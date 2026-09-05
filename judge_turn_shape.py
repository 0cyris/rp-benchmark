#!/usr/bin/env python3
"""LLM-judged response obligations per AI turn (turn shape, layer 2).

The rule-based detector in harness/turn_shape.py cannot reliably tell who a line
of dialogue is aimed at: seed rosters miss NPCs the model invents mid-scene, so
46-67% of quoted spans end up unattributable and the "NPC crosstalk is free" rule
— the whole point of the metric — barely runs. It also counts every question mark
as a demand (a rhetorical monologue scored 10), misses commands outside a 40-word
verb list, and misses demands outside quotation marks.

This asks a judge one narrow extraction question per turn instead: how many
separate things must the player's character respond to, with a quote for each.
Counting is done client-side from len(obligations) — a model-supplied integer is
never trusted (same reasoning as judge_session_flaw_hunter.py).

Output: results/turn_shape_judged.jsonl, append-only, one record per
(session_id, turn, judge). The file is its own checkpoint — rerunning skips work
already done, including for a different judge.

Usage:
    # Cost nothing, see what would be sent:
    python3 judge_turn_shape.py --dry-run --limit 5

    # Smoke test (~20 turns, one judge):
    python3 judge_turn_shape.py --judges claude_sonnet --limit 20

    # Pilot: 3 sessions per model, three judges, for inter-judge agreement
    python3 judge_turn_shape.py --n-per-model 3 \
        --judges claude_sonnet gpt_4_1 gemini_3_1_pro --concurrency 4

    # Full corpus, single judge
    python3 judge_turn_shape.py --judges claude_sonnet --concurrency 8
"""
import argparse
import json
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from harness import api
from harness.config import JUDGE_MODELS, REQUEST_DELAY_SECONDS

SOURCE = Path("results/multiturn_merged_all_v2.json")
SEEDS_FILE = Path("hf_dataset/_source/adversarial_seeds.json")
PROMPT_FILE = Path("prompts/judge_turn_shape.md")
RAW_OUT = Path("results/turn_shape_judged.jsonl")

# Judges beyond harness.config.JUDGE_MODELS, matching the round-3 second-judge
# pool in judge_round3_multi.py.
EXTRA_JUDGES = {
    "gemini_3_1_pro": "google/gemini-3.1-pro-preview",
    "gpt_latest": "~openai/gpt-latest",
}
ALL_JUDGES = {**JUDGE_MODELS, **EXTRA_JUDGES}

# Extraction, not composition: keep it deterministic. The output is a handful of
# quotes plus the shape flags for each turn in the batch.
JUDGE_CONFIG = {"temperature": 0.0, "max_tokens": 2500}

MIN_TURN_CHARS = 50


def load_done(path: Path) -> set:
    """Completed (session_id, turn, judge) keys.

    The JSONL is the checkpoint — there is no separate state file. Unlike
    judge_per_turn_failures.py the judge is part of the key, so a second judge
    over the same turns is not skipped as already-done.
    """
    done = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["session_id"], r["turn"], r["judge_key"]))
            except Exception:
                pass  # a torn final line costs one redone call
    return done


def stratified(sessions: list, n_per_model: int, rng: random.Random) -> list:
    """n_per_model sessions per model, biased toward distinct seeds."""
    by_model = defaultdict(list)
    for s in sessions:
        by_model[s["test_model"]].append(s)
    out = []
    for model in sorted(by_model):
        pool = by_model[model][:]
        rng.shuffle(pool)
        picked, seen = [], set()
        for s in pool:
            if len(picked) >= n_per_model:
                break
            if s["seed_id"] not in seen:
                seen.add(s["seed_id"])
                picked.append(s)
        for s in pool:  # backfill if the model has fewer distinct seeds
            if len(picked) >= n_per_model:
                break
            if s not in picked:
                picked.append(s)
        out.extend(picked)
    return out


def parse_judge_json(content: str) -> dict | None:
    """Fence-strip then brace-slice, as harness.api.judge_response does."""
    text = content.strip()
    if text.startswith("```"):
        text = "\n".join(
            l for l in text.split("\n") if not l.strip().startswith("```")
        )
    for cand in (text, text[text.find("{"):text.rfind("}") + 1] if "{" in text else ""):
        if not cand:
            continue
        try:
            parsed = json.loads(cand.strip())
            if isinstance(parsed, dict) and isinstance(parsed.get("turns"), list):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def build_work(sessions: list, seeds: dict, judges: dict, done: set,
               batch_size: int) -> list:
    """Batches of turns from one session, for one judge.

    The system prompt is ~1.4k tokens and dominates cost: sent once per turn it
    is 65% of all input tokens. Turns from the same session also share their
    scene block. Batching amortizes both. Batches stay small so a JSON parse
    failure costs a few turns rather than a whole session — the flaw-hunter
    session judge lost 66 of 336 sessions to unrecoverable JSON, and that is the
    failure mode being avoided here.
    """
    work = []
    for s in sessions:
        seed = seeds.get(s["seed_id"], {})
        sid = "%s::%s" % (s["test_model"], s["seed_id"])
        scene = {
            "character_name": s.get("character_name", "the character"),
            "character_setting": seed.get("character_setting", ""),
            "user_name": s.get("user_name", "the player"),
            "user_setting": seed.get("user_setting", ""),
        }
        turns = []
        for msg in s.get("dialogue", []):
            if msg.get("role") != "character" or msg.get("turn") == 0:
                continue  # turn 0 is the seed's opening, not model output
            content = (msg.get("content") or "").strip()
            if len(content) >= MIN_TURN_CHARS:
                turns.append({"turn": msg["turn"], "content": content})

        for jkey in judges:
            pending = [t for t in turns if (sid, t["turn"], jkey) not in done]
            for i in range(0, len(pending), batch_size):
                work.append({
                    "session_id": sid,
                    "model": s["test_model"],
                    "seed": s["seed_id"],
                    "judge_key": jkey,
                    "turns": pending[i:i + batch_size],
                    **scene,
                })
    return work


def build_user_content(w: dict) -> str:
    scene = (
        "<scene>\n"
        "The AI plays: %s\n%s\n\n"
        "The player plays: %s\n%s\n"
        "</scene>\n"
    ) % (
        w["character_name"], w["character_setting"],
        w["user_name"], w["user_setting"],
    )
    turns = "\n".join(
        '\n<turn n="%d">\n%s\n</turn>' % (t["turn"], t["content"])
        for t in w["turns"]
    )
    return scene + turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--out", default=str(RAW_OUT))
    ap.add_argument("--judges", nargs="+", default=["claude_sonnet"],
                    help="Short judge keys: %s" % ", ".join(sorted(ALL_JUDGES)))
    ap.add_argument("--n-per-model", type=int, default=None,
                    help="Stratified pilot: N sessions per model (default: all)")
    ap.add_argument("--max-models", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap total judge calls (smoke testing)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Turns per judge call. Amortizes the system prompt "
                         "(65%% of input tokens at batch size 1); keep it small "
                         "so a parse failure costs few turns.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print assembled prompts and a cost estimate, call nothing")
    args = ap.parse_args()

    unknown = [j for j in args.judges if j not in ALL_JUDGES]
    if unknown:
        ap.error("unknown judge(s): %s (known: %s)"
                 % (", ".join(unknown), ", ".join(sorted(ALL_JUDGES))))
    judges = {j: ALL_JUDGES[j] for j in args.judges}

    data = json.loads(Path(args.source).read_text())
    seen, sessions = set(), []
    for s in data["sessions"]:
        key = (s["test_model"], s["seed_id"])
        if key not in seen:
            seen.add(key)
            sessions.append(s)
    if args.max_models:
        keep = sorted({s["test_model"] for s in sessions})[:args.max_models]
        sessions = [s for s in sessions if s["test_model"] in keep]
    if args.n_per_model:
        sessions = stratified(sessions, args.n_per_model, random.Random(args.seed))

    seeds = {s["id"]: s for s in json.loads(SEEDS_FILE.read_text())}
    system_prompt = PROMPT_FILE.read_text()
    out_path = Path(args.out)
    done = load_done(out_path)
    work = build_work(sessions, seeds, judges, done, args.batch_size)
    if args.limit:
        work = work[:args.limit]

    n_turns = sum(len(w["turns"]) for w in work)
    print("Sessions: %d | judges: %s | done: %d turns | to do: %d turns in %d calls"
          " (batch %d)"
          % (len(sessions), ", ".join(judges), len(done), n_turns, len(work),
             args.batch_size))

    if args.dry_run:
        # Rough token estimate at ~4 chars/token; the real number comes back in
        # usage["cost"] once anything actually runs.
        sys_tok = len(system_prompt) / 4
        est = sum((sys_tok + len(build_user_content(w)) / 4) for w in work)
        out_est = 220 * n_turns
        print("\nEstimated input tokens: %.2fM  output: %.2fM"
              % (est / 1e6, out_est / 1e6))
        print("At ~$4.7/M blended (Sonnet 4, from this corpus's recorded usage):"
              " ~$%.2f\n" % ((est + out_est) / 1e6 * 4.7))
        for w in work[:2]:
            print("=" * 70)
            print("%s turns %s -> %s"
                  % (w["session_id"], [t["turn"] for t in w["turns"]],
                     w["judge_key"]))
            print("=" * 70)
            print(build_user_content(w)[:1500])
            print()
        return

    if not work:
        print("Nothing to do.")
        return

    write_lock = threading.Lock()
    counter = {"done": 0, "turns": 0, "errors": 0, "parse_errors": 0,
               "cost": 0.0}
    start = time.time()

    def _run_one(w: dict) -> dict:
        """Never raises — a failed call must not kill the pool."""
        try:
            resp = api.chat_completion(
                model=judges[w["judge_key"]],
                system_prompt=system_prompt,
                user_content=build_user_content(w),
                config=dict(JUDGE_CONFIG),  # copy: never mutate shared config
            )
        except Exception as e:
            return {"error": "%s: %s" % (type(e).__name__, e)}
        parsed = parse_judge_json(resp.get("content", ""))
        return {"resp": resp, "parsed": parsed}

    def _record(w: dict, res: dict):
        """Fan one batched response out to one JSONL row per turn.

        Rows stay per-turn whatever the batch size, so load_done, the analysis
        and the kappa script all key on (session_id, turn, judge) regardless of
        how the calls were grouped.
        """
        base = {
            "session_id": w["session_id"], "model": w["model"], "seed": w["seed"],
            "judge_key": w["judge_key"], "judge": judges[w["judge_key"]],
        }
        usage = (res.get("resp") or {}).get("usage")
        parsed = res.get("parsed")
        by_n = {}
        if parsed:
            for entry in parsed.get("turns", []):
                if isinstance(entry, dict) and entry.get("n") is not None:
                    try:
                        by_n[int(entry["n"])] = entry
                    except (TypeError, ValueError):
                        pass

        rows = []
        for i, t in enumerate(w["turns"]):
            rec = dict(base, turn=t["turn"])
            # Usage belongs to the batch, not the turn: attach it once so a
            # naive sum over rows is the true spend.
            if i == 0:
                rec["usage"] = usage
                rec["batch_size"] = len(w["turns"])
            if res.get("error"):
                rec["error"] = res["error"]
            elif parsed is None:
                rec["parse_error"] = True
                rec["raw_content"] = ((res.get("resp") or {}).get("content") or "")[:2000]
            elif t["turn"] not in by_n:
                # The judge dropped this turn from its array.
                rec["parse_error"] = True
                rec["raw_content"] = "turn %d missing from judge response" % t["turn"]
            else:
                entry = by_n[t["turn"]]
                raw = [o for o in entry.get("obligations", []) if isinstance(o, dict)]
                counted = [
                    o for o in raw
                    if o.get("addressee") == "player" and not o.get("rhetorical")
                ]
                rec["obligations"] = counted
                rec["n_obligations"] = len(counted)   # counted client-side
                rec["n_rhetorical"] = sum(1 for o in raw if o.get("rhetorical"))
                # Shape, per the GM-card spec's system-agnostic behavioral spine:
                # describe-then-stop, scene-as-map, never act for the player.
                rec["player_present"] = entry.get("player_present", True) is not False
                rec["handholds"] = entry.get("handholds")
                rec["handback"] = entry.get("handback")
                rec["pc_interiority_leak"] = bool(entry.get("pc_interiority_leak"))
                rec["crosstalk_present"] = bool(entry.get("crosstalk_present"))
                rec["npc_grading"] = bool(entry.get("npc_grading"))
                rec["monologue"] = bool(entry.get("monologue"))
                rec["manufactured_tension"] = bool(entry.get("manufactured_tension"))
                rec["notes"] = entry.get("notes", "")
            rows.append(rec)

        with write_lock:
            with open(out_path, "a") as f:
                for rec in rows:
                    f.write(json.dumps(rec) + "\n")
            if res.get("error"):
                counter["errors"] += 1
            elif parsed is None:
                counter["parse_errors"] += 1
            counter["cost"] += (usage or {}).get("cost") or 0.0
            counter["done"] += 1
            counter["turns"] += len(rows)
            n = counter["done"]
            if n % 25 == 0 or n == len(work):
                rate = n / max(time.time() - start, 1e-6)
                print("  %d/%d calls (%d turns)  %.1f/s  errors=%d parse_errors=%d"
                      "  $%.2f  eta %.0fs"
                      % (n, len(work), counter["turns"], rate, counter["errors"],
                         counter["parse_errors"], counter["cost"],
                         (len(work) - n) / max(rate, 1e-6)))

    if args.concurrency > 1:
        # Required: without this every worker serializes on the global 1.0s gate.
        api.set_min_interval(REQUEST_DELAY_SECONDS / args.concurrency)
        print("Concurrency: %d (min interval %.3fs)\n"
              % (args.concurrency, REQUEST_DELAY_SECONDS / args.concurrency))
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(_run_one, w): w for w in work}
            for fut in as_completed(futs):
                _record(futs[fut], fut.result())
    else:
        for w in work:
            _record(w, _run_one(w))

    print("\nDone. %d calls / %d turn records, %d API errors, %d parse errors,"
          " $%.2f actual cost."
          % (counter["done"], counter["turns"], counter["errors"],
             counter["parse_errors"], counter["cost"]))
    print("Wrote %s" % out_path)


if __name__ == "__main__":
    main()
