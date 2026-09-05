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
(session_id, turn, judge, prompt version). The file is its own checkpoint —
rerunning skips work already done, including for a different judge. The judge
prompt's content hash is part of the key, so editing the prompt re-judges rather
than silently leaving two schemas interleaved in one file.

Usage:
    # Cost nothing, see what would be sent:
    python3 judge_turn_shape.py --dry-run --limit 5

    # Smoke test (~20 turns, one judge):
    python3 judge_turn_shape.py --judges claude_sonnet --limit 20

    # Pilot: 3 sessions per model, three judges, for inter-judge agreement
    python3 judge_turn_shape.py --n-per-model 3 \
        --judges claude_sonnet gpt_4_1 gemini_3_1_pro --concurrency 4

    # Discrimination probe: the seeds built to elicit the shape failures
    python3 judge_turn_shape.py --concurrency 4 \
        --seeds adv_agency_emotional_climax_09 adv_agency_combat_10 \
                adv_pov_multi_npc_13 adv_passive_user_03 \
        --models mistral_small_creative claude_opus_4_6 llama_4_maverick \
                 gemini_2_5_flash

    # Full corpus, single judge
    python3 judge_turn_shape.py --judges claude_sonnet --concurrency 8

    # The purpose-built turn-shape corpus (both arms; see
    # run_turn_shape_generation.py). --seeds-file must match --source or the
    # scene block comes out empty.
    python3 judge_turn_shape.py --concurrency 8 \
        --source results/turn_shape_corpus.json \
        --seeds-file hf_dataset/_source/turn_shape_seeds.json \
        --out results/turn_shape_corpus_judged.jsonl

    # Simulator baseline: score the sim's own turns with the same metric, so
    # the models' bundling rate can be read against the floor set by what the
    # sim hands them.
    python3 judge_turn_shape.py --judge-role user --concurrency 4 \
        --source results/turn_shape_corpus.json \
        --seeds-file hf_dataset/_source/turn_shape_seeds.json \
        --out results/turn_shape_corpus_judged.jsonl
"""
import argparse
import hashlib
import json
import random
import sys
import threading
import time
from collections import Counter, defaultdict
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

# Reasoning judges need far more headroom: they spend the budget thinking before
# emitting a character of JSON. In the first pilot gemini_3_1_pro burned a median
# 2,397 reasoning tokens against a 2,500 cap, leaving ~100 for the answer, and
# truncated on 80% of its calls -- 367 unusable rows. Values match the round-3
# pool in judge_round3_multi.py:55, which hit the same wall.
JUDGE_MAX_TOKENS = {
    "gemini_3_1_pro": 16000,
    "gpt_latest": 8000,
}

MIN_TURN_CHARS = 50


def prompt_fingerprint(text: str) -> str:
    """Short content hash of the judge prompt.

    A content hash rather than a hand-maintained version constant: the failure
    this guards against is a human forgetting that the output schema moved, and
    a hash cannot forget.
    """
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_done(path: Path) -> tuple[set, set, Counter, set]:
    """Read the checkpoint.

    Returns (done, seen_any, by_sha, failed):
      done     — (session_id, turn, judge_key, prompt_sha) already judged
      seen_any — (session_id, turn, judge_key) judged under *any* prompt version
      by_sha   — row counts per prompt version, for reporting
      failed   — (session_id, turn, judge_key, prompt_sha) whose newest record is
                 a parse_error or API error, and so is a candidate for retry

    The JSONL is the checkpoint; there is no separate state file. The prompt
    fingerprint is part of the key because it has to be: keyed only on
    (session, turn, judge), a rerun after a prompt rewrite silently skips every
    previously judged turn and leaves one file holding two incompatible
    schemas — which is exactly what happened, and it produced a leaderboard
    that was pure artifact of which rows predated the change.
    """
    done, seen_any, by_sha, status = set(), set(), Counter(), {}
    if not path.exists():
        return done, seen_any, by_sha, set()
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                sha = r.get("prompt_sha", "legacy")
                key = (r["session_id"], r["turn"], r["judge_key"], sha)
                by_sha[sha] += 1
                done.add(key)
                seen_any.add((r["session_id"], r["turn"], r["judge_key"]))
                # Last record wins: a later successful retry clears an earlier
                # failure for the same key.
                status[key] = bool(r.get("parse_error") or r.get("error"))
            except Exception:
                pass  # a torn final line costs one redone call
    return done, seen_any, by_sha, {k for k, bad in status.items() if bad}


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
               batch_size: int, prompt_sha: str, seen_any: set | None = None,
               stale_only: bool = False, failed: set | None = None,
               retry_failed: bool = False, role: str = "character") -> list:
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
        # sid is part of the checkpoint key, so it must separate arms: two
        # sessions differing only by system prompt would otherwise look like
        # one already-judged session and the second would be skipped. Sessions
        # with no arm (every corpus before the turn-shape one) keep their
        # original ids, so existing .jsonl checkpoints stay valid.
        sid = "%s::%s" % (s["test_model"], s["seed_id"])
        if s.get("arm"):
            sid += "::%s" % s["arm"]
        if role != "character":
            sid += "#%s" % role
        # For the sim baseline (role="user") the two sides swap: the prompt
        # counts what is aimed at "the player's character", and when the turns
        # under test are the simulator's, the side being handed to is the AI's.
        # Same metric, other direction -- that is what makes it a baseline.
        scene = {
            "character_name": s.get("character_name", "the character"),
            "character_setting": seed.get("character_setting", ""),
            "user_name": s.get("user_name", "the player"),
            "user_setting": seed.get("user_setting", ""),
        }
        if role == "user":
            scene = {
                "character_name": scene["user_name"],
                "character_setting": scene["user_setting"],
                "user_name": scene["character_name"],
                "user_setting": scene["character_setting"],
            }
        turns = []
        for msg in s.get("dialogue", []):
            if msg.get("role") != role or msg.get("turn") in (0, 1):
                continue  # turns 0 and 1 come from the seed, not from a model
            if msg.get("is_challenge"):
                continue  # scripted seed input, identical across models
            content = (msg.get("content") or "").strip()
            if len(content) >= MIN_TURN_CHARS:
                turns.append({"turn": msg["turn"], "content": content})

        for jkey in judges:
            if retry_failed:
                # Parse-error rows are written *with* the current prompt_sha, so
                # load_done counts them as done and a plain rerun skips them.
                # That is deliberate (an unparseable turn should not retry
                # forever) but it means a config fix cannot reach them without
                # this flag.
                pending = [t for t in turns
                           if (sid, t["turn"], jkey, prompt_sha) in (failed or set())]
            else:
                pending = [t for t in turns
                           if (sid, t["turn"], jkey, prompt_sha) not in done]
            if stale_only:
                # Only turns that already have a row under some *other* prompt
                # version — the recovery path after a prompt edit.
                pending = [t for t in pending
                           if (sid, t["turn"], jkey) in (seen_any or set())]
            for i in range(0, len(pending), batch_size):
                work.append({
                    "session_id": sid,
                    "model": s["test_model"],
                    "seed": s["seed_id"],
                    "arm": s.get("arm"),
                    "role": role,
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
    ap.add_argument("--seeds-file", default=str(SEEDS_FILE),
                    help="Seed set matching --source (the scene block comes "
                         "from it). Default: %s" % SEEDS_FILE.name)
    ap.add_argument("--judge-role", default="character",
                    choices=["character", "user"],
                    help="Whose turns to judge. 'user' scores the simulator's "
                         "own turns as a baseline: the sim is told to push and "
                         "ask questions, and models mirror their interlocutor, "
                         "so its own bundling rate is the floor to report the "
                         "models against.")
    ap.add_argument("--judges", nargs="+", default=["claude_sonnet"],
                    help="Short judge keys: %s" % ", ".join(sorted(ALL_JUDGES)))
    ap.add_argument("--seeds", nargs="+", default=None,
                    help="Only these seed_ids (e.g. adv_pov_multi_npc_13)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Only these test_model keys. Unlike --max-models this "
                         "picks by name rather than taking the first N sorted.")
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
    ap.add_argument("--retry-failed", action="store_true",
                    help="Re-judge only rows that failed to parse at the current "
                         "prompt version (recovery after a judge config fix)")
    ap.add_argument("--stale-only", action="store_true",
                    help="Re-judge only turns already scored under an older "
                         "prompt version (recovery after a prompt edit)")
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
        # The arm belongs in the de-dup key: the turn-shape corpus runs every
        # (model, seed) twice, once per system-prompt arm, and without it the
        # second arm is silently discarded as a duplicate.
        key = (s["test_model"], s["seed_id"], s.get("arm"))
        if key not in seen:
            seen.add(key)
            sessions.append(s)
    if args.seeds:
        want = set(args.seeds)
        unknown = want - {s["seed_id"] for s in sessions}
        if unknown:
            ap.error("unknown seed(s): %s" % ", ".join(sorted(unknown)))
        sessions = [s for s in sessions if s["seed_id"] in want]
    if args.models:
        want = set(args.models)
        unknown = want - {s["test_model"] for s in sessions}
        if unknown:
            ap.error("unknown model(s): %s" % ", ".join(sorted(unknown)))
        sessions = [s for s in sessions if s["test_model"] in want]
    if args.max_models:
        keep = sorted({s["test_model"] for s in sessions})[:args.max_models]
        sessions = [s for s in sessions if s["test_model"] in keep]
    if args.n_per_model:
        sessions = stratified(sessions, args.n_per_model, random.Random(args.seed))

    seeds = {s["id"]: s for s in json.loads(Path(args.seeds_file).read_text())}
    system_prompt = PROMPT_FILE.read_text()
    prompt_sha = prompt_fingerprint(system_prompt)
    out_path = Path(args.out)
    done, seen_any, by_sha, failed = load_done(out_path)
    if args.retry_failed and args.stale_only:
        ap.error("--retry-failed and --stale-only are mutually exclusive")
    work = build_work(sessions, seeds, judges, done, args.batch_size,
                      prompt_sha, seen_any, args.stale_only,
                      failed, args.retry_failed, args.judge_role)
    if args.limit:
        work = work[:args.limit]

    n_turns = sum(len(w["turns"]) for w in work)
    print("Prompt version: %s" % prompt_sha)
    if failed:
        print("%d row(s) at this version failed to parse.%s"
              % (len(failed),
                 "" if args.retry_failed else " Use --retry-failed to redo them."))
    stale = {sha: n for sha, n in by_sha.items() if sha != prompt_sha}
    if stale:
        print("WARNING: %d existing row(s) from %d older prompt version(s): %s"
              % (sum(stale.values()), len(stale),
                 ", ".join("%s=%d" % kv for kv in sorted(stale.items()))))
        print("         They will be re-judged (or use --stale-only to do just"
              " those). Analysis excludes rows from other versions.")
    print("Sessions: %d | judges: %s | done at this version: %d turns |"
          " to do: %d turns in %d calls (batch %d)"
          % (len(sessions), ", ".join(judges), by_sha.get(prompt_sha, 0),
             n_turns, len(work), args.batch_size))

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
        # Per-judge config, copied per call. Deliberately NOT the shared-global
        # monkey-patch in judge_round3_multi.py:169 — that is not thread-safe and
        # this runner uses a ThreadPoolExecutor.
        config = dict(JUDGE_CONFIG)
        config["max_tokens"] = JUDGE_MAX_TOKENS.get(
            w["judge_key"], JUDGE_CONFIG["max_tokens"])
        try:
            resp = api.chat_completion(
                model=judges[w["judge_key"]],
                system_prompt=system_prompt,
                user_content=build_user_content(w),
                config=config,
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
            "prompt_sha": prompt_sha,
        }
        if w.get("arm"):
            base["arm"] = w["arm"]
        if w.get("role", "character") != "character":
            base["role"] = w["role"]
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
            if usage and usage.get("completion_tokens"):
                counter.setdefault("ctoks", defaultdict(list))
                counter["ctoks"][w["judge_key"]].append(usage["completion_tokens"])
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

    # Truncation check. A judge whose median completion sits at its cap is being
    # cut off mid-JSON, which shows up as parse errors rather than as an obvious
    # failure. This signal was already in the usage data during the first pilot
    # and nothing looked at it, so 367 rows were lost before anyone noticed.
    for jkey, toks in sorted(counter.get("ctoks", {}).items()):
        cap = JUDGE_MAX_TOKENS.get(jkey, JUDGE_CONFIG["max_tokens"])
        med = sorted(toks)[len(toks) // 2]
        if med >= cap * 0.98:
            print("WARNING: %s median completion %d tokens against a %d cap —"
                  " responses are being truncated. Raise JUDGE_MAX_TOKENS[%r]"
                  " and rerun with --retry-failed."
                  % (jkey, med, cap, jkey))
    print("Wrote %s" % out_path)


if __name__ == "__main__":
    main()
