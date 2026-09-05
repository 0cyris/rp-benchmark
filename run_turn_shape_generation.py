#!/usr/bin/env python3
"""Generate the purpose-built turn-shape corpus.

The adversarial corpus cannot answer the turn-shape question. 68% of its model
pairs share no seed, most models have 22 turns from 2 seeds, and its system
prompt (harness/multiturn.py) says nothing about turn shape at all -- so it can
only report unguided defaults, never whether a model *can* hold the floor when
asked. Three of its seeds also mandate a POV that contradicts the harness prompt.

This runs a fully crossed design instead: every model x every seed x two arms.

  unprompted  today's system prompt, byte-identical -- a replication of the
              existing corpus's conditions on clean seeds
  prompted    the same, plus prompts/system_turn_shape.md

Paired per (model, seed), so the arm delta is a within-subject effect and reads
as steerability: low unprompted + high prompted means "usable if you ask", low
in both means the model cannot do it.

Session judging is skipped (judge_models={}); the turn-shape judge is the only
scoring this corpus needs, so generation is the whole generation-side cost.

Usage:
    python3 run_turn_shape_generation.py --dry-run
    python3 run_turn_shape_generation.py --models gemini_2_5_flash --seeds ts_solo_quiet_01
    python3 run_turn_shape_generation.py --concurrency 8
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from harness.config import TEST_MODELS, RESULTS_DIR
from harness.multiturn import run_multiturn_benchmark, load_seeds

SEEDS_FILE = "turn_shape_seeds.json"
SHAPE_PROMPT = Path("prompts/system_turn_shape.md")
MERGED_OUT = RESULTS_DIR / "turn_shape_corpus.json"
NUM_TURNS = 10

# Twelve models: the pilot's extremes at both ends (so the new corpus can be
# cross-checked against the 639-row Sonnet pilot) plus the current frontier.
DEFAULT_MODELS = [
    # strong on shape in the pilot
    "gemini_2_5_flash", "llama_4_maverick", "deepseek_v3_2", "claude_opus_4_6",
    # weak on shape in the pilot
    "claude_sonnet_4_5", "claude_opus_4_7", "gemma_4_26b", "minimax_m2_7",
    # frontier refresh, not yet measured for shape
    "claude_opus_4_8", "gpt_5_5", "gemini_3_5_flash", "grok_4_3",
]

ARMS = ("unprompted", "prompted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--seeds", nargs="+", default=None,
                    help="Seed ids to run. Default: all in %s." % SEEDS_FILE)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--turns", type=int, default=NUM_TURNS)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print both arms' prompts and the work list; spend nothing.")
    args = ap.parse_args()

    unknown = [m for m in args.models if m not in TEST_MODELS]
    if unknown:
        sys.exit("Unknown models: %s\nAvailable: %s"
                 % (unknown, sorted(TEST_MODELS)))
    models = {m: TEST_MODELS[m] for m in args.models}

    seeds = load_seeds(seeds_file=SEEDS_FILE)
    if args.seeds:
        known = {s["id"] for s in seeds}
        missing = [s for s in args.seeds if s not in known]
        if missing:
            sys.exit("Unknown seeds: %s\nAvailable: %s" % (missing, sorted(known)))
        seeds = [s for s in seeds if s["id"] in args.seeds]
    seed_ids = [s["id"] for s in seeds]

    shape_extra = SHAPE_PROMPT.read_text().strip()

    n_sessions = len(models) * len(seed_ids) * len(args.arms)
    print("TURN-SHAPE CORPUS")
    print("  models  %d: %s" % (len(models), sorted(models)))
    print("  seeds   %d: %s" % (len(seed_ids), seed_ids))
    print("  arms    %s" % list(args.arms))
    print("  turns   %d  ->  %d model turns per session" % (args.turns, args.turns - 1))
    print("  sessions %d, ~%d judged turns\n"
          % (n_sessions, n_sessions * (args.turns - 1)))

    if args.dry_run:
        for arm in args.arms:
            print("=" * 72)
            print("ARM: %s" % arm)
            print("=" * 72)
            print(_preview_system(seeds[0], shape_extra if arm == "prompted" else None))
            print()
        print("Dry run: nothing sent.")
        return

    results, t0 = {}, time.time()
    for arm in args.arms:
        extra = shape_extra if arm == "prompted" else None
        print("\n" + "#" * 72)
        print("# ARM %s | %d sessions" % (arm, len(models) * len(seed_ids)))
        print("#" * 72, flush=True)
        try:
            res = run_multiturn_benchmark(
                test_models=models,
                judge_models={},          # generation only -- see module docstring
                seed_ids=seed_ids,
                num_turns=args.turns,
                concurrency=args.concurrency,
                seeds_file=SEEDS_FILE,
                char_system_extra=extra,
                arm=arm,
            )
        except Exception:
            print(traceback.format_exc(), flush=True)
            continue
        path = RESULTS_DIR / ("turn_shape_corpus_%s.json" % arm)
        path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print("  -> %s" % path, flush=True)
        results[arm] = res

    if not results:
        sys.exit("No arm completed.")

    merged = {
        "type": "turn_shape_corpus",
        "config": {
            "seeds_file": SEEDS_FILE,
            "num_turns": args.turns,
            "models": models,
            "seed_ids": seed_ids,
            "arms": list(results),
            "shape_prompt": shape_extra,
            "sources": {a: r["run_id"] for a, r in results.items()},
        },
        "sessions": [s for r in results.values() for s in r["sessions"]],
    }
    MERGED_OUT.write_text(json.dumps(merged, indent=2, ensure_ascii=False))

    ok = sum(1 for s in merged["sessions"] if "error" not in s)
    print("\n%d sessions (%d ok, %d errored) in %.1f min -> %s"
          % (len(merged["sessions"]), ok, len(merged["sessions"]) - ok,
             (time.time() - t0) / 60, MERGED_OUT))

    missing = _missing_cells(merged["sessions"], models, seed_ids, list(results))
    if missing:
        print("\nWARNING: %d empty (model, seed, arm) cells. The arm delta is a "
              "paired comparison, so these must be re-run, not averaged over:"
              % len(missing))
        for cell in missing[:20]:
            print("  %s x %s [%s]" % cell)


def _preview_system(seed, extra):
    """Reproduce run_session's char_system for eyeballing before spending."""
    s = (
        "You are roleplaying as %s. Stay in character at all times. "
        "Write in third-person past tense. Do NOT write actions or "
        "dialogue for %s — they are controlled by the user."
    ) % (seed["character_name"], seed["user_name"])
    if extra:
        s += "\n\n" + extra
    return s + "\n\n## Your Character\n%s" % seed["character_setting"]


def _missing_cells(sessions, models, seed_ids, arms):
    have = {(s["test_model"], s["seed_id"], s.get("arm"))
            for s in sessions if "error" not in s}
    return [(m, sd, a) for m in models for sd in seed_ids for a in arms
            if (m, sd, a) not in have]


if __name__ == "__main__":
    main()
