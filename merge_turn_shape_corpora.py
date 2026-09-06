#!/usr/bin/env python3
"""Merge every turn-shape generation file into one corpus.

The generator overwrites results/turn_shape_corpus.json each run, but the
harness also leaves a timestamped copy of every run that it never touches
again. Each session carries its own test_model / seed_id / arm, so the files
do not need to be identified by hand -- anything holding turn-shape sessions
is picked up, and (model, seed, arm) de-duplicates re-runs, newest file wins.

Usage:
    python3 merge_turn_shape_corpora.py 'results/multiturn_*.json' \
        results/turn_shape_corpus_all.json
"""
import json, sys, glob

SEEDS = {s["id"] for s in
         json.load(open("hf_dataset/_source/turn_shape_seeds.json"))}
paths = sorted(set(sum((glob.glob(p) for p in sys.argv[1:-1]), [])))
out_path = sys.argv[-1]

by_cell, skipped = {}, []
for p in paths:
    try:
        data = json.load(open(p))
    except (json.JSONDecodeError, OSError) as e:
        skipped.append((p, str(e)[:40]))
        continue
    kept = 0
    for s in data.get("sessions", []):
        if s.get("seed_id") not in SEEDS or "error" in s:
            continue
        by_cell[(s["test_model"], s["seed_id"], s.get("arm"))] = s
        kept += 1
    print("  %-46s %d turn-shape session(s)" % (p.split("/")[-1], kept))

sessions = list(by_cell.values())
json.dump({
    "type": "turn_shape_corpus",
    "config": {
        "seeds_file": "turn_shape_seeds.json",
        "merged_from": paths,
        "models": sorted({s["test_model"] for s in sessions}),
        "seed_ids": sorted({s["seed_id"] for s in sessions}),
        "arms": sorted({s.get("arm") for s in sessions if s.get("arm")}),
    },
    "sessions": sessions,
}, open(out_path, "w"), indent=2, ensure_ascii=False)

print("\n%d unique (model, seed, arm) cells -> %s" % (len(sessions), out_path))
for m in sorted({s["test_model"] for s in sessions}):
    cells = [(s["seed_id"], s.get("arm")) for s in sessions if s["test_model"] == m]
    print("  %-20s %d cells: %s" % (m, len(cells), sorted(cells)))
if skipped:
    print("skipped unreadable: %s" % skipped)
