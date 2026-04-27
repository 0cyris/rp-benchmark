#!/usr/bin/env python3
"""Fetch all votes from the live arena and split by mode.

Pulls https://arena.l3vi4th4n.ai/api/votes (the public endpoint that backs
analyze_*.py inputs) and partitions votes into per-mode JSONL files:
  - mode='arena'            → web/data/votes.jsonl  (single-message)
  - mode='multiturn_arena'  → data/multiturn_arena_votes.jsonl
  - mode='rubric'           → data/rubric_votes.jsonl

Usage:
    python3 fetch_arena_votes.py
"""
import json
from pathlib import Path
from urllib.request import urlopen


URL = "https://arena.l3vi4th4n.ai/api/votes"

DEST = {
    "arena": Path("web/data/votes.jsonl"),
    "multiturn_arena": Path("data/multiturn_arena_votes.jsonl"),
    "rubric": Path("data/rubric_votes.jsonl"),
}


def main():
    print(f"Fetching {URL}...")
    with urlopen(URL) as r:
        payload = json.load(r)
    votes = payload["votes"]
    print(f"  total votes: {len(votes)}")

    by_mode: dict[str, list[dict]] = {}
    for v in votes:
        by_mode.setdefault(v.get("mode", "unknown"), []).append(v)

    for mode, vs in by_mode.items():
        out = DEST.get(mode) or Path(f"data/{mode}_votes.jsonl")
        out.parent.mkdir(exist_ok=True, parents=True)
        with open(out, "w") as f:
            for v in vs:
                f.write(json.dumps(v) + "\n")
        print(f"  {mode}: {len(vs)} → {out}")


if __name__ == "__main__":
    main()
