#!/usr/bin/env python3
"""Analysis for the LLM-judged turn-shape metric (layer 2).

Reads results/turn_shape_judged.jsonl (produced by judge_turn_shape.py) and
reports four things:

  1. Per-model judged leaderboard — the bundling rate with a Wilson CI, mean
     obligations, and the puppeted-user rate the rule-based version could not see.
  2. The correlation gate — bundling vs composite, vs human multi-turn arena ELO,
     vs response length, and length-normalized. Low correlation with the composite
     is what would justify a separate leaderboard axis (METHODOLOGY 13.2).
  3. Rules vs judge — per-turn agreement between this and harness/turn_shape.py.
     If the judge merely reproduces the rule-based numbers, the spend bought
     nothing and that should be visible.
  4. Inter-judge agreement — Cohen's kappa on the bundled flag for every judge
     pair on turns they both scored. The benchmark's single-judge dependence is
     its most-documented weakness; this metric should not repeat it.

Note on kappa: analyze_round3_kappa.py uses quadratic-weighted kappa over a 1-5
Likert, which degenerates on a binary flag (both values clip into one level), so
plain unweighted Cohen's kappa is computed here instead.

Usage:
    python3 analyze_turn_shape_judged.py
"""
import json
import statistics as st
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from harness.turn_shape import analyze_turn, build_roster
from generate_profile_cards import wilson_ci
from analyze_method_correlations import spearman

JUDGED = Path("results/turn_shape_judged.jsonl")
SESSIONS_FILE = Path("results/multiturn_merged_all_v2.json")
SEEDS_FILE = Path("hf_dataset/_source/adversarial_seeds.json")
COMPOSITE_FILE = Path("results/composite_leaderboard.json")
MT_ARENA_FILE = Path("results/multiturn_arena_bayesian.json")
OUTPUT = Path("results/turn_shape_judged.json")

BUNDLED_THRESHOLD = 2


def load_judged() -> list[dict]:
    rows = []
    with open(JUDGED) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def cohens_kappa(a: list[bool], b: list[bool]) -> float | None:
    """Unweighted Cohen's kappa for two binary raters."""
    n = len(a)
    if n == 0:
        return None
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    exp = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if exp >= 1.0:
        return None  # both raters constant: kappa undefined
    return (obs - exp) / (1 - exp)


def safe_load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def rule_based_per_turn() -> dict:
    """{(session_id, turn): obligations} from the free rule-based detector."""
    data = json.loads(SESSIONS_FILE.read_text())
    seeds = {s["id"]: s for s in json.loads(SEEDS_FILE.read_text())}
    rosters = {sid: build_roster(s) for sid, s in seeds.items()}
    out, seen = {}, set()
    for sess in data["sessions"]:
        key = (sess["test_model"], sess["seed_id"])
        if key in seen:
            continue
        seen.add(key)
        roster = rosters.get(sess["seed_id"])
        if roster is None:
            continue
        sid = "%s::%s" % key
        for msg in sess["dialogue"]:
            if msg.get("role") != "character":
                continue
            r = analyze_turn(msg.get("content"), roster, sess.get("character_name"))
            if not r["empty"]:
                out[(sid, msg["turn"])] = r["obligations_loose"]
    return out


def main():
    if not JUDGED.exists():
        print("No %s yet — run judge_turn_shape.py first." % JUDGED)
        print("  smoke:  python3 judge_turn_shape.py --judges claude_sonnet --limit 20")
        return 1

    rows = load_judged()
    scored = [r for r in rows if "n_obligations" in r]
    failed = [r for r in rows if r.get("parse_error") or r.get("error")]
    judges = sorted({r["judge_key"] for r in rows})
    cost = sum((r.get("usage") or {}).get("cost") or 0.0 for r in rows)

    print("JUDGED TURN SHAPE — player-directed response obligations")
    print("%d turn records (%d scored, %d failed) across %d judge(s): %s"
          % (len(rows), len(scored), len(failed), len(judges), ", ".join(judges)))
    print("Actual spend: $%.2f\n" % cost)
    if not scored:
        print("Nothing scored successfully.")
        return 1

    # --- 1. per-model leaderboard, primary judge --------------------------
    primary = judges[0] if len(judges) == 1 else (
        "claude_sonnet" if "claude_sonnet" in judges else judges[0])
    per_model = defaultdict(list)
    for r in scored:
        if r["judge_key"] == primary:
            per_model[r["model"]].append(r)

    lb = []
    for model, rs in per_model.items():
        n = len(rs)
        k = sum(1 for r in rs if r["n_obligations"] >= BUNDLED_THRESHOLD)
        lo, hi = wilson_ci(k, n)
        lb.append({
            "model": model,
            "n_turns": n,
            "bundled_rate": round(k / n, 4),
            "bundled_ci": [round(lo, 4), round(hi, 4)],
            "mean_obligations": round(st.mean(r["n_obligations"] for r in rs), 3),
            "mean_rhetorical": round(
                st.mean(r.get("n_rhetorical", 0) for r in rs), 3),
            "crosstalk_rate": round(
                sum(1 for r in rs if r.get("crosstalk_present")) / n, 4),
            "puppeted_rate": round(
                sum(1 for r in rs if r.get("puppeted_user")) / n, 4),
        })
    lb.sort(key=lambda r: r["bundled_rate"])

    print("Primary judge: %s" % primary)
    print("%-26s %6s %18s %7s %8s %9s" % (
        "model", "turns", "bundled%", "mean", "cross%", "puppet%"))
    print("-" * 80)
    for r in lb:
        print("%-26s %6d  %5.1f [%4.1f-%4.1f] %7.2f %7.1f %8.1f" % (
            r["model"], r["n_turns"], r["bundled_rate"] * 100,
            r["bundled_ci"][0] * 100, r["bundled_ci"][1] * 100,
            r["mean_obligations"], r["crosstalk_rate"] * 100,
            r["puppeted_rate"] * 100))

    # --- 2. correlation gate ----------------------------------------------
    composite, mt_arena = {}, {}
    if (c := safe_load(COMPOSITE_FILE)):
        composite = {e["model"]: e["composite_score"] for e in c["leaderboard"]}
    if (a := safe_load(MT_ARENA_FILE)):
        mt_arena = {e["model"]: e["elo_mean"] for e in a["leaderboard"]}

    words = {}
    if (ts := safe_load(Path("results/turn_shape.json"))):
        words = {e["model"]: e["mean_words"] for e in ts["leaderboard"]}
    for r in lb:
        w = words.get(r["model"])
        r["obligations_per_1k_words"] = (
            round(r["mean_obligations"] / w * 1000, 3) if w else None)

    def rho(ref, field):
        pairs = [(r[field], ref[r["model"]]) for r in lb
                 if r["model"] in ref and r.get(field) is not None]
        if len(pairs) < 3:
            return None, len(pairs)
        return spearman([p[0] for p in pairs], [p[1] for p in pairs]), len(pairs)

    gate = {
        "bundled_vs_composite": rho(composite, "bundled_rate"),
        "bundled_vs_mt_arena_elo": rho(mt_arena, "bundled_rate"),
        "bundled_vs_mean_words": rho(words, "bundled_rate"),
        "per_1k_vs_composite": rho(composite, "obligations_per_1k_words"),
        "per_1k_vs_mt_arena_elo": rho(mt_arena, "obligations_per_1k_words"),
    }
    print("\nCORRELATION GATE (negative = bundling tracks a worse ranking)")
    for name, (v, n) in gate.items():
        print("  %-26s rho = %s (n=%d)"
              % (name, "n/a" if v is None else "%+.3f" % v, n))

    # --- 3. rules vs judge -------------------------------------------------
    print("\nRULES vs JUDGE (per turn, primary judge)")
    rules = rule_based_per_turn()
    pairs = [(rules[(r["session_id"], r["turn"])], r["n_obligations"])
             for r in scored if r["judge_key"] == primary
             and (r["session_id"], r["turn"]) in rules]
    if len(pairs) >= 3:
        rb = [p[0] >= BUNDLED_THRESHOLD for p in pairs]
        jb = [p[1] >= BUNDLED_THRESHOLD for p in pairs]
        agree = sum(1 for x, y in zip(rb, jb) if x == y) / len(pairs)
        k = cohens_kappa(rb, jb)
        print("  n=%d turns | exact count match %.1f%% | bundled agreement %.1f%%"
              % (len(pairs),
                 sum(1 for a_, b_ in pairs if a_ == b_) / len(pairs) * 100,
                 agree * 100))
        print("  bundled kappa = %s | count rho = %s"
              % ("n/a" if k is None else "%+.3f" % k,
                 "%+.3f" % spearman([p[0] for p in pairs], [p[1] for p in pairs])))
        print("  mean obligations: rules %.2f vs judge %.2f"
              % (st.mean(p[0] for p in pairs), st.mean(p[1] for p in pairs)))
    else:
        print("  not enough overlapping turns")

    # --- 4. inter-judge agreement ------------------------------------------
    inter = {}
    if len(judges) > 1:
        print("\nINTER-JUDGE AGREEMENT (bundled flag, turns both judges scored)")
        by_key = defaultdict(dict)
        for r in scored:
            by_key[(r["session_id"], r["turn"])][r["judge_key"]] = r["n_obligations"]
        for j1, j2 in combinations(judges, 2):
            both = [(v[j1], v[j2]) for v in by_key.values() if j1 in v and j2 in v]
            if len(both) < 3:
                continue
            b1 = [x >= BUNDLED_THRESHOLD for x, _ in both]
            b2 = [y >= BUNDLED_THRESHOLD for _, y in both]
            k = cohens_kappa(b1, b2)
            r_ = spearman([x for x, _ in both], [y for _, y in both])
            inter["%s|%s" % (j1, j2)] = {"n": len(both), "kappa": k, "rho": r_}
            print("  %-22s vs %-22s n=%4d  kappa=%s  rho=%s"
                  % (j1, j2, len(both),
                     "n/a" if k is None else "%+.3f" % k,
                     "n/a" if r_ is None else "%+.3f" % r_))
        print("  (kappa > 0.60 substantial, 0.40-0.60 moderate, < 0.40 poor)")

    OUTPUT.write_text(json.dumps({
        "n_records": len(rows), "n_scored": len(scored), "n_failed": len(failed),
        "judges": judges, "primary_judge": primary, "actual_cost_usd": round(cost, 4),
        "correlations": {k: {"rho": v[0], "n": v[1]} for k, v in gate.items()},
        "inter_judge": inter,
        "leaderboard": lb,
    }, indent=2))
    print("\nWrote %s" % OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
