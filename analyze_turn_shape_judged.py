#!/usr/bin/env python3
"""Analysis for the LLM-judged turn-shape metric (layer 2).

Reads results/turn_shape_judged.jsonl (produced by judge_turn_shape.py) and
reports four things:

  1. Per-model judged leaderboard, headlined by clean-handback rate: the share of
     live turns that describe-then-stop, leave something to act on, and stack at
     most one demand. Obligation count is one input, not the headline — per the
     GM-card spec a well-formed turn often asks nothing at all, carrying the handoff
     through the affordances in the scene ("scene-as-map"), and a bare "what do you
     do?" is the weaker handoff. What separates a good silent turn from a bad one is
     whether anything was left to act on.
  2. The correlation gate — clean-handback and demand-stacking against the
     composite, human multi-turn arena ELO, and response length. Low correlation
     with the composite is what would justify a separate leaderboard axis
     (METHODOLOGY 13.2).
  3. Rules vs judge — per-turn agreement between this and harness/turn_shape.py.
     If the judge merely reproduces the rule-based numbers, the spend bought
     nothing and that should be visible.
  4. Inter-judge agreement — Cohen's kappa on the clean-handback verdict for every
     judge pair on turns they both scored. The benchmark's single-judge dependence
     is its most-documented weakness; this metric should not repeat it.

Note on kappa: analyze_round3_kappa.py uses quadratic-weighted kappa over a 1-5
Likert, which degenerates on a binary flag (both values clip into one level), so
plain unweighted Cohen's kappa is computed here instead.

Usage:
    python3 analyze_turn_shape_judged.py
"""
import json
import statistics as st
import sys
from collections import Counter, defaultdict
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

# A turn counts as well-shaped when the player can still act, the turn stops where
# they act, it leaves purchase, and it does not stack demands or answer for them.
def is_clean(r: dict) -> bool:
    return (r.get("player_present", True)
            and r.get("handback") == "clean"
            and r.get("handholds") in ("map", "generic_prompt")
            and r.get("n_obligations", 0) <= 1
            and not r.get("pc_interiority_leak"))


def load_judged() -> list[dict]:
    rows = []
    with open(JUDGED) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def cohens_kappa(a: list[bool], b: list[bool]) -> tuple:
    """Unweighted Cohen's kappa, plus the raw numbers behind it.

    Returns (kappa, observed_agreement, positive_rate_a, positive_rate_b).
    kappa is None when it is undefined (no data, or both raters constant).

    The raw agreement matters as much as kappa here: clean_handback runs ~80%
    positive, and kappa is base-rate sensitive, so two judges agreeing on 90%
    of turns can still post a mediocre kappa purely from the skew. Reporting
    kappa alone would invite reading a usable metric as a broken one.
    """
    n = len(a)
    if n == 0:
        return None, None, None, None
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    exp = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if exp >= 1.0:
        return None, obs, pa1, pb1  # both raters constant: kappa undefined
    return (obs - exp) / (1 - exp), obs, pa1, pb1


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
    all_scored = [r for r in rows if "n_obligations" in r]
    failed = [r for r in rows if r.get("parse_error") or r.get("error")]
    cost = sum((r.get("usage") or {}).get("cost") or 0.0 for r in rows)

    # Never aggregate across prompt versions. A row written before the shape
    # fields existed has no handback, so is_clean() would score it unclean and
    # the ranking would become an artifact of which rows predate a prompt edit.
    shaped = [r for r in all_scored if "handback" in r]
    dominant = (Counter(r.get("prompt_sha", "legacy") for r in shaped)
                .most_common(1)[0][0] if shaped else None)
    scored = [r for r in shaped if r.get("prompt_sha", "legacy") == dominant]
    stale = [r for r in all_scored if r not in scored]

    judges = sorted({r["judge_key"] for r in scored}) or sorted(
        {r["judge_key"] for r in rows})

    print("JUDGED TURN SHAPE — response obligations and handback quality")
    print("%d turn records (%d scored, %d failed) across %d judge(s): %s"
          % (len(rows), len(all_scored), len(failed), len(judges),
             ", ".join(judges)))
    print("Prompt version: %s | Actual spend: $%.2f" % (dominant, cost))
    if stale:
        by_model = Counter(r["model"] for r in stale)
        print("\n" + "!" * 72)
        print("EXCLUDED %d row(s) written under a different prompt version."
              % len(stale))
        for m, n in sorted(by_model.items()):
            total = sum(1 for r in all_scored if r["model"] == m)
            note = "  <-- ALL rows stale, model not ranked" if n == total else ""
            print("   %-24s %3d/%d stale%s" % (m, n, total, note))
        print("Refresh them:  python3 judge_turn_shape.py --stale-only")
        print("!" * 72)
    print()
    if not scored:
        print("No rows at the current prompt version. Nothing to report.")
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
        live = [r for r in rs if r.get("player_present", True)]
        nl = len(live)
        k = sum(1 for r in live if is_clean(r))
        lo, hi = wilson_ci(k, nl) if nl else (0, 0)
        def rate(pred, rows=None):
            rows = rs if rows is None else rows
            return round(sum(1 for r in rows if pred(r)) / len(rows), 4) if rows else None
        lb.append({
            "model": model,
            "n_turns": n,
            "n_live_turns": nl,
            "clean_handback_rate": round(k / nl, 4) if nl else None,
            "clean_handback_ci": [round(lo, 4), round(hi, 4)],
            "dead_scene_rate": rate(lambda r: not r.get("player_present", True)),
            "demands_0": rate(lambda r: r["n_obligations"] == 0, live),
            "demands_1": rate(lambda r: r["n_obligations"] == 1, live),
            "demands_2plus": rate(
                lambda r: r["n_obligations"] >= BUNDLED_THRESHOLD, live),
            "handholds_map_rate": rate(lambda r: r.get("handholds") == "map", live),
            "generic_prompt_rate": rate(
                lambda r: r.get("handholds") == "generic_prompt", live),
            "no_opening_rate": rate(
                lambda r: r.get("handholds") == "none"
                or r.get("handback") == "no_opening", live),
            "over_resolved_rate": rate(
                lambda r: r.get("handback") == "over_resolved", live),
            "pc_interiority_leak_rate": rate(
                lambda r: r.get("pc_interiority_leak")),
            "npc_grading_rate": rate(lambda r: r.get("npc_grading")),
            "monologue_rate": rate(lambda r: r.get("monologue")),
            "manufactured_tension_rate": rate(
                lambda r: r.get("manufactured_tension")),
            "mean_obligations": round(st.mean(r["n_obligations"] for r in rs), 3),
        })
    lb.sort(key=lambda r: -(r["clean_handback_rate"] or 0))

    print("Primary judge: %s" % primary)
    print("Headline: clean handback = player present, stops where they act,"
          " leaves purchase, <=1 demand, no interiority leak.\n")
    print("%-24s %5s %17s %6s %6s %6s %6s %6s %6s" % (
        "model", "live", "clean handback%", "dem0", "dem1", "dem2+",
        "dead%", "noOpn%", "leak%"))
    print("-" * 92)
    for r in lb:
        print("%-24s %5d  %5.1f [%4.1f-%4.1f] %5.0f%% %5.0f%% %5.0f%% %5.0f%%"
              " %5.0f%% %5.0f%%" % (
            r["model"], r["n_live_turns"], (r["clean_handback_rate"] or 0) * 100,
            r["clean_handback_ci"][0] * 100, r["clean_handback_ci"][1] * 100,
            (r["demands_0"] or 0) * 100, (r["demands_1"] or 0) * 100,
            (r["demands_2plus"] or 0) * 100, (r["dead_scene_rate"] or 0) * 100,
            (r["no_opening_rate"] or 0) * 100,
            (r["pc_interiority_leak_rate"] or 0) * 100))

    sec = [(r["model"], r["npc_grading_rate"], r["monologue_rate"],
            r["manufactured_tension_rate"], r["handholds_map_rate"],
            r["generic_prompt_rate"]) for r in lb]
    print("\nSECONDARY (anti-patterns from the same spine)")
    print("%-24s %8s %8s %10s %8s %9s"
          % ("model", "grading%", "monolog%", "spawnTens%", "map%", "genericQ%"))
    print("-" * 74)
    for m, g, mo, mt, hm, gp in sec:
        print("%-24s %7.0f%% %7.0f%% %9.0f%% %7.0f%% %8.0f%%"
              % (m, (g or 0) * 100, (mo or 0) * 100, (mt or 0) * 100,
                 (hm or 0) * 100, (gp or 0) * 100))

    print("\nFIELD VARIANCE (a field that never varies is not contributing)")
    for field in ("handholds", "handback", "player_present", "pc_interiority_leak",
                  "npc_grading", "monologue", "manufactured_tension"):
        dist = Counter(r.get(field) for r in scored)
        flat = (" <-- CONSTANT, contributes nothing to the headline"
                if len(dist) == 1 else "")
        print("  %-22s %s%s"
              % (field, ", ".join("%s=%d" % kv for kv in dist.most_common()), flat))
    # Collinearity: if the three shape fields only ever move together they are
    # one signal wearing three names, and is_clean() reduces to that one signal.
    combos = Counter((r.get("player_present"), r.get("handholds"), r.get("handback"))
                     for r in scored)
    print("  distinct (present, handholds, handback) combos: %d" % len(combos))
    for c, n in combos.most_common():
        print("      %-40s %d" % (str(c), n))
    if len(combos) <= 2:
        print("  <-- COLLINEAR: these fields move together, so the headline"
              " reduces to one signal")

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
        r["mean_words"] = w

    def rho(ref, field):
        pairs = [(r[field], ref[r["model"]]) for r in lb
                 if r["model"] in ref and r.get(field) is not None]
        if len(pairs) < 3:
            return None, len(pairs)
        return spearman([p[0] for p in pairs], [p[1] for p in pairs]), len(pairs)

    gate = {
        "clean_handback_vs_composite": rho(composite, "clean_handback_rate"),
        "clean_handback_vs_mt_arena_elo": rho(mt_arena, "clean_handback_rate"),
        "clean_handback_vs_mean_words": rho(words, "clean_handback_rate"),
        "demands_2plus_vs_composite": rho(composite, "demands_2plus"),
        "demands_2plus_vs_mt_arena_elo": rho(mt_arena, "demands_2plus"),
        "per_1k_vs_composite": rho(composite, "obligations_per_1k_words"),
        "per_1k_vs_mt_arena_elo": rho(mt_arena, "obligations_per_1k_words"),
    }
    print("\nCORRELATION GATE (clean_handback is GOOD: positive = tracks a better"
          " ranking. demands_2plus is BAD: negative tracks better.)")
    for name, (v, n) in gate.items():
        print("  %-26s rho = %s (n=%d)"
              % (name, "n/a" if v is None else "%+.3f" % v, n))

    # --- 3. rules vs judge -------------------------------------------------
    print("\nRULES vs JUDGE (obligation counts per turn, primary judge)")
    rules = rule_based_per_turn()
    pairs = [(rules[(r["session_id"], r["turn"])], r["n_obligations"])
             for r in scored if r["judge_key"] == primary
             and (r["session_id"], r["turn"]) in rules]
    if len(pairs) >= 3:
        rb = [p[0] >= BUNDLED_THRESHOLD for p in pairs]
        jb = [p[1] >= BUNDLED_THRESHOLD for p in pairs]
        agree = sum(1 for x, y in zip(rb, jb) if x == y) / len(pairs)
        k, _, _, _ = cohens_kappa(rb, jb)
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
        print("\nINTER-JUDGE AGREEMENT (clean-handback verdict; rho on raw"
              " obligation counts)")
        by_key = defaultdict(dict)
        for r in scored:
            by_key[(r["session_id"], r["turn"])][r["judge_key"]] = (
                r["n_obligations"], is_clean(r))
        for j1, j2 in combinations(judges, 2):
            both = [(v[j1], v[j2]) for v in by_key.values() if j1 in v and j2 in v]
            if len(both) < 3:
                continue
            b1 = [x[1] for x, _ in both]
            b2 = [y[1] for _, y in both]
            k, obs, p1, p2 = cohens_kappa(b1, b2)
            r_ = spearman([x[0] for x, _ in both], [y[0] for _, y in both])
            inter["%s|%s" % (j1, j2)] = {
                "n": len(both), "kappa": k, "rho": r_,
                "observed_agreement": obs,
                "positive_rate": [p1, p2],
            }
            print("  %-20s vs %-20s n=%4d  kappa=%s  agree=%5.1f%%"
                  "  pos=%.0f%%/%.0f%%  rho=%s"
                  % (j1, j2, len(both),
                     "n/a" if k is None else "%+.3f" % k,
                     obs * 100, p1 * 100, p2 * 100,
                     "n/a" if r_ is None else "%+.3f" % r_))
        print("  (kappa > 0.60 substantial, 0.40-0.60 moderate, < 0.40 poor)")
        print("  Read kappa against agree% and pos%: a low kappa alongside high")
        print("  agreement and a lopsided positive rate is base-rate skew, not")
        print("  judges disagreeing about what a clean handback is.")

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
