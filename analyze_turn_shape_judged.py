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

On a corpus generated with two system-prompt arms (run_turn_shape_generation.py)
it also reports the arm contrast: the same model on the same seed with and
without a floor-discipline instruction, paired per cell. That delta is the
steerability reading — whether a model that bundles by default will stop when
asked — and it is the one thing the adversarial corpus structurally cannot
answer, since nothing in its system prompt addresses turn shape at all.

Note on kappa: analyze_round3_kappa.py uses quadratic-weighted kappa over a 1-5
Likert, which degenerates on a binary flag (both values clip into one level), so
plain unweighted Cohen's kappa is computed here instead.

Usage:
    python3 analyze_turn_shape_judged.py

    python3 analyze_turn_shape_judged.py \
        --judged results/turn_shape_corpus_judged.jsonl \
        --sessions results/turn_shape_corpus.json \
        --seeds-file hf_dataset/_source/turn_shape_seeds.json \
        --out results/turn_shape_corpus_judged.json
"""
import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from itertools import combinations
from math import comb
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

# Per-component inter-judge agreement measured on the 3-judge pilot
# (claude_sonnet / gpt_4_1 / gemini_3_1_pro, 459 turns each). Used to label
# reported rates when the loaded data has only one judge and live kappa cannot
# be computed. Recomputed from the data whenever >= 2 judges are present.
PILOT_KAPPA = {
    "player_present": 0.715,
    "crosstalk_present": 0.647,
    "handholds_not_none": 0.439,
    "monologue": 0.512,
    "demands": 0.466,
    "pc_interiority_leak": 0.314,
    "handback_clean": 0.285,
}


def reliability_label(kappa: float | None) -> str:
    """Landis-Koch bands, the same ones the kappa printout already uses."""
    if kappa is None:
        return "unknown"
    if kappa >= 0.6:
        return "substantial"
    if kappa >= 0.4:
        return "moderate"
    return "WEAK"


# The components behind the headline and the secondary flags, as predicates, so
# agreement can be measured on each one rather than only on their conjunction.
COMPONENTS = [
    ("player_present", lambda r: bool(r.get("player_present", True))),
    ("demands", lambda r: r.get("n_obligations", 0) <= 1),
    ("crosstalk_present", lambda r: bool(r.get("crosstalk_present"))),
    ("monologue", lambda r: bool(r.get("monologue"))),
    ("handholds_not_none", lambda r: r.get("handholds") != "none"),
    ("pc_interiority_leak", lambda r: bool(r.get("pc_interiority_leak"))),
    ("handback_clean", lambda r: r.get("handback") == "clean"),
]

def is_well_shaped(r: dict) -> bool:
    """The player can still act, and the turn handed them at most one thing.

    Reduced from a five-way conjunction after the three-judge pilot. The dropped
    terms are the two least reliable fields in the schema -- handback == clean
    falls to kappa +0.145 between judges and pc_interiority_leak to +0.215 --
    and a conjunction inherits the noise of its worst member, which is why the
    old verdict scored 0.25-0.40 while its best components reached +0.82.

    Measured on the pilot (459 turns x 3 judges), mean pairwise kappa:
        five-way AND         +0.347
        drop handback        +0.357
        drop handback+leak   +0.431
        this form            +0.479   (all pairs >= +0.398)

    It stays independent of the composite (rho = -0.103), of human multi-turn
    arena ELO (-0.070), and of response length (-0.456), with 36 points of
    spread across 21 models -- so the reliability was bought without losing
    what made the axis worth having.

    handholds is not needed here: handholds == "none" was collinear with
    player_present == False in every observation, so the "nothing to act on"
    case the GM-card spec cares about is already carried by player_present.
    The dropped fields are still recorded and still reported, with their
    measured agreement attached.
    """
    return r.get("player_present", True) and r.get("n_obligations", 0) <= 1


# Kept under the old name so nothing silently changes meaning mid-file.
is_clean = is_well_shaped


def load_judged(path: Path = JUDGED) -> list[dict]:
    rows = []
    with open(path) as f:
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


def _share(rows: list[dict], pred) -> float | None:
    """Fraction of rows satisfying pred, or None when there are none."""
    return (sum(1 for r in rows if pred(r)) / len(rows)) if rows else None


def _mean_words(rows: list[dict], turn_words: dict) -> float | None:
    """Mean words per turn for these judged rows, joined on (session_id, turn)."""
    vals = [turn_words[(r["session_id"], r["turn"])] for r in rows
            if (r["session_id"], r["turn"]) in turn_words]
    return round(st.mean(vals), 1) if vals else None


def sign_test(up: int, down: int) -> float | None:
    """Two-sided exact sign test on paired seeds. None when nothing moved.

    Each cell is only ~9 turns, so a per-model arm delta of +/-13 points can
    come out of sampling alone -- the synthetic check produced exactly that
    from a model planted with no real effect. With at most 6 paired seeds this
    can never beat p=0.031, which is the honest ceiling of the design and the
    reason the delta is reported with its p rather than on its own.
    """
    n = up + down
    if n == 0:
        return None
    k = min(up, down)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def rule_based_per_turn(sessions_file: Path = SESSIONS_FILE,
                        seeds_file: Path = SEEDS_FILE) -> tuple[dict, dict]:
    """Per-turn data read straight off the session corpus. No API calls.

    Returns ({(session_id, turn): rule_based_obligations},
             {(session_id, turn): word_count}).

    Words come back separately because the arm contrast needs them: the shape
    instruction cut mean turn length 19% in the smoke test, and the headline is
    negatively correlated with length (rho = -0.456 on the pilot), so "bought
    shape" and "bought brevity" are indistinguishable without length on the
    table. results/turn_shape.json cannot supply them -- it only covers the
    adversarial corpus, and has no per-arm split.
    """
    data = json.loads(sessions_file.read_text())
    seeds = {s["id"]: s for s in json.loads(seeds_file.read_text())}
    rosters = {sid: build_roster(s) for sid, s in seeds.items()}
    obligations, words, seen = {}, {}, set()
    for sess in data["sessions"]:
        key = (sess["test_model"], sess["seed_id"], sess.get("arm"))
        if key in seen:
            continue
        seen.add(key)
        # Must reproduce judge_turn_shape.build_work's id exactly, or the
        # rules-vs-judge join silently finds nothing to compare.
        sid = "%s::%s" % (sess["test_model"], sess["seed_id"])
        if sess.get("arm"):
            sid += "::%s" % sess["arm"]
        roster = rosters.get(sess["seed_id"])
        for msg in sess["dialogue"]:
            if msg.get("role") != "character":
                continue
            content = msg.get("content") or ""
            words[(sid, msg["turn"])] = len(content.split())
            if roster is None:
                continue  # unknown seed: no rule-based count, words still fine
            r = analyze_turn(content, roster, sess.get("character_name"))
            if not r["empty"]:
                obligations[(sid, msg["turn"])] = r["obligations_loose"]
    return obligations, words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", default=str(JUDGED),
                    help="Judge output .jsonl. Default: %s" % JUDGED)
    ap.add_argument("--sessions", default=str(SESSIONS_FILE),
                    help="Session corpus the judged rows came from (used only "
                         "for the free rule-based comparison).")
    ap.add_argument("--seeds-file", default=str(SEEDS_FILE),
                    help="Seed set matching --sessions.")
    ap.add_argument("--out", default=str(OUTPUT))
    args = ap.parse_args()

    judged_path = Path(args.judged)
    if not judged_path.exists():
        print("No %s yet — run judge_turn_shape.py first." % judged_path)
        print("  smoke:  python3 judge_turn_shape.py --judges claude_sonnet --limit 20")
        return 1

    rows = load_judged(judged_path)
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

    # Read once: the free rule-based counts and the per-turn word counts, both
    # straight off the session corpus.
    try:
        rules, turn_words = rule_based_per_turn(Path(args.sessions),
                                                Path(args.seeds_file))
    except (OSError, json.JSONDecodeError) as e:
        print("Could not read --sessions/--seeds-file (%s); skipping the"
              " rules comparison and the length column.\n" % e)
        rules, turn_words = {}, {}

    # --- seeds that cannot move the headline ------------------------------
    # A seed whose verdict is the same for every model in every arm carries no
    # ranking signal, and averaging it in dilutes the spread. ts_empty_room_06
    # is the built case: with no NPC in the scene nothing can address the
    # player, so n_obligations is structurally 0 and the headline returns a
    # free 100% whatever the model does. That seed is still worth running --
    # its target is handholds and interiority leak, reported below -- it just
    # must not be scored on a metric that cannot represent it.
    by_seed_verdicts = defaultdict(set)
    for r in scored:
        by_seed_verdicts[r["seed"]].add(is_clean(r))
    constant_seeds = {s for s, v in by_seed_verdicts.items() if len(v) == 1}

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
            "mean_words": _mean_words(rs, turn_words),
        })
        # The same headline over discriminating seeds only. Where a constant
        # seed exists this is the number that should drive the ranking; the
        # diluted one stays visible beside it rather than being replaced.
        disc = [r for r in live if r["seed"] not in constant_seeds]
        lb[-1]["clean_rate_excl_constant"] = (
            round(sum(1 for r in disc if is_clean(r)) / len(disc), 4)
            if disc else None)
    lb.sort(key=lambda r: -(r["clean_rate_excl_constant"]
                            if r["clean_rate_excl_constant"] is not None
                            else (r["clean_handback_rate"] or 0)))

    # --- component reliability ------------------------------------------
    by_key = defaultdict(dict)
    for r in scored:
        by_key[(r["session_id"], r["turn"])][r["judge_key"]] = r
    live_kappa = {}
    if len(judges) > 1:
        for name, fn in COMPONENTS:
            ks = []
            for j1, j2 in combinations(judges, 2):
                both = [(v[j1], v[j2]) for v in by_key.values()
                        if j1 in v and j2 in v]
                if len(both) < 3:
                    continue
                k, _, _, _ = cohens_kappa([fn(a) for a, _ in both],
                                          [fn(b) for _, b in both])
                if k is not None:
                    ks.append(k)
            if ks:
                live_kappa[name] = sum(ks) / len(ks)

    src = live_kappa or PILOT_KAPPA
    print("COMPONENT RELIABILITY (mean pairwise kappa, %s)"
          % ("this data, %d judges" % len(judges) if live_kappa
             else "from the 3-judge pilot — NOT from this single-judge data"))
    for name, _ in COMPONENTS:
        k = src.get(name)
        print("  %-22s %s  %s"
              % (name, "  n/a" if k is None else "%+.3f" % k,
                 reliability_label(k)))
    print("  headline = player_present AND demands<=1; the two weakest fields"
          " (handback_clean, pc_interiority_leak) are reported but NOT in it.")
    print()

    print("Primary judge: %s" % primary)
    print("Headline: well-shaped = player present AND <=1 demand."
          " Not comparable to the pilot's 5-way definition.")
    # The pilot's +0.479 was measured where player_present varied. In a
    # narrator corpus it never goes false, so the conjunction collapses to its
    # demands term and inherits that term's reliability instead. Say so rather
    # than carrying the better number over.
    if len({r.get("player_present", True) for r in scored}) == 1:
        print("  NOTE: player_present is constant in this data, so the headline"
              " has reduced to 'demands <= 1'.")
        print("        Reliability is that term's alone: kappa %+.3f (%s), not"
              " the pilot conjunction's +0.479."
              % (src.get("demands", 0.0), reliability_label(src.get("demands"))))
    if constant_seeds:
        print("  NOTE: %d seed(s) score constant for every model and arm (%s)."
              % (len(constant_seeds), ", ".join(sorted(constant_seeds))))
        print("        excl-const% drops them; it is the column that carries"
              " ranking signal.")
    print()
    print("%-24s %5s %17s %10s %6s %6s %6s %6s %7s %6s" % (
        "model", "live", "well-shaped%", "excl-const", "dem0", "dem1", "dem2+",
        "leak%", "meanObl", "words"))
    print("-" * 104)
    for r in lb:
        ec = r["clean_rate_excl_constant"]
        print("%-24s %5d  %5.1f [%4.1f-%4.1f] %9s %5.0f%% %5.0f%% %5.0f%%"
              " %5.0f%% %6.2f %6s" % (
            r["model"], r["n_live_turns"], (r["clean_handback_rate"] or 0) * 100,
            r["clean_handback_ci"][0] * 100, r["clean_handback_ci"][1] * 100,
            "-" if ec is None else "%.1f%%" % (ec * 100),
            (r["demands_0"] or 0) * 100, (r["demands_1"] or 0) * 100,
            (r["demands_2plus"] or 0) * 100,
            (r["pc_interiority_leak_rate"] or 0) * 100,
            r["mean_obligations"],
            "-" if r["mean_words"] is None else "%.0f" % r["mean_words"]))
    print("  meanObl is the continuous companion: judges agree on the count at"
          " rho +0.63..+0.75, better than on any binary derived from it.")
    print("  leak%% is reported at reliability '%s' — read it as indicative."
          % reliability_label(src.get("pc_interiority_leak")))

    # --- 1b. arm contrast: steerability ------------------------------------
    # Only fires on a corpus generated with two system-prompt arms. Paired per
    # (model, seed) cell: same model, same scene, one paragraph of prompt
    # different, so the delta is a within-subject effect and the seed-difficulty
    # spread that confounds the cross-model rate cancels out of it.
    arms = sorted({r["arm"] for r in scored if r.get("arm")})
    arm_report = {}
    if len(arms) == 2:
        base_arm, test_arm = arms if arms[0] == "unprompted" else arms[::-1]
        cells = defaultdict(list)
        for r in scored:
            if r["judge_key"] == primary and r.get("arm"):
                cells[(r["model"], r["seed"], r["arm"])].append(r)

        def cell_rate(rs):
            live = [x for x in rs if x.get("player_present", True)]
            return (sum(1 for x in live if is_clean(x)) / len(live)) if live else None

        # Report the headline delta beside a delta for each component, because
        # the headline alone can read zero while the instruction is plainly
        # working: on the NPC-free seed the smoke test cut interiority leak and
        # turn length sharply and the headline did not move a point, since with
        # no NPC present nothing can address the player. The dropped fields stay
        # out of the headline and are shown here with their agreement attached,
        # the same discipline the leaderboard's leak% column already uses.
        ARM_METRICS = [
            ("headline", None, lambda rs: cell_rate(rs)),
            ("demands<=1", "demands",
             lambda rs: _share(rs, lambda x: x.get("n_obligations", 0) <= 1)),
            ("map", "handholds_not_none",
             lambda rs: _share(rs, lambda x: x.get("handholds") == "map")),
            ("genericQ", "handholds_not_none",
             lambda rs: _share(rs, lambda x: x.get("handholds") == "generic_prompt")),
            ("leak", "pc_interiority_leak",
             lambda rs: _share(rs, lambda x: bool(x.get("pc_interiority_leak")))),
            ("over_res", "handback_clean",
             lambda rs: _share(rs, lambda x: x.get("handback") == "over_resolved")),
        ]

        print("\nARM CONTRAST — does asking for the shape get it? (paired by"
              " model x seed, primary judge)")
        print("%-24s %10s %10s %8s %10s %7s"
              % ("model", base_arm[:10], test_arm[:10], "delta", "seeds +/-", "p"))
        print("-" * 76)
        rowsy = []
        for model in sorted({m for m, _, _ in cells}):
            paired, comp_deltas, word_pairs = [], defaultdict(list), []
            for seed_id in sorted({s for m, s, _ in cells if m == model}):
                brs = cells.get((model, seed_id, base_arm), [])
                trs = cells.get((model, seed_id, test_arm), [])
                if not brs or not trs:
                    continue
                b, t = cell_rate(brs), cell_rate(trs)
                if b is not None and t is not None:
                    paired.append((b, t))
                for label, _, fn in ARM_METRICS[1:]:
                    bv, tv = fn(brs), fn(trs)
                    if bv is not None and tv is not None:
                        comp_deltas[label].append(tv - bv)
                bw = _mean_words(brs, turn_words)
                tw = _mean_words(trs, turn_words)
                if bw and tw:
                    word_pairs.append((bw, tw))
            if not paired:
                continue
            mb = st.mean(p[0] for p in paired)
            mt = st.mean(p[1] for p in paired)
            up = sum(1 for b, t in paired if t > b)
            down = sum(1 for b, t in paired if t < b)
            rowsy.append({
                "model": model, "n_paired_seeds": len(paired),
                base_arm: round(mb, 4), test_arm: round(mt, 4),
                "delta": round(mt - mb, 4), "seeds_up": up, "seeds_down": down,
                "sign_test_p": sign_test(up, down),
                "component_deltas": {k: round(st.mean(v), 4)
                                     for k, v in comp_deltas.items()},
                "words_delta_pct": (
                    round((st.mean(t for _, t in word_pairs)
                           / st.mean(b for b, _ in word_pairs) - 1) * 100, 1)
                    if word_pairs else None),
            })
        rowsy.sort(key=lambda r: -r["delta"])
        for r in rowsy:
            p_ = r["sign_test_p"]
            print("%-24s %9.0f%% %9.0f%% %+7.0f%% %6d/%d %7s"
                  % (r["model"], r[base_arm] * 100, r[test_arm] * 100,
                     r["delta"] * 100, r["seeds_up"], r["seeds_down"],
                     "n/a" if p_ is None else "%.3f" % p_))
        arm_report = {"base": base_arm, "test": test_arm, "per_model": rowsy}

        if rowsy:
            comp_labels = [lbl for lbl, _, _ in ARM_METRICS[1:]]
            print("\n  COMPONENT DELTAS (%s minus %s; the headline alone can"
                  " read zero while the" % (test_arm, base_arm))
            print("  instruction is plainly working — that is what the"
                  " NPC-free smoke test did)")
            # Built from the label list rather than a fixed slot count, so
            # adding a component cannot silently desynchronize the header.
            head = "  %-22s %9s" % ("model", "headline")
            head += "".join("%11s" % lbl for lbl in comp_labels)
            head += "%9s" % "words%"
            print(head)
            print("  " + "-" * (len(head) - 2))
            for r in rowsy:
                cd = r["component_deltas"]
                cells_txt = "".join(
                    "%11s" % ("-" if lbl not in cd else "%+.0f%%" % (cd[lbl] * 100))
                    for lbl in comp_labels)
                print("  %-22s %+8.0f%%%s %9s"
                      % (r["model"], r["delta"] * 100, cells_txt,
                         "-" if r["words_delta_pct"] is None
                         else "%+.0f%%" % r["words_delta_pct"]))
            print("  reliability: %s"
                  % "  ".join("%s=%s" % (lbl, reliability_label(src.get(key)))
                              for lbl, key, _ in ARM_METRICS[1:] if key))
            print("  leak and over_res are WEAK fields kept OUT of the"
                  " headline; they are shown here with")
            print("  their agreement attached, as indicative, not as a verdict."
                  " words% guards the other reading:")
            print("  a prompt that only shortens turns buys brevity, not shape"
                  " (headline vs length rho = -0.456).")

        if rowsy:
            print("  Read the delta as steerability. Low base + high delta ="
                  " usable if you ask for it;")
            print("  low in both = cannot hold the floor even when told to,"
                  " which is the finding that matters.")
            n_seeds = max(r["n_paired_seeds"] for r in rowsy)
            print("  p is a two-sided sign test over paired seeds; with %d of"
                  " them it bottoms out at %.3f,"
                  % (n_seeds, sign_test(n_seeds, 0) or 1.0))
            print("  so treat a delta whose p is not small as unmeasured, not"
                  " as zero.")
            if all(abs(r["delta"]) < 0.02 for r in rowsy):
                print("  <-- WARNING: delta ~0 for every model. Suspect the"
                      " instruction or its placement")
                print("      before believing that no model is steerable.")
    elif arms:
        print("\nOnly one arm present (%s); no steerability contrast to draw."
              % arms[0])

    # --- 1c. per-seed difficulty -------------------------------------------
    per_seed = defaultdict(list)
    for r in scored:
        if r["judge_key"] == primary and r.get("player_present", True):
            per_seed[r["seed"]].append(r)
    if len(per_seed) > 1:
        fully_crossed = len({
            (r["model"], r.get("arm")) for r in scored if r["judge_key"] == primary
        }) * len(per_seed) == len({
            (r["model"], r["seed"], r.get("arm")) for r in scored
            if r["judge_key"] == primary
        })
        print("\nPER-SEED (well-shaped%%, primary judge) — %s"
              % ("fully crossed, so this is seed difficulty"
                 if fully_crossed else
                 "NOT fully crossed: seed difficulty and model assignment are"
                 " confounded here"))
        for seed_id, rs in sorted(per_seed.items(),
                                  key=lambda kv: -sum(1 for r in kv[1] if is_clean(r)) / len(kv[1])):
            flag = ""
            if seed_id in constant_seeds:
                why = ("no NPC in the scene, so nothing can address the player"
                       " and obligations are structurally 0"
                       if all(r["n_obligations"] == 0 for r in rs)
                       else "same verdict for every model and arm")
                flag = "  <-- CONSTANT: %s; no ranking signal" % why
            print("  %-32s %5.1f%%  (n=%d, %d models)%s"
                  % (seed_id, sum(1 for r in rs if is_clean(r)) / len(rs) * 100,
                     len(rs), len({r["model"] for r in rs}), flag))
        if constant_seeds:
            disc = [r for rs in per_seed.values() for r in rs
                    if r["seed"] not in constant_seeds]
            if disc:
                print("  Headline over the %d discriminating seed(s): %.1f%%"
                      " (vs %.1f%% with the constant one(s) averaged in)."
                      % (len(per_seed) - len(constant_seeds),
                         sum(1 for r in disc if is_clean(r)) / len(disc) * 100,
                         sum(1 for rs in per_seed.values() for r in rs
                             if is_clean(r))
                         / sum(len(rs) for rs in per_seed.values()) * 100))

    sec = [(r["model"], r["npc_grading_rate"], r["monologue_rate"],
            r["manufactured_tension_rate"], r["handholds_map_rate"],
            r["generic_prompt_rate"]) for r in lb]
    print("\nSECONDARY (anti-patterns from the same spine) — reliability: "
          "grading=%s monolog=%s leak-not-shown-here"
          % (reliability_label(src.get("npc_grading")),
             reliability_label(src.get("monologue"))))
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

    Path(args.out).write_text(json.dumps({
        "n_records": len(rows), "n_scored": len(scored), "n_failed": len(failed),
        "judges": judges, "primary_judge": primary, "actual_cost_usd": round(cost, 4),
        "correlations": {k: {"rho": v[0], "n": v[1]} for k, v in gate.items()},
        "inter_judge": inter,
        "arm_contrast": arm_report,
        "constant_seeds": sorted(constant_seeds),
        "leaderboard": lb,
    }, indent=2))
    print("\nWrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
