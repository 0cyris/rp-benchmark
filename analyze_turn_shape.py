#!/usr/bin/env python3
"""Turn shape (conversational floor discipline) across the multi-turn sessions.

For every AI-character turn, counts the *response obligations* it hands back to
the player — questions and direct commands aimed at the player's character.
NPC-to-NPC crosstalk does not count. See harness/turn_shape.py for the rules.

This is layer 1 of the turn-shape work: a rule-based measurement plus the gate
that decides whether the rest is worth building. It answers two questions:

  1. Does turn shape rank-correlate with the existing composite score? A low rho
     means an independent latent that would earn its own leaderboard axis — the
     same argument that keeps Engagement separate (METHODOLOGY 13.2). A high rho
     means it is already captured and belongs as a rubric dimension instead.
  2. Does it predict human multi-turn preference (multiturn_arena_bayesian)?

Ambiguous spans (no resolvable addressee) are reported both ways — counted and
discarded — so the headline is an honest bracket rather than a false point
estimate. Pure prose statistics, no API calls.

Usage:
    python3 analyze_turn_shape.py
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from harness.turn_shape import analyze_turn, build_roster
from generate_profile_cards import wilson_ci
from analyze_method_correlations import spearman

# multiturn_merged_all_v2.json is a strict superset of the phase files (its
# config.sources lists them; the 96 overlapping sessions are byte-identical).
SESSIONS_FILE = Path("results/multiturn_merged_all_v2.json")
SEEDS_FILE = Path("hf_dataset/_source/adversarial_seeds.json")
COMPOSITE_FILE = Path("results/composite_leaderboard.json")
MT_ARENA_FILE = Path("results/multiturn_arena_bayesian.json")
OUTPUT = Path("results/turn_shape.json")


def load_sessions() -> list[dict]:
    """Unique (test_model, seed_id) sessions. There is no session id."""
    data = json.loads(SESSIONS_FILE.read_text())
    seen, out = set(), []
    for s in data["sessions"]:
        key = (s["test_model"], s["seed_id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def safe_load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def mean(xs):
    return round(st.mean(xs), 3) if xs else None


def main():
    sessions = load_sessions()
    seeds = {s["id"]: s for s in json.loads(SEEDS_FILE.read_text())}
    rosters = {sid: build_roster(seed) for sid, seed in seeds.items()}

    per_model = defaultdict(lambda: defaultdict(list))
    per_model_seed = defaultdict(lambda: defaultdict(list))
    per_seed = defaultdict(list)
    skipped_no_seed = set()
    empty_turns = 0

    for sess in sessions:
        model, seed_id = sess["test_model"], sess["seed_id"]
        roster = rosters.get(seed_id)
        if roster is None:
            skipped_no_seed.add(seed_id)
            continue
        char_name = sess.get("character_name")

        for msg in sess["dialogue"]:
            # role is the reliable discriminator; names collide (seed
            # adv_genre_shift_08 has character "Gabi (Narrator)", user "Gabi").
            if msg.get("role") != "character":
                continue
            r = analyze_turn(msg.get("content"), roster, char_name)
            if r["empty"]:
                empty_turns += 1
                continue

            m = per_model[model]
            m["obligations_loose"].append(r["obligations_loose"])
            m["obligations_strict"].append(r["obligations_strict"])
            m["terminal"].append(r["terminal_obligations"])
            m["shape_loose"].append(r["shape_score_loose"])
            m["shape_strict"].append(r["shape_score_strict"])
            m["bundled_loose"].append(r["bundled_loose"])
            m["bundled_strict"].append(r["bundled_strict"])
            m["words"].append(r["words"])
            m["blocks"].append(r["dialogue_blocks"])
            m["has_dialogue"].append(r["spans"] > 0)
            m["has_player_span"].append(r["player_spans"] > 0)
            m["ambiguous"].append(r["ambiguous_spans"])
            m["spans"].append(r["spans"])
            m["obl_per_1k"].append(
                r["obligations_loose"] / max(r["chars"], 1) * 1000
            )

            per_model_seed[model][seed_id].append(r["obligations_loose"])
            per_seed[seed_id].append(r["obligations_loose"])

    # --- per-model aggregation -------------------------------------------
    rows = []
    for model, m in per_model.items():
        n = len(m["obligations_loose"])
        k_loose = sum(m["bundled_loose"])
        k_strict = sum(m["bundled_strict"])
        lo_l, hi_l = wilson_ci(k_loose, n)
        lo_s, hi_s = wilson_ci(k_strict, n)
        total_spans = sum(m["spans"])
        rows.append({
            "model": model,
            "n_turns": n,
            "bundled_rate_loose": round(k_loose / n, 4),
            "bundled_ci_loose": [round(lo_l, 4), round(hi_l, 4)],
            "bundled_rate_strict": round(k_strict / n, 4),
            "bundled_ci_strict": [round(lo_s, 4), round(hi_s, 4)],
            "mean_obligations_loose": mean(m["obligations_loose"]),
            "mean_obligations_strict": mean(m["obligations_strict"]),
            "mean_terminal_obligations": mean(m["terminal"]),
            "mean_shape_score_loose": mean(m["shape_loose"]),
            "mean_shape_score_strict": mean(m["shape_strict"]),
            "obligations_per_1k_chars": mean(m["obl_per_1k"]),
            "mean_dialogue_blocks": mean(m["blocks"]),
            "mean_words": mean(m["words"]),
            "no_dialogue_rate": round(1 - sum(m["has_dialogue"]) / n, 4),
            "no_player_directed_rate": round(1 - sum(m["has_player_span"]) / n, 4),
            "ambiguous_span_rate": (
                round(sum(m["ambiguous"]) / total_spans, 4) if total_spans else None
            ),
        })
    rows.sort(key=lambda r: r["bundled_rate_loose"])

    # --- within-seed ranking (controls for seed cast size) ----------------
    # Several seeds mandate large casts, so a raw cross-model mean partly
    # measures seed assignment. Rank models within each seed, then average.
    seed_ranks = defaultdict(list)
    for seed_id in per_seed:
        scored = [
            (mdl, st.mean(v[seed_id]))
            for mdl, v in per_model_seed.items() if v.get(seed_id)
        ]
        scored.sort(key=lambda x: x[1])
        k = len(scored)
        for i, (mdl, _) in enumerate(scored):
            # Normalize to [0, 1]. Seeds cover different numbers of models
            # (12 seeds have all 21, 8 have only 13), so averaging raw ordinal
            # ranks would reward models that appear on the thinner seeds, where
            # the worst achievable rank is smaller.
            seed_ranks[mdl].append(i / (k - 1) if k > 1 else 0.5)
    within_seed = {m: round(st.mean(r), 4) for m, r in seed_ranks.items()}
    for r in rows:
        r["within_seed_position"] = within_seed.get(r["model"])

    # --- correlation gate --------------------------------------------------
    composite, mt_arena = {}, {}
    cdata = safe_load(COMPOSITE_FILE)
    if cdata:
        composite = {e["model"]: e["composite_score"] for e in cdata["leaderboard"]}
    adata = safe_load(MT_ARENA_FILE)
    if adata:
        mt_arena = {e["model"]: e["elo_mean"] for e in adata["leaderboard"]}

    def rho_against(ref: dict, field: str):
        pairs = [(r[field], ref[r["model"]]) for r in rows if r["model"] in ref]
        if len(pairs) < 3:
            return None, 0
        xs, ys = zip(*pairs)
        return spearman(list(xs), list(ys)), len(pairs)

    # Sign note: bundled_rate is "bad", composite/ELO are "good", so a NEGATIVE
    # rho here means bundling tracks with being ranked worse.
    rho_comp, n_comp = rho_against(composite, "bundled_rate_loose")
    rho_arena, n_arena = rho_against(mt_arena, "bundled_rate_loose")
    # Length-normalized readings. The raw rate tracks response length closely,
    # so these are the ones that say whether turn shape carries any signal of
    # its own once "writes longer turns" is divided out.
    rho_comp_n, _ = rho_against(composite, "obligations_per_1k_chars")
    rho_arena_n, _ = rho_against(mt_arena, "obligations_per_1k_chars")
    length_pairs = [(r["bundled_rate_loose"], r["mean_words"]) for r in rows]
    rho_length = spearman([p[0] for p in length_pairs], [p[1] for p in length_pairs])

    out = {
        "n_sessions": len(sessions),
        "n_models": len(rows),
        "n_turns_scored": sum(r["n_turns"] for r in rows),
        "n_turns_empty": empty_turns,
        "thresholds": {
            "bundled_threshold": 2,
            "penalty_per_obligation": 20,
            "max_penalty": 60,
        },
        "correlations": {
            "bundled_rate_vs_composite": {"rho": rho_comp, "n": n_comp},
            "bundled_rate_vs_mt_arena_elo": {"rho": rho_arena, "n": n_arena},
            "bundled_rate_vs_mean_words": {"rho": rho_length, "n": len(rows)},
            "obligations_per_1k_vs_composite": {"rho": rho_comp_n, "n": n_comp},
            "obligations_per_1k_vs_mt_arena_elo": {"rho": rho_arena_n, "n": n_arena},
        },
        "per_seed_mean_obligations": {
            s: round(st.mean(v), 3) for s, v in sorted(per_seed.items())
        },
        "leaderboard": rows,
        "limitations": [
            "Sessions were driven by a gemini-2.5-flash user simulator instructed "
            "to push the story and ask questions, so some bundling is a response "
            "to the simulated player rather than model habit.",
            "Addressee classification is heuristic; see ambiguous_span_rate and "
            "compare the loose/strict bracket before trusting a ranking.",
        ],
    }
    if skipped_no_seed:
        out["skipped_unknown_seeds"] = sorted(skipped_no_seed)
    OUTPUT.write_text(json.dumps(out, indent=2))

    # --- report ------------------------------------------------------------
    print("TURN SHAPE — player-directed response obligations per AI turn")
    print("%d sessions, %d models, %d turns scored (%d empty/null skipped)\n"
          % (len(sessions), len(rows), out["n_turns_scored"], empty_turns))
    print("%-26s %6s %16s %7s %7s %6s %7s %7s" % (
        "model", "turns", "bundled% (loose)", "strict", "mean", "term", "noDlg%", "amb%"))
    print("-" * 92)
    for r in rows:
        print("%-26s %6d  %5.1f [%4.1f-%4.1f] %6.1f %7.2f %6.2f %6.1f %6.1f" % (
            r["model"], r["n_turns"],
            r["bundled_rate_loose"] * 100,
            r["bundled_ci_loose"][0] * 100, r["bundled_ci_loose"][1] * 100,
            r["bundled_rate_strict"] * 100,
            r["mean_obligations_loose"], r["mean_terminal_obligations"],
            r["no_dialogue_rate"] * 100,
            (r["ambiguous_span_rate"] or 0) * 100,
        ))

    print("\nCORRELATION GATE (bundled_rate vs …; negative = bundling tracks worse ranking)")
    print("  composite_score   rho = %s (n=%d)" % (
        "n/a" if rho_comp is None else "%+.3f" % rho_comp, n_comp))
    print("  mt_arena_elo      rho = %s (n=%d)" % (
        "n/a" if rho_arena is None else "%+.3f" % rho_arena, n_arena))
    print("  mean_words        rho = %s (n=%d)  [length confound]" % (
        "n/a" if rho_length is None else "%+.3f" % rho_length, len(rows)))
    print("  length-normalized (obligations_per_1k_chars):")
    print("    composite       rho = %s" % (
        "n/a" if rho_comp_n is None else "%+.3f" % rho_comp_n))
    print("    mt_arena_elo    rho = %s" % (
        "n/a" if rho_arena_n is None else "%+.3f" % rho_arena_n))
    print("\nWrote %s" % OUTPUT)


if __name__ == "__main__":
    main()
