# What RP-Bench Means by "Good Roleplay"

And how much of that definition you can change.

This document answers two questions that the methodology docs answer only implicitly:

1. **What does this benchmark actually reward?** Not the mechanics — the taste.
2. **How much of that taste is a knob you can turn**, and where the knobs are.

For formulas and sample sizes see [`METHODOLOGY.md`](METHODOLOGY.md). For why the
benchmark is failure-oriented see [`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md).

---

## 1. The short answer

RP-Bench does not have one definition of good roleplay. It has **five stacked
definitions**, and they measurably disagree with each other:

| Layer | Operational definition of "good" | Where it lives |
|---|---|---|
| **Philosophy** | Good RP = RP that doesn't *break*. Reliability, not excellence. | `EXPERIMENT_DESIGN.md` §1 |
| **27-dim rubric** | Good RP = a specific craft aesthetic — restraint, subtext, friction, imperfection. | `prompts/judge_claude.md` |
| **Flaw hunter** | Good RP = the absence of 17 named, quotable defects. | `prompts/judge_flaw_hunter.md` |
| **Rule-based** | Good RP = prose that avoids a curated list of AI stylistic tics. | `harness/slop_detectors.py`, `harness/objective_metrics.py` |
| **Humans** | Good RP = whatever the arena voters clicked. | `data/multiturn_arena_votes.jsonl` |

The headline finding of the whole project is that these layers **do not converge**.
LLM-judge multi-turn scores rank-correlate at ρ = −0.43 with community ELO. The
flaw hunter agrees with real users on only 38.7% of swipe pairs (n=75) — worse
than its 50.7% disagreement rate. So when you read a leaderboard here, you are
reading *one* of these definitions, and the doc that generated it says which.

---

## 2. The philosophical layer: "not bad" beats "good"

The founding premise is a refusal to define quality:

> *"I can't tell you what's good RP, but I can certainly tell you what's bad RP."*

The design doc argues RP quality is not a uni-dimensional verifiable construct —
one player's purple prose is another's evocative writing — so the benchmark
measures **failure rates** instead, in the tradition of adverse-event rates in
medicine and failure rates in reliability engineering. The leaderboard is meant to
answer *"which model is least likely to ruin my session?"*, not *"which model
writes the best prose?"*

The community arena empirically supported this framing: attentive voters rejected
5 of 6 catch-pair failure archetypes at a 100% rate (refusal, repetition,
wrong-scene, truncation, meta-commentary). The 6th catch — *emotional hijack vs.
literary restraint* — split 74/26. **People agree on what's broken; they don't
agree on what's beautiful.**

Note the tension: the benchmark's stated philosophy is anti-quality-score, yet it
ships a 27-dimension quality rubric and a composite headline number. Sections 3–5
are that quality rubric's actual, quite opinionated taste.

---

## 3. The 27-dimension rubric: the benchmark's explicit aesthetic

`prompts/judge_claude.md` is the fullest statement of what this tool thinks good
roleplay is. It is derived from the HawThorne V.2 preset's Director system
(see `analysis/scoring_rubric_v2.md`), i.e. it encodes one RP community's craft
standards, not a neutral one.

### Tier 1 — Fundamentals (40% of the score)

Contract compliance. These are close to objective:

- **Agency respect** — never write the user's character's actions, dialogue, or interiority
- **Instruction adherence** — follow the card, system prompt, POV, tense
- **Continuity** — track names, injuries, promises, locations
- **Length calibration** — length proportional to the scene's weight, not maximal
- **Distinct voices** — NPCs identifiable by speech pattern alone
- **Scene grounding** — spatially coherent, persistent physical detail

### Tier 2 — Quality Control (35%)

This tier is where the taste lives. Every dimension is stated as an *anti-pattern*,
and the preferred pole is consistently **understatement over intensity**:

| Dimension | Scores low | Scores high |
|---|---|---|
| Anti-purple prose | ornamental adjectives, weather mirroring mood | restraint as style |
| Anti-repetition | "eyes darkened" for the fifth time | fresh description each scene |
| Anti-sycophancy | world bends toward the protagonist | NPCs with their own agendas, real friction |
| Anti-artificial-perfection | gruff mercenary delivering therapy-grade insight | characters who misread rooms and arrive too late |
| Show don't tell | "he felt sad" | a flinch, a tightened grip |
| Subtext | full transparency | two conversations at once |
| Pacing and restraint | racing every beat | silence as an instrument |
| Imperfect coping | stoic composure, dramatic unmasking | bad jokes, strained charm, exhaustion |

Read as a whole, the rubric's ideal response is **literary-realist**: emotionally
oblique, physically specific, unhurried, and slightly broken. It structurally
disfavors high-melodrama, wish-fulfilment, and comfort RP — modes many real RP
users actively want. This is the single most important thing to understand about
the benchmark's bias.

### Tier 3 — Genre Craft (25%)

Thirteen genre dimensions, of which the judge scores **only those it deems
applicable** to the scene: earned intimacy, atmospheric dread, structural comedy,
excavated truth, spatial precision, lived-in worlds, information architecture,
structural inevitability, threshold logic, emotional residue, erotic craft,
context integration, temporal reasoning.

Two are load-bearing for RP specifically rather than for writing in general:

- **Context integration** — lorebook/world-info woven through behavior and voice rather than dumped as exposition
- **Temporal reasoning** — clocks, fatigue, healing, light, drinks going cold

**Erotic craft** is the most explicitly opinionated dimension in the file: it
requires specific bodies ("which hand, where, how much pressure"), in-character
behavior during sex, **direct anatomical language and no euphemisms**, and prose
rhythm matching the act. That's a house style, and it is scored as correctness.

### The calibration regime

The judge is instructed to be "extremely demanding," to target an average of
**2.8–3.2 across dimensions**, and to treat 3 as "competent, no complaints." A 4
requires a citable line; a 5 is capped at 1–2 per evaluation. Combined with the
"cite it" rule, this is a deliberate anti-generosity correction — the benchmark
believes the default LLM-judge failure mode is scoring everything 4.0–4.5.

Scoring: `overall = 0.40·tier1 + 0.35·tier2 + 0.25·tier3`
(`harness/aggregate.py:165`), falling back to `0.55/0.45` when no tier-3
dimension applies.

---

## 4. The flaw hunter: good RP as absence of defects

`prompts/judge_flaw_hunter.md` is the philosophy layer made operational. Start at
100, deduct per **quoted** defect — no quote, no deduction:

- **−15 fatal**: agency violation, character break, wrong language/POV/tense
- **−8 major**: purple prose, recycled description, narrating emotions, convenient world, flat NPC voice, missing spatial awareness, skipped time logic
- **−3 minor**: generic physical detail, excessive modifiers, missed subtext opportunity, predictable beat, over- or under-written moment, samey sentence rhythm
- **+5 bonus** (max +15): a physical detail that could only belong to this character, real subtext with both layers legible, earned surprise

The deduction *ratios* are the value judgment: hijacking the user's character is
worth exactly five generic body-language phrases, and one purple-prose passage
costs nearly as much as three predictable beats.

Its calibration is severe by design — the judge is told a 2000-char response
should yield 5–10 flaws, and that competent AI RP lands at 55–70. Across 270
multi-turn sessions the observed mean was **36.1** with a floor of **−177**.

---

## 5. The rule-based layer: a curated anti-style

No LLM involved. `harness/slop_detectors.py` names the specific prose tics the
benchmark treats as slop:

- **Throat-clearing openers** — "the words landed", "silence fell", "the question hung in the air"
- **Negation-assertion** — "It wasn't X. It was Y." / "not a whisper but a demand"
- **Filter words** — "noticed that", "felt like", "realized how" (density-scored)
- plus fragmentary choppiness, micro-corrections, voice statements, blush/animation, self-negation loops, rhetorical questions, snappy triads

`harness/objective_metrics.py` adds length-normalized deductions for cliché
density (~120 curated phrases), low type-token ratio, monotonous sentence rhythm,
and bigram/trigram recycling.

These are the *most* opinionated signals in the repo — a literal blocklist of
sentence shapes — and, notably, the ones that agree with real users **best**
(objective 42.3%, slop 30.6%, vs. flaw hunter 38.7% with a higher disagreement
rate), especially in NSFW content where cliché density predicts rejection at ~60%.

---

## 6. The human layer: two different "goods"

The arenas reveal that "good roleplay" splits into two genuinely different latents:

- **Single-message arena** (2,013 votes): snap-judgment **engagement**. Uncorrelated with every LLM-judge method (ρ between −0.31 and −0.07).
- **Multi-turn arena** (223 votes, full dialogues): **sustained quality**. Correlates with the LLM-judge multi-turn Likert at ρ = +0.495 (p = 0.027, n = 20).

So the disagreement is not humans-vs-judges in general — it is
*humans-judging-snippets vs. anyone-judging-arcs*. This is why the composite
leaderboard reports Engagement as a **separate axis** rather than folding it in.

Caveats the repo states plainly: 338 self-selected RP-community voters, small
per-pair counts, and a single judge (Sonnet 4) whose aesthetic shapes every
Likert-derived ranking.

---

## 7. How tunable is it?

Very tunable in the scoring layers; much less tunable in the human data and the
already-published numbers. In rough order of effort:

### Tier A — No code changes (CLI flags)

| Knob | Effect on the definition |
|---|---|
| `--judge-mode standard\|flaw_hunter\|comparative` | Swap between "score the craft", "count the defects", and "just pick a winner" |
| `--judges claude_sonnet gpt_4_1 deepseek_r1` | Swap whose taste is authoritative. `deepseek_r1` is the permissive NSFW-capable judge added in round 3 |
| `--view dimensions` | Rank by any single dimension instead of the composite — the built-in answer to "I only care about agency" |
| `--adversarial` / `--seeds ...` | Change *which failures* the run stresses |
| `--language en\|ru` | Judge under Russian literary conventions instead of English ones (the rubric explicitly re-anchors "purple prose" for Russian) |
| `--types completion\|preference\|consistency\|ooc_correction\|degradation` | Change what competence is being probed |

This is the intended path. `EXPERIMENT_DESIGN.md` §6 argues *against* one scalar
and recommends per-dimension rankings — "for users who care about strict lore
adherence, rank by lore contradiction rate."

### Tier B — One-line constant edits

| What | Where | Currently |
|---|---|---|
| Tier weights | `harness/aggregate.py:165` | `0.40 / 0.35 / 0.25` |
| Combined score mix | `analyze_combined.py:36` | `0.5·flaw + 0.25·objective + 0.25·slop` |
| Composite leaderboard | `analyze_composite_score.py:37` | `0.35 mt_arena, 0.25 llm_judge, 0.20 rubric, 0.15 flaw, 0.05 behavioral` |
| Slop severity | `harness/slop_detectors.py` per-detector `weight` | `slop = 100 − min(40, Σweight × 3)` |
| Cliché / TTR / rhythm thresholds | `harness/objective_metrics.py:307-371` | caps of 30 / 20 / 10 / 20 |
| Sampling | `harness/config.py` `GENERATION_CONFIG` | `temp 0.8, top_p 0.95, 4096 max` |

Rebalancing Tier 1 up to 0.7 turns this into a pure reliability benchmark;
rebalancing Tier 3 up turns it into a prose-craft benchmark. Nothing structural
resists either change.

### Tier C — Prompt edits (the real lever)

The rubric is prose in `prompts/*.md`, not code. Everything in sections 3–4 above
is a paragraph you can rewrite:

- Delete or add dimensions (tier 3 is already scored à la carte)
- Change the deduction magnitudes, or drop `purple_prose` entirely — §8 of the methodology notes it dominates aggregates because the category is so broad
- Move the calibration target off 2.8–3.2, or remove the "cite it" rule
- Rewrite Erotic Craft if "no euphemisms" isn't your house style
- Invert an anti-pattern: strike "anti-artificial perfection" if you want competent-hero power fantasy scored well

Cost of doing so: you lose comparability with every published number, since all
results were produced under the current prompts.

### Tier D — Fit the rubric to *your* preferences

The repo ships machinery for this, and it is the most honest answer to
"is the subjectivity tunable":

- `learn_rubric_from_data.py` — extract a wide feature set from accepted/rejected swipe pairs and find which consistently differ
- `learn_rubric_classifier.py` — train logistic-regression / random-forest / gradient-boosting models on feature *deltas* to predict which response a user keeps
- `analyze_engagement_regressor.py` — the shipped instance of this: 12 rule-based features trained on 1,717 clean arena votes, ρ = +0.555 with human single-message ELO (p = 0.077), used to impute Engagement for 10 models

Feed your own swipe logs in and you get a preference model fit to you rather than
to Sonnet 4. The documented finding is sobering, though: **individual features
don't predict preference**, which is why the classifier script exists at all.

### Tier E — Not tunable without new data collection

- **Human arena ELO** — 2,013 + 223 votes from a specific self-selected RP demographic. You can't reweight your way to a different voter pool.
- **Published leaderboards** — every number was produced under one judge, one set of prompts, `n=1` per (model, seed) cell, no stochastic-variance capture.
- **Tier-3 applicability** — the judge decides which genre dimensions apply per scene. That routing is not exposed as a setting.
- **Position bias** — 64% single-pass flip rate; bidirectional judging is mandatory, not optional.
- **What's out of scope entirely** — creativity (risk-taking raises failure rates by construction), emotional resonance, character *deepening* over time, long-context/compression failures beyond 12 turns, and any model's ceiling under a real SillyTavern preset (everything is tested raw).

---

## 8. Practical recipes

**"I want the strict-DM leaderboard."**
`--judge-mode flaw_hunter --adversarial`, then read `--view dimensions` for
agency, continuity and temporal reasoning. Optionally raise tier-1 weight in
`harness/aggregate.py:165`.

**"I want engaging prose, not correctness."**
Ignore the composite entirely; read the Engagement axis and the single-message
arena ELO. They are deliberately not blended, because they don't correlate.

**"I like melodrama and comfort RP."**
The Tier 2 rubric is working against you by design. Edit
`prompts/judge_claude.md` — drop anti-artificial-perfection and anti-sycophancy,
soften anti-purple-prose — and treat the rule-based scores (which track user
preference better than the judge does) as your primary signal.

**"I want it to match *me*."**
Collect swipe pairs, run `learn_rubric_classifier.py`, and use the fitted model.
Expect a modest signal; that's the honest state of the art here.

---

## 9. One-paragraph summary

RP-Bench's official position is that good roleplay is **roleplay that doesn't
break** — agency intact, instructions followed, continuity held, time coherent —
because that's the only part people reliably agree on. Layered on top is an
unofficial and much stronger aesthetic: restrained, oblique, physically specific,
character-imperfect literary realism with a world that pushes back. That second
layer is one community's taste, it is inherited from a specific SillyTavern
preset, it disagrees with real arena voters about half the time, and the repo
says so out loud. Nearly all of it is tunable — CLI flags for the cheap version,
constants for the weights, and plain-English prompt edits for the taste itself —
but the human vote data underneath, and the comparability of the published
rankings, are not.
