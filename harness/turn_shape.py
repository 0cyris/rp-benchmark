"""Turn-shape detectors — conversational floor discipline.

Measures how many *response obligations* one AI turn hands back to the player:
questions and direct commands aimed at the player's character. A turn that ends
on one open beat is easy to answer in character; a turn that stacks several
demands forces the player to reply to things that, in a real exchange, would
have been interleaved.

Only player-directed speech counts. NPC-to-NPC crosstalk is free — a scene where
one NPC addresses the player, a second interjects at the first, and the first
rebukes them is ONE obligation, not three. That keeps the metric off NPC banter,
which is usually good writing, and on the thing that actually breaks turn-taking.

All detection is rule-based — no LLM judge, no network, deterministic.

Companion analysis script: analyze_turn_shape.py
"""
import re
import unicodedata


# ============================================================
# TUNABLES
# ============================================================
# One obligation is the target shape. Zero is fine too (a beat that moves and
# leaves the floor open). The penalty applies only above one.
PENALTY_PER_OBLIGATION = 20
MAX_PENALTY = 60

# A turn is "bundled" once it hands back this many player-directed obligations.
BUNDLED_THRESHOLD = 2

# Window (chars) searched around a quoted span for an attribution frame
# ("she said to you", "he turned to Ash").
ATTRIBUTION_WINDOW_BEFORE = 120
ATTRIBUTION_WINDOW_AFTER = 60

# Two adjacent quoted spans belong to the same block (one split utterance,
# `"Hi," he said, "there."`) when the gap is short and carries a dialogue tag.
SPLIT_UTTERANCE_MAX_GAP = 40


# ============================================================
# LEXICONS
# ============================================================
SECOND_PERSON = re.compile(r"\b(you|your|yours|yourself)\b", re.IGNORECASE)

DIALOGUE_TAGS = re.compile(
    r"\b(said|asked|replied|answered|murmured|muttered|whispered|called|"
    r"added|continued|offered|snapped|breathed|managed|repeated|echoed|"
    r"began|finished|noted|observed|countered|pressed|urged)\b",
    re.IGNORECASE,
)

# Verbs that open an imperative aimed at the listener.
IMPERATIVE_VERBS = {
    "tell", "show", "give", "come", "follow", "choose", "pick", "answer",
    "speak", "explain", "describe", "look", "wait", "stop", "help", "bring",
    "take", "leave", "sit", "stand", "open", "close", "listen", "watch",
    "hold", "move", "go", "get", "put", "start", "finish", "try", "think",
    "decide", "walk", "run", "stay", "drop", "pass", "hand", "read", "write",
}

# Address forms that are always aimed at some other NPC, never at the player
# (unless the player carries the same title, which rule 1 catches first).
COURT_ADDRESS = re.compile(
    r"\b(your\s+(majesty|grace|highness|eminence|worship|lordship|ladyship))\b",
    re.IGNORECASE,
)

# Titles worth splitting off a harvested NPC name ("Queen Isara" -> "Queen").
NAME_TITLES = {
    "queen", "king", "prince", "princess", "lord", "lady", "ser", "sir",
    "commander", "captain", "chief", "maester", "master", "priestess",
    "priest", "director", "doctor", "dr", "professor", "sergeant", "general",
    "admiral", "engineer", "officer", "ambassador", "herald", "elder",
}

# Words a capitalized-token sweep must never mistake for a character name.
NAME_STOPWORDS = {
    "you", "your", "the", "a", "an", "and", "but", "or", "if", "when", "this",
    "that", "there", "they", "them", "their", "he", "she", "it", "i", "we",
    "write", "each", "five", "four", "three", "two", "one", "npcs", "npc",
    "user", "ooc", "pov", "yes", "no", "ok", "okay", "god", "gods",
}


# ============================================================
# NORMALIZATION
# ============================================================
def normalize(text: str, character_name: str | None = None) -> str:
    """Canonicalize a stored response before any parsing.

    Three corpus quirks make raw text unsafe to parse:

    1. Curly vs straight quotes split cleanly by model (gpt_4_1 is majority
       curly, ten models are 0% curly), so un-normalized quote counting would
       measure model identity rather than shape.
    2. Leading `CharacterName:` prefixes survive in the stored sessions —
       `harness.multiturn._strip_name_prefix` was added after they were
       generated. grok_4_1 emits one on ~36% of turns.
    3. Some responses are None, blank, or start with stray whitespace.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = (text.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'")
                .replace("«", '"').replace("»", '"'))
    if character_name:
        # Same shape as multiturn._strip_name_prefix. re.escape is mandatory:
        # names include "Gabi (Narrator)" and "Narrator (Heist Scene)".
        text = re.sub(
            r"^(?:\s*%s\s*:\s*)+" % re.escape(character_name),
            "", text, flags=re.IGNORECASE,
        )
    return text.strip()


# ============================================================
# SCENE ROSTER
# ============================================================
def _capitalized_names(text: str) -> set[str]:
    """Capitalized sequences that look like names.

    Sentence-initial capitals are skipped. Character settings are ordinary
    prose — "Cynical, observant, burned out." / "Notices things other people
    miss." — and harvesting those openers would seed the NPC list with words
    like "has" and "notices", which then match inside ordinary dialogue and
    misclassify player-directed lines as crosstalk.
    """
    out = set()
    # Separator is spaces only — \s+ would splice across newlines and turn
    # big-card section headers into names ("Appearance\nMaren").
    for m in re.finditer(r"\b([A-Z][a-z'’]+(?:[ ]+[A-Z][a-z'’]+){0,2})\b", text):
        cand = m.group(1).strip()
        if cand.lower() in NAME_STOPWORDS:
            continue
        if all(w.lower() in NAME_STOPWORDS for w in cand.split()):
            continue
        preceding = text[:m.start()].rstrip()
        if not preceding or preceding[-1] in ".!?:;\n-*•":
            continue  # sentence- or line-initial: prose, not a name
        # Require a recognized title as the head word. Free prose in a big
        # character card is full of capitalized non-people — section headers
        # ("Core Personality", "Communication Rules"), places ("Medbay"),
        # adjectives ("Newtonian", "Dwarvish") — and any of those left in the
        # NPC list would match inside ordinary dialogue and silently reclassify
        # player-directed lines as crosstalk. Title-led is the reliable signal.
        if cand.split()[0].lower() not in NAME_TITLES:
            continue
        out.add(cand)
    return out


def _looks_like_name(text: str) -> bool:
    """Is a bullet-list capture a character name rather than a sentence?

    Big-card seeds bullet whole rules ("- Maren has a severe allergy to
    shellfish."), which must not become address terms.
    """
    words = text.split()
    if not words or len(words) > 3 or text.endswith("."):
        return False
    return all(w[:1].isupper() for w in words if w)


def build_roster(seed: dict) -> dict:
    """Derive player and NPC address terms from a seed definition.

    Returns {"player": [...], "npc": [...]} of lowercase terms. Player terms
    win ties: rule 1 of the addressee ladder fires before the NPC rule.
    """
    character_name = (seed.get("character_name") or "").strip()
    user_name = (seed.get("user_name") or "").strip()
    char_setting = seed.get("character_setting") or ""
    user_setting = seed.get("user_setting") or ""

    player = {user_name.lower()} if user_name else set()
    # A role-noun user_name ("Junior Officer") is also addressed by its head
    # word ("Officer"), which is how NPCs actually call the player.
    for word in user_name.split():
        if word.lower() in NAME_TITLES:
            player.add(word.lower())
    for cand in _capitalized_names(user_setting):
        if user_name and cand.lower() in user_name.lower():
            player.add(cand.lower())

    npc = set()
    # Bullet-listed casts, e.g. "- Queen Isara (cold, calculating)".
    for m in re.finditer(r"^\s*[-*]\s*([A-Z][^(\n,]{1,40})", char_setting, re.M):
        cand = m.group(1).strip()
        if _looks_like_name(cand):
            npc.add(cand.lower())
    for cand in _capitalized_names(char_setting):
        npc.add(cand.lower())

    # Split titles off names: "Queen Isara" also answers to "Isara" and "Queen".
    # Only for title-led names — splitting an arbitrary phrase would readmit
    # the generic head words the harvest just excluded.
    for name in list(npc):
        parts = name.split()
        if len(parts) > 1 and parts[0] in NAME_TITLES:
            npc.add(parts[0])
            npc.add(" ".join(parts[1:]))

    # A speaker never vocatives themselves, and the player is never an NPC.
    npc.discard(character_name.lower())
    for part in character_name.split():
        npc.discard(part.lower())
    npc -= player
    # Drop any NPC term that is a word of a player term. Seed 20's player is
    # "Dr. Kofi Osei", so a bare "Kofi" left in the NPC set would send every
    # line addressing the player into the crosstalk bucket.
    player_words = {w.strip(".,'") for term in player for w in term.split()}
    npc = {t for t in npc
           if len(t) > 2 and not set(t.split()) & player_words}

    return {"player": sorted(player), "npc": sorted(npc)}


def _mentions(text: str, terms) -> bool:
    return any(
        re.search(r"\b%s\b" % re.escape(t), text, re.IGNORECASE) for t in terms
    )


# ============================================================
# QUOTED SPANS
# ============================================================
def extract_spans(text: str) -> list[dict]:
    """Quoted dialogue spans with their offsets. Text must be normalized.

    Only straight double quotes delimit dialogue — apostrophes are far too
    common inside prose to treat as delimiters.
    """
    spans = []
    for m in re.finditer(r'"([^"]{1,2000})"', text):
        inner = m.group(1).strip()
        if inner:
            spans.append({"text": inner, "start": m.start(), "end": m.end()})
    return spans


def count_blocks(text: str, spans: list[dict]) -> int:
    """Quoted spans separated by narration. Diagnostic only — never scored.

    Adjacent spans merge when the gap is short and carries a dialogue tag,
    which is one utterance split around its attribution.
    """
    if not spans:
        return 0
    blocks = 1
    for prev, cur in zip(spans, spans[1:]):
        gap = text[prev["end"]:cur["start"]]
        split_utterance = (
            len(gap) <= SPLIT_UTTERANCE_MAX_GAP and DIALOGUE_TAGS.search(gap)
        )
        if not split_utterance:
            blocks += 1
    return blocks


# ============================================================
# ADDRESSEE CLASSIFICATION
# ============================================================
def classify_addressee(text: str, span: dict, roster: dict) -> str:
    """Return "player", "npc", or "ambiguous" for one quoted span.

    Heuristic ladder, first match wins. Ordering carries real weight: the NPC
    vocative rule must beat the bare second-person rule, or "Your Majesty, if
    I may—" would read as player-directed on its "Your".
    """
    inner = span["text"]

    # 1. Player vocative.
    if _mentions(inner, roster["player"]):
        return "player"

    # 2. NPC vocative, including court address forms.
    if COURT_ADDRESS.search(inner) or _mentions(inner, roster["npc"]):
        return "npc"

    # 3. Attribution frame in the surrounding narration.
    before = text[max(0, span["start"] - ATTRIBUTION_WINDOW_BEFORE):span["start"]]
    after = text[span["end"]:span["end"] + ATTRIBUTION_WINDOW_AFTER]
    frame = before + " " + after
    if re.search(r"\b(to|at|toward|towards|asked|told)\s+you\b", frame, re.I):
        return "player"
    for term in roster["npc"]:
        if re.search(r"\b(to|at|toward|towards)\s+(the\s+)?%s\b" % re.escape(term),
                     frame, re.IGNORECASE):
            return "npc"

    # 4. Second person with nothing competing.
    if SECOND_PERSON.search(inner):
        return "player"

    return "ambiguous"


# ============================================================
# OBLIGATIONS
# ============================================================
def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def count_questions(span_text: str) -> int:
    """Question marks are the obligation, not sentences — "Who? When? Why?"
    is three demands even though it reads as one breath."""
    return span_text.count("?")


def count_imperatives(span_text: str) -> int:
    """Sentences opening on a command verb aimed at the listener."""
    n = 0
    for sent in _sentences(span_text):
        if sent.rstrip().endswith("?"):
            continue  # already counted as a question
        # Drop a leading vocative: "Ambassador, tell me what you want."
        stripped = re.sub(r"^[A-Z][\w'’]*,\s+", "", sent)
        m = re.match(r"[\"'\*\-—\s]*([A-Za-z]+)", stripped)
        if m and m.group(1).lower() in IMPERATIVE_VERBS:
            n += 1
    return n


def analyze_turn(text: str, roster: dict, character_name: str | None = None) -> dict:
    """Score one AI-character turn.

    Emits both an inclusive and an exclusive reading of ambiguous spans so the
    caller can report a bracket instead of a false point estimate.
    """
    norm = normalize(text, character_name)
    if not norm:
        return {
            "empty": True, "chars": 0, "words": 0,
            "spans": 0, "dialogue_blocks": 0,
            "player_spans": 0, "npc_spans": 0, "ambiguous_spans": 0,
            "questions": 0, "imperatives": 0,
            "obligations_loose": 0, "obligations_strict": 0,
            "terminal_obligations": 0,
            "bundled_loose": False, "bundled_strict": False,
            "shape_score_loose": None, "shape_score_strict": None,
        }

    spans = extract_spans(norm)
    final_para_start = norm.rfind("\n\n")

    counts = {"player": 0, "npc": 0, "ambiguous": 0}
    q_player = i_player = 0
    q_amb = i_amb = 0
    terminal = 0

    for span in spans:
        who = classify_addressee(norm, span, roster)
        counts[who] += 1
        q = count_questions(span["text"])
        imp = count_imperatives(span["text"])
        if who == "player":
            q_player += q
            i_player += imp
            if span["start"] > final_para_start:
                terminal += q + imp
        elif who == "ambiguous":
            q_amb += q
            i_amb += imp

    loose = q_player + i_player                    # ambiguous discarded
    strict = loose + q_amb + i_amb                 # ambiguous counted

    def score(obligations):
        return 100 - min(
            MAX_PENALTY, PENALTY_PER_OBLIGATION * max(0, obligations - 1)
        )

    return {
        "empty": False,
        "chars": len(norm),
        "words": len(norm.split()),
        "spans": len(spans),
        "dialogue_blocks": count_blocks(norm, spans),
        "player_spans": counts["player"],
        "npc_spans": counts["npc"],
        "ambiguous_spans": counts["ambiguous"],
        "questions": q_player,
        "imperatives": i_player,
        "obligations_loose": loose,
        "obligations_strict": strict,
        "terminal_obligations": terminal,
        "bundled_loose": loose >= BUNDLED_THRESHOLD,
        "bundled_strict": strict >= BUNDLED_THRESHOLD,
        "shape_score_loose": score(loose),
        "shape_score_strict": score(strict),
    }
