"""Deterministic phase, strategy, and restatement-lens selection (no LLM)."""

from __future__ import annotations

import re
from typing import Optional

from core.schemas import (
    AnalyzerState,
    LENS_FORBIDDEN_BY_MECHANISM,
    LENS_PRIMARY_BY_MECHANISM,
    PHASES,
    STRATEGIES,
    SessionState,
    StrategyDecision,
)


# Lens keyword tables (case-insensitive substring matches)

LENS_KEYWORDS: dict[str, list[str]] = {
    "A": [  # physical / somatic
        "dard", "sir", "pet", "seena", "chest", "thakan", "weight",
        "neend", "nahi aati", "kamar", "sar", "aankh", "bukhar",
        "badan", "chakkar", "ulti", "bhookh",
    ],
    "B": [  # role / duty
        "zimmedari", "kaam", "job", "career", "padhai", "naukri",
        "responsibility", "paisa kamana", "ghar chalana", "emi", "loan",
        "duty", "role", "position",
    ],
    "C": [  # relational
        "ghar", "papa", "mummy", "maa", "bhai", "behan", "wife", "husband",
        "family", "log", "samaj", "rishtedaar", "dost", "sasural", "saas",
    ],
    "D": [  # time / duration
        "kab tak", "kitne din", "saalon se", "bahut time", "itne saal",
        "kab khatam", "kitna aur", "lamba", "arsaa",
    ],
    "E": [  # feared consequence
        "dar", "kya hoga", "future", "nahi hoga", "barbaad", "fail",
        "khatam", "doob jayenge", "bezzati", "insult", "reject",
        "nikaal denge",
    ],
    "F": [  # emotional overwhelm (also acts as the default fallback)
        "ro", "rona", "aansu", "toot", "bikhar", "pagal", "udaas",
    ],
}

# Lenses are checked A → E first; F is the catch-all default.
_LENS_PRIORITY: list[str] = ["A", "B", "C", "D", "E"]


# Help-seeking detector
# When the seeker EXPLICITLY asks for help/advice/solution, we override the
# Analyzer's receptiveness reading to "high". This unblocks the gate when the
# Analyzer (which is distress-biased due to dataset skew) keeps reporting low
# receptiveness even after the user has clearly opened up.
#
# Patterns are matched case-insensitively as substrings (with word boundaries
# where Hindi inflections allow it). False positives are tolerable here — the
# worst that happens is one over-eager phase advance, which the next turn's
# Analyzer reading will correct.
_HELP_SEEKING_PATTERNS: list[str] = [
    r"\bkya\s+kar(?:u|oo|oon|na|na\s+chahiye)\b",
    r"\bkaise\s+kar(?:u|oo|oon|na)\b",
    r"\bkya\s+karna\s+(?:chahiye|hai)\b",
    r"\bsolution(?:s)?\b",
    r"\bhelp\s+kar(?:o|u|do|na)\b",
    r"\bhelp\s+chahiye\b",
    r"\badvice\b",
    r"\bsuggest(?:ion)?(?:s)?\b",
    r"\bbatao(?:\s+na)?\b",
    r"\bbata\s+(?:do|na)\b",
    r"\bbata(?:do|na)\b",
    r"\btum\s+(?:kuch\s+)?bata\b",
    r"\bsamjhao\b",
    r"\braasta\s+(?:bata|nikalo|dikhao)\b",
    r"\bidea\s+do\b",
    r"\bkuch\s+karo\b",
    r"\bwhat\s+(?:should|do)\s+i\s+do\b",
    r"\bany\s+(?:advice|suggestion|tip|idea)\b",
    r"\bplease\s+help\b",
]
_COMPILED_HELP = [
    re.compile(p, re.IGNORECASE | re.UNICODE) for p in _HELP_SEEKING_PATTERNS
]


def detect_help_seeking(text: Optional[str]) -> bool:
    """Return True if `text` contains an explicit help/advice request.

    Used by `compute_phase` to upgrade `user_receptiveness` to "high" when the
    seeker is clearly opening the door for guidance. Pure regex, ~10 µs.
    """
    if not text:
        return False
    for pat in _COMPILED_HELP:
        if pat.search(text):
            return True
    return False


def _effective_receptiveness(
    analyzer_state: AnalyzerState,
    seeker_text: Optional[str],
) -> str:
    """Treat explicit help-seeking as receptiveness=high regardless of what
    the Analyzer said. Otherwise return the Analyzer's reading verbatim."""
    if seeker_text and detect_help_seeking(seeker_text):
        return "high"
    return analyzer_state.user_receptiveness


# Anti-stuck heuristic
# After 4 consecutive Exploration turns (without intensity escalating to 5+),
# we force-promote to Insight. The seeker has been heard enough; staying in
# Exploration any longer makes the bot feel hollow / circular (the symptom
# the user reported in the screenshot from 2026-05-02).
_STUCK_THRESHOLD = 4


def _is_stuck_in_exploration(session: SessionState, intensity: int) -> bool:
    """True if the last `_STUCK_THRESHOLD` phases were all Exploration AND we
    are not in genuine high-distress territory (intensity ≥ 5)."""
    if intensity >= 5:
        return False
    last = session.phase_history[-_STUCK_THRESHOLD:]
    return (
        len(last) >= _STUCK_THRESHOLD
        and all(p == "Exploration" for p in last)
    )


# After 3 consecutive Insight turns at moderate-or-better intensity, we
# graduate to Action. Without this rule the bot stays in "reflection of
# feelings" mode and the user's "kya karu?" is met with another mirror
# instead of a concrete suggestion. This is the symptom the user
# described directly: "the model don't know when actually to move to Action".
_INSIGHT_STUCK_THRESHOLD = 3


def _is_stuck_in_insight(session: SessionState, intensity: int) -> bool:
    """True if we've been in Insight long enough at moderate intensity that
    the seeker is ready for Action. Uses both `phase_history` (last N) AND
    the new `turns_in_current_phase` journey marker so the rule fires even
    if there were brief Exploration interludes inside an Insight stretch."""
    if intensity >= 4:
        # Still too distressed for Action — Insight is the right place.
        return False
    last_phase = session.phase_history[-1] if session.phase_history else None
    if last_phase != "Insight":
        return False
    # Path A: hard signal from journey marker.
    if session.turns_in_current_phase >= _INSIGHT_STUCK_THRESHOLD:
        return True
    # Path B: legacy phase_history check (still useful for sessions
    # restored from before the journey markers existed).
    last_n = session.phase_history[-_INSIGHT_STUCK_THRESHOLD:]
    return (
        len(last_n) >= _INSIGHT_STUCK_THRESHOLD
        and all(p == "Insight" for p in last_n)
    )


# FUNCTION 1: compute_phase

def compute_phase(
    analyzer_state: AnalyzerState,
    session: SessionState,
    seeker_text: Optional[str] = None,
) -> str:
    """
    Determine the current conversation phase.

    The Phase Gate is sacred — Action strategies NEVER appear during Exploration.

    `seeker_text` (optional) lets us detect explicit help-seeking like
    "kya karu yaar" / "solution batao", which upgrades receptiveness to "high"
    and lets the gate skip ahead. Backward-compatible: if omitted, behavior
    matches the original receptiveness-blind logic.

    Rules (evaluated in order, first match wins):
      0.  risk_signal active                              → "Exploration"
      1.  New problem detected                            → "Exploration"
      2.  Anti-stuck (Exploration): 4× Exploration AND
          intensity ≤ 4                                   → "Insight"  (force advance)
      2b. Anti-stuck (Insight): 3× Insight AND
          intensity ≤ 3                                   → "Action"   (force advance)
      3.  effective_receptiveness == "high":
          3a. Insight in last 2 phases AND intensity ≤ 3  → "Action"
          3b. intensity ≤ 4                               → "Insight"
      4.  Intensity >= 4                                  → "Exploration"
      5.  Intensity == 3 AND last phase was "Exploration" → "Insight"
      6.  Intensity == 3 AND last phase was NOT "Exploration" → "Exploration"
      7.  Intensity <= 2 AND "Insight" in last 2 phases   → "Action"
      8.  Intensity <= 2 AND "Insight" NOT in last 2 phases → "Insight"
      9.  Default                                         → "Exploration"
    """
    intensity = analyzer_state.emotion_intensity
    last_phase: Optional[str] = session.phase_history[-1] if session.phase_history else None
    last_two_phases: list[str] = session.phase_history[-2:] if session.phase_history else []
    receptiveness = _effective_receptiveness(analyzer_state, seeker_text)

    # Rule 0: crisis — never advance phase, hold space and let safety handle escalation.
    if analyzer_state.risk_signal:
        return "Exploration"

    # Rule 1: a new problem resets the conversation; we must re-validate before advising.
    if analyzer_state.is_new_problem:
        return "Exploration"

    # Rule 2 (NEW): anti-stuck — break out of an Exploration loop when the seeker
    # is no longer in acute distress. Without this, the bot circles for 5+ turns
    # if the Analyzer keeps reading intensity=4 (very common bias).
    if _is_stuck_in_exploration(session, intensity):
        return "Insight"

    # Rule 2b (NEW): anti-stuck for Insight — graduate to Action once we've
    # spent enough time reflecting at moderate intensity. Symmetric counterpart
    # to Rule 2; together they prevent both "endless mirroring" pathologies.
    if _is_stuck_in_insight(session, intensity):
        return "Action"

    # Rule 3 (NEW): high receptiveness wins — the seeker has clearly opened up
    # ("Solution kya h?", "Tum batao na", "Kya karu?"). Don't keep mirroring;
    # advance the phase. Skip directly to Action if Insight has already happened.
    if receptiveness == "high" and intensity <= 4:
        if "Insight" in last_two_phases and intensity <= 3:
            return "Action"
        return "Insight"

    # Rule 4: high distress — too flooded for advice.
    if intensity >= 4:
        return "Exploration"

    # Rule 5: moderate distress, already explored — natural progression to Insight.
    if intensity == 3 and last_phase == "Exploration":
        return "Insight"

    # Rule 6: moderate distress but no prior Exploration — must explore first.
    if intensity == 3 and last_phase != "Exploration":
        return "Exploration"

    # Rule 7: low distress AND we have already done Insight recently — ready for Action.
    if intensity <= 2 and "Insight" in last_two_phases:
        return "Action"

    # Rule 8: low distress but no recent Insight — must consolidate via Insight first.
    if intensity <= 2 and "Insight" not in last_two_phases:
        return "Insight"

    # Rule 9: ultra-defensive default.
    return "Exploration"


# Phase decision explainer (debug / observability)
def explain_phase_decision(
    analyzer_state: AnalyzerState,
    session: SessionState,
    seeker_text: Optional[str] = None,
) -> str:
    """Return a human-readable string describing WHY the phase came out the
    way it did. Pure inspection — recomputes against the same rule order as
    `compute_phase` and surfaces the first matching rule. Used by the dev UI
    so a developer can see why the bot is or isn't advancing."""
    intensity = analyzer_state.emotion_intensity
    receptiveness = _effective_receptiveness(analyzer_state, seeker_text)
    help_seeking = detect_help_seeking(seeker_text or "")
    last_phase = session.phase_history[-1] if session.phase_history else None
    last_two = session.phase_history[-2:] if session.phase_history else []

    if analyzer_state.risk_signal:
        return f"R0 risk_signal active → locked Exploration"
    if analyzer_state.is_new_problem:
        return "R1 new_problem → Exploration"
    if _is_stuck_in_exploration(session, intensity):
        return (
            f"R2 anti-stuck ({_STUCK_THRESHOLD}× Exploration, int={intensity}) "
            "→ forced Insight"
        )
    if _is_stuck_in_insight(session, intensity):
        return (
            f"R2b anti-stuck ({_INSIGHT_STUCK_THRESHOLD}× Insight, "
            f"int={intensity}) → forced Action"
        )
    if receptiveness == "high" and intensity <= 4:
        tag = " (help-seeking)" if help_seeking else ""
        if "Insight" in last_two and intensity <= 3:
            return f"R3a high recept{tag} + prior Insight + int≤3 → Action"
        return f"R3b high recept{tag} + int≤4 → Insight"
    if intensity >= 4:
        return f"R4 int={intensity}≥4, recept={receptiveness} → Exploration"
    if intensity == 3 and last_phase == "Exploration":
        return "R5 int=3 + last=Exploration → Insight"
    if intensity == 3:
        return "R6 int=3 + last≠Exploration → Exploration"
    if intensity <= 2 and "Insight" in last_two:
        return f"R7 int≤2 + recent Insight → Action"
    if intensity <= 2:
        return "R8 int≤2 + no recent Insight → Insight"
    return "R9 default → Exploration"


# FUNCTION 2: compute_allowed_strategies

def compute_allowed_strategies(phase: str, session: SessionState) -> list[str]:
    """
    Return the strategies allowed in the current phase.

    Exploration → ["RESTATEMENT_OR_PARAPHRASING", "QUESTION"]
    Insight     → ["REFLECTION_OF_FEELINGS", "AFFIRMATION_AND_REASSURANCE", "SELF_DISCLOSURE"]
                  (SELF_DISCLOSURE removed if already used in this session)
    Action      → ["PROVIDING_SUGGESTIONS", "EXECUTION", "INFORMATION"]
    """
    if phase not in PHASES:
        raise ValueError(f"Unknown phase: {phase!r}. Must be one of {PHASES}.")

    if phase == "Exploration":
        return ["RESTATEMENT_OR_PARAPHRASING", "QUESTION"]

    if phase == "Insight":
        allowed = [
            "REFLECTION_OF_FEELINGS",
            "AFFIRMATION_AND_REASSURANCE",
            "SELF_DISCLOSURE",
        ]
        # Self-disclosure is a one-shot intimacy device; never repeat in the same session.
        if session.self_disclosure_used:
            allowed = [s for s in allowed if s != "SELF_DISCLOSURE"]
        return allowed

    # phase == "Action"
    return ["PROVIDING_SUGGESTIONS", "EXECUTION", "INFORMATION"]


# FUNCTION 3: compute_strategy

def _next_in_allowed(strategy: str, allowed: list[str]) -> str:
    """Cyclically pick the next allowed strategy after `strategy`. Falls back to allowed[0]."""
    if not allowed:
        # Defensive — should never happen because every phase has allowed strategies.
        return STRATEGIES[0]
    if strategy in allowed:
        idx = allowed.index(strategy)
        return allowed[(idx + 1) % len(allowed)]
    return allowed[0]


# Supporter-text repetition guard
# If the last two SAATHI replies share too much vocabulary, the strategy
# selector will rotate to a different allowed strategy. This catches the
# "Yeh dar waqai bahut bada hai..." → "Yeh dar sach me itna ovewhelming hai..."
# loop the user complained about.
_REPETITION_JACCARD_THRESHOLD = 0.55
_REPETITION_MIN_TOKENS = 5  # ignore very short replies
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _supporter_repetition_detected(session: SessionState) -> bool:
    """True if the last two Supporter turns are >55% lexically similar."""
    supporter_texts: list[str] = [
        t.text for t in session.turn_history if t.speaker == "Supporter"
    ]
    if len(supporter_texts) < 2:
        return False
    a = set(_TOKEN_RE.findall(supporter_texts[-1].lower()))
    b = set(_TOKEN_RE.findall(supporter_texts[-2].lower()))
    if len(a) < _REPETITION_MIN_TOKENS or len(b) < _REPETITION_MIN_TOKENS:
        return False
    union = a | b
    if not union:
        return False
    return (len(a & b) / len(union)) >= _REPETITION_JACCARD_THRESHOLD


def compute_strategy(
    phase: str,
    analyzer_state: AnalyzerState,
    session: SessionState,
) -> str:
    """
    Select the specific strategy. Deterministic — no LLM.

    Rule order (first match wins):

    S5 ACTIVE PROTOCOLS (per dataset/prompts/conversation_generation.txt §S5).
    These are HARD overrides — they fire regardless of phase.
      P1. risk_signal set (moderate/high)              → RESTATEMENT_OR_PARAPHRASING
          (Hold phase, mirror only; safety follow-up handled by safety agent.)
      P2. stigma_cue == True                           → RESTATEMENT_OR_PARAPHRASING
          ("next Supporter turn = Restatement only,
           shorter than previous turn, ends with soft pause not a question.")

    Anti-repetition rules (apply after S5 protocols):
      A1. Same strategy in last 2 entries of strategy_history → next in allowed list.
      A2. "QUESTION" appears twice consecutively              → RESTATEMENT_OR_PARAPHRASING.

    Phase-specific rules:
      EXPLORATION:
        E1. intensity >= 5 AND receptiveness == "low"  → RESTATEMENT_OR_PARAPHRASING
        E2. intensity == 4 AND receptiveness == "medium" → alternate RESTATEMENT/QUESTION
        E3. intensity == 4 AND receptiveness == "low"  → RESTATEMENT_OR_PARAPHRASING
        E4. Default Exploration                        → RESTATEMENT_OR_PARAPHRASING

      INSIGHT:
        I1. coping_mech == "Somatization"              → REFLECTION_OF_FEELINGS
            (Physical symptoms — reflect the emotional root.)
        I2. Default Insight                            → REFLECTION_OF_FEELINGS

      ACTION:
        Ac1. Default Action                            → PROVIDING_SUGGESTIONS

      Final fallback                                   → allowed_strategies[0]
    """
    allowed = compute_allowed_strategies(phase, session)
    history = session.strategy_history
    last_two = history[-2:] if len(history) >= 2 else []
    last_strategy: Optional[str] = history[-1] if history else None

    # ---- S5 ACTIVE PROTOCOLS (highest priority, override every other rule) ----
    # P1: risk_signal present (moderate/high) — Restatement-only, hold phase.
    if analyzer_state.risk_signal:
        return "RESTATEMENT_OR_PARAPHRASING"

    # P2: stigma_cue True — Restatement-only, soft pause (no question).
    if analyzer_state.stigma_cue is True:
        return "RESTATEMENT_OR_PARAPHRASING"

    # ---- Supporter-text repetition guard ----
    # If SAATHI's last two replies were too similar lexically, force a
    # different strategy in the current phase's allowed list. This breaks
    # the "every reply sounds like a paraphrase of the last" loop.
    if len(allowed) > 1 and _supporter_repetition_detected(session):
        for s in allowed:
            if s != last_strategy:
                return s

    # ---- Anti-repetition (override layer) ----
    if len(last_two) == 2 and last_two[0] == last_two[1]:
        # A2 is more specific (QUESTION → RESTATEMENT). Apply it first to avoid ambiguity.
        if last_two[0] == "QUESTION":
            if "RESTATEMENT_OR_PARAPHRASING" in allowed:
                return "RESTATEMENT_OR_PARAPHRASING"
            # Fallback if RESTATEMENT not allowed in current phase.
            return _next_in_allowed(last_two[0], allowed)
        # A1 generic: pick the next strategy after the repeated one.
        return _next_in_allowed(last_two[0], allowed)

    # ---- Phase-specific rules ----
    intensity = analyzer_state.emotion_intensity
    receptiveness = analyzer_state.user_receptiveness

    if phase == "Exploration":
        # E1: severely flooded — mirror, do not interrogate.
        if intensity >= 5 and receptiveness == "low":
            return "RESTATEMENT_OR_PARAPHRASING"

        # E2: high but workable distress with medium openness — alternate RESTATEMENT/QUESTION.
        if intensity == 4 and receptiveness == "medium":
            if last_strategy == "RESTATEMENT_OR_PARAPHRASING":
                return "QUESTION"
            return "RESTATEMENT_OR_PARAPHRASING"

        # E3: high distress + closed-off — stay safe with mirroring.
        if intensity == 4 and receptiveness == "low":
            return "RESTATEMENT_OR_PARAPHRASING"

        # E4: default Exploration.
        return "RESTATEMENT_OR_PARAPHRASING"

    if phase == "Insight":
        # I0 (NEW — B4): SELF_DISCLOSURE bias for rapport-built sessions.
        # If we've already done at least one Insight turn (REFLECTION) and the
        # seeker is engaging well (intensity ≤ 3, not new problem), bring out
        # SELF_DISCLOSURE for warmth. Currently the strategy was ~never used.
        # Constraints: only fire once per session (one-shot intimacy device),
        # not when stigma_cue is set (the seeker is closed off — bad time to
        # share), and only after we've genuinely listened first.
        if (
            "SELF_DISCLOSURE" in allowed
            and not session.self_disclosure_used
            and not analyzer_state.stigma_cue
            and not analyzer_state.is_new_problem
            and analyzer_state.emotion_intensity <= 3
            and session.turn_count >= 3
            and "REFLECTION_OF_FEELINGS" in session.strategy_history
        ):
            return "SELF_DISCLOSURE"

        # I1: physical symptoms — reflect the emotional root underneath.
        if analyzer_state.current_coping_mech == "Somatization":
            if "REFLECTION_OF_FEELINGS" in allowed:
                return "REFLECTION_OF_FEELINGS"

        # I2: default Insight.
        if "REFLECTION_OF_FEELINGS" in allowed:
            return "REFLECTION_OF_FEELINGS"

        # All Insight strategies exhausted — fall through to final fallback.

    if phase == "Action":
        # Ac0 (NEW — B3): post-suggestion follow-up rotation.
        # If we just gave PROVIDING_SUGGESTIONS last turn AND we're still in
        # Action, rotate to EXECUTION (= "did you try it? how was it?"). This
        # closes the loop the way a real friend would — instead of giving
        # another generic tip, we check in on the previous one.
        if (
            last_strategy == "PROVIDING_SUGGESTIONS"
            and "EXECUTION" in allowed
        ):
            return "EXECUTION"

        # Ac1: default Action.
        if "PROVIDING_SUGGESTIONS" in allowed:
            return "PROVIDING_SUGGESTIONS"

    # ---- Final fallback ----
    return allowed[0]


# FUNCTION 4: compute_lens

def _contains_keyword(haystack: str, keyword: str) -> bool:
    """Case-insensitive substring match. Multi-word keywords are matched as substrings."""
    if not haystack:
        return False
    return keyword.lower() in haystack.lower()


def compute_lens(
    analyzer_state: AnalyzerState,
    seeker_text: Optional[str] = None,
) -> Optional[str]:
    """
    Decide which restatement lens (A-F) the Generator should use.

    Pure keyword matching — no LLM. Searches `coping_shade_signal` and (optionally)
    the broader `seeker_text`. First keyword match in priority order A → E wins;
    otherwise F.

    Mechanism compatibility (per dataset/prompts/conversation_generation.txt §S4):
      - Lens A is FORBIDDEN for Sequential coping.
      - Lens F is FORBIDDEN for Somatization coping.
    Forbidden lenses are skipped in the priority scan, and any final fallback
    re-routes to the mechanism's preferred lens (or the first allowed lens).

    Note: callers must only use the returned lens when the selected strategy is
    RESTATEMENT_OR_PARAPHRASING. compute_full_strategy enforces that.
    """
    haystack_parts: list[str] = []
    if analyzer_state.coping_shade_signal:
        haystack_parts.append(analyzer_state.coping_shade_signal)
    if seeker_text:
        haystack_parts.append(seeker_text)
    haystack = " \n ".join(haystack_parts).lower()

    mechanism = analyzer_state.current_coping_mech
    forbidden = LENS_FORBIDDEN_BY_MECHANISM.get(mechanism, set())

    def _allowed(letter: str) -> bool:
        return letter not in forbidden

    # Priority scan A → E, skipping forbidden lenses.
    if haystack.strip():
        for lens_letter in _LENS_PRIORITY:
            if not _allowed(lens_letter):
                continue
            for kw in LENS_KEYWORDS[lens_letter]:
                if _contains_keyword(haystack, kw):
                    return lens_letter

        # Lens F is the explicit overwhelm scan (only if allowed for the mechanism).
        if _allowed("F"):
            for kw in LENS_KEYWORDS["F"]:
                if _contains_keyword(haystack, kw):
                    return "F"

    # ---- Fallback when no keyword matched (or haystack empty) ----
    # 1) Use mechanism's primary lens if it isn't forbidden (it never should be).
    primary = LENS_PRIMARY_BY_MECHANISM.get(mechanism)
    if primary and _allowed(primary):
        return primary

    # 2) First allowed lens in canonical order A..F.
    for letter in ("A", "B", "C", "D", "E", "F"):
        if _allowed(letter):
            return letter

    # Should never happen — every mechanism has at least 5 allowed lenses.
    return "F"


# FUNCTION 5: compute_full_strategy

def compute_full_strategy(
    analyzer_state: AnalyzerState,
    session: SessionState,
    seeker_text: Optional[str] = None,
) -> StrategyDecision:
    """
    End-to-end deterministic decision: phase → strategy → (optional) lens.

    `seeker_text` is forwarded to `compute_phase` (to detect explicit
    help-seeking) and to `compute_lens` (broader keyword surface for lens
    selection). Optional for backward compatibility.
    """
    phase = compute_phase(analyzer_state, session, seeker_text)
    strategy = compute_strategy(phase, analyzer_state, session)
    lens: Optional[str] = None
    if strategy == "RESTATEMENT_OR_PARAPHRASING":
        lens = compute_lens(analyzer_state, seeker_text)

    return StrategyDecision(
        current_phase=phase,
        selected_strategy=strategy,
        restatement_lens=lens,
    )


# Self-tests

if __name__ == "__main__":
    from core.schemas import AnalyzerState, SessionState

    # Test 1: High intensity, low receptiveness, Duty_Based → Exploration +
    #         RESTATEMENT with Lens B (Duty_Based primary).
    state1 = AnalyzerState(
        emotion_type="exhaustion", emotion_intensity=5,
        problem_type="Employment_Livelihood",
        current_coping_mech="Duty_Based",
        coping_shade_signal="machine ki tarah chal raha hoon",
        user_receptiveness="low", is_new_problem=False, stigma_cue=False, risk_signal=None,
    )
    session1 = SessionState(session_id="test", user_id="test")
    result1 = compute_full_strategy(state1, session1)
    print(f"Test 1: {result1}")
    assert result1.current_phase == "Exploration"
    assert result1.selected_strategy == "RESTATEMENT_OR_PARAPHRASING"
    assert result1.restatement_lens == "B"  # Duty_Based primary lens

    # Test 2: Moderate intensity after Exploration with stigma → phase advances to
    # Insight, but S5 P2 (stigma_cue override) forces RESTATEMENT_OR_PARAPHRASING.
    state2 = AnalyzerState(
        emotion_type="shame", emotion_intensity=3,
        problem_type="Family_Dynamics",
        current_coping_mech="Relational_Preservation",
        coping_shade_signal="papa ko kya bataunga",
        user_receptiveness="medium", is_new_problem=False, stigma_cue=True, risk_signal=None,
    )
    session2 = SessionState(session_id="test", user_id="test", phase_history=["Exploration"])
    result2 = compute_full_strategy(state2, session2)
    print(f"Test 2: {result2}")
    assert result2.current_phase == "Insight"
    assert result2.selected_strategy == "RESTATEMENT_OR_PARAPHRASING"
    assert result2.restatement_lens == "C"  # "papa" -> relational lens

    # Test 3: Low intensity after Insight → Action + PROVIDING_SUGGESTIONS.
    state3 = AnalyzerState(
        emotion_type="hope", emotion_intensity=2,
        problem_type="Academic_Pressure",
        current_coping_mech="Sequential",
        coping_shade_signal="ek kaam khatam kar leta hoon",
        user_receptiveness="high", is_new_problem=False, stigma_cue=False, risk_signal=None,
    )
    session3 = SessionState(
        session_id="test", user_id="test",
        phase_history=["Exploration", "Insight"],
    )
    result3 = compute_full_strategy(state3, session3)
    print(f"Test 3: {result3}")
    assert result3.current_phase == "Action"
    assert result3.selected_strategy == "PROVIDING_SUGGESTIONS"

    # Test 4: New problem mid-session → Exploration regardless of history.
    state4 = AnalyzerState(
        emotion_type="panic", emotion_intensity=2,
        problem_type="Financial_Debt",
        current_coping_mech="Duty_Based",
        coping_shade_signal="EMI nahi bhar paunga",
        user_receptiveness="high", is_new_problem=True, stigma_cue=False, risk_signal=None,
    )
    session4 = SessionState(
        session_id="test", user_id="test",
        phase_history=["Exploration", "Insight", "Action"],
    )
    result4 = compute_full_strategy(state4, session4)
    print(f"Test 4: {result4}")
    assert result4.current_phase == "Exploration"
    assert result4.restatement_lens == "B"  # "EMI" → role/duty

    # ---- Extra targeted tests for anti-repetition + lens ----

    # Anti-repetition A1: same strategy twice → next in allowed list.
    session_rep = SessionState(
        session_id="t", user_id="t",
        phase_history=["Exploration", "Exploration"],
        strategy_history=["RESTATEMENT_OR_PARAPHRASING", "RESTATEMENT_OR_PARAPHRASING"],
    )
    state_rep = AnalyzerState(
        emotion_type="fear", emotion_intensity=4, problem_type="Employment",
        current_coping_mech="Duty_Based",
        coping_shade_signal="kaam chala nahi paa raha",
        user_receptiveness="medium", is_new_problem=False, stigma_cue=False, risk_signal=None,
    )
    result_rep = compute_full_strategy(state_rep, session_rep)
    print(f"Anti-rep A1 test: {result_rep}")
    assert result_rep.current_phase == "Exploration"
    assert result_rep.selected_strategy == "QUESTION"

    # Anti-repetition A2: QUESTION twice → RESTATEMENT_OR_PARAPHRASING.
    session_q2 = SessionState(
        session_id="t", user_id="t",
        phase_history=["Exploration", "Exploration"],
        strategy_history=["QUESTION", "QUESTION"],
    )
    result_q2 = compute_full_strategy(state_rep, session_q2)
    print(f"Anti-rep A2 test: {result_q2}")
    assert result_q2.selected_strategy == "RESTATEMENT_OR_PARAPHRASING"

    # Lens C (relational): mention of "papa" should pick lens C.
    state_c = AnalyzerState(
        emotion_type="shame", emotion_intensity=4, problem_type="Family_Conflict",
        current_coping_mech="Relational_Preservation",
        coping_shade_signal="papa ki nazar mein gir gaya hoon",
        user_receptiveness="low", is_new_problem=False, stigma_cue=False, risk_signal=None,
    )
    result_c = compute_full_strategy(state_c, SessionState(session_id="t", user_id="t"))
    print(f"Lens C test: {result_c}")
    assert result_c.selected_strategy == "RESTATEMENT_OR_PARAPHRASING"
    assert result_c.restatement_lens == "C"

    # Insight + Somatization → REFLECTION_OF_FEELINGS (I2 rule).
    state_som = AnalyzerState(
        emotion_type="exhaustion", emotion_intensity=3,
        problem_type="Health_Chronic_Illness",
        current_coping_mech="Somatization",
        coping_shade_signal="seena bhari rehta hai",
        user_receptiveness="medium", is_new_problem=False, stigma_cue=False, risk_signal=None,
    )
    session_som = SessionState(
        session_id="t", user_id="t",
        phase_history=["Exploration"],
    )
    result_som = compute_full_strategy(state_som, session_som)
    print(f"Insight Somatization test: {result_som}")
    assert result_som.current_phase == "Insight"
    assert result_som.selected_strategy == "REFLECTION_OF_FEELINGS"

    # Mechanism compatibility — Sequential's PRIMARY lens is F. With an explicit
    # overwhelm phrase like "rona", the lens should resolve to F.
    state_seq = AnalyzerState(
        emotion_type="despair", emotion_intensity=5,
        problem_type="Family_Dynamics",
        current_coping_mech="Sequential",
        coping_shade_signal="raat bhar rona aata hai",
        user_receptiveness="low", is_new_problem=False, stigma_cue=False, risk_signal=None,
    )
    result_seq = compute_full_strategy(state_seq, SessionState(session_id="t", user_id="t"))
    print(f"Sequential primary-lens-F test: {result_seq}")
    assert result_seq.selected_strategy == "RESTATEMENT_OR_PARAPHRASING"
    assert result_seq.restatement_lens == "F"

    # Sequential coping with a BODY shade ("sir mein dard") MUST NOT pick Lens A
    # (forbidden for Sequential). Should fall through and end on Sequential's
    # primary F.
    state_seq_body = AnalyzerState(
        emotion_type="exhaustion", emotion_intensity=5,
        problem_type="Health_Chronic_Illness",
        current_coping_mech="Sequential",
        coping_shade_signal="sir mein dard rehta hai",
        user_receptiveness="low", is_new_problem=False, stigma_cue=False, risk_signal=None,
    )
    result_seq_body = compute_full_strategy(state_seq_body, SessionState(session_id="t", user_id="t"))
    print(f"Sequential lens-A-forbidden test: {result_seq_body}")
    assert result_seq_body.restatement_lens != "A", "Lens A is forbidden for Sequential"

    # Somatization with body shade should pick Lens A (Somatization primary, allowed).
    state_a = AnalyzerState(
        emotion_type="fatigue", emotion_intensity=5,
        problem_type="Health_Chronic_Illness",
        current_coping_mech="Somatization",
        coping_shade_signal="sir mein dard rehta hai",
        user_receptiveness="low", is_new_problem=False, stigma_cue=False, risk_signal=None,
    )
    result_a = compute_full_strategy(state_a, SessionState(session_id="t", user_id="t"))
    print(f"Somatization lens-A test: {result_a}")
    assert result_a.selected_strategy == "RESTATEMENT_OR_PARAPHRASING"
    assert result_a.restatement_lens == "A"

    # Somatization with overwhelm vocabulary MUST NOT collapse to Lens F.
    state_som_overwhelm = AnalyzerState(
        emotion_type="overwhelm", emotion_intensity=5,
        problem_type="Health_Chronic_Illness",
        current_coping_mech="Somatization",
        coping_shade_signal="rona aa raha hai bas",
        user_receptiveness="low", is_new_problem=False, stigma_cue=False, risk_signal=None,
    )
    result_som_ov = compute_full_strategy(state_som_overwhelm, SessionState(session_id="t", user_id="t"))
    print(f"Somatization overwhelm test (F forbidden): {result_som_ov}")
    assert result_som_ov.restatement_lens != "F", "Lens F is forbidden for Somatization"

    # Category normalization — folder name should normalize to canonical.
    state_norm = AnalyzerState(
        emotion_type="anxious",  # alias -> "anxiety"
        emotion_intensity=4,
        problem_type="academic-presuure",  # folder typo -> Academic_Pressure
        current_coping_mech="Duty Based",  # alias -> Duty_Based
        coping_shade_signal="padhai ka pressure",
        user_receptiveness="LOW", is_new_problem=False, stigma_cue=False, risk_signal=None,
    )
    print(
        f"Normalization test: emotion={state_norm.emotion_type!r}, "
        f"problem={state_norm.problem_type!r}, coping={state_norm.current_coping_mech!r}"
    )
    assert state_norm.emotion_type == "anxiety"
    assert state_norm.problem_type == "Academic_Pressure"
    assert state_norm.current_coping_mech == "Duty_Based"
    assert state_norm.user_receptiveness == "low"

    # ---- S5 ACTIVE PROTOCOLS overrides ----

    # P2 (stigma_cue) overrides Action phase: even though intensity says Action,
    # stigma_cue forces RESTATEMENT_OR_PARAPHRASING.
    state_stigma_action = AnalyzerState(
        emotion_type="shame", emotion_intensity=2,
        problem_type="Gender_Identity",
        current_coping_mech="Relational_Preservation",
        coping_shade_signal="log kya kahenge",
        user_receptiveness="medium", is_new_problem=False, stigma_cue=True, risk_signal=None,
    )
    session_stigma_action = SessionState(
        session_id="t", user_id="t",
        phase_history=["Exploration", "Insight"],
    )
    result_stigma_action = compute_full_strategy(state_stigma_action, session_stigma_action)
    print(f"S5 P2 stigma override (Action phase): {result_stigma_action}")
    assert result_stigma_action.current_phase == "Action"  # phase still computed normally
    assert result_stigma_action.selected_strategy == "RESTATEMENT_OR_PARAPHRASING"

    # P1 (risk_signal) overrides everything else, regardless of phase or coping.
    state_risk = AnalyzerState(
        emotion_type="despair", emotion_intensity=3,
        problem_type="Health_Chronic_Illness",
        current_coping_mech="Sequential",
        coping_shade_signal="ab aage jeena bhi mushkil hai",
        user_receptiveness="low", is_new_problem=False, stigma_cue=False,
        risk_signal="ab aage jeena bhi mushkil hai",
    )
    session_risk = SessionState(
        session_id="t", user_id="t",
        phase_history=["Exploration"],
    )
    result_risk = compute_full_strategy(state_risk, session_risk)
    print(f"S5 P1 risk override: {result_risk}")
    assert result_risk.selected_strategy == "RESTATEMENT_OR_PARAPHRASING"

    # Stigma override beats anti-repetition: even if last 2 strategies were
    # RESTATEMENT (which A1 would normally rotate away from), stigma_cue
    # forces RESTATEMENT to repeat.
    state_stigma_rep = AnalyzerState(
        emotion_type="shame", emotion_intensity=4,
        problem_type="Family_Dynamics",
        current_coping_mech="Relational_Preservation",
        coping_shade_signal="papa ka kya hoga",
        user_receptiveness="low", is_new_problem=False, stigma_cue=True, risk_signal=None,
    )
    session_stigma_rep = SessionState(
        session_id="t", user_id="t",
        strategy_history=["RESTATEMENT_OR_PARAPHRASING", "RESTATEMENT_OR_PARAPHRASING"],
        phase_history=["Exploration", "Exploration"],
    )
    result_stigma_rep = compute_full_strategy(state_stigma_rep, session_stigma_rep)
    print(f"S5 P2 beats anti-repetition: {result_stigma_rep}")
    assert result_stigma_rep.selected_strategy == "RESTATEMENT_OR_PARAPHRASING"

    # ---- NEW: Help-seeking detector ----
    assert detect_help_seeking("Solution kya h yar iska") is True
    assert detect_help_seeking("Tum help karo na") is True
    assert detect_help_seeking("Kya karu yaar ab?") is True
    assert detect_help_seeking("Batao na kya karna chahiye") is True
    assert detect_help_seeking("What should I do now?") is True
    # Negative cases — distress expressions WITHOUT explicit asks for help.
    assert detect_help_seeking("Bas dukhi hoon, kuch samajh nahi aa raha") is False
    assert detect_help_seeking("Bas dukhi hoon") is False
    assert detect_help_seeking("Aaj bahut bura din tha") is False
    assert detect_help_seeking("") is False
    assert detect_help_seeking(None) is False
    print("Help-seeking detector tests passed.")

    # ---- NEW: Receptiveness-driven phase advancement ----
    # Reproduce the screenshot scenario: intensity=4 throughout, but seeker
    # explicitly asks "Solution kya h yar iska" — phase MUST advance off Exploration.
    state_screenshot = AnalyzerState(
        emotion_type="fear", emotion_intensity=4,
        problem_type="Academic_Pressure",
        current_coping_mech="Sequential",
        coping_shade_signal="results ka dar",
        user_receptiveness="medium",  # Analyzer reads medium, NOT high
        is_new_problem=False, stigma_cue=False, risk_signal=None,
    )
    session_screenshot = SessionState(
        session_id="t", user_id="t",
        phase_history=["Exploration", "Exploration", "Exploration"],
        strategy_history=[
            "RESTATEMENT_OR_PARAPHRASING", "QUESTION", "RESTATEMENT_OR_PARAPHRASING"
        ],
    )
    # Without seeker_text, behavior is unchanged (Exploration).
    res_no_text = compute_full_strategy(state_screenshot, session_screenshot)
    print(f"Screenshot scenario WITHOUT help-seeking text: {res_no_text}")
    assert res_no_text.current_phase == "Exploration"

    # WITH the explicit help-seeking text → must advance to Insight.
    res_with_text = compute_full_strategy(
        state_screenshot, session_screenshot,
        seeker_text="Solution kya h yar iska",
    )
    print(f"Screenshot scenario WITH 'Solution kya h yar iska': {res_with_text}")
    assert res_with_text.current_phase == "Insight", (
        f"Expected Insight after explicit help-seeking, got {res_with_text.current_phase}"
    )
    assert res_with_text.selected_strategy in {
        "REFLECTION_OF_FEELINGS", "AFFIRMATION_AND_REASSURANCE", "SELF_DISCLOSURE"
    }
    print("Help-seeking → phase advance test PASSED.")

    # ---- NEW: Help-seeking + prior Insight → Action ----
    state_action_ready = AnalyzerState(
        emotion_type="hope", emotion_intensity=3,
        problem_type="Academic_Pressure",
        current_coping_mech="Sequential",
        coping_shade_signal="kuch karna chahiye",
        user_receptiveness="medium",  # Analyzer says medium, but help-seeking upgrades it
        is_new_problem=False, stigma_cue=False, risk_signal=None,
    )
    session_action_ready = SessionState(
        session_id="t", user_id="t",
        phase_history=["Exploration", "Insight"],
        strategy_history=["QUESTION", "REFLECTION_OF_FEELINGS"],
    )
    res_action = compute_full_strategy(
        state_action_ready, session_action_ready,
        seeker_text="Tum batao na kya karu",
    )
    print(f"Help-seeking + prior Insight + int=3 → Action: {res_action}")
    assert res_action.current_phase == "Action"
    assert res_action.selected_strategy == "PROVIDING_SUGGESTIONS"

    # ---- NEW: Anti-stuck rule ----
    # 4 consecutive Exploration turns + intensity 4 → forced Insight.
    session_stuck = SessionState(
        session_id="t", user_id="t",
        phase_history=["Exploration", "Exploration", "Exploration", "Exploration"],
        strategy_history=[
            "RESTATEMENT_OR_PARAPHRASING", "QUESTION",
            "RESTATEMENT_OR_PARAPHRASING", "QUESTION",
        ],
    )
    state_stuck = AnalyzerState(
        emotion_type="fear", emotion_intensity=4,
        problem_type="Academic_Pressure",
        current_coping_mech="Sequential",
        coping_shade_signal="dar lag raha hai",
        user_receptiveness="low",  # even low receptiveness — anti-stuck still fires
        is_new_problem=False, stigma_cue=False, risk_signal=None,
    )
    res_stuck = compute_full_strategy(state_stuck, session_stuck)
    print(f"Anti-stuck (4× Exploration, int=4): {res_stuck}")
    assert res_stuck.current_phase == "Insight", (
        f"Expected forced Insight, got {res_stuck.current_phase}"
    )

    # Anti-stuck does NOT fire when intensity is genuinely high (5+).
    state_stuck_severe = state_stuck.model_copy(update={"emotion_intensity": 5})
    res_stuck_severe = compute_full_strategy(state_stuck_severe, session_stuck)
    print(f"Anti-stuck does NOT fire at int=5: {res_stuck_severe}")
    assert res_stuck_severe.current_phase == "Exploration"

    # Anti-stuck does NOT fire when risk_signal is set.
    state_stuck_risk = state_stuck.model_copy(
        update={"risk_signal": "ab nahi sambhal pa raha"}
    )
    res_stuck_risk = compute_full_strategy(state_stuck_risk, session_stuck)
    print(f"Anti-stuck does NOT fire on risk_signal: {res_stuck_risk}")
    assert res_stuck_risk.current_phase == "Exploration"
    assert res_stuck_risk.selected_strategy == "RESTATEMENT_OR_PARAPHRASING"

    # ---- NEW: Supporter-text repetition guard ----
    from core.schemas import TurnRecord
    session_repetition = SessionState(
        session_id="t", user_id="t",
        phase_history=["Exploration", "Exploration"],
        strategy_history=[
            "RESTATEMENT_OR_PARAPHRASING", "RESTATEMENT_OR_PARAPHRASING"
        ],
        turn_history=[
            TurnRecord(
                turn_id=1, speaker="Seeker",
                text="results ka dar lag raha hai",
            ),
            TurnRecord(
                turn_id=2, speaker="Supporter",
                text="results ka dar tumhe andar se ghabrahat deta hai na",
            ),
            TurnRecord(turn_id=3, speaker="Seeker", text="haan sach me"),
            TurnRecord(
                turn_id=4, speaker="Supporter",
                text="results ka dar sach me itna ghabrahat deta hai na",
            ),
        ],
    )
    state_repetition = AnalyzerState(
        emotion_type="fear", emotion_intensity=4,
        problem_type="Academic_Pressure",
        current_coping_mech="Sequential",
        coping_shade_signal="dar lag raha hai",
        user_receptiveness="medium",
        is_new_problem=False, stigma_cue=False, risk_signal=None,
    )
    res_rep = compute_full_strategy(state_repetition, session_repetition)
    print(f"Supporter-repetition guard: {res_rep}")
    # last_strategy was RESTATEMENT_OR_PARAPHRASING, so guard rotates to QUESTION.
    assert res_rep.selected_strategy == "QUESTION", (
        f"Expected QUESTION (rotated away from repeated RESTATEMENT), "
        f"got {res_rep.selected_strategy}"
    )

    # ---- NEW: explain_phase_decision smoke test ----
    why = explain_phase_decision(
        state_screenshot, session_screenshot,
        seeker_text="Solution kya h yar iska",
    )
    print(f"explain_phase_decision (help-seeking): {why}")
    assert "high recept" in why and "help-seeking" in why

    why_stuck = explain_phase_decision(state_stuck, session_stuck)
    print(f"explain_phase_decision (anti-stuck): {why_stuck}")
    assert "anti-stuck" in why_stuck

    why_locked = explain_phase_decision(
        AnalyzerState(
            emotion_type="fear", emotion_intensity=5,
            problem_type="Academic_Pressure",
            current_coping_mech="Sequential",
            coping_shade_signal="bahut dar",
            user_receptiveness="low", is_new_problem=False,
            stigma_cue=False, risk_signal=None,
        ),
        SessionState(session_id="t", user_id="t"),
    )
    print(f"explain_phase_decision (intensity-locked): {why_locked}")
    assert "int=5" in why_locked and "Exploration" in why_locked

    # ---- NEW B4: SELF_DISCLOSURE bias in Insight (rapport built) ----
    # Need intensity=3 + last phase=Exploration so phase computes to Insight
    # (Rule 5). At intensity≤2 we'd jump to Action via Rule 7.
    state_rapport = AnalyzerState(
        emotion_type="acceptance", emotion_intensity=3,
        problem_type="Academic_Pressure",
        current_coping_mech="Sequential",
        coping_shade_signal="thoda samajh aa raha hai",
        user_receptiveness="medium", is_new_problem=False,
        stigma_cue=False, risk_signal=None,
    )
    session_rapport = SessionState(
        session_id="t", user_id="t",
        turn_count=4,
        phase_history=["Exploration", "Exploration", "Insight", "Exploration"],
        strategy_history=[
            "RESTATEMENT_OR_PARAPHRASING", "QUESTION",
            "REFLECTION_OF_FEELINGS", "RESTATEMENT_OR_PARAPHRASING",
        ],
        intensity_trajectory=[4, 4, 3, 3],
    )
    res_disclose = compute_full_strategy(state_rapport, session_rapport)
    print(f"B4 SELF_DISCLOSURE bias (rapport built): {res_disclose}")
    assert res_disclose.current_phase == "Insight"
    assert res_disclose.selected_strategy == "SELF_DISCLOSURE", (
        f"expected SELF_DISCLOSURE, got {res_disclose.selected_strategy}"
    )

    # B4 negative: stigma_cue blocks SELF_DISCLOSURE bias
    state_stigma_block = state_rapport.model_copy(update={"stigma_cue": True})
    res_no_disclose = compute_full_strategy(state_stigma_block, session_rapport)
    print(f"B4 stigma blocks SELF_DISCLOSURE: {res_no_disclose}")
    assert res_no_disclose.selected_strategy == "RESTATEMENT_OR_PARAPHRASING", (
        "stigma_cue should override to RESTATEMENT (S5 P2), got "
        f"{res_no_disclose.selected_strategy}"
    )

    # B4 negative: already used → no second SELF_DISCLOSURE
    session_already_used = session_rapport.model_copy(
        update={"self_disclosure_used": True}
    )
    res_used = compute_full_strategy(state_rapport, session_already_used)
    print(f"B4 self_disclosure_used → REFLECTION fallback: {res_used}")
    assert res_used.selected_strategy == "REFLECTION_OF_FEELINGS"

    # ---- NEW B3: post-Action follow-up rotation → EXECUTION ----
    state_action_followup = AnalyzerState(
        emotion_type="hope", emotion_intensity=2,
        problem_type="Academic_Pressure",
        current_coping_mech="Sequential",
        coping_shade_signal="dekhta hoon kar ke",
        user_receptiveness="medium", is_new_problem=False,
        stigma_cue=False, risk_signal=None,
    )
    session_post_action = SessionState(
        session_id="t", user_id="t",
        turn_count=5,
        phase_history=["Exploration", "Exploration", "Insight", "Insight", "Action"],
        strategy_history=[
            "RESTATEMENT_OR_PARAPHRASING", "QUESTION",
            "REFLECTION_OF_FEELINGS", "REFLECTION_OF_FEELINGS",
            "PROVIDING_SUGGESTIONS",
        ],
        intensity_trajectory=[4, 4, 3, 3, 2],
        self_disclosure_used=False,
    )
    res_followup = compute_full_strategy(state_action_followup, session_post_action)
    print(f"B3 post-Action follow-up (last=PROVIDING_SUGGESTIONS): {res_followup}")
    assert res_followup.current_phase == "Action"
    assert res_followup.selected_strategy == "EXECUTION", (
        f"expected EXECUTION follow-up, got {res_followup.selected_strategy}"
    )

    # Scenario: 3 consecutive Insight turns at int=3 with low receptiveness
    # (so R3 wouldn't fire). Without R2b, the bot would stay in Insight
    # for a 4th time and the user gets another reflection instead of help.
    state_stuck_insight = AnalyzerState(
        emotion_type="confusion",
        emotion_intensity=3,
        problem_type="Academic_Pressure",
        current_coping_mech="Sequential",
        coping_shade_signal="kya karu",
        user_receptiveness="low",   # so R3 doesn't fire
        is_new_problem=False,
        stigma_cue=False,
        risk_signal=None,
    )
    session_stuck_insight = SessionState(
        session_id="t_r2b",
        user_id="u_r2b",
        phase_history=["Exploration", "Insight", "Insight", "Insight"],
        strategy_history=[
            "RESTATEMENT_OR_PARAPHRASING",
            "REFLECTION_OF_FEELINGS",
            "REFLECTION_OF_FEELINGS",
            "REFLECTION_OF_FEELINGS",
        ],
        intensity_trajectory=[4, 3, 3, 3],
        coping_trajectory=["Sequential"] * 4,
        turn_count=4,
        # Journey markers as set by SessionManager.update_after_turn:
        phase_first_reached={"Exploration": 1, "Insight": 2},
        turns_in_current_phase=3,
    )
    res_stuck_ins = compute_full_strategy(state_stuck_insight, session_stuck_insight)
    print(
        f"R2b stuck-in-Insight (3× Insight, int=3, low recept): {res_stuck_ins}"
    )
    assert res_stuck_ins.current_phase == "Action", (
        f"expected forced Action, got {res_stuck_ins.current_phase}"
    )

    # Same setup but intensity=5 — should NOT fire (genuine high distress).
    state_stuck_insight_hot = state_stuck_insight.model_copy(
        update={"emotion_intensity": 5}
    )
    res_stuck_ins_hot = compute_full_strategy(
        state_stuck_insight_hot, session_stuck_insight
    )
    print(
        f"R2b does NOT fire at int=5 (still distressed): {res_stuck_ins_hot}"
    )
    assert res_stuck_ins_hot.current_phase != "Action", (
        f"R2b mis-fired at high intensity → got {res_stuck_ins_hot.current_phase}"
    )

    # explain_phase_decision should surface R2b cleanly.
    reason_r2b = explain_phase_decision(state_stuck_insight, session_stuck_insight)
    print(f"explain_phase_decision (R2b): {reason_r2b}")
    assert reason_r2b.startswith("R2b "), f"R2b explainer missing: {reason_r2b!r}"

    print("\nAll phase gate tests passed!")
