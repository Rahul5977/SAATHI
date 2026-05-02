"""
SAATHI deterministic phase + strategy + lens engine.

This module is the "brain" between the Analyzer and the Generator.
It runs with **zero LLM calls** — every decision is pure Python and
fully unit-testable.

Public functions
----------------
- compute_phase(analyzer_state, session) -> str
- compute_allowed_strategies(phase, session) -> list[str]
- compute_strategy(phase, analyzer_state, session) -> str
- compute_lens(analyzer_state, seeker_text=None) -> Optional[str]
- compute_full_strategy(analyzer_state, session) -> StrategyDecision
"""

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



# FUNCTION 1: compute_phase

def compute_phase(analyzer_state: AnalyzerState, session: SessionState) -> str:
    """
    Determine the current conversation phase.

    The Phase Gate is sacred — Action strategies NEVER appear during Exploration.

    Rules (evaluated in order, first match wins):
      1. New problem detected → "Exploration"
      2. Intensity >= 4 → "Exploration"
      3. Intensity == 3 AND last phase was "Exploration" → "Insight"
      4. Intensity == 3 AND last phase was NOT "Exploration" → "Exploration"
      5. Intensity <= 2 AND "Insight" appears in last 2 phases → "Action"
      6. Intensity <= 2 AND "Insight" NOT in last 2 phases → "Insight"
      7. Default → "Exploration"
    """
    intensity = analyzer_state.emotion_intensity
    last_phase: Optional[str] = session.phase_history[-1] if session.phase_history else None
    last_two_phases: list[str] = session.phase_history[-2:] if session.phase_history else []

    # Rule 1: a new problem resets the conversation; we must re-validate before advising.
    if analyzer_state.is_new_problem:
        return "Exploration"

    # Rule 2: high distress — too flooded for advice.
    if intensity >= 4:
        return "Exploration"

    # Rule 3: moderate distress, already explored — natural progression to Insight.
    if intensity == 3 and last_phase == "Exploration":
        return "Insight"

    # Rule 4: moderate distress but no prior Exploration — must explore first.
    if intensity == 3 and last_phase != "Exploration":
        return "Exploration"

    # Rule 5: low distress AND we have already done Insight recently — ready for Action.
    if intensity <= 2 and "Insight" in last_two_phases:
        return "Action"

    # Rule 6: low distress but no recent Insight — must consolidate via Insight first.
    if intensity <= 2 and "Insight" not in last_two_phases:
        return "Insight"

    # Rule 7: ultra-defensive default.
    return "Exploration"



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
        # I1: physical symptoms — reflect the emotional root underneath.
        if analyzer_state.current_coping_mech == "Somatization":
            if "REFLECTION_OF_FEELINGS" in allowed:
                return "REFLECTION_OF_FEELINGS"

        # I2: default Insight.
        if "REFLECTION_OF_FEELINGS" in allowed:
            return "REFLECTION_OF_FEELINGS"

        # All Insight strategies exhausted — fall through to final fallback.

    if phase == "Action":
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
) -> StrategyDecision:
    """
    End-to-end deterministic decision: phase → strategy → (optional) lens.
    """
    phase = compute_phase(analyzer_state, session)
    strategy = compute_strategy(phase, analyzer_state, session)
    lens: Optional[str] = None
    if strategy == "RESTATEMENT_OR_PARAPHRASING":
        lens = compute_lens(analyzer_state)

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

    print("\nAll phase gate tests passed!")
