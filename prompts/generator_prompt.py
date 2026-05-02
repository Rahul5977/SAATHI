"""
Generator (Agent 2) prompt builder.

Composes the chat-completions messages array used by the Generator LLM to
produce the final Hinglish response shown to the seeker.

Inputs flow in from:
  - `core.schemas.AnalyzerState`        — output of Agent 1 (Analyzer)
  - `core.schemas.StrategyDecision`     — output of `core.phase_gate.compute_full_strategy`
  - `core.schemas.TurnRecord` list      — recent conversation history
  - `retrieval.generator_retriever`     — pre-formatted few-shot block + negative example

Layout of the produced messages:
  1. SYSTEM — SAATHI identity + persona profile + phase/strategy/lens directive
              + coping-shade mirror requirement + stigma cue + risk override
              (if any) + the WHAT-NOT-TO-DO negative example
  2. USER   — retrieved few-shot examples + recent history + new seeker text +
              compact "generate now" checklist

The Generator is expected to reply with PLAIN TEXT (no JSON, no labels).
Downstream safety/post-processing is handled by `agents/safety_checker.py`.
"""

from __future__ import annotations

from typing import Optional

from config import (
    SAATHI_CARE_TAG_FREQ,
    SAATHI_EMOJI_ENABLED,
    SAATHI_FACTS_WINDOW,
)
from core.schemas import AnalyzerState, SessionState, StrategyDecision, TurnRecord
from prompts.analyzer_prompt import format_history
from prompts.system_prompts import (
    LENS_DESCRIPTIONS,
    PHASE_INSTRUCTIONS,
    SAATHI_SYSTEM_PROMPT,
    STRATEGY_HINTS,
)


# ---------------------------------------------------------------------------
# Persona profiles
# ---------------------------------------------------------------------------
# `hindi_ratio` is the target proportion of Hindi-origin words in SAATHI's
# response (0.0 = pure English, 1.0 = pure Hindi). `note` is a human-readable
# vocabulary hint dropped into the system prompt verbatim.
PERSONA_PROFILES: dict[str, dict] = {
    "P0":  {"name": "Unknown",            "hindi_ratio": 0.60, "note": "Neutral register, standard Hinglish"},
    "P1":  {"name": "Rural student",      "hindi_ratio": 0.80, "note": "Simple Hindi-heavy, avoid English jargon, use desi expressions"},
    "P2":  {"name": "Small-town student", "hindi_ratio": 0.70, "note": "Mix of Hindi and basic English, relatable tone"},
    "P3":  {"name": "Female student",     "hindi_ratio": 0.65, "note": "Warm, empathetic register, gender-sensitive"},
    "P4":  {"name": "IT professional",    "hindi_ratio": 0.50, "note": "Comfortable with English tech terms, metro vocabulary"},
    "P5":  {"name": "First-gen college",  "hindi_ratio": 0.85, "note": "Very simple vocabulary, no fancy English, high Hindi ratio"},
    "P6":  {"name": "Postgrad student",   "hindi_ratio": 0.55, "note": "More English, academic vocabulary okay"},
    "P7":  {"name": "Village background", "hindi_ratio": 0.95, "note": "Almost pure Hindi, very simple words, avoid ALL English"},
    "P8":  {"name": "Working student",    "hindi_ratio": 0.60, "note": "Practical, work-related vocabulary"},
    "P9":  {"name": "Creative field",     "hindi_ratio": 0.55, "note": "Expressive, emotional vocabulary"},
    "P10": {"name": "Medical student",    "hindi_ratio": 0.50, "note": "Can use health terms but not clinical psychology terms"},
    "P11": {"name": "English-medium",     "hindi_ratio": 0.30, "note": "More English-heavy, comfortable with English emotional vocab"},
    "P12": {"name": "Competitive exam",   "hindi_ratio": 0.70, "note": "UPSC/GATE/CAT vocabulary, pressure-focused"},
}


# Phase-specific one-line execution hints repeated in the USER message
# checklist. Kept short on purpose — the heavy explanation is in
# PHASE_INSTRUCTIONS in the SYSTEM message.
_PHASE_GEN_HINT: dict[str, str] = {
    "Exploration": "Mirror their feelings. Use their coping shade phrase. Show you heard them. NO advice.",
    "Insight":     "Gently reflect what's beneath their words. No direct advice yet.",
    "Action":      "Offer ONE small, practical, concrete step tied to a fact they actually mentioned.",
}

# Number of trailing turns of history shown to the Generator (older turns are
# already summarized in `analyzer_state` from prior Analyzer calls).
_HISTORY_WINDOW = 6


# ---------------------------------------------------------------------------
# Care-gesture pool — small one-liners SAATHI may slip in occasionally to
# feel like a friend who notices basic things, not a therapy bot.
# ---------------------------------------------------------------------------
_CARE_TAG_POOL: list[str] = [
    "Aaj khaana time pe khaya?",
    "Neend ho rahi hai theek se?",
    "Pani le pehle, phir baat karte hain.",
    "Thoda chai pi?",
    "Ek lambi saans le pehle.",
    "5 second ke liye aankhe band kar.",
]


# ---------------------------------------------------------------------------
# Voice modulation by emotional intensity — a one-line tone hint we inject
# into the system prompt so the Generator calibrates register to the moment.
# ---------------------------------------------------------------------------
def _voice_modulation(intensity: int) -> str:
    """Return a tone hint string keyed to intensity (1-6)."""
    if intensity >= 6:
        return (
            "TONE: minimal, holding. Crisis register. 1-2 short sentences "
            "max. No questions. No metaphors. Just 'Main hoon yahan' energy."
        )
    if intensity >= 5:
        return (
            "TONE: soft, slow, no humor. Severe distress. Short sentences. "
            "No similes. No 'jaise...'. Hold space, don't try to lift."
        )
    if intensity >= 4:
        return (
            "TONE: warm and grounded. They're flooded but functioning. "
            "Mirror gently, don't perform empathy. Skip the poetry."
        )
    if intensity == 3:
        return (
            "TONE: warm and conversational. Match their tempo. A small light "
            "touch is OK if it fits the moment."
        )
    # intensity 1-2: low distress, room for personality.
    return (
        "TONE: warm, friendly, peer-to-peer. Light humor allowed if it "
        "fits. Keep it short and natural — they're settling."
    )


def _select_care_tag(turn_count: int, last_care_turn: int, intensity: int) -> Optional[str]:
    """Decide whether to inject a care-gesture line this turn.

    Skip during high distress (intensity >= 5) — a "khaana khaya?" while the
    seeker is breaking down feels jarring. Otherwise, fire every
    `SAATHI_CARE_TAG_FREQ` turns.
    """
    if SAATHI_CARE_TAG_FREQ <= 0 or intensity >= 5 or turn_count <= 0:
        return None
    turns_since = turn_count - last_care_turn
    if turns_since < SAATHI_CARE_TAG_FREQ:
        return None
    # Rotate by turn_count so we don't always pick the same tag.
    return _CARE_TAG_POOL[turn_count % len(_CARE_TAG_POOL)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _format_facts_block(
    new_facts: list[str],
    facts_log: list[str],
) -> Optional[str]:
    """Render the CONCRETE FACTS block, or return None if there's nothing
    to show. De-duplicates while preserving recency order."""
    seen: set[str] = set()
    ordered: list[str] = []
    # Newest facts first (they were just extracted), then older log entries.
    for f in (new_facts or []) + list(reversed(facts_log[-SAATHI_FACTS_WINDOW:] or [])):
        key = f.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(f.strip())
    if not ordered:
        return None
    bulleted = "\n".join(f"  - {f}" for f in ordered[:SAATHI_FACTS_WINDOW])
    return (
        "CONCRETE FACTS the seeker has shared so far (USE AT LEAST ONE BY "
        "NAME if it fits this turn — never speak only in abstractions when "
        "specifics exist):\n" + bulleted
    )


def _length_hint(seeker_text: str) -> str:
    """Tell the Generator how long the reply should be, based on seeker
    input length. This is the single biggest 'feels like a bot' tell."""
    n_words = len((seeker_text or "").split())
    if n_words <= 5:
        return (
            "LENGTH BUDGET: seeker sent a SHORT message — your reply must be "
            "1 short sentence (max 12 words). An interjection alone is fine: "
            "\"Hmm.\" / \"Acha.\" / \"Bata bata.\""
        )
    if n_words <= 20:
        return (
            "LENGTH BUDGET: seeker's message is medium-length — reply with "
            "1-2 sentences (max ~30 words). Stay conversational."
        )
    return (
        "LENGTH BUDGET: seeker shared a longer message — reply with up to "
        "3 sentences. NEVER more. If you have nothing extra, say less."
    )


def build_generator_prompt(
    seeker_text: str,
    analyzer_state: AnalyzerState,
    strategy_decision: StrategyDecision,
    conversation_history: list[TurnRecord],
    retrieved_examples: str,
    negative_example: str,
    persona_code: str = "P0",
    session: Optional[SessionState] = None,
) -> list[dict]:
    """Build the chat-completions messages array for Agent 2 (Generator).

    Args:
        seeker_text:           The new seeker turn (raw).
        analyzer_state:        Output of Agent 1.
        strategy_decision:     Output of `phase_gate.compute_full_strategy`.
        conversation_history:  Full history; we slice to the last
                               `_HISTORY_WINDOW` turns inside this function.
        retrieved_examples:    Already formatted by
                               `GeneratorRetriever.format_for_prompt(...)`.
        negative_example:      Already formatted by
                               `GeneratorRetriever.format_negative_example()`.
        persona_code:          One of `PERSONA_PROFILES` keys; falls back to
                               "P0" if unknown.
        session:               Optional `SessionState` used to inject the
                               rolling facts log, care-gesture rotation, and
                               continuity hints. If omitted (e.g. unit tests),
                               the prompt still works but loses the
                               session-aware niceties.

    Returns:
        A list of `{"role": ..., "content": ...}` dicts ready to pass to
        `BaseLLM.generate_text(...)` or `generate_stream(...)`.
    """
    persona  = PERSONA_PROFILES.get(persona_code, PERSONA_PROFILES["P0"])
    phase    = strategy_decision.current_phase
    strategy = strategy_decision.selected_strategy
    lens     = strategy_decision.restatement_lens

    if phase not in PHASE_INSTRUCTIONS:
        # Defensive: should never trigger because StrategyDecision validates
        # phase via `core.schemas.PHASES`.
        raise ValueError(
            f"Unknown phase '{phase}'. Expected one of {list(PHASE_INSTRUCTIONS)}."
        )

    hindi_pct = int(round(persona["hindi_ratio"] * 100))

    # ---- Session-derived hints (graceful fallback when session is None) ----
    facts_log: list[str] = list(session.facts_log) if session else []
    turn_count: int = session.turn_count if session else 0
    last_care_turn: int = session.last_care_tag_turn if session else 0

    facts_block = _format_facts_block(analyzer_state.concrete_facts, facts_log)
    care_tag = _select_care_tag(
        turn_count=turn_count,
        last_care_turn=last_care_turn,
        intensity=analyzer_state.emotion_intensity,
    )
    voice_hint = _voice_modulation(analyzer_state.emotion_intensity)
    length_hint = _length_hint(seeker_text)

    # ---------------- SYSTEM message ----------------
    system_parts: list[str] = [
        SAATHI_SYSTEM_PROMPT,
        "",
        "## YOUR PERSONA FOR THIS CONVERSATION",
        f"Persona: {persona_code} ({persona['name']})",
        f"Hindi word ratio target: {hindi_pct}% Hindi words in your response",
        f"Vocabulary note: {persona['note']}",
        "",
        "## YOUR CURRENT TASK",
        f"PHASE: {phase}",
        f"INSTRUCTION: {PHASE_INSTRUCTIONS[phase]}",
        "",
        f"STRATEGY TO USE: {strategy}",
    ]

    if strategy in STRATEGY_HINTS:
        system_parts.append(f"STRATEGY EXECUTION: {STRATEGY_HINTS[strategy]}")

    if lens and lens in LENS_DESCRIPTIONS:
        system_parts.append(
            f"RESTATEMENT LENS: {lens} — {LENS_DESCRIPTIONS[lens]}"
        )

    system_parts.extend([
        "",
        voice_hint,
        "",
        length_hint,
        "",
        f"COPING SHADE TO MIRROR: \"{analyzer_state.coping_shade_signal}\"",
        "→ Echo this phrase (or a direct paraphrase) once in your response — "
        "naturally, not mechanically. Don't quote it as a label.",
        "",
        "STIGMA CUE: " + (
            "YES — Be extra gentle. Do NOT push for disclosure or openness. "
            "Respect the shame."
            if analyzer_state.stigma_cue
            else "No — Normal warmth."
        ),
    ])

    if facts_block is not None:
        system_parts.extend(["", facts_block])

    if care_tag is not None:
        system_parts.extend([
            "",
            "CARE TAG (optional): if it fits naturally and intensity is low, "
            f"you MAY end with this single short line: \"{care_tag}\". "
            "Skip it if the moment is heavy.",
        ])

    if SAATHI_EMOJI_ENABLED and analyzer_state.emotion_intensity <= 3:
        system_parts.extend([
            "",
            "EMOJI: at most ONE emoji is allowed at the end if it adds warmth "
            "(🙏, 💪, 😅, 🫂). Skip entirely if the moment is heavy.",
        ])

    # If the Analyzer flagged a crisis phrase, the deterministic phase_gate
    # has already forced strategy=RESTATEMENT_OR_PARAPHRASING. We surface the
    # override here so the Generator stays grounded in the seeker's words and
    # never tries to "fix" or redirect during a crisis turn.
    if analyzer_state.risk_signal:
        system_parts.extend([
            "",
            "## ACTIVE SAFETY OVERRIDE",
            f"The seeker expressed a high-risk signal: "
            f"\"{analyzer_state.risk_signal}\".",
            "Stay with them. Mirror exactly. Do NOT advise, redirect to "
            "professionals, minimize, or rush to action — a separate safety "
            "module handles HITL escalation. Your only job here is to make "
            "them feel heard right now.",
        ])

    system_parts.extend(["", negative_example])

    # ---------------- USER message ----------------
    history_window = conversation_history[-_HISTORY_WINDOW:]
    history_text = format_history(history_window)
    gen_hint = _PHASE_GEN_HINT[phase]

    user_parts: list[str] = [
        "RETRIEVED EXAMPLES FROM SIMILAR CONVERSATIONS:",
        "(Study these for tone and Hinglish ratio. Do NOT copy poetry/metaphors "
        "from them verbatim — your reply should sound natural, not literary.)",
        "",
        retrieved_examples,
        "",
        f"CONVERSATION HISTORY (last {len(history_window)} turns):",
        history_text,
        "",
        "SEEKER JUST SAID:",
        f"\"{seeker_text}\"",
        "",
        "Generate the SAATHI response now.",
        f"• Length: per LENGTH BUDGET above (seeker wrote {len(seeker_text.split())} words)",
        f"• Hinglish ({hindi_pct}% Hindi words)",
        f"• Echo coping shade: \"{analyzer_state.coping_shade_signal}\"",
        f"• Strategy: {strategy}",
        f"• {gen_hint}",
        "• Reference at least one CONCRETE FACT by name if any are listed above",
        "• Open with a one-line VALIDATION before reflecting (\"Haan yaar...\", "
        "\"Bhai, this is tough.\", \"Oof.\")",
        "• If the seeker brought a positive frame, AMPLIFY it — don't reflect on it",
        "• NO poetry/simile if you've used one in the last 2 turns. Plain Hinglish.",
        "• ONE gentle question OR one validating observation at the end. Not both.",
        "• Do NOT use any prohibited clinical vocabulary",
        "• Return ONLY the response text — no JSON, no labels, no explanation",
    ]

    return [
        {"role": "system", "content": "\n".join(system_parts)},
        {"role": "user",   "content": "\n".join(user_parts)},
    ]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from core.schemas import AnalyzerState, StrategyDecision, TurnRecord

    # Realistic inputs that mirror a real pipeline turn.
    analyzer_state = AnalyzerState(
        emotion_type="exhaustion",
        emotion_intensity=5,
        problem_type="Employment_Livelihood",
        current_coping_mech="Duty_Based",
        coping_shade_signal="machine ki tarah chal raha hoon",
        user_receptiveness="low",
        is_new_problem=False,
        stigma_cue=False,
        risk_signal=None,
    )
    strategy_decision = StrategyDecision(
        current_phase="Exploration",
        selected_strategy="RESTATEMENT_OR_PARAPHRASING",
        restatement_lens="B",  # role/duty frame
    )
    history = [
        TurnRecord(turn_id=1, speaker="Seeker",
                   text="Bhai aaj subah se bahut load hai.",
                   emotion="exhaustion", intensity=4),
        TurnRecord(turn_id=2, speaker="Supporter",
                   text="Subah se hi yeh load uth gaya — kaafi bhaari shuruaat lagti hai.",
                   strategy="RESTATEMENT_OR_PARAPHRASING", phase="Exploration"),
    ]
    retrieved = (
        "--- EXAMPLE 1 ---\n"
        "STRATEGY: RESTATEMENT_OR_PARAPHRASING  |  LENS: N/A  |  COPING: Duty_Based\n"
        "PHASE: Exploration  |  INTENSITY: 5  |  EMOTION: exhaustion\n\n"
        "SEEKER SAID:\n\"Roz 15 ghante kaam, ruk nahi sakta.\"\n\n"
        "SAATHI SAID:\n\"15 ghante ka yeh non-stop chalna tumhe andar se "
        "nichod raha hai, par engine band karne ka option hi nahi mil raha.\"\n"
    )
    negative = (
        "--- WHAT NOT TO DO ---\n"
        "❌ WRONG: \"It sounds like you might be experiencing burnout...\"\n"
        "✅ RIGHT: \"Machine ki tarah chalte chalte thak gaye ho...\"\n"
    )

    # ---- Case 1: standard Exploration turn ----
    msgs = build_generator_prompt(
        seeker_text=(
            "Aaj bhi wahi haal hai bhai. Office, ghar, EMI — sab ek saath. "
            "Machine ki tarah chal raha hoon par ruk bhi nahi sakta."
        ),
        analyzer_state=analyzer_state,
        strategy_decision=strategy_decision,
        conversation_history=history,
        retrieved_examples=retrieved,
        negative_example=negative,
        persona_code="P2",
    )

    print("=" * 78)
    print("CASE 1 — Exploration / RESTATEMENT_OR_PARAPHRASING / lens B / P2")
    print("=" * 78)
    print(f"messages: {len(msgs)}  "
          f"(system={len(msgs[0]['content'])} chars, user={len(msgs[1]['content'])} chars)")

    sys_text = msgs[0]["content"]
    user_text = msgs[1]["content"]

    # Hard contract checks: every required block must appear in the output.
    required_in_system = [
        "SAATHI",                                         # identity
        "Persona: P2",                                    # persona injection
        "Hindi word ratio target: 70%",                   # ratio
        "PHASE: Exploration",                             # phase
        PHASE_INSTRUCTIONS["Exploration"],                # phase instruction
        "STRATEGY TO USE: RESTATEMENT_OR_PARAPHRASING",   # strategy
        "RESTATEMENT LENS: B —",                          # lens
        LENS_DESCRIPTIONS["B"],                           # lens detail
        "COPING SHADE TO MIRROR: \"machine ki tarah chal raha hoon\"",
        "STIGMA CUE: No",                                 # stigma branch
        "WHAT NOT TO DO",                                 # negative example
    ]
    for needle in required_in_system:
        assert needle in sys_text, f"missing in SYSTEM: {needle!r}"

    required_in_user = [
        "RETRIEVED EXAMPLES",
        "CONVERSATION HISTORY (last 2 turns):",
        "Aaj bhi wahi haal hai bhai",                     # seeker text
        "Hinglish (70% Hindi words)",
        "machine ki tarah chal raha hoon",                # mirror coping shade
        "Strategy: RESTATEMENT_OR_PARAPHRASING",
        "Mirror their feelings",                          # phase gen hint
    ]
    for needle in required_in_user:
        assert needle in user_text, f"missing in USER: {needle!r}"

    # Should NOT contain the safety override block when risk_signal is None.
    assert "ACTIVE SAFETY OVERRIDE" not in sys_text
    print("  contract checks OK ✓")

    # ---- Case 2: stigma_cue=True, no lens (non-restatement strategy) ----
    analyzer_stigma = analyzer_state.model_copy(update={
        "emotion_type": "shame",
        "stigma_cue": True,
        "coping_shade_signal": "papa ko kya bataunga",
        "current_coping_mech": "Relational_Preservation",
        "problem_type": "Academic_Pressure",
    })
    decision_q = StrategyDecision(
        current_phase="Insight",
        selected_strategy="QUESTION",
        restatement_lens=None,
    )
    msgs2 = build_generator_prompt(
        seeker_text="Papa ko kya bataunga is baar.",
        analyzer_state=analyzer_stigma,
        strategy_decision=decision_q,
        conversation_history=history,
        retrieved_examples=retrieved,
        negative_example=negative,
        persona_code="P3",
    )
    s2 = msgs2[0]["content"]
    assert "STIGMA CUE: YES" in s2, "stigma_cue=True branch failed"
    assert "RESTATEMENT LENS:" not in s2, "lens line should be absent"
    assert "PHASE: Insight" in s2
    assert "ACTIVE SAFETY OVERRIDE" not in s2
    print("\nCASE 2 — Insight / QUESTION / no lens / P3 / stigma=YES  ✓")

    # ---- Case 3: risk_signal present -> ACTIVE SAFETY OVERRIDE block ----
    analyzer_risk = analyzer_state.model_copy(update={
        "emotion_type": "despair",
        "emotion_intensity": 6,
        "current_coping_mech": "Sequential",
        "coping_shade_signal": "teen saal barbaad kar diye",
        "risk_signal": "sab khatam kar doon, koi fayda nahi raha",
    })
    decision_safety = StrategyDecision(
        current_phase="Exploration",
        selected_strategy="RESTATEMENT_OR_PARAPHRASING",
        restatement_lens=None,
    )
    msgs3 = build_generator_prompt(
        seeker_text="Ab kuch nahi bacha. Sab khatam kar doon, koi fayda nahi raha.",
        analyzer_state=analyzer_risk,
        strategy_decision=decision_safety,
        conversation_history=history,
        retrieved_examples=retrieved,
        negative_example=negative,
        persona_code="P0",
    )
    s3 = msgs3[0]["content"]
    assert "ACTIVE SAFETY OVERRIDE" in s3, "safety override block missing"
    assert "sab khatam kar doon" in s3, "risk phrase not surfaced"
    print("CASE 3 — Crisis turn / safety override block injected           ✓")

    # ---- Case 4: unknown persona falls back to P0 cleanly ----
    msgs4 = build_generator_prompt(
        seeker_text="test",
        analyzer_state=analyzer_state,
        strategy_decision=strategy_decision,
        conversation_history=[],
        retrieved_examples=retrieved,
        negative_example=negative,
        persona_code="P_DOES_NOT_EXIST",
    )
    assert "Persona: P_DOES_NOT_EXIST (Unknown)" in msgs4[0]["content"]
    assert "Hindi word ratio target: 60%" in msgs4[0]["content"]
    assert "CONVERSATION HISTORY (last 0 turns):" in msgs4[1]["content"]
    print("CASE 4 — Unknown persona falls back to P0, empty history OK     ✓")

    print("\ngenerator_prompt.py — all checks passed ✓")
