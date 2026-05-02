"""
Analyzer (Agent 1) prompt builder.

The Analyzer's ONLY job is structured extraction — it never composes a reply
to the seeker. It reads (history + previous AnalyzerState + new seeker turn)
and emits a JSON object that conforms to `core.schemas.AnalyzerState`.

This module:
  - Builds the messages array in OpenAI chat-completions format.
  - Pins the schema enum lists to `core.schemas` so the prompt cannot drift
    from the Pydantic contract.
  - Ships 4 static few-shot exemplars covering the four canonical coping
    mechanisms + a high-risk crisis case.

Used by `agents/analyzer.py` (built later) which will:
  llm.generate_json(messages=build_analyzer_prompt(...))
"""

from __future__ import annotations

from typing import Optional

from core.schemas import (
    AnalyzerState,
    COPING_MECHANISMS,
    EMOTION_TYPES,
    PROBLEM_TYPES,
    RECEPTIVENESS_VALUES,
    TurnRecord,
)


# ---------------------------------------------------------------------------
# History formatter
# ---------------------------------------------------------------------------
def format_history(turns: list[TurnRecord]) -> str:
    """Render a `TurnRecord` list into a compact chat transcript with
    per-turn metadata tags the LLM can use as state-tracking hints."""
    if not turns:
        return "No previous turns — this is the start of the conversation."

    lines: list[str] = []
    for t in turns:
        meta = ""
        if t.strategy:
            meta += f" [strategy={t.strategy}]"
        if t.emotion and t.intensity is not None:
            meta += f" [emotion={t.emotion}, intensity={t.intensity}]"
        elif t.emotion:
            meta += f" [emotion={t.emotion}]"
        lines.append(f"{t.speaker}: {t.text}{meta}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt — built once at import time from the canonical schema lists
# ---------------------------------------------------------------------------
def _build_system_message() -> str:
    """Compose the Analyzer's system message. Enum lists come straight from
    `core.schemas` so adding a new emotion/category there auto-updates the
    prompt without manual sync."""
    emotion_pipe = " | ".join(f'"{e}"' for e in EMOTION_TYPES)
    problem_pipe = " | ".join(f'"{p}"' for p in PROBLEM_TYPES)
    coping_pipe  = " | ".join(f'"{c}"' for c in COPING_MECHANISMS)
    recept_pipe  = " | ".join(f'"{r}"' for r in RECEPTIVENESS_VALUES)

    return f"""\
You are the Analyzer module of SAATHI. Your ONLY job is structured information extraction. You do NOT generate any response to the seeker. You read their message and extract their emotional/behavioral state as a JSON object.

Be PRECISE. Extract what is ACTUALLY PRESENT in the text. Do not infer emotions the seeker hasn't expressed. Do not project. If something isn't clearly present, use conservative defaults.

CRITICAL RULE — coping_shade_signal: This must be the EXACT phrase from the seeker's text that reveals HOW they are coping. Not a summary. Not your interpretation. Their literal words. Example: if they say "machine ki tarah chal raha hoon" — that IS the coping shade signal. Copy it exactly (lowercase, trimmed). If the seeker's whole message is a coping shade, copy the most concentrated 4-12 word fragment.

CRITICAL RULE — risk_signal: If the seeker expresses any thought of self-harm, suicide, hopelessness about being alive, or wanting to "end it" / "khatam kar doon" / "mar jaun" / "jeena nahi chahta" — copy the EXACT trigger phrase into `risk_signal` and set `emotion_intensity` to 6. Otherwise return null. Err on the side of flagging — false positives are cheaper than misses.

OUTPUT FORMAT — Return ONLY a valid JSON object matching this schema:
{{
    "emotion_type": {emotion_pipe},
    "emotion_intensity": 1-6 (1=resolved, 2=mild, 3=moderate, 4=high, 5=severe, 6=crisis),
    "problem_type": {problem_pipe},
    "current_coping_mech": {coping_pipe},
    "coping_shade_signal": "<exact phrase from seeker text>",
    "user_receptiveness": {recept_pipe},
    "is_new_problem": true | false,
    "stigma_cue": true | false,
    "risk_signal": "<exact crisis phrase>" or null,
    "concrete_facts": ["<short fact 1>", "<short fact 2>", ...]
}}

CONCRETE_FACTS — extract specific, hard details the seeker mentions THIS turn, so the next reply can reference them by name instead of speaking abstractly. Examples of good facts:
  - "PPT presentation pending"
  - "exam in 1 day"
  - "papa retired last month"
  - "mummy's surgery next week"
  - "interview at TCS Tuesday"
  - "Bangalore se shift hua 2 mahine pehle"
  - "girlfriend with whom 4 years relationship"
Rules:
  - 0 to 5 facts per turn (most turns will have 0-2).
  - Each fact: ≤12 words, in English or Hinglish, noun-phrase style, no full sentences.
  - Only extract facts that are NEW or CLARIFYING. Don't repeat facts from previous turns.
  - Skip emotional descriptions ("dar lag raha") — those go in coping_shade_signal, not here.
  - If nothing concrete, return [].

INTENSITY CALIBRATION:
- 6 (crisis):   Active suicidal ideation, self-harm, "I want to end it", "sab khatam kar doon"
- 5 (severe):   Extreme distress, complete shutdown, "I can't take this anymore", "tut gaya hoon"
- 4 (high):     Significant distress but still functioning, "bahut zyada ho gaya hai"
- 3 (moderate): Noticeable distress, processing feelings, "samajh nahi aa raha"
- 2 (mild):     Low distress, reflecting, some hope, "thoda better hai"
- 1 (resolved): Calm, forward-looking, "ab soch raha hoon"

ANTI-INFLATION RULES (very important — common mistake):
- Brief acknowledgements like "Sach me", "Haan", "Theek hai", "Ok", "Sahi kaha" are NOT high-intensity. They are de-escalation signals — usually intensity 2-3, NOT 4. Intensity must DROP after the seeker accepts a reflection.
- Do not just carry forward the previous intensity. Re-read the new message in isolation first, then check whether the new content contains escalation words (more body symptoms, more catastrophising, more hopelessness) or settling words (agreement, partial insight, calmer tone). Adjust accordingly.
- A 1-3 word reply is rarely intensity 4+. Reserve intensity 4 for messages that contain new distressing content.

RECEPTIVENESS CALIBRATION:
- low:    Flooded, shut down, venting, not ready for any input
- medium: Processing, somewhat open to reflection but not advice
- high:   Asking for help, seeking input. Triggers: "kya karu(n)?", "solution batao", "tum batao na", "help karo", "advice do", "what should I do?", "kaise karu", "raasta dikhao", "samjhao". When ANY of these appear, you MUST set user_receptiveness="high" — it overrides any other read.

COPING MECHANISM SIGNALS:
- Duty_Based:               "karna hi padega", "machine ki tarah", role/obligation framing
- Relational_Preservation:  "papa ko kya bataunga", shame about disappointing family, "kalesh nahi"
- Somatization:             "sir mein dard", "neend nahi aati", body-pain framing of distress
- Sequential:               "ek aur kaam nipta loon phir dekhunga", task-by-task survival

stigma_cue is True when the seeker shows shame, fear of being judged, fear of family/social discovery, or anti-help-seeking sentiment ("log kya kahenge", "natak bol denge", "kamzor samjhenge").

Return ONLY the JSON object. No explanation, no preamble, no markdown fencing.\
"""


_ANALYZER_SYSTEM_MESSAGE = _build_system_message()


# ---------------------------------------------------------------------------
# Static few-shot exemplars
# Cover all 4 coping mechanisms + a high-risk crisis case. problem_type values
# use the canonical C1-C8 names from `core.schemas.PROBLEM_TYPES`.
# ---------------------------------------------------------------------------
_FEW_SHOTS: list[dict[str, str]] = [
    # Example 1 — Duty_Based + exhaustion (Employment_Livelihood, first turn)
    {
        "user": (
            "CONVERSATION HISTORY:\n"
            "No previous turns — this is the start of the conversation.\n\n"
            "PREVIOUS STATE: None — this is the first turn.\n\n"
            "NEW SEEKER MESSAGE:\n"
            "\"Bhai aaj subah se bus machine ki tarah chal raha hoon. Office "
            "ka kaam, ghar ka kaam, EMI bharni hai, papa ki zameen ka masla "
            "hai. Ruk-ne ka time hi nahi milta. Kabhi kabhi lagta hai ki "
            "thak ke gir jaunga lekin ruk nahi sakta.\""
        ),
        "assistant": (
            '{"emotion_type": "exhaustion", "emotion_intensity": 5, '
            '"problem_type": "Employment_Livelihood", '
            '"current_coping_mech": "Duty_Based", '
            '"coping_shade_signal": "machine ki tarah chal raha hoon", '
            '"user_receptiveness": "low", "is_new_problem": false, '
            '"stigma_cue": false, "risk_signal": null, '
            '"concrete_facts": ["EMI bharna hai", "papa ki zameen ka masla"]}'
        ),
    },
    # Example 2 — Relational_Preservation + shame (Academic_Pressure)
    {
        "user": (
            "CONVERSATION HISTORY:\n"
            "Seeker: Marks bahut kharab aaye hain is baar.\n"
            "Supporter: Kitna mushkil hoga tumhare liye yeh result dekhna. "
            "[strategy=RESTATEMENT_OR_PARAPHRASING]\n\n"
            "PREVIOUS STATE: {\"emotion_type\": \"panic\", "
            "\"emotion_intensity\": 4, \"problem_type\": \"Academic_Pressure\", "
            "\"current_coping_mech\": \"Relational_Preservation\", "
            "\"coping_shade_signal\": \"marks bahut kharab\", "
            "\"user_receptiveness\": \"medium\", \"is_new_problem\": false, "
            "\"stigma_cue\": false, \"risk_signal\": null}\n\n"
            "NEW SEEKER MESSAGE:\n"
            "\"Papa ko kya bataunga... unki itni umeedein hain mujhse. Puri "
            "colony mein bolta phirta hai mera beta engineer banega. Ab kya "
            "munh dikhaunga. Kisi ko batane ka mann hi nahi karta.\""
        ),
        "assistant": (
            '{"emotion_type": "shame", "emotion_intensity": 5, '
            '"problem_type": "Academic_Pressure", '
            '"current_coping_mech": "Relational_Preservation", '
            '"coping_shade_signal": "papa ko kya bataunga", '
            '"user_receptiveness": "low", "is_new_problem": false, '
            '"stigma_cue": true, "risk_signal": null, '
            '"concrete_facts": ["marks bahut kharab is baar"]}'
        ),
    },
    # Example 3 — Somatization + overwhelm (Employment_Livelihood, mid-conv)
    {
        "user": (
            "CONVERSATION HISTORY:\n"
            "Seeker: Kaam bahut zyada ho gaya hai.\n"
            "Supporter: Itna kaam ek saath — lagta hai bahut bhaari pad raha "
            "hai. [strategy=RESTATEMENT_OR_PARAPHRASING]\n"
            "Seeker: Haan aur ab health bhi kharab ho rahi hai.\n\n"
            "PREVIOUS STATE: {\"emotion_type\": \"overwhelm\", "
            "\"emotion_intensity\": 4, \"problem_type\": \"Employment_Livelihood\", "
            "\"current_coping_mech\": \"Duty_Based\", "
            "\"coping_shade_signal\": \"kaam bahut zyada\", "
            "\"user_receptiveness\": \"medium\", \"is_new_problem\": false, "
            "\"stigma_cue\": false, \"risk_signal\": null}\n\n"
            "NEW SEEKER MESSAGE:\n"
            "\"Sir mein subah se dard ho raha hai. Pet bhi kharab rehta hai "
            "aaj kal. Doctor ke paas jaane ka time nahi hai. Bas kaam khatam "
            "karna hai deadline se pehle, phir dekhunga.\""
        ),
        "assistant": (
            '{"emotion_type": "overwhelm", "emotion_intensity": 4, '
            '"problem_type": "Employment_Livelihood", '
            '"current_coping_mech": "Somatization", '
            '"coping_shade_signal": "sir mein subah se dard ho raha hai", '
            '"user_receptiveness": "low", "is_new_problem": false, '
            '"stigma_cue": false, "risk_signal": null, '
            '"concrete_facts": ["sir aur pet mein dard", "deadline aane wali"]}'
        ),
    },
    # Example 4 — Sequential + despair WITH risk signal (Academic_Pressure)
    {
        "user": (
            "CONVERSATION HISTORY:\n"
            "Seeker: UPSC ka result aaya, phir se fail.\n"
            "Supporter: Do baar try karke bhi result na aaye — bahut toota hua "
            "mehsoos ho raha hoga. [strategy=REFLECTION_OF_FEELINGS]\n\n"
            "PREVIOUS STATE: {\"emotion_type\": \"despair\", "
            "\"emotion_intensity\": 5, \"problem_type\": \"Academic_Pressure\", "
            "\"current_coping_mech\": \"Sequential\", "
            "\"coping_shade_signal\": \"phir se fail\", "
            "\"user_receptiveness\": \"low\", \"is_new_problem\": false, "
            "\"stigma_cue\": true, \"risk_signal\": null}\n\n"
            "NEW SEEKER MESSAGE:\n"
            "\"Ab kya fayda hai bhai. Teen saal barbaad kar diye. Na job hai "
            "na kuch. Ghar wale bhi thak gaye hain. Kabhi kabhi lagta hai sab "
            "khatam kar doon, koi fayda nahi raha.\""
        ),
        "assistant": (
            '{"emotion_type": "despair", "emotion_intensity": 6, '
            '"problem_type": "Academic_Pressure", '
            '"current_coping_mech": "Sequential", '
            '"coping_shade_signal": "teen saal barbaad kar diye", '
            '"user_receptiveness": "low", "is_new_problem": false, '
            '"stigma_cue": true, '
            '"risk_signal": "sab khatam kar doon, koi fayda nahi raha", '
            '"concrete_facts": ["UPSC fail second time", "3 saal preparation", "no job"]}'
        ),
    },
    # Example 5 — LOW intensity / brief agreement after rapport
    # Critical calibration: short Hinglish acknowledgements like "Sach me",
    # "Haan thoda samajh aaya" should NOT be scored as intensity 4. They
    # signal de-escalation — the seeker is settling, not flooding.
    {
        "user": (
            "CONVERSATION HISTORY:\n"
            "Seeker: Results ka bahut dar hai.\n"
            "Supporter: Results ka dar tumhe andar se itna ghabrata hai na — "
            "jaise har waqt yeh soch chal rahi hai. "
            "[strategy=RESTATEMENT_OR_PARAPHRASING]\n\n"
            "PREVIOUS STATE: {\"emotion_type\": \"fear\", "
            "\"emotion_intensity\": 4, \"problem_type\": \"Academic_Pressure\", "
            "\"current_coping_mech\": \"Sequential\", "
            "\"coping_shade_signal\": \"results ka dar\", "
            "\"user_receptiveness\": \"medium\", \"is_new_problem\": false, "
            "\"stigma_cue\": false, \"risk_signal\": null}\n\n"
            "NEW SEEKER MESSAGE:\n"
            "\"Sach me. Haan thoda samajh aa raha hai.\""
        ),
        "assistant": (
            '{"emotion_type": "acceptance", "emotion_intensity": 2, '
            '"problem_type": "Academic_Pressure", '
            '"current_coping_mech": "Sequential", '
            '"coping_shade_signal": "thoda samajh aa raha hai", '
            '"user_receptiveness": "medium", "is_new_problem": false, '
            '"stigma_cue": false, "risk_signal": null, '
            '"concrete_facts": []}'
        ),
    },
    # Example 6 — EXPLICIT help-seeking → user_receptiveness MUST be "high"
    # Critical calibration: phrases like "Solution kya h?", "Tum batao na",
    # "Kya karu yaar?" are textbook receptiveness=high. The seeker is
    # actively opening the door for guidance. Intensity is moderate (3),
    # not high — the help-seeking itself shows some self-regulation.
    {
        "user": (
            "CONVERSATION HISTORY:\n"
            "Seeker: Results aane wale hain, neend nahi aati.\n"
            "Supporter: Results ka itna pressure hai ki neend bhi cheen le "
            "raha hai — yeh boojh sach mein bahut bhaari hai. "
            "[strategy=REFLECTION_OF_FEELINGS]\n"
            "Seeker: Haan exactly.\n\n"
            "PREVIOUS STATE: {\"emotion_type\": \"fear\", "
            "\"emotion_intensity\": 3, \"problem_type\": \"Academic_Pressure\", "
            "\"current_coping_mech\": \"Sequential\", "
            "\"coping_shade_signal\": \"neend nahi aati\", "
            "\"user_receptiveness\": \"medium\", \"is_new_problem\": false, "
            "\"stigma_cue\": false, \"risk_signal\": null}\n\n"
            "NEW SEEKER MESSAGE:\n"
            "\"Yaar tum batao na, ab main kya karoon? Koi solution batao iska.\""
        ),
        "assistant": (
            '{"emotion_type": "fear", "emotion_intensity": 3, '
            '"problem_type": "Academic_Pressure", '
            '"current_coping_mech": "Sequential", '
            '"coping_shade_signal": "ab main kya karoon", '
            '"user_receptiveness": "high", "is_new_problem": false, '
            '"stigma_cue": false, "risk_signal": null, '
            '"concrete_facts": ["results aane wale", "neend nahi aati"]}'
        ),
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_analyzer_prompt(
    new_seeker_text: str,
    conversation_history: list[TurnRecord],
    previous_analyzer_state: Optional[AnalyzerState],
) -> list[dict]:
    """Build the complete chat-completions messages array for the Analyzer.

    Layout (in order):
      1. system   — role + extraction rules + canonical schema enums
      2. user/asst x4 — static few-shot exemplars (one per coping mechanism)
      3. user     — actual call: history + previous state + new seeker text
    """
    messages: list[dict] = [{"role": "system", "content": _ANALYZER_SYSTEM_MESSAGE}]

    for ex in _FEW_SHOTS:
        messages.append({"role": "user",      "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})

    history_text = format_history(conversation_history)
    if previous_analyzer_state is not None:
        prev_state_text = previous_analyzer_state.model_dump_json(indent=2)
    else:
        prev_state_text = "None — this is the first turn."

    final_user = (
        f"CONVERSATION HISTORY (last {len(conversation_history)} turns):\n"
        f"{history_text}\n\n"
        f"PREVIOUS STATE: {prev_state_text}\n\n"
        f"NEW SEEKER MESSAGE:\n"
        f"\"{new_seeker_text}\"\n\n"
        "Extract the state. Return JSON only."
    )
    messages.append({"role": "user", "content": final_user})
    return messages


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    # Empty-history smoke
    msgs = build_analyzer_prompt(
        new_seeker_text=(
            "Aaj phir se sab gadbad ho gaya. Ghar wale baat nahi kar rahe, "
            "office mein boss naraz hai, neend bhi nahi aati raat ko. Bas "
            "machine ki tarah chal raha hoon ki ruk gaya toh sab gir jayega."
        ),
        conversation_history=[],
        previous_analyzer_state=None,
    )

    print("=" * 78)
    print("ANALYZER PROMPT — empty history")
    print("=" * 78)
    print(f"Total messages: {len(msgs)}  "
          f"(1 system + {(len(msgs) - 2)} few-shot turns + 1 final user)")
    print(f"System length:  {len(msgs[0]['content']):,} chars")
    print(f"Final user msg: {len(msgs[-1]['content']):,} chars")
    print()

    # Verify schema enum injection actually happened
    sys_msg = msgs[0]["content"]
    assert "Academic_Pressure" in sys_msg, "C1 problem missing"
    assert "Marriage_Rishta"   in sys_msg, "C3 problem missing"
    assert "Migration_Displacement" in sys_msg, "C8 problem missing"
    assert "Duty_Based"        in sys_msg, "coping missing"
    assert "Sequential"        in sys_msg, "coping missing"
    assert "exhaustion"        in sys_msg, "emotion missing"
    print("Schema enum injection verified ✓")

    # Verify few-shot assistant outputs all parse as valid AnalyzerState
    print("\nValidating few-shot assistant outputs against AnalyzerState...")
    for i, ex in enumerate(_FEW_SHOTS, 1):
        payload = json.loads(ex["assistant"])
        state = AnalyzerState(**payload)
        print(
            f"  shot {i}: {state.problem_type:<24} "
            f"{state.current_coping_mech:<24} "
            f"emotion={state.emotion_type:<11} "
            f"int={state.emotion_intensity} "
            f"risk={'YES' if state.risk_signal else 'no'}"
        )

    # With history + previous state
    prev = AnalyzerState(
        emotion_type="overwhelm",
        emotion_intensity=4,
        problem_type="Employment_Livelihood",
        current_coping_mech="Duty_Based",
        coping_shade_signal="kaam bahut zyada",
        user_receptiveness="medium",
        is_new_problem=False,
        stigma_cue=False,
        risk_signal=None,
    )
    history = [
        TurnRecord(turn_id=1, speaker="Seeker",
                   text="Kaam bahut zyada ho gaya hai.",
                   emotion="overwhelm", intensity=4),
        TurnRecord(turn_id=2, speaker="Supporter",
                   text="Itna kaam ek saath — lagta hai bahut bhaari pad raha hai.",
                   strategy="RESTATEMENT_OR_PARAPHRASING", phase="Exploration"),
    ]
    msgs2 = build_analyzer_prompt(
        new_seeker_text=(
            "Sir mein dard hai, neend nahi aa rahi. Doctor ke paas jaane ka "
            "time bhi nahi."
        ),
        conversation_history=history,
        previous_analyzer_state=prev,
    )
    print(f"\nWith history+prev: {len(msgs2)} messages, "
          f"final-user len = {len(msgs2[-1]['content'])}")

    print("\nanalyzer_prompt.py — all checks passed ✓")
