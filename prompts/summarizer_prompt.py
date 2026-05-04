"""
Summarizer (Agent 4) prompt builder.

The Summarizer's job is structured compression: read the conversation so far
+ the previous summary (if any) and emit a fresh `SessionSummary` JSON.

It is called every N turns (or when history exceeds a soft cap) so that the
Generator and phase_gate can reason about turns 1..K even after they fall
out of the 16-turn rolling window.

Like the Analyzer, this module emits NO seeker-facing text. Pure structured
extraction.

Why structured (vs free-form running summary)?
  - The Generator prompt template needs deterministic slots
    (`seeker_goal`, `key_facts`, `emotional_arc`) — free-text would force
    extra parsing.
  - Structured fields make it trivial for the eval harness to assert
    "the summary remembered fact X by turn N".
  - Field-level caps prevent runaway summaries from blowing the prompt
    budget.
"""

from __future__ import annotations

from typing import Optional

from core.schemas import (
    AnalyzerState,
    SessionSummary,
    SessionState,
    TurnRecord,
    UserProfile,
)
from prompts.analyzer_prompt import format_history


# ---------------------------------------------------------------------------
# System prompt — composed at import time
# ---------------------------------------------------------------------------
SUMMARIZER_SYSTEM_PROMPT = """\
You are the Summarizer module of SAATHI. Your ONLY job is structured compression of an ongoing conversation.

You do NOT generate any reply for the seeker. You do NOT advise or analyze emotions in real-time (the Analyzer does that). You read everything that has happened so far and produce a JSON `SessionSummary` that lets the rest of the system reason about turns that have fallen out of the rolling history window.

Think of yourself as the "memory" of a friend who zoned out for a minute and now wants a 30-second catch-up. Be FACTUAL. Be CONCRETE. Avoid abstract psychology language.

OUTPUT FORMAT — Return ONLY a valid JSON object with these EXACT fields:
{
    "narrative":         "<= 80 words, past tense, third-person about the seeker. Captures the arc.>",
    "seeker_goal":       "<= 20 words OR null. The single most pressing thing the seeker is trying to resolve. e.g. 'Calm down enough to study maths revision in 1 week before JEE Advanced.'>",
    "key_facts":         ["<= 12 items, each <= 12 words. Hard, persistent details: exam dates, places, family members, medications. NEVER feelings or interpretations.>"],
    "emotional_arc":     "<one phrase, <= 12 words. e.g. 'panicked → settling' or 'ashamed and closed → opening up cautiously'>",
    "phase_journey":     "<one phrase, <= 12 words. e.g. 'Exploration→Insight by turn 4' or 'still in Exploration, intensity dropping'>",
    "open_threads":      ["<= 5 items. Things the bot has opened (questions, suggestions) that the seeker hasn't yet responded to. Empty list if none.>"],
    "generated_at_turn": <integer — the turn count at which this summary was generated>
}

HARD RULES (violations cause failure):
1. Output JSON ONLY. No markdown fences, no commentary, no explanation.
2. `narrative` is past-tense and third-person. Never "I" or "you".
3. `key_facts` are NOUN PHRASES, never sentences. e.g. "JEE Advanced in 1 week" YES, "The seeker is preparing for JEE Advanced and the exam is in one week" NO.
4. NEVER include the seeker's literal verbatim emotional outpouring in `key_facts`. Only hard facts: dates, places, names, numbers, medications, jobs, exam IDs.
5. NEVER project emotions the seeker hasn't actually expressed. If they said "thoda dar lag raha hai" the arc is "mild fear", not "panicked despair".
6. If a previous summary is provided, BUILD ON IT — don't drop facts that are still relevant. Append/refine. Only remove facts the seeker has explicitly contradicted.
7. `open_threads` should be empty if every supporter turn so far has been pure validation — only fill it when the bot has actually asked something or offered something concrete.
8. If the seeker has expressed a clear ASK ("kya karu?", "solution batao", "tum batao na"), the `seeker_goal` MUST capture what they're asking for help with.
"""


# ---------------------------------------------------------------------------
# Few-shot examples — keep tiny; this agent is mechanical
# ---------------------------------------------------------------------------
_FEW_SHOTS: list[dict] = [
    {
        "role": "user",
        "content": (
            "PREVIOUS SUMMARY: (none — first summarization)\n\n"
            "CONVERSATION SO FAR:\n"
            "Seeker: Sab theek hai, bas kuch dino me JEE advanced hai, padhne ka mann ni ho rha hai or thoda thoda dar bhi lag raha hai.\n"
            "Supporter: JEE advanced ke pehle aise lagna, waqai challenging hai. Padhne ka mann nahi lag raha, aur andar se thoda darr bhi hai — bilkul samajh sakta hoon.\n"
            "Seeker: 1 hafte me paper hai yaar, break kaisi lunga\n"
            "Supporter: Bhai, JEE advanced sir pe pehle break lena toh mushkil lagta hai. Lagta hai andar se pressure aur ghabrahat mile ek saath baithe hain.\n"
            "Seeker: Ha aisa hi lagta hai...par krna kya h ye ni samjh ata mujhe..karu kya 1 hafte abhi mai\n\n"
            "LATEST ANALYZER STATE:\n"
            "{ \"emotion_type\": \"confusion\", \"emotion_intensity\": 3, \"problem_type\": \"Academic_Pressure\", \"current_coping_mech\": \"Sequential\", \"user_receptiveness\": \"high\", \"is_new_problem\": false, \"stigma_cue\": false, \"risk_signal\": null, \"concrete_facts\": [\"JEE Advanced in 1 week\"] }\n\n"
            "TURN COUNT: 3"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{"
            "\"narrative\": \"Seeker is preparing for JEE Advanced, exam is in 1 week. They have lost motivation to study and are feeling some fear. They asked the supporter for direction on what to do in the remaining week.\","
            " \"seeker_goal\": \"Figure out what concrete thing to do in the 1 week before JEE Advanced.\","
            " \"key_facts\": [\"JEE Advanced in 1 week\", \"motivation to study has dropped\"],"
            " \"emotional_arc\": \"mild fear and confusion, opening up\","
            " \"phase_journey\": \"Exploration→Insight by turn 2, opened to Action by turn 3\","
            " \"open_threads\": [\"seeker asked for help on what to do this week — supporter has not yet given a concrete suggestion\"],"
            " \"generated_at_turn\": 3"
            "}"
        ),
    },
    {
        "role": "user",
        "content": (
            "PREVIOUS SUMMARY:\n"
            "{\"narrative\": \"Seeker is preparing for JEE Advanced, exam is in 1 week. They have lost motivation to study and are feeling some fear. They asked the supporter for direction.\", \"seeker_goal\": \"Figure out what concrete thing to do in 1 week before JEE Advanced.\", \"key_facts\": [\"JEE Advanced in 1 week\", \"motivation to study has dropped\"], \"emotional_arc\": \"mild fear and confusion, opening up\", \"phase_journey\": \"Exploration→Insight by turn 2, opened to Action by turn 3\", \"open_threads\": [\"seeker asked for help on what to do this week\"], \"generated_at_turn\": 3}\n\n"
            "RECENT TURNS (since last summary):\n"
            "Supporter: Haan yaar, JEE advanced ke pehle aise sochna bilkul normal hai. Ek kaam kar, ek din ke liye sirf maths revise karne ka plan bana le. Karke batana kaisa laga.\n"
            "Seeker: Achi advice hai... note kar liya maine ye, aur kya karna chahiye mujhe mann shant krne k liye\n\n"
            "LATEST ANALYZER STATE:\n"
            "{ \"emotion_type\": \"confusion\", \"emotion_intensity\": 2, \"problem_type\": \"Academic_Pressure\", \"current_coping_mech\": \"Sequential\", \"user_receptiveness\": \"high\", \"is_new_problem\": false, \"stigma_cue\": false, \"risk_signal\": null, \"concrete_facts\": [] }\n\n"
            "TURN COUNT: 4"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{"
            "\"narrative\": \"Seeker is preparing for JEE Advanced in 1 week. They had lost motivation but are now opening up. The supporter suggested a 1-day maths revision plan; seeker accepted it and is now asking for grounding techniques to calm their mind.\","
            " \"seeker_goal\": \"Calm down enough to follow through on a 1-day maths revision plan before JEE Advanced.\","
            " \"key_facts\": [\"JEE Advanced in 1 week\", \"motivation to study has dropped\", \"agreed to 1-day maths revision plan\"],"
            " \"emotional_arc\": \"mild fear → settling, more receptive\","
            " \"phase_journey\": \"Exploration→Insight→Action by turn 3, settled in Action\","
            " \"open_threads\": [\"supporter suggested 1-day maths revision plan — awaiting seeker follow-up next session\", \"seeker asked for additional calming techniques\"],"
            " \"generated_at_turn\": 4"
            "}"
        ),
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_summarizer_prompt(
    session: SessionState,
    profile: Optional[UserProfile] = None,
    turns_since_last_summary: Optional[list[TurnRecord]] = None,
) -> list[dict]:
    """Build the chat-completions messages array for the Summarizer LLM.

    Args:
        session: The full session state — `turn_history`, `summary` (if any),
            `latest_analyzer_state`, `phase_history`, `turn_count`.
        profile: Optional cross-session UserProfile. When the conversation
            is fresh (turn ≤ 1) we surface `last_session_summary` so the
            new summary acknowledges continuity.
        turns_since_last_summary: If provided, the model sees only the new
            turns + the previous summary (incremental). If None, the model
            sees the full `turn_history` (full re-summarize).

    Returns:
        A list of `{"role": ..., "content": ...}` dicts.
    """
    prev_summary_blob: str
    if session.summary is not None:
        # Compact the previous summary to a single JSON line so the
        # incremental flow is unambiguous.
        prev_summary_blob = session.summary.model_dump_json()
    elif profile is not None and profile.last_session_summary:
        prev_summary_blob = (
            f"(no in-session summary yet — but the user's LAST SESSION ended with: "
            f"\"{profile.last_session_summary}\")"
        )
    else:
        prev_summary_blob = "(none — first summarization)"

    if turns_since_last_summary is not None:
        section_label = "RECENT TURNS (since last summary):"
        history = turns_since_last_summary
    else:
        section_label = "CONVERSATION SO FAR:"
        history = session.turn_history

    history_text = format_history(history)

    analyzer_blob: str
    if session.latest_analyzer_state is not None:
        # Strip transient narrative-y fields the summarizer doesn't need.
        a = session.latest_analyzer_state
        analyzer_blob = (
            "{"
            f"\"emotion_type\": \"{a.emotion_type}\", "
            f"\"emotion_intensity\": {a.emotion_intensity}, "
            f"\"problem_type\": \"{a.problem_type}\", "
            f"\"current_coping_mech\": \"{a.current_coping_mech}\", "
            f"\"user_receptiveness\": \"{a.user_receptiveness}\", "
            f"\"is_new_problem\": {str(a.is_new_problem).lower()}, "
            f"\"stigma_cue\": {str(a.stigma_cue).lower()}, "
            f"\"risk_signal\": "
            f"{('null' if a.risk_signal is None else repr(a.risk_signal))}, "
            f"\"concrete_facts\": {list(a.concrete_facts)}"
            "}"
        )
    else:
        analyzer_blob = "(no analyzer state available)"

    user_content = (
        f"PREVIOUS SUMMARY:\n{prev_summary_blob}\n\n"
        f"{section_label}\n{history_text}\n\n"
        f"LATEST ANALYZER STATE:\n{analyzer_blob}\n\n"
        f"TURN COUNT: {session.turn_count}"
    )

    messages: list[dict] = [
        {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
        *_FEW_SHOTS,
        {"role": "user", "content": user_content},
    ]
    return messages


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sess = SessionState(session_id="s1", user_id="u1", turn_count=4)
    sess.turn_history = [
        TurnRecord(turn_id=1, speaker="Seeker", text="Sab theek hai, bas JEE advanced 1 hafte me hai", emotion="fear", intensity=3),
        TurnRecord(turn_id=2, speaker="Supporter", text="Samajh sakta hoon, kafi pressure hoga.", strategy="REFLECTION_OF_FEELINGS", phase="Exploration"),
        TurnRecord(turn_id=3, speaker="Seeker", text="Kya karu yaar, kuch samajh ni aa raha", emotion="confusion", intensity=3),
        TurnRecord(turn_id=4, speaker="Supporter", text="Ek din ke liye sirf maths revise kar.", strategy="PROVIDING_SUGGESTIONS", phase="Action"),
    ]
    sess.latest_analyzer_state = AnalyzerState(
        emotion_type="confusion",
        emotion_intensity=3,
        problem_type="Academic_Pressure",
        current_coping_mech="Sequential",
        coping_shade_signal="kuch samajh ni aa raha",
        user_receptiveness="high",
        is_new_problem=False,
        stigma_cue=False,
        risk_signal=None,
        concrete_facts=["JEE Advanced in 1 week"],
    )

    msgs = build_summarizer_prompt(sess)
    assert msgs[0]["role"] == "system"
    assert "Summarizer module" in msgs[0]["content"]
    # System + 4 few-shots + 1 real turn
    assert len(msgs) == 1 + 4 + 1
    last = msgs[-1]["content"]
    assert "JEE advanced" in last
    assert "TURN COUNT: 4" in last
    assert "PREVIOUS SUMMARY:" in last
    print("messages built OK; total =", len(msgs))
    print(f"system_chars = {len(msgs[0]['content'])}")
    print(f"user_chars   = {len(msgs[-1]['content'])}")

    # Incremental flow
    sess.summary = SessionSummary(
        narrative="Seeker preparing for JEE Advanced in 1 week.",
        seeker_goal="Figure out what to do this week.",
        key_facts=["JEE Advanced in 1 week"],
        emotional_arc="mild fear",
        phase_journey="Exploration→Insight",
        generated_at_turn=2,
    )
    msgs2 = build_summarizer_prompt(
        sess,
        turns_since_last_summary=sess.turn_history[-2:],
    )
    last2 = msgs2[-1]["content"]
    assert "RECENT TURNS (since last summary):" in last2
    assert "JEE Advanced in 1 week" in last2  # from prev summary blob
    print("incremental flow OK")

    # Cross-session continuity
    from core.schemas import UserProfile
    sess2 = SessionState(session_id="s2", user_id="u1", turn_count=1)
    sess2.turn_history = [
        TurnRecord(turn_id=1, speaker="Seeker", text="Hi, wapas aaya hoon", emotion="calm", intensity=2),
    ]
    profile = UserProfile(
        user_id="u1",
        last_session_summary="Seeker prepped for JEE Adv in 1 week, tried breathing exercise.",
    )
    msgs3 = build_summarizer_prompt(sess2, profile=profile)
    last3 = msgs3[-1]["content"]
    assert "LAST SESSION ended with" in last3, "cross-session hint missing"
    print("cross-session hint surfaced in PREVIOUS SUMMARY block: OK")

    print("\nprompts/summarizer_prompt.py — all checks passed ✓")
