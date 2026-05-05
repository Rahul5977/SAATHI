"""Periodic session summarization for long-running chats (Agent 4)."""

from __future__ import annotations

import logging
from typing import Optional

from config import SUMMARIZER_TEMPERATURE
from core.schemas import SessionState, SessionSummary, TurnRecord, UserProfile
from llm import get_llm
from prompts.summarizer_prompt import build_summarizer_prompt


logger = logging.getLogger(__name__)


class Summarizer:
    """Stateless wrapper around the Summarizer LLM call."""

    def __init__(self) -> None:
        self.llm = get_llm("summarizer")

    async def summarize(
        self,
        session: SessionState,
        profile: Optional[UserProfile] = None,
        turns_since_last_summary: Optional[list[TurnRecord]] = None,
    ) -> SessionSummary:
        """Produce a fresh `SessionSummary` reflecting `session` + (optionally)
        `profile.last_session_summary` for cross-session continuity.

        `turns_since_last_summary` lets the caller pass an incremental slice
        when an existing `session.summary` is present — the LLM then sees
        only the new turns + the previous summary blob, which is much
        cheaper for long sessions.

        ALWAYS returns a valid SessionSummary. Never raises.
        """
        try:
            messages = build_summarizer_prompt(
                session=session,
                profile=profile,
                turns_since_last_summary=turns_since_last_summary,
            )
            result: SessionSummary = await self.llm.generate_json(
                messages=messages,
                response_schema=SessionSummary,
                temperature=SUMMARIZER_TEMPERATURE,
                max_tokens=600,
            )
            # Stamp the actual turn count we summarized at — the LLM's value
            # is hint-only, the orchestrator owns the truth.
            result.generated_at_turn = session.turn_count

            logger.info(
                "Summarizer: turn=%s | goal=%s | facts=%s | arc=%s",
                session.turn_count,
                (result.seeker_goal or "(none)")[:80],
                len(result.key_facts),
                result.emotional_arc[:60],
            )
            return result

        except Exception as e:
            logger.error(
                "Summarizer failed at turn=%s, falling back: %s",
                session.turn_count, e, exc_info=True,
            )
            return self._safe_default(session, profile)

    # Internals
    @staticmethod
    def _safe_default(
        session: SessionState,
        profile: Optional[UserProfile],
    ) -> SessionSummary:
        """Build a minimal carry-forward summary when the LLM call fails.

        Strategy:
          - If a previous in-session summary exists, return it verbatim with
            an updated `generated_at_turn`. This is strictly better than a
            blank summary because at least the previously-extracted goal
            and key_facts persist.
          - Else if the user has a cross-session profile with a last
            summary, surface it as the narrative so the bot still sounds
            continuous.
          - Else build a stub from `analyzer_state.concrete_facts` +
            `phase_history`.
        """
        if session.summary is not None:
            carry = session.summary.model_copy(deep=True)
            carry.generated_at_turn = session.turn_count
            return carry

        facts: list[str] = list(session.facts_log[-12:])
        if session.latest_analyzer_state is not None:
            for f in session.latest_analyzer_state.concrete_facts or []:
                if f and f not in facts:
                    facts.append(f)

        narrative = ""
        if profile is not None and profile.last_session_summary:
            narrative = (
                f"(continuity from last session) {profile.last_session_summary}"
            )[:780]

        # Phase journey in compact form: "Exploration→Insight→Action"
        seen_phases: list[str] = []
        for p in session.phase_history:
            if not seen_phases or seen_phases[-1] != p:
                seen_phases.append(p)
        phase_journey = "→".join(seen_phases)[:160] if seen_phases else ""

        return SessionSummary(
            narrative=narrative,
            seeker_goal=None,
            key_facts=facts,
            emotional_arc="",
            phase_journey=phase_journey,
            open_threads=[],
            generated_at_turn=session.turn_count,
        )


# Smoke test
if __name__ == "__main__":
    import asyncio

    from core.schemas import AnalyzerState

    async def _smoke():
        sess = SessionState(session_id="s1", user_id="u1", turn_count=4)
        sess.facts_log = ["JEE Advanced in 1 week"]
        sess.phase_history = ["Exploration", "Insight", "Action", "Action"]
        sess.latest_analyzer_state = AnalyzerState(
            emotion_type="confusion", emotion_intensity=2,
            problem_type="Academic_Pressure", current_coping_mech="Sequential",
            coping_shade_signal="kya karu", user_receptiveness="high",
            is_new_problem=False, stigma_cue=False, risk_signal=None,
            concrete_facts=["1-day maths revision plan"],
        )

        # _safe_default exercises ALL branches without an LLM call.
        s_default = Summarizer._safe_default(sess, profile=None)
        assert s_default.generated_at_turn == 4
        assert "JEE Advanced in 1 week" in s_default.key_facts
        assert "1-day maths revision plan" in s_default.key_facts
        assert s_default.phase_journey == "Exploration→Insight→Action"
        print("safe_default (no profile, no prev summary): OK")

        # With profile continuity
        prof = UserProfile(
            user_id="u1",
            last_session_summary="Last session, seeker prepped for JEE Adv.",
        )
        s_default2 = Summarizer._safe_default(sess, profile=prof)
        assert "continuity from last session" in s_default2.narrative
        print("safe_default (with profile continuity): OK")

        # With existing in-session summary -> carry forward
        sess.summary = SessionSummary(
            narrative="prev narrative",
            seeker_goal="prev goal",
            key_facts=["prev fact"],
            emotional_arc="prev arc",
            phase_journey="prev journey",
            generated_at_turn=2,
        )
        s_default3 = Summarizer._safe_default(sess, profile=None)
        assert s_default3.narrative == "prev narrative"
        assert s_default3.generated_at_turn == 4   # bumped to current turn
        print("safe_default (carry forward existing summary): OK")

    asyncio.run(_smoke())
    print("\nagents/summarizer.py — fallback paths OK ✓")
    print("(LLM path requires OPENAI_API_KEY; not exercised here.)")
