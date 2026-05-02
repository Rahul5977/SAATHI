"""
Agent 1 — Analyzer.

Reads (new seeker turn + history + previous AnalyzerState) and returns a
fresh `AnalyzerState`. NO response generation happens here — that's the
Generator's job.

Backend-agnostic: speaks only to `llm.get_llm("analyzer")`. Switching
LLM_BACKEND from "openai" to "local" requires no changes here.

Failure policy:
  - LLM/parse errors NEVER crash the pipeline. We log and return a
    conservative default state so the rest of the turn can proceed.
  - The default uses a moderate-distress profile (intensity=3, receptiveness=
    medium) so phase_gate stays in Exploration and the Generator stays in
    safe territory.
"""

from __future__ import annotations

import logging

from config import ANALYZER_TEMPERATURE
from core.schemas import AnalyzerState, SessionState, TurnRecord
from llm import get_llm
from prompts.analyzer_prompt import build_analyzer_prompt


logger = logging.getLogger(__name__)


class Analyzer:
    """Stateless wrapper around the Analyzer LLM call."""

    def __init__(self) -> None:
        self.llm = get_llm("analyzer")

    async def analyze(
        self,
        new_seeker_text: str,
        conversation_history: list[TurnRecord],
        previous_analyzer_state: AnalyzerState | None,
        session: SessionState,
    ) -> AnalyzerState:
        """Extract structured emotional/behavioral state from the seeker's
        latest message. Always returns a valid AnalyzerState (never raises)."""
        try:
            messages = build_analyzer_prompt(
                new_seeker_text=new_seeker_text,
                conversation_history=conversation_history,
                previous_analyzer_state=previous_analyzer_state,
            )
            result: AnalyzerState = await self.llm.generate_json(
                messages=messages,
                response_schema=AnalyzerState,
                temperature=ANALYZER_TEMPERATURE,
                max_tokens=400,
            )
            risk_str = (
                f"YES: {result.risk_signal}" if result.risk_signal else "none"
            )
            logger.info(
                "Analyzer: emotion=%s, intensity=%s, coping=%s, "
                "shade='%s', risk=%s",
                result.emotion_type,
                result.emotion_intensity,
                result.current_coping_mech,
                (result.coping_shade_signal or "")[:50],
                risk_str,
            )
            return result

        except Exception as e:
            logger.error("Analyzer failed, falling back to safe default: %s",
                         e, exc_info=True)
            return self._safe_default(new_seeker_text, previous_analyzer_state)

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    @staticmethod
    def _safe_default(
        new_seeker_text: str,
        previous: AnalyzerState | None,
    ) -> AnalyzerState:
        """Build a conservative AnalyzerState when the LLM call fails.

        Strategy: prefer carrying forward the previous state's high-signal
        fields (problem_type, current_coping_mech) since those rarely change
        turn-to-turn — only the emotion/intensity/shade are fresh. This keeps
        the conversation coherent across an LLM hiccup.
        """
        problem_type = (
            previous.problem_type if previous else "Academic_Pressure"
        )
        coping_mech = (
            previous.current_coping_mech if previous else "Duty_Based"
        )
        shade = (new_seeker_text or "").strip()[:60] or "(no input)"

        return AnalyzerState(
            emotion_type="overwhelm",
            emotion_intensity=3,
            problem_type=problem_type,
            current_coping_mech=coping_mech,
            coping_shade_signal=shade,
            user_receptiveness="medium",
            is_new_problem=False,
            stigma_cue=False,
            risk_signal=None,
        )
