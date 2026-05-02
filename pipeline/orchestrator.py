"""
Pipeline orchestrator — single entry point per seeker turn.

Per-turn flow:

  1. Load (or create) the SessionState for `session_id`.
  2. Run SafetyChecker.check(...) and Analyzer.analyze(...) IN PARALLEL.
     They share no state, so this is ~free latency reduction.
  3. If the safety checker requires HITL escalation, yield the static
     CRISIS_RESPONSE, persist the turn with a placeholder StrategyDecision,
     and return — DO NOT call the Generator on a crisis turn.
  4. Otherwise, compute (phase, strategy, lens) deterministically via
     `core.phase_gate.compute_full_strategy` (zero LLM calls).
  5. Stream tokens from the Generator, accumulating the full text.
  6. Persist the completed turn to the session store.

This module is the ONLY place that knows how the agents wire together.
The FastAPI / WebSocket layer just calls `orchestrator.run(...)` and
forwards the yielded tokens to the client.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from agents.analyzer import Analyzer
from agents.generator import Generator
from agents.safety import CRISIS_RESPONSE, SafetyChecker
from core.phase_gate import compute_full_strategy, explain_phase_decision
from core.schemas import StrategyDecision
from pipeline.session import SessionManager


logger = logging.getLogger(__name__)


# Static placeholder strategy used to log a crisis turn into session history.
# We persist this so trajectory analytics (`session.strategy_history`) reflect
# that a crisis interrupted normal flow.
_CRISIS_STRATEGY = StrategyDecision(
    current_phase="Exploration",
    selected_strategy="INFORMATION",
    restatement_lens=None,
)


class PipelineOrchestrator:
    """Owns the singleton agent instances and routes a turn end-to-end."""

    def __init__(self) -> None:
        self.analyzer = Analyzer()
        self.generator = Generator()
        self.safety_checker = SafetyChecker()
        self.session_manager = SessionManager()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    async def run(
        self,
        session_id: str,
        user_id: str,
        seeker_text: str,
    ) -> AsyncGenerator[str, None]:
        """Run a single seeker turn end-to-end. Yields response tokens as
        they arrive (or one full crisis response in the safety-trip case)."""

        # ---- 1. Load / create session ----
        session = await self.session_manager.get_or_create(session_id, user_id)

        # ---- 2. Safety + Analyzer in parallel ----
        # `gather` propagates the FIRST exception. Both agents already swallow
        # their own internal failures and return safe defaults, so an unhandled
        # crash here would be a real bug we want to surface — not silence.
        safety_task = asyncio.create_task(
            self.safety_checker.check(seeker_text, session)
        )
        analyzer_task = asyncio.create_task(
            self.analyzer.analyze(
                new_seeker_text=seeker_text,
                conversation_history=session.get_recent_history(6),
                previous_analyzer_state=session.latest_analyzer_state,
                session=session,
            )
        )
        safety_flags, analyzer_state = await asyncio.gather(
            safety_task, analyzer_task
        )

        # ---- 3. Crisis short-circuit ----
        if safety_flags.requires_hitl:
            logger.warning(
                "CRISIS DETECTED on session=%s trigger=%r",
                session_id, safety_flags.trigger_phrase,
            )
            yield CRISIS_RESPONSE

            # Persist the crisis turn so it appears in trajectory + history
            # and so `session.hitl_escalated` flips to True for downstream
            # systems (HITL queue, analytics).
            try:
                await self.session_manager.update_after_turn(
                    session=session,
                    seeker_text=seeker_text,
                    supporter_text=CRISIS_RESPONSE,
                    analyzer_state=analyzer_state,
                    strategy_decision=_CRISIS_STRATEGY,
                    safety_flags=safety_flags,
                    phase_decision_reason="R0 risk_signal active → locked Exploration (CRISIS)",
                )
            except Exception as e:
                logger.error(
                    "Failed to persist crisis turn for session=%s: %s",
                    session_id, e, exc_info=True,
                )
            return

        # ---- 4. Deterministic phase + strategy + lens ----
        # Pass `seeker_text` so phase_gate can detect explicit help-seeking
        # phrases ("solution kya h?", "tum batao na", "kya karu?") and
        # upgrade receptiveness=high accordingly. Without this, the gate
        # stays locked on the Analyzer's distress-biased reading.
        strategy_decision = compute_full_strategy(
            analyzer_state, session, seeker_text=seeker_text
        )
        phase_decision_reason = explain_phase_decision(
            analyzer_state, session, seeker_text=seeker_text
        )

        logger.info(
            "Pipeline session=%s turn=%s | phase=%s strategy=%s lens=%s "
            "(emotion=%s int=%s coping=%s) | %s",
            session_id,
            session.turn_count + 1,
            strategy_decision.current_phase,
            strategy_decision.selected_strategy,
            strategy_decision.restatement_lens,
            analyzer_state.emotion_type,
            analyzer_state.emotion_intensity,
            analyzer_state.current_coping_mech,
            phase_decision_reason,
        )

        # ---- 5. Stream the generator output ----
        full_response_parts: list[str] = []
        try:
            async for token in self.generator.generate_stream(
                seeker_text=seeker_text,
                analyzer_state=analyzer_state,
                strategy_decision=strategy_decision,
                conversation_history=session.get_recent_history(6),
                session=session,
            ):
                full_response_parts.append(token)
                yield token
        except Exception as e:
            # Generator already wraps its own stream in a fallback — this only
            # fires if something completely unexpected happens between us and
            # the generator object itself.
            logger.error(
                "Orchestrator: generator stream crashed: %s", e, exc_info=True,
            )

        full_response = "".join(full_response_parts).strip()

        # ---- 6. Persist completed turn ----
        try:
            await self.session_manager.update_after_turn(
                session=session,
                seeker_text=seeker_text,
                supporter_text=full_response,
                analyzer_state=analyzer_state,
                strategy_decision=strategy_decision,
                safety_flags=safety_flags,
                phase_decision_reason=phase_decision_reason,
            )
        except Exception as e:
            # Don't punish the user with an exception after they've already
            # seen the response. Log loudly and move on.
            logger.error(
                "Failed to persist turn for session=%s: %s",
                session_id, e, exc_info=True,
            )

        logger.info(
            "Turn %s complete on session=%s | response_len=%s chars",
            session.turn_count, session_id, len(full_response),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def close(self) -> None:
        """Release Redis connection. Call on application shutdown."""
        await self.session_manager.close()
