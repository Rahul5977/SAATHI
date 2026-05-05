"""Wires safety, analyzer, phase gate, generator, session store, and memory per turn."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, Optional

from agents.analyzer import Analyzer
from agents.generator import Generator
from agents.safety import CRISIS_RESPONSE, SafetyChecker
from agents.summarizer import Summarizer
from config import (
    SAATHI_SUMMARY_EVERY_N_TURNS,
    SAATHI_SUMMARY_HISTORY_TRIGGER,
    SAATHI_SUMMARY_INCREMENTAL_WINDOW,
)
from core.phase_gate import compute_full_strategy, explain_phase_decision
from core.schemas import SessionState, StrategyDecision, UserProfile
from pipeline.memory import MemoryManager
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
        self.summarizer = Summarizer()
        self.session_manager = SessionManager()
        self.memory_manager = MemoryManager()

    # Internal helpers
    async def _hydrate_new_session(
        self, session: SessionState
    ) -> tuple[SessionState, UserProfile]:
        """Brand-new session — pull the user's cross-session profile and
        seed the session with continuity hints. Always returns a profile
        (creating it if absent). Mutates `session` in place + persists it.
        """
        profile = await self.memory_manager.get_or_create(session.user_id)
        await self.memory_manager.register_session_start(profile)

        # Snapshot the profile onto the session so the Generator prompt can
        # reach it without an extra Redis call per turn.
        session.user_profile_snapshot = profile

        # Seed facts_log with the user's persistent life facts so even the
        # FIRST turn of session N+1 can reference "JEE Advanced ke baad ka
        # plan" without waiting for the Analyzer to re-extract them.
        if profile.key_life_facts:
            seed = list(profile.key_life_facts)
            seen = {f.lower() for f in session.facts_log}
            for f in seed:
                if f.lower() not in seen:
                    session.facts_log.append(f)
                    seen.add(f.lower())

        # Persona continuity — adopt the locked persona from prior sessions
        # if we haven't decided one yet for this fresh session.
        if profile.persona_code and session.persona_code == "P0":
            session.persona_code = profile.persona_code

        await self.session_manager.save_session(session)
        return session, profile

    async def _maybe_summarize(
        self,
        session: SessionState,
        profile: Optional[UserProfile],
    ) -> None:
        """Decide whether the Summarizer should run this turn and, if so,
        run it and fold the result back into `session`. Best-effort: any
        failure is logged but never raises.

        Cadence:
          - Run every `SAATHI_SUMMARY_EVERY_N_TURNS` turns starting at turn 4.
          - OR run if the rolling history exceeds
            `SAATHI_SUMMARY_HISTORY_TRIGGER` AND we haven't summarized in
            the last 2 turns.
          - Skip turn 1 entirely — there's nothing meaningful to summarize.
        """
        if SAATHI_SUMMARY_EVERY_N_TURNS <= 0:
            return  # disabled
        if session.turn_count < 2:
            return

        last_summary_turn = (
            session.summary.generated_at_turn if session.summary else 0
        )
        turns_since = session.turn_count - last_summary_turn
        cadence_due = (
            session.turn_count >= SAATHI_SUMMARY_EVERY_N_TURNS
            and turns_since >= SAATHI_SUMMARY_EVERY_N_TURNS
        )
        history_overflow = (
            len(session.turn_history) >= SAATHI_SUMMARY_HISTORY_TRIGGER
            and turns_since >= 2
        )
        if not (cadence_due or history_overflow):
            return

        try:
            # Incremental when we already have a summary; full re-summarize
            # otherwise (cheaper than full every time on long sessions).
            if session.summary is not None:
                window = SAATHI_SUMMARY_INCREMENTAL_WINDOW
                turns_since_last = session.turn_history[-window:]
                summary = await self.summarizer.summarize(
                    session=session,
                    profile=profile,
                    turns_since_last_summary=turns_since_last,
                )
            else:
                summary = await self.summarizer.summarize(
                    session=session,
                    profile=profile,
                    turns_since_last_summary=None,
                )

            session.summary = summary
            await self.session_manager.save_session(session)
            logger.info(
                "Summarizer fired at turn=%s (incremental=%s, facts=%s)",
                session.turn_count,
                session.summary is not None and last_summary_turn > 0,
                len(summary.key_facts),
            )

            # Fold the freshly extracted key_facts into the user's persistent
            # profile so they survive across sessions.
            if profile is not None and summary.key_facts:
                await self.memory_manager.merge_session_facts(
                    profile, summary.key_facts,
                )
        except Exception as e:
            logger.error(
                "Summarizer pass failed at turn=%s: %s",
                session.turn_count, e, exc_info=True,
            )

    # Main entry point
    async def run(
        self,
        session_id: str,
        user_id: str,
        seeker_text: str,
    ) -> AsyncGenerator[str, None]:
        """Run a single seeker turn end-to-end. Yields response tokens as
        they arrive (or one full crisis response in the safety-trip case)."""

        # ---- 1. Load / create session (hydrate cross-session profile on creation) ----
        existing = await self.session_manager.get_session(session_id)
        if existing is None:
            session = await self.session_manager.create_session(session_id, user_id)
            session, profile = await self._hydrate_new_session(session)
        else:
            session = existing
            # Ongoing session — load the (possibly-updated) profile but do NOT
            # bump session counters or reseed facts_log. We just want the
            # current snapshot in case the bot already handled a partial turn.
            profile = await self.memory_manager.get(session.user_id)
            if profile is None:
                # Race condition or stale state — recreate and snapshot.
                profile = await self.memory_manager.get_or_create(session.user_id)
                session.user_profile_snapshot = profile

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
            # No Generator pass — clear stale retrieval debug from prior turns.
            session.latest_retrieval_debug = []
            session.latest_retrieval_query = None
            session.latest_retrieval_filter_level = None
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

        # ---- 7. Cross-session profile update (best-effort) ----
        # Fold this turn's concrete facts straight into the long-term profile
        # so even if the user ghosts mid-session, hard facts persist.
        try:
            if profile is not None and analyzer_state.concrete_facts:
                await self.memory_manager.merge_session_facts(
                    profile, analyzer_state.concrete_facts,
                )
            # Also tag the recurring theme by problem_type so we can answer
            # "is this a returning topic for them?" later.
            if profile is not None:
                await self.memory_manager.update_recurring_themes(
                    profile, [analyzer_state.problem_type],
                )
        except Exception as e:
            logger.error(
                "Profile fact-merge crashed for user=%s: %s",
                user_id, e, exc_info=True,
            )

        # ---- 8. Maybe summarize (in-thread; small model, ~1s) ----
        await self._maybe_summarize(session, profile)

        logger.info(
            "Turn %s complete on session=%s | response_len=%s chars",
            session.turn_count, session_id, len(full_response),
        )

    # Lifecycle
    async def close_session(self, session_id: str) -> None:
        """Roll a session into the user's long-term profile and unlink it.

        The API layer SHOULD call this on explicit session-end (e.g. user
        clicks "End conversation" in the UI). It's safe to omit; profiles
        will still be incrementally updated each turn — this just gives a
        clean point to fold the final SessionSummary into the profile.
        """
        sess = await self.session_manager.get_session(session_id)
        if sess is None:
            return
        profile = await self.memory_manager.get(sess.user_id)
        if profile is None:
            return
        # Make absolutely sure we have a current summary before folding.
        if sess.summary is None and sess.turn_count > 0:
            await self._maybe_summarize(sess, profile)
            sess = await self.session_manager.get_session(session_id) or sess
        await self.memory_manager.apply_session_close(profile, sess)
        logger.info(
            "Closed session=%s | folded %s turns into user=%s profile",
            session_id, sess.turn_count, sess.user_id,
        )

    async def close(self) -> None:
        """Release all Redis connections. Call on application shutdown."""
        await self.session_manager.close()
        await self.memory_manager.close()
