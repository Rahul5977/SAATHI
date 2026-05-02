"""
Redis-backed session persistence with in-memory fallback.

Each session is stored as a JSON-serialized `SessionState` under the key
``session:{session_id}`` with a 24h TTL. If Redis cannot be reached on the
first call, we transparently degrade to a per-process in-memory dict so dev
loops, tests, and demo runs work without a Redis server.

Concurrency note: a single SessionManager instance is safe to share across
async tasks within one process. The Redis client is built lazily on first
use to avoid event-loop-binding issues.
"""

from __future__ import annotations

import logging
from typing import Optional

import redis.asyncio as redis

from config import REDIS_URL
from core.schemas import (
    AnalyzerState,
    SafetyFlags,
    SessionState,
    StrategyDecision,
    TurnRecord,
)


logger = logging.getLogger(__name__)


# Hard cap on stored conversation history (8 exchanges = 16 turns). Older
# turns are summarized via the rolling AnalyzerState carried on SessionState.
_HISTORY_CAP = 16

# Session TTL in Redis: 24 hours of inactivity = fresh start next time.
_SESSION_TTL_SECONDS = 86_400


class SessionManager:
    """Redis-backed session store with in-memory fallback."""

    def __init__(self) -> None:
        self._redis: Optional[redis.Redis] = None
        self._memory_store: dict[str, str] = {}
        self._use_redis: bool = True
        self.ttl: int = _SESSION_TTL_SECONDS

    # ------------------------------------------------------------------
    # Internal: lazy Redis connection with one-shot fallback flip
    # ------------------------------------------------------------------
    async def _get_redis(self) -> Optional[redis.Redis]:
        if not self._use_redis:
            return None
        if self._redis is None:
            try:
                self._redis = redis.from_url(REDIS_URL, decode_responses=True)
                await self._redis.ping()
                logger.info("Connected to Redis at %s", REDIS_URL)
            except Exception as e:
                logger.warning(
                    "Redis not available (%s) — falling back to in-memory store",
                    e,
                )
                self._use_redis = False
                # Best-effort close on the half-built client.
                if self._redis is not None:
                    try:
                        await self._redis.aclose()
                    except Exception:
                        pass
                self._redis = None
        return self._redis

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """Fetch an existing session or return None."""
        key = self._key(session_id)
        r = await self._get_redis()
        if r is not None:
            data = await r.get(key)
        else:
            data = self._memory_store.get(key)

        if not data:
            return None
        try:
            return SessionState.model_validate_json(data)
        except Exception as e:
            # Schema drift or corrupted blob — treat as no session so the
            # caller will create a fresh one. Don't crash the request.
            logger.error(
                "Corrupt session payload for %s (%s) — discarding",
                session_id, e,
            )
            return None

    async def save_session(self, session: SessionState) -> None:
        """Persist a session (with TTL refresh)."""
        data = session.model_dump_json()
        key = self._key(session.session_id)
        r = await self._get_redis()
        if r is not None:
            await r.setex(key, self.ttl, data)
        else:
            self._memory_store[key] = data

    async def create_session(
        self, session_id: str, user_id: str
    ) -> SessionState:
        """Create + persist a new SessionState."""
        session = SessionState(session_id=session_id, user_id=user_id)
        await self.save_session(session)
        logger.info("Created new session: %s (user=%s)", session_id, user_id)
        return session

    async def get_or_create(
        self, session_id: str, user_id: str
    ) -> SessionState:
        """Convenience: return existing session or create a new one."""
        existing = await self.get_session(session_id)
        if existing is not None:
            return existing
        return await self.create_session(session_id, user_id)

    async def update_after_turn(
        self,
        session: SessionState,
        seeker_text: str,
        supporter_text: str,
        analyzer_state: AnalyzerState,
        strategy_decision: StrategyDecision,
        safety_flags: SafetyFlags,
        phase_decision_reason: Optional[str] = None,
    ) -> SessionState:
        """Apply a completed turn (seeker → SAATHI) to `session` and persist.
        Mutates `session` in place AND returns it for chaining.

        `phase_decision_reason` is a human-readable string from
        `core.phase_gate.explain_phase_decision` — purely diagnostic, surfaced
        to the dev UI but never read by any agent.
        """
        session.turn_count += 1

        # ---- trajectories (append) ----
        session.phase_history.append(strategy_decision.current_phase)
        session.strategy_history.append(strategy_decision.selected_strategy)
        session.intensity_trajectory.append(analyzer_state.emotion_intensity)
        session.coping_trajectory.append(analyzer_state.current_coping_mech)

        # ---- turn history (append both turns, then cap) ----
        seeker_turn = TurnRecord(
            turn_id=session.turn_count * 2 - 1,
            speaker="Seeker",
            text=seeker_text,
            emotion=analyzer_state.emotion_type,
            intensity=analyzer_state.emotion_intensity,
        )
        supporter_turn = TurnRecord(
            turn_id=session.turn_count * 2,
            speaker="Supporter",
            text=supporter_text,
            strategy=strategy_decision.selected_strategy,
            phase=strategy_decision.current_phase,
        )
        session.turn_history.append(seeker_turn)
        session.turn_history.append(supporter_turn)
        if len(session.turn_history) > _HISTORY_CAP:
            session.turn_history = session.turn_history[-_HISTORY_CAP:]

        # ---- snapshot the latest agent outputs ----
        session.latest_analyzer_state = analyzer_state
        session.latest_strategy_decision = strategy_decision
        session.latest_safety_flags = safety_flags
        if phase_decision_reason is not None:
            session.latest_phase_decision_reason = phase_decision_reason

        # ---- one-shot triggers ----
        if strategy_decision.selected_strategy == "SELF_DISCLOSURE":
            session.self_disclosure_used = True
            session.self_disclosure_turn = session.turn_count
        if safety_flags.requires_hitl:
            session.hitl_escalated = True

        await self.save_session(session)
        return session

    async def close(self) -> None:
        """Release the Redis connection. Safe to call multiple times."""
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception as e:
                logger.warning("Error closing Redis client: %s", e)
            finally:
                self._redis = None
