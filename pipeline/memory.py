"""
Cross-session memory layer.

`MemoryManager` is the persistent, user-keyed counterpart to `SessionManager`.
Where SessionManager tracks one conversation, MemoryManager tracks the
*user* — facts, themes, summaries — across every conversation they've ever
had with SAATHI.

Storage:
  - Redis key `user_profile:{user_id}` (string, JSON-serialized UserProfile).
  - 90-day TTL refreshed on every write. (We don't want indefinite retention
    — a stale 8-month-old profile is more harmful than helpful.)
  - In-memory fallback when Redis is unavailable (mirrors `SessionManager`).

API surface kept narrow on purpose:
  - `get_or_create(user_id)`           — hydrate at session start
  - `save(profile)`                    — persist after any update
  - `register_session_start(profile)`  — bump counters, refresh `last_seen_at`
  - `apply_session_close(...)`         — fold a finished session's summary
                                         into the long-term profile
  - `merge_session_facts(profile,
                         new_facts)`   — incremental fact accumulation
                                         (called every turn from orchestrator)

This module has zero LLM dependencies. The decision of WHAT to remember is
made by the Summarizer agent (`agents/summarizer.py`); MemoryManager just
holds the bytes.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import redis.asyncio as redis

from config import REDIS_URL
from core.schemas import SessionState, SessionSummary, UserProfile


logger = logging.getLogger(__name__)


# 90 days. Long enough that returning users feel remembered, short enough
# that abandoned profiles eventually evict.
_PROFILE_TTL_SECONDS = 90 * 24 * 60 * 60

# Hard caps applied at write time so a runaway summarizer can't grow the
# profile blob unboundedly.
_MAX_RECURRING_THEMES = 20
_MAX_KEY_LIFE_FACTS = 30


class MemoryManager:
    """Redis-backed cross-session profile store with in-memory fallback."""

    def __init__(self) -> None:
        self._redis: Optional[redis.Redis] = None
        self._memory_store: dict[str, str] = {}
        self._use_redis: bool = True
        self.ttl: int = _PROFILE_TTL_SECONDS

    # ------------------------------------------------------------------
    # Internal: lazy Redis connection (mirrors SessionManager)
    # ------------------------------------------------------------------
    async def _get_redis(self) -> Optional[redis.Redis]:
        if not self._use_redis:
            return None
        if self._redis is None:
            try:
                self._redis = redis.from_url(REDIS_URL, decode_responses=True)
                await self._redis.ping()
                logger.info(
                    "MemoryManager connected to Redis at %s", REDIS_URL,
                )
            except Exception as e:
                logger.warning(
                    "MemoryManager: Redis not available (%s) — using in-memory store",
                    e,
                )
                self._use_redis = False
                if self._redis is not None:
                    try:
                        await self._redis.aclose()
                    except Exception:
                        pass
                self._redis = None
        return self._redis

    @staticmethod
    def _key(user_id: str) -> str:
        return f"user_profile:{user_id}"

    # ------------------------------------------------------------------
    # Public API: load / save
    # ------------------------------------------------------------------
    async def get(self, user_id: str) -> Optional[UserProfile]:
        """Fetch an existing profile. Returns None on miss or corruption."""
        key = self._key(user_id)
        r = await self._get_redis()
        data = await r.get(key) if r is not None else self._memory_store.get(key)
        if not data:
            return None
        try:
            return UserProfile.model_validate_json(data)
        except Exception as e:
            logger.error(
                "MemoryManager: corrupt profile for %s (%s) — discarding",
                user_id, e,
            )
            return None

    async def save(self, profile: UserProfile) -> None:
        """Persist a profile (with TTL refresh)."""
        # Apply hard caps defensively. Validators on `UserProfile` already
        # cap on construction, but a long-running process that mutates lists
        # in place could otherwise grow them past the limits before save.
        profile.recurring_themes = profile.recurring_themes[-_MAX_RECURRING_THEMES:]
        profile.key_life_facts = profile.key_life_facts[-_MAX_KEY_LIFE_FACTS:]

        data = profile.model_dump_json()
        key = self._key(profile.user_id)
        r = await self._get_redis()
        if r is not None:
            await r.setex(key, self.ttl, data)
        else:
            self._memory_store[key] = data

    async def get_or_create(self, user_id: str) -> UserProfile:
        """Return existing profile or create a fresh one. The fresh profile
        is NOT saved here — caller is expected to call `save` after any
        meaningful mutation (or `register_session_start`)."""
        existing = await self.get(user_id)
        if existing is not None:
            return existing
        logger.info("MemoryManager: creating new profile for user=%s", user_id)
        fresh = UserProfile(user_id=user_id)
        fresh.touch()
        return fresh

    # ------------------------------------------------------------------
    # Public API: lifecycle hooks called from the orchestrator
    # ------------------------------------------------------------------
    async def register_session_start(self, profile: UserProfile) -> UserProfile:
        """Called when a new session opens for this user. Bumps the session
        counter, refreshes `last_seen_at`, persists. Idempotent enough — if
        the same session_id reopens after a transient socket close, this
        will double-count, but that's a minor bookkeeping cost."""
        profile.sessions_count += 1
        profile.touch()
        await self.save(profile)
        return profile

    async def apply_session_close(
        self,
        profile: UserProfile,
        finished_session: SessionState,
    ) -> UserProfile:
        """Fold a completed session into the long-term profile.

        Specifically:
          - `last_session_summary` <- finished_session.summary.narrative
          - `last_session_goal`    <- finished_session.summary.seeker_goal
          - `key_life_facts`       <- merged with summary.key_facts (capped + deduped)
          - `total_turns`          <- bumped by finished_session.turn_count
          - `last_seen_at`         <- now

        Idempotent on a per-session basis: if you call this twice for the
        same session, totals will inflate but text fields will simply
        re-overwrite themselves. (The orchestrator calls this exactly once
        per session close, so in practice this is fine.)
        """
        summary = finished_session.summary
        if summary is not None:
            if summary.narrative:
                profile.last_session_summary = summary.narrative
            if summary.seeker_goal:
                profile.last_session_goal = summary.seeker_goal
            if summary.key_facts:
                profile.key_life_facts = _dedupe_extend(
                    profile.key_life_facts, summary.key_facts,
                    cap=_MAX_KEY_LIFE_FACTS,
                )
        profile.total_turns += int(finished_session.turn_count or 0)
        profile.touch()
        await self.save(profile)
        return profile

    async def merge_session_facts(
        self,
        profile: UserProfile,
        new_facts: Iterable[str],
    ) -> UserProfile:
        """Incrementally fold session-level concrete facts into the profile.

        Called from the orchestrator after each turn so even if the user
        ghosts the session mid-conversation, their volatile facts (exam
        date, name, hometown) are still persisted. Saves only if at least
        one new fact was actually added — avoids a Redis round-trip per turn
        when the seeker hasn't said anything new.
        """
        before = len(profile.key_life_facts)
        profile.key_life_facts = _dedupe_extend(
            profile.key_life_facts, list(new_facts),
            cap=_MAX_KEY_LIFE_FACTS,
        )
        if len(profile.key_life_facts) != before:
            profile.touch()
            await self.save(profile)
        return profile

    async def update_recurring_themes(
        self,
        profile: UserProfile,
        new_themes: Iterable[str],
    ) -> UserProfile:
        """Add new recurring-theme tags (e.g. problem_type or analyst-derived
        labels) to the profile. Auto-saves if anything was actually added.
        """
        before = len(profile.recurring_themes)
        profile.recurring_themes = _dedupe_extend(
            profile.recurring_themes, list(new_themes),
            cap=_MAX_RECURRING_THEMES,
        )
        if len(profile.recurring_themes) != before:
            profile.touch()
            await self.save(profile)
        return profile

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def close(self) -> None:
        """Release the Redis connection. Safe to call multiple times."""
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception as e:
                logger.warning("MemoryManager: error closing Redis client: %s", e)
            finally:
                self._redis = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dedupe_extend(
    existing: list[str],
    incoming: list[str],
    cap: int,
) -> list[str]:
    """Append `incoming` to `existing` (case-insensitive de-dup) and trim
    from the FRONT to `cap`. Most-recent items win when we hit the cap."""
    seen: set[str] = {x.strip().lower() for x in existing if x and x.strip()}
    out: list[str] = list(existing)
    for item in incoming or []:
        if not item:
            continue
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    if len(out) > cap:
        out = out[-cap:]
    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio

    async def _smoke():
        # Force in-memory fallback by pointing at a definitely-down Redis.
        mm = MemoryManager()
        mm._use_redis = False  # noqa: SLF001 — test internals, not prod path

        # Fresh profile
        prof = await mm.get_or_create("user_test_001")
        assert prof.user_id == "user_test_001"
        assert prof.sessions_count == 0
        await mm.register_session_start(prof)
        assert prof.sessions_count == 1
        print("create + register_session_start: OK")

        # Round-trip through the in-memory store
        loaded = await mm.get("user_test_001")
        assert loaded is not None and loaded.sessions_count == 1
        print("load round-trip: OK")

        # Incremental fact merge
        await mm.merge_session_facts(loaded, ["JEE Advanced in 1 week", "from Kota"])
        await mm.merge_session_facts(loaded, ["JEE Advanced in 1 week", "papa retired last month"])
        assert "JEE Advanced in 1 week" in loaded.key_life_facts
        assert "papa retired last month" in loaded.key_life_facts
        # Dedup check — only 3 unique entries despite 4 inserts.
        assert len(loaded.key_life_facts) == 3
        print("merge_session_facts (dedup + persist): OK")

        # Theme accumulation
        await mm.update_recurring_themes(loaded, ["academic stress", "academic stress", "family pressure"])
        assert loaded.recurring_themes == ["academic stress", "family pressure"]
        print("update_recurring_themes: OK")

        # Session close — fold summary in
        sess = SessionState(session_id="s1", user_id="user_test_001", turn_count=8)
        sess.summary = SessionSummary(
            narrative="Seeker prepping for JEE Adv in 1 week. Tried breathing pause, helped slightly.",
            seeker_goal="Calm down enough to study maths revision",
            key_facts=["JEE Advanced in 1 week", "studying in Kota", "papa retired last month"],
            emotional_arc="panicked → settling",
            phase_journey="Exploration→Insight→Action by turn 5",
            generated_at_turn=5,
        )
        await mm.apply_session_close(loaded, sess)
        assert "JEE" in (loaded.last_session_summary or "")
        assert "Calm down" in (loaded.last_session_goal or "")
        assert "studying in Kota" in loaded.key_life_facts
        assert loaded.total_turns == 8
        print("apply_session_close: OK")

        # Cap enforcement
        for i in range(50):
            await mm.merge_session_facts(loaded, [f"fact_{i}"])
        assert len(loaded.key_life_facts) == _MAX_KEY_LIFE_FACTS
        print(f"cap enforcement (key_life_facts <= {_MAX_KEY_LIFE_FACTS}): OK")

        await mm.close()

    asyncio.run(_smoke())
    print("\npipeline/memory.py — all checks passed ✓")
