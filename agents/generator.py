"""
Agent 2 — Generator.

Streams the final Hinglish response shown to the seeker.

Pipeline per turn:
  1. Retrieve top-K conversation examples from FAISS (composite-prefix query).
  2. Build the chat-completions messages array via `build_generator_prompt`.
  3. Stream tokens from the LLM, yielding each chunk to the caller.
  4. After streaming completes, run a non-blocking prohibited-vocabulary
     audit. Currently we LOG only — we don't rewrite mid-stream because the
     bytes are already on their way to the user. Regeneration on violation
     is a future enhancement (see TODO below).

Backend-agnostic: speaks only to `llm.get_llm("generator")`.
"""

from __future__ import annotations

import logging
import re
from typing import AsyncGenerator

from config import GENERATOR_TEMPERATURE, INDEX_PATH, SAATHI_CARE_TAG_FREQ
from core.prohibited_words import check_prohibited
from core.schemas import (
    AnalyzerState,
    SessionState,
    StrategyDecision,
    TurnRecord,
)
from llm import get_llm
from prompts.generator_prompt import _CARE_TAG_POOL, build_generator_prompt
from retrieval.generator_retriever import GeneratorRetriever


logger = logging.getLogger(__name__)


# Static fallback if the LLM stream collapses entirely. Hinglish, neutral,
# coping-shade-agnostic, safe in every phase. Better than emitting nothing.
_FALLBACK_RESPONSE = (
    "Main samajh sakta hoon yeh kitna mushkil hai. Thoda aur bataoge?"
)


# Soft-warning phrases — if the generator produces 2+ of these we log loudly
# (and in a future iteration may force a regeneration). These are the exact
# stock phrases the bot was over-using in screenshot evidence: every reply
# ended up sounding like the same poetic counsellor template.
_OVERUSED_PHRASES: list[str] = [
    r"\bkabhi kabhi yeh sab\b",
    r"\bhar waqt sar pe\b",
    r"\bsaans lene ki bhi fursat\b",
    r"\bjaise sab kuch ek saath\b",
    r"\bek chhupi hui umeed\b",
    r"\bupheaval ke beech\b",
    r"\blagta hai tumhare andar\b",
    r"\bsamajh sakta hoon\b",
    r"\bbohot overwhelming\b",
    r"\bbahut overwhelming\b",
]
_OVERUSED_RE = [re.compile(p, re.IGNORECASE) for p in _OVERUSED_PHRASES]


def _detect_overused_phrases(text: str) -> list[str]:
    """Return the list of overused-phrase regexes that matched. Used only
    for soft-warning logging today; later we may use this to trigger a
    regeneration with a 'do not use these phrases' system note."""
    if not text:
        return []
    hits: list[str] = []
    for raw, rx in zip(_OVERUSED_PHRASES, _OVERUSED_RE):
        if rx.search(text):
            hits.append(raw.strip(r"\b"))
    return hits


class Generator:
    """Owns the FAISS retriever instance + the Generator LLM client."""

    def __init__(self) -> None:
        self.llm = get_llm("generator")
        # Loading the FAISS index is ~100 MB on disk + dim*N float32s in RAM,
        # so we do it ONCE per process and reuse across turns.
        self.retriever = GeneratorRetriever(INDEX_PATH)

    async def generate_stream(
        self,
        seeker_text: str,
        analyzer_state: AnalyzerState,
        strategy_decision: StrategyDecision,
        conversation_history: list[TurnRecord],
        session: SessionState,
    ) -> AsyncGenerator[str, None]:
        """Yield response tokens as they arrive. Always yields *something* —
        falls back to a safe canned line if everything blows up."""

        # ---- STEPS 1 & 2: retrieve + format examples ----
        try:
            examples = await self.retriever.retrieve(
                seeker_text=seeker_text,
                strategy=strategy_decision.selected_strategy,
                coping_mech=analyzer_state.current_coping_mech,
                phase=strategy_decision.current_phase,
                intensity=analyzer_state.emotion_intensity,
                persona_code=session.persona_code,
                emotion=analyzer_state.emotion_type,
                top_k=6,
            )
            formatted_examples = self.retriever.format_for_prompt(examples)
            negative_example = self.retriever.format_negative_example()
            logger.info(
                "Retrieved %d few-shot examples (strategy=%s, coping=%s, phase=%s)",
                len(examples),
                strategy_decision.selected_strategy,
                analyzer_state.current_coping_mech,
                strategy_decision.current_phase,
            )
        except Exception as e:
            logger.error("Retrieval failed (continuing with empty pool): %s",
                         e, exc_info=True)
            formatted_examples = (
                "(No examples available — generate from your training)"
            )
            negative_example = ""

        # ---- STEP 3: build prompt (now session-aware) ----
        messages = build_generator_prompt(
            seeker_text=seeker_text,
            analyzer_state=analyzer_state,
            strategy_decision=strategy_decision,
            conversation_history=conversation_history,
            retrieved_examples=formatted_examples,
            negative_example=negative_example,
            persona_code=session.persona_code,
            session=session,
        )

        # ---- STEP 4: stream tokens ----
        full_response = ""
        try:
            async for token in self.llm.generate_stream(
                messages=messages,
                temperature=GENERATOR_TEMPERATURE,
                max_tokens=250,
            ):
                full_response += token
                yield token
        except Exception as e:
            logger.error("Generator streaming failed: %s", e, exc_info=True)
            # If we already streamed some bytes, append the fallback so the
            # user gets a complete sentence. If nothing streamed, emit just
            # the fallback.
            yield _FALLBACK_RESPONSE
            full_response = (full_response + " " + _FALLBACK_RESPONSE).strip()

        # ---- STEP 5: post-stream prohibited-vocab audit (log only) ----
        # TODO: when we have headroom, on violation re-call the LLM with a
        # "you used <X> — rewrite using <Y> instead" reminder and replace the
        # stored turn (the user already saw the violating text — that ship has
        # sailed for this turn, but we can prevent it from being indexed into
        # future memory).
        try:
            has_prohibited, found_words = check_prohibited(full_response)
            if has_prohibited:
                logger.warning(
                    "PROHIBITED WORDS in Generator output: %s | response[:200]=%r",
                    found_words,
                    full_response[:200],
                )
        except Exception as e:
            # Audit must never break the turn.
            logger.error("Prohibited-vocab audit crashed: %s", e, exc_info=True)

        # ---- STEP 6: overused-phrase audit (log only, for now) ----
        try:
            overused = _detect_overused_phrases(full_response)
            if len(overused) >= 2:
                logger.warning(
                    "OVER-POETIC reply detected (%d filler phrases: %s) — "
                    "consider tightening the system prompt or regenerating. "
                    "response[:200]=%r",
                    len(overused), overused, full_response[:200],
                )
            elif overused:
                logger.info(
                    "Note: 1 stock phrase used: %s (acceptable, monitor)",
                    overused,
                )
        except Exception as e:
            logger.error("Overused-phrase audit crashed: %s", e, exc_info=True)

        # ---- STEP 7: update rolling session.facts_log + care-tag turn ----
        # We do NOT call session_manager.save here — the orchestrator's
        # `update_after_turn` runs right after this and persists the whole
        # session. We just mutate the in-memory session object.
        try:
            new_facts = list(analyzer_state.concrete_facts or [])
            if new_facts:
                seen = {f.strip().lower() for f in session.facts_log}
                for f in new_facts:
                    key = f.strip().lower()
                    if key and key not in seen:
                        session.facts_log.append(f.strip())
                        seen.add(key)
                # Cap the rolling log so it doesn't grow unbounded.
                _MAX_FACTS = 32
                if len(session.facts_log) > _MAX_FACTS:
                    session.facts_log = session.facts_log[-_MAX_FACTS:]

            # If the prompt would have suggested a care tag this turn AND any
            # of the pool phrases actually appeared in the reply, mark the
            # care-tag turn so we don't re-fire too soon.
            if (
                SAATHI_CARE_TAG_FREQ > 0
                and any(tag in full_response for tag in _CARE_TAG_POOL)
            ):
                session.last_care_tag_turn = session.turn_count + 1
        except Exception as e:
            logger.error("facts_log/care-tag update crashed: %s", e, exc_info=True)
