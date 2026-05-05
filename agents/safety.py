"""Regex-first crisis detection with optional LLM backup (Agent 3)."""

from __future__ import annotations

import json
import logging
import re

from config import SAFETY_TEMPERATURE
from core.schemas import SafetyFlags, SessionState
from llm import get_llm


logger = logging.getLogger(__name__)


# Stage 1 — regex patterns (compiled once at import)
CRISIS_PATTERNS: dict[str, list[str]] = {
    "suicidal_ideation": [
        r"mar\s*jaana", r"marna\s*chahta", r"marna\s*chahti",
        r"zindagi\s*khatam", r"jeena\s*nahi", r"jeene\s*ka\s*mann\s*nahi",
        r"suicide", r"khatam\s*kar\s*(?:loon|loo|leta|leti|doon|du)",
        r"sab\s*khatam\s*kar\s*d(?:oon|u|eta)",
        r"khud\s*ko\s*khatam", r"duniya\s*chhod",
        r"maut", r"mar\s*jaau", r"nahi\s*rehna",
    ],
    "self_harm": [
        r"khud\s*ko\s*hurt", r"khud\s*ko\s*dard",
        r"apne\s*aap\s*ko.*(?:marna|katna|hurt|nuksaan)",
        r"self[\s-]*harm", r"cut\s*kar(?:na|ta|ti)",
        r"nuksaan\s*pahunchana",
    ],
    "severe_hopelessness": [
        r"koi\s*fayda\s*nahi", r"kuch\s*nahi\s*bacha",
        r"sab\s*khatam.*(?:hai|ho\s*gaya)",
        r"ummeed\s*nahi.*(?:rahi|hai|bachee)",
        r"koi\s*raasta\s*nahi", r"ab\s*kuch\s*nahi\s*ho\s*sakta",
        r"haar\s*gaya.*(?:hoon|hun|hu)", r"haar\s*gayi.*hoon",
    ],
}

COMPILED_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    category: [re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns]
    for category, patterns in CRISIS_PATTERNS.items()
}


# Static crisis response (validated copy)
CRISIS_RESPONSE = (
    "Tum jo share kar rahe ho, woh bahut important hai aur main sunta hoon. "
    "Main chahta hoon ki tum safe raho. Please abhi iVARS helpline pe call karo: "
    "9152987821 — wahan trained log hain jo samjhenge, 24 ghante available hain. "
    "Kya tum abhi safe ho?"
)


# SafetyChecker
class SafetyChecker:
    """Cascading risk classifier."""

    def __init__(self) -> None:
        self.llm = get_llm("safety")

    async def check(self, text: str, session: SessionState) -> SafetyFlags:
        """Return a SafetyFlags verdict for `text` in the context of
        `session`. NEVER raises — failures degrade to no-flag."""

        # ---- Stage 1: regex pattern match ----
        stage1 = self._stage1_regex(text or "")
        if stage1 is not None:
            return stage1

        # ---- Stage 2: session-level escalation heuristics ----
        stage2 = self._stage2_history(session)
        if stage2 is not None:
            return stage2

        # ---- Stage 3: LLM classifier ----
        return await self._stage3_llm(text or "")

    # Stage 1
    @staticmethod
    def _stage1_regex(text: str) -> SafetyFlags | None:
        for category, compiled_list in COMPILED_PATTERNS.items():
            for pattern in compiled_list:
                match = pattern.search(text)
                if match:
                    trigger = match.group()
                    logger.warning(
                        "SAFETY STAGE 1 HIT: category=%s, trigger=%r",
                        category, trigger,
                    )
                    return SafetyFlags(
                        requires_hitl=True,
                        risk_level="high",
                        trigger_phrase=trigger,
                    )
        return None

    # Stage 2
    @staticmethod
    def _stage2_history(session: SessionState) -> SafetyFlags | None:
        # Sustained high intensity across the last 2+ turns.
        recent_intensities = session.intensity_trajectory[-3:]
        if (
            len(recent_intensities) >= 2
            and all(i >= 5 for i in recent_intensities)
        ):
            logger.warning(
                "SAFETY STAGE 2: sustained high intensity (last 3 = %s)",
                recent_intensities,
            )
            return SafetyFlags(
                requires_hitl=True,
                risk_level="high",
                trigger_phrase="sustained_high_intensity",
            )

        # Prior turn carried a risk signal AND the latest analyzer state also
        # has one. (We only have one prior AnalyzerState in SessionState, so
        # the "multiple risk signals" check collapses into "the most recent
        # one fired and intensity is staying high".)
        latest = session.latest_analyzer_state
        if (
            latest is not None
            and latest.risk_signal
            and latest.emotion_intensity >= 5
        ):
            logger.warning(
                "SAFETY STAGE 2: prior turn flagged risk + intensity>=5 "
                "(risk=%r, intensity=%s)",
                latest.risk_signal, latest.emotion_intensity,
            )
            return SafetyFlags(
                requires_hitl=True,
                risk_level="high",
                trigger_phrase="pattern_escalation",
            )

        return None

    # Stage 3
    async def _stage3_llm(self, text: str) -> SafetyFlags:
        if not text.strip():
            return SafetyFlags(
                requires_hitl=False, risk_level="none", trigger_phrase=None,
            )

        llm_messages = [
            {
                "role": "system",
                "content": (
                    "You are a safety classifier. Assess the risk level of "
                    "the following message. Focus on: suicidal ideation, "
                    "self-harm intent, severe hopelessness, giving up on "
                    "life. Reply with ONLY a JSON object: "
                    '{"risk_level": "none" | "low" | "medium" | "high", '
                    '"reason": "brief explanation"}'
                ),
            },
            {"role": "user", "content": f'Message to assess: "{text}"'},
        ]

        try:
            response_text = await self.llm.generate_text(
                messages=llm_messages,
                temperature=SAFETY_TEMPERATURE,
                max_tokens=80,
            )
            parsed = self._parse_classifier_json(response_text)
            risk = parsed.get("risk_level") or "none"
            reason = (parsed.get("reason") or "")[:100]

            if risk not in ("none", "low", "medium", "high"):
                logger.warning(
                    "SAFETY STAGE 3 returned unknown risk_level=%r — defaulting to none",
                    risk,
                )
                risk = "none"

            if risk in ("medium", "high"):
                logger.warning(
                    "SAFETY STAGE 3 LLM: risk=%s, reason=%s", risk, reason,
                )
                return SafetyFlags(
                    requires_hitl=(risk == "high"),
                    risk_level=risk,
                    trigger_phrase=reason or None,
                )

            return SafetyFlags(
                requires_hitl=False,
                risk_level=risk,
                trigger_phrase=None,
            )

        except Exception as e:
            logger.error(
                "SAFETY STAGE 3 LLM classifier failed (defaulting to no-flag): %s",
                e, exc_info=True,
            )
            return SafetyFlags(
                requires_hitl=False, risk_level="none", trigger_phrase=None,
            )

    # JSON parser tolerant of ```json fences
    @staticmethod
    def _parse_classifier_json(response_text: str) -> dict:
        cleaned = (response_text or "").strip()
        # Strip ```json ... ``` or ``` ... ``` fencing.
        if cleaned.startswith("```"):
            inner = cleaned.split("```")
            # Pieces: ['', 'json\n{...}\n', ''] or ['', '{...}', '']
            if len(inner) >= 2:
                cleaned = inner[1].strip()
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].lstrip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Last-ditch: pull the first {...} blob.
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise
