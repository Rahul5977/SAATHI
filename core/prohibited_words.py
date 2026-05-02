"""
SAATHI prohibited clinical vocabulary.

The Generator must never sound like a textbook therapist or a diagnostic chatbot.
This module provides:
  - PROHIBITED_ALL: a flat lowercase list to scan generated text against
  - check_prohibited(text): returns (has_violations, list_of_found_words)
  - get_replacement_suggestions(): mapping from clinical term -> SAATHI-style alternative
"""

from __future__ import annotations

import re
from typing import Tuple

PROHIBITED_CLINICAL: list[str] = [
    "depression", "depressed", "anxiety disorder", "anxiety",
    "trauma", "traumatic", "ptsd", "triggers", "triggered",
    "boundaries", "boundary", "toxic", "toxicity",
    "gaslighting", "gaslight", "mental health", "mental illness",
    "therapy", "therapist", "counseling", "counselor",
    "diagnose", "diagnosis", "diagnosed",
    "disorder", "syndrome", "condition",
    "medication", "medicine", "antidepressant", "antianxiety",
    "psychiatrist", "psychologist", "psychiatric",
    "burnout", "burned out", "burnt out",
    "self-care", "coping mechanism", "coping strategy",
    "emotional intelligence", "mindfulness",
    "cognitive behavioral", "CBT",
    "narcissist", "narcissistic",
    "codependent", "codependency",
    "attachment style", "avoidant",
    "bipolar", "schizophrenia", "OCD",
    "panic attack", "panic disorder",
]

# Hindi / Hinglish equivalents that are equally clinical or stigmatizing
PROHIBITED_HINDI: list[str] = [
    "mansik rog", "mansik bimari", "pagal", "paagal",
    "dawai", "goliyan", "psychiatrist", "psychologist",
]

# Therapist-y advisory phrases banned in S7 of conversation_generation.txt.
# These are MULTI-WORD phrases that use substring matching.
PROHIBITED_PHRASES: list[str] = [
    # English advisory cliches (S7)
    "see a therapist",
    "set boundaries",
    "prioritise yourself",
    "prioritize yourself",
    "you are so strong",
    "you're so strong",
    "cut them off",
    "cut him off",
    "cut her off",
    # Toxic-positivity Hinglish lines explicitly banned in S7
    "sab theek ho jayega",
    "sab thik ho jayega",
    "everything will be fine",
]

# Single deduplicated lowercase list — used by check_prohibited at runtime.
PROHIBITED_ALL: list[str] = sorted(
    {
        w.lower().strip()
        for w in (PROHIBITED_CLINICAL + PROHIBITED_HINDI + PROHIBITED_PHRASES)
        if w.strip()
    }
)


def check_prohibited(text: str) -> Tuple[bool, list[str]]:
    """
    Check whether `text` contains any prohibited clinical vocabulary.

    Matching is:
      - case-insensitive
      - whole-word for single-token entries (so "OCD" doesn't fire on "decode")
      - substring for multi-token entries like "panic attack" or "anxiety disorder"

    Returns (has_prohibited, list_of_found_words_in_canonical_form).
    The returned list is deduplicated and preserves first-occurrence order.
    """
    if not isinstance(text, str) or not text:
        return (False, [])

    text_lower = text.lower()
    found: list[str] = []
    seen: set[str] = set()

    for word in PROHIBITED_ALL:
        if word in seen:
            continue
        if " " in word or "-" in word:
            # Multi-token / hyphenated entries: substring match is sufficient.
            if word in text_lower:
                found.append(word)
                seen.add(word)
        else:
            # Single-token entries: require a word boundary so "anxiety" doesn't
            # match inside "anxieties" -> actually it should match; but "ocd"
            # must NOT match "decode". \b handles both correctly because the
            # boundary is between alnum and non-alnum.
            pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
            if pattern.search(text_lower):
                found.append(word)
                seen.add(word)

    return (len(found) > 0, found)


def get_replacement_suggestions() -> dict[str, str]:
    """
    Mapping of prohibited clinical terms to SAATHI-appropriate alternatives.

    Used in the Generator system prompt to teach the model what to say instead
    of clinical jargon. Phrasing is intentionally Hinglish and embodied.
    """
    return {
        "depression": "bahut udaasi / andar se toot-sa jaana",
        "burnout": "thak ke choor ho jaana / machine ki tarah chal-te rehna",
        "anxiety": "andar se ghabrahat / dil mein bechainee",
        "therapy": "kisi se baat karna jo samjhe",
        "boundaries": "apne liye thoda waqt nikalna",
        "toxic": "woh rishta jo andar se todta hai",
        "triggers": "woh baatein jo andar tak chubhti hain",
        "self-care": "apna khayal rakhna",
        "trauma": "woh gahri chot jo andar hai",
        "coping mechanism": "jaise tum deal kar rahe ho",
        "panic attack": "achanak itni ghabrahat ki saans na aaye",
    }
