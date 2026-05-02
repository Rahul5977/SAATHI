"""
SAATHI core schemas.

Pydantic v2 contracts shared across the pipeline. The taxonomy here is the
canonical source of truth and aligns with `dataset/prompts/conversation_generation.txt`
and `dataset/prompts/cot_generation.txt`.

Key design choices:
  - PROBLEM_TYPES uses the C1-C8 taxonomy from the dataset prompts (with a
    parallel CATEGORY_CODE_MAP / CATEGORY_LABEL_MAP for code/label lookup).
  - EMOTION_TYPES covers the most common canonical labels seen in the dataset
    plus an alias map for high-frequency variants ("anxious" -> "anxiety", etc.).
  - RESTATEMENT_LENSES includes mechanism compatibility metadata so callers
    (e.g. phase_gate.compute_lens) can enforce dataset rules at runtime.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Emotion taxonomy
# ---------------------------------------------------------------------------
# Canonical lowercase labels the Analyzer LLM is allowed to return.
EMOTION_TYPES: list[str] = [
    # Core distress (per Prompt 2 base set)
    "exhaustion", "panic", "shame", "grief", "numbness", "fear",
    "helplessness", "despair", "guilt", "overwhelm", "resignation",
    # Frequent variants observed in the actual dataset
    "anxiety", "sadness", "frustration", "anger", "isolation",
    "loneliness", "humiliation", "worthlessness", "suffocation",
    "alienation", "betrayal", "resentment", "confusion", "fatigue",
    "vulnerability", "inadequacy", "hopelessness", "pressure",
    "trapped",
    # Recovery / positive (used when intensity drops in Action phase)
    "hope", "relief", "calm", "gratitude", "acceptance", "determination",
    "clarity", "peace", "validation", "readiness",
]

# Adjective / participle forms -> canonical noun forms.
EMOTION_ALIASES: dict[str, str] = {
    "anxious": "anxiety",
    "ashamed": "shame",
    "guilty": "guilt",
    "exhausted": "exhaustion",
    "fatigued": "fatigue",
    "tired": "fatigue",
    "weary": "fatigue",
    "helpless": "helplessness",
    "hopeful": "hope",
    "hopeless": "hopelessness",
    "humiliated": "humiliation",
    "isolated": "isolation",
    "lonely": "loneliness",
    "alienated": "alienation",
    "betrayed": "betrayal",
    "resentful": "resentment",
    "frustrated": "frustration",
    "sad": "sadness",
    "panicked": "panic",
    "fearful": "fear",
    "scared": "fear",
    "worried": "fear",
    "trapped/helpless": "trapped",
    "suffocated": "suffocation",
    "numb": "numbness",
    "overwhelmed": "overwhelm",
    "relieved": "relief",
    "calmer": "calm",
    "calmness": "calm",
    "grateful": "gratitude",
    "thankful": "gratitude",
    "determined": "determination",
    "ready": "readiness",
    "willing": "readiness",
    "validated": "validation",
    "resolute": "determination",
    "resigned": "resignation",
    "vulnerable": "vulnerability",
    "worthless": "worthlessness",
    "hurt": "vulnerability",
    "burnout": "exhaustion",
}


# ---------------------------------------------------------------------------
# Problem (context) taxonomy — aligned with dataset CATEGORY MAP
# ---------------------------------------------------------------------------
PROBLEM_TYPES: list[str] = [
    "Academic_Pressure",        # C1
    "Family_Dynamics",          # C2
    "Marriage_Rishta",          # C3
    "Employment_Livelihood",    # C4
    "Financial_Debt",           # C5
    "Gender_Identity",          # C6
    "Health_Chronic_Illness",   # C7
    "Migration_Displacement",   # C8
]

# Canonical -> short C-code (used by retriever filters and CoT chain).
CATEGORY_CODE_MAP: dict[str, str] = {
    "Academic_Pressure":      "C1",
    "Family_Dynamics":        "C2",
    "Marriage_Rishta":        "C3",
    "Employment_Livelihood":  "C4",
    "Financial_Debt":         "C5",
    "Gender_Identity":        "C6",
    "Health_Chronic_Illness": "C7",
    "Migration_Displacement": "C8",
}

# Canonical -> human label exactly as written in the dataset prompts.
CATEGORY_LABEL_MAP: dict[str, str] = {
    "Academic_Pressure":      "Academic Pressure",
    "Family_Dynamics":        "Family Dynamics",
    "Marriage_Rishta":        "Marriage & Rishta",
    "Employment_Livelihood":  "Employment & Livelihood",
    "Financial_Debt":         "Financial Debt",
    "Gender_Identity":        "Gender & Identity",
    "Health_Chronic_Illness": "Health & Chronic Illness",
    "Migration_Displacement": "Migration & Displacement",
}

# Loose dataset folder names + prompt-side variants -> canonical PROBLEM_TYPE.
CATEGORY_FOLDER_MAP: dict[str, str] = {
    # Spec labels
    "academic pressure":          "Academic_Pressure",
    "family dynamics":            "Family_Dynamics",
    "marriage & rishta":          "Marriage_Rishta",
    "marriage and rishta":        "Marriage_Rishta",
    "employment & livelihood":    "Employment_Livelihood",
    "employment and livelihood":  "Employment_Livelihood",
    "financial debt":             "Financial_Debt",
    "gender & identity":          "Gender_Identity",
    "gender and identity":        "Gender_Identity",
    "health & chronic illness":   "Health_Chronic_Illness",
    "health and chronic illness": "Health_Chronic_Illness",
    "migration & displacement":   "Migration_Displacement",
    "migration and displacement": "Migration_Displacement",
    # Codes
    "c1": "Academic_Pressure", "c2": "Family_Dynamics",
    "c3": "Marriage_Rishta",   "c4": "Employment_Livelihood",
    "c5": "Financial_Debt",    "c6": "Gender_Identity",
    "c7": "Health_Chronic_Illness", "c8": "Migration_Displacement",
    # situation_generation.txt enum strings
    "academic_pressure":   "Academic_Pressure",
    "placement_stress":    "Employment_Livelihood",
    "family_conflict":     "Family_Dynamics",
    "workplace_hierarchy": "Employment_Livelihood",
    "marriage_pressure":   "Marriage_Rishta",
    "financial_burden":    "Financial_Debt",
    "grief_illness":       "Health_Chronic_Illness",
    "social_stigma":       "Gender_Identity",
    "identity_conflict":   "Gender_Identity",
    # Real dataset folder names (typos preserved on purpose)
    "academic-presuure": "Academic_Pressure",
    "academic-pressure": "Academic_Pressure",
    "employement":       "Employment_Livelihood",
    "employment":        "Employment_Livelihood",
    "financial":         "Financial_Debt",
    "gender":            "Gender_Identity",
    "marriage":          "Marriage_Rishta",
    "migration":         "Migration_Displacement",
    "health":            "Health_Chronic_Illness",
    "familial-and-interpersonal-conflicts": "Family_Dynamics",
}


# ---------------------------------------------------------------------------
# Coping mechanisms — matches dataset COPING MAP exactly
# ---------------------------------------------------------------------------
COPING_MECHANISMS: list[str] = [
    "Duty_Based", "Relational_Preservation", "Somatization", "Sequential",
]

COPING_ALIASES: dict[str, str] = {
    "duty based":              "Duty_Based",
    "duty-based":              "Duty_Based",
    "relational preservation": "Relational_Preservation",
    "relational-preservation": "Relational_Preservation",
    "relational_pres":         "Relational_Preservation",
    "somatic":                 "Somatization",
    "sequential coping":       "Sequential",
    "emotion-first":           "Sequential",
}


# ---------------------------------------------------------------------------
# Strategies & phases (unchanged — align with dataset)
# ---------------------------------------------------------------------------
STRATEGIES: list[str] = [
    "RESTATEMENT_OR_PARAPHRASING", "QUESTION", "REFLECTION_OF_FEELINGS",
    "AFFIRMATION_AND_REASSURANCE", "SELF_DISCLOSURE", "PROVIDING_SUGGESTIONS",
    "EXECUTION", "INFORMATION",
]

PHASES: list[str] = ["Exploration", "Insight", "Action"]


# ---------------------------------------------------------------------------
# Restatement lenses — labels and mechanism compatibility (per S4 of
# dataset/prompts/conversation_generation.txt)
# ---------------------------------------------------------------------------
RESTATEMENT_LENSES: dict[str, str] = {
    "A": "physical/somatic frame",
    "B": "role/duty frame",
    "C": "relational frame",
    "D": "time/duration frame",
    "E": "feared consequence frame",
    "F": "emotional overwhelm frame",
}

# Mechanism -> set of lenses that are FORBIDDEN for that mechanism.
# Sourced from S4 of dataset/prompts/conversation_generation.txt:
#   "Lens A FORBIDDEN for Sequential"
#   "Lens F FORBIDDEN for Somatization"
LENS_FORBIDDEN_BY_MECHANISM: dict[str, set[str]] = {
    "Sequential":   {"A"},
    "Somatization": {"F"},
    "Duty_Based":            set(),
    "Relational_Preservation": set(),
}

# Mechanism -> primary/preferred lens (informational; phase_gate uses it as a hint).
LENS_PRIMARY_BY_MECHANISM: dict[str, str] = {
    "Somatization":            "A",
    "Duty_Based":              "B",
    "Relational_Preservation": "C",
    "Sequential":              "F",
}

RECEPTIVENESS_VALUES: list[str] = ["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fuzzy_match(value: str, allowed: list[str]) -> Optional[str]:
    """Case-insensitive substring match either direction. Returns canonical or None."""
    if not isinstance(value, str):
        return None
    v = value.lower().strip()
    if not v:
        return None
    for canon in allowed:
        c = canon.lower()
        if c == v or c in v or v in c:
            return canon
    return None


def normalize_category(value: str) -> Optional[str]:
    """
    Normalize a free-form category string (folder name, label, or code) to a
    canonical PROBLEM_TYPE. Returns None if no mapping is found.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    key = value.strip().lower().replace("_", " ").replace("-", "-")
    # First try the explicit folder/label map (lowercased keys).
    if key in CATEGORY_FOLDER_MAP:
        return CATEGORY_FOLDER_MAP[key]
    # Try the underscore-preserving key as well.
    key_us = value.strip().lower()
    if key_us in CATEGORY_FOLDER_MAP:
        return CATEGORY_FOLDER_MAP[key_us]
    # Try matching against canonical PROBLEM_TYPES directly.
    return _fuzzy_match(value, PROBLEM_TYPES)


def normalize_emotion(value: str) -> Optional[str]:
    """Normalize a free-form emotion to a canonical EMOTION_TYPES value."""
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.lower().strip()
    if v in EMOTION_TYPES:
        return v
    if v in EMOTION_ALIASES:
        return EMOTION_ALIASES[v]
    return _fuzzy_match(v, EMOTION_TYPES)


def normalize_coping(value: str) -> Optional[str]:
    """Normalize a free-form coping mechanism to canonical COPING_MECHANISMS."""
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip()
    if v in COPING_MECHANISMS:
        return v
    v_low = v.lower()
    if v_low in COPING_ALIASES:
        return COPING_ALIASES[v_low]
    # Tolerate spaces/dashes ("Duty Based" -> "Duty_Based")
    v_norm = v.replace(" ", "_").replace("-", "_")
    if v_norm in COPING_MECHANISMS:
        return v_norm
    return _fuzzy_match(v, COPING_MECHANISMS)


# ---------------------------------------------------------------------------
# AnalyzerState — output of Agent 1 (Analyzer)
# ---------------------------------------------------------------------------
class AnalyzerState(BaseModel):
    """Captures the seeker's emotional/behavioral state."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    emotion_type: str
    emotion_intensity: int = Field(
        description="1=resolved, 2=mild, 3=moderate, 4=high, 5=severe, 6=crisis"
    )
    problem_type: str
    current_coping_mech: str
    coping_shade_signal: str  # EXACT phrase from seeker — never paraphrased
    user_receptiveness: Literal["low", "medium", "high"]
    is_new_problem: bool
    stigma_cue: bool
    risk_signal: Optional[str] = None  # EXACT crisis phrase if detected, else None

    @field_validator("emotion_intensity")
    @classmethod
    def validate_intensity(cls, v: int) -> int:
        if not 1 <= v <= 6:
            raise ValueError(f"emotion_intensity must be 1-6, got {v}")
        return v

    @field_validator("emotion_type")
    @classmethod
    def validate_emotion(cls, v: str) -> str:
        canon = normalize_emotion(v)
        if canon is None:
            raise ValueError(
                f"emotion_type must be normalizable to one of {EMOTION_TYPES}, got '{v}'"
            )
        return canon

    @field_validator("problem_type")
    @classmethod
    def validate_problem(cls, v: str) -> str:
        if v in PROBLEM_TYPES:
            return v
        canon = normalize_category(v)
        if canon is not None:
            return canon
        raise ValueError(
            f"problem_type must be normalizable to one of {PROBLEM_TYPES}, got '{v}'"
        )

    @field_validator("current_coping_mech")
    @classmethod
    def validate_coping(cls, v: str) -> str:
        canon = normalize_coping(v)
        if canon is None:
            raise ValueError(
                f"current_coping_mech must be one of {COPING_MECHANISMS}, got '{v}'"
            )
        return canon

    @field_validator("user_receptiveness", mode="before")
    @classmethod
    def validate_receptiveness(cls, v):
        # Runs BEFORE the Literal[...] check so we can normalize "LOW" -> "low".
        if isinstance(v, str):
            v_norm = v.lower().strip()
            if v_norm in RECEPTIVENESS_VALUES:
                return v_norm
            raise ValueError(
                f"user_receptiveness must be one of {RECEPTIVENESS_VALUES}, got '{v}'"
            )
        return v


# ---------------------------------------------------------------------------
# StrategyDecision — output of phase_gate.py (deterministic)
# ---------------------------------------------------------------------------
class StrategyDecision(BaseModel):
    """Output of the deterministic strategy engine. No LLM involved."""

    model_config = ConfigDict(extra="ignore")

    current_phase: str
    selected_strategy: str
    restatement_lens: Optional[str] = None

    @field_validator("current_phase")
    @classmethod
    def validate_phase(cls, v: str) -> str:
        if v not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got '{v}'")
        return v

    @field_validator("selected_strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        if v not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got '{v}'")
        return v

    @field_validator("restatement_lens")
    @classmethod
    def validate_lens(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if v not in RESTATEMENT_LENSES:
            raise ValueError(
                f"restatement_lens must be one of {list(RESTATEMENT_LENSES.keys())}, got '{v}'"
            )
        return v


# ---------------------------------------------------------------------------
# SafetyFlags — output of safety checker
# ---------------------------------------------------------------------------
class SafetyFlags(BaseModel):
    """Output of the safety checker."""

    model_config = ConfigDict(extra="ignore")

    requires_hitl: bool = False
    risk_level: Literal["none", "low", "medium", "high"] = "none"
    trigger_phrase: Optional[str] = None


# ---------------------------------------------------------------------------
# TurnRecord — single turn in conversation history
# ---------------------------------------------------------------------------
class TurnRecord(BaseModel):
    """A single turn in conversation history."""

    model_config = ConfigDict(extra="ignore")

    turn_id: int
    speaker: Literal["Seeker", "Supporter"]
    text: str
    emotion: Optional[str] = None
    intensity: Optional[int] = None
    strategy: Optional[str] = None
    phase: Optional[str] = None


# ---------------------------------------------------------------------------
# SessionState — persisted in Redis between turns
# ---------------------------------------------------------------------------
class SessionState(BaseModel):
    """Complete session state, persisted in Redis between turns."""

    model_config = ConfigDict(extra="ignore")

    session_id: str
    user_id: str
    persona_code: str = "P0"
    persona_locked: bool = False
    turn_count: int = 0
    phase_history: list[str] = Field(default_factory=list)
    strategy_history: list[str] = Field(default_factory=list)
    intensity_trajectory: list[int] = Field(default_factory=list)
    coping_trajectory: list[str] = Field(default_factory=list)
    self_disclosure_used: bool = False
    self_disclosure_turn: Optional[int] = None
    latest_analyzer_state: Optional[AnalyzerState] = None
    latest_strategy_decision: Optional[StrategyDecision] = None
    latest_safety_flags: Optional[SafetyFlags] = None
    turn_history: list[TurnRecord] = Field(default_factory=list)
    hitl_escalated: bool = False

    def get_recent_history(self, n: int = 6) -> list[TurnRecord]:
        """Return last n turns of conversation history."""
        if n <= 0:
            return []
        return self.turn_history[-n:]

    def get_last_strategies(self, n: int = 2) -> list[str]:
        """Return last n strategies used."""
        if n <= 0:
            return []
        return self.strategy_history[-n:]
