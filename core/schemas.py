"""Pydantic models and taxonomies shared across the SAATHI pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Emotion taxonomy
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


# Problem (context) taxonomy — aligned with dataset CATEGORY MAP
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


# Coping mechanisms — matches dataset COPING MAP exactly
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


# Strategies & phases (unchanged — align with dataset)
STRATEGIES: list[str] = [
    "RESTATEMENT_OR_PARAPHRASING", "QUESTION", "REFLECTION_OF_FEELINGS",
    "AFFIRMATION_AND_REASSURANCE", "SELF_DISCLOSURE", "PROVIDING_SUGGESTIONS",
    "EXECUTION", "INFORMATION",
]

PHASES: list[str] = ["Exploration", "Insight", "Action"]


# Restatement lenses — labels and mechanism compatibility (per S4 of
# dataset/prompts/conversation_generation.txt)
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


# Helpers
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


# AnalyzerState — output of Agent 1 (Analyzer)
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
    # Concrete details the seeker mentioned this turn — short noun-phrases
    # the Generator can reference back to ("PPT presentation", "exam in 1 day",
    # "papa retired last month"). Lets SAATHI sound like it actually heard
    # them, instead of speaking only in abstractions ("academics ka pressure").
    # Max 5 items, each <= 12 words.
    concrete_facts: list[str] = Field(default_factory=list)

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


# StrategyDecision — output of phase_gate.py (deterministic)
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


# SafetyFlags — output of safety checker
class SafetyFlags(BaseModel):
    """Output of the safety checker."""

    model_config = ConfigDict(extra="ignore")

    requires_hitl: bool = False
    risk_level: Literal["none", "low", "medium", "high"] = "none"
    trigger_phrase: Optional[str] = None


# TurnRecord — single turn in conversation history
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


# RetrievalDebugItem — one few-shot row surfaced to the dev UI / meta
class RetrievalDebugItem(BaseModel):
    """Snapshot of a single retrieved dataset record for debugging. Filled by
    `agents.generator` after each successful `GeneratorRetriever.retrieve`."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    conversation_id: str = ""
    faiss_score: Optional[float] = None
    final_score: Optional[float] = None
    strategy: str = ""
    phase: str = ""
    emotion: str = ""
    seeker_preview: str = Field(default="", max_length=200)


# SessionSummary — periodic compressed view of the session
class SessionSummary(BaseModel):
    """Structured running summary of the session, written by the Summarizer
    agent every few turns. Lets the Generator stay coherent past the 16-turn
    history cap WITHOUT carrying the full history forward each call.

    The fields here are intentionally short — the goal is "what would a friend
    remember about this conversation if they zoned out for a minute and then
    rejoined?", NOT "transcribe everything verbatim".
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    # ≤ ~80 words. Past tense, third-person about the seeker. Captures the
    # arc of the conversation so far. e.g. "Seeker is preparing for JEE
    # Advanced, exam in 1 week. Started panicked, asked for help around turn
    # 3 ('kya karu?'), tried a 5-min breathing pause and reported it helped
    # slightly. Currently working through study-plan options."
    narrative: str = Field(default="", max_length=800)

    # The single most pressing thing the seeker is trying to resolve. Stays
    # pinned across turns so the bot never loses the thread. Can be None on
    # turn 1 before there's enough signal.
    seeker_goal: Optional[str] = Field(default=None, max_length=200)

    # Hard, persistent details from the session — used to seed `facts_log` if
    # it gets pruned. e.g. ["JEE Advanced in 1 week", "from Bangalore",
    # "papa retired last month"].
    key_facts: list[str] = Field(default_factory=list)

    # One-phrase emotional arc. e.g. "panicked → settling" or
    # "ashamed and closed → opening up cautiously".
    emotional_arc: str = Field(default="", max_length=160)

    # One-phrase phase journey. e.g. "Exploration→Insight by turn 4" or
    # "still in Exploration, intensity dropping".
    phase_journey: str = Field(default="", max_length=160)

    # Threads the bot has opened but not yet closed — questions asked,
    # suggestions offered awaiting feedback, etc. Lets the bot follow up
    # naturally instead of jumping topic.
    open_threads: list[str] = Field(default_factory=list)

    # Bookkeeping — which turn this summary was generated at. Used by the
    # orchestrator to decide whether the summary is stale.
    generated_at_turn: int = 0

    @field_validator("key_facts")
    @classmethod
    def _cap_facts(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for f in v or []:
            if not f:
                continue
            key = f.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(f.strip())
        return ordered[:12]

    @field_validator("open_threads")
    @classmethod
    def _cap_threads(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for t in v or []:
            if not t:
                continue
            key = t.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(t.strip())
        return ordered[:5]


# UserProfile — cross-session memory keyed by user_id
class UserProfile(BaseModel):
    """Persistent, cross-session memory for a single user. Stored in Redis
    under `user_profile:{user_id}` independently of any session.

    On session creation, the orchestrator hydrates `facts_log` from the
    profile's `key_life_facts` so the bot can say "JEE Advanced ke baad
    ka kya plan banaya tha pichli baar?" on day 2 — that's the difference
    between a chatbot and a friend.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    user_id: str

    # Persona that the bot has settled on for this user across sessions.
    # Once `persona_locked=True` for a session, the bot stops re-evaluating it.
    persona_code: str = "P0"

    # Optional — the user's name if they ever shared it. Keeps the bot from
    # asking "tumhara naam kya hai?" every session.
    display_name: Optional[str] = None

    # Recurring topic threads across sessions, e.g. ["academic stress",
    # "father's pressure", "Hostel loneliness"]. Updated incrementally —
    # never overwritten in full.
    recurring_themes: list[str] = Field(default_factory=list)

    # Hard life facts the bot should never forget. e.g. ["IIT JEE aspirant",
    # "studying in Kota since 2024", "family in Patna"]. Higher bar than
    # session facts_log — these survive a session ending.
    key_life_facts: list[str] = Field(default_factory=list)

    # Compressed summary of the LAST completed session. Lets the bot open
    # session N+1 with continuity. e.g. "Last time we talked, you were 1 week
    # before JEE Advanced. You tried a 5-min breathing pause that helped a
    # bit. You said you'd revise maths the next day. Kya hua?"
    last_session_summary: Optional[str] = Field(default=None, max_length=800)

    # The goal of the previous session (for follow-up).
    last_session_goal: Optional[str] = Field(default=None, max_length=200)

    # When the user last interacted (UTC ISO string). String is used (not
    # datetime) so the JSON serialization stays trivial across redis.
    last_seen_at: Optional[str] = None

    # Counters.
    sessions_count: int = 0
    total_turns: int = 0

    @field_validator("recurring_themes")
    @classmethod
    def _cap_themes(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for t in v:
            if not t:
                continue
            key = t.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(t.strip())
        return ordered[:20]

    @field_validator("key_life_facts")
    @classmethod
    def _cap_life_facts(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for f in v:
            if not f:
                continue
            key = f.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(f.strip())
        return ordered[:30]

    def touch(self) -> None:
        """Refresh `last_seen_at` to now (UTC ISO)."""
        self.last_seen_at = datetime.now(timezone.utc).isoformat()


# SessionState — persisted in Redis between turns
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
    # Human-readable explanation of why the most recent phase came out as it
    # did (e.g. "R3b high recept (help-seeking) + int≤4 → Insight"). Surfaced
    # in the dev UI debug panel; not used by any agent logic.
    latest_phase_decision_reason: Optional[str] = None
    turn_history: list[TurnRecord] = Field(default_factory=list)
    hitl_escalated: bool = False
    # Rolling de-duplicated log of concrete facts mentioned across the whole
    # session. Generator pulls the most recent ~8 here so it can reference
    # things from earlier turns even after they fall out of `turn_history`.
    facts_log: list[str] = Field(default_factory=list)
    # Last care-tag turn — used by the care-gesture rotation in the Generator
    # so we don't ask "khaana khaya?" twice in a row.
    last_care_tag_turn: int = 0

    # Periodic compressed view written by the Summarizer agent. Lets the
    # Generator reason about turns 1-N even after they fall out of the
    # 16-turn history cap. None until the first summarization fires (around
    # turn 4).
    summary: Optional[SessionSummary] = None

    # The turn at which the bot first reached each phase. Helps phase_gate
    # answer "have we ever reached Insight in this session?" without
    # walking the entire history. Maintained incrementally in
    # `SessionManager.update_after_turn`.
    phase_first_reached: dict[str, int] = Field(default_factory=dict)

    # How many consecutive turns we've spent in the current phase. Used by
    # the anti-stuck heuristic in `core.phase_gate`.
    turns_in_current_phase: int = 0

    # Snapshot of the cross-session profile that was hydrated when this
    # session was created. The orchestrator reads this when building the
    # Generator prompt so cross-session continuity ("pichli baar tum yeh
    # bata rahe the...") is possible. Live profile updates go through
    # `MemoryManager`, not here.
    user_profile_snapshot: Optional[UserProfile] = None

    # Filled at the end of each successful Generator pass; cleared on
    # retrieval failure or crisis short-circuit (or overwritten next turn).
    latest_retrieval_debug: list[RetrievalDebugItem] = Field(default_factory=list)
    latest_retrieval_query: Optional[str] = None
    latest_retrieval_filter_level: Optional[str] = None

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

    def has_reached_phase(self, phase: str) -> bool:
        """True if the conversation has ever reached `phase` so far."""
        return phase in self.phase_first_reached
