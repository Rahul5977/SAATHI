"""Expectation helpers and `run_checks` for YAML `expect:` blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        marker = "✓" if self.passed else "✗"
        return f"  {marker} {self.name}{(' — ' + self.detail) if self.detail else ''}"


# Observation contract — the runner builds one of these per turn
@dataclass
class TurnObservation:
    """Everything we captured about a single turn — passed to every check."""
    turn_index: int
    seeker_text: str
    response_text: str
    phase: Optional[str] = None
    strategy: Optional[str] = None
    lens: Optional[str] = None
    decision_reason: Optional[str] = None
    analyzer: Optional[dict] = None
    safety_risk: Optional[str] = None
    safety_trigger: Optional[str] = None
    summary: Optional[dict] = None  # SessionSummary as dict, may be None
    facts_log: Optional[list[str]] = None
    phase_history: Optional[list[str]] = None


# Atomic check primitives
def check_phase(obs: TurnObservation, expected: str) -> CheckResult:
    actual = obs.phase
    return CheckResult(
        name=f"phase == {expected!r}",
        passed=(actual == expected),
        detail=("" if actual == expected else f"got {actual!r}"),
    )


def check_phase_in(obs: TurnObservation, allowed: list[str]) -> CheckResult:
    return CheckResult(
        name=f"phase ∈ {allowed!r}",
        passed=(obs.phase in allowed),
        detail=("" if obs.phase in allowed else f"got {obs.phase!r}"),
    )


def check_strategy(obs: TurnObservation, expected: str) -> CheckResult:
    return CheckResult(
        name=f"strategy == {expected!r}",
        passed=(obs.strategy == expected),
        detail=("" if obs.strategy == expected else f"got {obs.strategy!r}"),
    )


def check_strategy_in(obs: TurnObservation, allowed: list[str]) -> CheckResult:
    return CheckResult(
        name=f"strategy ∈ {allowed!r}",
        passed=(obs.strategy in allowed),
        detail=("" if obs.strategy in allowed else f"got {obs.strategy!r}"),
    )


def check_strategy_not(obs: TurnObservation, forbidden: list[str]) -> CheckResult:
    bad = obs.strategy in forbidden
    return CheckResult(
        name=f"strategy ∉ {forbidden!r}",
        passed=(not bad),
        detail=(f"forbidden strategy {obs.strategy!r} was selected" if bad else ""),
    )


def check_decision_reason_matches(obs: TurnObservation, pattern: str) -> CheckResult:
    text = obs.decision_reason or ""
    matched = re.search(pattern, text) is not None
    return CheckResult(
        name=f"decision_reason matches /{pattern}/",
        passed=matched,
        detail=("" if matched else f"reason was {text!r}"),
    )


def check_safety_risk(obs: TurnObservation, expected: str) -> CheckResult:
    actual = (obs.safety_risk or "").lower()
    return CheckResult(
        name=f"safety_risk == {expected!r}",
        passed=(actual == expected.lower()),
        detail=("" if actual == expected.lower() else f"got {actual!r}"),
    )


# Response-text checks
def check_response_contains_any(
    obs: TurnObservation, needles: list[str]
) -> CheckResult:
    text = (obs.response_text or "").lower()
    hits = [n for n in needles if n.lower() in text]
    return CheckResult(
        name=f"response contains any of {needles!r}",
        passed=bool(hits),
        detail=(
            f"matched {hits!r}"
            if hits
            else f"none matched in response[:120]={obs.response_text[:120]!r}"
        ),
    )


def check_response_contains_all(
    obs: TurnObservation, needles: list[str]
) -> CheckResult:
    text = (obs.response_text or "").lower()
    misses = [n for n in needles if n.lower() not in text]
    return CheckResult(
        name=f"response contains all of {needles!r}",
        passed=(not misses),
        detail=(
            ""
            if not misses
            else f"missing {misses!r} in response[:120]={obs.response_text[:120]!r}"
        ),
    )


def check_response_no_phrases(
    obs: TurnObservation, banned: list[str]
) -> CheckResult:
    text = (obs.response_text or "").lower()
    found = [b for b in banned if b.lower() in text]
    return CheckResult(
        name=f"response avoids banned phrases {banned!r}",
        passed=(not found),
        detail=("" if not found else f"used {found!r}"),
    )


def check_response_max_words(obs: TurnObservation, max_words: int) -> CheckResult:
    n = len((obs.response_text or "").split())
    return CheckResult(
        name=f"response_word_count <= {max_words}",
        passed=(n <= max_words),
        detail=(f"got {n} words" if n > max_words else f"got {n} words"),
    )


def check_response_min_words(obs: TurnObservation, min_words: int) -> CheckResult:
    n = len((obs.response_text or "").split())
    return CheckResult(
        name=f"response_word_count >= {min_words}",
        passed=(n >= min_words),
        detail=(f"got {n} words" if n < min_words else f"got {n} words"),
    )


def check_response_matches(obs: TurnObservation, pattern: str) -> CheckResult:
    text = obs.response_text or ""
    matched = re.search(pattern, text, flags=re.IGNORECASE) is not None
    return CheckResult(
        name=f"response matches /{pattern}/i",
        passed=matched,
        detail=("" if matched else f"response[:200]={text[:200]!r}"),
    )


# Analyzer / memory sub-spec checks
def _bound_check(value: Any, bounds: dict) -> tuple[bool, str]:
    """Check `value` against {ge, le, gt, lt, eq} bounds. Returns (ok, detail)."""
    if value is None:
        return False, "value is None"
    try:
        v = value
        for op, target in bounds.items():
            if op == "ge" and not (v >= target): return False, f"{v} < {target}"
            if op == "le" and not (v <= target): return False, f"{v} > {target}"
            if op == "gt" and not (v >  target): return False, f"{v} <= {target}"
            if op == "lt" and not (v <  target): return False, f"{v} >= {target}"
            if op == "eq" and not (v == target): return False, f"{v} != {target}"
        return True, f"value={v}"
    except TypeError as e:
        return False, f"cannot compare {value!r}: {e}"


def check_analyzer(obs: TurnObservation, spec: dict) -> list[CheckResult]:
    """Evaluate sub-checks against the recorded analyzer dict."""
    results: list[CheckResult] = []
    az = obs.analyzer or {}
    for field, rule in (spec or {}).items():
        if field == "intensity":
            if isinstance(rule, dict):
                ok, detail = _bound_check(az.get("emotion_intensity"), rule)
                results.append(CheckResult(
                    name=f"analyzer.intensity {rule}", passed=ok, detail=detail,
                ))
            else:
                got = az.get("emotion_intensity")
                results.append(CheckResult(
                    name=f"analyzer.intensity == {rule}",
                    passed=(got == rule),
                    detail=("" if got == rule else f"got {got}"),
                ))
        elif field == "emotion":
            got = az.get("emotion_type")
            if isinstance(rule, list):
                results.append(CheckResult(
                    name=f"analyzer.emotion ∈ {rule}",
                    passed=(got in rule),
                    detail=("" if got in rule else f"got {got!r}"),
                ))
            else:
                results.append(CheckResult(
                    name=f"analyzer.emotion == {rule!r}",
                    passed=(got == rule),
                    detail=("" if got == rule else f"got {got!r}"),
                ))
        elif field == "receptiveness":
            got = az.get("user_receptiveness")
            results.append(CheckResult(
                name=f"analyzer.receptiveness == {rule!r}",
                passed=(got == rule),
                detail=("" if got == rule else f"got {got!r}"),
            ))
        elif field == "coping":
            got = az.get("current_coping_mech")
            results.append(CheckResult(
                name=f"analyzer.coping == {rule!r}",
                passed=(got == rule),
                detail=("" if got == rule else f"got {got!r}"),
            ))
        elif field == "stigma_cue":
            got = bool(az.get("stigma_cue"))
            results.append(CheckResult(
                name=f"analyzer.stigma_cue == {rule}",
                passed=(got == bool(rule)),
                detail=("" if got == bool(rule) else f"got {got}"),
            ))
        elif field == "risk_signal_set":
            got = az.get("risk_signal") is not None
            results.append(CheckResult(
                name=f"analyzer.risk_signal_set == {rule}",
                passed=(got == bool(rule)),
                detail=("" if got == bool(rule) else f"got risk_signal={az.get('risk_signal')!r}"),
            ))
        elif field == "concrete_facts_contains_any":
            facts = [f.lower() for f in az.get("concrete_facts", []) or []]
            hits = [n for n in (rule or []) if any(n.lower() in f for f in facts)]
            results.append(CheckResult(
                name=f"analyzer.concrete_facts contains any of {rule!r}",
                passed=bool(hits),
                detail=(f"matched {hits!r}" if hits else f"facts={facts!r}"),
            ))
        else:
            results.append(CheckResult(
                name=f"analyzer.{field} (unknown rule)",
                passed=False,
                detail=f"unknown analyzer rule {field!r}",
            ))
    return results


def check_memory(obs: TurnObservation, spec: dict) -> list[CheckResult]:
    """Evaluate memory-layer sub-checks against the recorded summary."""
    results: list[CheckResult] = []
    summary = obs.summary or {}
    facts_log = obs.facts_log or []

    for field, rule in (spec or {}).items():
        if field == "summary_present":
            ok = bool(summary) == bool(rule)
            results.append(CheckResult(
                name=f"memory.summary_present == {rule}",
                passed=ok,
                detail=("" if ok else f"got summary={'yes' if summary else 'no'}"),
            ))
        elif field == "seeker_goal_contains":
            goal = (summary.get("seeker_goal") or "").lower()
            needles = rule if isinstance(rule, list) else [rule]
            hits = [n for n in needles if n.lower() in goal]
            results.append(CheckResult(
                name=f"memory.seeker_goal contains any of {needles!r}",
                passed=bool(hits),
                detail=(f"matched {hits!r}" if hits else f"goal={goal!r}"),
            ))
        elif field == "key_facts_contains_any":
            facts = [f.lower() for f in summary.get("key_facts", []) or []]
            needles = rule if isinstance(rule, list) else [rule]
            hits = [n for n in needles if any(n.lower() in f for f in facts)]
            results.append(CheckResult(
                name=f"memory.key_facts contains any of {needles!r}",
                passed=bool(hits),
                detail=(f"matched {hits!r}" if hits else f"key_facts={facts!r}"),
            ))
        elif field == "facts_log_contains_any":
            log = [f.lower() for f in facts_log]
            needles = rule if isinstance(rule, list) else [rule]
            hits = [n for n in needles if any(n.lower() in f for f in log)]
            results.append(CheckResult(
                name=f"memory.facts_log contains any of {needles!r}",
                passed=bool(hits),
                detail=(f"matched {hits!r}" if hits else f"facts_log={log!r}"),
            ))
        elif field == "open_threads_contains_any":
            threads = [t.lower() for t in summary.get("open_threads", []) or []]
            needles = rule if isinstance(rule, list) else [rule]
            hits = [n for n in needles if any(n.lower() in t for t in threads)]
            results.append(CheckResult(
                name=f"memory.open_threads contains any of {needles!r}",
                passed=bool(hits),
                detail=(f"matched {hits!r}" if hits else f"threads={threads!r}"),
            ))
        else:
            results.append(CheckResult(
                name=f"memory.{field} (unknown rule)",
                passed=False,
                detail=f"unknown memory rule {field!r}",
            ))
    return results


# The orchestrator — turns a per-turn `expect:` spec into CheckResult list
_TOP_LEVEL_DISPATCH = {
    "phase":                  check_phase,
    "phase_in":               check_phase_in,
    "strategy":               check_strategy,
    "strategy_in":            check_strategy_in,
    "strategy_not":           check_strategy_not,
    "decision_reason_matches": check_decision_reason_matches,
    "safety_risk":            check_safety_risk,
    "response_contains_any":  check_response_contains_any,
    "response_contains_all":  check_response_contains_all,
    "response_no_phrases":    check_response_no_phrases,
    "response_max_words":     check_response_max_words,
    "response_min_words":     check_response_min_words,
    "response_matches":       check_response_matches,
}


def run_checks(obs: TurnObservation, spec: Optional[dict]) -> list[CheckResult]:
    """Evaluate the entire `expect:` block of one turn. Returns one
    CheckResult per atomic assertion (a single key may produce multiple,
    e.g. `analyzer:`)."""
    results: list[CheckResult] = []
    if not spec:
        return results
    for key, value in spec.items():
        if key == "analyzer":
            results.extend(check_analyzer(obs, value or {}))
        elif key == "memory":
            results.extend(check_memory(obs, value or {}))
        elif key in _TOP_LEVEL_DISPATCH:
            results.append(_TOP_LEVEL_DISPATCH[key](obs, value))
        else:
            results.append(CheckResult(
                name=f"unknown rule '{key}'",
                passed=False,
                detail=f"check key {key!r} is not registered",
            ))
    return results


# Smoke test
if __name__ == "__main__":
    obs = TurnObservation(
        turn_index=1,
        seeker_text="kya karu yaar 1 hafte me JEE advanced hai",
        response_text="Bhai JEE advanced ke pehle aise lagna bilkul samajh sakta hoon.",
        phase="Insight",
        strategy="REFLECTION_OF_FEELINGS",
        lens=None,
        decision_reason="R3b high recept (help-seeking) + int≤4 → Insight",
        analyzer={
            "emotion_type": "fear",
            "emotion_intensity": 3,
            "user_receptiveness": "high",
            "current_coping_mech": "Sequential",
            "stigma_cue": False,
            "risk_signal": None,
            "concrete_facts": ["JEE Advanced in 1 week"],
        },
        safety_risk="none",
        summary={
            "seeker_goal": "Figure out what to do this week before JEE Advanced.",
            "key_facts": ["JEE Advanced in 1 week"],
            "open_threads": ["seeker asked for direction"],
        },
        facts_log=["JEE Advanced in 1 week"],
        phase_history=["Exploration", "Insight"],
    )

    spec = {
        "phase": "Insight",
        "strategy_in": ["REFLECTION_OF_FEELINGS", "AFFIRMATION_AND_REASSURANCE"],
        "decision_reason_matches": r"^R3",
        "safety_risk": "none",
        "response_contains_any": ["JEE", "exam"],
        "response_max_words": 30,
        "response_no_phrases": ["bohot overwhelming"],
        "analyzer": {
            "intensity": {"ge": 2, "le": 4},
            "receptiveness": "high",
            "risk_signal_set": False,
            "concrete_facts_contains_any": ["JEE"],
        },
        "memory": {
            "summary_present": True,
            "seeker_goal_contains": ["JEE"],
            "key_facts_contains_any": ["JEE Advanced"],
            "facts_log_contains_any": ["JEE Advanced"],
        },
    }

    results = run_checks(obs, spec)
    fails = [r for r in results if not r.passed]
    for r in results:
        print(r)
    assert not fails, f"unexpected failures: {fails}"
    print(f"\n{len(results)} checks ran, all passed ✓")

    # Now flip a few things to ensure failures surface as data
    obs_bad = TurnObservation(
        turn_index=1, seeker_text="x", response_text="bohot overwhelming hai",
        phase="Action", strategy="PROVIDING_SUGGESTIONS",
    )
    bad_results = run_checks(obs_bad, {
        "phase": "Insight",
        "response_no_phrases": ["bohot overwhelming"],
    })
    bad_fails = [r for r in bad_results if not r.passed]
    assert len(bad_fails) == 2, f"expected 2 failures, got {bad_fails}"
    print("failure surfacing: OK")

    print("\neval/checks.py — all checks passed ✓")
