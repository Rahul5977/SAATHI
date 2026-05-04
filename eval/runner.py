"""
Eval runner: drive the orchestrator through hand-authored YAML scenarios
and capture observations + assertion results per turn.

YAML scenario schema (see `eval/golden/README.md`):

    name: phase_advances_on_help_seeking
    description: Bot should advance from Exploration to Insight when seeker asks for help
    user_id: u_test                       # optional; auto-derived from name otherwise
    persona_code: P0                      # optional; default P0
    seed_profile:                         # optional; pre-populates a UserProfile
      sessions_count: 0
      key_life_facts: []
    turns:
      - say: "Sab theek hai, bas thoda thaka hua hoon"
        expect:
          phase: Exploration
      - say: "Kya karu yaar, kuch samajh ni aa raha"
        expect:
          phase: Insight
          decision_reason_matches: "^R3"

The runner produces a `ScenarioResult` per scenario. The CLI in
`eval/__main__.py` aggregates results and prints a terminal report.

Determinism:
  - Generator runs at the temperature defined in `config.GENERATOR_TEMPERATURE`.
    Eval mode does NOT alter agent temperatures (we want to see realistic
    output behavior). Tests are written using SHAPE assertions
    (regex / contains / structural), not exact-string matches.
  - Each scenario uses a fresh in-memory session AND profile (no Redis
    leakage between tests).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# Import after the path is resolved by being a submodule of `eval`.
from eval.checks import CheckResult, TurnObservation, run_checks


logger = logging.getLogger(__name__)


GOLDEN_DIR = Path(__file__).parent / "golden"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class TurnResult:
    turn_index: int
    seeker_text: str
    response_text: str
    observation: TurnObservation
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


@dataclass
class ScenarioResult:
    name: str
    description: str
    turns: list[TurnResult] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: Optional[str] = None  # set on harness-level failure

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        return all(t.passed for t in self.turns)

    @property
    def total_checks(self) -> int:
        return sum(len(t.checks) for t in self.turns)

    @property
    def failed_checks(self) -> int:
        return sum(
            sum(1 for c in t.checks if not c.passed) for t in self.turns
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_scenarios(
    pattern: Optional[str] = None,
    directory: Optional[Path] = None,
) -> list[dict]:
    """Load all YAML scenarios from `directory` (default: eval/golden/).
    If `pattern` is provided, only scenarios whose name matches the regex
    are returned."""
    directory = directory or GOLDEN_DIR
    scenarios: list[dict] = []
    for path in sorted(directory.rglob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text())
        except yaml.YAMLError as e:
            logger.error("YAML parse failed for %s: %s", path, e)
            continue
        if data is None:
            continue
        if "name" not in data:
            data["name"] = path.stem
        data["__path__"] = str(path)
        scenarios.append(data)

    if pattern:
        rx = re.compile(pattern, re.IGNORECASE)
        scenarios = [s for s in scenarios if rx.search(s.get("name", ""))]
    return scenarios


# ---------------------------------------------------------------------------
# Single-scenario runner
# ---------------------------------------------------------------------------
async def run_scenario(scenario: dict) -> ScenarioResult:
    """Execute one YAML scenario end-to-end against a fresh in-memory
    orchestrator + memory store. Returns a `ScenarioResult` even on harness
    failure (with `.error` set) so the CLI can keep going."""

    # IMPORTANT: import the orchestrator inside the function so a single
    # FAISS load amortizes across all scenarios — the singleton lives on
    # the orchestrator instance we create here.
    from pipeline.orchestrator import PipelineOrchestrator
    from core.schemas import UserProfile

    name = scenario.get("name", "(unnamed)")
    description = scenario.get("description", "")

    result = ScenarioResult(name=name, description=description)
    started = time.time()

    # Each scenario gets a fresh orchestrator (so SessionManager + MemoryManager
    # state is isolated). The FAISS index in Generator is reloaded though —
    # that's the cost of test isolation. For a fast suite we'd amortize it.
    orchestrator = PipelineOrchestrator()
    # Force in-memory stores so tests are hermetic regardless of dev Redis state.
    orchestrator.session_manager._use_redis = False        # noqa: SLF001
    orchestrator.memory_manager._use_redis = False         # noqa: SLF001

    try:
        # Optionally pre-seed a UserProfile so cross-session tests can run.
        seed = scenario.get("seed_profile") or {}
        user_id = scenario.get("user_id") or f"u_{name[:24]}"
        session_id = scenario.get("session_id") or f"s_{uuid.uuid4().hex[:10]}"

        if seed:
            profile = UserProfile(user_id=user_id, **seed)
            await orchestrator.memory_manager.save(profile)

        for i, turn in enumerate(scenario.get("turns") or [], start=1):
            seeker_text = (turn.get("say") or "").strip()
            if not seeker_text:
                logger.warning(
                    "Scenario %s turn %d has no `say:` — skipping", name, i,
                )
                continue

            tokens: list[str] = []
            try:
                async for tok in orchestrator.run(
                    session_id=session_id,
                    user_id=user_id,
                    seeker_text=seeker_text,
                ):
                    tokens.append(tok)
            except Exception as e:
                logger.error(
                    "Scenario %s turn %d: orchestrator crashed: %s",
                    name, i, e, exc_info=True,
                )
                result.error = f"turn {i}: orchestrator crashed: {e}"
                break

            response_text = "".join(tokens).strip()

            # Snapshot the persisted SessionState to extract decision/memory data.
            session = await orchestrator.session_manager.get_session(session_id)
            obs = _build_observation(
                turn_index=i,
                seeker_text=seeker_text,
                response_text=response_text,
                session=session,
            )

            # Run the YAML's `expect:` block against this observation.
            check_results = run_checks(obs, turn.get("expect"))

            tr = TurnResult(
                turn_index=i,
                seeker_text=seeker_text,
                response_text=response_text,
                observation=obs,
                checks=check_results,
            )
            result.turns.append(tr)

            # Optional fail-fast within a scenario via `stop_on_fail: true`
            if scenario.get("stop_on_fail") and not tr.passed:
                break

        # Optional cross-scenario hook: scenario can declare a `then:` block to
        # close the session and start a new one (cross-session test). Out of
        # scope for v1 — keep simple.

    except Exception as e:
        logger.error("Scenario %s harness failure: %s", name, e, exc_info=True)
        result.error = f"harness failure: {e}"
    finally:
        try:
            await orchestrator.close()
        except Exception:
            pass

    result.duration_seconds = time.time() - started
    return result


def _build_observation(
    turn_index: int,
    seeker_text: str,
    response_text: str,
    session,  # SessionState | None
) -> TurnObservation:
    """Project the persisted SessionState into a `TurnObservation`."""
    obs = TurnObservation(
        turn_index=turn_index,
        seeker_text=seeker_text,
        response_text=response_text,
    )
    if session is None:
        return obs

    sd = session.latest_strategy_decision
    if sd is not None:
        obs.phase = sd.current_phase
        obs.strategy = sd.selected_strategy
        obs.lens = sd.restatement_lens

    obs.decision_reason = session.latest_phase_decision_reason

    az = session.latest_analyzer_state
    if az is not None:
        obs.analyzer = az.model_dump()

    sf = session.latest_safety_flags
    if sf is not None:
        obs.safety_risk = sf.risk_level
        obs.safety_trigger = sf.trigger_phrase

    if session.summary is not None:
        obs.summary = session.summary.model_dump()

    obs.facts_log = list(session.facts_log)
    obs.phase_history = list(session.phase_history)
    return obs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_terminal_report(
    results: list[ScenarioResult], verbose: bool = False
) -> str:
    """Render results as a fixed-width terminal report."""
    lines: list[str] = []
    total_checks = sum(r.total_checks for r in results)
    total_failed = sum(r.failed_checks for r in results)
    total_scenarios = len(results)
    failed_scenarios = sum(1 for r in results if not r.passed)

    lines.append("=" * 78)
    lines.append("SAATHI eval — golden scenario results")
    lines.append("=" * 78)
    for r in results:
        marker = "PASS" if r.passed else "FAIL"
        lines.append(
            f"\n[{marker}] {r.name}  ({r.duration_seconds:.1f}s, "
            f"{r.failed_checks}/{r.total_checks} failed)"
        )
        if r.description:
            lines.append(f"       {r.description}")
        if r.error:
            lines.append(f"       ERROR: {r.error}")
            continue
        # Only show per-turn detail for failures (or if -v).
        for tr in r.turns:
            show_turn = verbose or not tr.passed
            if not show_turn:
                continue
            preview = (tr.response_text[:80] + "…") if len(tr.response_text) > 80 else tr.response_text
            lines.append(
                f"  turn {tr.turn_index}: seeker={tr.seeker_text[:60]!r}"
            )
            lines.append(f"           saathi=  {preview!r}")
            obs = tr.observation
            lines.append(
                f"           phase={obs.phase} strategy={obs.strategy} "
                f"reason={(obs.decision_reason or '')!r}"
            )
            for c in tr.checks:
                if verbose or not c.passed:
                    lines.append(f"      {c}")

    lines.append("\n" + "-" * 78)
    lines.append(
        f"SUMMARY: {total_scenarios - failed_scenarios}/{total_scenarios} "
        f"scenarios passed,  "
        f"{total_checks - total_failed}/{total_checks} checks passed"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point used by `python -m eval`
# ---------------------------------------------------------------------------
async def run_all(pattern: Optional[str] = None, verbose: bool = False) -> int:
    """Load and run all golden scenarios, print the report, return exit code
    (0 = all passed, 1 = at least one failure)."""
    scenarios = load_scenarios(pattern=pattern)
    if not scenarios:
        print(f"No scenarios found in {GOLDEN_DIR}"
              + (f" matching {pattern!r}" if pattern else ""))
        return 1

    results: list[ScenarioResult] = []
    for s in scenarios:
        logger.info("Running scenario: %s", s.get("name"))
        results.append(await run_scenario(s))

    print(format_terminal_report(results, verbose=verbose))
    return 0 if all(r.passed for r in results) else 1


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m eval",
        description="Run SAATHI golden eval scenarios.",
    )
    p.add_argument(
        "-k", "--pattern",
        help="Regex; only scenarios whose name matches this run.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show every check (passing + failing), not just failures.",
    )
    p.add_argument(
        "--log-level", default="WARNING",
        help="Logging level for noisy submodules.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(run_all(pattern=args.pattern, verbose=args.verbose))


if __name__ == "__main__":
    raise SystemExit(main())
