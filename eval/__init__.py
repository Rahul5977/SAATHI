"""SAATHI evaluation harness.

Drive the orchestrator through hand-authored YAML scenarios and assert that
the per-turn observations (phase, strategy, response text, analyzer state,
session memory) match expectations. Failures localize to a specific turn
and a specific assertion so regressions are debuggable.

Run with:
    python -m eval                  # all golden tests
    python -m eval -k memory        # tests whose name matches "memory"
    python -m eval -v               # full per-turn diff on failure

See `eval/golden/README.md` for the YAML schema.
"""
