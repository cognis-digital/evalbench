"""EVALBENCH — offline LLM / agent eval harness with regression gates.

No network, no third-party deps. Define test cases + assertions in a JSON
suite, run them against recorded model outputs (or a pluggable provider),
and gate CI on a pass-rate threshold + regression against a baseline.
"""
from .core import (
    Assertion,
    TestCase,
    Suite,
    CaseResult,
    RunReport,
    load_suite,
    run_suite,
    compare_baseline,
    ASSERTIONS,
)

TOOL_NAME = "evalbench"
TOOL_VERSION = "1.0.0"

__all__ = [
    "Assertion",
    "TestCase",
    "Suite",
    "CaseResult",
    "RunReport",
    "load_suite",
    "run_suite",
    "compare_baseline",
    "ASSERTIONS",
    "TOOL_NAME",
    "TOOL_VERSION",
]
