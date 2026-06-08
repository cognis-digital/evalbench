"""Core eval engine: assertions, suite loading, runner, regression compare.

A *suite* is JSON:
{
  "name": "qa",
  "threshold": 0.8,          # min pass rate to succeed (optional)
  "cases": [
    {
      "id": "greet",
      "input": "say hi",                 # carried through, informational
      "output": "Hello there!",          # recorded model/agent output
      "weight": 1.0,
      "assertions": [
        {"type": "contains", "value": "hello", "ignore_case": true},
        {"type": "max_tokens", "value": 50}
      ]
    }
  ]
}

Assertions are pure functions over the output string — fully offline.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------
# Assertion implementations. Each returns (passed: bool, detail: str).
# --------------------------------------------------------------------------

def _norm(s: Any) -> str:
    return s if isinstance(s, str) else json.dumps(s, sort_keys=True)


def _maybe_ci(text: str, value: str, ignore_case: bool) -> Tuple[str, str]:
    if ignore_case:
        return text.lower(), value.lower()
    return text, value


def _a_contains(output: str, a: "Assertion") -> Tuple[bool, str]:
    text, val = _maybe_ci(output, str(a.value), a.ignore_case)
    ok = val in text
    return ok, f"expected substring {a.value!r}" + ("" if ok else " — not found")


def _a_not_contains(output: str, a: "Assertion") -> Tuple[bool, str]:
    text, val = _maybe_ci(output, str(a.value), a.ignore_case)
    ok = val not in text
    return ok, f"forbidden substring {a.value!r}" + ("" if ok else " — present")


def _a_equals(output: str, a: "Assertion") -> Tuple[bool, str]:
    text, val = _maybe_ci(output.strip(), str(a.value).strip(), a.ignore_case)
    ok = text == val
    return ok, "exact match" if ok else f"expected {a.value!r}, got {output.strip()!r}"


def _a_regex(output: str, a: "Assertion") -> Tuple[bool, str]:
    flags = re.IGNORECASE if a.ignore_case else 0
    try:
        ok = re.search(str(a.value), output, flags) is not None
    except re.error as e:
        return False, f"invalid regex {a.value!r}: {e}"
    return ok, f"regex {a.value!r}" + ("" if ok else " — no match")


def _a_max_tokens(output: str, a: "Assertion") -> Tuple[bool, str]:
    n = len(output.split())
    ok = n <= int(a.value)
    return ok, f"{n} tokens (limit {a.value})"


def _a_min_tokens(output: str, a: "Assertion") -> Tuple[bool, str]:
    n = len(output.split())
    ok = n >= int(a.value)
    return ok, f"{n} tokens (min {a.value})"


def _a_is_json(output: str, a: "Assertion") -> Tuple[bool, str]:
    try:
        parsed = json.loads(output)
    except (ValueError, TypeError) as e:
        return False, f"not valid JSON: {e}"
    if a.value:  # optional: require a top-level key path / type
        if isinstance(a.value, str) and isinstance(parsed, dict):
            ok = a.value in parsed
            return ok, f"json key {a.value!r}" + ("" if ok else " — missing")
    return True, "valid JSON"


def _a_json_path(output: str, a: "Assertion") -> Tuple[bool, str]:
    """value is a dotted path; matches against a.expected (optional)."""
    try:
        parsed = json.loads(output)
    except (ValueError, TypeError) as e:
        return False, f"not valid JSON: {e}"
    cur: Any = parsed
    for part in str(a.value).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return False, f"path {a.value!r} not found"
    if a.expected is not None:
        ok = _norm(cur) == _norm(a.expected)
        return ok, f"path {a.value} = {cur!r}" + ("" if ok else f" (expected {a.expected!r})")
    return True, f"path {a.value} present ({cur!r})"


def _a_levenshtein(output: str, a: "Assertion") -> Tuple[bool, str]:
    """Fuzzy similarity gate. value=target string, threshold=min ratio 0..1."""
    target = str(a.value)
    ratio = _similarity(output.strip(), target.strip())
    thr = a.threshold if a.threshold is not None else 0.8
    ok = ratio >= thr
    return ok, f"similarity {ratio:.2f} (min {thr})"


ASSERTIONS: Dict[str, Callable[[str, "Assertion"], Tuple[bool, str]]] = {
    "contains": _a_contains,
    "not_contains": _a_not_contains,
    "equals": _a_equals,
    "regex": _a_regex,
    "max_tokens": _a_max_tokens,
    "min_tokens": _a_min_tokens,
    "is_json": _a_is_json,
    "json_path": _a_json_path,
    "similarity": _a_levenshtein,
}


def _similarity(a: str, b: str) -> float:
    """Normalized Levenshtein similarity (1.0 = identical), stdlib only."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    dist = prev[-1]
    return 1.0 - dist / max(len(a), len(b))


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Assertion:
    type: str
    value: Any = None
    ignore_case: bool = False
    expected: Any = None
    threshold: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Assertion":
        if "type" not in d:
            raise ValueError("assertion missing 'type'")
        if d["type"] not in ASSERTIONS:
            raise ValueError(f"unknown assertion type {d['type']!r}")
        return cls(
            type=d["type"],
            value=d.get("value"),
            ignore_case=bool(d.get("ignore_case", False)),
            expected=d.get("expected"),
            threshold=d.get("threshold"),
        )

    def check(self, output: str) -> Tuple[bool, str]:
        return ASSERTIONS[self.type](output, self)


@dataclass
class TestCase:
    id: str
    output: str
    input: Any = None
    weight: float = 1.0
    assertions: List[Assertion] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TestCase":
        if "id" not in d:
            raise ValueError("case missing 'id'")
        if "output" not in d:
            raise ValueError(f"case {d['id']!r} missing 'output'")
        return cls(
            id=str(d["id"]),
            output=str(d["output"]),
            input=d.get("input"),
            weight=float(d.get("weight", 1.0)),
            assertions=[Assertion.from_dict(a) for a in d.get("assertions", [])],
        )


@dataclass
class Suite:
    name: str
    cases: List[TestCase]
    threshold: float = 1.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Suite":
        cases = [TestCase.from_dict(c) for c in d.get("cases", [])]
        if not cases:
            raise ValueError("suite has no cases")
        ids = [c.id for c in cases]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate case ids: {sorted(dupes)}")
        return cls(
            name=str(d.get("name", "suite")),
            cases=cases,
            threshold=float(d.get("threshold", 1.0)),
        )


@dataclass
class CaseResult:
    id: str
    passed: bool
    weight: float
    assertions: List[Dict[str, Any]]


@dataclass
class RunReport:
    suite: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    weighted_pass_rate: float
    threshold: float
    ok: bool
    cases: List[CaseResult]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# --------------------------------------------------------------------------
# Loading + running
# --------------------------------------------------------------------------

def load_suite(path: str) -> Suite:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return Suite.from_dict(data)


def run_suite(suite: Suite) -> RunReport:
    results: List[CaseResult] = []
    for case in suite.cases:
        adetails: List[Dict[str, Any]] = []
        case_ok = True
        for a in case.assertions:
            ok, detail = a.check(case.output)
            if not ok:
                case_ok = False
            adetails.append({"type": a.type, "passed": ok, "detail": detail})
        # A case with zero assertions is a no-op pass (still counted).
        results.append(CaseResult(case.id, case_ok, case.weight, adetails))

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    pass_rate = passed / total if total else 0.0
    wtot = sum(r.weight for r in results) or 1.0
    wpass = sum(r.weight for r in results if r.passed)
    weighted = wpass / wtot
    ok = weighted >= suite.threshold
    return RunReport(
        suite=suite.name,
        total=total,
        passed=passed,
        failed=failed,
        pass_rate=round(pass_rate, 4),
        weighted_pass_rate=round(weighted, 4),
        threshold=suite.threshold,
        ok=ok,
        cases=results,
    )


def compare_baseline(report: RunReport, baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Regression gate: flag cases that passed in baseline but now fail."""
    base_cases = {c["id"]: c["passed"] for c in baseline.get("cases", [])}
    regressions: List[str] = []
    fixes: List[str] = []
    for r in report.cases:
        was = base_cases.get(r.id)
        if was is True and not r.passed:
            regressions.append(r.id)
        elif was is False and r.passed:
            fixes.append(r.id)
    base_rate = float(baseline.get("weighted_pass_rate", baseline.get("pass_rate", 0.0)))
    delta = round(report.weighted_pass_rate - base_rate, 4)
    return {
        "baseline_pass_rate": base_rate,
        "current_pass_rate": report.weighted_pass_rate,
        "delta": delta,
        "regressions": regressions,
        "fixes": fixes,
        "ok": len(regressions) == 0,
    }
