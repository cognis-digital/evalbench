"""Deep tests for evalbench's eval harness: assertions + regression gate.

Covers: every assertion type, the mini JSON-Schema validator, json-path,
similarity/levenshtein metrics, suite evaluation + scoring/weighting, the
regression-gate diff (newly_failing / score_drop / missing / strict), the
bundled suite, and the CLI surface (run/demo/gate/types, table+json,
non-zero exit on findings). No network.
"""

import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evalbench import (  # noqa: E402
    TOOL_NAME,
    TOOL_VERSION,
    BUNDLED_SUITE,
    EvalError,
    evaluate_case,
    evaluate_suite,
    diff_runs,
    validate_schema,
    resolve_json_path,
    cosine_similarity,
    levenshtein,
    levenshtein_ratio,
)
from evalbench import cli  # noqa: E402

DEMO = os.path.join(ROOT, "demos", "02-deep", "evalbench")
BASELINE = os.path.join(DEMO, "baseline_suite.json")
CANDIDATE = os.path.join(DEMO, "candidate_suite.json")


def _one(spec, output, **metrics):
    case = {"id": "t", "output": output, "assert": [spec]}
    case.update(metrics)
    return evaluate_case(case).asserts[0]


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #

def test_metadata():
    assert TOOL_NAME == "evalbench"
    assert isinstance(TOOL_VERSION, str) and TOOL_VERSION.count(".") >= 1


# --------------------------------------------------------------------------- #
# Text metrics
# --------------------------------------------------------------------------- #

def test_cosine_similarity():
    assert abs(cosine_similarity("hello world", "hello world") - 1.0) < 1e-9
    assert cosine_similarity("a b c", "x y z") == 0.0
    mid = cosine_similarity("the quick brown fox", "the quick red fox")
    assert 0.5 < mid < 1.0


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("abc", "abc") == 0
    assert levenshtein_ratio("abc", "abc") == 1.0
    assert 0.0 < levenshtein_ratio("abc", "abd") < 1.0


# --------------------------------------------------------------------------- #
# JSON-Schema validator
# --------------------------------------------------------------------------- #

def test_schema_valid_and_invalid():
    schema = {
        "type": "object",
        "required": ["id", "n"],
        "properties": {
            "id": {"type": "string", "pattern": r"^A-\d+$"},
            "n": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "additionalProperties": False,
    }
    assert validate_schema({"id": "A-1", "n": 5}, schema) == []
    # bad pattern, out-of-range, missing required (n), extra prop
    errs = validate_schema({"id": "B-1", "x": 1}, schema)
    joined = " ".join(errs)
    assert "pattern" in joined
    assert "missing required" in joined
    assert "additional properties" in joined
    # out-of-range maximum
    assert "maximum" in " ".join(validate_schema({"id": "A-1", "n": 99}, schema))


def test_schema_bool_not_integer():
    # bool must not satisfy integer/number
    assert validate_schema(True, {"type": "integer"})


def test_schema_enum_and_array():
    schema = {"type": "array", "items": {"type": "string", "enum": ["a", "b"]},
              "minItems": 1}
    assert validate_schema(["a", "b"], schema) == []
    assert validate_schema(["a", "c"], schema)
    assert validate_schema([], schema)


def test_json_path():
    obj = {"a": {"b": [{"c": 7}]}}
    assert resolve_json_path(obj, "a.b[0].c") == (True, 7)
    assert resolve_json_path(obj, "a.b[5].c")[0] is False
    assert resolve_json_path(obj, "a.z")[0] is False


# --------------------------------------------------------------------------- #
# Every assertion type
# --------------------------------------------------------------------------- #

def test_string_asserts():
    assert _one({"type": "contains", "value": "cat"}, "a cat sat").passed
    assert not _one({"type": "contains", "value": "dog"}, "a cat sat").passed
    assert _one({"type": "icontains", "value": "CAT"}, "a cat sat").passed
    assert _one({"type": "not-contains", "value": "dog"}, "a cat").passed
    assert _one({"type": "equals", "value": "Hi", "normalize": True}, "  hi ").passed
    assert _one({"type": "starts-with", "value": "the"}, "the end").passed
    assert _one({"type": "ends-with", "value": "end"}, "the end").passed


def test_regex_asserts():
    assert _one({"type": "regex", "value": r"\d{3}-\d{4}"}, "call 555-1234").passed
    assert _one({"type": "not-regex", "value": r"\bidiot\b"}, "polite text").passed
    assert not _one({"type": "regex", "value": r"^\d+$"}, "abc").passed


def test_json_asserts():
    assert _one({"type": "json-valid"}, '{"a":1}').passed
    assert not _one({"type": "json-valid"}, "{not json").passed
    sa = _one({"type": "json-schema",
               "schema": {"type": "object", "required": ["a"]}}, '{"a":1}')
    assert sa.passed
    jp = _one({"type": "json-path", "path": "a", "value": 1}, '{"a":1}')
    assert jp.passed
    jp2 = _one({"type": "json-path", "path": "a", "value": 2}, '{"a":1}')
    assert not jp2.passed


def test_similarity_and_levenshtein_asserts():
    s = _one({"type": "similarity", "value": "hello world", "threshold": 0.9},
             "hello world")
    assert s.passed and s.score == 1.0
    lv = _one({"type": "levenshtein", "value": "color", "threshold": 0.7}, "colour")
    assert lv.passed
    s2 = _one({"type": "similarity", "value": "totally different phrase",
               "threshold": 0.9}, "nothing alike here")
    assert not s2.passed


def test_metric_asserts():
    assert _one({"type": "length", "min": 1, "max": 5}, "abc").passed
    assert not _one({"type": "length", "min": 10}, "abc").passed
    assert _one({"type": "word-count", "min": 2, "max": 2}, "two words").passed
    assert _one({"type": "latency", "max": 100}, "x", latency_ms=50).passed
    assert not _one({"type": "latency", "max": 100}, "x", latency_ms=200).passed
    assert _one({"type": "cost", "max": 0.01}, "x", cost_usd=0.005).passed


def test_composite_asserts():
    allof = _one({"type": "all-of", "asserts": [
        {"type": "contains", "value": "a"},
        {"type": "contains", "value": "b"},
    ]}, "a and b")
    assert allof.passed
    anyof = _one({"type": "any-of", "asserts": [
        {"type": "contains", "value": "x"},
        {"type": "contains", "value": "b"},
    ]}, "a and b")
    assert anyof.passed
    bad = _one({"type": "all-of", "asserts": [
        {"type": "contains", "value": "a"},
        {"type": "contains", "value": "z"},
    ]}, "a only")
    assert not bad.passed


def test_unknown_type_raises():
    try:
        _one({"type": "nope"}, "x")
        assert False, "expected EvalError"
    except EvalError:
        pass


# --------------------------------------------------------------------------- #
# Scoring / weighting / advisory
# --------------------------------------------------------------------------- #

def test_advisory_assert_does_not_fail_case():
    case = {"id": "c", "output": "hello", "assert": [
        {"type": "contains", "value": "hello"},
        {"type": "contains", "value": "MISSING", "required": False},
    ]}
    res = evaluate_case(case)
    assert res.passed  # advisory failure does not sink the case
    assert res.score < 1.0  # but it still drags the score down


def test_weighting_affects_score():
    case = {"id": "c", "output": "hello", "assert": [
        {"type": "contains", "value": "hello", "weight": 3.0},
        {"type": "contains", "value": "X", "weight": 1.0, "required": False},
    ]}
    res = evaluate_case(case)
    # weighted: (3*1 + 1*0) / 4 = 0.75
    assert abs(res.score - 0.75) < 1e-6


# --------------------------------------------------------------------------- #
# Bundled suite
# --------------------------------------------------------------------------- #

def test_bundled_suite_all_pass():
    run = evaluate_suite(BUNDLED_SUITE)
    assert run.total == 4
    assert run.ok
    assert run.pass_rate == 1.0


# --------------------------------------------------------------------------- #
# Regression gate
# --------------------------------------------------------------------------- #

def test_diff_detects_newly_failing():
    base = evaluate_suite(json.load(open(BASELINE, encoding="utf-8"))).to_dict()
    cand = evaluate_suite(json.load(open(CANDIDATE, encoding="utf-8"))).to_dict()
    findings = diff_runs(base, cand)
    kinds = {f.kind for f in findings}
    ids = {f.case_id for f in findings}
    assert "newly_failing" in kinds
    assert "json-weather" in ids


def test_diff_clean_when_identical():
    base = evaluate_suite(BUNDLED_SUITE).to_dict()
    assert diff_runs(base, base) == []


def test_diff_missing_case():
    base = {"cases": [{"id": "x", "passed": True, "score": 1.0, "asserts": []}],
            "pass_rate": 1.0, "mean_score": 1.0}
    cand = {"cases": [], "pass_rate": 0.0, "mean_score": 0.0}
    findings = diff_runs(base, cand)
    assert any(f.kind == "missing" for f in findings)


def test_diff_score_drop_tolerance():
    base = {"cases": [{"id": "x", "passed": True, "score": 1.0, "asserts": []}]}
    cand = {"cases": [{"id": "x", "passed": True, "score": 0.8, "asserts": []}]}
    assert any(f.kind == "score_drop" for f in diff_runs(base, cand, tolerance=0.0))
    assert diff_runs(base, cand, tolerance=0.3) == []


def test_diff_strict_run_regression():
    base = {"cases": [], "pass_rate": 1.0, "mean_score": 1.0}
    cand = {"cases": [], "pass_rate": 0.5, "mean_score": 0.5}
    findings = diff_runs(base, cand, strict=True)
    assert any(f.kind == "run_regression" for f in findings)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _capture(argv):
    out, err = io.StringIO(), io.StringIO()
    old_o, old_e = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        code = cli.main(argv)
    finally:
        sys.stdout, sys.stderr = old_o, old_e
    return code, out.getvalue(), err.getvalue()


def test_cli_version_and_types():
    code, out, _ = _capture(["types", "--format", "json"])
    assert code == 0
    data = json.loads(out)
    assert data["tool"] == "evalbench"
    assert len(data["assertion_types"]) >= 15


def test_cli_demo_passes():
    code, out, _ = _capture(["demo", "--format", "json"])
    assert code == 0
    data = json.loads(out)
    assert data["ok"] is True
    assert data["total"] == 4


def test_cli_run_table():
    code, out, _ = _capture(["run", BASELINE])
    assert code == 0
    assert "pass_rate" in out


def test_cli_gate_detects_regression_nonzero_exit(tmp_path):
    code, out, _ = _capture(["gate", BASELINE, CANDIDATE, "--format", "json"])
    assert code == 1  # non-zero exit on findings
    data = json.loads(out)
    assert data["passed"] is False
    assert any(f["case_id"] == "json-weather" for f in data["findings"])


def test_cli_gate_clean_zero_exit(tmp_path):
    # gating a suite against itself => no regressions
    code, out, _ = _capture(["gate", BASELINE, BASELINE, "--format", "json"])
    assert code == 0
    assert json.loads(out)["passed"] is True


def test_cli_save_roundtrip(tmp_path):
    saved = tmp_path / "run.json"
    code, _, _ = _capture(["run", BASELINE, "--save", str(saved), "--format", "json"])
    assert code == 0
    doc = json.loads(saved.read_text(encoding="utf-8"))
    assert doc["tool"] == "evalbench"
    assert doc["total"] == 3
