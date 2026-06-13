"""Offline smoke tests for evalbench. No network."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evalbench import TOOL_NAME, TOOL_VERSION
from evalbench.cli import main
from evalbench.core import (
    EvalError,
    diff_runs,
    evaluate_case,
    evaluate_suite,
)

DEMO = os.path.join(
    os.path.dirname(__file__), "..", "demos", "01-basic", "support_suite.json"
)


def _case(output, *asserts):
    return evaluate_case({"id": "c", "output": output, "asserts": list(asserts)})


class TestAssertions(unittest.TestCase):
    def test_contains_ci(self):
        r = _case("well hello there",
                  {"type": "contains", "value": "HELLO", "ignore_case": True})
        self.assertTrue(r.passed)

    def test_not_contains(self):
        ok = _case("all good", {"type": "not-contains", "value": "error"})
        bad = _case("fatal error", {"type": "not-contains", "value": "error"})
        self.assertTrue(ok.passed)
        self.assertFalse(bad.passed)

    def test_word_count(self):
        self.assertTrue(_case("one two", {"type": "word-count", "max": 2}).passed)
        self.assertFalse(
            _case("one two three", {"type": "word-count", "max": 2}).passed)

    def test_json_valid_and_path(self):
        out = '{"action": "refund", "n": 3}'
        self.assertTrue(_case(out, {"type": "json-valid"}).passed)
        self.assertTrue(
            _case(out, {"type": "json-path", "path": "action",
                        "value": "refund"}).passed)
        self.assertFalse(
            _case(out, {"type": "json-path", "path": "missing"}).passed)

    def test_invalid_regex_rejected(self):
        with self.assertRaises(EvalError):
            _case("x", {"type": "regex", "value": "("})

    def test_similarity(self):
        spec = {"type": "similarity", "value": "hello world", "threshold": 0.8}
        self.assertTrue(_case("hello world", spec).passed)
        self.assertFalse(_case("totally different", spec).passed)

    def test_unknown_type_rejected(self):
        with self.assertRaises(EvalError):
            _case("x", {"type": "nope"})


class TestSuiteRun(unittest.TestCase):
    def test_demo_suite_passes(self):
        with open(DEMO, encoding="utf-8") as fh:
            run = evaluate_suite(json.load(fh))
        self.assertTrue(run.ok)
        self.assertEqual(run.total, 4)
        self.assertEqual(run.passed_cases, 4)

    def test_required_failure_fails_run(self):
        run = evaluate_suite({
            "name": "t",
            "cases": [
                {"id": "good", "output": "yes",
                 "asserts": [{"type": "contains", "value": "yes"}]},
                {"id": "bad", "output": "no",
                 "asserts": [{"type": "contains", "value": "yes"}]},
            ],
        })
        self.assertFalse(run.ok)
        self.assertEqual(run.passed_cases, 1)

    def test_regression_detected(self):
        suite = {"name": "r",
                 "cases": [{"id": "c1", "output": "bad",
                            "asserts": [{"type": "contains", "value": "good"}]}]}
        candidate = evaluate_suite(suite).to_dict()
        baseline = {"pass_rate": 1.0, "mean_score": 1.0,
                    "cases": [{"id": "c1", "passed": True, "score": 1.0}]}
        findings = diff_runs(baseline, candidate)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].case_id, "c1")
        self.assertEqual(findings[0].kind, "newly_failing")


class TestCli(unittest.TestCase):
    def test_version(self):
        self.assertEqual(TOOL_NAME, "evalbench")
        self.assertTrue(TOOL_VERSION)

    def test_run_demo_exit_zero(self):
        self.assertEqual(main(["run", DEMO, "--format", "json"]), 0)

    def test_bundled_demo_exit_zero(self):
        self.assertEqual(main(["demo", "--format", "json"]), 0)

    def test_types_cmd(self):
        self.assertEqual(main(["types", "--format", "json"]), 0)

    def test_missing_file_exit_two(self):
        self.assertEqual(main(["run", "does_not_exist.json"]), 2)


if __name__ == "__main__":
    unittest.main()
