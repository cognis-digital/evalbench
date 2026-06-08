"""Offline smoke tests for evalbench. No network."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evalbench import TOOL_NAME, TOOL_VERSION
from evalbench.cli import main
from evalbench.core import (
    Assertion,
    Suite,
    compare_baseline,
    run_suite,
)

DEMO = os.path.join(
    os.path.dirname(__file__), "..", "demos", "01-basic", "support_suite.json"
)


class TestAssertions(unittest.TestCase):
    def test_contains_ci(self):
        a = Assertion(type="contains", value="HELLO", ignore_case=True)
        ok, _ = a.check("well hello there")
        self.assertTrue(ok)

    def test_not_contains(self):
        a = Assertion(type="not_contains", value="error")
        self.assertTrue(a.check("all good")[0])
        self.assertFalse(a.check("fatal error")[0])

    def test_max_tokens(self):
        a = Assertion(type="max_tokens", value=2)
        self.assertTrue(a.check("one two")[0])
        self.assertFalse(a.check("one two three")[0])

    def test_is_json_and_path(self):
        out = '{"action": "refund", "n": 3}'
        self.assertTrue(Assertion(type="is_json").check(out)[0])
        self.assertTrue(
            Assertion(type="json_path", value="action", expected="refund").check(out)[0]
        )
        self.assertFalse(
            Assertion(type="json_path", value="missing").check(out)[0]
        )

    def test_regex_invalid(self):
        ok, detail = Assertion(type="regex", value="(").check("x")
        self.assertFalse(ok)
        self.assertIn("invalid regex", detail)

    def test_similarity(self):
        a = Assertion(type="similarity", value="hello world", threshold=0.8)
        self.assertTrue(a.check("hello world")[0])
        self.assertFalse(a.check("totally different")[0])

    def test_unknown_type_rejected(self):
        with self.assertRaises(ValueError):
            Assertion.from_dict({"type": "nope"})


class TestSuiteRun(unittest.TestCase):
    def test_demo_suite_passes(self):
        with open(DEMO, encoding="utf-8") as fh:
            suite = Suite.from_dict(json.load(fh))
        report = run_suite(suite)
        self.assertEqual(report.failed, 0)
        self.assertTrue(report.ok)
        self.assertEqual(report.total, 4)

    def test_threshold_gate_fails(self):
        suite = Suite.from_dict({
            "name": "t",
            "threshold": 1.0,
            "cases": [
                {"id": "good", "output": "yes",
                 "assertions": [{"type": "contains", "value": "yes"}]},
                {"id": "bad", "output": "no",
                 "assertions": [{"type": "contains", "value": "yes"}]},
            ],
        })
        report = run_suite(suite)
        self.assertFalse(report.ok)
        self.assertEqual(report.passed, 1)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            Suite.from_dict({
                "name": "d",
                "cases": [
                    {"id": "x", "output": "a"},
                    {"id": "x", "output": "b"},
                ],
            })

    def test_regression_detected(self):
        suite = Suite.from_dict({
            "name": "r",
            "cases": [{"id": "c1", "output": "bad",
                       "assertions": [{"type": "contains", "value": "good"}]}],
        })
        report = run_suite(suite)
        baseline = {"weighted_pass_rate": 1.0,
                    "cases": [{"id": "c1", "passed": True}]}
        comp = compare_baseline(report, baseline)
        self.assertEqual(comp["regressions"], ["c1"])
        self.assertFalse(comp["ok"])


class TestCli(unittest.TestCase):
    def test_version(self):
        self.assertEqual(TOOL_NAME, "evalbench")
        self.assertTrue(TOOL_VERSION)

    def test_run_demo_exit_zero(self):
        rc = main(["run", DEMO, "--format", "json"])
        self.assertEqual(rc, 0)

    def test_assertions_cmd(self):
        rc = main(["--format", "json", "assertions"])
        self.assertEqual(rc, 0)

    def test_missing_file_exit_two(self):
        rc = main(["run", "does_not_exist.json"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
