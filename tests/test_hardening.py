"""Hardening tests: error paths, edge cases, and bad-input handling.

Covers:
  - Missing / unreadable input files -> exit 2 with clear message.
  - Malformed JSON input -> exit 2 with clear message.
  - Negative --tolerance -> exit 2.
  - evaluate_suite with non-list cases / non-dict case items -> EvalError.
  - output_file pointing to a missing path -> EvalError with clear message.
  - mcp_server module imports without crashing.
  - Edge cases: empty output string, zero-division guards in scoring.
"""
from __future__ import annotations

import io
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:  # noqa: E402
    sys.path.insert(0, ROOT)

from evalbench.cli import main  # noqa: E402
from evalbench.core import EvalError, evaluate_case, evaluate_suite  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _capture(argv):
    out, err = io.StringIO(), io.StringIO()
    old_o, old_e = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        code = main(argv)
    finally:
        sys.stdout, sys.stderr = old_o, old_e
    return code, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- #
# CLI: bad file paths
# --------------------------------------------------------------------------- #

class TestCliFileErrors(unittest.TestCase):

    def test_missing_suite_file_returns_exit_2(self):
        code, _out, err = _capture(["run", "/no/such/file.json"])
        self.assertEqual(code, 2)
        self.assertIn("evalbench", err)

    def test_malformed_json_returns_exit_2(self, tmp_path=None):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                        delete=False, encoding="utf-8") as fh:
            fh.write("{not valid json!!!")
            path = fh.name
        try:
            code, _out, err = _capture(["run", path])
            self.assertEqual(code, 2)
            self.assertIn("evalbench", err)
        finally:
            os.unlink(path)

    def test_gate_missing_baseline_returns_exit_2(self):
        code, _out, err = _capture(["gate", "/no/baseline.json", "/no/candidate.json"])
        self.assertEqual(code, 2)
        self.assertIn("evalbench", err)


# --------------------------------------------------------------------------- #
# CLI: negative tolerance
# --------------------------------------------------------------------------- #

class TestCliToleranceValidation(unittest.TestCase):

    def test_negative_tolerance_returns_exit_2(self):
        # Use the bundled demo files for baseline/candidate — they exist.
        demo = os.path.join(ROOT, "demos", "02-deep", "evalbench", "baseline_suite.json")
        code, _out, err = _capture(["gate", demo, demo, "--tolerance", "-0.5"])
        self.assertEqual(code, 2)
        self.assertIn("tolerance", err)


# --------------------------------------------------------------------------- #
# Core: evaluate_suite guards
# --------------------------------------------------------------------------- #

class TestEvaluateSuiteGuards(unittest.TestCase):

    def test_cases_not_a_list_raises_eval_error(self):
        with self.assertRaises(EvalError) as ctx:
            evaluate_suite({"cases": "not-a-list"})
        self.assertIn("list", str(ctx.exception))

    def test_case_item_not_a_dict_raises_eval_error(self):
        with self.assertRaises(EvalError) as ctx:
            evaluate_suite({"cases": ["not-a-dict"]})
        self.assertIn("cases[0]", str(ctx.exception))

    def test_suite_not_a_dict_raises_eval_error(self):
        with self.assertRaises(EvalError) as ctx:
            evaluate_suite(["list", "not", "dict"])
        self.assertIn("JSON object", str(ctx.exception))

    def test_empty_cases_raises_eval_error(self):
        with self.assertRaises(EvalError):
            evaluate_suite({"cases": []})


# --------------------------------------------------------------------------- #
# Core: output_file path errors
# --------------------------------------------------------------------------- #

class TestOutputFileErrors(unittest.TestCase):

    def test_missing_output_file_raises_eval_error(self):
        case = {
            "id": "t",
            "output_file": "/no/such/file.txt",
            "assert": [{"type": "contains", "value": "x"}],
        }
        with self.assertRaises(EvalError) as ctx:
            evaluate_case(case)
        self.assertIn("not found", str(ctx.exception))
        self.assertIn("t", str(ctx.exception))

    def test_empty_output_file_path_raises_eval_error(self):
        case = {
            "id": "t2",
            "output_file": "",
            "assert": [{"type": "contains", "value": "x"}],
        }
        with self.assertRaises(EvalError) as ctx:
            evaluate_case(case)
        self.assertIn("non-empty string", str(ctx.exception))


# --------------------------------------------------------------------------- #
# Core: edge cases in evaluation
# --------------------------------------------------------------------------- #

class TestEvaluationEdgeCases(unittest.TestCase):

    def test_empty_output_string_does_not_crash(self):
        case = {
            "id": "empty",
            "output": "",
            "assert": [
                {"type": "contains", "value": "x"},
                {"type": "length", "min": 0, "max": 100},
                {"type": "word-count", "min": 0, "max": 5},
            ],
        }
        result = evaluate_case(case)
        # contains "x" fails, length passes, word-count passes
        self.assertFalse(result.passed)
        self.assertIsInstance(result.score, float)

    def test_all_advisory_assertions_do_not_zero_divide(self):
        """If all assertions are advisory, total_weight is still >0 (weight defaults to 1.0)."""
        case = {
            "id": "adv",
            "output": "hello",
            "assert": [
                {"type": "contains", "value": "MISSING", "required": False},
            ],
        }
        result = evaluate_case(case)
        self.assertTrue(result.passed)  # no required assertions failed
        self.assertGreaterEqual(result.score, 0.0)


# --------------------------------------------------------------------------- #
# mcp_server: module imports without crashing
# --------------------------------------------------------------------------- #

class TestMcpServerImport(unittest.TestCase):

    def test_mcp_server_imports_cleanly(self):
        import importlib
        mod = importlib.import_module("evalbench.mcp_server")
        self.assertTrue(callable(mod.serve))


if __name__ == "__main__":
    unittest.main()
