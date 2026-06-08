"""EVALBENCH command-line interface.

Subcommands:
  run      <suite.json> [--baseline b.json] [--out report.json]
  list     <suite.json>          # list cases + assertion counts
  assertions                     # list supported assertion types

Exit codes: 0 = ok, 1 = gate failed (threshold or regression), 2 = usage/error.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import TOOL_NAME, TOOL_VERSION
from .core import ASSERTIONS, compare_baseline, load_suite, run_suite


def _print_table(report_dict: dict, comparison: Optional[dict]) -> None:
    r = report_dict
    print(f"suite: {r['suite']}")
    print(f"{'CASE':<24} {'RESULT':<7} {'WEIGHT':>6}  ASSERTIONS")
    print("-" * 64)
    for c in r["cases"]:
        res = "PASS" if c["passed"] else "FAIL"
        adesc = ", ".join(
            f"{a['type']}{'✓' if a['passed'] else '✗'}" for a in c["assertions"]
        ) or "(none)"
        print(f"{c['id']:<24} {res:<7} {c['weight']:>6.2f}  {adesc}")
    print("-" * 64)
    print(
        f"passed {r['passed']}/{r['total']}  "
        f"weighted={r['weighted_pass_rate']:.2%}  "
        f"threshold={r['threshold']:.2%}  "
        f"=> {'OK' if r['ok'] else 'BELOW THRESHOLD'}"
    )
    if comparison is not None:
        print(
            f"baseline {comparison['baseline_pass_rate']:.2%} -> "
            f"{comparison['current_pass_rate']:.2%}  "
            f"(delta {comparison['delta']:+.2%})"
        )
        if comparison["regressions"]:
            print("REGRESSIONS: " + ", ".join(comparison["regressions"]))
        if comparison["fixes"]:
            print("fixes: " + ", ".join(comparison["fixes"]))


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        suite = load_suite(args.suite)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    report = run_suite(suite)
    rd = report.to_dict()

    comparison = None
    if args.baseline:
        try:
            with open(args.baseline, "r", encoding="utf-8") as fh:
                baseline = json.load(fh)
        except (OSError, ValueError) as e:
            print(f"error: cannot read baseline: {e}", file=sys.stderr)
            return 2
        comparison = compare_baseline(report, baseline)
        rd["comparison"] = comparison

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(rd, fh, indent=2)
        except OSError as e:
            print(f"error: cannot write report: {e}", file=sys.stderr)
            return 2

    if args.format == "json":
        print(json.dumps(rd, indent=2))
    else:
        _print_table(rd, comparison)

    gate_ok = report.ok and (comparison is None or comparison["ok"])
    return 0 if gate_ok else 1


def _cmd_list(args: argparse.Namespace) -> int:
    try:
        suite = load_suite(args.suite)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    payload = {
        "suite": suite.name,
        "threshold": suite.threshold,
        "cases": [
            {"id": c.id, "weight": c.weight, "assertions": len(c.assertions)}
            for c in suite.cases
        ],
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(f"suite: {suite.name} (threshold {suite.threshold:.2%})")
        for c in payload["cases"]:
            print(f"  {c['id']:<24} weight={c['weight']:.2f} assertions={c['assertions']}")
    return 0


def _cmd_assertions(args: argparse.Namespace) -> int:
    names = sorted(ASSERTIONS)
    if args.format == "json":
        print(json.dumps({"assertions": names}, indent=2))
    else:
        print("supported assertion types:")
        for n in names:
            print(f"  {n}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description="Offline LLM/agent eval harness with regression gates.",
    )
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    p.add_argument(
        "--format", choices=["table", "json"], default="table",
        help="output format (default: table)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="run a suite and gate on threshold/regression")
    pr.add_argument("suite", help="path to suite JSON")
    pr.add_argument("--baseline", help="baseline report JSON for regression gate")
    pr.add_argument("--out", help="write full report JSON to this path")
    pr.set_defaults(func=_cmd_run)

    pl = sub.add_parser("list", help="list cases in a suite")
    pl.add_argument("suite", help="path to suite JSON")
    pl.set_defaults(func=_cmd_list)

    pa = sub.add_parser("assertions", help="list supported assertion types")
    pa.set_defaults(func=_cmd_assertions)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
