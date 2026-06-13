"""EVALBENCH command-line interface.

Subcommands:
  run     Evaluate a suite (JSON) and emit a run result; non-zero exit on case failure.
  gate    Compare a candidate run against a baseline run; non-zero exit on regressions.
  types   List supported assertion types.
  demo    Run the bundled golden-set suite (no files needed).

Exit codes: 0 success, 1 findings (case failure / regression), 2 usage/IO error.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import core


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _emit(text: str, output: str | None) -> None:
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"[{core.TOOL_NAME}] wrote {output}", file=sys.stderr)
    else:
        print(text)


def _run_suite(suite: dict, args) -> int:
    run = core.evaluate_suite(suite)
    if args.save:
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(run.to_dict(), fh, indent=2)
        print(f"[{core.TOOL_NAME}] saved run to {args.save}", file=sys.stderr)

    if args.format == "json":
        _emit(json.dumps(run.to_dict(), indent=2), args.output)
    else:
        _emit(core.render_run_table(run), args.output)
    return 0 if run.ok else 1


def _cmd_run(args) -> int:
    suite = _load_json(args.suite)
    return _run_suite(suite, args)


def _cmd_demo(args) -> int:
    return _run_suite(core.BUNDLED_SUITE, args)


def _as_run(doc: dict) -> dict:
    """Accept either a saved run-result JSON or a raw suite.

    A suite has cases carrying assertion specs (``assert``/``asserts``); a saved
    run result has cases carrying outcome fields (``passed``/``score``). When a
    suite is given, evaluate it on the fly so the gate always sees run results.
    """
    cases = doc.get("cases") or []
    if cases and ("assert" in cases[0] or "asserts" in cases[0]):
        return core.evaluate_suite(doc).to_dict()
    return doc


def _cmd_gate(args) -> int:
    baseline = _as_run(_load_json(args.baseline))
    candidate = _as_run(_load_json(args.candidate))

    findings = core.diff_runs(baseline, candidate,
                              tolerance=args.tolerance, strict=args.strict)

    if args.format == "json":
        _emit(json.dumps({
            "tool": core.TOOL_NAME,
            "passed": not findings,
            "tolerance": args.tolerance,
            "strict": args.strict,
            "findings": [f.to_dict() for f in findings],
        }, indent=2), args.output)
    else:
        _emit(core.render_diff_table(findings, args.tolerance), args.output)
    return 1 if findings else 0


def _cmd_types(args) -> int:
    types = [
        ("contains", "substring present (ignore_case opt)"),
        ("icontains", "case-insensitive contains"),
        ("not-contains", "substring must be absent"),
        ("equals", "exact match (normalize/ignore_case opt)"),
        ("regex", "re.search must match"),
        ("not-regex", "re.search must NOT match"),
        ("starts-with", "output startswith value"),
        ("ends-with", "output endswith value"),
        ("json-valid", "output parses as JSON"),
        ("json-schema", "validate against mini JSON-Schema"),
        ("json-path", "dotted path resolves (== value opt)"),
        ("similarity", "TF-cosine >= threshold"),
        ("levenshtein", "edit-distance ratio >= threshold"),
        ("length", "char length within [min,max]"),
        ("word-count", "token count within [min,max]"),
        ("latency", "latency_ms <= max"),
        ("cost", "cost_usd <= max"),
        ("all-of", "all sub-asserts pass"),
        ("any-of", "any sub-assert passes"),
    ]
    if args.format == "json":
        print(json.dumps({"tool": core.TOOL_NAME, "version": core.TOOL_VERSION,
                          "assertion_types": [{"type": t, "desc": d} for t, d in types]},
                         indent=2))
        return 0
    print(f"{core.TOOL_NAME} v{core.TOOL_VERSION} — assertion types\n")
    for t, d in types:
        print(f"  {t:<16}{d}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evalbench",
        description=f"{core.TOOL_NAME} — offline eval harness with a regression gate.",
    )
    p.add_argument("--version", action="version",
                   version=f"{core.TOOL_NAME} {core.TOOL_VERSION}")
    sub = p.add_subparsers(dest="command")

    def add_fmt(sp):
        sp.add_argument("--format", choices=["table", "json"], default="table")
        sp.add_argument("--output", help="write report to this file")

    run = sub.add_parser("run", help="evaluate a suite (JSON)")
    run.add_argument("suite", help="path to a suite JSON file")
    run.add_argument("--save", help="save the run result JSON (for later gating)")
    add_fmt(run)
    run.set_defaults(func=_cmd_run)

    demo = sub.add_parser("demo", help="run the bundled golden-set suite")
    demo.add_argument("--save", help="save the run result JSON")
    add_fmt(demo)
    demo.set_defaults(func=_cmd_demo)

    gate = sub.add_parser("gate", help="compare candidate vs baseline run")
    gate.add_argument("baseline", help="baseline run-result JSON (or suite)")
    gate.add_argument("candidate", help="candidate run-result JSON (or suite)")
    gate.add_argument("--tolerance", type=float, default=0.0,
                      help="max allowed score drop before flagging (default 0.0)")
    gate.add_argument("--strict", action="store_true",
                      help="also flag overall pass_rate / mean_score regressions")
    add_fmt(gate)
    gate.set_defaults(func=_cmd_gate)

    types = sub.add_parser("types", help="list supported assertion types")
    types.add_argument("--format", choices=["table", "json"], default="table")
    types.set_defaults(func=_cmd_types)

    # "assertions" is an alias for "types" (backward-compat)
    assertions = sub.add_parser("assertions", help="list supported assertion types (alias for types)")
    assertions.add_argument("--format", choices=["table", "json"], default="table")
    assertions.set_defaults(func=_cmd_types)

    return p


def _reorder_global_flags(argv: list[str]) -> list[str]:
    """Move global flags (--format, --output) that appear before a subcommand to after it.

    Allows: evalbench --format json assertions  (as well as the normal form)
    """
    _SUBCOMMANDS = {"run", "demo", "gate", "types", "assertions"}
    # find position of first recognised subcommand
    sub_pos = None
    for i, arg in enumerate(argv):
        if arg in _SUBCOMMANDS:
            sub_pos = i
            break
    if sub_pos is None or sub_pos == 0:
        return argv  # nothing to reorder

    pre = argv[:sub_pos]   # flags before subcommand
    rest = argv[sub_pos:]  # subcommand + its args

    # extract --format and --output with values from pre-subcommand portion
    hoisted: list[str] = []
    remaining_pre: list[str] = []
    i = 0
    while i < len(pre):
        tok = pre[i]
        if tok in ("--format", "--output") and i + 1 < len(pre):
            hoisted.extend([tok, pre[i + 1]])
            i += 2
        else:
            remaining_pre.append(tok)
            i += 1

    return remaining_pre + rest + hoisted


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = _reorder_global_flags(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"[{core.TOOL_NAME}] error: {exc}", file=sys.stderr)
        return 2
    except (core.EvalError, json.JSONDecodeError) as exc:
        print(f"[{core.TOOL_NAME}] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
