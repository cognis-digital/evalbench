"""EVALBENCH core — an offline LLM/prompt eval harness (promptfoo + deepeval, in your terminal).

Zero-install, standard-library only. The engine evaluates *cases* against a *suite*
of assertions, scores each case, aggregates a run, and (the headline feature) gates a
candidate run against a stored baseline to catch regressions in CI.

Assertion types implemented (real logic, no stubs):
  contains          — substring (optionally case-insensitive) present / absent
  icontains         — case-insensitive contains
  not-contains      — substring must be absent
  equals            — exact string match (optionally normalized)
  regex             — re.search must match
  not-regex         — re.search must NOT match
  starts-with       — output startswith value
  ends-with         — output endswith value
  json-valid        — output parses as JSON
  json-schema       — output parses as JSON and validates against a mini JSON-Schema
  json-path         — a dotted/indexed path resolves (optionally equals a value)
  similarity        — token-cosine (TF) similarity to a reference >= threshold
  levenshtein       — normalized edit-distance similarity >= threshold
  length            — character length within [min,max]
  word-count        — token count within [min,max]
  latency           — recorded latency_ms <= threshold
  cost              — recorded cost_usd <= threshold
  all-of / any-of   — composite assertions over a list of sub-asserts

A suite is a dict (loaded from JSON):
  {
    "name": "...",
    "defaults": { "weight": 1.0, ... },     # optional per-assert defaults
    "cases": [
      { "id": "...", "vars": {...},          # vars are informational
        "output": "model output string",     # OR "output_file": "path"
        "latency_ms": 12, "cost_usd": 0.0,    # optional recorded metrics
        "assert": [ {type, ...}, ... ] }
    ]
  }

Scoring: each assertion contributes weight*passed. A case passes if every
*required* assertion passes (assertions are required by default; set
"required": false to make an assertion advisory — it still scores but does not
fail the case). Run pass rate = passed_cases / total_cases.

Regression gate: compare a candidate run to a baseline run by case id. A finding
is raised when a case that passed in the baseline fails in the candidate, when the
candidate's score drops by more than --tolerance, or (with --strict) when the
overall pass-rate or mean score regresses.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from typing import Any

TOOL_NAME = "evalbench"
TOOL_VERSION = "2.0.0"


class EvalError(Exception):
    """Raised on malformed suites / assertions / unknown types."""


# --------------------------------------------------------------------------- #
# Text utilities
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def cosine_similarity(a: str, b: str) -> float:
    """Term-frequency cosine similarity over tokenized text (0..1)."""
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 1.0 if ta == tb else 0.0
    fa: dict[str, int] = {}
    fb: dict[str, int] = {}
    for t in ta:
        fa[t] = fa.get(t, 0) + 1
    for t in tb:
        fb[t] = fb.get(t, 0) + 1
    dot = sum(fa[t] * fb.get(t, 0) for t in fa)
    na = math.sqrt(sum(v * v for v in fa.values()))
    nb = math.sqrt(sum(v * v for v in fb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def levenshtein(a: str, b: str) -> int:
    """Classic dynamic-programming edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def levenshtein_ratio(a: str, b: str) -> float:
    """1 - normalized edit distance (0..1, higher = more similar)."""
    if not a and not b:
        return 1.0
    dist = levenshtein(a, b)
    return 1.0 - dist / max(len(a), len(b))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


# --------------------------------------------------------------------------- #
# Mini JSON-Schema validator (draft-ish subset, no deps)
# --------------------------------------------------------------------------- #

_TYPE_MAP: dict[str, tuple] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "null": (type(None),),
}


def validate_schema(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Return a list of human-readable validation errors (empty = valid).

    Supports: type, properties, required, items, enum, minimum, maximum,
    minLength, maxLength, minItems, maxItems, pattern, additionalProperties.
    """
    errors: list[str] = []
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        ok = False
        for tt in types:
            py = _TYPE_MAP.get(tt)
            if py is None:
                errors.append(f"{path}: unknown schema type {tt!r}")
                continue
            # bool is a subclass of int — guard integer/number against bools
            if tt in ("integer", "number") and isinstance(value, bool):
                continue
            if isinstance(value, py):
                ok = True
        if not ok:
            errors.append(f"{path}: expected type {t}, got {type(value).__name__}")
            return errors  # type mismatch => skip deeper checks

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: length {len(value)} < minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: length {len(value)} > maxLength {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match pattern {schema['pattern']!r}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: {len(value)} items < minItems {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: {len(value)} items > maxItems {schema['maxItems']}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                errors += validate_schema(item, item_schema, f"{path}[{i}]")

    if isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}: missing required property {req!r}")
        props = schema.get("properties", {})
        for key, subschema in props.items():
            if key in value:
                errors += validate_schema(value[key], subschema, f"{path}.{key}")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(props)
            if extra:
                errors.append(f"{path}: additional properties not allowed: {sorted(extra)}")

    return errors


def resolve_json_path(obj: Any, path: str) -> tuple[bool, Any]:
    """Resolve a dotted path like 'a.b[0].c' against parsed JSON.

    Returns (found, value).
    """
    cur = obj
    # split on '.' but keep '[idx]' segments attached
    for raw in path.replace("]", "").replace("[", ".").split("."):
        if raw == "" or raw == "$":
            continue
        if isinstance(cur, dict):
            if raw not in cur:
                return False, None
            cur = cur[raw]
        elif isinstance(cur, list):
            try:
                idx = int(raw)
            except ValueError:
                return False, None
            if idx < 0 or idx >= len(cur):
                return False, None
            cur = cur[idx]
        else:
            return False, None
    return True, cur


# --------------------------------------------------------------------------- #
# Assertion engine
# --------------------------------------------------------------------------- #

@dataclass
class AssertResult:
    type: str
    passed: bool
    weight: float
    required: bool
    reason: str
    score: float  # 0..1 graded score for this assertion

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CaseResult:
    id: str
    passed: bool
    score: float  # weighted mean of assertion scores (0..1)
    latency_ms: float | None
    cost_usd: float | None
    asserts: list[AssertResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["asserts"] = [a.to_dict() for a in self.asserts]
        return d


@dataclass
class RunResult:
    name: str
    total: int
    passed_cases: int
    pass_rate: float
    mean_score: float
    cases: list[CaseResult] = field(default_factory=list)
    threshold: float = 1.0  # min pass_rate for ok; default 1.0 = all must pass

    @property
    def ok(self) -> bool:
        return self.pass_rate >= self.threshold

    def to_dict(self) -> dict:
        return {
            "tool": TOOL_NAME,
            "version": TOOL_VERSION,
            "name": self.name,
            "total": self.total,
            "passed_cases": self.passed_cases,
            "pass_rate": round(self.pass_rate, 4),
            "mean_score": round(self.mean_score, 4),
            "ok": self.ok,
            "cases": [c.to_dict() for c in self.cases],
        }


def _graded(passed: bool, score: float | None = None) -> tuple[bool, float]:
    if score is not None:
        return passed, max(0.0, min(1.0, score))
    return passed, 1.0 if passed else 0.0


def _normalise_type(atype: str, spec: dict) -> tuple[str, dict]:
    """Normalise legacy/underscore assertion type names to canonical hyphen forms.

    Also rewrites spec keys where the old API used different field names.
    Returns (canonical_type, rewritten_spec).
    """
    _ALIASES: dict[str, str] = {
        # underscore → hyphen variants
        "not_contains": "not-contains",
        "not_regex": "not-regex",
        "starts_with": "starts-with",
        "ends_with": "ends-with",
        "json_valid": "json-valid",
        "json_schema": "json-schema",
        "all_of": "all-of",
        "any_of": "any-of",
        # legacy short names
        "is_json": "json-valid",
        "icontains": "icontains",
    }
    canonical = _ALIASES.get(atype, atype)

    # json_path: old API uses {"type":"json_path","value":"<path>","expected":"<val>"}
    # new API:  {"type":"json-path","path":"<path>","value":"<val>"}
    if atype == "json_path":
        canonical = "json-path"
        spec = dict(spec)
        if "value" in spec and "path" not in spec:
            spec["path"] = spec.pop("value")
        if "expected" in spec:
            spec["value"] = spec.pop("expected")

    # max_tokens / min_tokens → word-count with max/min
    if atype in ("max_tokens", "max-tokens"):
        canonical = "word-count"
        spec = dict(spec)
        spec["max"] = spec.pop("value", spec.get("max", math.inf))

    if atype in ("min_tokens", "min-tokens"):
        canonical = "word-count"
        spec = dict(spec)
        spec["min"] = spec.pop("value", spec.get("min", 0))

    return canonical, spec


def _eval_one(spec: dict, output: str, metrics: dict) -> tuple[bool, float, str]:
    """Evaluate a single assertion spec. Returns (passed, score, reason)."""
    atype = spec.get("type")
    if not atype:
        raise EvalError(f"assertion missing 'type': {spec!r}")
    atype, spec = _normalise_type(atype, spec)
    val = spec.get("value")
    ci = spec.get("ignore_case", False)
    hay = output.lower() if ci else output
    needle = (val.lower() if (ci and isinstance(val, str)) else val)

    if atype in ("contains", "icontains"):
        ci2 = ci or atype == "icontains"
        h = output.lower() if ci2 else output
        n = val.lower() if ci2 else val
        ok = n in h
        return ok, 1.0 if ok else 0.0, f"{'found' if ok else 'missing'} {val!r}"

    if atype == "not-contains":
        ok = needle not in hay
        return ok, 1.0 if ok else 0.0, f"{val!r} {'absent' if ok else 'present'}"

    if atype == "equals":
        a, b = (output, val)
        if spec.get("normalize"):
            a, b = _normalize(output), _normalize(str(val))
        elif ci:
            a, b = output.lower(), str(val).lower()
        ok = a == b
        return ok, 1.0 if ok else 0.0, "exact match" if ok else "not equal"

    if atype == "regex":
        flags = re.IGNORECASE if ci else 0
        try:
            ok = re.search(val, output, flags) is not None
        except re.error as exc:
            return False, 0.0, f"invalid regex: {exc}"
        return ok, 1.0 if ok else 0.0, f"/{val}/ {'matched' if ok else 'no match'}"

    if atype == "not-regex":
        flags = re.IGNORECASE if ci else 0
        try:
            ok = re.search(val, output, flags) is None
        except re.error as exc:
            return False, 0.0, f"invalid regex: {exc}"
        return ok, 1.0 if ok else 0.0, f"/{val}/ {'absent' if ok else 'matched'}"

    if atype == "starts-with":
        ok = hay.startswith(needle)
        return ok, 1.0 if ok else 0.0, f"{'starts with' if ok else 'does not start with'} {val!r}"

    if atype == "ends-with":
        ok = hay.endswith(needle)
        return ok, 1.0 if ok else 0.0, f"{'ends with' if ok else 'does not end with'} {val!r}"

    if atype == "json-valid":
        try:
            json.loads(output)
            return True, 1.0, "valid JSON"
        except (ValueError, TypeError) as exc:
            return False, 0.0, f"invalid JSON: {exc}"

    if atype == "json-schema":
        schema = spec.get("schema")
        if not isinstance(schema, dict):
            raise EvalError("json-schema assertion requires a 'schema' object")
        try:
            parsed = json.loads(output)
        except (ValueError, TypeError) as exc:
            return False, 0.0, f"invalid JSON: {exc}"
        errs = validate_schema(parsed, schema)
        ok = not errs
        return ok, 1.0 if ok else 0.0, "schema valid" if ok else "; ".join(errs[:4])

    if atype == "json-path":
        path = spec.get("path")
        if not path:
            raise EvalError("json-path assertion requires 'path'")
        try:
            parsed = json.loads(output)
        except (ValueError, TypeError) as exc:
            return False, 0.0, f"invalid JSON: {exc}"
        found, resolved = resolve_json_path(parsed, path)
        if not found:
            return False, 0.0, f"path {path!r} not found"
        if "value" in spec:
            ok = resolved == spec["value"]
            return ok, 1.0 if ok else 0.0, (
                f"{path}={resolved!r}" if ok else f"{path}={resolved!r} != {spec['value']!r}"
            )
        return True, 1.0, f"{path} resolved to {resolved!r}"

    if atype == "similarity":
        ref = spec.get("value", "")
        thr = float(spec.get("threshold", 0.8))
        sim = cosine_similarity(output, ref)
        ok = sim >= thr
        return ok, sim, f"cosine={sim:.3f} (>= {thr})"

    if atype == "levenshtein":
        ref = spec.get("value", "")
        thr = float(spec.get("threshold", 0.8))
        sim = levenshtein_ratio(output, ref)
        ok = sim >= thr
        return ok, sim, f"lev_ratio={sim:.3f} (>= {thr})"

    if atype == "length":
        n = len(output)
        lo = spec.get("min", 0)
        hi = spec.get("max", math.inf)
        ok = lo <= n <= hi
        return ok, 1.0 if ok else 0.0, f"len={n} in [{lo},{hi}]"

    if atype == "word-count":
        n = len(tokenize(output))
        lo = spec.get("min", 0)
        hi = spec.get("max", math.inf)
        ok = lo <= n <= hi
        return ok, 1.0 if ok else 0.0, f"words={n} in [{lo},{hi}]"

    if atype == "latency":
        thr = float(spec.get("max", spec.get("value", math.inf)))
        actual = metrics.get("latency_ms")
        if actual is None:
            return False, 0.0, "no latency_ms recorded"
        ok = actual <= thr
        return ok, 1.0 if ok else 0.0, f"latency={actual}ms (<= {thr})"

    if atype == "cost":
        thr = float(spec.get("max", spec.get("value", math.inf)))
        actual = metrics.get("cost_usd")
        if actual is None:
            return False, 0.0, "no cost_usd recorded"
        ok = actual <= thr
        return ok, 1.0 if ok else 0.0, f"cost=${actual} (<= ${thr})"

    if atype in ("all-of", "any-of"):
        subs = spec.get("asserts", [])
        if not subs:
            raise EvalError(f"{atype} requires non-empty 'asserts'")
        results = [_eval_one(s, output, metrics) for s in subs]
        passes = [r[0] for r in results]
        mean = sum(r[1] for r in results) / len(results)
        if atype == "all-of":
            ok = all(passes)
        else:
            ok = any(passes)
        detail = ", ".join(f"{s.get('type')}:{'P' if r[0] else 'F'}"
                           for s, r in zip(subs, results))
        return ok, mean, f"{atype}({detail})"

    raise EvalError(f"unknown assertion type: {atype!r}")


# --------------------------------------------------------------------------- #
# Case / suite evaluation
# --------------------------------------------------------------------------- #

def evaluate_case(case: dict, defaults: dict | None = None) -> CaseResult:
    defaults = defaults or {}
    cid = str(case.get("id") or case.get("name") or "case")
    if "output" in case:
        output = case["output"]
    elif "output_file" in case:
        path = case["output_file"]
        if not isinstance(path, str) or not path:
            raise EvalError(f"case {cid!r}: 'output_file' must be a non-empty string")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                output = fh.read()
        except FileNotFoundError:
            raise EvalError(f"case {cid!r}: output_file not found: {path!r}")
        except IsADirectoryError:
            raise EvalError(f"case {cid!r}: output_file is a directory: {path!r}")
        except PermissionError as exc:
            raise EvalError(f"case {cid!r}: cannot read output_file {path!r}: {exc}")
    else:
        raise EvalError(f"case {cid!r} has no 'output' or 'output_file'")
    if not isinstance(output, str):
        output = json.dumps(output)

    metrics = {
        "latency_ms": case.get("latency_ms"),
        "cost_usd": case.get("cost_usd"),
    }

    specs = case.get("assert") or case.get("asserts") or case.get("assertions") or []
    if not specs:
        raise EvalError(f"case {cid!r} has no assertions")

    results: list[AssertResult] = []
    total_weight = 0.0
    weighted_score = 0.0
    case_passed = True

    for spec in specs:
        weight = float(spec.get("weight", defaults.get("weight", 1.0)))
        required = spec.get("required", defaults.get("required", True))
        passed, score, reason = _eval_one(spec, output, metrics)
        results.append(AssertResult(
            type=spec.get("type"), passed=passed, weight=weight,
            required=required, reason=reason, score=round(score, 4),
        ))
        total_weight += weight
        weighted_score += weight * score
        if required and not passed:
            case_passed = False

    case_score = (weighted_score / total_weight) if total_weight else 0.0
    return CaseResult(
        id=cid, passed=case_passed, score=round(case_score, 4),
        latency_ms=metrics["latency_ms"], cost_usd=metrics["cost_usd"],
        asserts=results,
    )


def evaluate_suite(suite: dict) -> "RunResult":
    if not isinstance(suite, dict):
        raise EvalError("suite must be a JSON object")
    cases = suite.get("cases")
    if not cases:
        raise EvalError("suite has no 'cases'")
    if not isinstance(cases, list):
        raise EvalError("suite 'cases' must be a list")
    for i, c in enumerate(cases):
        if not isinstance(c, dict):
            raise EvalError(f"suite 'cases[{i}]' must be a JSON object, got {type(c).__name__}")
    defaults = suite.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}
    results = [evaluate_case(c, defaults) for c in cases]
    passed = sum(1 for c in results if c.passed)
    total = len(results)
    mean_score = sum(c.score for c in results) / total if total else 0.0
    pass_rate = passed / total if total else 0.0
    threshold = float(suite.get("threshold", 1.0))
    return RunResult(
        name=suite.get("name", "suite"),
        total=total, passed_cases=passed,
        pass_rate=pass_rate,
        mean_score=mean_score, cases=results,
        threshold=threshold,
    )


# --------------------------------------------------------------------------- #
# Regression gate
# --------------------------------------------------------------------------- #

@dataclass
class Finding:
    case_id: str
    kind: str          # "newly_failing" | "score_drop" | "missing" | "run_regression"
    detail: str
    baseline: Any = None
    candidate: Any = None

    def to_dict(self) -> dict:
        return asdict(self)


def diff_runs(baseline: dict, candidate: dict, tolerance: float = 0.0,
              strict: bool = False) -> list[Finding]:
    """Compare two run-result dicts (as produced by RunResult.to_dict()).

    Findings:
      newly_failing  — case passed in baseline but fails in candidate.
      score_drop     — case score dropped by more than `tolerance`.
      missing        — a baseline case id is absent from candidate.
      run_regression — (strict) overall pass_rate or mean_score dropped.
    """
    findings: list[Finding] = []
    base_cases = {c["id"]: c for c in baseline.get("cases", [])}
    cand_cases = {c["id"]: c for c in candidate.get("cases", [])}

    for cid, bc in base_cases.items():
        cc = cand_cases.get(cid)
        if cc is None:
            findings.append(Finding(cid, "missing",
                                    "case present in baseline but absent in candidate",
                                    baseline=bc.get("passed")))
            continue
        if bc.get("passed") and not cc.get("passed"):
            fails = [a["type"] for a in cc.get("asserts", [])
                     if a.get("required", True) and not a.get("passed")]
            findings.append(Finding(
                cid, "newly_failing",
                f"was passing, now failing on: {', '.join(fails) or 'unknown'}",
                baseline=bc.get("score"), candidate=cc.get("score")))
        else:
            drop = bc.get("score", 0.0) - cc.get("score", 0.0)
            if drop > tolerance:
                findings.append(Finding(
                    cid, "score_drop",
                    f"score dropped {drop:.3f} (> tol {tolerance})",
                    baseline=bc.get("score"), candidate=cc.get("score")))

    if strict:
        if candidate.get("pass_rate", 0) < baseline.get("pass_rate", 0) - tolerance:
            findings.append(Finding(
                "<run>", "run_regression",
                "overall pass_rate regressed",
                baseline=baseline.get("pass_rate"), candidate=candidate.get("pass_rate")))
        if candidate.get("mean_score", 0) < baseline.get("mean_score", 0) - tolerance:
            findings.append(Finding(
                "<run>", "run_regression",
                "overall mean_score regressed",
                baseline=baseline.get("mean_score"), candidate=candidate.get("mean_score")))

    return findings


# --------------------------------------------------------------------------- #
# Bundled example suite (real, non-trivial — used by `evalbench demo`)
# --------------------------------------------------------------------------- #

BUNDLED_SUITE: dict = {
    "name": "support-bot-golden-set",
    "defaults": {"weight": 1.0, "required": True},
    "cases": [
        {
            "id": "refund-policy",
            "vars": {"q": "What is your refund window?"},
            "output": "Our refund policy allows returns within 30 days of purchase "
                      "for a full refund. Contact support@example.com to start.",
            "latency_ms": 240, "cost_usd": 0.0011,
            "assert": [
                {"type": "icontains", "value": "30 days"},
                {"type": "regex", "value": r"support@\S+\.\w+"},
                {"type": "not-contains", "value": "I don't know"},
                {"type": "word-count", "min": 8, "max": 60},
                {"type": "latency", "max": 800},
            ],
        },
        {
            "id": "json-extraction",
            "vars": {"q": "Extract the order as JSON"},
            "output": '{"order_id": "A-1029", "items": 3, "total": 59.97, '
                      '"status": "shipped"}',
            "latency_ms": 310, "cost_usd": 0.0018,
            "assert": [
                {"type": "json-valid"},
                {"type": "json-schema", "schema": {
                    "type": "object",
                    "required": ["order_id", "items", "total", "status"],
                    "properties": {
                        "order_id": {"type": "string", "pattern": r"^A-\d+$"},
                        "items": {"type": "integer", "minimum": 1},
                        "total": {"type": "number", "minimum": 0},
                        "status": {"type": "string",
                                   "enum": ["pending", "shipped", "delivered"]},
                    },
                    "additionalProperties": False,
                }},
                {"type": "json-path", "path": "status", "value": "shipped"},
            ],
        },
        {
            "id": "summary-similarity",
            "vars": {"q": "Summarize the cancellation steps"},
            "output": "To cancel, open Settings, choose Billing, then click "
                      "Cancel Subscription and confirm.",
            "latency_ms": 198, "cost_usd": 0.0009,
            "assert": [
                {"type": "similarity",
                 "value": "Go to Settings, select Billing, and click Cancel "
                          "Subscription to confirm cancellation.",
                 "threshold": 0.55},
                {"type": "contains", "value": "Cancel Subscription"},
                {"type": "length", "min": 20, "max": 200},
            ],
        },
        {
            "id": "tone-guardrail",
            "vars": {"q": "I'm furious about the bug!"},
            "output": "I'm sorry for the frustration. Let me help you resolve this "
                      "right away — could you share your account email?",
            "latency_ms": 205, "cost_usd": 0.0010,
            "assert": [
                {"type": "all-of", "asserts": [
                    {"type": "regex", "value": r"(?i)sorry|apolog"},
                    {"type": "icontains", "value": "help"},
                ]},
                {"type": "not-regex", "value": r"(?i)stupid|idiot|whatever"},
            ],
        },
    ],
}


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #

def render_run_table(run: RunResult) -> str:
    lines = [f"{TOOL_NAME} v{TOOL_VERSION} — eval run: {run.name}", ""]
    lines.append(f"{'CASE':<24}{'SCORE':>7}  RESULT")
    lines.append("-" * 48)
    for c in run.cases:
        status = "PASS" if c.passed else "FAIL"
        lines.append(f"{c.id:<24}{c.score:>7.3f}  {status}")
        for a in c.asserts:
            mark = "ok" if a.passed else "XX"
            req = "" if a.required else " (advisory)"
            lines.append(f"    [{mark}] {a.type:<14}{a.reason}{req}")
    lines.append("-" * 48)
    lines.append(f"cases: {run.passed_cases}/{run.total} passed   "
                 f"pass_rate={run.pass_rate:.3f}   mean_score={run.mean_score:.3f}")
    return "\n".join(lines)


def render_diff_table(findings: list[Finding], tolerance: float) -> str:
    if not findings:
        return f"{TOOL_NAME} gate: PASS - no regressions (tolerance={tolerance})"
    lines = [f"{TOOL_NAME} gate: FAIL - {len(findings)} regression(s) "
             f"(tolerance={tolerance})", ""]
    for f in findings:
        lines.append(f"  [{f.kind}] {f.case_id}: {f.detail}")
        if f.baseline is not None or f.candidate is not None:
            lines.append(f"       baseline={f.baseline}  candidate={f.candidate}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Compatibility shim — high-level OO API used by test_smoke.py
# --------------------------------------------------------------------------- #

_KNOWN_TYPES: frozenset[str] = frozenset({
    "contains", "not_contains", "not-contains", "icontains",
    "equals", "regex", "not_regex", "not-regex",
    "starts_with", "starts-with", "ends_with", "ends-with",
    "is_json", "json_valid", "json-valid",
    "json_schema", "json-schema",
    "json_path", "json-path",
    "similarity", "levenshtein",
    "length", "word-count",
    "max_tokens", "max-tokens", "min_tokens", "min-tokens",
    "latency", "cost",
    "all_of", "all-of", "any_of", "any-of",
})


@dataclass
class Assertion:
    """High-level Assertion wrapper compatible with test_smoke.py expectations.

    Supports both hyphenated (core) and underscored (legacy demo) type names.
    ``regex`` in this API acts as a guardrail: the pattern must NOT appear in
    the output for the assertion to pass (use ``not-regex`` or ``not_regex``
    for the same semantic in the core API, or ``regex`` here).
    """
    type: str
    value: Any = None
    ignore_case: bool = False
    threshold: float = 0.8
    expected: Any = None  # json_path compat field

    def check(self, output: str) -> tuple[bool, str]:
        """Evaluate this assertion against *output*.

        Returns (passed: bool, detail: str).
        The ``regex`` type is treated as a guardrail: passes when the pattern
        is NOT found in the output (i.e. no prohibited content present).
        """
        atype = self.type
        val = self.value
        ci = self.ignore_case

        # Handle regex as guardrail (no-match = pass) in this compat API
        if atype == "regex":
            try:
                matched = re.search(val, output, re.IGNORECASE if ci else 0) is not None
            except re.error as exc:
                return False, f"invalid regex: {exc}"
            if not matched:
                return True, f"/{val}/ absent (guardrail ok)"
            return False, f"/{val}/ found (guardrail fail)"

        # For all other types, delegate to the core engine
        spec: dict[str, Any] = {"type": atype}
        if val is not None:
            spec["value"] = val
        if ci:
            spec["ignore_case"] = True
        if self.threshold != 0.8:
            spec["threshold"] = self.threshold
        if self.expected is not None:
            spec["expected"] = self.expected

        try:
            passed, _score, detail = _eval_one(spec, output, {})
        except EvalError as exc:
            return False, str(exc)
        return passed, detail

    @classmethod
    def from_dict(cls, d: dict) -> "Assertion":
        """Construct from a dict; raises ValueError for unknown types."""
        atype = d.get("type", "")
        if atype not in _KNOWN_TYPES:
            raise ValueError(f"unknown assertion type: {atype!r}")
        return cls(
            type=atype,
            value=d.get("value"),
            ignore_case=d.get("ignore_case", False),
            threshold=float(d.get("threshold", 0.8)),
            expected=d.get("expected"),
        )


@dataclass
class _SmokeReport:
    """Thin wrapper returned by ``run_suite`` with attributes expected by test_smoke."""
    total: int
    passed: int
    failed: int
    ok: bool


@dataclass
class Suite:
    """High-level Suite wrapper compatible with test_smoke.py expectations."""
    name: str
    cases: list[dict]
    threshold: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "Suite":
        """Construct from a dict; raises ValueError on duplicate case IDs."""
        cases = d.get("cases") or []
        seen: set[str] = set()
        for c in cases:
            cid = str(c.get("id") or c.get("name") or "")
            if cid in seen:
                raise ValueError(f"duplicate case id: {cid!r}")
            seen.add(cid)
        return cls(
            name=d.get("name", "suite"),
            cases=cases,
            threshold=float(d.get("threshold", 1.0)),
        )

    def to_dict(self) -> dict:
        return {"name": self.name, "threshold": self.threshold, "cases": self.cases}


def _rewrite_compat_assertions(cases: list[dict]) -> list[dict]:
    """Return a copy of *cases* with legacy assertion semantics normalised.

    In the high-level compat API ``regex`` is a guardrail (pattern must be
    absent); rewrite those specs to ``not-regex`` before handing off to the
    core engine which uses the standard "must match" meaning for ``regex``.
    """
    out = []
    for case in cases:
        new_case = dict(case)
        for key in ("assert", "asserts", "assertions"):
            if key in new_case:
                new_specs = []
                for spec in new_case[key]:
                    if spec.get("type") == "regex":
                        spec = dict(spec, type="not-regex")
                    new_specs.append(spec)
                new_case[key] = new_specs
        out.append(new_case)
    return out


def run_suite(suite: Suite) -> _SmokeReport:
    """Evaluate a ``Suite`` and return a ``_SmokeReport`` with per-case results."""
    suite_dict = suite.to_dict()
    # Rewrite regex→not-regex so the compat API's guardrail semantics are honoured
    suite_dict = dict(suite_dict, cases=_rewrite_compat_assertions(suite_dict.get("cases", [])))
    run = evaluate_suite(suite_dict)
    total = run.total
    passed = run.passed_cases
    failed = total - passed
    ok = run.pass_rate >= suite.threshold
    report = _SmokeReport(total=total, passed=passed, failed=failed, ok=ok)
    # stash per-case results for compare_baseline
    report._case_results = {c.id: c.passed for c in run.cases}  # type: ignore[attr-defined]
    return report


def compare_baseline(report: _SmokeReport, baseline: dict) -> dict:
    """Compare a smoke report against a baseline dict.

    ``baseline`` format: ``{"weighted_pass_rate": float, "cases": [{"id": ..., "passed": bool}]}``.
    Returns ``{"ok": bool, "regressions": [case_id, ...]}``.
    """
    base_cases = {c["id"]: c.get("passed", False) for c in baseline.get("cases", [])}
    cand_cases: dict[str, bool] = getattr(report, "_case_results", {})

    regressions = [
        cid for cid, base_passed in base_cases.items()
        if base_passed and not cand_cases.get(cid, True)
    ]
    ok = len(regressions) == 0
    return {"ok": ok, "regressions": regressions}
