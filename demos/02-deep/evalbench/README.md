# evalbench — deep demo: assertions + regression gate

This demo shows the headline feature set: rich assertion types and a CI-style
regression gate comparing a candidate run to a baseline.

## 1. Run the bundled golden set (no files needed)

```sh
python -m evalbench demo
python -m evalbench demo --format json
```

## 2. Evaluate a suite and save the baseline

```sh
python -m evalbench run baseline_suite.json --save baseline_run.json
```

`baseline_suite.json` defines 3 cases exercising `icontains`, `not-contains`,
`word-count`, `latency`, `json-valid`, `json-schema` (with `enum`, `minimum`,
`additionalProperties: false`), `json-path`, `similarity`, and `contains`.

## 3. Gate a candidate against the baseline

```sh
python -m evalbench gate baseline_suite.json candidate_suite.json --strict
```

`candidate_suite.json` deliberately regresses the `json-weather` case:
the model now returns `"condition": "foggy"` (not in the `enum`) and an extra
`humidity` field (forbidden by `additionalProperties: false`), and the
`json-path` no longer equals `"sunny"`. The gate detects a `newly_failing`
finding and exits non-zero — exactly what you want failing a CI step.

`gate` accepts either saved run-result JSON or raw suites for both arguments and
evaluates suites on the fly.
