# Demo 01 — Basic agent eval with a regression gate

This demo evaluates recorded outputs from a customer-support agent against a
small suite of offline assertions, then shows how to gate CI on regressions.

## The suite

`support_suite.json` defines four cases drawn from a support bot:

- `greet` — must greet the user and stay short.
- `refund_json` — must emit valid JSON with an `action` field equal to `refund`.
- `no_pii` — must NOT leak a raw credit-card number (regex guard).
- `summary_len` — summary must be at least 8 tokens (not a one-word dodge).

The suite `threshold` is `0.75`, so at least 75% (weighted) must pass.

## Run it

```sh
python -m evalbench run demos/01-basic/support_suite.json
```

Expected: all four cases pass, exit code `0`.

## Inspect / machine-read

```sh
python -m evalbench list demos/01-basic/support_suite.json --format json
python -m evalbench run  demos/01-basic/support_suite.json --format json
```

## Wire it into CI as a regression gate

Save a known-good baseline, then fail the build if any previously-passing
case regresses:

```sh
# 1. record the current run as the baseline (committed to the repo)
python -m evalbench run demos/01-basic/support_suite.json \
    --format json --out baseline.json

# 2. in CI, run again and compare
python -m evalbench run demos/01-basic/support_suite.json \
    --baseline baseline.json
```

If a case that passed in `baseline.json` now fails, `evalbench` prints a
`REGRESSIONS:` line and exits non-zero — perfect for `ci-for-agents`.
