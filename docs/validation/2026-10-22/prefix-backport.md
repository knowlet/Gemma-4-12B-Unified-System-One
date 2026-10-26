# CPU BF16 prefix alignment backport

This repair carries the existing CPU-only numerical correction from PR #9,
commit `04121b2871cd02ddebe033cd39e9b1151adb298e`, onto the PR #10 source
baseline `c0fdf89da3ea36aa3ed6f0551e504f2f9fa252c5`. It does not merge PR #9
or change PR #10's base. It adds a bounded production correction to a PR
whose earlier changes were test-only.

## Source provenance and scope

The following files are byte-identical to that PR #9 commit:

- `src/s1/browser_prefix.py`
- `src/s1/browser_prefix_cache.py`
- `tests/test_browser_prefix_alignment.py`

The existing `tests/test_browser_prefix.py` and backend tests are unchanged.
All materialized baseline source/test files were verified against their Git
blob identities at the PR #10 head before editing. The local validation tree
is a source subset, not a checkout with the remote commit's Git ancestry.

CPU BF16 SDPA is sensitive to the physical key-axis reduction geometry.
The correction preserves masked absolute columns in transient independent
cache forks, while retaining native sliding-window storage. It changes no
winner rule, tolerance, model input, candidate projection, attention backend,
or default model precision. FP32/eager/non-CPU paths remain native.
Additional transient attention memory/work is intentional; no speedup is claimed.

## Local CPU validation

Environment: Linux x86_64, Python 3.12.14, PyTorch 2.8.0+cpu,
Transformers 5.17.0, NumPy 2.5.4, Pydantic 2.14.1, pytest 9.1.1,
Pillow 12.3.0, Ruff 0.17.1. No weights were downloaded. Tests construct
the existing seeded tiny Gemma model locally; this is not trained-12B acceptance.

Commands use `PYTHONPATH=src HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2`.

- Baseline: `.venv/bin/python -m pytest -q tests/test_browser_prefix.py -k real_gemma_prefix --tb=short`
  reproduces **1 failed, 5 passed, 46 deselected**. The original goal-first
  BF16 winner assertion fails with `item-43` versus `item-37`.
- Corrected: `.venv/bin/python -m pytest -q tests/test_browser_prefix.py tests/test_browser_prefix_alignment.py tests/test_browser_prefix_backend.py --tb=short`
  passes **94 tests**, with no skips. This includes 52 original prefix tests,
  15 existing alignment regressions, and 27 backend tests.
- Ruff lint, Ruff formatting check, and `git diff --check` pass for the changed code.

## Remaining boundaries

The full inference suite was not run locally in this source-subset tree.
Hosted CI for a newly published head must be checked separately.
The October 18 CI run 39108252586 is historical: its checkout was merge-ref
`03f6728`, combining `c0fdf89` with `a72426a`, not a new exact-head result.
Its two failures were this BF16 assertion and missing historical wire bytes.

The original `native-original-wire.json` and its historical integrity chain
remain unavailable. No replacement capture, synthesized fixture, skipped test,
xfail, or rewritten historical evidence is included. GPU, trained-12B,
live-browser, production deployment, merge, and release acceptance remain
outside this repair.
