# Gemma 4 Unified System One

Native text, image and audio decisions from `google/gemma-4-12B-it`:
one multimodal backbone forward, multiple answer slots, selected LM-head rows,
and typed Choice / Noul / Score probabilities. No text generation or JSON parsing.

This project extends [system-one-open](https://github.com/mithalouni/system-one-open).
The original E2B experiments and demos remain available; their historical results
are not measurements of this Unified model. See [upstream notes](docs/upstream.md).

## Development

```bash
uv sync --extra api
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv build
```

Inference and model tests: `uv sync --extra inference --extra api`.
Cloud tooling: add `--extra modal`; Laya comparisons: add `--extra laya`.
`uv.lock` pins all optional dependencies. No model download occurs during the default tests.

See [architecture](docs/architecture.md), [benchmarking](docs/benchmarking.md),
and [Modal deployment](docs/modal.md) for runnable examples and limitations.

[Decision Benchmark v2](docs/benchmark-v2.md) includes capability/budget planning
and a sequential runner with grouped text fixtures and reproducible result records:

```bash
uv run --no-sync s1 eval run --profile text-smoke --enable-model uniform \
  --output artifacts/evaluation/text-smoke-run
```

## Layout

| Path | Purpose |
| --- | --- |
| `src/s1/` | Installable decision SDK, validation, multimodal inference, API and evaluation |
| `apps/modal/` | Cloud entrypoints; retained upstream training and demos |
| `tests/` | Offline contracts, numerical regression tests and tiny-model integration |
| `examples/` | Requests and a small benchmark fixture, not an accuracy leaderboard |
| `scripts/` | Historical report generation |
| `docs/` | Design, evaluation protocol and deployment instructions |
| `results/`, `media/`, `report.html` | Historical upstream artifacts, retained unchanged |

The new API exposes `/v1/systemone` and `/decide`; local CLI usage is `uv run s1 --help`.
