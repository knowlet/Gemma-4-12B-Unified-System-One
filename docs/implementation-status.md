# Decision Benchmark v2 implementation status

Current requested scope: complete the remaining evaluation tooling, retain honest
not-run states for unavailable models/data/hardware, test changes, and keep
conventional commits. Do not publish or claim live model measurements from fixtures.

- [x] E0 registry/preflight/budget planning (`f3136f0`).
- [x] E1 reference execution/artifacts (`2228e8a`).
- [x] E2 hard/soft/ordinal metrics, split audit, source-group bootstrap, and
  calibration-only fitting/application (`dc4f9fe`).
- [x] E1 additional provider protocols, low-cost baselines, dataset import/provenance.
- [x] E3 native independent Gemma batching, G0–G4 controlled readouts,
  synthetic N/K/length/reuse preparation, order perturbation and paired behavior tools.
- [x] E4 fixed concurrency/open-loop scheduling, SLO accounting, goodput and timing.
- [x] E5 multimodal counterfactual/pipeline/retention comparisons.
- [x] E6 matched few-shot support, SetFit and three-seed CE/CE+Brier training,
  separate adapters, content hashes and training/test overlap rejection.
- [x] E7 resettable workflow evaluation, routed cost accounting, paired policy
  comparisons, noninferiority gate and standalone reports.
- [x] End-to-end offline acceptance and documentation.

Local verification: **455 tests passed, none skipped**, including real tiny Gemma
FP32/BF16 media forwards, three-seed CE/CE+Brier updates, separate adapter reload,
TF-IDF/LR fit, local NLI/cross-encoder forwards, and SetFit training/save/reload.
Ruff lint/format, `uv lock --check --offline`, wheel/sdist build and the offline
acceptance script passed. CI definitions also run acceptance and install the optional
baseline runtimes in the inference job; remote CI has not been run by this task.

External validation still required: actual model weights/API credentials, suitable
GPU hardware, representative held-out media/workflow data and paid service budgets.
CPU-only PyTorch 2.8.0 and Transformers 5.17.0 were installed locally for real tiny
multimodal forwards; no pretrained weights are needed for these tests.

The delivered reference scope does not enable shared-prefix/packed-attention
optimizations, quantization/MLX runtimes, high-cardinality K>52, crash-resume or
production browser/coding-agent environments. These are explicit extensions, not
unverified entries presented as supported capabilities. The workflow environments
and acceptance outputs are fixtures; representative live model measurement remains
an external validation stage.
