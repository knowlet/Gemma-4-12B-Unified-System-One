# Decision Benchmark v2 implementation status

Current requested scope: complete the remaining evaluation tooling, retain honest
not-run states for unavailable models/data/hardware, test changes, and keep
conventional commits. Do not publish or claim live model measurements from fixtures.

- [x] E0 registry/preflight/budget planning (`f3136f0`).
- [x] E1 reference execution/artifacts (`2228e8a`).
- [x] E2 hard/soft/ordinal metrics, split audit, source-group bootstrap, and
  calibration-only fitting/application implemented (validation in progress).
- [ ] E1 additional provider protocols, low-cost baselines, dataset import/provenance.
- [ ] E3 native independent Gemma batching, controlled readouts and sweep/parity tools.
- [ ] E4 fixed concurrency/open-loop scheduling, deadlines, goodput and timing.
- [ ] E5 multimodal counterfactual/pipeline/retention comparisons.
- [ ] E6 matched few-shot support/training recipes and seed comparisons.
- [ ] E7 resettable workflow evaluation, cost accounting and standalone reports.
- [ ] End-to-end offline acceptance, documentation and conventional commits.

External validation still required: actual model weights/API credentials, suitable
GPU hardware, representative held-out media/workflow data and paid service budgets.
CPU-only PyTorch 2.8.0 and Transformers 5.17.0 were installed locally for real tiny
multimodal forwards; no pretrained weights are needed for these tests.
