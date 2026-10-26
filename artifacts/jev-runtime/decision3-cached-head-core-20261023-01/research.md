# Cached Decision3 head numerical core: pre-implementation decision

2026-10-23. Parent HEAD03d65691099d5ffad3b7b38ba847c5a58448c6aa; the
worktree was clean before the independent new candidates. The root owns only
this new directory. This is an unselected numerical primitive, not a data loader,
model artifact writer, training command or promoted policy.

| Reference | Evidence read | Decision |
| --- | --- | --- |
| Existing repository MIT readout | src/s1/browser_readout.py; scripts/breakthrough_training.py:fit_head; scripts/jev_browser_fit_readout.py | Reuse immutable-base/provenance design. Its52 LM rows, bias, softcap and metadata differ from D3; do not change it. |
| Publisher d3-edge runtime | Pinned cab71e6ad3461b8086d51ddb572c656babd63b0b/d3_runtime.py:logits/probabilities, source SHA adf7f7aa085674b5682e492d8b2e3cc4f11c67b0d153515c742e907e9b550917 | Retain255-code FP32 projection, legal-count mask and temperature1. The feature observer separately proves same-call replay using the genuine runtime. |
| PyTorch | Official2.14 Module and Adam API/source; v2.13.0 LICENSE | Standard forward hooks and optimizer available; installed runtime is2.13, and2.13 docs could not be opened. No new framework behavior is assumed without actual proof. |
| scikit-learn logistic regression | Public main sklearn/linear_model/_logistic.py,2026-10-23 | General coefficient fitting does not directly preserve this immutable W0+delta and per-example legal mask; no new dependency or copied source. |
| NumPy2.5.3 | Installed BSD-3-Clause LICENSE.txt; repository's existing NumPy dependency | Use array arithmetic for a small explicit masked CE+L2 SGD core. Verify gradients with independent finite differences and closed-form cases. This is a separate numerical profile, not publisher-native equivalence. |

Primary API references (read2026-10-23):
- https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html
- https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html
- https://github.com/pytorch/pytorch/blob/v2.13.0/LICENSE
- https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_logistic.py

No implementation is copied from those external projects. The equations are
z=hW0^T+hDelta^T; illegal codes>=N have zero probability; soft-target CE plus
0.5*lambda*||Delta||^2, with FP32 arrays and fixed temperature1. Soft targets
allow multiple independently authored valid actions without inventing a tie.
Only Delta changes; no bias, DONE penalty, calibration or encoder gradient.

The primitive accepts arrays only and has no file/network/CLI/artifact API. A
future owned caller must accept only independently COMMITTED, authenticated
SELF_AUTHORED_TRAIN feature chunks and enforce split/family exclusion. This
module alone proves no data eligibility, memory fit, training success or real
browser improvement. Actual model/readout/features/deltas remain ignored local
payloads; no real array or parameters are created in source tests.
