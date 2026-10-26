# Cached Decision3 delta-head numerical core

An unselected array primitive for a future owned caller. It computes the original
255-code form hW0^T + hDelta^T, applies each row's legal-option mask at temperature1,
and fits only Delta using soft-target cross entropy, L2 and deterministic SGD.
It has no file loader, model constructor, checkpoint writer or training CLI.

Root observed14 bounded synthetic NumPy tests and Ruff pass. Independent source
review found an accepted soft-target rounding case that required multiplying the
softmax by the actual target row mass in the gradient. The corrected source and
exact [0.25,-0.25] regression passed; validation.json binds the checked bytes.
The peer review preserves its earlier pending-test wording with this receipt as
the subsequent root test observation.

This is a separate NumPy FP32 profile. Synthetic learning is not publisher-native
numerical equivalence, real feature eligibility, browser improvement or an
official score. A future owned wrapper must enforce the actual255x1024 readout,
authenticate immutable baseline/features, permit only COMMITTED train chunks,
exclude held-out families, enforce the entire transaction deadline and accept
parameters only after original owner/root/closure checks. Actual arrays and
parameters remain local ignored payloads. No such wrapper or model is selected.

See research.md for the pre-implementation reuse decision and limitations,
independent-source-review.json for the resolved gradient finding, and
validation.json for portable test scope. All new source resides in this directory;
existing runtime, model, benchmark, resource owner and browser code is unchanged.
