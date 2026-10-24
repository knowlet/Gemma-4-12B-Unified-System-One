# GX10 memory and CPU browser integration: source findings

This is a source investigation and held preparation, not a new benchmark,
training result or browser success. The existing full06 Decision3 evaluation
continues in its original session and resource profile. Candidate checks and
execution remain held until that work and its original owner are terminal.

The historical training gate in `scripts/jev_browser_fit_public_actions.py`
calculates its required available memory as
`R + 2*23,919,549,408 + 30GiB + 256MiB`, where
`R=max(24GiB, ceil(MemTotal/5))`. With the declared arithmetic example
`MemTotal=130,595,864,576`, this is `106,438,961,908` bytes. This number is an
admission estimate, not a measured allocation peak or a hardware capacity
measurement. The existing code applies the same prospective formula after
loading; it does not subtract a resident model allowance. A refusal remains a
valid refusal under that profile.

The original native40GiB profile is unchanged. An isolated candidate at
`artifacts/jev-runtime/native28-resource-candidate-20261024-01` instead declares
separate28GiB native-allocator and memcg limits, with live-stop readings strictly
below27.5GiB. It retains the original reserve and proposes explicit floors
`R+2*28GiB+4GiB` before load and `R+28GiB+8GiB` after genuine model verification.
Those are `90,543,682,356` and `64,773,878,580` bytes in the same arithmetic
example. No actual admission or fit measurement is implied. These two caps are
separate; they do not establish a combined physical UMA hard cap.

The held entry retains all142 original training rows and37 development rows,
the original full BF1612B model, context8192 rejection policy, shared-elements-v2
compiler, q/v LoRA, batch1, full-vocabulary target-position cross entropy and
unchanged trainer. Its fixed two updates followed by all37 development rows are
only a memory diagnostic. They do not cover every training row or establish
full-schedule capacity. Neither success nor failure exports an adapter or a
normal adapter manifest. `SOURCE_ONLY=True` currently refuses execution before
reading a plan or loading dependencies. Source checks, a real external
guardian, exact source selection and fresh admission remain required.

Separately, `src/s1/evaluation/decision3.py:configuration_blockers` rejects
non-CUDA devices, and the constructor in `src/s1/jev_decision3_native.py` applies
that blocker. A real CPU browser endpoint therefore needs its own honest CPU
constructor and resource identity. The original `predict` parsing, ordered
questions, response normalization and named-wire ABI can be retained. The
batch4 smoke witness also requires exactly four rows, while a minimal browser
request can contain fewer; a CPU diagnostic must observe the real1–4 row
batches rather than insert fake questions.

This CPU integration limitation does not explain or invalidate the earlier
GPU browser failures. A source-only HTTP diagnostic is being prepared to
separate wiring from model decisions using an invented two-button page and
the unmodified native selections. No new CPU HTTP, browser or GPU qualification
has been established. The two JevBench suites remain separate; no browser
success or competitor win follows from their scores.
