This is an isolated, unvalidated SOURCE-ONLY28GiB memory diagnostic candidate.
The directory was created exclusively and `resource_gate.py` was copied from
the current `src/s1/browser_native_resources.py`, then edited only in this new
namespace. No production, frozen H15,40GiB resource profile, fitter, adapter,
launcher, model, data, current NN selection or historical receipt was changed.

During authoring and source review, no candidate import, AST, builder, formatter,
test, dry run, host/Driver/model observation, GPU work or service was performed.
Small source SHA/size records and independent text reviews are source evidence
only. Executed validation, freeze, actual admission and capacity remain PENDING.
`memory_probe.py` is a separate, held entry candidate: `SOURCE_ONLY=True`
refuses execution before reading a plan or loading dependencies. It composes
all179 original rows, two original training updates and37 development rows;
it never saves an adapter or a normal manifest on success or failure. There
is no external guardian implementation or accepted exact execution selection.
Its non-null guardian plan field is provenance only, not a live proof. Merely
flipping the source hold does not complete those missing requirements.
`plan.pending.json` is metadata, not a runnable command specification.

The new profile is `gx10-uma-native-28gib-memory-diagnostic-v1`, schema is
`jev-native-resource-gate-gx10-uma-28gib-diagnostic-v1`, and expected unit is
`jev-native28-memory-diagnostic-20261024-01.service` under the original explicit
user1001/app.slice namespace. Actual same-boot/unit/invocation identity and
Driver/Torch/host totals still require fresh future validation.

| Policy | New value |
| --- | --- |
| Native allocator cap | 28GiB =30,064,771,072 bytes |
| Separate memcg cap | 28GiB =30,064,771,072 bytes |
| CUDA reserved live stop | strictly below27.5GiB =29,527,900,160 bytes |
| memcg current live stop | strictly below27.5GiB =29,527,900,160 bytes |
| Swap | zero |
| R | max(24GiB, ceil(MemTotal/5)), unchanged |
| Startup required available | R+2*28GiB+4GiB |
| Verified-loaded required available | R+28GiB+8GiB |

For declared arithmetic MemTotal130,595,864,576, R=26,119,172,916,
startup=90,543,682,356 and loaded=64,773,878,580 bytes. These values are formula
examples, not observations by this author. Unlike the original40GiB profile,
this explicitly distinct candidate does not take a maximum with the old
prospective floor. The original formula remains diagnostic-only:
`old_train_required_bytes = R + 2*23,919,549,408 + 30GiB + 256MiB`.
The old CUDA-free predicate stays41GiB, regardless of the new28GiB cap.
Neither original rule or historical refusal is changed or reinterpreted.

The candidate owns its28GiB `allocator_cap` helper. It validates the same selected
UUID, GB10 name and native backend; sets `(CAP-4MiB)/actual_total` once and checks
the actual fraction readback. It never initializes40GiB first, calls frozen
H15's allocator initializer/floor, monkeypatches H15 or changes cached globals.
Unchanged byte-pinned H15 identity/Driver/origin mechanisms and the original
three component descriptors remain the source of those identity checks.

Raw allocator peaks and memcg peaks/OOM counters are retained. Gate checks keep
the raw host/unit readback before applying refusals, and retain the native
snapshot before validating it. A constructor-time unit refusal attaches
`error.native28_resource_evidence` because no admitted phase journal yet exists;
a future entry must persist that object without replacing the primary error.
An external guardian and partial-update/cleanup receipts remain necessary.
Phase checks are sampling, not prevention of every transient allocation or a
guarantee that a failing native call returns. The two caps are separate and are
never added into, or asserted as, a combined physical UMA hard cap.

Original142train/37development compilation, phase order, loaded-model checker,
batch1/context/trainer/LoRA behavior, process-local release schedule, no swap or
resident credit,3hour bound, identity checks and component capture interfaces
are retained. No data or training code is included or modified. Normal native
fit and adapter verification still accept only their existing profiles and
must reject this profile. No usable adapter manifest is produced here.

`test_resource_gate.py` contains stdlib fake SOURCE for arithmetic,28-only
allocator setup, wrong-device/backend refusal, memcg stop equality, old-cap/OOM
refusal, retained constructor/phase failure readings and CUDA stop equality.
The fake CUDA-bound test stubs identity checks; it is not real Driver evidence.
`test_memory_probe.py` adds eight unexecuted stdlib fake-test sources for the
held entry and full compiler composition; see `actor-source-notes.md`.
None of these tests was imported or run. Root must review, release bounded
checks and pin exact sources before any separate measurement-entry/guardian
integration. A successful small memory measurement would still not prove that
the entire unchanged142/37 training schedule fits or improves browser quality.
