# Native28 memory probe entry: held source candidate

Only `memory_probe.py`, `test_memory_probe.py` and this note were authored for
the entry task. No candidate import, AST parse, formatter, builder, dry run,
test, host/unit observation, data/model/native payload read, GPU action, service
or Git action was executed. The original native/core/public fitter files were
read as sources; their small source SHA values were recorded with `sha256sum`.
The existing full06 model work was left untouched.

`SOURCE_ONLY = True` deliberately refuses `execute` before reading its plan or
loading any dependency. Default CLI behavior prints held diagnostic metadata
using stdlib only. This is not a runnable or qualified training handoff. There
is no actor plan, source freeze, reviewed exact command, independent guardian,
admission or actual closure evidence. Do not clear the guard merely to obtain
a passing run: finish those pieces, review and check the source, then reselect
the changed actor source and all dependent pins.

The future plan contract is a draft. Its non-null `root_guardian_selection`
field is provenance, not proof of a live guardian. The guardian's independent
identity, resource observations, stop behavior and accepted closure still need
implementation and review. No source code here substitutes a Boolean or file
for those external guarantees.

## Actual source composition

The entry loads the original native fitter helpers by SHA and preserves the
original helper/module namespace instead of monkeypatching it. Recorded source
pins are:

| Source | SHA256 |
| --- | --- |
| `scripts/jev_browser_fit_native_actions.py` | `8f127b338d32b079e814090a828fd3a608271fac5a711216d131c52b509988d5` |
| `scripts/jev_browser_fit_actions.py` | `709eb702ee1a926dc6bea3740d28e880246f1be944fdb6d06deb4bec915dd474` |
| `scripts/jev_browser_fit_public_actions.py` | `a38bbab599f6f8ec8117d1b912e02913330ea7bb11cd67e4e6793ed80da01ca7` |

Original native `_preflight_paths`, `_environment`, `_sources`, `_checkpoint`,
`_construct_model`, `_check_model` and `_final_checks` are reused directly.
The existing source capture retains its approved caller labels and actual
original files. It does not claim the new entry is an approved normal fitter.
A separate five-file `diagnostic-sources` namespace captures the entry, new
resource gate and three original composition sources, with current and captured
copy verification at finalization. The future plan must also select the exact
original S1/runtime source inventory identity before weights are loaded.

All 179 original historical rows are loaded, frozen and checked: 142 train and
37 development, including the original ID/group/request-fingerprint separation
and original/frozen byte revalidation. The optional initial parent is captured
and split-checked through the existing APIs. The original BF16 checkpoint and
full model/head/FP32 q/v LoRA checks are retained. The training context is the
original fitter's 8192 with rejection rather than truncation; this is separate
from the unchanged 16384-context Decision3 benchmark.

Every original row is compiled with the unchanged `kernel.compile_actions`,
original `shared-elements-v2`, native target labels and CFG/trie composition.
The original `kernel.train_actions` is called with all 142 compiled train rows
and all 37 development rows, fixed two updates, seed42 and learning rate1e-4.
Its batch1/full-vocabulary target-position CE, non-reentrant checkpointing,
FP32 q/v-only AdamW, frozen full head and development `no_grad()` logic are not
changed. Two updates follow the original deterministic shuffle. They do not
cover every training row or establish worst-case/full120-step capacity.

The new resource Gate surrounds the same pre-model/parent/compile/optimizer and
native train/development forward phases. Original training exceptions retain
their partial update ledger and unknown in-flight effects. New resource
constructor errors preserve `native28_resource_evidence`; phase failures remain
in the resource journal. Hook removal, original source/data revalidation and
the separate diagnostic source namespace are finalized independently so one
cleanup error does not skip all remaining evidence. A literal elapsed check and
the new unit's selected3h deadline still require real execution verification.

## Output and qualification boundary

The output creates its own `*` ignore file. It may retain frozen inputs, source
copies, compile/CFG records, original fitting metrics and a private
`diagnostic-receipt.json`. Every receipt remains `qualified=false`,
`training_qualified=false`, `browser_qualified=false`. There is no call to
`_save_adapter` or normal adapter publication/verification on either success or
failure. No adapter directory or accepted adapter manifest is produced. Normal
loader support for a future formal28GiB training artifact remains a separate
unimplemented change; this diagnostic cannot inherit old40GiB qualification.

Eight authored stdlib fake tests cover default dry/held execution, raw resource
and partial-training error retention, forward-hook mode/refusal semantics,
strict fixed plan values, duplicate plan keys, complete179 compiler composition
without wire mutation and missing-population refusal. They have not been
imported or executed. They cannot establish Torch/GPU memory, bytecode/body
equivalence, actual data validity, guardian behavior or capacity. Root must run
appropriate bounded checks only after the current NN work and original owner
closure are terminal, then complete the missing guardian/selection before any
diagnostic execution.
