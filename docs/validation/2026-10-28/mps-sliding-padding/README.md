# MPS fully masked sliding-padding witness

Exact CPU preprocessing of both failed external requests finds fully masked
**padding** query rows in native sliding attention. Real query rows and answer
slots have valid keys; candidate masks retain 6/41 or 6/43 options.

| Page observation | Independent prompt lengths | Padding gap | Fully masked padding rows |
| --- | --- | ---: | ---: |
| Wikipedia main page | 2243 / 3331 | 1088 | 65 |
| Graph theory | 2257 / 3488 | 1231 | 208 |

The configured sliding window is 1024. All three failures for each page share
the same logical request fingerprint. Input tensors and exact four-dimensional
boolean masks are contiguous. The
[exact-request witness](exact-request-mask-witness.json) records versions,
executed-source/mask-constructor hashes, shapes, slots and masked-row counts.
It did not load model weights or execute attention.

A separate synthetic kernel probe then used identical seeded finite Q/K/V and a
sliding mask with 77 fully masked padding rows on CPU and MPS, in both FP32 and
BF16. CPU outputs remained finite. MPS produced 2,464 nonfinite padding values
in the first attention output; feeding that output to a second masked attention
made all 3,200 real-token values nonfinite. Thus masked padding can contaminate
real queries on this installed PyTorch 2.8.0 MPS runtime. This kernel experiment
does not by itself prove where the full checkpoint first becomes nonfinite.

[report.json](report.json) preserves exact counts, seed, mask/source hashes and
environment. [probe.py.txt](probe.py.txt) is the exact executed source; its bytes
are kept unformatted. Reproduce with `python probe.py.txt` in the recorded
PyTorch environment. The probe uses no model, labels, answer repair or CUDA.
The version-pinned [PyTorch MPS attention source](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/mps/operations/Attention.mm)
is a primary implementation reference.

The MPS guard now partitions compatible rows in their original order so every
padding gap is strictly smaller than the minimum native sliding window. It keeps
complete prompts/media, answer order, calibration and eight-prompt limits. It
does not change masks, replace nonfinite probabilities or retry on another model.
CPU/CUDA batching is unchanged.

The repaired [full-checkpoint control](patched-native-control.json) uses the same
failed Main_Page input and safe observer copies. Batched execution now makes
two native forwards (2243/3331 tokens). All four decision answer maps—batched
and exact singletons, before and after generation—are identical, with zero
probability difference and no invalid distributions. Each decision phase's 102
tensor observations contains zero nonfinite values. All 6,656 generation
observations are also finite, but the helper still produces invalid JSON. The
numerical fix therefore does not establish helper suitability or task completion.

The earlier instrumented control is
[explicitly invalidated](../mps-observer-safety/README.md) because its combined
copy/conversion could corrupt telemetry and mutate masks. It is excluded from
the above claim. The uninstrumented external baseline, exact CPU masks, kernel
witness and repaired patched control provide separate evidence. The
[patched control archive](patched-native-control-evidence.zip) retains all raw
responses, safe telemetry, input/source hashes and checkpoint identity. Times
include observer synchronization and are not speed evidence. Actual browser
replay and both JevBench suites remain separate acceptance steps.
