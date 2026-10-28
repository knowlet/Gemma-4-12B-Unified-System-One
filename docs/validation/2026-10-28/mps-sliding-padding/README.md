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

The next full-checkpoint control replays the unchanged archived batched request
and exact per-question singleton forwards before and after native text
generation. Observers record finite/nonfinite values at each layer and readout.
Only that control can connect the kernel witness to this model's actual failure.
A proposed MPS batching guard derives its length bound from the native sliding
window; it must retain prompt/media/order/calibration and pass real validation
before being called a fix.
