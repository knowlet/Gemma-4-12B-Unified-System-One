# MPS graph-cache maintenance control (2026-10-28)

The browser pilot reaches both destination states but exceeds the frozen 180 s
budget. Whole-call MPS maintenance is enabled in that pilot. No numerical-shape
optimization has passed the unchanged 0.01 probability/argmax gate.

This bounded investigation reuses the existing native full-prompt inference,
development-only snapshots, process guards and synchronized measurement code.
It does not replace PyTorch, alter model precision, use another model or change
production cleanup defaults.

## Source comparison and decision

| Source | Relevant mechanism | Decision and limits |
| --- | --- | --- |
| PyTorch, BSD-3-Clause; [commit 350cda97](https://github.com/pytorch/pytorch/commit/350cda97a10d843e1df5e556ffe67df5def10fbe), `aten/src/ATen/mps/MPSHooks.mm::emptyCache` and `aten/src/ATen/native/mps/OperationUtils.h::MPSGraphCache::clear` | Clearing the allocator also deletes compiled graphs under the serial queue. The cache is keyed by operation, shape and dtype. | Use the public API directly. Fewer clears might avoid recompilation; the source does not prove a speed improvement for this model. No upstream code is copied. |
| [Actual installed Torch build](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/mps/MPSAllocator.mm), allocator `emptyCache` and memory accounting | Device allocations and cached resources are different populations; measurements need synchronization and separate active/driver bytes. | Keep normal watermarks and a pre-dispatch driver guard. Memory observations are snapshots, not peaks. |
| Existing S1 `scripts/diagnose_mps_prefix_shapes.py`, `src/s1/evaluation/gemma.py::predict_batch` | Frozen native development requests and complete independent prompts. | Reuse the input/source/checkpoint guards. Preserve the exact per-request model path and shapes. Do not use rejected prefix reuse. |

The query scope is the relevant upstream PyTorch cache implementation and the
installed source commit, not a general runtime replacement survey. The prior
isolated 2.14.1 controls establish bounded memory improvement with per-call
clearing; they do not establish safety without clearing.

## Proposed bounded experiment

Run the same three pre-existing development observations in chronological order
in each ABBA block. A clears after every whole prediction; B retains allocator
and compiled graph caches inside a three-call block. Both start each block from
a synchronized cache clear and include the initial call, final cleanup and
cleanup failures in recorded evidence. No warmup or failed sample is discarded.
Only the resource policy changes; no KV state crosses requests.

Use one isolated MPS/BF16/SDPA checkpoint sequentially after all earlier model
processes terminate. Record all raw answers, input/source hashes, synchronized
call/block latency, and active/driver memory before and after each call/clear.
Stop before another inference if driver allocation reaches 0.8 of the device
recommended maximum. Any inference, observation or cleanup error terminates the
experiment, preserving the failure. Compare every repeated request across A/B
with the original 0.01 maximum probability difference and identical argmax gate.
Keep memory and numerical gates separate from timing; passing this small control
would still require actual browser and full benchmark acceptance before any
production default changes.
