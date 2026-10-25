# GX10 native GPU qualification

2026-10-25, worktree based on `e085f8f513c6a36681d2dea03b0a44b2873ebf54`.
This extends the existing GX10 execution-gap and local-successor inventory
research. It proposes a fresh runtime probe, not a browser or model-quality result.

The next requirement is to run the actual pinned Decision3 Edge model on GB10,
with all four invented questions intact, including the singleton, at both short
and the previously defined 4,579-token maximum long input. The CPU5/CPU10 receipts
and source profiles remain immutable; a GPU result cannot inherit their runtime
qualification. Both JevBench scores and the full browser comparison remain pending
for the newly trained 109-row candidate.

| Reference | Mechanism and source evidence | Decision |
| --- | --- | --- |
| Publisher Decision3, Apache-2.0 | The locally authenticated `cab71e6ad3461b8086d51ddb572c656babd63b0b` `modeling_d3.py` delegates AutoModel construction to `d3_runtime.py:D3`; its `probabilities` uses final-position FP32 readout, legal-code masking and temperature 1. The manifest records Apache-2.0 for runtime and weights. | Use this exact revision through the existing project provider. No moving-main upgrade or encoder rewrite. |
| Existing project, MIT | `src/s1/evaluation/decision3.py:Decision3Backend` already selects CUDA BF16, full package verification and native batch size 8. The existing CPU probe supplies invented token construction and response checks, but its placement and observer contracts explicitly require CPU. | Extend the existing integration with a separately owned GPU actor. Keep batch size 8; four requested questions form one actual four-row partial batch. Reuse only device-independent helpers. |
| PyTorch CUDA documentation | GPU operations are asynchronous; host timings need synchronization. The installed Torch `2.13.0+cu130` passed a bounded invented BF16 operation on SM12.1, but this proves neither full-model kernel compatibility nor model speed. | Fence before timing and after native forward, readout replay and cleanup. Record real selected package/module origins. |

The publisher's newer upstream server and vLLM engine were found in official
search results. They are separate implementations and revisions, so they cannot
replace this fixed native baseline without a new comparison. The pinned local
runtime source, not those newer documentation claims, determines the probe.

The new actor must authenticate its selected sources, full local model cache and
runtime before model import; use the selected Python3.13 interpreter and overlay;
remain offline; bind boot, host memory, GPU UUID and its actual owned cgroup;
retain conservative memory admission and no swap credit; and preserve all native
heads and their order. A distinct bounded unit must contain the actor and its
children. No existing CPU plan is edited or relabelled.

Each complete call keeps the original 20-second limit, including parse/prepare,
forward, CUDA fences, every legal probability/ordered-argmax comparison, source
checks and durable evidence. The original whole-actor 120-second watchdog includes
construction and final cleanup. Same-call final-norm observation must return None
and keep only the detached final-position features; replay on the proper device
must use the unchanged FP32 readout. Failures remain failures, with no warmup or
second run silently substituted for the first result.

The initial scope contains no browser input, teacher label, optimizer or quality
claim. Successful short/long execution is a prerequisite for separate new-delta
numerical qualification and the complete ten-case V7 browser comparison. The
Heaven accuracy/calibration suite and probability-coherence suite must be freshly
scored and reported separately for a newly deployed candidate.

References read on 2026-10-25:

- [Pinned publisher runtime](https://huggingface.co/vllm-sr/d3-edge/blob/cab71e6ad3461b8086d51ddb572c656babd63b0b/d3_runtime.py), also inspected from the authenticated local snapshot. The pinned README web fetch was unavailable; licensing was read from the pinned local manifest.
- [Publisher current AutoModel wrapper](https://huggingface.co/vllm-sr/d3-edge/blob/main/modeling_d3.py), discovery only; not selected as execution bytes.
- [Publisher current native server](https://huggingface.co/vllm-sr/d3-edge/blob/main/d3_server.py), discovery only.
- [PyTorch CUDA asynchronous execution](https://docs.pytorch.org/docs/main/notes/cuda.html#asynchronous-execution).
- [Existing GX10 capacity research](jev-decision3-local-successor-inventory.md).

The first actual GPU attempt constructed the tracked model in about 3.40 seconds,
but its invented `operation/target/text/singleton` questions were then rejected by
the browser-only adapter before any observed encoder forward. That adapter also
requires browser operation semantics and corresponding named target heads. This
is a diagnostic routing error, not a Decision3 answer or speed result. The failed
attempt remains quarantined, with its original primary and cleanup errors. The
external owner completed scoped process/cgroup cleanup; no success is inferred
from construction alone.

The correction is to validate these unchanged diagnostic questions normally as
`Decision3NativeRequest` and call the genuine provider directly. The production
browser adapter keeps its stricter schema. The data-free regression in
`tests/test_jev_decision3_native.py` proves both the browser refusal before any
backend call and exact ordered native serialization at 38 and 255 target choices,
including the singleton. The new runtime attempt still needs fresh source pins,
actual CUDA forwards, numerical replay, deadlines and completed cleanup.
