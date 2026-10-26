# Actual four-core GX10 and Decision3 Edge CPU measurements

Four-core CPU inference now works for both the full Q8 S1 runner and the smaller Decision3 Edge. Q8 is still far too slow for the selected browser deadline; Edge completed a genuine small request quickly enough to proceed to the two separate JevBench campaigns. No new benchmark score or browser win follows from these probes.

The new selected worker uses four Cortex-X925 cores16–19, a400% CPU quota,2GiB memory and zero swap. Initial host admission remains reserve26,119,172,916 bytes plus the entire2GiB cap; runtime reserve and original read-only Comfy observer remain unchanged. CPU0 in the earlier one-thread probe is an A725, so this changes core type as well as parallelism. It is not a pure four-times scaling comparison.

| Actual probe | Input tokens | Measured seconds | Scope |
| --- | ---: | ---: | --- |
| Previous Q8 CPU0, one thread |68|No response by60.000893|Startup-inclusive deadline failure|
| Q8 CPU4, full context2048 |68|33.239148|Startup plus first valid response|
| Q8 CPU4, full context2048 |1667|464.663419|Post-warm request; valid finite logits|
| Q8 CPU4, compact context2048 |68|33.689711|Startup plus first valid response|
| Decision3 Edge CPU4 |103|0.252889|One genuine native request; model loading excluded|

Edge loaded in3.073386 seconds; its whole actor lasted11.490934 seconds, and the external controlled operation14.206463 seconds. It used pinned `vllm-sr/d3-edge` revision `cab71e6ad3461b8086d51ddb572c656babd63b0b`, batch1, BF16 backbone, original FP32 readout and publisher CPU reference kernels. One model constructor and one forward ran; CUDA stayed uninitialized. Full20-file source/model authentication happened inside the owned runtime before loading. No final tensor rehash was performed by this smoke. The original CUDA-only project backend remains unchanged.

The independent saved Q8 audit recomputed3.766100394 effective owned CPU cores over the completed full2048 case (1,875.191366 CPU seconds over497.91327094 wall seconds). Four sampled native tasks had affinity16–19 and changing CPU ticks. This verifies real parallel work, while sampled masks do not prove every thread's complete lifetime confinement. The full1667-token response alone exceeds20 seconds by a wide margin; remaining synthetic compact-cache research was deliberately cancelled to prioritize real small-model scoring. Compact long/8192/cache parity are incomplete, not failed numerical comparisons or deadline outcomes.

Cancellation sent exactly one pidfd-targeted SIGINT to the external controller after verifying its boot/PID/start/argv/executable/UID. Its existing finally path used the original authenticated unit close. The original closure's first scan saw a zombie and its second was empty, leaving `whole_unit_exit_confirmed=false`; that historical value remains unchanged. Later read-only scans through the exact original authority found two empty selected-cohort scans, stable absent-manager observations and an absent cgroup. They supplement the original closure rather than rewriting it. No ComfyUI operation or unrelated signal occurred.

Both probes reached the2GiB cgroup peak. Q8's last retained `memory.events` snapshot had max277,916 and oom/oom_kill/oom_group_kill0; Edge had max122 and the same zero OOM counters. These are sampled cap-pressure observations, not proof of the exact bottleneck or a continuous-time no-OOM guarantee. Edge's original closure completed with two empty scans and an absent cgroup.

The source-qualified Edge campaign is now running Benchmark Heaven public231 and then JevBench coherence MINI, one shared CPU model constructor, concurrency1, fresh outputs/cache, original pinned tasks and scorers. It declares the official180-second benchmark call default and6,500-second whole-campaign deadline; those are separate from browser20-second qualification. Suite latency distributions exclude model loading. Final scores require all231 public tasks and1,248 coherence requests, final source/model authentication and owned-unit closure. Use the companion live snapshot and existing exec session78352; do not restart from a stale state file.

Previously completed Gemma BF16 results remain separate:195/231(84.42%) Benchmark Heaven public accuracy and41.26/100 coherence MINI. Best genuine browser result remains4/10 for its earlier profile. Edge scores, Decision3 Lite execution, complete six-model comparison and victory remain pending at this snapshot. Source preparation for a distinct Lite3GiB CPU worker uses the stronger initial reserve-plus3GiB floor; it is not an admitted current run or a change to this2GiB worker.
