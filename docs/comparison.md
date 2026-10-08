# Accuracy, memory, and latency comparison

This report is generated from saved evidence. New campaign rows and historical Modal results retain their own model, precision, hardware, and workload identities.

Campaign: `20261007-jev-omni-matched-comparison`; status: **completed**.
No missing or blocked result is counted as a successful evaluation.

## Shared evaluation contract

BoolQ uses the same 128 fixed validation cases, with dataset SHA-256 `7bdb25dc3d8a0371bb73909fe55597de86f66e808bc42d1a5a1cb17d08f44d41`. Errors and rejected requests remain in the full denominator. Repeated seeds reuse this test set; they are not additional independent test examples.
Native media uses 52 examples: 32 MNIST images and 20 FSDD recordings (six audio speaker groups). Text-only models have unsupported native media. OCR/ASR pipelines must be measured separately with preprocessing time and memory. The historical M2 source-label oracle is diagnostic and is never ranked as media accuracy.

GB means decimal bytes / 1,000,000,000; GiB means bytes / 1,073,741,824. Model footprint, CUDA peak allocated, CUDA peak reserved, process-lifetime peak RSS, and GPU capacity are different quantities. The registered model tensor footprint can exclude auxiliary quantizer state; CUDA peaks include resident weights and the measured workload. Device used is an instantaneous device-wide observation, includes CUDA context and potentially other processes, and is not a per-process peak. Allocation below 8 GiB does not prove deployment on an 8 GiB GPU: reserved memory and device usage may be higher. Acceptance caps use decimal GB, so an 8.344 GB allocation fails an 8.000 GB cap even though it is 7.771 GiB. Historical 24 GB / 48 GB figures are nominal 12B weight estimates, not measured deployment minima. Quantization also retains some layers, scales, adapters, and activations at higher precision.

The configured Modal tier is A100-80GB, but the allocator may provide PCIe or SXM variants. The table preserves each observed GPU SKU. Accuracy aggregates preserve that hardware list; latency remains per model/workload and is not pooled across SKUs.

## Current campaign: identical BoolQ, native media, and memory harness

| Model / format | Run status | BoolQ accuracy | Native media accuracy | Recorded tensor footprint GB (GiB) | CUDA peak allocated / reserved GB (GiB) | Device used GB (GiB) | BoolQ mean / p99 ms | Load C1 / C16 p99 ms | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-base-bf16 / bfloat16 | completed | 86.72% (111/128; valid 128) | 65.38% (34/52; valid 52) | 23.92 (22.28) | 24.45 (22.77) / 24.84 (23.13) | 25.72 (23.96) | 87.67 / 92.94 | 103.66 / 1680.92 | NVIDIA A100 80GB PCIe |
| gemma-ce128-s0-bf16 / bfloat16 | completed | 90.62% (116/128; valid 128) | 65.38% (34/52; valid 52) | 23.96 (22.32) | 24.50 (22.82) / 24.89 (23.18) | 25.77 (24.00) | 155.92 / 167.18 | 167.63 / 2577.48 | NVIDIA A100-SXM4-80GB |
| gemma-ce128-s0-int8 / int8 / bfloat16 | completed | 92.19% (118/128; valid 128) | 61.54% (32/52; valid 52) | 13.06 (12.17) | 13.90 (12.94) / 16.40 (15.27) | 17.29 (16.10) | 447.38 / 705.97 | 527.93 / 8242.73 | NVIDIA A100-SXM4-80GB |
| gemma-ce128-s0-nf4 / nf4 / bfloat16 | completed | 85.94% (110/128; valid 128) | 55.77% (29/52; valid 52) | 7.61 (7.09) | 8.34 (7.77) / 8.56 (7.97) | 9.57 (8.92) | 183.95 / 190.13 | 213.75 / 3203.85 | NVIDIA A100-SXM4-80GB |
| gemma-ce128-s1-bf16 / bfloat16 | completed | 91.41% (117/128; valid 128) | 61.54% (32/52; valid 52) | 23.96 (22.32) | 24.50 (22.82) / 24.89 (23.18) | 25.77 (24.00) | 127.49 / 222.09 | 293.24 / 2057.94 | NVIDIA A100-SXM4-80GB |
| gemma-ce128-s1-int8 / int8 / bfloat16 | completed | 90.62% (116/128; valid 128) | 63.46% (33/52; valid 52) | 13.06 (12.17) | 13.90 (12.94) / 16.08 (14.98) | 16.98 (15.81) | 595.10 / 1108.96 | 622.10 / 9415.70 | NVIDIA A100-SXM4-80GB |
| gemma-ce128-s1-nf4 / nf4 / bfloat16 | completed | 86.72% (111/128; valid 128) | 57.69% (30/52; valid 52) | 7.61 (7.09) | 8.34 (7.77) / 8.46 (7.88) | 9.34 (8.70) | 159.82 / 172.95 | 197.48 / 3258.40 | NVIDIA A100 80GB PCIe |
| gemma-ce128-s2-bf16 / bfloat16 | completed | 91.41% (117/128; valid 128) | 63.46% (33/52; valid 52) | 23.96 (22.32) | 24.50 (22.82) / 24.90 (23.19) | 25.78 (24.01) | 130.54 / 142.50 | 148.78 / 2290.12 | NVIDIA A100-SXM4-80GB |
| gemma-ce128-s2-int8 / int8 / bfloat16 | completed | 91.41% (117/128; valid 128) | 61.54% (32/52; valid 52) | 13.06 (12.17) | 13.89 (12.94) / 16.08 (14.98) | 16.98 (15.81) | 440.44 / 549.87 | 467.76 / 7431.93 | NVIDIA A100-SXM4-80GB |
| gemma-ce128-s2-nf4 / nf4 / bfloat16 | completed | 89.06% (114/128; valid 128) | 55.77% (29/52; valid 52) | 7.61 (7.09) | 8.34 (7.77) / 8.56 (7.97) | 9.44 (8.79) | 226.28 / 249.27 | 244.42 / 3718.61 | NVIDIA A100-SXM4-80GB |
| laya-general / provider | completed | 82.03% (105/128; valid 128) | unsupported native media | 1.69 (1.57) | 2.77 (2.58) / 3.08 (2.87) | 3.97 (3.69) | 32.99 / 65.43 | 48.53 / 768.88 | NVIDIA A100-SXM4-80GB |
| decider-local / bfloat16 | completed | 88.28% (113/128; valid 128) | unsupported native media | 3.76 (3.51) | 4.25 (3.95) / 4.39 (4.09) | 5.28 (4.91) | 75.65 / 126.85 | 106.69 / 1454.40 | NVIDIA A100-SXM4-80GB |
| kev-local / bfloat16 | completed | 71.88% (92/128; valid 128) | unsupported native media | 1.51 (1.40) | 2.00 (1.87) / 2.18 (2.03) | 3.07 (2.86) | 120.34 / 135.49 | 147.55 / 2216.28 | NVIDIA A100 80GB PCIe |
| agentjev-local / float32 | completed | 72.66% (93/128; valid 128) | unsupported native media | 2.39 (2.23) | 3.70 (3.45) / 3.90 (3.63) | 4.78 (4.45) | 40.79 / 63.14 | 48.49 / 930.34 | NVIDIA A100-SXM4-80GB |
| gemma-ce128-s0-nf4-recovered / nf4 / bfloat16 | completed | 88.28% (113/128; valid 128) | 55.77% (29/52; valid 52) | 7.61 (7.09) | 8.34 (7.77) / 8.56 (7.97) | 9.44 (8.79) | 192.18 / 201.70 | 206.51 / 3258.32 | NVIDIA A100-SXM4-80GB |
| gemma-ce128-s1-nf4-recovered / nf4 / bfloat16 | completed | 90.62% (116/128; valid 128) | 57.69% (30/52; valid 52) | 7.61 (7.09) | 8.34 (7.77) / 8.56 (7.97) | 9.44 (8.79) | 182.39 / 191.54 | 195.69 / 3120.67 | NVIDIA A100-SXM4-80GB |
| gemma-ce128-s2-nf4-recovered / nf4 / bfloat16 | completed | 92.19% (118/128; valid 128) | 55.77% (29/52; valid 52) | 7.61 (7.09) | 8.34 (7.77) / 8.56 (7.97) | 9.44 (8.79) | 173.40 / 182.12 | 189.39 / 3014.78 | NVIDIA A100-SXM4-80GB |
| clef-local / bfloat16 | completed | 89.84% (115/128; valid 128) | supported subset 84.38% (27/32; valid 32); 20/52 unsupported | 54.97 (51.19) | 55.66 (51.84) / 55.86 (52.02) | 56.75 (52.85) | 185.02 / 245.80 | 252.91 / 3259.15 | NVIDIA A100-SXM4-80GB |
| jev-omni-local / bfloat16 | completed | 87.50% (112/128; valid 128) | 57.69% (30/52; valid 52) | 23.92 (22.28) | 24.38 (22.71) / 24.58 (22.89) | 25.47 (23.72) | 81.29 / 111.94 | 133.79 / 1504.21 | NVIDIA A100-SXM4-80GB |

Decider and Kev use eager PyTorch reference execution; their live worker logs report missing optional causal-conv1d and flash-linear-attention kernels. Published CUDA-graph or compiled GPU timings describe different runtimes. AgentJev uses the pinned current coding checkpoint, not the older typed-decision checkpoint. INT8 timing includes the logging behavior of its recorded run. Early verbose cast-warning attempts remain in the audit trail; newer receipts identify the once-per-process logging filter. The filter changes logging only, and the reruns preserve accuracy. Runtime and hardware variation prevent attributing a timing difference solely to logging.

The selected Decider and Kev rows use corrected full-module tensor footprints. Earlier partial-module footprints remain only in superseded attempts in the audit index; the first Kev footprint excluded its small FP32 pointer head. CUDA peaks cover the full process in both versions. Treat measured CUDA allocation and device usage as separate deployment-memory observations.

Clef is the pinned full 27B release, evaluated in the shared PyTorch reference runtime without optional flash-linear-attention or causal-conv1d acceleration. These timings describe this local runtime, not an optimized hosted Clef service or the smaller Clef-Flash model.

Jev-Omni uses the pinned merged BF16 multimodal backbone and FP32 stored decision head, with BF16 autocast and one native forward per question. Its recorded runtime is PyTorch 2.10.0 / Transformers 5.17.0; older campaign rows retain their own package versions. The adapter computes softmax in float32 to preserve a complete normalized candidate distribution and passes existing 16 kHz float32 audio directly, avoiding the publisher's temporary WAV/ffmpeg roundtrip. Source code, weights and processor share the same immutable revision. No test fitting, CUDA graphs, compilation or prefix cache is used. These measurements include native media preprocessing and must not be compared as identical runtimes to the publisher's optimized H200 timings.

## Current native-media retention

| Model | MNIST | FSDD | Media quality and support | Media mean / p99 ms |
| --- | --- | --- | --- | --- |
| gemma-base-bf16 | 28/32 (87.50%) | 6/20 (30.00%) | 65.38% (34/52; valid 52) | 106.58 / 118.55 |
| gemma-ce128-s0-bf16 | 28/32 (87.50%) | 6/20 (30.00%) | 65.38% (34/52; valid 52) | 163.50 / 173.94 |
| gemma-ce128-s0-int8 | 28/32 (87.50%) | 4/20 (20.00%) | 61.54% (32/52; valid 52) | 449.03 / 476.77 |
| gemma-ce128-s0-nf4 | 28/32 (87.50%) | 1/20 (5.00%) | 55.77% (29/52; valid 52) | 194.45 / 204.26 |
| gemma-ce128-s1-bf16 | 28/32 (87.50%) | 4/20 (20.00%) | 61.54% (32/52; valid 52) | 193.20 / 337.40 |
| gemma-ce128-s1-int8 | 28/32 (87.50%) | 5/20 (25.00%) | 63.46% (33/52; valid 52) | 547.94 / 676.69 |
| gemma-ce128-s1-nf4 | 29/32 (90.62%) | 1/20 (5.00%) | 57.69% (30/52; valid 52) | 173.48 / 190.23 |
| gemma-ce128-s2-bf16 | 28/32 (87.50%) | 5/20 (25.00%) | 63.46% (33/52; valid 52) | 140.13 / 154.83 |
| gemma-ce128-s2-int8 | 29/32 (90.62%) | 3/20 (15.00%) | 61.54% (32/52; valid 52) | 446.81 / 489.27 |
| gemma-ce128-s2-nf4 | 28/32 (87.50%) | 1/20 (5.00%) | 55.77% (29/52; valid 52) | 234.77 / 247.77 |
| gemma-ce128-s0-nf4-recovered | 28/32 (87.50%) | 1/20 (5.00%) | 55.77% (29/52; valid 52) | 204.29 / 212.32 |
| gemma-ce128-s1-nf4-recovered | 29/32 (90.62%) | 1/20 (5.00%) | 57.69% (30/52; valid 52) | 194.38 / 203.72 |
| gemma-ce128-s2-nf4-recovered | 28/32 (87.50%) | 1/20 (5.00%) | 55.77% (29/52; valid 52) | 184.47 / 195.50 |
| clef-local | 27/32 (84.38%) | unsupported (20 cases) | supported subset 84.38% (27/32; valid 32); 20/52 unsupported | 335.77 / 565.13 |
| jev-omni-local | 28/32 (87.50%) | 2/20 (10.00%) | 57.69% (30/52; valid 52) | 100.70 / 114.71 |

These are the same raw images and recordings in every current model row. Text-only providers remain unsupported for native media; no oracle description, OCR, or ASR output is inserted into these scores.
Models with partial native support report accuracy over their eligible subset and expose unsupported cases separately. An image-only result over 32 MNIST cases is not a full 52-case image/audio score; unsupported audio has no accuracy. Errors within a supported task remain in that task's denominator. Media latency uses successful supported requests, so an image-only timing covers a different modality mix from Gemma's combined image/audio timing.


Accuracy targets (>89% or ≥90%) and memory targets (~12 GB or 6–8 GB) are acceptance goals. A format name, successful model load, or nominal bit count does not establish that a target was met. Each quantized CE adapter must be evaluated in its own row.

## Current campaign load results

Load percentiles include the workload identified below; they are separate from serial BoolQ evaluation time. Successful-request percentiles exclude failures, which remain visible in request counts. Small request counts provide screening estimates of p99.
The current harness runs one model in one process and serializes inference with a service lock, without batching or request-result caching. Loopback HTTP excludes WAN latency. Each model receives six cells of 128 requests (768 total): closed-loop concurrency 1, 4, 16, and 64, plus fixed and Poisson arrivals at 5 requests/second with at most 16 outstanding requests. The latency SLO is 1,000 ms. These measurements include this runtime's queueing; they do not estimate an optimized provider's batched throughput.

| Model | Arrival | Concurrency | Successful / requests | p50 ms | p95 ms | p99 ms | SLO misses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-base-bf16 | closed_loop | 1 | 128 / 128 | 92.31 | 99.48 | 103.66 | 0 |
| gemma-base-bf16 | closed_loop | 4 | 128 / 128 | 378.91 | 403.26 | 440.59 | 0 |
| gemma-base-bf16 | closed_loop | 16 | 128 / 128 | 1517.13 | 1668.72 | 1680.92 | 119 |
| gemma-base-bf16 | closed_loop | 64 | 128 / 128 | 6067.96 | 6461.07 | 6509.69 | 121 |
| gemma-base-bf16 | fixed | 16 | 128 / 128 | 97.17 | 102.34 | 106.43 | 0 |
| gemma-base-bf16 | poisson | 16 | 128 / 128 | 106.05 | 308.47 | 364.62 | 0 |
| gemma-ce128-s0-bf16 | closed_loop | 1 | 128 / 128 | 153.51 | 160.89 | 167.63 | 0 |
| gemma-ce128-s0-bf16 | closed_loop | 4 | 128 / 128 | 617.16 | 640.14 | 645.16 | 0 |
| gemma-ce128-s0-bf16 | closed_loop | 16 | 128 / 128 | 2461.18 | 2554.68 | 2577.48 | 122 |
| gemma-ce128-s0-bf16 | closed_loop | 64 | 128 / 128 | 9835.55 | 10130.39 | 10519.98 | 126 |
| gemma-ce128-s0-bf16 | fixed | 16 | 128 / 128 | 161.09 | 172.82 | 177.71 | 0 |
| gemma-ce128-s0-bf16 | poisson | 16 | 128 / 128 | 338.14 | 1009.01 | 1195.33 | 7 |
| gemma-ce128-s0-int8 | closed_loop | 1 | 128 / 128 | 447.97 | 471.80 | 527.93 | 0 |
| gemma-ce128-s0-int8 | closed_loop | 4 | 128 / 128 | 1685.25 | 1955.56 | 2052.80 | 126 |
| gemma-ce128-s0-int8 | closed_loop | 16 | 128 / 128 | 7050.27 | 8223.57 | 8242.73 | 126 |
| gemma-ce128-s0-int8 | closed_loop | 64 | 128 / 128 | 27597.22 | 28754.42 | 28787.17 | 127 |
| gemma-ce128-s0-int8 | fixed | 16 | 128 / 128 | 14715.67 | 28080.53 | 29276.62 | 125 |
| gemma-ce128-s0-int8 | poisson | 16 | 128 / 128 | 15539.62 | 29081.76 | 30318.62 | 126 |
| gemma-ce128-s0-nf4 | closed_loop | 1 | 128 / 128 | 189.08 | 200.41 | 213.75 | 0 |
| gemma-ce128-s0-nf4 | closed_loop | 4 | 128 / 128 | 767.55 | 797.01 | 823.03 | 0 |
| gemma-ce128-s0-nf4 | closed_loop | 16 | 128 / 128 | 3110.49 | 3182.88 | 3203.85 | 124 |
| gemma-ce128-s0-nf4 | closed_loop | 64 | 128 / 128 | 12159.92 | 12539.03 | 12563.58 | 124 |
| gemma-ce128-s0-nf4 | fixed | 16 | 128 / 128 | 193.10 | 197.41 | 199.52 | 0 |
| gemma-ce128-s0-nf4 | poisson | 16 | 128 / 128 | 1683.47 | 2728.94 | 2853.70 | 88 |
| gemma-ce128-s1-bf16 | closed_loop | 1 | 128 / 128 | 148.70 | 230.94 | 293.24 | 0 |
| gemma-ce128-s1-bf16 | closed_loop | 4 | 128 / 128 | 479.51 | 528.31 | 539.51 | 0 |
| gemma-ce128-s1-bf16 | closed_loop | 16 | 128 / 128 | 1987.07 | 2052.73 | 2057.94 | 121 |
| gemma-ce128-s1-bf16 | closed_loop | 64 | 128 / 128 | 7665.26 | 7902.45 | 8230.78 | 124 |
| gemma-ce128-s1-bf16 | fixed | 16 | 128 / 128 | 120.89 | 129.96 | 150.66 | 0 |
| gemma-ce128-s1-bf16 | poisson | 16 | 128 / 128 | 324.69 | 1878.92 | 2055.34 | 39 |
| gemma-ce128-s1-int8 | closed_loop | 1 | 128 / 128 | 558.42 | 600.35 | 622.10 | 0 |
| gemma-ce128-s1-int8 | closed_loop | 4 | 128 / 128 | 2142.06 | 2305.80 | 2508.49 | 127 |
| gemma-ce128-s1-int8 | closed_loop | 16 | 128 / 128 | 9017.53 | 9366.16 | 9415.70 | 127 |
| gemma-ce128-s1-int8 | closed_loop | 64 | 128 / 128 | 36687.90 | 37643.66 | 37768.86 | 127 |
| gemma-ce128-s1-int8 | fixed | 16 | 128 / 128 | 23663.47 | 44612.57 | 46538.10 | 126 |
| gemma-ce128-s1-int8 | poisson | 16 | 128 / 128 | 24411.38 | 44777.56 | 46440.64 | 127 |
| gemma-ce128-s1-nf4 | closed_loop | 1 | 128 / 128 | 180.05 | 186.33 | 197.48 | 1 |
| gemma-ce128-s1-nf4 | closed_loop | 4 | 128 / 128 | 672.12 | 739.87 | 751.94 | 0 |
| gemma-ce128-s1-nf4 | closed_loop | 16 | 128 / 128 | 2717.82 | 3224.33 | 3258.40 | 123 |
| gemma-ce128-s1-nf4 | closed_loop | 64 | 128 / 128 | 11353.21 | 12112.38 | 12230.30 | 125 |
| gemma-ce128-s1-nf4 | fixed | 16 | 128 / 128 | 181.80 | 194.13 | 200.79 | 0 |
| gemma-ce128-s1-nf4 | poisson | 16 | 128 / 128 | 1125.40 | 1862.12 | 2029.53 | 67 |
| gemma-ce128-s2-bf16 | closed_loop | 1 | 128 / 128 | 134.93 | 140.79 | 148.78 | 0 |
| gemma-ce128-s2-bf16 | closed_loop | 4 | 128 / 128 | 553.32 | 571.03 | 575.82 | 0 |
| gemma-ce128-s2-bf16 | closed_loop | 16 | 128 / 128 | 2219.95 | 2277.28 | 2290.12 | 122 |
| gemma-ce128-s2-bf16 | closed_loop | 64 | 128 / 128 | 8777.20 | 9173.29 | 9509.85 | 126 |
| gemma-ce128-s2-bf16 | fixed | 16 | 128 / 128 | 137.69 | 143.31 | 150.76 | 0 |
| gemma-ce128-s2-bf16 | poisson | 16 | 128 / 128 | 268.23 | 725.63 | 897.24 | 0 |
| gemma-ce128-s2-int8 | closed_loop | 1 | 128 / 128 | 437.08 | 453.10 | 467.76 | 0 |
| gemma-ce128-s2-int8 | closed_loop | 4 | 128 / 128 | 1765.70 | 1846.31 | 1876.78 | 126 |
| gemma-ce128-s2-int8 | closed_loop | 16 | 128 / 128 | 6831.27 | 7370.03 | 7431.93 | 126 |
| gemma-ce128-s2-int8 | closed_loop | 64 | 128 / 128 | 26716.12 | 28636.18 | 28688.12 | 127 |
| gemma-ce128-s2-int8 | fixed | 16 | 128 / 128 | 15329.02 | 32115.56 | 33433.79 | 126 |
| gemma-ce128-s2-int8 | poisson | 16 | 128 / 128 | 15198.77 | 27649.96 | 28775.12 | 126 |
| gemma-ce128-s2-nf4 | closed_loop | 1 | 128 / 128 | 226.58 | 235.71 | 244.42 | 0 |
| gemma-ce128-s2-nf4 | closed_loop | 4 | 128 / 128 | 911.14 | 926.35 | 928.85 | 0 |
| gemma-ce128-s2-nf4 | closed_loop | 16 | 128 / 128 | 3674.94 | 3713.89 | 3718.61 | 124 |
| gemma-ce128-s2-nf4 | closed_loop | 64 | 128 / 128 | 14608.59 | 14816.52 | 14823.61 | 125 |
| gemma-ce128-s2-nf4 | fixed | 16 | 128 / 128 | 2733.87 | 4644.39 | 4811.68 | 109 |
| gemma-ce128-s2-nf4 | poisson | 16 | 128 / 128 | 3584.34 | 5730.00 | 6084.19 | 102 |
| laya-general | closed_loop | 1 | 128 / 128 | 38.39 | 43.03 | 48.53 | 0 |
| laya-general | closed_loop | 4 | 128 / 128 | 163.59 | 187.34 | 203.91 | 0 |
| laya-general | closed_loop | 16 | 128 / 128 | 647.23 | 725.45 | 768.88 | 0 |
| laya-general | closed_loop | 64 | 128 / 128 | 2524.88 | 2958.89 | 3010.08 | 110 |
| laya-general | fixed | 16 | 128 / 128 | 39.21 | 44.32 | 48.62 | 0 |
| laya-general | poisson | 16 | 128 / 128 | 39.57 | 82.77 | 99.78 | 0 |
| decider-local | closed_loop | 1 | 128 / 128 | 79.79 | 93.36 | 106.69 | 0 |
| decider-local | closed_loop | 4 | 128 / 128 | 325.16 | 358.73 | 376.17 | 0 |
| decider-local | closed_loop | 16 | 128 / 128 | 1302.04 | 1426.80 | 1454.40 | 118 |
| decider-local | closed_loop | 64 | 128 / 128 | 5041.69 | 5679.77 | 5721.96 | 119 |
| decider-local | fixed | 16 | 128 / 128 | 79.97 | 92.38 | 103.28 | 0 |
| decider-local | poisson | 16 | 128 / 128 | 92.69 | 251.73 | 307.50 | 0 |
| kev-local | closed_loop | 1 | 128 / 128 | 125.97 | 137.33 | 147.55 | 0 |
| kev-local | closed_loop | 4 | 128 / 128 | 512.32 | 581.60 | 942.14 | 0 |
| kev-local | closed_loop | 16 | 128 / 128 | 2041.57 | 2189.12 | 2216.28 | 121 |
| kev-local | closed_loop | 64 | 128 / 128 | 8084.43 | 8666.09 | 8673.90 | 122 |
| kev-local | fixed | 16 | 128 / 128 | 128.17 | 138.74 | 148.61 | 0 |
| kev-local | poisson | 16 | 128 / 128 | 204.21 | 578.98 | 689.72 | 0 |
| agentjev-local | closed_loop | 1 | 128 / 128 | 44.52 | 47.31 | 48.49 | 0 |
| agentjev-local | closed_loop | 4 | 128 / 128 | 186.20 | 196.50 | 213.50 | 0 |
| agentjev-local | closed_loop | 16 | 128 / 128 | 738.87 | 902.40 | 930.34 | 0 |
| agentjev-local | closed_loop | 64 | 128 / 128 | 2920.14 | 3439.50 | 3572.88 | 112 |
| agentjev-local | fixed | 16 | 128 / 128 | 47.00 | 49.72 | 53.37 | 0 |
| agentjev-local | poisson | 16 | 128 / 128 | 48.02 | 118.41 | 133.49 | 0 |
| gemma-ce128-s0-nf4-recovered | closed_loop | 1 | 128 / 128 | 196.17 | 203.19 | 206.51 | 0 |
| gemma-ce128-s0-nf4-recovered | closed_loop | 4 | 128 / 128 | 788.90 | 801.39 | 805.96 | 0 |
| gemma-ce128-s0-nf4-recovered | closed_loop | 16 | 128 / 128 | 3141.63 | 3230.93 | 3258.32 | 124 |
| gemma-ce128-s0-nf4-recovered | closed_loop | 64 | 128 / 128 | 12563.27 | 13007.30 | 13033.31 | 125 |
| gemma-ce128-s0-nf4-recovered | fixed | 16 | 128 / 128 | 196.49 | 225.86 | 229.67 | 0 |
| gemma-ce128-s0-nf4-recovered | poisson | 16 | 128 / 128 | 1907.59 | 2940.52 | 3092.07 | 87 |
| gemma-ce128-s1-nf4-recovered | closed_loop | 1 | 128 / 128 | 187.42 | 194.36 | 195.69 | 0 |
| gemma-ce128-s1-nf4-recovered | closed_loop | 4 | 128 / 128 | 757.55 | 769.39 | 773.18 | 0 |
| gemma-ce128-s1-nf4-recovered | closed_loop | 16 | 128 / 128 | 3052.05 | 3103.72 | 3120.67 | 124 |
| gemma-ce128-s1-nf4-recovered | closed_loop | 64 | 128 / 128 | 12296.92 | 12534.29 | 12547.75 | 124 |
| gemma-ce128-s1-nf4-recovered | fixed | 16 | 128 / 128 | 193.39 | 218.63 | 231.89 | 0 |
| gemma-ce128-s1-nf4-recovered | poisson | 16 | 128 / 128 | 1682.66 | 2686.07 | 2828.98 | 87 |
| gemma-ce128-s2-nf4-recovered | closed_loop | 1 | 128 / 128 | 182.20 | 187.54 | 189.39 | 0 |
| gemma-ce128-s2-nf4-recovered | closed_loop | 4 | 128 / 128 | 739.92 | 765.64 | 772.40 | 0 |
| gemma-ce128-s2-nf4-recovered | closed_loop | 16 | 128 / 128 | 2954.04 | 3000.72 | 3014.78 | 124 |
| gemma-ce128-s2-nf4-recovered | closed_loop | 64 | 128 / 128 | 11583.08 | 11933.83 | 11939.12 | 124 |
| gemma-ce128-s2-nf4-recovered | fixed | 16 | 128 / 128 | 179.44 | 185.85 | 191.83 | 0 |
| gemma-ce128-s2-nf4-recovered | poisson | 16 | 128 / 128 | 1016.26 | 1738.34 | 1899.61 | 65 |
| clef-local | closed_loop | 1 | 128 / 128 | 191.30 | 213.89 | 252.91 | 0 |
| clef-local | closed_loop | 4 | 128 / 128 | 733.43 | 819.03 | 1015.72 | 2 |
| clef-local | closed_loop | 16 | 128 / 128 | 2921.19 | 3226.86 | 3259.15 | 123 |
| clef-local | closed_loop | 64 | 128 / 128 | 11769.86 | 12231.95 | 12271.34 | 124 |
| clef-local | fixed | 16 | 128 / 128 | 194.92 | 275.17 | 443.24 | 0 |
| clef-local | poisson | 16 | 128 / 128 | 1251.75 | 1969.64 | 2079.98 | 68 |
| jev-omni-local | closed_loop | 1 | 128 / 128 | 83.89 | 90.82 | 133.79 | 0 |
| jev-omni-local | closed_loop | 4 | 128 / 128 | 343.39 | 376.89 | 392.15 | 0 |
| jev-omni-local | closed_loop | 16 | 128 / 128 | 1419.97 | 1484.35 | 1504.21 | 118 |
| jev-omni-local | closed_loop | 64 | 128 / 128 | 5642.65 | 5966.29 | 5976.50 | 119 |
| jev-omni-local | fixed | 16 | 128 / 128 | 90.24 | 99.98 | 113.78 | 0 |
| jev-omni-local | poisson | 16 | 128 / 128 | 101.23 | 291.61 | 379.29 | 0 |

## Quantization and target checks

The implemented paths use bitsandbytes INT8 or NF4 with BF16 surrounding layers and NF4 compute. The INT8 kernel internally casts activations to FP16 before quantizing them. The tied input embedding / LM head and media projections remain dense because the candidate readout indexes floating-point head rows. Existing CE-128 adapters are reused across the original three formats and all three predeclared seeds; no seed is selected using test accuracy. Any NF4 recovery extension is labeled separately because it adds training updates and changes adapter weights. CPU offload is disabled. FP8 is not an implemented path in this campaign. The [quantized inference guide](quantization.md) documents local CLI and Modal serving flags.

The pinned checkpoint contains 11,959,730,176 parameters: 10,899,947,520 eligible linear parameters and 1,059,782,656 parameters retained at BF16, including a 1,006,632,960-parameter tied embedding/head counted once. Therefore the raw parameter-only floors are **23.919 GB BF16**, **13.020 GB INT8**, and **7.570 GB NF4**. These are derived counts, not GPU measurements; quantizer metadata, adapters, activations and workspaces add memory. A strict 12 GB INT8 allocation target cannot be met by this linear-only conversion. The NF4 8 GB target has little room beyond its raw parameters and must be checked against live peak allocation. See the [metadata-only sizing receipt](validation/2026-10-01/quantization-sizing.json).

| Quantized model | Accuracy/memory target status | BoolQ | Peak allocated GB (GiB) | Target cap GB | Accuracy ≥90% | Memory target met |
| --- | --- | --- | --- | --- | --- | --- |
| gemma-ce128-s0-int8 | failed | 92.19% (118/128; valid 128) | 13.90 (12.94) | 12.00 | True | False |
| gemma-ce128-s0-nf4 | failed | 85.94% (110/128; valid 128) | 8.34 (7.77) | 8.00 | False | False |
| gemma-ce128-s1-int8 | failed | 90.62% (116/128; valid 128) | 13.90 (12.94) | 12.00 | True | False |
| gemma-ce128-s1-nf4 | failed | 86.72% (111/128; valid 128) | 8.34 (7.77) | 8.00 | False | False |
| gemma-ce128-s2-int8 | failed | 91.41% (117/128; valid 128) | 13.89 (12.94) | 12.00 | True | False |
| gemma-ce128-s2-nf4 | failed | 89.06% (114/128; valid 128) | 8.34 (7.77) | 8.00 | False | False |
| gemma-ce128-s0-nf4-recovered | failed | 88.28% (113/128; valid 128) | 8.34 (7.77) | 8.00 | False | False |
| gemma-ce128-s1-nf4-recovered | failed | 90.62% (116/128; valid 128) | 8.34 (7.77) | 8.00 | True | False |
| gemma-ce128-s2-nf4-recovered | failed | 92.19% (118/128; valid 128) | 8.34 (7.77) | 8.00 | True | False |

| CE-128 format | 3-seed evidence | Completed seeds | Mean accuracy [min, max] | Worst peak allocated GB (GiB) | Paired mean delta, pp / reference | Accuracy/memory target | All run stages complete | Observed GPU SKUs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bf16 | complete | [0, 1, 2] | 91.15% [90.62%, 91.41%] | 24.50 (22.82) | not measured | not_applicable | True | NVIDIA A100-SXM4-80GB |
| int8 | complete | [0, 1, 2] | 91.41% [90.62%, 92.19%] | 13.90 (12.94) | +0.260 vs BF16 | failed | True | NVIDIA A100-SXM4-80GB |
| nf4 | complete | [0, 1, 2] | 87.24% [85.94%, 89.06%] | 8.34 (7.77) | -3.906 vs BF16 | failed | True | NVIDIA A100 80GB PCIe; NVIDIA A100-SXM4-80GB |
| nf4-recovered | complete | [0, 1, 2] | 90.36% [88.28%, 92.19%] | 8.34 (7.77) | +3.125 vs original NF4 | failed | True | NVIDIA A100-SXM4-80GB |

Three-seed aggregates require all predeclared seeds (0, 1, 2), complete BoolQ coverage, and matching GPU identity or configured GPU tier. Observed SKUs remain explicit. Missing seeds are never dropped from an average. Memory uses the worst peak across seeds, and paired deltas compare the same seed against its newly measured BF16 row (or original NF4 for the extra-training recovery). These are three models evaluated on the same 128 examples, not 384 independent test examples. Accuracy/memory target status is separate from completion of every load stage and its latency SLOs.

**Exploratory NF4 recovery:** an additional fixed 100-update QLoRA continuation is a separate method, using the 256-case training pool and 16-case calibration split. The parent CE adapter has 100 optimizer updates; the fixed continuation adds 100 at batch size 1 (100 training cases, less than one epoch), for a 200-update recipe. Its adapter weights differ from the original CE adapter, and it must not be described as a controlled quantization-only comparison. The original NF4 scores and every seed remain visible.
The combined summary includes quality results and artifact hashes for all three recovery seeds. Individual recovery receipts and prediction artifacts are not included in this repository snapshot.

## Current paired BoolQ comparisons

Deltas are candidate minus reference, in percentage points. Every computed interval uses the same 128 recorded requests and source-group bootstrap; the 95% intervals are descriptive and unadjusted for multiple comparisons. Same-seed quantization comparisons additionally require identical checkpoint and adapter identities. Explicit recovery pairs permit a changed adapter only when its recorded parent digest and fixed training-data recipe are verified; these compare extra training, not quantization alone. Missing or mismatched evidence does not produce an interval.

| Reference | Candidate | Comparison status | Accuracy delta, pp | 95% CI, pp | Source groups |
| --- | --- | --- | --- | --- | --- |
| gemma-ce128-s0-bf16 | gemma-ce128-s0-int8 | computed | +1.562 | [+0.000, +3.906] | 128 |
| gemma-ce128-s0-bf16 | gemma-ce128-s0-nf4 | computed | -4.688 | [-9.375, +0.000] | 128 |
| gemma-ce128-s1-bf16 | gemma-ce128-s1-int8 | computed | -0.781 | [-2.344, +0.000] | 128 |
| gemma-ce128-s1-bf16 | gemma-ce128-s1-nf4 | computed | -4.688 | [-9.375, +0.000] | 128 |
| gemma-ce128-s2-bf16 | gemma-ce128-s2-int8 | computed | +0.000 | [-3.125, +3.125] | 128 |
| gemma-ce128-s2-bf16 | gemma-ce128-s2-nf4 | computed | -2.344 | [-7.031, +2.344] | 128 |
| gemma-base-bf16 | laya-general | computed | -4.688 | [-13.301, +3.926] | 128 |
| gemma-base-bf16 | decider-local | computed | +1.562 | [-6.250, +9.375] | 128 |
| gemma-base-bf16 | kev-local | computed | -14.844 | [-24.219, -6.250] | 128 |
| gemma-base-bf16 | agentjev-local | computed | -14.062 | [-23.438, -3.906] | 128 |
| gemma-base-bf16 | clef-local | computed | +3.125 | [-3.906, +10.156] | 128 |
| gemma-base-bf16 | jev-omni-local | computed | +0.781 | [-6.250, +7.812] | 128 |
| gemma-ce128-s0-nf4 | gemma-ce128-s0-nf4-recovered | computed | +2.344 | [-1.562, +6.250] | 128 |
| gemma-ce128-s1-nf4 | gemma-ce128-s1-nf4-recovered | computed | +3.906 | [-0.781, +8.594] | 128 |
| gemma-ce128-s2-nf4 | gemma-ce128-s2-nf4-recovered | computed | +3.125 | [-1.562, +7.812] | 128 |

Target checks apply to complete 128-case BoolQ observations and measured CUDA allocation. They do not certify the total capacity of a smaller GPU, reserve space for another process, or imply that load-latency SLOs passed. Tiny random-weight CUDA tests verify conversion/readout/adapter/media execution but do not establish full 12B accuracy or memory acceptance.

## Historical Modal A100 results (2026-10-01)

| Model | BoolQ accuracy [seed min, max] | Seeds | Native media accuracy | Weight size (estimate only) | Peak VRAM | BoolQ wall/decision ms | Hardware |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 12B base (BF16) | 86.72% (111/128; valid 128) | 1 | 65.38% (34/52; valid 52) | ≈24 GB nominal | not recorded | 87.23 | NVIDIA A100-SXM4-80GB |
| Gemma 12B base (parity control) (FP32) | 86.72% (111/128; valid 128) | 1 | 63.46% (33/52; valid 52) | ≈48 GB nominal | not recorded | not recorded | NVIDIA A100-SXM4-80GB |
| Laya general (recorded SDK runtime) | 82.03% (105/128; valid 128) | 1 | unsupported native media | unknown | not recorded | not recorded | NVIDIA A100-SXM4-80GB |
| prior, 8/class (CPU runtime) | 50.00% [50.00%, 50.00%] | 3 | not measured | unknown | not recorded | not recorded | CPU |
| prior, 32/class (CPU runtime) | 50.00% [50.00%, 50.00%] | 3 | not measured | unknown | not recorded | not recorded | CPU |
| prior, 128/class (CPU runtime) | 50.00% [50.00%, 50.00%] | 3 | not measured | unknown | not recorded | not recorded | CPU |
| setfit, 8/class (CPU runtime) | 51.30% [46.09%, 53.91%] | 3 | not measured | unknown | not recorded | not recorded | CPU |
| setfit, 32/class (CPU runtime) | 48.70% [46.09%, 52.34%] | 3 | not measured | unknown | not recorded | not recorded | CPU |
| setfit, 128/class (CPU runtime) | 49.48% [48.44%, 50.00%] | 3 | not measured | unknown | not recorded | not recorded | CPU |
| tfidf, 8/class (CPU runtime) | 49.22% [47.66%, 50.78%] | 3 | not measured | unknown | not recorded | not recorded | CPU |
| tfidf, 32/class (CPU runtime) | 47.14% [44.53%, 51.56%] | 3 | not measured | unknown | not recorded | not recorded | CPU |
| tfidf, 128/class (CPU runtime) | 52.34% [52.34%, 52.34%] | 3 | not measured | unknown | not recorded | not recorded | CPU |
| Gemma 12B + ce, 8/class (BF16) | 88.02% [82.03%, 92.19%] | 3 | 61.54% (32/52; valid 52); seed 0 only | ≈24 GB nominal | not recorded | 87.16–97.29 | NVIDIA A100-SXM4-80GB |
| Gemma 12B + ce, 32/class (BF16) | 88.80% [87.50%, 90.62%] | 3 | 61.54% (32/52; valid 52); seed 0 only | ≈24 GB nominal | not recorded | 95.52–112.70 | NVIDIA A100-SXM4-80GB |
| Gemma 12B + ce, 128/class (BF16) | 91.15% [90.62%, 91.41%] | 3 | 65.38% (34/52; valid 52); seed 0 only | ≈24 GB nominal | not recorded | 88.39–114.85 | NVIDIA A100-SXM4-80GB |
| Gemma 12B + ce_brier, 8/class (BF16) | 87.50% [82.03%, 92.19%] | 3 | 63.46% (33/52; valid 52); seed 0 only | ≈24 GB nominal | not recorded | 86.89–109.57 | NVIDIA A100-SXM4-80GB |
| Gemma 12B + ce_brier, 32/class (BF16) | 88.80% [87.50%, 90.62%] | 3 | 61.54% (32/52; valid 52); seed 0 only | ≈24 GB nominal | not recorded | 105.00–117.69 | NVIDIA A100-SXM4-80GB |
| Gemma 12B + ce_brier, 128/class (BF16) | 90.89% [90.62%, 91.41%] | 3 | 63.46% (33/52; valid 52); seed 0 only | ≈24 GB nominal | not recorded | 98.36–112.92 | NVIDIA A100-SXM4-80GB |

Historical CE latency is total held-out evaluation wall time divided by 128 decisions, shown per seed as a range. It is not a service p99 and must not be assigned to a quantized model. The 91.15% CE-128 result averages three seeds; seed 0 is 116/128 (90.625%). The media retention measurements for trained models use seed 0 only.

Historical base BF16 media accuracy is 28/32 (87.50%) on MNIST and 6/20 (30.00%) on FSDD. These are digit recognition subsets, not broad visual/audio capability tests. CE-128 seed 0 improves BoolQ by 3.906 percentage points versus base, with descriptive paired 95% source-group CI [-1.562, +9.375] pp; this does not establish a statistically accepted improvement. Laya's point difference versus base is -4.688 pp with CI [-13.301, +3.926] pp.

## Historical latency and load, by workload

| Stage | Workload | Successful / requests | Offered requests/s | p99 ms | 1,000 ms SLO misses |
| --- | --- | --- | --- | --- | --- |
| services | fixed-c1-n64 | 64 / 64 | not recorded | 104.193 | 0 |
| services | fixed-c4-n64 | 64 / 64 | not recorded | 111.980 | 0 |
| services | fixed-c16-n64 | 64 / 64 | not recorded | 101.036 | 0 |
| services | fixed-c64-n64 | 64 / 64 | not recorded | 116.311 | 0 |
| services | poisson-c1-n64 | 64 / 64 | not recorded | 463.222 | 0 |
| services | poisson-c4-n64 | 64 / 64 | not recorded | 555.386 | 0 |
| services | poisson-c16-n64 | 64 / 64 | not recorded | 331.405 | 0 |
| services | poisson-c64-n64 | 64 / 64 | not recorded | 627.252 | 0 |
| services | fixed-c16-n10000 | 10000 / 10000 | not recorded | 105.027 | 0 |
| stress | closed_loop-c1-n128 | 128 / 128 | closed loop | 97.757 | 0 |
| stress | closed_loop-c4-n128 | 128 / 128 | closed loop | 416.697 | 0 |
| stress | closed_loop-c16-n128 | 128 / 128 | closed loop | 1564.742 | 118 |
| stress | closed_loop-c64-n128 | 128 / 128 | closed loop | 6692.645 | 120 |
| stress | fixed-c1-n128 | 128 / 128 | 13.077 | 2400.033 | 82 |
| stress | fixed-c4-n128 | 128 / 128 | 13.077 | 2300.284 | 80 |
| stress | fixed-c16-n128 | 128 / 128 | 13.077 | 2368.989 | 79 |
| stress | fixed-c64-n128 | 128 / 128 | 13.077 | 2452.714 | 85 |
| stress | poisson-c1-n128 | 128 / 128 | 13.077 | 3185.145 | 87 |
| stress | poisson-c4-n128 | 128 / 128 | 13.077 | 2476.947 | 94 |
| stress | poisson-c16-n128 | 128 / 128 | 13.077 | 2446.876 | 99 |
| stress | poisson-c64-n128 | 128 / 128 | 13.077 | 3678.667 | 96 |

| Stage | Precision | Direct G4 workload | p99 ms |
| --- | --- | --- | --- |
| checkpoint | torch.bfloat16 | g0-g4-contract | 64.918 |
| checkpoint | torch.bfloat16 | native-batch-1 | 71.579 |
| checkpoint-fp32 | torch.float32 | g0-g4-contract | 339.548 |
| checkpoint-fp32 | torch.float32 | native-batch-1 | 504.424 |

The legacy 10,000-request service result is 105.027 ms p99 (block-bootstrap 95% CI 102.413–106.950 ms). Its offered rate and actual peak inflight were not recorded. The later stress harness measured queueing: closed-loop concurrency 16 reached 1,564.742 ms p99 and concurrency 64 reached 6,692.645 ms, with SLO misses. The legacy result therefore does not imply 105 ms p99 at genuine concurrency 16. All these service tests used loopback HTTP and do not include WAN latency.

BF16 native-batch and question-independence parity checks failed in the historical run; the FP32 control passed. Direct-call timing and accuracy depend on precision, batch shape, context, and number of candidates. All original findings remain in the historical evidence.

## Practical strengths and limits

- Clef records 89.84% BoolQ accuracy, 185.02 ms serial mean latency, and 55.66 (51.84) GB (GiB) peak CUDA allocation. Its image result is reported over the 32 MNIST cases; native audio is unsupported. The matched BoolQ comparison against Gemma base is shown separately from media coverage.
- Jev-Omni records 87.50% BoolQ accuracy, 81.29 ms serial mean latency, and 24.38 (22.71) GB (GiB) peak CUDA allocation. Native media records 57.69% (30/52) across images and audio. The report retains the paired BoolQ interval and runtime differences; the point estimate alone does not establish an accuracy ordering.
- Gemma CE-128 BF16 records 91.15% mean BoolQ accuracy across all three seeds, with worst measured CUDA allocation 24.50 (22.82) GB (GiB). It accepts raw images and audio, but media accuracy varies by task and high concurrency adds queueing.
- Decider records 88.28%, compared with Gemma base 86.72% and CE-128 BF16 mean 91.15%. Its peak CUDA allocation is 4.25 (3.95) GB (GiB); native image/audio inputs are unsupported. These are point estimates, not proof of a population-level accuracy ordering.
- Laya records 82.03%, 32.99 ms serial mean latency (the lowest measured mean in the selected rows), and 2.77 (2.58) GB (GiB) peak allocation. This is a small, fast text-only option with lower BoolQ accuracy than Gemma CE-128.
- INT8 records 91.41% mean accuracy [90.62%, 92.19%], with worst peak CUDA allocation 13.90 (12.94) GB (GiB). Serial mean latency spans 440.44–595.10 ms across the recorded runs. The current runtime exceeds the strict 12 GB allocation target; INT8 is not a measured latency improvement over BF16 here. The combined per-seed accuracy/memory acceptance target fails.
- NF4 records 87.24% mean accuracy [85.94%, 89.06%], with worst peak CUDA allocation 8.34 (7.77) GB (GiB). Serial mean latency spans 159.82–226.28 ms across the recorded runs. The original quantization-only method misses the requested >89–90% accuracy target. FSDD falls to 1/20 for every seed, so the media regression is substantial. The combined per-seed accuracy/memory acceptance target fails.
- Exploratory recovered NF4 records 90.36% mean accuracy [88.28%, 92.19%], with worst peak CUDA allocation 8.34 (7.77) GB (GiB). Serial mean latency spans 173.40–192.18 ms across the recorded runs. This separate method adds 100 fixed training updates per seed. Its mean crosses 90%, but seeds [0] remain at or below 89%, so it does not establish an all-seed quality pass. The continuation does not recover audio accuracy: FSDD remains 1/20 for every seed. The combined per-seed accuracy/memory acceptance target fails.
- With 128 BoolQ cases, one decision changes accuracy by 0.78125 percentage points. Passing a point threshold is small-sample screening, not a guarantee for other tasks, distributions, or production load. Peak allocation alone is not a minimum GPU capacity.

## Evidence and reproduction

Historical source: [live-summary.json](validation/2026-10-01/live-summary.json) (SHA-256 `14f355dc3ac6861bce76bf50a2d2467bb2456cee04aa105b16291770452d1571`). The [full historical report](validation/2026-10-01/live-validation.md) includes prediction hashes, model revisions, Modal jobs, calibration, learning curves, and provenance. The [September 30 smoke report](validation/2026-09-30/README.md) used A100 40GB and five fixture decisions; those smoke results are not an accuracy ranking.

Campaign source: [campaign receipt](validation/2026-10-07/comparison-summary.json) (SHA-256 `2d3c3374dd9dca5ae80a0bc03ef6ce5e4f78afcc355dbd0596371ac82508886d`).

Jev-Omni's complete receipt, BoolQ/media predictions and manifests, request records, and all six raw HTTP load cells are published with [verified source hashes](validation/2026-10-07/jev-omni-evidence.json). [Jev-Omni evidence and replay instructions](validation/2026-10-07/jev-omni.md) record the native-runtime policy and exact paired confidence interval. Earlier Clef evidence remains in the [October 2 archive](validation/2026-10-02/clef-evidence.json).

The combined receipt retains **31 attempts**, with **19 selected model rows**. Selection uses the latest completed attempt by recorded start time, or the latest compatible incomplete attempt when none completed; it never selects by test accuracy. Superseded failures and incompatible pre-execution attempts remain in its audit index.

| Model | Checkpoint | Pinned revision | Process peak RSS GB (GiB) | Context limit | Context policy |
| --- | --- | --- | --- | --- | --- |
| gemma-base-bf16 | google/gemma-4-12B-it | 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 | 29.45 (27.43) | 16384 | reject |
| gemma-ce128-s0-bf16 | google/gemma-4-12B-it | 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 | 29.59 (27.55) | 16384 | reject |
| gemma-ce128-s0-int8 | google/gemma-4-12B-it | 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 | 34.81 (32.42) | 16384 | reject |
| gemma-ce128-s0-nf4 | google/gemma-4-12B-it | 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 | 29.60 (27.57) | 16384 | reject |
| gemma-ce128-s1-bf16 | google/gemma-4-12B-it | 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 | 29.58 (27.55) | 16384 | reject |
| gemma-ce128-s1-int8 | google/gemma-4-12B-it | 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 | 34.81 (32.42) | 16384 | reject |
| gemma-ce128-s1-nf4 | google/gemma-4-12B-it | 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 | 29.60 (27.56) | 16384 | reject |
| gemma-ce128-s2-bf16 | google/gemma-4-12B-it | 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 | 29.59 (27.56) | 16384 | reject |
| gemma-ce128-s2-int8 | google/gemma-4-12B-it | 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 | 34.81 (32.42) | 16384 | reject |
| gemma-ce128-s2-nf4 | google/gemma-4-12B-it | 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 | 29.60 (27.57) | 16384 | reject |
| laya-general | convaiinnovations/laya | 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 | 7.78 (7.25) | 512 | truncate |
| decider-local | Mapika/decider-2b | 533964dae8be954c5b5e19fa4948e48408094c1e | 9.18 (8.55) | 32768 | reject |
| kev-local | jaredpalmer/kev-0.8b | 9a45d25eb2ab761841196625383fa1dff0e56c1e | 7.24 (6.75) | 8192 | reject |
| agentjev-local | aimeigaoshou/agent-jev | 0e2593e6e6c0eade0700712ac13c4389aa7654cc | 10.86 (10.11) | 2048 | reject |
| gemma-ce128-s0-nf4-recovered | google/gemma-4-12B-it | 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 | 30.54 (28.44) | 16384 | reject |
| gemma-ce128-s1-nf4-recovered | google/gemma-4-12B-it | 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 | 30.57 (28.47) | 16384 | reject |
| gemma-ce128-s2-nf4-recovered | google/gemma-4-12B-it | 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 | 30.57 (28.47) | 16384 | reject |
| clef-local | Cloudflare/clef | 2f3de3dd85f379784083b0814d997ab627200f0c | 60.78 (56.60) | 16384 | reject |
| jev-omni-local | akhilaaa3/Jev-Omni | 5addda86ddee081a68fb067477ea100c221b8917 | 29.84 (27.79) | 16384 | reject |

Follow the [original-data preparation and restoration steps](modal.md#reproduce-the-october-1-matched-comparison) before launching a fresh run. The restored native media is byte-identical to the archived dataset; its [recovery audit](validation/2026-10-01/media-rebuild.json) retains the reconstruction diagnostics without changing the benchmark input identity.

```bash
uv run --no-sync modal run apps/modal/compare.py --run your-unique-run
# Exploratory additional-training method; reports all three seeds separately.
uv run --no-sync modal run apps/modal/recover_nf4.py --run your-recovery-run
uv run --no-sync python scripts/collect_comparison.py \
  artifacts/comparison/your-unique-run/campaign.json \
  artifacts/comparison/your-recovery-run/campaign.json \
  --campaign-id your-comparison-id \
  --output docs/validation/2026-10-07/comparison-summary.json
uv run --no-sync python scripts/summarize_comparison.py \
  --historical docs/validation/2026-10-01/live-summary.json \
  --campaign docs/validation/2026-10-07/comparison-summary.json \
  --output docs/comparison.md
```

For split campaigns or retries, pass all campaign JSON paths to the collector; it retains their attempt history and never selects a model by test score. The renderer rejects changed BoolQ/media dataset hashes and inconsistent accuracy denominators. Missing measurements remain explicit. See the campaign runner's `--help` for live execution options; generating this document does not launch GPU jobs.
