# Release evaluation

Metrics use the complete frozen populations. Each runtime fits temperature on the separate 256-case calibration split.

| Population | Backend | Cases | Accuracy | NLL | Brier | ECE-15 | p50 / p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| fresh_boolq | base_cuda | 256 | 80.08% | 0.4409 | 0.2874 | 0.0671 | 88.99 / 92.87 |
| fresh_boolq | trained_cuda | 256 | 89.84% | 0.2727 | 0.1580 | 0.0521 | 88.96 / 92.60 |
| fresh_boolq | trained_mlx | 256 | 89.84% | 0.2705 | 0.1564 | 0.0455 | 1506.87 / 2854.65 |
| old_boolq | base_cuda | 128 | 86.72% | 0.3316 | 0.1983 | 0.0799 | 87.96 / 91.72 |
| old_boolq | trained_cuda | 128 | 92.19% | 0.2498 | 0.1373 | 0.0586 | 90.25 / 93.35 |
| old_boolq | trained_mlx | 128 | not_run | — | — | — | — |
| media_all | base_cuda | 52 | 65.38% | 1.0500 | 0.4168 | 0.1574 | 112.26 / 113.98 |
| media_all | trained_cuda | 52 | 65.38% | 1.0726 | 0.4236 | 0.2184 | 111.59 / 113.65 |
| media_all | trained_mlx | 52 | 65.38% | 1.0820 | 0.4258 | 0.1310 | 3282.42 / 3471.44 |
| mnist | base_cuda | 32 | 87.50% | 0.4641 | 0.1877 | 0.1779 | 112.73 / 115.03 |
| mnist | trained_cuda | 32 | 87.50% | 0.5261 | 0.2073 | 0.2146 | 112.42 / 113.81 |
| mnist | trained_mlx | 32 | 87.50% | 0.5182 | 0.2059 | 0.1982 | 3323.09 / 3533.95 |
| fsdd | base_cuda | 20 | 30.00% | 1.9873 | 0.7835 | 0.1824 | 88.72 / 93.71 |
| fsdd | trained_cuda | 20 | 30.00% | 1.9471 | 0.7695 | 0.2806 | 91.04 / 93.28 |
| fsdd | trained_mlx | 20 | 30.00% | 1.9841 | 0.7776 | 0.1506 | 1528.55 / 1719.40 |

CUDA includes backend preprocessing and response construction; MLX uses prepared tensors. These timings do not establish a cross-runtime speedup.

Fresh BoolQ paired accuracy differences (target minus reference; 95% source-group bootstrap CI):

- base_to_trained_cuda: +9.77 percentage points [+5.86, +14.06], n=256; 10,000 resamples, seed 42.
- trained_cuda_to_mlx: +0.00 percentage points [+0.00, +0.00], n=256; 10,000 resamples, seed 42.

Calibration temperatures: base_cuda=2.518552; trained_cuda=2.783034; trained_mlx=2.714418.

descriptive fixed-population results; no test-selected acceptance gate.

Release identity: `2a708e5f4acfea4b688e88ba338aa62991608db875fef039ee6b4f33c3a74d73`
