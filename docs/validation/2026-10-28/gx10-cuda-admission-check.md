# GX10 CUDA availability check

On 2026-10-28 around 16:07 Asia/Taipei, a short synthetic BF16 matrix
multiplication completed on `cuda:0`, identified by PyTorch as NVIDIA GB10.
The interpreter reported PyTorch `2.13.0+cu130` and CUDA `13.0`; a 1024 by
1024 all-ones multiplication produced the expected finite value 1024. The
process synchronized CUDA before exiting. The ComfyUI queue was empty and
system available memory exceeded the 34,709,107,508-byte admission threshold
immediately before importing PyTorch.

This is a device and arithmetic check, not Decision3 model inference, a
benchmark, a measurement of sustained performance, or proof of GPU exclusivity.
GB10 uses shared system memory and `nvidia-smi` did not report total or free
device memory. ComfyUI and desktop processes retained GPU contexts. No other
work was stopped or unloaded.

The user directed subsequent inference on this host toward CUDA; MPS inference
is to be handled separately on a MacBook M1 Max. The previous CPU timing
experiment remains a separate measurement, not a CUDA comparison.
