"""Measured process and CUDA memory, with explicit peak-window semantics."""

from __future__ import annotations

import resource
import sys


def reset_memory_peak(device="cuda"):
    """Start a CUDA peak window; call after model load and any warmup forwards.

    Allocated peaks include resident weights. Reserved peaks can retain allocator
    blocks created during loading; no cache eviction changes the timed workload.
    Process peak RSS is lifetime-wide and cannot be reset by this function.
    """
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def memory_snapshot(module=None, device="cuda"):
    """Bytes observed now/within the last explicit CUDA peak window.

    The registered tensor footprint excludes activation/workspace allocations
    and may exclude auxiliary quantizer state; CUDA allocation is the deployment
    measurement. Device-wide usage also includes other processes and CUDA context.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(rss if sys.platform == "darwin" else rss * 1024)
    result = {
        "process_peak_rss_bytes": rss_bytes,
        "process_peak_rss_kib": rss_bytes / 1024,
        "process_peak_scope": "process_lifetime",
    }
    if module is not None:
        if hasattr(module, "get_memory_footprint"):
            footprint = module.get_memory_footprint()
        else:
            tensors = {id(tensor): tensor for tensor in (*module.parameters(), *module.buffers())}
            footprint = sum(
                tensor.nelement() * tensor.element_size() for tensor in tensors.values()
            )
        result.update(
            model_footprint_bytes=int(footprint),
            model_footprint_scope="registered_parameters_and_buffers",
        )
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        result.update(
            gpu_name=torch.cuda.get_device_name(device),
            gpu_total_bytes=total,
            device_used_bytes=total - free,
            allocated_bytes=torch.cuda.memory_allocated(device),
            reserved_bytes=torch.cuda.memory_reserved(device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            cuda_peak_scope="since_last_reset_or_process_start",
        )
    return result
