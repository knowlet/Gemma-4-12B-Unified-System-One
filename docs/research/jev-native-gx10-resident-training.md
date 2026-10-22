The October 22, 2026 GX10 observation has118,257,770,496 available bytes. The
native fitter still repeats its106,438,961,088-byte prospective startup floor
after loading the model. A closed prior actual07 attempt had119,927,267,328
before loading and94,667,264,000 before parent injection, then refused with zero
updates. Repeating that launch would not establish full-run capacity.

Research precedes implementation. This is a separate explicitly selected
resource profile, preserving the original default and all historical refusals.
No model quality or qualification follows from source changes or fake tests.

| Reference | Inspected mechanism and license | Decision |
| --- | --- | --- |
| Current native/public fitter | `jev_browser_fit_native_actions.py:fit_native_actions`, `jev_browser_fit_public_actions.py:ResourceGate`, repository MIT | Keep original prospective profile unchanged; bind a separate profile to the same actual12B checks and unchanged142train/37development full-vocabulary CE. |
| Existing GX10 native memory research | Frozen `public-native-fit-memory-uma-probe-20261020-01/native_uma_peak_probe_01.py`, SHA `2ed7cda3054da62e7440bf57e4bfae15e26ee6d22de8b80aede6c8f53973e3cb`; `research_floor`, `uma_validate_identity`, `uma_memory_decision`, `Research`; repository MIT | Reuse startup max(original106.439GB, R+2*40GiB+4GiB)=116.313GB and loaded R+40GiB+8GiB=77.659GB. Retain40GiB per-process native allocator and own-unit limits,0swap, soft allocator stop39.5GiB and live host reserve. These remain separate limits, not a combined CUDA/memcg hard-bound claim. A previous A arm completed two disposable native updates; full142-step capacity is unproven. |
| PyTorch official CUDA documentation/source | `torch/cuda/memory.py:memory_allocated/memory_reserved/set_per_process_memory_fraction`, BSD-3-Clause; main observed October22 | Read actual process allocator counters/backend/fraction; distinguish tensor allocation, cached reservation and global device reporting. Cap only this process, release only its unused cache at controlled stage boundaries. No inference that allocated bytes equal all physical device consumption. |
| NVIDIA DGX Spark documentation | Known Issues, UMA reporting and iGPU framebuffer guidance | Require actual GB10/integrated device/UUID/total identity. Keep raw CUDA counters, but do not substitute them for Linux host availability or claim a discrete VRAM pool. No third-party implementation copied. |

Primary sources inspected:

- https://docs.pytorch.org/docs/main/notes/cuda.html
- https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.memory_allocated.html
- https://github.com/pytorch/pytorch/blob/main/torch/cuda/memory.py
- https://github.com/pytorch/pytorch/blob/main/LICENSE
- https://docs.nvidia.com/dgx/dgx-spark/known-issues.html

The new profile must authenticate the same boot/process/unit, actual native
allocator readback and GB10 identity. Transition to loaded thresholds only after
the existing full12B/BF16/fullhead/176qv-LoRA checks pass. Record original
prospective would-refuse separately. Never rewrite an old gate outcome or
borrow old95-source admission. Preserve all179 inputs, exact targets, optimizer,
development zero-updates, primary failures, partial weights and whole-unit
closure evidence. No Comfy interrupt/unload/free or global cache changes.

Validation: fake boundary, wrong-device/unit/boot/backend/fraction, forged
loaded transition and artifact mutation refusals; actual full12B run under
fresh conservative startup admission and external owned-unit guardian. Then
strict exported-artifact verification and matched real Jev browser trials.
Source/fake success cannot substitute for real model, update or browser evidence.
