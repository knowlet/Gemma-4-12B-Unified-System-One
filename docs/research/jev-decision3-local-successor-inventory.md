# Decision3 GX10 local successor inventory

2026-10-22 metadata-only investigation. All six pinned checkpoints are present locally, and all six use the architecture supported by the existing publisher-native backend. The smallest five are numerically eligible for fresh admission under the unchanged memory formulas. No model was loaded, no CUDA context was created, and no service or unit was started or changed during this investigation. These findings establish candidates and implementation work, not runtime qualification or scores.

## Exact local checkpoint selections

The registry is `src/s1/evaluation/decision3.py:CHECKPOINTS`. Each cache directory is exactly `artifacts/jev-local-hf/hub/models--vllm-sr--<name>/snapshots/<revision>`. The following manifest hashes were recomputed from the local small files and match the registry. File counts and aggregate sizes include `MODEL_MANIFEST.json` plus the CORE runtime selection, excluding documentation, licenses, requirements and assets.

| Model ID | Immutable revision | MODEL_MANIFEST.json SHA-256 | Selected files | Selected bytes |
| --- | --- | --- | ---: | ---: |
| vllm-sr/d3 | `225be8b1e2270becab69c5fe05b26f99cbbdd66b` | `4959df7ee964c2fad24a8e853df8e3cda9bae61636fa5650cc1a0fed78539677` | 31 | 52,199,243,753 |
| vllm-sr/d3-flash | `8b35c853b90e777595ec36bacae820fe08ea077a` | `abdafb9f4a3e39595a54b7ca977e6bd7a3279c57745231319e3a171a43976e65` | 24 | 16,812,829,752 |
| vllm-sr/d3-mini | `97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e` | `bc638e17982411f6ad0f524d33d0dc1f0db81ad7995c1bfdf93dba7674df5e48` | 22 | 9,104,396,884 |
| vllm-sr/d3-nano | `aeb70d4dd53ed25b855c3b0a66394b8fe265db7f` | `db2139c5b5b7060f8f95150ec70ca2fe7e69e6bf4249e5766df5388bf35977f2` | 20 | 4,451,751,168 |
| vllm-sr/d3-lite | `9c10703193afc7c7d28c88201c0364bfbbf58320` | `31f0ff94ff26619c0a063ccac803edcfe34c56c92e5c9c50b703b62893f125dd` | 21 | 1,730,220,475 |
| vllm-sr/d3-edge | `cab71e6ad3461b8086d51ddb572c656babd63b0b` | `3bae8989a6feb36a66bf4d8c44112dc63199d0dc69f564dc895fe368b1e08e59` | 21 | 1,212,854,825 |

The total is 139 files and 85,511,296,857 bytes. The metadata reads reproduce `artifacts/jev-runtime/decision3-GX10-cache-download-20261020-01/root-all6-post-d3-storage-inventory-01.json`. Previous closed transfer/storage proofs are in that directory's `root-{d3,flash,mini,nano}-storage-verification-01.json`; lite/edge staging evidence is under `artifacts/jev-runtime/decision3-GX10-cache-staging-20261020-01/`. Those historical proofs and this metadata inventory do not replace the initial and final full SHA/header authentication required by the actual worker.

## Architecture compatibility

Every selected `config.json` declares `model_type=qwen3_5`, `architectures=[Qwen3_5Model]` and `auto_map.AutoModel=modeling_d3.D3Model`. Every selected `decision_config.json` declares `d3-code-readout-v1`, 255 codes, FP32 readout, last pooling, noncausal full attention, temperature 1.0 and `max_length=null`.

| Model | Publisher base ID | Text width / layers | Pinned loaded parameters | Backbone + readout file bytes | Readout tensor shape |
| --- | --- | ---: | ---: | ---: | --- |
| d3 | Qwen/Qwen3.8-27B | 5,120 / 64 | 26,086,635,760 | 52,176,025,120 | `[255,5120]` FP32 |
| d3-flash | Qwen/Qwen3.5-9B | 4,096 / 32 | 8,393,739,504 | 16,789,656,976 | `[255,4096]` FP32 |
| d3-mini | Qwen/Qwen3.5-4B | 2,560 / 32 | 4,539,918,336 | 9,081,227,360 | `[255,2560]` FP32 |
| d3-nano | Qwen/Qwen3.5-2B | 2,048 / 24 | 2,213,763,904 | 4,428,644,200 | `[255,2048]` FP32 |
| d3-lite | Qwen/Qwen3.5-0.8B | 1,024 / 24 | 853,247,040 | 1,707,072,248 | `[255,1024]` FP32 |
| d3-edge | Qwen/Qwen3.5-0.8B | 1,024 / 12 | 594,582,720 | 1,189,722,528 | `[255,1024]` FP32 |

The publisher architecture dispatch is explicit in `modeling_d3.py:D3Model` and `d3_runtime.py:backbone_class`: Qwen3-VL is selected only for `qwen3_vl`; these six configurations select `Qwen3_5Model`. The full d3 base ID does not require a separate Qwen3.8 architecture class for these pinned serialized files. The local copies are also available under `artifacts/jev-d3-native/20261015-native-source-01/<model>/`.

The five small files read per snapshot included configuration and publisher entry/runtime sources. All six `modeling_d3.py` hashes are `f8f8db23ff58f1a990ffa24977d744a50e1b4446d02a64c22070967705b6dbef`, and all six `d3_fast.py` hashes are `68a1fb9a2c3bde9ac2763bf504decae592ca12722698fbde1b39a31262955535`. Full d3's `d3_runtime.py` is `2d90e93ac06b9ca4a84e0610ab9acfa2d6793ea4b40eec1b4a3da47577bdea37`; the other five use `adf7f7aa085674b5682e492d8b2e3cc4f11c67b0d153515c742e907e9b550917`. A successor must authenticate the selected model's own complete manifest, including these differences.

`Decision3Backend` already selects all six registry identities and calls publisher `AutoModel.from_pretrained(..., trust_remote_code=True, local_files_only=True, dtype="bfloat16", batch_size=8, verify="full")`. `src/s1/jev_decision3_native.py:JevDecision3NativeAdapter` accepts their unchanged complete 255-option native payloads. The bootstrap geometry checks cover exact loaded parameter count, BF16 CUDA backbone, FP32 readout, native batch 8, temperature 1.0, disabled fast/permutation modes and matched project context 8192. No model-specific adapter or tensor conversion is missing. Actual CUDA construction, selected kernels, text48 warmup and numerical behavior remain unmeasured.

The explicit local runtime selection is `artifacts/jev-runtime/decision3-GX10-local-runtime-20261020-01/runtime-selection-01.json`: aarch64, Python 3.13 at `/home/albert_hsieh/ComfyUI/.venv/bin/python`, torch `2.13.0+cu130`, torchvision `0.28.0+cu130`, transformers `5.17.0`, tokenizers `0.23.3`, safetensors `0.8.0`, accelerate `1.15.0`, huggingface-hub `1.33.0` and numpy `2.5.2`. The venv path is an interpreter selection; using it is not permission to control ComfyUI. The selection pins source/metadata, not every indirect DSO payload. The actual worker must revalidate selected origins and record `runtime_info()` and `kernels` after warmup.

## Unchanged memory policy on the current boot

The observed boot is `90eeafaa-7c2f-45f0-8b45-f74ef2ee1005`. Current `T=MemTotal=130,595,864,576` bytes. This differs from the historical note's previous boot by one 4 KiB page; the historical receipt remains unchanged.

For this boot, `R=max(24 GiB,ceil(T/5))=26,119,172,916` bytes and fixed BF16 Gemma helper bytes `B=23,919,549,408`. Existing formulas are:

- CORE at every gate: `R + 2D + 2 GiB`, where `D` is the registry's resident file bytes. No resident or swap credit.
- Native admission before a declared helper loads: `R + D + max(D,B) + B`.
- Native admission after that helper is actually resident: `R + D + max(D,B)`. The current observation must already include helper residency; this removes only the additional helper allocation, retaining the complete declared transient.
- Q95 helper load and each forward: `R + 2B = 73,958,271,732` bytes.

| Model | CORE available floor | Native before helper | Native with helper resident | Capacity status under these formulas |
| --- | ---: | ---: | ---: | --- |
| d3 | 132,618,706,804 | 154,390,772,564 | 130,471,223,156 | CORE exceeds physical RAM; refuse |
| d3-flash | 61,845,970,516 | 90,747,928,708 | 66,828,379,300 | Fresh admission candidate |
| d3-mini | 46,429,111,284 | 83,039,499,092 | 59,119,949,684 | Fresh admission candidate |
| d3-nano | 37,123,944,964 | 78,386,915,932 | 54,467,366,524 | Fresh admission candidate |
| d3-lite | 31,680,801,060 | 75,665,343,980 | 51,745,794,572 | Fresh admission candidate |
| d3-edge | 30,646,101,620 | 75,147,994,260 | 51,228,444,852 | Fresh admission candidate |

Thus flash, mini, nano, lite and edge have no unconditional physical-RAM contradiction. With the helper resident, their independent host thresholds are all below the Q95 forward threshold of 73,958,271,732 bytes. A real co-resident run still needs fresh phase observations satisfying every applicable gate, CUDA free-memory checks and complete cgroup observations. A pre-load free-memory sample does not prove that the post-load helper forward can admit. These independent formulas are necessary gates, not an existing measured simultaneous peak bound.

The full d3 CORE floor exceeds this boot's entire physical RAM by **2,022,842,228 bytes**, before counting any unavailable RAM. This is a policy capacity refusal, not a measured failed inference or a model-quality score. Its unchanged-policy baseline requires a larger-memory host. Any future different capacity profile would need its own measured and enforced bounds, separately reviewed evidence and explicit identity; this proposal does not reduce a floor or subtract resident D3 bytes.

## Current-boot identity boundary

H15's historical `artifacts/jev-runtime/public-native-fit-memory-uma-probe-20261020-01/native_uma_peak_probe_01.py` remains byte-pinned at SHA `2ed7cda3054da62e7440bf57e4bfae15e26ee6d22de8b80aede6c8f53973e3cb`, 118,654 bytes. Its `UMA_TOTAL=130595860480`, `uma_host_identity` and historical CUDA-total checks are tied to the previous boot's total and must not be altered or presented as new-boot qualification. Source17 is a distinct frozen Decision3 source projection: its listed 17 files do not contain H15; its CORE gates already take an explicit expected boot and MemTotal. Keep both historical boundaries intact.

The new local owner must additionally bind each raw host, Driver and Torch observation to the same freshly selected boot, positive exact MemTotal and GPU UUID. For the observed boot, read-only metadata evidence is `artifacts/jev-runtime/gx10-boot-identity-20261022-01/metadata-01.json`: raw Driver total equals host total 130,595,864,576, UUID `e3429caa-b057-cbe0-917d-d3b85b313be7`, name NVIDIA GB10, integrated=1, device count=1 and no current CUDA context before or after the query. NVML memory queries returned unsupported. These are identity observations, not allocator admission, warmup or memory-usage measurements.

Current `src/s1/browser_native_resources.py:host_identity` and `uma_identity` supply reusable typed identity checks for complete host and Driver/Torch metadata. Reuse requires its actual new source pin and complete expected input fields, preserving raw totals and exact equality on every phase; it does not confer H15 qualification. Its `Runtime` and native training `Gate` also own a separate 40 GiB allocator/memcg profile and exact training-unit contract. They must not be imported as the Decision3 resource policy or silently applied to the Q95 service. A small separately inventoried identity-only collector/checker is the appropriate seam; retain unchanged Decision3, Q95, CUDA-free and cgroup floors alongside it.

## Smallest executable successor

The existing component is `artifacts/jev-runtime/decision3-GX10-warm-successor-20261020-01/gx10_warm_bootstrap.py`. Its `CACHE_CANDIDATES=("d3-lite","d3-edge")` remains an explicit validation restriction even though all six downloads are now present. Its CLI deliberately refuses `--run`, and `Components` owns no listener, scheduler, response delivery or process supervisor.

A first actual baseline can select **d3-edge**, which the unchanged component already accepts, and run the two original browser groups serially. It does not require editing that component to widen its allowlist. The following are the missing executable functions/contracts; names in the first column are proposed successor APIs, not claims that those functions already exist.

| Successor responsibility | Reusable implementation | Required concrete behavior |
| --- | --- | --- |
| `seal_local_plan` | Bootstrap `source_inventory`, `RuntimeSelection`, frozen source17 | Pin a new actual source union containing controller, worker transport, helper checker and reused gateway code. Preserve source17 and the component's old projection separately; keep their existing SHA checks. Select a fresh boot/MemTotal/UUID, nonce, runtime selection and empty owned output directory. |
| `start_q95_helper` / `check_live_q95` | `scripts/jev_qualify_public_field_syntax_bf16.py` qualification/serve checks; older field-JSON observer's process/listener/private-credential mechanisms | Launch an owned qualified Q95 helper, bind exact PID/start/PGID/SID/boot, loopback listener ownership and actual authenticated request access. Recheck full independent helper and co-resident admission from fresh observations at each required phase. Return the bootstrap's exact evidence shape with a persisted receipt SHA. `/healthz` alone bypasses bearer authentication and cannot prove credential binding. |
| `owned_worker_main` | Bootstrap `bootstrap`, frozen worker `process_request` | Create the offline environment and fresh publisher module cache before dynamic imports. Authenticate full selected cache, runtime and original sources; construct the real native backend and complete publisher text48 warmup. Emit nonce/model/PID/start/UUID-bound READY only after all checks. Bind the existing helper-resident memory policy explicitly in the successor's live checks; the bootstrap backend call itself currently leaves `helper_already_loaded` at its default. |
| `local_gateway` / `run_browser_group` | `apps/modal/decision3_browser_service.py:Gateway` and `Session`; frozen worker `process_request` | Implement local loopback transport and owned-child framing, queue/deadline/publication/delivery handling. Preserve exact `request_id,payload_json,http_sha256,remaining_seconds` worker envelope and original HTTP body. Modal mounts, H200 initialization and cloud RPC require local replacements. Run known-four and public-six with their original 64/96 envelope limits, 16 steps and judges. |
| `close_arm` / `publish_result` | Existing worker close/final-evidence mechanisms and owned-session closure semantics | Authenticate final sources/cache/runtime/UUID, fence CUDA, preserve raw and late outputs, close the backend, and prove owned worker and descendants absent before another arm. Handle partial constructor failure at the whole-process owner. Report completed browser results, admission refusal, timeout and delivery failure distinctly. |

Keep Q95's exact profile `field-syntax-preserving-base-helper-v1`, source count 95, qualification SHA `d33bfcac49f90c47b626c3915baf3d5fcbfb2e32522bfaccbbfec59dce35d8db`, 20 seconds, 128 tokens and context 8192. `live_helper_check` currently validates a callback result's fields and booleans; it does not supply the fresh process/listener/credential/admission checker above. The older observer/controller sources are under `artifacts/jev-runtime/field-json-qualified-browser-retest-20261018-01/` and must be adapted with a new Q95 identity rather than relabelled as already matched.

The preserved original manifest is `artifacts/jev-runtime/decision3-matched-v7-warm-readiness-20261018-01/source-inventory17.json`, SHA `d8c886416bfc54192a3f861a8cb386e25754606a42410317056d2eef2f638985`; its frozen `source-candidate-01/jev_decision3_browser_worker.py` SHA is `f46fae94b759e373fce18d8e75abcac261b4e1f7c79bead0614cbed0e3d4f7d7`. Do not monkeypatch H15, the old source17/18 inventories or `CACHE_CANDIDATES`. Widening to flash/mini/nano belongs to an explicit separately inventoried successor configuration/bootstrap using their exact selections above. All six models remain the required comparison population, including the largest model's capacity refusal until an admissible host is available.

Before the first real load, narrow CPU integration verification should cover actual loopback HTTP and owned fake-child READY/request/CLOSED framing, stale/wrong nonce/model/PID/UUID/helper credentials, partial startup, deadline crossings and incomplete cleanup. After that, the first evidence target is an actual edge × known-four/public-six run with Q95 resident, full cache/runtime authentication, completed text48 warmup, raw request evidence and confirmed closure. New Gemma comparison arms must run under that same selected Q95 regime. Historical Q95 results and the NF4 in-process helper cannot substitute for matched new controls.

## Current source17 integration check

The final transport handoff checked all17 current source17 paths:16 still match; `src/s1/jev_generated.py` is now SHA `c356e0eb3c046cb0787d7be0c5960803006231e8bdb414de644ae2167e72d849` /23,569 bytes, compared with historical `582b520615cf123367436622f4e5ca6f5d94abb227449d866ec64cbb8adc9b3c` /18,825 bytes. The old bootstrap therefore refuses the current tree before construction. Preserve the historical manifest and projection; actual execution needs an explicit current source union or separately authenticated frozen runtime. The new stdlib transport does not import project modules and grants no whole-tree continuation without external kernel ownership proof. Exact check: `artifacts/jev-runtime/decision3-local-transport-20261022-01/integration-gap-01.json`.
