# Fresh Q95 helper binding

`live_helper.py` is a stdlib, parent-process callback. It starts no actor,
listener, service, unit, framework or model. The separate owner must first
launch a real Q95 service from a complete selected frozen source root, then
seal its actual process/listener identity. Historical source copies, a saved
qualification or these CPU fixtures do not establish a live model helper.

```python
checker = LiveHelperChecker(
    selection_blob=externally_selected_descriptor_bytes,
    selection_sha=externally_selected_sha256,
    receipt_directory=fresh_existing_private_directory,
)
evidence = checker(checker.binding, "before-cache-and-framework")
```

The callback returns exactly `binding`, `phase`, `helper_admitted`,
`co_resident_admitted` and `receipt_sha256`. It returns admission only after
persisting the sanitized receipt exclusively. Failure raises `BindingError`;
the callback cannot be authorized by a caller-supplied boolean. Each callback
has a three-second admission deadline; receipts are capped at 512. No later
callback admission is inferred from a previous successful receipt.

The descriptor has these exact top-level fields:

| Field | Authority |
| --- | --- |
| `schema_id` | `decision3-q95-live-helper-selection-v1` |
| `binding` | The bootstrap's exact public helper binding: profile, qualification SHA, source count, boot, PID/start/PGID/SID, endpoint, seconds, tokens, context. Q95 remains profile `field-syntax-preserving-base-helper-v1`, source95, qualification `d33bfcac49f90c47b626c3915baf3d5fcbfb2e32522bfaccbbfec59dce35d8db`, 20s/128/8192. |
| `worker` | Exactly `pid,start_ticks,pgid,sid,boot_id,exe,argv,cgroup`. Live UID, identity and command are reread around the check. PID=PGID=SID is required. |
| `listener` | Exactly `pid,address,port,inode`; IPv4 `127.0.0.1`, a unique listening inode actually held in that PID's fd table. |
| `service_directory`, `private_config` | Ordinary selected paths under the repository runtime tree; config is exactly `service_directory/controller-config.json`, own UID, mode0600, one hardlink. Stopping/closed markers refuse admission. |
| `interpreter`, `source_root` | The selected command is `[interpreter,source_root/scripts/jev_qualify_public_field_syntax_bf16.py,--internal-child,private_config]`; actual executable and argv must match. The source root contains all95 exact selected files. |
| `qualification`, `execution`, `runtime`, `registry` | Each is `{path,sha256,size_bytes}` with repository-relative ordinary path. Fixed SHAs are Q95 `d33bfc...`, execution `95f2eab0be9646da18694b18c872e044db09ba2fb1c09708077b01b24d58ae44`, runtime `702f7ff3cf194d1db56f86474e48ffc88cdfa8c9c6d3930467e3c12ea6492016`, and registry `aaa21ab6f2ecf5ae0e00694b43755541dc75fc0310ba10d945ade2a4511add52`. |
| `expected_memtotal` | Positive same-boot host total. |
| `co_resident` | Exactly `model,resident_file_bytes,cgroup`. All six models remain selected; resident bytes must equal the pinned registry AST. The callback's actual parent process must occupy this independently selected D3 unit cgroup. |

The config's qualification/public artifact/input/plan/execution/runtime identity,
paths, boot, MemTotal and output must agree with the selected metadata. The
private config is credential authority. Old Q93 policy/text env-key equality
does not apply: the D3 policy and Q95 text keys are different. Credentials never
enter public selections, argv, receipts or logs.

For the parent browser only, `checker.text_connection()` first performs a fresh
complete check and returns an **in-memory-only** mapping with `endpoint`,
`model="generated-actions-none"`, and `api_key`. Do not serialize or print that
mapping or the resulting browser environment. The model child receives no Q95
credential. Read-only properties `checker.binding` (a detached copy),
`checker.model`, `checker.resident_bytes` and `checker.selection_sha` let the
owner compare its independently selected arm and helper before construction.

Authentication uses the exact source-derived non-inference schema refusal. A
wrong credential must produce the default error envelope without allocating a
handler request ID or trace. A selected credential must produce `helper_schema`
422 with a fresh `X-Public-Request-ID`; the checker joins that ID to the actual
persisted Q95 raw record and requires no prefix, prepared prompt, API call,
generation or forward events, with restored guard and the empty-object payload
hash. An arbitrary 500, an arbitrary 422 or `/healthz` is insufficient. Real Q95
HTTP behavior remains unmeasured until the root owner starts that service.

Memory policy remains exact and separate. Q95 requires **host** available memory
at least `max(24GiB,ceil(MemTotal/5)) + 2*23919549408`, without resident/swap
credit. Its actual cgroup current/max/events are recorded; the host reserve is
not imposed inside a 40GiB helper cgroup. D3 retains CORE's
`min(host available, selected cgroup+ancestor headroom) >= reserve + 2D + 2GiB`.
It is therefore checked against actual selected unit limits, including the
separately approved 80GiB D3 profile. Existing native CUDA/workload checks and
outer unit/cohort ownership remain external requirements. This callback grants
neither model qualification nor whole-tree/no-escape/continuation authority.

`q95-source-runtime-inventory-01.json` verifies three exact95 source copies and
all348 installed and captured runtime files (7,050,328 bytes). Six current
project files differ from Q95. The source-only copies do not by themselves
contain the qualifier's required ROOT-relative artifacts/caches. A separate
frozen-root assembly is separately recorded under
`decision3-q95-frozen-root-20261022-01`. Its `serve-plan-02.json` and saved-byte
preflight now include the confined tensor link, created by the separate owner
only after the native training unit closed. A real service and its live
admission remain pending. No model payload was read or linked by this checker
task.

`auth-protocol-proof.json` records the source-derived status and handler join.
The default Starlette error handler is a separately pinned supporting runtime
file, outside historical runtime348; this extra ref must be included in the
real owner's selected source population. It does not expand or relabel the
historical runtime descriptor.

If Q95 assembly is deferred, a **separately declared** possible alternative is
`matched-current-nf4-field-helper-v1`: current NF4 base helper with the adapter
disabled and explicit 20s/128/8192 limits. It is not implemented or qualified
here, must not inherit Q95 identity, and would require new Gemma controls plus
all six Decision3 identities under that same helper condition. Historical Q95
scores cannot substitute for those matched controls.

Tests use synthetic metadata with deliberately overridden test-only pins and a
real owned, timed stdlib loopback fake. They exercise PID/session/listener,
private config and authentication joins, failures, memory floors, source drift,
receipt quota/persistence and split credentials. They prove the checker
contract, not actual Q95 execution. Every fake child is reaped in fixture
cleanup; it also has an independent 15-second alarm. No real model, GPU,
Comfy endpoint, systemd unit or cloud service is called by these tests.
