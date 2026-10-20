The next change connects the measured `shared-elements-v2` inference format to
native action supervision. At baseline `736fc7f`, `compile_actions` and
`compile_action_plan` still require literal raw JSON, so an existing raw-prompt
LoRA cannot establish that the new representation was trained. The 23 authored
form rows were admitted as data, with zero compiled rows and zero fit updates.
This work prepares training; model and browser quality remain unmeasured.

Research on October 21, 2026 used the local pinned sources and official upstream
documentation/source. No new dependency or external implementation is copied.

| Reference | Evidence read | Decision |
| --- | --- | --- |
| Existing native action fitter, repository MIT | `scripts/jev_browser_fit_actions.py:compile_actions`, `scripts/jev_cfg_action_loss.py:compile_action_plan`, `scripts/jev_owned_form_progress.py:compile_wire_projection` | Extend these paths. Preserve original teacher/head checks, full processor tensors, exact prefix boundary, native EOS, 32-token completion reserve and loss math. Default raw records remain identical. |
| Hugging Face TRL, Apache-2.0 | Official `trl/trainer/sft_trainer.py`, prompt-completion preparation, prefix-mismatch warning and completion masks; LICENSE read | Its separation of prompt and completion supports the existing approach. Do not adopt a new trainer or its truncation path; native exact-prefix rejection already provides the required contract. |
| Hugging Face Transformers, Apache-2.0 | Official chat-template documentation and LICENSE | Retain the actual inference generation prefix, append the audited action and native stop, and check every prefix tensor. A completed-assistant template is not assumed interchangeable with the inference template. |

Official sources were read at `main` on the research date; no immutable upstream
commit is claimed:

- https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py
- https://github.com/huggingface/trl/blob/main/LICENSE
- https://huggingface.co/docs/trl/sft_trainer
- https://huggingface.co/docs/transformers/chat_templating
- https://github.com/huggingface/transformers/blob/main/LICENSE

The compiler accepts an explicit `prompt_format`, defaulting to
`raw-json-v1`, and use production `format_action_request`. Explicit shared rows
record the requested format and exact representation metadata; effective raw
fallback remains distinguishable from a raw experiment. CFG validation must
reconstruct that same requested format, authenticate its metadata and original
wire, and retain the full original legal inventory. Historical raw records need
no additional fields. Unknown or inconsistent modes must reject.

Tests cover raw compatibility, selected V2 and raw fallback,
teacher-free prefixes, all original candidates, changed representation metadata,
wrong-format validation, tensor boundary/mask/EOS/context refusal, and unchanged
target/grammar inventories. The initial lightweight run had 42 passing tests and
62 skips. After correcting the invalid singleton fixture, the bounded GX10 ML
environment passed all 104 tests. Its CUDA enumeration warning is retained; CUDA
was not initialized. The earlier interrupted attempt remains separate evidence.
Actual CPU preparation should reuse all 23 authored
rows and the original public population under the proven 3 GiB resource profile.
Historical data evidence and current execution sources must be separately
authenticated; old source-handoff qualification cannot be silently transferred.
No model forward, optimizer update, native deployment or browser success follows
from processor/CFG preparation.
