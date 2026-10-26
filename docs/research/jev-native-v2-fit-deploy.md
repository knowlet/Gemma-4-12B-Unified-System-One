The next implementation connects all 179 historical native action rows to the
existing action-token trainer and an explicitly selected V2 deployment format.
The previous CPU preparation refused after processor loading and compiled zero
rows. Its evidence remains unchanged. This work cannot establish task success
without actual model fitting and matched browser trials.

Research on October 21, 2026 inspected the current repository and official
upstream source, with two search intents: adapter checkpoint metadata and
prompt/completion training contracts. No new dependency or upstream code is
copied. Upstream references were read at `main`; no immutable upstream SHA is
claimed.

| Reference | Source and license evidence | Decision |
| --- | --- | --- |
| Existing native action fitter and strict adapter verifier | `scripts/jev_browser_fit_actions.py:compile_actions/train_actions`, `src/s1/browser_action_adapter.py:_training_evidence/_verify_action_adapter`; repository MIT | Reuse exact prefix checks, completion-only full-vocabulary CE, FP32 q/v-only LoRA, frozen base/head, split/lineage checks and helper isolation. Keep legacy raw manifest/data validation intact. Add native-data orchestration and a separate artifact branch. |
| Existing immutable historical data reader | `artifacts/jev-runtime/v2-action-training-prepare-20261021-01/historical_data_projection.py`, SHA `abefbe19229b557c618dbf90f0722b7d71e92552b94c85a6c2289521968d05b5` | Reuse the separately authenticated 45 data files and all 179 rows. Do not construct an old admitted-collection DTO or waive old source-handoff guards. Freeze original evidence for independent replay. |
| Hugging Face PEFT | `src/peft/peft_model.py:save_pretrained/from_pretrained`, checkpoint documentation and Apache-2.0 LICENSE | Keep safetensors and the existing strict LoRA configuration. Add a repository manifest contract for prompt semantics; adapter weights/config alone do not record our complete input representation. |
| Hugging Face TRL | `trl/trainer/sft_trainer.py`, `sft_config.py:chat_template_path/completion_only_loss`, Apache-2.0 LICENSE | Reuse the stricter existing native exact-generation-prefix trainer. Do not add TRL, truncation or a different optimizer/loss objective during this format change. |

Official sources:

- https://github.com/huggingface/peft/blob/main/src/peft/peft_model.py
- https://huggingface.co/docs/peft/main/developer_guides/checkpoint
- https://github.com/huggingface/peft/blob/main/LICENSE
- https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py
- https://github.com/huggingface/trl/blob/main/trl/trainer/sft_config.py
- https://github.com/huggingface/trl/blob/main/LICENSE

The implementation adds a native historical-data bridge, a separate V2 artifact
kind, and explicit format propagation from the native browser CLI. The artifact
binds the requested V2 format, original/shared instruction hashes, production
formatter/codec identity and complete-user-text byte selection rule. Effective
raw fallback is valid within the requested V2 profile. Reject train/serve profile
mismatches before weights load and at the generated-action boundary.

All 142 train and 37 development rows remain in scope. The existing BF16 training
base and NF4 deployment recipe use the explicit `generated-actions-nf4` prompt
identity; these bytes are not evidence for a `generated-actions-none` experiment.
This change keeps the original full-vocabulary CE objective. CFG compilation
checks legality and inventories; it does not imply B/C CFG loss training.

Validation must cover raw compatibility, native inventories above 52 candidates,
shared and raw-fallback records, full split/data preservation, receipt/compiled
contract mutation, preweight and direct-API format refusal, helper isolation,
genuine tiny-CPU LoRA updates and development zero updates. The actual 12B fit,
resource qualification, NF4 deployment and full Jev/Decision3 comparison remain
required evidence; tiny tests or frozen data hashes cannot substitute for them.
