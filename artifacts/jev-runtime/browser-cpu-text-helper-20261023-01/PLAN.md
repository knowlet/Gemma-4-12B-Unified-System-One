Scope and implementation decision, 2026-10-23.

The existing Decision3 classifier cannot produce arbitrary field text. The existing Gemma syntax helper binds a 262144-token SentencePiece/ByteFallback decoder, Gemma EOS IDs and PEFT guard, and its original loading admission is unavailable. The completed source research at `../browser-cpu-text-helper-research-20261023-01/README.md` compared cached Qwen3-1.7B NF4, official Qwen2.5-0.5B-Instruct and official SmolLM2-135M-Instruct. Root selected the official 0.5B BF16 model for a distinct shared CPU helper profile.

| Source inspected | Role in this artifact |
| --- | --- |
| Browser Use Jev, upstream `1231850a0bf1a0c0341fe408ef1668dbbfdfac46`, `jev_ultrafast/model.py:field_text/field_context`, `questions.py:TEXT_VALUE` | Direct protocol compatibility and unchanged instruction literal. No browser/executor code copied or modified. |
| Existing `scripts/jev_field_syntax_helper.py:generate_field_text` and `jev_native_v2_browser.py` | Design reference for actual-EOS refusal, durable evidence, whole-request deadline and fail-closed publication. No inherited qualification and no Gemma grammar rebinding. |
| Selected local Transformers 5.17.0, `models/qwen2/modeling_qwen2.py`, `generation/utils.py`, `generation/stopping_criteria.py` | Direct dependency APIs for CPU SDPA, `generate`, `logits_to_keep`, forward hook, streamer and deadline criteria. Native API source hashes appear in `source-selection.json`. |
| [Official Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775), [official config](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/config.json), [Apache-2.0 license](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE) | Selected identity, architecture, license and native Transformers generation. Root-provided bounded publisher metadata supplies known weight SHA256 and other Git blob identities; no model/tokenizer payload was read. |

The implementation uses the existing Python/Transformers stack with a small stdlib ASGI boundary. It adds no serving, grammar, quantization, browser or model-selection dependency. Research is reused from before this implementation; this is reference-informed work, not a clean-room claim.

Completed source stage:

1. Author the strict protocol, lazy native runtime and durable ASGI boundary in this new directory.
2. Keep acquisition/runtime/owner unknowns explicit in a pending immutable selection.
3. Run stdlib fake boundaries and source checks; retain exact results and frozen hashes.

Pending root-owned dependencies:

1. Independently reviewed immutable acquisition/stream verification. Complete eight ordinary selected model files and actual SHA256 values; keep the 988-MB model out of Git.
2. Root-selected fresh combined cap and explicit whole-host admission, source/native-library selection, authenticated external owner verifier/ticket and real hard watchdog. Do not change existing owners.
3. Genuine independent field-generation qualification with raw tokens/EOS, CPU kernel/numerical checks, whole-20-second timing and peak combined memory/swap evidence.
4. Matched browser controls with the same helper and unchanged task/judge/options/16-step loop. Record all failures and unknown delivery honestly. Keep both JevBench populations separate.

No root-owned pending step is performed or authorized by this source artifact. No commit or push was made.
