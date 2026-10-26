This is a source-only downloader for the official public
[Qwen2.5-0.5B-Instruct checkpoint](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775).
The root fetched only 8,219 bytes of immutable publisher API metadata. No
checkpoint, tokenizer payload, framework, generator or browser was loaded here.
The Apache-2.0 license is one of the nine selected package files.

The selected revision is `7ae557604adf67be50417f59c2c2f167def9a775`.
Publisher metadata binds the 988,097,824-byte safetensors file to SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
Each other selected file is verified with its original Git blob hash, including
the blob header and exact size, while its actual SHA-256 is collected in the
same stream. These latter SHA-256 values are deliberately unknown until a real
successful download. Unlisted files, oversized/short/corrupt responses, HTTP206,
existing destinations and symlink paths cannot produce a returned success.
Failed partial files remain for diagnosis; there is no automatic retry or
resume. The public Q8 stream-hash-before-rename fetcher is the local reference;
the new implementation adds Git blob identities, bounded whole time and an
explicit owned-process check.

Actual acquisition remains pending independent review and a root command spec.
It must wait for the active Edge campaign's authenticated closure. Reuse the
existing **download owner02**: CPU0, one-core quota, 2 GiB memory, swap0 and the
already reviewed 32 GiB per-file limit. The CPU4 inference owners' 256 MiB
per-file limit cannot accommodate this checkpoint; no inference owner is
modified. Owner02 still requires fresh whole-host admission at reserve plus
its entire 2 GiB cap and the original read-only ComfyUI gate. This downloader
pins owner02's existing source and reuses its process identity, original
observer and host gate. Inside the unit it rechecks actual cgroup, affinity,
quota, memory/swap and inherited file limits. A 600-second whole timer, 30-second
maximum network waits, 1 MiB stream chunks, exact per-file byte limits and a
whole-package disk check bound the operation. The generic owner's independent
runtime/stop/closure checks remain necessary.

Do not infer success from a receipt filename alone. Qualification requires a
zero actor and owner exit, complete nine-file identity rows, unchanged sources,
the final host gate, no error/failure/late receipt, and authenticated external
closure. A late or failed success receipt is retained under a different name;
an evidence failure remains a failed command even if a data file was already
published. The SHA-256 values establish the downloaded streams, not permanent
file immutability or a complete Python/DSO closure. A future model loader must
authenticate the selected actual files itself inside its admitted runtime.

The 12 stdlib tests use independent tiny synthetic byte streams and mock
responses; they open no socket, tensor or tokenizer payload and do no model,
benchmark, ComfyUI or active-unit operation. They cover corrupt same-length
content, excess/short bodies, HTTP status/length, Git and LFS paths, deadline,
failed evidence, existing/symlink destinations and refusal outside the owner.
They establish no actual download, model fit, field generation or latency.
The separate helper implementation will use native model generation with the
unchanged field request and strict JSON/EOS/whole-20-second authorization. No
benchmark task, field answer, DOM, credential, model or runtime binary belongs
in a commit.

Status: SOURCE ONLY. No root runtime selection, actual acquisition, helper
qualification, browser score or goal completion.
