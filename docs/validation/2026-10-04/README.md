# Model publication — October 4, 2026

The complete trained BF16 and multimodal MLX 8-bit packages are public on Hugging
Face. The [publication receipt](publication-status.json) records both as
`published_and_verified`. Training, calibration and evaluation remain the
[October 2/3 release measurements](../../release.md); publication adds file and
access verification, with no new inference results.

The separate [GGUF release evidence](gguf/README.md) adds October 4 Q8_0/F16
conversion, native S1 execution, independent calibration, complete evaluation
and [verified public publication](gguf/hub-gguf-publication.json). It preserves
the BF16/MLX receipts below unchanged and records its own new measurements.
The [combined publication status](gguf/all-publication-status.json) lists all
three verified public packages with their separate immutable revisions.

| Format | Immutable weights revision | Published package revision | Weight bytes |
| --- | --- | --- | ---: |
| BF16 | [`66626de5cbcf8c5fecb8e9b58710805a02a42f67`](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One/tree/66626de5cbcf8c5fecb8e9b58710805a02a42f67) | [`a66f836b56605039fe040f330180e336d19b3362`](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One/tree/a66f836b56605039fe040f330180e336d19b3362) | 23,919,549,408 |
| MLX 8-bit | [`e304ce0b87148933ce43d5ef9bed779f048accc8`](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-MLX-8bit/tree/e304ce0b87148933ce43d5ef9bed779f048accc8) | [`a5f89b400f7ef63e162f22866c206ed67cf8f282`](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-MLX-8bit/tree/a5f89b400f7ef63e162f22866c206ed67cf8f282) | 12,754,909,484 |
| GGUF Q8_0 + F16 projector | [`5940bdbe292b33ac86450093a528d39b13c3a8f4`](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-GGUF/tree/5940bdbe292b33ac86450093a528d39b13c3a8f4) | [`c418d37a17689fae554f7fc7d78db05ff7d52cfb`](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-GGUF/tree/c418d37a17689fae554f7fc7d78db05ff7d52cfb) | 12,791,679,008 |

Final cards and package manifests were committed after the core weights. The
[published BF16 card](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One/blob/a66f836b56605039fe040f330180e336d19b3362/README.md)
and [published MLX card](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-MLX-8bit/blob/a5f89b400f7ef63e162f22866c206ed67cf8f282/README.md)
use their package revisions above. The BF16/MLX inference/evaluation source remains
`d4ea1ef79af2e22f666374e529fa56c1de9d3f53`, a separate identity from either Hub
revision. The [reproduction guide](../../release-reproduce.md) downloads the
pinned BF16 weight revision while retaining that historical runtime.

## Verification evidence

- [Before-upload verification](preupload-verification.json) records local package
  files and SHA-256 hashes checked before publication.
- [BF16 weight receipt](hub-bf16-weights.json) and
  [MLX weight receipt](hub-mlx-8bit-weights.json) record the core upload revisions.
- [BF16 publication receipt](hub-bf16-publication.json) and
  [MLX publication receipt](hub-mlx-8bit-publication.json) bind the final package
  revisions to matching remote file sizes, LFS SHA-256 and Git blob identities.
  They also record anonymous metadata access for the BF16 weight file and all
  three MLX shards, plus downloaded public cards, manifests and calibration
  sidecars. Complete weight files were not redownloaded and no new remote
  inference was run.

Exact published card/manifest snapshots are retained here:

| Format | Card snapshot | Artifact manifest | Manifest SHA-256 |
| --- | --- | --- | --- |
| BF16 | [README](model-cards/bf16/README.md) | [Manifest](model-cards/bf16/artifact-manifest.json) | `f38d7f50060eabcbed6f73d16404ad42eb6ba1bddc79b68052afc13cab8598eb` |
| MLX 8-bit | [README](model-cards/mlx-8bit/README.md) | [Manifest](model-cards/mlx-8bit/artifact-manifest.json) | `9be087926c59c11671200d559c9bf6b8e8dbffebc491411d42229c15a4cce197` |

Each artifact manifest hashes its packaged files except itself. The original
[training artifact receipt](../2026-10-02/release/artifacts.json),
[MLX conversion receipt](../2026-10-02/release/mlx-conversion-manifest.json),
evaluation reports, and historical cards remain unchanged. The
[October 3 publication-status receipt](../2026-10-02/release/publication-status.json)
still records the credential limitation at that time; this directory records
the completed October 4 publication separately.
