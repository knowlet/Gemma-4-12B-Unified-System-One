# Native NVFP4 release

The [public Hugging Face package](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-NVFP4/tree/5c4f3c2cb5de2be66b9e51160613e868b33c311f)
was published and anonymously verified on October 7, 2026. All 48 files match
their preupload SHA-256 and size identities. See the
[publication receipt](validation/2026-10-07/nvfp4/publication.json) and
[complete measured results](validation/2026-10-07/nvfp4/README.md).
The [public Hub load test](validation/2026-10-07/nvfp4/hub-inference.json)
also passed native text/image/audio inference on B200 at that exact revision.

This release converts the **trained, merged BF16 System One checkpoint** into
packed NVFP4 language-model linear weights. The input embedding, candidate LM
head and native image/audio modules remain BF16. NVFP4 is distinct from the
existing bitsandbytes NF4 path.

The runtime uses the native Transformers 5.17 NVFP4 integration and the
[pinned Blackwell kernel](https://huggingface.co/kernels/kernels-community/nvfp4-gemm/tree/a66348abcb8cc22da69f2b77a3f5d4e01748f495).
Weights use E2M1 values, FP8 E4M3 scales per 16 values, and a float32 global
scale. The kernel uses dynamic NVFP4 activations for prefill; its one/two-row
path uses W4A16 GEMV. Both swizzled and row-major scales are retained in the
artifact, so total storage includes more than four bits per weight.

Transformers 5.17 supports conversion but does not support saving/reloading its
native NVFP4 modules. The S1 runtime therefore provides the explicit
`s1-transformers-nvfp4-v1` safetensors format. It preserves the exact packed
buffers and validates their dimensions, dtypes, checksums and tied-weight
aliases on load. Use the bundled S1 runtime; this is not a ModelOpt checkpoint
or a claim of compatibility with vLLM, TensorRT-LLM or stock `from_pretrained`.
See the [upstream implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/integrations/nvfp4.py)
and [quantizer limitations](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/quantizers/quantizer_nvfp4.py).

## Runtime

Use a Blackwell CUDA GPU. The measured environment uses B200, PyTorch 2.10.0,
Transformers 5.17.0 and kernels 0.16.0. A100/H100, CPU and MPS are rejected;
there is no automatic precision fallback or CPU offload. GB10 and other
Blackwell hardware require a compatible kernel build and separate validation.

Create a separate environment to preserve the project's historical PyTorch
2.8 benchmark environment:

```bash
uv venv --python 3.12 artifacts/nvfp4-env
uv pip install --python artifacts/nvfp4-env/bin/python -r requirements-nvfp4.txt -e .
```

Use `s1 decide`, `s1 benchmark` or `UnifiedDecisionModel` with
`quantization="nvfp4"`, CUDA and BF16 floating layers. Pass a merged BF16 source
to quantize while loading, or a packed S1 NVFP4 directory to restore the saved
buffers directly. Existing PEFT adapters must be merged in BF16 before
quantization.

```bash
artifacts/nvfp4-env/bin/s1 decide examples/request.json \
  --backend gemma --model knowlet/Gemma-4-12B-Unified-System-One-NVFP4 \
  --revision 5c4f3c2cb5de2be66b9e51160613e868b33c311f \
  --device cuda --precision bfloat16 --quantization nvfp4
```

## Reproduce conversion and validation

The source is
[`knowlet/Gemma-4-12B-Unified-System-One`](https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One/tree/a66f836b56605039fe040f330180e336d19b3362).
Every source checkpoint file must match the archived training release receipt.
The existing Modal workflow reads the source from
`/vol/releases/20261002-boolq-release-01/merged` on
`gemma-unified-system-one`. It does not train or modify that source.

```bash
mkdir -p artifacts/release/datasets
tar -xzf docs/validation/2026-10-02/release-datasets.tar.gz \
  -C artifacts/release/datasets
uv build --wheel --out-dir artifacts/nvfp4/runtime
uv run --no-sync modal run apps/modal/nvfp4_release.py --stage probe
uv run --no-sync modal run apps/modal/nvfp4_release.py \
  --stage release --run your-unique-run
uv run --no-sync modal run apps/modal/nvfp4_release.py \
  --stage verify --run your-unique-run
```

The bounded B200 release job checks the frozen dataset identities, evaluates
BF16, converts and saves packed NVFP4, then frees and reloads the saved weights.
Text/image/audio candidate logits must be exactly preserved across save/reload.
Only the 256 disjoint calibration cases fit the NVFP4 temperature. All 256
fresh BoolQ cases, 52 historical media cases and 128 historical BoolQ cases
must finish successfully. A separate process imports the exact bundled wheel
and replays all 436 evaluation cases; probabilities must agree within 1e-6
and every decision must agree before publication is allowed.

The historical 128-case population is separate from the fresh 256-case release
population. [Jev-Omni's matched comparison](validation/2026-10-07/jev-comparison.md)
uses the former. Its A100/PyTorch 2.10 latency must not be presented as a
same-hardware NVFP4 speed comparison.

`scripts/publish_nvfp4.py` accepts only an explicit file whitelist. It checks
the packed artifact, complete evaluation, calibration, and exact wheel replay
before uploading, then verifies every file at the immutable public Hub commit.
`--validate-only` runs the publication gates without uploading.

For direct publication from the existing Modal Volume, first run
`apps/modal/publish_nvfp4.py --stage prepare --run your-unique-run` and review
the returned model card and exact whitelist. Then use `--stage publish` with
the same run. The existing local Hub login is passed as a temporary Modal
secret; credential files are not included in the artifact or runtime image.
