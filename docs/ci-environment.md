# CI environment and evidence boundaries

The contract and inference jobs check out the canonical
`browser-use/jev-ultrafast` source at
`1231850a0bf1a0c0341fe408ef1668dbbfdfac46` into
`artifacts/jev-runtime/upstream`. Native-agent/profile tests require these actual
sources; they are not replaced with mocks or a moving upstream branch. The
browser extra supplies Playwright and HTTP/2 dependencies. Live browser tests
retain their existing explicit opt-in; this setup does not enable them.

The inference job runs `scripts/prepare_ci_jev.py` before pytest. It retrieves only
four public tokenizer/template metadata files from
`knowlet/Gemma-4-12B-Unified-System-One` revision
`a66f836b56605039fe040f330180e336d19b3362`, verifies the existing `ASSET_SHA` pins,
and populates the collector's expected cache directory. It downloads no weights,
uses no authentication, and never regenerates acceptance evidence. Existing
cache entries are also checked and corrupt entries cause a hard failure.

## Outstanding captured regression input

`test_native_compiler_real_cached_Rust_full58_and_no_pruning` still requires the
original untracked capture:

`artifacts/jev-browser/20261015-public-native-training-start-readiness-01/postgres18-limit-offset/native-original-wire.json`

The tracked start-readiness report describes 58 offered PostgreSQL targets but
does not contain the original wire bytes or their individual digest. A fresh
public-page capture cannot prove byte identity with that historical input. The
original file and SHA-256 must be supplied and reviewed before this regression
can be made self-contained. The test and its assertions are retained unchanged;
CI must report its absence rather than fabricate a replacement or skip it.

## Formatting and provenance

`validation/2026-10-16/ci-format-source-identity.json` records the source digests
before/after Ruff formatting and AST-equivalence checks. Historical validation
receipts, processor closure hashes, and `public_action_env_probe.py`'s
`ARTIFACT_SHA` intentionally remain unchanged. That Modal probe is bound to the
pre-format artifact source, so running it against the newly formatted checkout
will correctly refuse the source mismatch. Use the historical pinned checkout
for historical evidence; obtaining a new source-bound qualification requires a
separate verified capture, not rewriting an old receipt. No Modal job or model
execution is part of this CI repair.

The synthetic qualification tests now isolate their temporary runtime module
origins with pytest monkeypatch restoration. Production source-origin validation
is unchanged and still rejects foreign, missing, or modified loaded modules.
