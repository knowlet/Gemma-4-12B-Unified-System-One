"""Synthetic-only checks for the browser-independent runtime integrity helper."""

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jev_decision3_runtime_integrity as integrity
import pytest

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def test_extract_preserves_original_definitions_statically():
    old = ast.parse((ROOT / "scripts/jev_decision3_link_navigation_features.py").read_text())
    new = ast.parse((ROOT / "scripts/jev_decision3_runtime_integrity.py").read_text())
    names = {
        "_need",
        "_sha",
        "_file_sha",
        "_validate_manifest",
        "_tensor_sha",
        "_validate_provider_spec",
    }
    constants = {"MODEL_ID", "REVISION", "MANIFEST_SHA256", "MODEL_SHA256", "W0_SHA256"}

    def items(tree):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        } | {
            node.targets[0].id: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in constants
        }

    assert len(items(old)) == len(names) + len(constants)
    assert items(old) == items(new)


def test_imports_never_load_browser_or_cases():
    script = """\
import sys
sys.path.insert(0, 'scripts')
import jev_decision3_runtime_integrity
import jev_decision3_jevbench_heaven_owner
import jev_decision3_jevbench_coherence_owner
blocked = ('jev_browser_benchmark', 'jev_decision3_link_navigation_pipeline',
           'jev_decision3_link_navigation_scenarios', 'jev_public_web_cases',
           'jev_heldout_public_cases')
assert all(name not in sys.modules for name in blocked)
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-B", "-s", "-c", script], cwd=ROOT, check=True)


@pytest.fixture
def example(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    root = cache / "snapshot"
    root.mkdir(parents=True)
    (root / "model.bin").write_bytes(b"synthetic-model")
    manifest = {
        "schema": "d3-package-manifest/1",
        "identity": {"model_sha256": "synthetic-model-id"},
        "files_sha256": {
            "model.bin": digest(b"synthetic-model"),
            "NOTICE": digest(b"not read"),
        },
    }
    raw = json.dumps(manifest, sort_keys=True).encode()
    (root / "MODEL_MANIFEST.json").write_bytes(raw)
    monkeypatch.setattr(integrity, "MANIFEST_SHA256", digest(raw))
    monkeypatch.setattr(integrity, "MODEL_SHA256", "synthetic-model-id")
    return root, cache, SimpleNamespace(root=root, manifest=manifest)


def test_manifest_validates_runtime_and_rejects_changed_or_missing_file(example):
    root, cache, runtime = example
    assert integrity._validate_manifest(runtime, cache) == digest(
        (root / "MODEL_MANIFEST.json").read_bytes()
    )
    (root / "model.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="model file differs"):
        integrity._validate_manifest(runtime, cache)
    (root / "model.bin").unlink()
    with pytest.raises(ValueError, match="ordinary absolute pinned file"):
        integrity._validate_manifest(runtime, cache)


def test_manifest_rejects_runtime_mismatch_and_outside_cache(example, tmp_path):
    root, cache, runtime = example
    runtime.manifest = {}
    with pytest.raises(ValueError, match="loaded runtime manifest differs"):
        integrity._validate_manifest(runtime, cache)
    runtime.root = tmp_path
    with pytest.raises(ValueError, match="within selected cache"):
        integrity._validate_manifest(runtime, cache)
    runtime.root = root
    (root / "model.bin").unlink()
    target = tmp_path / "outside"
    target.write_bytes(b"synthetic-model")
    (root / "model.bin").symlink_to(target)
    with pytest.raises(ValueError, match="ordinary absolute pinned file"):
        integrity._validate_manifest(runtime, cache)


def test_manifest_rejects_traversal_and_incorrect_identity(example, monkeypatch):
    root, cache, runtime = example
    runtime.manifest["files_sha256"]["../outside"] = "0" * 64
    raw = json.dumps(runtime.manifest, sort_keys=True).encode()
    (root / "MODEL_MANIFEST.json").write_bytes(raw)
    monkeypatch.setattr(integrity, "MANIFEST_SHA256", digest(raw))
    with pytest.raises(ValueError, match="unsafe model file path"):
        integrity._validate_manifest(runtime, cache)
    runtime.manifest["identity"]["model_sha256"] = "other"
    raw = json.dumps(runtime.manifest, sort_keys=True).encode()
    (root / "MODEL_MANIFEST.json").write_bytes(raw)
    monkeypatch.setattr(integrity, "MANIFEST_SHA256", digest(raw))
    with pytest.raises(ValueError, match="checkpoint identity"):
        integrity._validate_manifest(runtime, cache)


def test_provider_and_tensor_checks_use_supplied_objects_only():
    spec = SimpleNamespace(
        adapter="decision3",
        model_id=integrity.MODEL_ID,
        revision=integrity.REVISION,
        device="cuda:0",
        precision="bfloat16",
        context_limit=16384,
    )
    integrity._validate_provider_spec(spec)
    spec.precision = "other"
    with pytest.raises(ValueError, match="pinned native provider"):
        integrity._validate_provider_spec(spec)

    class Tensor:
        def detach(self):
            return self

        def to(self, *, device):
            assert device == "cpu"
            return self

        def contiguous(self):
            return self

        def numpy(self):
            return self

        def tobytes(self):
            return b"synthetic-tensor"

    assert integrity._tensor_sha(Tensor()) == digest(b"synthetic-tensor")


def test_old_source_pins_refuse_missing_new_helper():
    script = """\
import inspect, pathlib, sys
sys.path[:0] = ['scripts', 'src']
import jev_decision3_candidate_browser_child as child
import jev_decision3_candidate_http_owner as common
import jev_decision3_runtime_integrity as helper
import jev_decision3_jevbench_heaven_owner as heaven
import jev_decision3_jevbench_coherence_owner as coherence
import run_jevbench_public
import jevbench_public_request_inventory
from s1.backends import normalize_response
from s1.contracts import DecisionRequest, validate_distribution
from s1.evaluation.decision3 import native_payload, normalize_native_response
class Dummy: pass
Dummy.__module__ = heaven.__name__
heaven.Dummy = Dummy
backend = Dummy()
backend.model = Dummy()
backend.runtime = Dummy()
common._verify_source_pins = lambda pins: None
for owner, source, config in (
    (heaven, run_jevbench_public, run_jevbench_public.CONFIG),
    (coherence, jevbench_public_request_inventory, jevbench_public_request_inventory.COHERENCE_CONFIG),
):
    paths = [owner.__file__, child.__file__, common.__file__, source.__file__, config,
             inspect.getfile(DecisionRequest), inspect.getfile(normalize_response),
             inspect.getfile(validate_distribution), inspect.getfile(native_payload),
             inspect.getfile(normalize_native_response), inspect.getfile(type(backend)),
             inspect.getfile(type(backend.model)), inspect.getfile(type(backend.runtime))]
    pins = {str(pathlib.Path(path).absolute()): '0' * 64 for path in paths}
    pins[str(pathlib.Path(config).absolute())] = owner.CONFIG_SHA256
    try:
        owner._actual_sources({'source_pins': pins}, backend)
    except ValueError as error:
        assert 'sources must be pinned' in str(error), error
    else:
        raise AssertionError('old source plan unexpectedly accepted')
    pins[str(pathlib.Path(helper.__file__).absolute())] = '0' * 64
    owner._actual_sources({'source_pins': pins}, backend)
assert 'jev_decision3_link_navigation_features' not in sys.modules
assert 'jev_decision3_link_navigation_pipeline' not in sys.modules
assert 'jev_browser_benchmark' not in sys.modules
"""
    subprocess.run([sys.executable, "-B", "-s", "-c", script], cwd=ROOT, check=True)


def test_changed_source_pin_is_refused(tmp_path):
    import jev_decision3_candidate_http_owner as common

    source = tmp_path / "synthetic-source.py"
    source.write_bytes(b"initial")
    pins = {str(source): digest(b"initial")}
    common._verify_source_pins(pins)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="selected source bytes changed"):
        common._verify_source_pins(pins)
