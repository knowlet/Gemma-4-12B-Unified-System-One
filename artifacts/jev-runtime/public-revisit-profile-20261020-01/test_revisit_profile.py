"""Pure source materialization/verification; no browser or model dependency."""

import ast
import builtins
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
SOURCE = ROOT / "scripts/jev_observation_profile_revisit.py"
CANONICAL = (
    ROOT
    / "artifacts/jev-runtime/public-native-hover-settled-v8-20261019-01/profile-01/parent_v5/parent/canonical"
)
SPEC = importlib.util.spec_from_file_location("pure_revisit_profile_subject", SOURCE)
P = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = P
SPEC.loader.exec_module(P)


@pytest.fixture(autouse=True)
def forbidden_actors(monkeypatch):
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        if name.split(".", 1)[0] in {
            "torch",
            "transformers",
            "modal",
            "ctypes",
            "playwright",
            "selenium",
            "httpx",
        }:
            raise AssertionError("Forbidden runtime/actor import: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    return P.build_profile(P.read_canonical_sources(CANONICAL), expected_source_sha256=sha(SOURCE))


def prepare(tmp_path):
    output = tmp_path / "profile"
    result = P.materialize_profile(CANONICAL, output, expected_source_sha256=sha(SOURCE))
    return output, result


def verify(output, result, **changes):
    kwargs = {
        "expected_manifest_sha256": result["manifest_sha256"],
        "expected_source_sha256": sha(SOURCE),
    }
    kwargs.update(changes)
    return P.verify_materialized_profile(output, **kwargs)


def test_runtime_is_parent7_plus_core_with_only_snapshot_identity_change():
    bundle = build()
    base = P.V8.build_profile(P.read_canonical_sources(CANONICAL), expected_source_sha256=P.V8_SHA)
    assert len(bundle.runtime_files) == 8
    assert set(bundle.runtime_files) == set(base.runtime_files) | {
        "jev_ultrafast/observation_revisit.py"
    }
    for label in base.runtime_files:
        if label not in {P.SNAPSHOT, "jev_ultrafast/agent.py", "jev_ultrafast/model.py"}:
            assert bundle.runtime_files[label] == base.runtime_files[label]
    assert (
        bundle.runtime_files[P.SNAPSHOT].replace(
            ("id:'" + P.PROFILE + "'").encode(), ("id:'" + P.V8.PROFILE + "'").encode(), 1
        )
        == base.runtime_files[P.SNAPSHOT]
    )
    assert (
        sha(ROOT / P.DEPENDENCIES["observation_revisit.py"][1])
        == "80249cdd386619b9121160c94d3a071f586cfa40d75977a7e751138175a6b5d0"
    )
    manifest = bundle.materialized_manifest()
    assert (
        len(manifest["artifact_files"]) + 1
        == manifest["artifact_file_count_including_manifest"]
        == 59
    )
    assert manifest["old_V8_or_Q95_qualification_covers_new_prompt"] is False
    assert manifest["limits"] == {"files": 64, "total_bytes": 16 * 1024**2, "file_bytes": 1024**2}


def test_capture_adapter_preserves_original_traversal_body_bytes_and_AST():
    parent = Path(P.V8.__file__).read_text()
    adapter = Path(P.TREE.__file__).read_text()
    old = next(
        n
        for n in ast.parse(parent).body
        if isinstance(n, ast.FunctionDef) and n.name == "_capture_tree"
    )
    new = next(
        n
        for n in ast.parse(adapter).body
        if isinstance(n, ast.FunctionDef) and n.name == "capture_tree"
    )
    old_body = "\n".join(parent.splitlines()[old.body[1].lineno - 1 : old.end_lineno])
    new_body = "\n".join(adapter.splitlines()[new.body[1].lineno - 1 : new.end_lineno])
    assert old_body == new_body
    assert [ast.dump(n, include_attributes=False) for n in old.body[1:]] == [
        ast.dump(n, include_attributes=False) for n in new.body[1:]
    ]
    assert P.V8.MAX_FILES == P.MAX_FILES == 64 and P.V8.MAX_TOTAL_BYTES == P.MAX_TOTAL_BYTES


def test_materialize_verify_closes_exact_population_and_all_byte_pins(tmp_path):
    output, result = prepare(tmp_path)
    runtime = verify(output, result)
    manifest = runtime.manifest()
    assert type(runtime) is P.VerifiedMaterializedProfile
    assert result["artifact_files"] == 59 and result["browser_model_calls"] == 0
    assert runtime.runtime_root == output / "runtime"
    assert runtime.read_state == (runtime.runtime_root / P.SNAPSHOT).read_text()
    assert runtime.bundle.marker_expression == runtime.marker_expression
    assert P.PROFILE in runtime.read_state and runtime.manifest_sha256 == sha(
        output / "profile-manifest.json"
    )
    for label, pin in manifest["artifact_files"].items():
        path = output / label
        assert path.stat().st_size == pin["size_bytes"] and sha(path) == pin["sha256"]
    assert set(P._capture(output, read_contents=False)[1]) == P._labels()
    assert not (output / ".profile-manifest.pending").exists()


def test_copied_helper_can_verify_self_contained_profile_in_fresh_stdlib_process(tmp_path):
    output, result = prepare(tmp_path)
    script = """import importlib.util,sys,json
from pathlib import Path
p=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('copied_profile_verifier',p)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
r=m.verify_materialized_profile(Path(sys.argv[2]),expected_manifest_sha256=sys.argv[3],expected_source_sha256=sys.argv[4])
assert not any(n.split('.')[0] in {'torch','transformers','httpx','playwright','selenium','modal'} for n in sys.modules)
print(json.dumps({'profile':r.manifest()['profile'],'runtime_files':len(r.manifest()['runtime_source_sha256']),'actor_imports':False}))
"""
    r = subprocess.run(
        [
            sys.executable,
            "-B",
            "-s",
            "-c",
            script,
            str(output / "sources/jev_observation_profile_revisit.py"),
            str(output),
            result["manifest_sha256"],
            sha(SOURCE),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == {
        "profile": P.PROFILE,
        "runtime_files": 8,
        "actor_imports": False,
    }


@pytest.mark.parametrize(
    "change",
    [
        "runtime",
        "unknown-file",
        "unknown-directory",
        "symlink",
        "manifest",
        "helper-sha",
        "profile",
    ],
)
def test_tamper_unknown_population_and_wrong_external_pins_refuse(tmp_path, change):
    output, result = prepare(tmp_path)
    kwargs = {}
    if change == "runtime":
        with (output / "runtime/jev_ultrafast/model.py").open("ab") as stream:
            stream.write(b"\n# unreviewed change\n")
    elif change == "unknown-file":
        (output / "unknown.py").write_text("# unknown\n")
    elif change == "unknown-directory":
        (output / "unknown").mkdir()
    elif change == "symlink":
        path = output / "runtime/jev_ultrafast/model.py"
        path.unlink()
        path.symlink_to(output / "runtime/jev_ultrafast/agent.py")
    elif change == "manifest":
        kwargs["expected_manifest_sha256"] = "0" * 64
    elif change == "helper-sha":
        kwargs["expected_source_sha256"] = "0" * 64
    else:
        kwargs["profile"] = P.V8.PROFILE
    with pytest.raises(ValueError):
        verify(output, result, **kwargs)


def test_fresh_disjoint_output_is_required_without_overwrite(tmp_path):
    output, _result = prepare(tmp_path)
    before = (output / "profile-manifest.json").read_bytes()
    with pytest.raises(ValueError, match="Fresh disjoint"):
        P.materialize_profile(CANONICAL, output, expected_source_sha256=sha(SOURCE))
    assert (output / "profile-manifest.json").read_bytes() == before
    with pytest.raises(ValueError, match="Fresh disjoint"):
        P.materialize_profile(CANONICAL, CANONICAL / "child", expected_source_sha256=sha(SOURCE))


@pytest.mark.parametrize(
    "limit,value", [("MAX_FILES", 1), ("MAX_SOURCE_BYTES", 1), ("MAX_TOTAL_BYTES", 2)]
)
def test_explicit_capture_limits_are_enforced(limit, value, tmp_path):
    (tmp_path / "a").write_bytes(b"aa")
    (tmp_path / "b").write_bytes(b"bb")
    kwargs = {
        "labels": {"a", "b"},
        "read_contents": True,
        "helpers": P.V8,
        "MAX_SOURCE_BYTES": 100,
        "MAX_FILES": 64,
        "MAX_TOTAL_BYTES": 100,
    }
    kwargs[limit] = value
    with pytest.raises(ValueError, match="bound"):
        P.TREE.capture_tree(tmp_path, **kwargs)


def test_settling_sidecar_changes_identity_only_and_preserves_primary_failure():
    browser = SimpleNamespace(
        observation_settling_records=[{"terminal_unknown": False, "retained": [1, 2]}]
    )
    saved = []
    result = P.write_settling_sidecar(
        browser, 0, "fake", lambda _path, value, _secrets: saved.append(value), []
    )
    assert result == saved[0] and result["profile"] == P.PROFILE
    assert result["schema_id"] == "jev-revisit-native-observation-settling-sidecar-v1"
    assert result["records"] == browser.observation_settling_records
    primary = ValueError("original primary")

    def fail(*_args):
        raise RuntimeError("secondary writer")

    with pytest.raises(ValueError) as captured:
        P.write_settling_sidecar(browser, 0, "fake", fail, [], primary)
    assert captured.value is primary and any(
        "secondary writer" in note for note in primary.__notes__
    )
