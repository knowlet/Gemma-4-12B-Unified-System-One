"""Isolated FIND_TEXT bundle composition with invented sources; no browser or ML."""

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTAL = "jev-find-text-experimental-v1"


def find_builder():
    path = ROOT / "scripts/jev_find_text_browser.py"
    spec = importlib.util.spec_from_file_location("find_text_browser_builder_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_v7_builder(builder):
    """An invented predecessor tests composition without reading any task set."""
    runner = (
        b"def run(repeats, max_steps, text_endpoint, text_model, text_helper):\n"
        + builder.GUARD
        + builder.CONFIGURATION
        + b"    }\n"
        + b"    if True:\n"
        + b"        if True:\n"
        + b"            if True:\n"
        + b"                if True:\n"
        + builder.TASK_SPECS
        + b"                    ]\n"
        + b"                    for item in [1]:\n"
        + b"                        try:\n"
        + builder.CONSTRUCTOR
        + b"                        finally:\n"
        + b"                            pass\n"
    )
    files = {
        "runner.py": runner,
        "runner-profile.patch": b"# Invented predecessor patch.\n",
        "launcher.py": b"# Invented predecessor launcher.\n",
        "helper.py": b"# Invented common helper, no tasks or judges.\n",
    }
    calls = []

    def verify(bundle, manifest):
        calls.append(("verify", Path(bundle)))
        assert {entry.name for entry in Path(bundle).iterdir()} == {
            *manifest["files"],
            "bundle-manifest.json",
        }
        for name, record in manifest["files"].items():
            assert builder._record((Path(bundle) / name).read_bytes()) == record
        assert json.loads((Path(bundle) / "bundle-manifest.json").read_bytes()) == manifest

    def prepare(bundle, *, case_module, source_dir):
        calls.append(("prepare", case_module, Path(source_dir)))
        bundle.mkdir()
        for name, blob in files.items():
            (bundle / name).write_bytes(blob)
        manifest = {
            "case_module": case_module,
            "profile_helper_sha256": "1" * 64,
            "files": {name: builder._record(blob) for name, blob in files.items()},
        }
        (bundle / "bundle-manifest.json").write_bytes(builder._encode(manifest))
        return bundle, manifest

    return SimpleNamespace(prepare_bundle=prepare, verify_bundle=verify), files, calls


def prepared_fake_bundle(monkeypatch, tmp_path, *, case_module="original"):
    builder = find_builder()
    predecessor, files, calls = fake_v7_builder(builder)
    source = tmp_path / "repo/scripts"
    source.mkdir(parents=True)
    references = {"fake-selector.py": {"sha256": "2" * 64, "size_bytes": 19}}
    monkeypatch.setattr(builder, "_base_builder", lambda root: predecessor)
    monkeypatch.setattr(builder, "_source_references", lambda root: copy.deepcopy(references))
    bundle, manifest = builder.prepare_bundle(
        tmp_path / "bundle", case_module=case_module, source_dir=source
    )
    return builder, bundle, manifest, files, calls


@pytest.mark.parametrize("case_module", ["original", "heldout"])
def test_new_bundle_preserves_predecessor_and_declares_only_four_runner_seams(
    monkeypatch, tmp_path, case_module
):
    builder, bundle, manifest, files, calls = prepared_fake_bundle(
        monkeypatch, tmp_path, case_module=case_module
    )
    assert calls[0] == ("prepare", case_module, tmp_path / "repo/scripts")
    assert manifest["action_profile"] == EXPERIMENTAL
    assert manifest["source_delta"]["additional_runner_seams"] == 4
    assert manifest["source_delta"]["inverse_bytes_equal"] is True
    assert manifest["source_delta"]["inverse_AST_equal"] is True
    assert manifest["model_executed"] is False
    assert manifest["runtime_qualification_attested"] is False
    for name, original in files.items():
        assert (bundle / builder.BASE_DIRECTORY / name).read_bytes() == original
        if name not in {"runner.py", "runner-profile.patch"}:
            assert (bundle / name).read_bytes() == original
    assert (bundle / "runner.py").read_bytes() == builder._derive(
        files["runner.py"], tmp_path / "repo"
    )
    builder.verify_bundle(bundle, manifest)


def test_new_bundle_refuses_unrelated_runner_edit_even_with_matching_manifest(
    monkeypatch, tmp_path
):
    builder, bundle, manifest, _, _ = prepared_fake_bundle(monkeypatch, tmp_path)
    runner = bundle / "runner.py"
    runner.write_bytes(runner.read_bytes() + b"\nUNDECLARED_POLICY_CHANGE = True\n")
    manifest["files"]["runner.py"] = builder._record(runner.read_bytes())
    (bundle / "bundle-manifest.json").write_bytes(builder._encode(manifest))
    with pytest.raises(ValueError, match="source/inverse proof"):
        builder.verify_bundle(bundle, manifest)


def test_new_bundle_refuses_extra_population(monkeypatch, tmp_path):
    builder, bundle, manifest, _, _ = prepared_fake_bundle(monkeypatch, tmp_path)
    (bundle / "unselected.py").write_text("# Unselected module.\n")
    with pytest.raises(ValueError, match="Unexpected or nonordinary"):
        builder.verify_bundle(bundle, manifest)


def test_new_bundle_refuses_changed_original_base(monkeypatch, tmp_path):
    builder, bundle, manifest, _, _ = prepared_fake_bundle(monkeypatch, tmp_path)
    (bundle / builder.BASE_DIRECTORY / "helper.py").write_text("# Changed original helper.\n")
    with pytest.raises(AssertionError):
        builder.verify_bundle(bundle, manifest)


def test_new_runner_seams_refuse_a_second_constructor():
    builder = find_builder()
    _, files, _ = fake_v7_builder(builder)
    with pytest.raises(ValueError, match="Unique FIND_TEXT"):
        builder._derive(files["runner.py"] + builder.CONSTRUCTOR, ROOT)


def derived_input_guard():
    builder = find_builder()
    _, files, _ = fake_v7_builder(builder)
    tree = ast.parse(builder._derive(files["runner.py"], ROOT))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    assert all(isinstance(statement, ast.If) for statement in function.body[:2])
    # Execute only the actual injected validation, without selecting an Agent,
    # importing the factory or reading any upstream/task/runtime input.
    function.body = function.body[:2] + [ast.Return(value=ast.Constant(value=True))]
    namespace = {}
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, "<find-text-derived-input-guard>", "exec"), namespace)
    return namespace["run"]


@pytest.mark.parametrize(
    "changes",
    [
        {"max_steps": 15},
        {"max_steps": 17},
        {"max_steps": 16.0},
        {"max_steps": True},
        {"text_endpoint": None},
        {"text_endpoint": " "},
        {"text_model": None},
        {"text_model": ""},
        {"text_helper": object()},
    ],
)
def test_actual_derived_guard_refuses_budget_or_callback_helper(changes):
    values = {
        "repeats": 1,
        "max_steps": 16,
        "text_endpoint": "http://127.0.0.1:1",
        "text_model": "fake-text-helper",
        "text_helper": None,
    }
    values.update(changes)
    with pytest.raises(ValueError, match="FIND_TEXT requires16"):
        derived_input_guard()(**values)


def test_actual_derived_guard_accepts_only_the_explicit_http_shape():
    assert derived_input_guard()(1, 16, "http://127.0.0.1:1", "fake-text-helper", None) is True


@pytest.mark.parametrize(
    "label,expected",
    [
        (
            "scripts/jev_browser_profile_v7_benchmark.py",
            "31450f9002ad216c32b77751997eb27748c491d7bc757c4410294d3704fdfb9e",
        ),
        (
            "scripts/jev_browser_benchmark.py",
            "fe1410675e4decb06a8491a5a83d2ec965629c99f8912b797446c4cb2c155f6c",
        ),
    ],
)
def test_default_v7_builder_and_canonical_entry_bytes_remain_unchanged(label, expected):
    assert hashlib.sha256((ROOT / label).read_bytes()).hexdigest() == expected


def test_new_builder_binds_selector_and_three_frozen_components_without_importing_them():
    builder = find_builder()
    refs = builder._source_references(ROOT)
    assert set(refs) == {builder.ACTION_SELECTOR, *builder.COMPONENTS}
    assert refs[builder.ACTION_SELECTOR]["sha256"] == builder.ACTION_SELECTOR_SHA256
    assert all(0 < ref["size_bytes"] <= builder.MAX_SOURCE_BYTES for ref in refs.values())
