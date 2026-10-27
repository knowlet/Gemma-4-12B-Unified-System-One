"""The isolated launcher changes one import, never serving/training or policy."""

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import jev_browser_heldout_benchmark as LAUNCHER
import pytest

ROOT = Path(__file__).resolve().parents[1]


def cli(output):
    return [
        "--upstream-path",
        "/upstream",
        "--endpoint",
        "http://127.0.0.1:8123/v1/systemone",
        "--model",
        "explicit-policy-model",
        "--output-dir",
        str(output),
    ]


def test_prepare_preserves_original_and_only_changes_one_import_ast(tmp_path):
    original_path = ROOT / "scripts/jev_browser_benchmark.py"
    original = original_path.read_bytes()
    bundle, manifest = LAUNCHER.prepare_bundle(tmp_path / "bundle")
    assert original_path.read_bytes() == original
    assert (bundle / "original_jev_browser_benchmark.py").read_bytes() == original
    derived = (bundle / "runner.py").read_bytes()
    assert derived == original.replace(LAUNCHER.ORIGINAL_IMPORT, LAUNCHER.ISOLATED_IMPORT, 1)
    tree = ast.parse(derived)
    substitutions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "jev_heldout_public_cases"
    ]
    assert len(substitutions) == 1
    substitutions[0].module = "jev_public_web_cases"
    assert ast.dump(tree, include_attributes=False) == ast.dump(
        ast.parse(original), include_attributes=False
    )
    diff = (bundle / "runner-import.patch").read_text()
    assert [
        line for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")
    ] == ["-        from jev_public_web_cases import ("]
    assert [
        line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")
    ] == ["+        from jev_heldout_public_cases import ("]
    assert manifest["status"] == "source_prepared_not_site_qualified_or_evaluated"
    assert manifest["readiness"]["native_site_preflight_verified"] is False
    assert len(manifest["cases"]) == 6
    LAUNCHER.verify_bundle(bundle, manifest)
    with pytest.raises(FileExistsError):
        LAUNCHER.prepare_bundle(bundle)


@pytest.mark.parametrize("name", ["jev_browser_benchmark.py", "jev_public_web_cases.py"])
def test_original_source_pin_failure_precedes_any_bundle_creation(tmp_path, name):
    source = tmp_path / "source"
    source.mkdir()
    for file in (
        "jev_browser_benchmark.py",
        "jev_public_web_cases.py",
        "jev_heldout_public_cases.py",
    ):
        (source / file).write_bytes((ROOT / "scripts" / file).read_bytes())
    with (source / name).open("ab") as handle:
        handle.write(b"\n# changed\n")
    with pytest.raises(ValueError, match="source pins"):
        LAUNCHER.prepare_bundle(tmp_path / "bundle", source_dir=source)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    "name",
    ["runner.py", "jev_heldout_public_cases.py", "jev_public_web_cases.py", "bundle-manifest.json"],
)
def test_bundle_changes_are_detected(tmp_path, name):
    bundle, manifest = LAUNCHER.prepare_bundle(tmp_path / "bundle")
    with (bundle / name).open("ab") as handle:
        handle.write(b"\n ")
    if name == "bundle-manifest.json":
        value = json.loads((bundle / name).read_text())
        value["selection_sha256"] = "0" * 64
        (bundle / name).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="changed"):
        LAUNCHER.verify_bundle(bundle, manifest)


def test_prepare_only_never_executes_a_child_or_imports_an_ml_runtime(tmp_path):
    # A separate plain Python interpreter also detects accidental ML/browser
    # imports, independently of the parent's pytest module population.
    script = """
import runpy, sys
sys.path.insert(0, sys.argv.pop(1))
source = sys.argv.pop(1)
try:
    runpy.run_path(source, run_name='__main__')
except SystemExit as exc:
    if exc.code:
        raise
assert not any(name.split('.')[0] in {'torch','transformers','playwright','peft'} for name in sys.modules)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-s",
            "-c",
            script,
            str(ROOT / "scripts"),
            str(ROOT / "scripts/jev_browser_heldout_benchmark.py"),
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--prepare-only",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["model_executed"] is False
    assert not (tmp_path / "bundle/launch-receipt.json").exists()


def test_defaults_select_only_fixed_cases_and_original_declared_observations(tmp_path):
    forwarded, values = LAUNCHER._arguments(cli(tmp_path / "run"))
    assert values.max_steps == 16 and values.repeats == 1 and values.seed == 20261014
    assert forwarded[: len(cli(tmp_path / "run"))] == cli(tmp_path / "run")
    assert forwarded[
        forwarded.index("--public-cases") + 1 : forwarded.index("--observation-profile")
    ] == list(LAUNCHER.PUBLIC_WEB_CASES)
    assert forwarded[-2:] == ["--observation-profile-helper-sha256", LAUNCHER.PROFILE_SHA256]
    subset = cli(tmp_path / "run") + [
        "--public-cases",
        "heldout-public-004",
        "heldout-public-005",
        "--observation-profile=composed-visible-text-v3",
        "--observation-profile-helper-sha256=" + LAUNCHER.PROFILE_SHA256,
    ]
    forwarded, values = LAUNCHER._arguments(subset)
    assert forwarded == subset
    assert values.public_cases == ["heldout-public-004", "heldout-public-005"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--tasks", "upstream_travel"],
        ["--task", "upstream_travel"],
        ["--live-url", "https://en.wikipedia.org/wiki/Spoon"],
        ["--travel-challenges-only"],
        ["--suite-split", "holdout"],
        ["--max-steps", "17"],
        ["--repeats", "2"],
        ["--seed", "20261018"],
        ["--public-cases", "heldout-public-000", "heldout-public-000"],
        ["--public-cases", "public-python-counter-most-common-v1"],
        ["--observation-profile", "native-1231850"],
        ["--observation-profile-helper-sha256", "0" * 64],
    ],
)
def test_no_suite_mixing_budget_change_or_unknown_case(tmp_path, extra):
    with pytest.raises(SystemExit):
        LAUNCHER._arguments(cli(tmp_path / "run") + extra)


@pytest.mark.parametrize("elapsed_ms", [1.0, 8123.5], ids=["ordinary", "port-digits-in-time"])
def test_delegate_exact_copied_runner_without_task_success_claim(tmp_path, monkeypatch, elapsed_ms):
    calls = []

    def run(command, *, check):
        calls.append((command, check))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(LAUNCHER.subprocess, "run", run)
    ticks = iter([0.0, elapsed_ms / 1000])
    monkeypatch.setattr(LAUNCHER, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    args = cli(tmp_path / "run")
    assert LAUNCHER.main(["--bundle-dir", str(tmp_path / "bundle"), "--", *args]) == 0
    command, check = calls[0]
    assert command[:4] == [sys.executable, "-B", "-s", str(tmp_path / "bundle/runner.py")]
    assert command[4 : 4 + len(args)] == args
    assert check is False
    receipt = json.loads((tmp_path / "bundle/launch-receipt.json").read_text())
    assert receipt["status"] == "finished" and receipt["source_integrity_verified"] is True
    assert receipt["child_exit_is_task_success"] is False
    assert receipt["public_case_ids"] == list(LAUNCHER.PUBLIC_WEB_CASES)
    assert receipt["selection_case_count"] == receipt["requested_case_count"] == 6
    assert receipt["evaluation_scope"] == "predeclared_six_cases"
    assert (
        receipt["bundle_manifest_sha256"]
        == hashlib.sha256((tmp_path / "bundle/bundle-manifest.json").read_bytes()).hexdigest()
    )
    assert receipt["elapsed_ms"] == pytest.approx(elapsed_ms)
    if elapsed_ms == 8123.5:
        assert "8123" in str(receipt["elapsed_ms"])
    endpoint = args[args.index("--endpoint") + 1]
    assert endpoint not in json.dumps(receipt)  # Actual wire/credentials are not re-logged.
    assert urlsplit(endpoint).netloc not in json.dumps(receipt)
    assert (
        receipt["argument_sha256"]
        == hashlib.sha256(json.dumps(command[4:], ensure_ascii=False).encode()).hexdigest()
    )


def test_child_failure_and_midrun_source_failure_both_remain_visible(tmp_path, monkeypatch):
    def run(command, *, check):
        Path(command[3]).write_bytes(b"changed")
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(LAUNCHER.subprocess, "run", run)
    assert (
        LAUNCHER.main(["--bundle-dir", str(tmp_path / "bundle"), "--", *cli(tmp_path / "run")]) == 9
    )
    receipt = json.loads((tmp_path / "bundle/launch-receipt.json").read_text())
    assert receipt["returncode"] == 9
    assert receipt["source_integrity_verified"] is False
    assert "source changed" in receipt["source_integrity_error"]


def test_primary_child_exception_is_preserved_alongside_integrity_failure(tmp_path, monkeypatch):
    def run(command, *, check):
        Path(command[3]).unlink()
        raise RuntimeError("actual child start failed")

    monkeypatch.setattr(LAUNCHER.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="actual child start failed"):
        LAUNCHER.main(["--bundle-dir", str(tmp_path / "bundle"), "--", *cli(tmp_path / "run")])
    receipt = json.loads((tmp_path / "bundle/launch-receipt.json").read_text())
    assert receipt["error"] == "RuntimeError: actual child start failed"
    assert receipt["source_integrity_verified"] is False


def test_existing_run_output_refused_before_preparation_or_child(tmp_path, monkeypatch):
    monkeypatch.setattr(LAUNCHER.subprocess, "run", lambda *a, **k: pytest.fail("child called"))
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(SystemExit):
        LAUNCHER.main(["--bundle-dir", str(tmp_path / "bundle"), "--", *cli(run)])
    assert not (tmp_path / "bundle").exists()
