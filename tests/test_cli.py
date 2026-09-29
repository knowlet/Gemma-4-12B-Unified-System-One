import json
import subprocess
import sys


def test_installed_cli_and_report(tmp_path):
    report = tmp_path / "uniform.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "s1.cli",
            "benchmark",
            "--backend",
            "uniform",
            "examples/benchmarks/smoke.jsonl",
            "--output",
            str(report),
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(report.read_text())["counts"] == {"ok": 5}


def test_default_package_import_does_not_load_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import s1.cli, sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
