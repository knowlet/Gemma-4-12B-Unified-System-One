import json
import subprocess
import sys

import pytest


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


@pytest.mark.parametrize("command", ["decide", "benchmark", "serve"])
def test_quantized_adapter_options_reach_gemma_backend(monkeypatch, tmp_path, command):
    from types import SimpleNamespace

    from s1 import api, cli
    from s1.backends import UniformBackend

    calls = []

    def gemma(model, **kwargs):
        calls.append((model, kwargs))
        return UniformBackend()

    monkeypatch.setattr(cli, "GemmaBackend", gemma)
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda *a, **kw: None))
    monkeypatch.setattr(api, "create_app", lambda *a, **kw: object())
    argv = [
        command,
        "--model",
        "owner/base",
        "--revision",
        "a" * 40,
        "--device",
        "cuda:1",
        "--precision",
        "bfloat16",
        "--quantization",
        "nf4",
        "--adapter-path",
        str(tmp_path / "ce-adapter"),
    ]
    if command == "decide":
        argv.append("examples/request.json")
    elif command == "benchmark":
        argv.extend(["examples/benchmarks/smoke.jsonl", "--output", str(tmp_path / "run.json")])
    assert cli.main(argv) == 0
    assert calls == [
        (
            "owner/base",
            {
                "revision": "a" * 40,
                "device": "cuda:1",
                "precision": "bfloat16",
                "quantization": "nf4",
                "adapter_path": str(tmp_path / "ce-adapter"),
            },
        )
    ]


@pytest.mark.parametrize("backend", ["uniform", "laya", "http"])
@pytest.mark.parametrize(
    "option",
    [
        ["--quantization", "int8"],
        ["--adapter-path", "adapter"],
        ["--precision", "float32"],
        ["--dtype", "float16"],
        ["--attn-implementation", "sdpa"],
    ],
)
def test_non_gemma_rejects_gemma_inference_settings(monkeypatch, capsys, backend, option):
    from s1 import cli

    def fail_load(*args, **kwargs):
        pytest.fail("invalid settings must fail before constructing any backend")

    for name in ("GemmaBackend", "UniformBackend", "LayaBackend", "HTTPBackend"):
        monkeypatch.setattr(cli, name, fail_load)
    with pytest.raises(SystemExit) as failure:
        cli.main(["decide", "examples/request.json", "--backend", backend, *option])
    assert failure.value.code == 2
    assert "require --backend gemma" in capsys.readouterr().err


@pytest.mark.parametrize(
    "option",
    [
        ["--device", "cpu"],
        ["--device", "mps"],
        ["--precision", "float32"],
        ["--dtype", "float32"],
        ["--dtype", "float16"],
    ],
)
def test_invalid_quantized_precision_fails_before_loading(monkeypatch, capsys, option):
    from s1 import cli

    monkeypatch.setattr(cli, "GemmaBackend", lambda *a, **kw: pytest.fail("must not load"))
    with pytest.raises(SystemExit) as failure:
        cli.main(["decide", "examples/request.json", "--quantization", "nf4", *option])
    assert failure.value.code == 2
    assert "requires CUDA and bfloat16" in capsys.readouterr().err


def test_quantization_is_not_a_training_cli_option(capsys):
    from s1 import cli

    with pytest.raises(SystemExit) as failure:
        cli.main(
            [
                "train",
                "--train-data",
                "train.jsonl",
                "--calibration-data",
                "cal.jsonl",
                "--output",
                "adapter",
                "--quantization",
                "nf4",
            ]
        )
    assert failure.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_gemma_cli_forwards_runtime_settings(tmp_path, decision_request, monkeypatch):
    from s1 import cli
    from s1.backends import UniformBackend

    request_path = tmp_path / "request.json"
    request_path.write_text(decision_request.model_dump_json())
    calls = []

    def backend(model, **kwargs):
        calls.append((model, kwargs))
        return UniformBackend()

    monkeypatch.setattr(cli, "GemmaBackend", backend)
    assert (
        cli.main(
            [
                "decide",
                "--model",
                "owner/model",
                "--device",
                "mps",
                "--dtype",
                "float16",
                "--attn-implementation",
                "eager",
                str(request_path),
            ]
        )
        == 0
    )
    assert calls == [
        (
            "owner/model",
            {
                "revision": None,
                "device": "mps",
                "precision": None,
                "quantization": "none",
                "adapter_path": None,
                "dtype": "float16",
                "attn_implementation": "eager",
            },
        )
    ]


def test_conflicting_precision_aliases_fail_before_loading(monkeypatch, capsys):
    from s1 import cli

    monkeypatch.setattr(cli, "GemmaBackend", lambda *a, **kw: pytest.fail("must not load"))
    with pytest.raises(SystemExit) as failure:
        cli.main(
            [
                "decide",
                "examples/request.json",
                "--device",
                "cuda",
                "--precision",
                "bfloat16",
                "--dtype",
                "float32",
            ]
        )
    assert failure.value.code == 2
    assert "must agree" in capsys.readouterr().err


def test_training_cli_forwards_runtime_settings(tmp_path, benchmark_case, monkeypatch):
    pytest.importorskip("torch")
    from s1 import cli, training, unified

    paths = {}
    for split in ("train", "calibration"):
        case = benchmark_case.model_copy(deep=True)
        case.split = split
        case.id = split
        paths[split] = tmp_path / f"{split}.jsonl"
        paths[split].write_text(case.model_dump_json() + "\n")
    calls = []

    class Model:
        def __init__(self, name, **kwargs):
            calls.append(kwargs)

        def save(self, path):
            path.mkdir(parents=True)

    monkeypatch.setattr(unified, "UnifiedDecisionModel", Model)
    monkeypatch.setattr(training, "train_model", lambda *args, **kwargs: {"steps": 0})
    assert (
        cli.main(
            [
                "train",
                "--train-data",
                str(paths["train"]),
                "--calibration-data",
                str(paths["calibration"]),
                "--output",
                str(tmp_path / "output"),
                "--device",
                "mps",
                "--dtype",
                "bfloat16",
                "--attn-implementation",
                "sdpa",
            ]
        )
        == 0
    )
    assert calls[0]["device"] == "mps"
    assert calls[0]["dtype"] == "bfloat16"
    assert calls[0]["attn_implementation"] == "sdpa"
