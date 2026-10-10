"""CPU integration preflight for the expensive, source-pinned research cell.

Execute the production profile closure with a fake model and benchmark backend.
The real typed answers, calibration objective, report serialization and auditor
are retained; no Modal connection or GPU inference is used.
"""

import ast
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import shutil
import sys
import threading
import types
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1.contracts import DecisionRequest, answer_from_probabilities
from s1.evaluation.datasets import EvaluationCase, load_cases

ROOT = Path(__file__).parents[1]
APP_PATH = ROOT / "apps/modal/breakthrough.py"
PROTOCOL_PATH = ROOT / "configs/experiments/jevbench-breakthrough-20261008.json"
PROTOCOL_SHA256 = "3a8dcceba9fbc0c7ce31bddbb85c0d10acdfa855ddf237f4d5e81982ec6a926d"


@pytest.fixture
def torch_module():
    return pytest.importorskip("torch")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def app_module(monkeypatch):
    class Image:
        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()

        def __getattr__(self, name):
            return lambda *args, **kwargs: self

    class App:
        app_id = "cpu-preflight"

        def __init__(self, name):
            self.name = name

        def function(self, **kwargs):
            return lambda fn: fn

        def local_entrypoint(self):
            return lambda fn: fn

    modal = types.ModuleType("modal")
    modal.App, modal.Image = App, Image
    modal.Volume = SimpleNamespace(from_name=lambda *a, **k: SimpleNamespace(commit=lambda: None))
    modal.is_local = lambda: True
    modal.current_function_call_id = lambda: "cpu-preflight-call"
    monkeypatch.delenv("S1_HF_SECRET", raising=False)
    monkeypatch.setitem(sys.modules, "modal", modal)
    return load_module("breakthrough_app_preflight", APP_PATH)


def closure(app, **bindings):
    """Run real nested definitions without the model/download setup prelude."""
    source = ast.parse(APP_PATH.read_text())
    run = next(
        node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_run"
    )
    nodes = [
        node
        for node in run.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in {"Predictor", "evaluate", "load"}
    ]
    namespace = {
        **app.__dict__,
        "hashlib": hashlib,
        "sys": sys,
        "DecisionRequest": DecisionRequest,
        "answer_from_probabilities": answer_from_probabilities,
        **bindings,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(APP_PATH), "exec"), namespace)
    return SimpleNamespace(**namespace)


def typed_request(state="observation", *, media=False):
    return DecisionRequest.model_validate(
        {
            "state": state,
            "questions": {
                "pick": {
                    "type": "choice",
                    "instructions": "Choose the observed item.",
                    "criteria": {"zeta": "First candidate", "alpha": "Second candidate"},
                },
                "accept": {"type": "noul", "instructions": "Does the condition hold?"},
                "grade": {
                    "type": "score",
                    "instructions": "Choose the applicable band.",
                    "criteria": {"-2": "low", "5": "middle", "9": "high"},
                },
            },
            "media": [{"type": "audio", "samples": [0.1, -0.1]}] if media else [],
        }
    )


def annotated(split, state, *, media=False):
    request = typed_request(state, media=media)
    return EvaluationCase(
        id=f"{split}-{state}",
        group_id=f"group-{state}",
        split=split,
        request=request,
        gold={q.id: q.labels()[-1] for q in request.questions},
        soft_gold={
            q.id: dict(zip(q.labels(), [0.2, 0.8] if len(q.labels()) == 2 else [0.1, 0.2, 0.7]))
            for q in request.questions
        },
    )


@pytest.fixture
def profile(app_module, monkeypatch, tmp_path, torch_module):
    torch = torch_module
    training = load_module(
        "breakthrough_training_preflight", ROOT / "scripts/breakthrough_training.py"
    )
    captures, benchmarks, commits = [], [], []

    def capture(model, request, prompt, head=None):
        captures.append({"request": request.model_dump(mode="json"), "head": head})
        return [
            {
                "question": question,
                "logits": 2
                * torch.tensor(
                    [0.2, 0.8] if len(question.labels()) == 2 else [0.1, 0.2, 0.7]
                ).log(),
                "sequence_tokens": 27,
            }
            for question in request.questions
        ]

    monkeypatch.setattr(training, "capture", capture)
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 456)
    cal, test = [annotated("calibration", "calibration")], [annotated("test", "heldout")]
    native = annotated("test", "native", media=True)
    release = {
        "test": [annotated("test", "released-text")],
        "media": [annotated("test", "released-media", media=True)],
        "regression_text": [annotated("test", "released-regression")],
    }

    def public(predict, tasks_root, output, identity):
        answers = predict("public", typed_request().named_questions_payload())
        benchmarks.append({"output": Path(output).name, "identity": identity, "response": answers})
        # Deliberately retain run_public's flat return shape; it has no ["summary"].
        return {"n_correct": 1, "n_tasks": 1, "identity": identity}

    class Report:
        def __init__(self, backend):
            self.overall, self.interval = 0.75, [0.6, 0.9]
            self.dimensions = {kind: 0.75 for kind in ("REP", "BAT", "MEA", "LOG", "CHO")}
            self.errors = []
            self.scores = {"overall": self.overall, "pillars": self.dimensions}
            self.backend = backend

        def save(self, path):
            training.write_json(
                path,
                {
                    "scores": self.scores,
                    "result": {"errors": self.errors, "backend": self.backend},
                    "settings": {},
                },
            )
            return Path(path)

    def coherence(backend, **kwargs):
        assert kwargs["suite"] == "jevbench-mini" and kwargs["concurrency"] == 1
        questions = typed_request().named_questions_payload()
        for question_order in (questions, dict(reversed(list(questions.items())))):
            returned = backend.fn("coherence", question_order)
            assert list(returned) == list(question_order)
        return Report(backend.name)

    jb = types.ModuleType("jevbench")
    jb.from_callable = lambda fn, *, name: SimpleNamespace(fn=fn, name=name)
    jb.evaluate = coherence
    monkeypatch.setitem(sys.modules, "jevbench", jb)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    receipt = {"gpu": "CPU preflight mock", "execution": "independent native", "profiles": []}
    env = closure(
        app_module,
        torch=torch,
        destination=tmp_path,
        source_hashes={"campaign": hashlib.sha256(APP_PATH.read_bytes()).hexdigest()},
        receipt=receipt,
        calibration=cal,
        test=test,
        release=release,
        capture=capture,
        records=training.records,
        fit_temperatures=training.fit_temperatures,
        summarize=training.summarize,
        write_json=training.write_json,
        run_public=public,
        load_cases=lambda path: [native],
        volume=SimpleNamespace(commit=lambda: commits.append(True)),
    )
    return SimpleNamespace(
        env=env,
        model=SimpleNamespace(lm=SimpleNamespace(eval=lambda: None)),
        captures=captures,
        benchmarks=benchmarks,
        commits=commits,
        receipt=receipt,
        native=native,
        root=tmp_path,
    )


@pytest.mark.inference
def test_profile_has_auditable_identity_and_complete_native_media(profile):
    name, revision = "released-user-control", "a" * 40
    profile.env.evaluate(
        profile.model, name, "user_question", {"model_id": "model", "revision": revision}
    )
    root = profile.root / name
    receipt = json.loads((root / "receipt.json").read_text())
    assert receipt["identity"]["coherence_commit"] == profile.env.COHERENCE
    assert receipt["identity"]["benchmarkheaven_commit"] == profile.env.HEAVEN
    assert receipt["identity"] == profile.benchmarks[0]["identity"]
    assert receipt["identity"]["reported_cost_usd"] is None
    assert receipt["official_rank"] is None
    audit = load_module(
        "breakthrough_campaign_auditor", ROOT / "scripts/summarize_jevbench_campaign.py"
    )
    audited = audit.coherence_summary(root / "coherence-mini.json", receipt)
    assert audited["overall"] == 0.75 and audited["errors"] == 0
    report = json.loads((root / "coherence-mini.json").read_text())
    assert report["result"]["backend"] == f"{name}@{revision}"
    assert any(
        row["request"] == profile.native.request.model_dump(mode="json") for row in profile.captures
    )
    assert json.loads((root / "native-fixture.json").read_text())[0]["modalities"] == ["audio"]
    regression = json.loads((root / "regression.json").read_text())
    assert set(regression) == {"test", "media", "regression_text"}
    assert all(row["n_questions"] == 3 and row["n_correct"] == 3 for row in regression.values())
    assert receipt["memory_peak_scope"].startswith("Whole profile")


@pytest.mark.inference
def test_all_primitive_calibration_replay_uses_frozen_logits_without_forward(profile):
    profile.env.evaluate(
        profile.model,
        "released-user-control",
        "user_question",
        {"model_id": "model", "revision": "a" * 40},
    )
    assert [row["output"] for row in profile.benchmarks] == [
        "public231",
        "public231-unit",
        "public231-published_global",
    ]
    raw, unit, published = [row["response"] for row in profile.benchmarks]
    assert raw["runtime"]["forward_calls"] == 3
    assert unit["runtime"]["forward_calls"] == published["runtime"]["forward_calls"] == 0
    assert (
        unit["runtime"]["raw_logits"]
        == published["runtime"]["raw_logits"]
        == raw["runtime"]["raw_logits"]
    )
    assert (
        len(profile.captures) == 9
    )  # cal/test, public, three release sets, native, two coherence requests
    for key, answer in raw["answers"].items():
        assert sum(answer["probabilities"].values()) == pytest.approx(1.0)
        assert (
            list(answer["probabilities"])
            == typed_request().questions[["pick", "accept", "grade"].index(key)].labels()
        )
    assert raw["answers"]["accept"]["noul"] == raw["answers"]["accept"]["probabilities"]["true"]
    assert set(raw["answers"]["grade"]["probabilities"]) == {"-2", "5", "9"}
    assert raw["answers"]["pick"]["choice"] == unit["answers"]["pick"]["choice"] == "alpha"
    assert raw["answers"]["pick"]["confidence"] < unit["answers"]["pick"]["confidence"]
    assert "NOT model inference latency" in profile.benchmarks[1]["identity"]["execution"]


@pytest.mark.inference
@pytest.mark.parametrize("tamper", ["id", "labels", "count", "unseen"])
def test_predictor_replay_rejects_wrong_request_or_label_identity(profile, tamper):
    predictor = profile.env.Predictor(
        profile.model, "user_question", {kind: 1.0 for kind in ("choice", "noul", "score")}
    )
    request = typed_request()
    predictor(request.state, request.named_questions_payload())
    saved = next(iter(predictor.raw.values()))
    if tamper == "id":
        saved[0]["id"] = "different-question"
    elif tamper == "labels":
        saved[0]["labels"].reverse()
    elif tamper == "count":
        saved.pop()
    predictor.replay = True
    with pytest.raises((ValueError, KeyError)):
        predictor(
            "unseen" if tamper == "unseen" else request.state, request.named_questions_payload()
        )
    assert len(profile.captures) == 1


@pytest.mark.parametrize(
    "run,stage", [("", "ablate"), ("../escape", "train"), ("Uppercase", "train"), ("x", "delete")]
)
def test_run_destination_rejects_unsafe_or_unknown_scope(app_module, run, stage):
    with pytest.raises(ValueError):
        app_module._path(run, stage)


def test_export_retains_trained_weights_for_declared_hashes(app_module, monkeypatch, tmp_path):
    weights = {
        "decision-head.pt": b"immutable-head",
        "mixed-lora/adapter_model.safetensors": b"immutable-lora",
    }
    for name, content in weights.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (tmp_path / "receipt.json").write_text(
        json.dumps({name: hashlib.sha256(value).hexdigest() for name, value in weights.items()})
    )
    monkeypatch.setattr(app_module, "_path", lambda *args: tmp_path)
    with zipfile.ZipFile(io.BytesIO(app_module.export("cpu-preflight", "train"))) as exported:
        assert set(weights) <= set(exported.namelist())
        for name, content in weights.items():
            assert exported.read(name) == content


@pytest.mark.inference
def test_load_pins_processor_bytes_and_seeds_trainable_adapter(app_module, tmp_path, torch_module):
    torch = torch_module
    files = {
        "config.json": b"{}",
        "model.safetensors": b"frozen-backbone",
        "tokenizer.model": b"frozen-tokenizer",
        "processor_config.json": b'{"sample_rate":16000}',
        "chat_template.jinja": b"frozen-template",
        "s1_config.json": b'{"temperature":2.7830344470383452}',
    }
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    helper = load_module(
        "checkpoint_identity_preflight", ROOT / "scripts/prepare_mlx_validation.py"
    )
    adapters = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    seeds, constructors, downloads = [], [], []

    def constructor(directory, **options):
        assert seeds == [42]
        constructors.append((directory, options))
        return SimpleNamespace(lm=adapters)

    env = closure(
        app_module,
        torch=SimpleNamespace(manual_seed=lambda value: seeds.append(value)),
        snapshot_download=lambda model, **options: (
            downloads.append((model, options)) or str(tmp_path)
        ),
        checkpoint_identity=helper.checkpoint_identity,
        UnifiedDecisionModel=constructor,
    )
    model, identity = env.load("released", lora=True)
    assert identity["model_id"] == app_module.WEIGHTS["released"][0]
    assert identity["revision"] == downloads[0][1]["revision"] == app_module.WEIGHTS["released"][1]
    for name in set(files) - {"model.safetensors"}:
        assert identity["processor_files"][name] == {
            "sha256": hashlib.sha256(files[name]).hexdigest(),
            "size_bytes": len(files[name]),
        }
    assert constructors[0][1]["temperature"] == 1.0
    assert constructors[0][1]["lora"]["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert all(parameter.dtype == torch.float32 for parameter in model.lm.parameters())


@pytest.fixture
def setup_preflight(app_module, monkeypatch, tmp_path, torch_module):
    """Execute real full-population validation, stopping before model loading."""
    torch = torch_module
    if not (ROOT / "artifacts/release/datasets/manifest.json").is_file():
        pytest.skip("requires locally prepared, source-pinned native release datasets")
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import breakthrough_training
    import huggingface_hub

    data = tmp_path / "data"
    generator = load_module(
        "breakthrough_data_preflight", ROOT / "scripts/prepare_breakthrough_data.py"
    )
    generator.prepare(data, train=1024, calibration=192, test=384)
    native = tmp_path / "native-regression.jsonl"
    native.write_bytes((ROOT / "examples/benchmarks/mps.jsonl").read_bytes())
    release_data = tmp_path / "release-data"
    release_data.mkdir()
    for source in (ROOT / "artifacts/release/datasets").iterdir():
        target = release_data / source.name
        if source.name == "manifest.json":
            target.write_bytes(source.read_bytes())
        else:
            target.symlink_to(source)
    mappings = {
        "/workspace/scripts": ROOT / "scripts",
        "/workspace/configs": ROOT / "configs",
        "/workspace/data": data,
        "/workspace/release-data": release_data,
        "/workspace/native-regression.jsonl": native,
    }

    def path(value):
        value = str(value)
        for prefix, mapped in mappings.items():
            if value == prefix or value.startswith(prefix + "/"):
                return mapped / value[len(prefix) :].lstrip("/")
        return Path(value)

    class SetupComplete(Exception):
        pass

    receipts, downloads = [], []

    def stop_before_load(path, value):
        receipts.append(value)
        raise SetupComplete

    monkeypatch.setattr(app_module, "Path", path)
    monkeypatch.setattr(app_module, "_path", lambda *args: tmp_path / "cloud-output")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU mock")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "cpu-preflight")
    monkeypatch.setattr(breakthrough_training, "write_json", stop_before_load)
    import release_training

    monkeypatch.setattr(release_training, "Path", path)
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **k: downloads.append(True)
    )
    return SimpleNamespace(
        run=lambda: app_module._run("cpu-preflight", "ablate"),
        complete=SetupComplete,
        receipts=receipts,
        downloads=downloads,
        data=data,
        release_data=release_data,
        native=native,
        mappings=mappings,
        output=tmp_path / "cloud-output",
    )


@pytest.mark.inference
def test_full_frozen_populations_and_executed_helpers_are_recorded_before_load(setup_preflight):
    with pytest.raises(setup_preflight.complete):
        setup_preflight.run()
    receipt = setup_preflight.receipts[0]
    assert receipt["split_audit"]["valid"]
    assert receipt["split_audit"]["split_counts"] == {
        "train": 1024,
        "calibration": 192,
        "test": 384,
    }
    assert receipt["release_data"]["train"]["cases"] == 2048
    assert receipt["release_data"]["media"]["cases"] == 52
    assert receipt["release_data"]["regression_text"]["cases"] == 128
    for key in ("release_data_loader", "checkpoint_identity_helper", "prospective_protocol"):
        archived = setup_preflight.output / "sources" / f"{key}.txt"
        assert hashlib.sha256(archived.read_bytes()).hexdigest() == receipt["source_files"][key]
    protocol_archive = setup_preflight.output / "sources/prospective_protocol.txt"
    assert protocol_archive.read_bytes() == PROTOCOL_PATH.read_bytes()
    assert receipt["source_files"]["prospective_protocol"] == PROTOCOL_SHA256
    protocol = json.loads(protocol_archive.read_text())
    for name, directory in (
        ("synthetic", setup_preflight.data),
        ("release_regression", setup_preflight.release_data),
    ):
        assert (
            hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
            == protocol["data"][name]["manifest_sha256"]
        )
    assert (
        hashlib.sha256(setup_preflight.native.read_bytes()).hexdigest()
        == protocol["data"]["native_fixture"]["file_sha256"]
    )
    native_cases = load_cases(setup_preflight.native)
    assert len(native_cases) == protocol["data"]["native_fixture"]["cases"] == 8
    assert (
        sum(len(case.request.questions) for case in native_cases)
        == protocol["data"]["native_fixture"]["questions"]
        == 29
    )
    assert not setup_preflight.downloads


@pytest.mark.inference
def test_changed_synthetic_bytes_are_rejected_before_any_model_load(setup_preflight):
    with (setup_preflight.data / "calibration.jsonl").open("a") as stream:
        stream.write("\n")  # Same valid cases, changed frozen bytes.
    with pytest.raises(ValueError, match="checksum|pin"):
        setup_preflight.run()
    assert not setup_preflight.receipts and not setup_preflight.downloads


@pytest.mark.inference
@pytest.mark.parametrize(
    "target", ["synthetic_manifest", "release_manifest", "native_fixture", "synthetic_provenance"]
)
def test_changed_frozen_manifest_fixture_or_auxiliary_bytes_fail_before_load(
    setup_preflight, target
):
    files = {
        "synthetic_manifest": setup_preflight.data / "manifest.json",
        "release_manifest": setup_preflight.release_data / "manifest.json",
        "native_fixture": setup_preflight.native,
        "synthetic_provenance": setup_preflight.data / "provenance.json",
    }
    # Valid JSON/JSONL with identical parsed content; byte pins must still reject.
    with files[target].open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum|pin"):
        setup_preflight.run()
    assert not setup_preflight.receipts and not setup_preflight.downloads


@pytest.mark.inference
def test_changed_prospective_protocol_bytes_fail_before_load(setup_preflight, tmp_path):
    configs = tmp_path / "configs"
    target = configs / "experiments" / PROTOCOL_PATH.name
    target.parent.mkdir(parents=True)
    shutil.copyfile(PROTOCOL_PATH, target)
    with target.open("a") as stream:
        stream.write("\n")
    setup_preflight.mappings["/workspace/configs"] = configs
    with pytest.raises(ValueError, match="protocol|pin"):
        setup_preflight.run()
    assert not setup_preflight.receipts and not setup_preflight.downloads


@pytest.fixture
def entrypoint(app_module, monkeypatch, tmp_path, request):
    """Run the actual local entrypoint and downloader with only provider I/O faked."""
    run, stage = "cpu-entrypoint", getattr(request, "param", "ablate")
    prefix = f"breakthrough/{run}/{stage}"
    source = b"# preserved executed campaign\n"
    gpu_receipt = {
        "run": run,
        "stage": stage,
        "status": "completed",
        "profiles": [{"name": "released-current"}],
        "source_files": {"campaign": hashlib.sha256(source).hexdigest()},
    }

    class Volume:
        def __init__(self):
            self.files = {
                "receipt.json": json.dumps(gpu_receipt).encode() + b"\n",
                "sources/campaign.txt": source,
                **{f"raw/{i:03}.json": json.dumps({"case": i}).encode() for i in range(205)},
            }
            if stage == "train":
                self.files.update(
                    {
                        "decision-head.pt": b"preserved research head",
                        "train-features.pt": b"preserved research features",
                        "mixed-lora/adapter/adapter_model.safetensors": b"preserved research adapter",
                    }
                )
            self.listdir = SimpleNamespace(aio=self.list)
            self.read_file = SimpleNamespace(aio=self.read)
            self.active, self.peak, self.lists = 0, 0, 0
            self.fail_member = None

        async def list(self, path, recursive):
            assert path == prefix and recursive
            self.lists += 1
            return [
                SimpleNamespace(path=f"{prefix}/{name}", size=len(data), type=1, mtime=1)
                for name, data in self.files.items()
            ]

        async def read(self, path):
            name = path.removeprefix(prefix + "/")
            if name == self.fail_member:
                raise FileExistsError("simulated exclusive-write failure")
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0)
                yield self.files[name]
            finally:
                self.active -= 1

        def commit(self):
            raise AssertionError("local export must not mutate the Volume")

    calls, volume = [], Volume()

    def experiment(run_arg, stage_arg):
        calls.append((run_arg, stage_arg))
        assert volume.lists == 0  # A terminal GPU call precedes every read.
        return gpu_receipt

    def serial_export(*args):
        raise AssertionError("default entrypoint must not launch the serial remote ZIP export")

    monkeypatch.setattr(app_module, "ROOT", tmp_path)
    monkeypatch.setattr(app_module, "volume", volume)
    monkeypatch.setattr(app_module, "experiment", SimpleNamespace(remote=experiment))
    monkeypatch.setattr(app_module, "export", SimpleNamespace(remote=serial_export))
    destination = tmp_path / "artifacts/jevbench/breakthrough-20261008/runs" / run / stage
    return SimpleNamespace(
        app=app_module,
        volume=volume,
        receipt=gpu_receipt,
        calls=calls,
        destination=destination,
        run=run,
        stage=stage,
    )


@pytest.mark.parametrize("entrypoint", ["ablate", "train"], indirect=True)
def test_default_entrypoint_downloads_complete_source_bound_evidence_without_remote_export(
    entrypoint, capsys
):
    entrypoint.app.main(entrypoint.run, entrypoint.stage)
    assert entrypoint.calls == [(entrypoint.run, entrypoint.stage)]
    assert entrypoint.volume.peak == 16 and entrypoint.volume.lists == 2
    for name, data in entrypoint.volume.files.items():
        assert (entrypoint.destination / name).read_bytes() == data
    assert not (entrypoint.destination / "results.zip").exists()
    metadata = entrypoint.destination / ".evidence-download"
    state = json.loads((metadata / "receipt.json").read_text())
    assert state["status"] == "completed" and not state["errors"]
    assert state["expected_files"] == len(state["members"]) == len(entrypoint.volume.files)
    assert state["executed_sources_verified"] == 1
    assert state["settings"] == {
        "concurrency": 16,
        "file_timeout_seconds": 120,
        "attempts": 3,
        "remote_mutations": False,
        "gpu_calls": 0,
    }
    helper_source = (ROOT / "scripts/export_breakthrough_evidence.py").read_bytes()
    assert (metadata / "export_breakthrough_evidence.py.txt").read_bytes() == helper_source
    assert state["source_sha256"] == hashlib.sha256(helper_source).hexdigest()
    assert (
        state["gpu_receipt_sha256"]
        == hashlib.sha256(entrypoint.volume.files["receipt.json"]).hexdigest()
    )
    for member in state["members"]:
        data = entrypoint.volume.files[member["path"]]
        assert member["sha256"] == hashlib.sha256(data).hexdigest()
        assert member["size_bytes"] == len(data)
    output = capsys.readouterr().out
    assert '"downloaded_files": 200' in output and "Saved " in output


def test_default_entrypoint_recovers_failed_gpu_evidence_and_reraises_original_error(
    entrypoint, monkeypatch
):
    gpu_error = RuntimeError("original GPU-stage failure")
    entrypoint.receipt["status"] = "failed"
    entrypoint.receipt["error"] = str(gpu_error)
    raw_receipt = json.dumps(entrypoint.receipt).encode() + b"\n"
    entrypoint.volume.files["receipt.json"] = raw_receipt

    def fail(*args):
        raise gpu_error

    monkeypatch.setattr(entrypoint.app, "experiment", SimpleNamespace(remote=fail))
    with pytest.raises(RuntimeError) as caught:
        entrypoint.app.main(entrypoint.run, entrypoint.stage)
    assert caught.value is gpu_error
    assert (entrypoint.destination / "receipt.json").read_bytes() == raw_receipt
    state = json.loads((entrypoint.destination / ".evidence-download/receipt.json").read_text())
    assert state["status"] == "completed" and state["gpu_receipt_status"] == "failed"


def test_default_entrypoint_retains_partial_transfer_and_propagates_failure(entrypoint, capsys):
    entrypoint.volume.fail_member = "raw/000.json"
    with pytest.raises(ValueError, match="download incomplete"):
        entrypoint.app.main(entrypoint.run, entrypoint.stage)
    state = json.loads((entrypoint.destination / ".evidence-download/receipt.json").read_text())
    assert state["status"] == "failed"
    assert state["errors"] == [
        {"path": "raw/000.json", "error_type": "FileExistsError", "attempts": 1}
    ]
    assert (entrypoint.destination / "receipt.json").read_bytes() == entrypoint.volume.files[
        "receipt.json"
    ]
    assert "Saved " not in capsys.readouterr().out


def test_default_entrypoint_preserves_gpu_error_when_recovery_also_fails(entrypoint, monkeypatch):
    gpu_error = RuntimeError("original GPU-stage failure")
    entrypoint.volume.fail_member = "raw/000.json"
    entrypoint.receipt["status"] = "failed"
    entrypoint.volume.files["receipt.json"] = json.dumps(entrypoint.receipt).encode() + b"\n"

    def fail(*args):
        raise gpu_error

    monkeypatch.setattr(entrypoint.app, "experiment", SimpleNamespace(remote=fail))
    with pytest.raises(RuntimeError) as caught:
        entrypoint.app.main(entrypoint.run, entrypoint.stage)
    assert caught.value is gpu_error
    assert isinstance(caught.value.__cause__, ValueError)
    assert "download incomplete" in str(caught.value.__cause__)


def test_default_entrypoint_refuses_existing_destination_before_gpu_call(entrypoint):
    entrypoint.destination.mkdir(parents=True)
    existing = entrypoint.destination / "preserved.txt"
    existing.write_bytes(b"keep existing evidence")
    with pytest.raises(FileExistsError):
        entrypoint.app.main(entrypoint.run, entrypoint.stage)
    assert entrypoint.calls == [] and entrypoint.volume.lists == 0
    assert existing.read_bytes() == b"keep existing evidence"


@pytest.fixture
def installed_modal():
    """Load the optional real SDK before app_module temporarily installs its fake."""
    sdk = pytest.importorskip("modal")
    cli = importlib.import_module("modal.cli.run")
    async_utils = importlib.import_module("modal._utils.async_utils")
    click_testing = importlib.import_module("click.testing")
    return SimpleNamespace(
        sdk=sdk,
        cli=cli,
        synchronize_api=async_utils.synchronize_api,
        CliRunner=click_testing.CliRunner,
    )


def test_actual_modal_cli_sync_entrypoint_download_runs_outside_provider_event_loop(
    installed_modal, entrypoint, monkeypatch
):
    """Exercise the SDK's CLI and synchronizer while replacing only provider setup/I/O."""
    monkeypatch.setitem(sys.modules, "modal", installed_modal.sdk)
    sdk_app = installed_modal.sdk.App("offline-entrypoint-boundary")
    sdk_entrypoint = sdk_app.local_entrypoint()(entrypoint.app.main)
    observations = {}

    @asynccontextmanager
    async def provider_context(app, **kwargs):
        assert app.name == "offline-entrypoint-boundary"
        observations["provider_thread"] = threading.get_ident()
        observations["provider_loop"] = asyncio.get_running_loop()
        yield app

    # Modal's real run_app also wraps an async context manager with this SDK API.
    monkeypatch.setattr(
        installed_modal.cli,
        "run_app",
        installed_modal.synchronize_api(provider_context),
    )
    download = entrypoint.app._download_evidence

    def observed_download(*args):
        observations["user_thread"] = threading.get_ident()
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        return download(*args)  # Real asyncio.run and complete bounded downloader.

    monkeypatch.setattr(entrypoint.app, "_download_evidence", observed_download)
    command = installed_modal.cli._get_click_command_for_local_entrypoint(sdk_app, sdk_entrypoint)
    result = installed_modal.CliRunner().invoke(
        command,
        ["--run", entrypoint.run, "--stage", entrypoint.stage],
        obj={
            "detach": False,
            "name": None,
            "env": None,
            "interactive": False,
            "show_progress": False,
            "show_timestamps": False,
            "result_path": None,
        },
    )
    assert result.exit_code == 0, result.output
    assert result.exception is None and "Saved " in result.output
    assert observations["provider_thread"] != observations["user_thread"]
    assert entrypoint.calls == [(entrypoint.run, entrypoint.stage)]
    assert entrypoint.volume.peak == 16 and entrypoint.volume.lists == 2
    state = json.loads((entrypoint.destination / ".evidence-download/receipt.json").read_text())
    assert state["status"] == "completed" and len(state["members"]) == 207
