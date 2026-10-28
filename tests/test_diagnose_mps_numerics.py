"""No-model lifecycle and CPU-only observational tests for the MPS diagnostic."""

from __future__ import annotations

import copy
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_mps_numerics", ROOT / "scripts/diagnose_mps_numerics.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class Question:
    def __init__(self, value):
        self.__dict__.update(copy.deepcopy(value))

    def model_copy(self, deep=False):
        return copy.deepcopy(self)


class Request:
    def __init__(self, value):
        self.state = copy.deepcopy(value["state"])
        self.media = copy.deepcopy(value.get("media", []))
        self.questions = [Question(q) for q in value["questions"]]

    @classmethod
    def model_validate(cls, value):
        return cls(value)

    def model_copy(self, deep=False, update=None):
        result = copy.deepcopy(self)
        result.__dict__.update(update or {})
        return result


def request():
    return {
        "state": {"page": "original observed controls"},
        "media": [],
        "questions": [
            {
                "id": "operation",
                "type": "choice",
                "instructions": "Select observed operation",
                "criteria": {"a": "A", "b": "B"},
            },
            {
                "id": "target",
                "type": "choice",
                "instructions": "Select observed target",
                "criteria": {"x": "X", "y": "Y"},
            },
        ],
    }


def helper():
    return {
        "instruction": "Return the field value as JSON.",
        "context": {"goal": "Fill the visible field from the goal"},
        "max_new_tokens": 128,
        "do_sample": False,
    }


def maintenance(error=None):
    return {
        "before": {
            "current_allocated_bytes": 20,
            "driver_allocated_bytes": 80,
            "recommended_max_bytes": 100,
        },
        "after": {
            "current_allocated_bytes": 20,
            "driver_allocated_bytes": 24,
            "recommended_max_bytes": 100,
        },
        "error": error,
        "wall_ms": 0,
    }


@contextmanager
def no_observers(model, runtime, events, phase, *, decision_callbacks=True):
    events.append({"phase": phase, "event": "fake lifecycle observer"})
    yield


def test_selected_inputs_preserve_complete_request_and_original_helper_budget(tmp_path):
    decision, text = tmp_path / "decision.json", tmp_path / "helper.json"
    RUNNER.write_json(
        decision, [{"mapped_request": request(), "error": {"message": "original failure"}}]
    )
    RUNNER.write_json(
        text,
        [
            {
                **helper(),
                "raw_text": "malformed prior text",
                "error": {"message": "prior parse error"},
            }
        ],
    )
    actual_request, actual_helper = RUNNER.read_inputs(decision, 1, text, 1)
    assert actual_request == request() and actual_helper == helper()
    assert "raw_text" not in actual_helper and "error" not in actual_helper
    with pytest.raises(ValueError, match="one-based"):
        RUNNER.read_inputs(decision, 0, text, 1)


def test_all_fixed_phases_run_without_fallback_and_singletons_keep_exact_questions(
    tmp_path, monkeypatch
):
    seen = []

    class Backend:
        model = SimpleNamespace(device="mps")
        question_mode = "independent"

        def predict(self, value):
            seen.append(copy.deepcopy(value))
            if len(value.questions) > 1:
                raise ValueError("original invalid probabilities")
            return {"answers": {value.questions[0].id: {"unchanged": "raw singleton response"}}}

    runtime = SimpleNamespace(
        contracts=SimpleNamespace(DecisionRequest=Request),
        torch=SimpleNamespace(mps=SimpleNamespace(synchronize=lambda: None)),
    )
    guard = SimpleNamespace(maintain_mps_allocator=lambda torch: maintenance())
    monkeypatch.setattr(RUNNER, "observe_model", no_observers)
    monkeypatch.setattr(
        RUNNER,
        "generate_native",
        lambda *args: {
            "raw_text": "malformed text",
            "error": {"type": "JSONDecodeError", "message": "raw preserved"},
        },
    )
    original = request()
    rows = RUNNER.execute_experiment(Backend(), runtime, guard, original, helper(), tmp_path)
    assert [row["phase"] for row in rows] == list(RUNNER.PHASES)
    assert [len(value.questions) for value in seen] == [2, 1, 1, 2, 1, 1]
    assert (
        rows[0]["errors"][0]["message"]
        == rows[3]["errors"][0]["message"]
        == "original invalid probabilities"
    )
    assert (
        rows[1]["merged_answers"]
        == rows[4]["merged_answers"]
        == {
            "operation": {"unchanged": "raw singleton response"},
            "target": {"unchanged": "raw singleton response"},
        }
    )
    assert rows[2]["calls"][0]["response"]["raw_text"] == "malformed text"
    assert sum(row["generation_calls"] for row in rows) == 1
    assert original == request()
    for index in [0, 3]:
        assert seen[index].state == original["state"]
    for value in seen:
        for question in value.questions:
            expected = next(q for q in original["questions"] if q["id"] == question.id)
            assert question.__dict__ == expected
    assert rows[0]["request_sha256"] == rows[3]["request_sha256"] == RUNNER.json_hash(original)


def test_helper_only_runs_one_generation_without_decision_validation_or_callbacks(
    tmp_path, monkeypatch
):
    seen = []

    class Backend:
        model = SimpleNamespace(device="mps")

        def predict(self, value):
            pytest.fail("helper-only must not dispatch a decision")

    @contextmanager
    def observers(model, runtime, events, phase, *, decision_callbacks=True):
        assert phase == "text_generation" and decision_callbacks is False
        seen.append("generation_observers")
        yield

    def generation(model, torch, original_helper):
        assert original_helper == helper()
        seen.append("generation")
        return {
            "raw_text": '```json\n{"text": "Graph theory"}\n```',
            "generated_token_ids": [1, 2, 3],
            "error": {"type": "JSONDecodeError", "message": "strict JSON failed"},
        }

    # No contracts module is provided: helper-only must not validate a
    # DecisionRequest or construct any decision execution path.
    runtime = SimpleNamespace(torch=SimpleNamespace(mps=SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setattr(RUNNER, "observe_model", observers)
    monkeypatch.setattr(RUNNER, "generate_native", generation)
    original = request()
    rows = RUNNER.execute_experiment(
        Backend(),
        runtime,
        SimpleNamespace(maintain_mps_allocator=lambda torch: maintenance()),
        original,
        helper(),
        tmp_path,
        helper_only=True,
    )
    assert seen == ["generation_observers", "generation"]
    assert RUNNER.experiment_phases(True) == ("text_generation",)
    assert [row["phase"] for row in rows] == ["text_generation"]
    assert rows[0]["backend_calls"] == 0 and rows[0]["generation_calls"] == 1
    assert rows[0]["question_order"] == []
    assert rows[0]["request_sha256"] == RUNNER.json_hash(original)
    assert rows[0]["calls"][0]["response"]["generated_token_ids"] == [1, 2, 3]
    assert rows[0]["calls"][0]["response"]["raw_text"].startswith("```json")
    assert rows[0]["errors"] == [{"type": "JSONDecodeError", "message": "strict JSON failed"}]
    assert json.loads((tmp_path / "results.json").read_text()) == rows
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "results.json",
        "text_generation.json",
    ]


def test_helper_only_real_observer_does_not_access_decision_projection_modules():
    handles = []

    class HookModule:
        def register_forward_hook(self, callback):
            handle = SimpleNamespace(removed=False)
            handle.remove = lambda: setattr(handle, "removed", True)
            handles.append(handle)
            return handle

        def register_forward_pre_hook(self, callback, **kwargs):
            return self.register_forward_hook(callback)

    backbone = HookModule()
    backbone.language_model = SimpleNamespace(layers=[HookModule()])
    model = SimpleNamespace(backbone=backbone, head=HookModule(), lm=HookModule())
    # Neither candidate-projection module is present. Generation observers
    # still install/remove their hooks without accessing either module.
    with RUNNER.observe_model(
        model, SimpleNamespace(), [], "text_generation", decision_callbacks=False
    ):
        assert len(handles) == 6 and not any(handle.removed for handle in handles)
    assert all(handle.removed for handle in handles)


def test_subfailure_is_preserved_without_manufacturing_merged_answers(tmp_path, monkeypatch):
    class Backend:
        model = SimpleNamespace(device="mps")

        def predict(self, value):
            if value.questions[0].id == "target":
                raise RuntimeError("target failed")
            return {"answers": {value.questions[0].id: {"raw": True}}}

    runtime = SimpleNamespace(
        contracts=SimpleNamespace(DecisionRequest=Request),
        torch=SimpleNamespace(mps=SimpleNamespace(synchronize=lambda: None)),
    )
    monkeypatch.setattr(RUNNER, "observe_model", no_observers)
    monkeypatch.setattr(RUNNER, "generate_native", lambda *args: {"raw_text": "{}"})
    rows = RUNNER.execute_experiment(
        Backend(),
        runtime,
        SimpleNamespace(maintain_mps_allocator=lambda torch: maintenance()),
        request(),
        helper(),
        tmp_path,
    )
    assert rows[1]["merged_answers"] is rows[4]["merged_answers"] is None
    assert rows[1]["calls"][1]["error"]["message"] == "target failed"
    assert rows[1]["calls"][0]["response"]["answers"]["operation"] == {"raw": True}


def test_inference_error_survives_maintenance_error():
    def operation():
        raise RuntimeError("original inference failure")

    runtime = SimpleNamespace(torch=SimpleNamespace(mps=SimpleNamespace(synchronize=lambda: None)))
    result = RUNNER.measured_call(
        operation,
        runtime,
        SimpleNamespace(
            maintain_mps_allocator=lambda torch: maintenance(
                {"type": "RuntimeError", "message": "trim failed"}
            )
        ),
    )
    assert result["error"]["message"] == "original inference failure"
    assert result["allocator_maintenance"]["error"]["message"] == "trim failed"


def fixtures(tmp_path):
    source = tmp_path / "original"
    (source / "src/s1").mkdir(parents=True)
    (source / "src/s1/unified.py").write_text("original frozen Unified bytes")
    (source / "src/s1/backends.py").write_text("original frozen backend bytes")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    rows = []
    for name in [
        "model.safetensors",
        "config.json",
        "s1_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        path = checkpoint / name
        path.write_text("tiny fixture")
        rows.append(
            {"path": name, "size_bytes": path.stat().st_size, "sha256": RUNNER.digest(path)}
        )
    RUNNER.write_json(checkpoint / "artifact-manifest.json", {"files": rows})
    decision, text = tmp_path / "decision.json", tmp_path / "text.json"
    RUNNER.write_json(decision, [{"mapped_request": request()}])
    RUNNER.write_json(text, [helper()])
    return SimpleNamespace(
        source_root=source,
        checkpoint=checkpoint,
        decision_trace=decision,
        helper_trace=text,
        decision_record=1,
        helper_record=1,
        output=tmp_path / "output",
        question_mode="independent",
        helper_only=False,
        wait_for_pid=[1, 2],
        poll_seconds=0.01,
    )


@pytest.mark.parametrize("missing_package", [False, True])
def test_capture_freeze_and_wait_precede_any_torch_or_model_import(
    tmp_path, monkeypatch, missing_package
):
    args = fixtures(tmp_path)
    events = []

    def wait(handle, save, seconds):
        assert (
            args.output / "sources/src/s1/unified.py"
        ).read_text() == "original frozen Unified bytes"
        events.append(f"terminal-{handle['pid']}")

    guard = SimpleNamespace(
        ALLOCATOR_POLICY={"name": "test maintenance"},
        package_version=lambda name: None if missing_package else "test-version",
        capture_handle=lambda pid: {"pid": pid, "identity": "original identity"},
        wait_handle=wait,
        maintain_mps_allocator=lambda torch: maintenance(),
    )
    monkeypatch.setattr(RUNNER, "load_module", lambda *args: guard)
    monkeypatch.setattr(RUNNER.importlib.metadata, "version", lambda name: "test-version")

    class Backend:
        def __init__(self, *args, **kwargs):
            events.append("model-load")
            assert kwargs["device"] == "mps" and kwargs["dtype"] == "bfloat16"
            self.model = SimpleNamespace(device="mps")
            self.metadata = {"model": "test full Unified"}

    def runtime(output):
        assert events == ["terminal-1", "terminal-2"]
        events.append("torch-import")
        return SimpleNamespace(backends=SimpleNamespace(GemmaBackend=Backend), torch=None)

    monkeypatch.setattr(RUNNER, "load_runtime", runtime)
    monkeypatch.setattr(RUNNER, "execute_experiment", lambda *args, **kwargs: [])
    assert RUNNER.run(args) == 0
    assert events == ["terminal-1", "terminal-2", "torch-import", "model-load"]
    assert (args.output / "sources/scripts/resume_mps_benchmarks.py").is_file()
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["packages"]["torch"] == (None if missing_package else "test-version")
    assert receipt["phases"] == list(RUNNER.PHASES) and len(receipt["phases"]) == 5
    assert receipt["helper_only"] is False


def test_helper_only_receipt_declares_phase_and_no_numerical_acceptance(tmp_path, monkeypatch):
    args = fixtures(tmp_path)
    args.helper_only = True
    args.wait_for_pid = []
    guard = SimpleNamespace(
        ALLOCATOR_POLICY={},
        package_version=lambda name: "test-version",
        maintain_mps_allocator=lambda torch: maintenance(),
    )

    class Backend:
        model = SimpleNamespace(device="mps")
        metadata = {"model": "test full Unified"}

        def __init__(self, *args, **kwargs):
            pass

        def predict(self, value):
            pytest.fail("helper-only must not dispatch a decision")

    runtime = SimpleNamespace(
        backends=SimpleNamespace(GemmaBackend=Backend),
        torch=SimpleNamespace(mps=SimpleNamespace(synchronize=lambda: None)),
    )
    monkeypatch.setattr(RUNNER, "load_module", lambda *args: guard)
    monkeypatch.setattr(RUNNER, "load_runtime", lambda output: runtime)
    monkeypatch.setattr(RUNNER, "observe_model", no_observers)
    monkeypatch.setattr(RUNNER, "generate_native", lambda *args: {"raw_text": '{"text":"value"}'})
    assert RUNNER.run(args) == 0
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["phases"] == ["text_generation"]
    assert receipt["helper_only"] is True
    assert receipt["numerical_acceptance"] == receipt["scoring"] == "not performed"
    assert receipt["backend_calls"] == 0 and receipt["generation_calls"] == 1


@pytest.mark.parametrize("helper_only", [False, True])
def test_helper_only_cli_is_opt_in(tmp_path, monkeypatch, helper_only):
    seen = []
    monkeypatch.setattr(RUNNER, "run", lambda args: seen.append(args) or 0)
    argv = []
    for flag in ["decision-trace", "helper-trace", "checkpoint", "source-root", "output"]:
        argv.extend(["--" + flag, str(tmp_path / flag)])
    argv.extend(["--decision-record", "1", "--helper-record", "1"])
    if helper_only:
        argv.append("--helper-only")
    assert RUNNER.main(argv) == 0
    assert len(seen) == 1 and seen[0].helper_only is helper_only


def test_changed_frozen_input_blocks_model_loading(tmp_path, monkeypatch):
    args = fixtures(tmp_path)

    def wait(handle, save, seconds):
        (args.output / "sources/inputs/decision-trace.json").write_text("mutated")

    guard = SimpleNamespace(
        ALLOCATOR_POLICY={},
        package_version=lambda name: "test-version",
        capture_handle=lambda pid: {"pid": pid},
        wait_handle=wait,
    )
    monkeypatch.setattr(RUNNER, "load_module", lambda *args: guard)
    monkeypatch.setattr(RUNNER.importlib.metadata, "version", lambda name: "test-version")
    monkeypatch.setattr(RUNNER, "load_runtime", lambda output: pytest.fail("model must not load"))
    with pytest.raises(ValueError, match="changed while queued"):
        RUNNER.run(args)
    assert json.loads((args.output / "receipt.json").read_text())["status"] == "error"


def test_checkpoint_rejects_mutated_weights_before_model_load(tmp_path):
    args = fixtures(tmp_path)
    identity = RUNNER.digest(args.checkpoint / "artifact-manifest.json")
    RUNNER.verify_checkpoint(args.checkpoint, identity)
    (args.checkpoint / "model.safetensors").write_text("tampered")
    with pytest.raises(ValueError, match="member differs"):
        RUNNER.verify_checkpoint(args.checkpoint, identity)


def test_cpu_tensor_statistics_separates_padded_nonfinite_values_without_mutation():
    torch = pytest.importorskip("torch")
    tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [float("nan"), float("inf")]]])
    mask = torch.tensor([[1, 1, 0]])
    before = tensor.clone()
    stats = RUNNER.tensor_statistics(tensor, torch, mask, max_values=2)
    assert stats["all"]["values"] == 6 and stats["all"]["finite"] == 4
    assert stats["valid_query_rows"]["nan"] == 0
    assert stats["padded_query_rows"]["nan"] == stats["padded_query_rows"]["positive_inf"] == 1
    torch.testing.assert_close(tensor, before, equal_nan=True)


@pytest.mark.parametrize("row_offset", [0, 1, 68, 136])
def test_offset_views_copy_original_dtype_before_cpu_cast_and_preserve_inputs(row_offset):
    torch = pytest.importorskip("torch")
    transfers = []

    class TransferSpy:
        def __init__(self, tensor):
            self.tensor = tensor

        @property
        def shape(self):
            return self.tensor.shape

        @property
        def ndim(self):
            return self.tensor.ndim

        @property
        def dtype(self):
            return self.tensor.dtype

        @property
        def device(self):
            return "mps"

        def detach(self):
            return TransferSpy(self.tensor.detach())

        def reshape(self, *shape):
            return TransferSpy(self.tensor.reshape(*shape))

        def __getitem__(self, key):
            return TransferSpy(self.tensor[key])

        def to(self, *args, **kwargs):
            assert args == () and kwargs == {"device": "cpu"}, (
                "combined device/dtype transfer is forbidden"
            )
            transfers.append((self.tensor.dtype, self.tensor.storage_offset()))
            # Return a CPU tensor in the *original* dtype. Subsequent .float()
            # or .bool() must therefore be a separate CPU-only conversion.
            return self.tensor.clone()

    base = torch.full((205, 3840), 0.5, dtype=torch.bfloat16)
    mask_base = torch.zeros(205, dtype=torch.int64)
    mask_base[row_offset : row_offset + 68] = 1
    values = base[row_offset : row_offset + 68]
    masks = mask_base[row_offset : row_offset + 68]
    base_before, mask_before = base.clone(), mask_base.clone()
    stats = RUNNER.tensor_statistics(
        TransferSpy(values), torch, TransferSpy(masks), max_values=262144
    )
    assert stats["all"]["values"] == 68 * 3840
    assert stats["all"]["finite"] == 68 * 3840
    assert stats["all"]["finite_min"] == stats["all"]["finite_max"] == 0.5
    assert stats["valid_query_rows"]["values"] == 68 * 3840
    assert transfers[0] == (torch.int64, row_offset)
    assert any(
        dtype == torch.bfloat16 and offset == row_offset * 3840 for dtype, offset in transfers
    )
    torch.testing.assert_close(base, base_before, atol=0, rtol=0)
    torch.testing.assert_close(mask_base, mask_before, atol=0, rtol=0)


def test_new_observer_policy_explicitly_invalidates_prior_instrumented_comparison():
    assert RUNNER.TELEMETRY_POLICY["combined_device_dtype_transfer"] is False
    invalidation = RUNNER.TELEMETRY_POLICY["prior_observer_invalidation"]
    assert invalidation["status"] == "invalid_for_unchanged_model_comparisons"
    assert "20261028-native-control-01" in invalidation["run"]
    assert "int64 input masks" in invalidation["reason"]
