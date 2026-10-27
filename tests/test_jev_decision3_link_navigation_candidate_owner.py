"""Synthetic candidate owner contracts; no model, GPU or systemd unit is started."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import jev_decision3_link_navigation_candidate_owner as owner
import numpy as np
import pytest
from test_jev_decision3_link_navigation_gpu_owner import plan as feature_plan


def candidate_plan(tmp_path):
    result = feature_plan(tmp_path)
    result["schema"] = owner.SCHEMA
    result["deltas"] = {
        head: {
            "path": str(tmp_path / (head + ".npy")),
            "file_sha256": "a" * 64,
            "fp32_sha256": "b" * 64,
        }
        for head in ("operation", "click_target")
    }
    return result


def test_candidate_plan_source_gate_and_resource_limits(tmp_path):
    plan = candidate_plan(tmp_path)
    owner._validate_plan(plan)
    with pytest.raises(ValueError, match="independent review"):
        owner.supervise(tmp_path / "absent", "a" * 64)
    args = owner._unit_args(
        "jev-link-candidate-test.service", tmp_path / "plan", "a" * 64, tmp_path, plan
    )
    assert "--property=MemoryMax=" + str(owner.MEMORY_MAX) in args
    assert "--property=MemorySwapMax=0" in args
    assert "--property=CPUQuota=500%" in args
    assert "--property=AllowedCPUs=15-19" in args
    assert "--property=RuntimeMaxSec=120" in args
    assert str(Path(owner.__file__).absolute()) in args
    for mutate in (
        lambda x: x.update(shard_size=4),
        lambda x: x["deltas"].pop("operation"),
        lambda x: x["deltas"]["click_target"].update(fp32_sha256="bad"),
        lambda x: x["deltas"]["click_target"].update(path="relative.npy"),
    ):
        value = candidate_plan(tmp_path)
        mutate(value)
        with pytest.raises(ValueError):
            owner._validate_plan(value)


def test_exact_delta_file_and_matrix_hashes_required(tmp_path):
    plan = candidate_plan(tmp_path)
    for head, ref in plan["deltas"].items():
        data = np.full((255, 1024), 1 if head == "operation" else 2, dtype="<f4")
        path = Path(ref["path"])
        np.save(path, data, allow_pickle=False)
        ref["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        ref["fp32_sha256"] = hashlib.sha256(data.tobytes()).hexdigest()
    torch = SimpleNamespace(from_numpy=lambda value: value)
    loaded = owner._load_deltas(plan, torch)
    assert set(loaded) == {"operation", "click_target"}
    assert loaded["operation"].shape == (255, 1024)
    plan["deltas"]["operation"]["fp32_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="FP32"):
        owner._load_deltas(plan, torch)
    plan["deltas"]["operation"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="delta file"):
        owner._load_deltas(plan, torch)


def test_completed_predict_is_bound_to_wire_and_witness(tmp_path, monkeypatch):
    import copy
    import json

    import jev_decision3_link_navigation_adapter as adapter
    import jev_decision3_link_navigation_adapter_audit as audit
    import jev_decision3_link_navigation_features as features

    import s1.evaluation.decision3 as decision3
    import s1.jev_decision3_native as native

    monkeypatch.setattr(owner, "SOURCE_ONLY", False)
    plan = candidate_plan(tmp_path)
    plan["cache_root"] = str(tmp_path)
    wire = {
        "model": features.ALIAS,
        "state": "synthetic",
        "questions": {
            "operation": {"type": "choice", "criteria": {"a": "A", "b": "B"}, "instructions": "x"},
            "other": {"criteria": {"c": "C", "d": "D"}, "type": "choice", "instructions": "x"},
        },
    }
    payload = {
        "state": wire["state"],
        "questions": {
            name: {"type": q["type"], "instructions": q["instructions"], "criteria": q["criteria"]}
            for name, q in wire["questions"].items()
        },
    }
    digest = features._sha(features._canonical(wire))
    shard = {
        "index": 0,
        "ordered_identity_sha256": "1" * 64,
        "receipt_sha256": "2" * 64,
        "plan_sha256": "3" * 64,
        "wires": [{"payload": wire, "wire_sha256": digest}],
    }
    monkeypatch.setattr(features, "feature_only_shards", lambda *_args, **_kwargs: [shard])
    monkeypatch.setattr(features, "_check_sources", lambda *_args: None)
    monkeypatch.setattr(features, "_validate_manifest", lambda *_args: None)
    monkeypatch.setattr(
        native, "parse_native_payload", lambda value, **_kwargs: (value, None, None, None)
    )
    monkeypatch.setattr(decision3, "native_payload", lambda _parsed: copy.deepcopy(payload))
    monkeypatch.setattr(owner, "_load_deltas", lambda *_args: {})
    monkeypatch.setattr(owner.inspect, "getfile", lambda _obj: __file__)
    backend = SimpleNamespace(model=SimpleNamespace(), runtime=SimpleNamespace())
    required = [adapter.__file__, adapter.reference.__file__, audit.__file__, __file__]
    plan["source_pins"] = {str(Path(p).absolute()): "0" * 64 for p in required}
    raw_response = {"answers": {"operation": "synthetic"}}
    receipts = []
    behavior = {"fail_after_witness": False, "wrong_wire": False, "reorder": None, "closed": 0}

    def writer(root, *, maximum_calls):
        assert maximum_calls == 1
        root.mkdir()

        def sink():
            folder = root / "call-0000"
            folder.mkdir()
            actual = copy.deepcopy(payload)
            if behavior["reorder"] == "heads":
                actual["questions"] = dict(reversed(list(actual["questions"].items())))
            if behavior["reorder"] == "criteria":
                criteria = actual["questions"]["operation"]["criteria"]
                actual["questions"]["operation"]["criteria"] = dict(
                    reversed(list(criteria.items()))
                )
            doc = {
                "metadata": {
                    "native_payload": {} if behavior["wrong_wire"] else actual,
                    "composed_raw": raw_response,
                },
                "witness_sha256": "5" * 64,
            }
            raw = json.dumps(doc, separators=(",", ":")).encode()
            (folder / "witness.json").write_bytes(raw)
            receipts.append(
                {
                    "directory": str(folder),
                    "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                    "witness_sha256": "5" * 64,
                    "delta_sha256": {k: v["fp32_sha256"] for k, v in plan["deltas"].items()},
                }
            )

        return sink, receipts

    class FakeAdapter:
        def __init__(self, provider, _deltas, _expected, _torch, *, identity_guard, witness_sink):
            assert identity_guard(provider, {})
            self.sink = witness_sink

        def predict(self, _parsed, *, deadline):
            deadline()
            self.sink()
            if behavior["fail_after_witness"]:
                raise RuntimeError("deadline after witness")
            return {
                "decision3_provider_evidence": {
                    "native_payload": payload,
                    "raw_response": raw_response,
                }
            }

        def close(self):
            behavior["closed"] += 1

    monkeypatch.setattr(audit, "make_witness_writer", writer)
    monkeypatch.setattr(adapter, "DualHeadDecision3Backend", FakeAdapter)
    output = tmp_path / "ok"
    output.mkdir()
    result = owner._capture_candidate(plan, object(), backend, object(), output, lambda: None)
    assert result["completed"][0]["wire_sha256"] == digest
    assert len(result["completed"][0]["response_sha256"]) == 64
    assert len(result["completed"][0]["raw_response_sha256"]) == 64
    owner._verify_candidate_result({"capture": result}, shard, plan, output)
    response_path = output / result["completed"][0]["response_file"]
    original_response = response_path.read_bytes()
    response_path.write_bytes(original_response + b" ")
    with pytest.raises(ValueError, match="response file changed"):
        owner._verify_candidate_result({"capture": result}, shard, plan, output)
    response_path.write_bytes(original_response)
    assert result["model_score"] is None and result["fit_eligible"] is False
    assert behavior["closed"] == 1
    receipts.clear()
    behavior["fail_after_witness"] = True
    output = tmp_path / "failure"
    output.mkdir()
    with pytest.raises(RuntimeError, match="after witness"):
        owner._capture_candidate(plan, object(), backend, object(), output, lambda: None)
    assert len(receipts) == 1 and behavior["closed"] == 2
    receipts.clear()
    behavior.update(fail_after_witness=False, wrong_wire=True)
    output = tmp_path / "mismatch"
    output.mkdir()
    with pytest.raises(ValueError, match="same-request witness"):
        owner._capture_candidate(plan, object(), backend, object(), output, lambda: None)
    assert behavior["closed"] == 3
    for reordered in ("heads", "criteria"):
        receipts.clear()
        behavior.update(wrong_wire=False, reorder=reordered)
        output = tmp_path / reordered
        output.mkdir()
        with pytest.raises(ValueError, match="same-request witness"):
            owner._capture_candidate(plan, object(), backend, object(), output, lambda: None)
    plan["source_pins"].pop(str(Path(adapter.reference.__file__).absolute()))
    with pytest.raises(ValueError, match="actual source pins"):
        owner._capture_candidate(
            plan, object(), backend, object(), tmp_path / "unused", lambda: None
        )


def test_real_native_dto_preserves_state_heads_and_criteria_not_field_order():
    import copy

    import jev_decision3_link_navigation_features as features

    from s1.evaluation.decision3 import native_payload
    from s1.jev_decision3_native import parse_native_payload

    wire = {
        "model": features.ALIAS,
        "state": {"a": 1, "b": [2, 3]},
        "questions": {
            "operation": {
                "criteria": {"CLICK": "A", "DONE": "B"},
                "type": "choice",
                "instructions": "x",
            },
            "click_target": {
                "type": "choice",
                "criteria": {"a": "C", "b": "D"},
                "instructions": "y",
            },
        },
    }
    parsed, _, _, _ = parse_native_payload(wire, model_identity=features.ALIAS)
    native = native_payload(parsed)
    original = {"state": wire["state"], "questions": wire["questions"]}
    assert list(wire["questions"]["operation"]) != list(native["questions"]["operation"])
    assert owner._same_native(native, original)
    modified = copy.deepcopy(original)
    modified["questions"] = dict(reversed(list(modified["questions"].items())))
    assert not owner._same_native(native, modified)
    modified = copy.deepcopy(original)
    criteria = modified["questions"]["operation"]["criteria"]
    modified["questions"]["operation"]["criteria"] = dict(reversed(list(criteria.items())))
    assert not owner._same_native(native, modified)
    modified = copy.deepcopy(original)
    modified["state"] = dict(reversed(list(modified["state"].items())))
    assert not owner._same_native(native, modified)
    modified = copy.deepcopy(original)
    modified["questions"]["operation"]["instructions"] = "changed"
    assert not owner._same_native(native, modified)


def test_drop_exception_frames_releases_owned_model_and_preserves_primary():
    import gc
    import weakref

    class Model:
        pass

    def create_error():
        model = Model()
        reference = weakref.ref(model)

        def fail():
            assert model is not None
            try:
                raise ValueError("cause")
            except ValueError as cause:
                raise RuntimeError("primary") from cause

        try:
            fail()
        except RuntimeError as error:
            return error, reference

    error, reference = create_error()
    assert reference() is not None
    assert error.__cause__.__traceback__ is not None
    stack = owner._drop_exception_frames(error)
    gc.collect()
    assert reference() is None
    assert type(error) is RuntimeError and error.args == ("primary",)
    assert error.__traceback__ is None and error.__cause__.__traceback__ is None
    assert any(frame["function"] == "fail" for frame in stack)


def test_child_error_releases_model_before_cuda_cleanup(tmp_path, monkeypatch):
    import json
    import weakref

    import jev_decision3_link_navigation_features as features
    import jev_decision3_link_navigation_pipeline as pipeline

    import s1.evaluation.decision3 as decision3

    monkeypatch.setattr(owner, "SOURCE_ONLY", False)
    monkeypatch.setenv("INVOCATION_ID", "a" * 32)
    plan = {
        "output_dir": str(tmp_path),
        "cache_root": str(tmp_path),
        "modules_cache": str(tmp_path),
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
        "source_pins": {},
        "collection_dir": str(tmp_path),
        "receipt_sha256": "b" * 64,
        "collection_plan_sha256": "c" * 64,
    }
    plan_sha = "d" * 64
    (tmp_path / "go.json").write_text(json.dumps({"invocation": "a" * 32, "plan_sha256": plan_sha}))
    monkeypatch.setattr(owner, "_load_plan", lambda *_args: plan)
    monkeypatch.setattr(owner, "_own_cgroup", lambda *_args: None)
    monkeypatch.setattr(owner, "_bind_project_source", lambda *_args: None)
    monkeypatch.setattr(owner, "_verify_source_pins", lambda *_args: None)
    monkeypatch.setattr(owner.systemd, "process_identity", lambda *_args: {"startticks": 1})
    monkeypatch.setattr(owner.systemd, "boot_id", lambda: "boot")
    monkeypatch.setattr(features, "_check_sources", lambda *_args: None)
    monkeypatch.setattr(pipeline, "admit_training_collection", lambda *_args, **_kwargs: object())
    refs = []

    class Model:
        pass

    class Provider:
        def __init__(self):
            self.model = Model()
            refs.append(weakref.ref(self.model))

        def close(self):
            pass

    def fail_after_model_load(_plan, _admitted, backend, _torch, _output, _check):
        assert backend.model is not None
        raise RuntimeError("original forward failure")

    monkeypatch.setattr(decision3, "Decision3Backend", lambda *_args: Provider())
    monkeypatch.setattr(decision3, "decision3_spec", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(owner, "_capture_candidate", fail_after_model_load)
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda _index: SimpleNamespace(
            uuid=plan["gpu_uuid"], major=12, minor=1, name="synthetic"
        ),
        is_bf16_supported=lambda **_kwargs: True,
        synchronize=lambda *_args: None,
        memory_allocated=lambda _index: int(bool(refs and refs[0]() is not None)),
        empty_cache=lambda: None,
    )
    torch = SimpleNamespace(
        set_num_threads=lambda *_args: None,
        set_num_interop_threads=lambda *_args: None,
        set_float32_matmul_precision=lambda *_args: None,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))),
        cuda=cuda,
        version=SimpleNamespace(cuda="synthetic"),
        _C=SimpleNamespace(),
    )
    monkeypatch.setattr(owner, "_bind_overlay", lambda *_args: torch)
    with pytest.raises(RuntimeError, match="original forward failure"):
        owner._child(tmp_path / "plan", plan_sha, "synthetic.service")
    status = json.loads((tmp_path / "worker-result.json").read_text())
    assert status["status"] == "failed_no_accepted_features"
    assert status["error_type"] == "RuntimeError"
    assert status["final_cuda_allocated"] == 0
    assert refs[0]() is None
    assert any(frame["function"] == "fail_after_model_load" for frame in status["error_stack"])
