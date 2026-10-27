"""Synthetic admission/planning checks; no model, browser or GPU is used."""

import copy

import jev_decision3_link_navigation_features as features
import jev_decision3_link_navigation_pipeline as pipeline
import jev_decision3_text_history_collection as base
import pytest


def admitted_rows(count=3):
    blobs = []
    for index in range(count):
        payload = {
            "model": features.ALIAS,
            "state": {"page": index},
            "questions": {
                "operation": {
                    "type": "choice",
                    "instructions": "Choose action",
                    "criteria": {"CLICK": "click", "DONE": "done"},
                },
                "click_target": {
                    "type": "choice",
                    "instructions": "Choose target",
                    "criteria": {"t2": "other", "t1": "target"},
                },
            },
        }
        row = {
            "schema_id": base.ROW_SCHEMA,
            "split": "train",
            "id": f"case-{index}",
            "payload": payload,
            "teacher_choice": {"operation": "CLICK", "target": "t1"},
            "valid_next_targets": {"CLICK": ["t1"]},
            "loss_mask": {"operation": True, "click_target": True},
            "wire_sha256": base._sha(base._encode(payload)),
            "fit_eligible": False,
        }
        blobs.append(base._encode(row))
    receipt, plan = "a" * 64, "b" * 64
    seal = base._digest(
        {"rows": [blob.decode() for blob in blobs], "receipt": receipt, "plan": plan}
    )
    return pipeline.AdmittedLinkCollection(
        tuple(blobs), receipt, plan, seal, pipeline._ADMISSION_PROOF
    )


def test_only_verified_admission_can_be_planned():
    with pytest.raises(ValueError, match="admission"):
        features.feature_only_shards(object(), shard_size=2)
    admitted = admitted_rows()
    forged = pipeline.AdmittedLinkCollection(
        admitted._rows, admitted.receipt_sha256, admitted.plan_sha256, admitted._seal, object()
    )
    with pytest.raises(ValueError, match="admission"):
        features.feature_only_shards(forged, shard_size=2)
    changed = pipeline.AdmittedLinkCollection(
        admitted._rows[:-1],
        admitted.receipt_sha256,
        admitted.plan_sha256,
        admitted._seal,
        pipeline._ADMISSION_PROOF,
    )
    with pytest.raises(ValueError, match="changed"):
        features.feature_only_shards(changed, shard_size=2)


def test_shards_strip_supervision_preserve_wire_and_positions():
    admitted = admitted_rows()
    original = pipeline.compile_wire_projection(admitted)
    shards = features.feature_only_shards(admitted, shard_size=2)
    assert [(s["start"], s["stop"], s["total"]) for s in shards] == [(0, 2, 3), (2, 3, 3)]
    assert shards[0]["ordered_identity_sha256"] == shards[1]["ordered_identity_sha256"]
    wires = [item for shard in shards for item in shard["wires"]]
    assert [item["case_id"] for item in wires] == [row["case_id"] for row in original]
    assert all(
        set(item) == {"global_ordinal", "case_id", "wire_sha256", "payload"} for item in wires
    )
    assert [item["payload"] for item in wires] == [row["payload"] for row in original]
    assert list(wires[0]["payload"]["questions"]) == ["operation", "click_target"]
    assert list(wires[0]["payload"]["questions"]["click_target"]["criteria"]) == ["t2", "t1"]
    assert all(
        shard["fit_eligible"] is False and shard["status"] == "planned_not_captured"
        for shard in shards
    )
    wires[0]["payload"]["state"]["page"] = "modified"
    assert pipeline.compile_wire_projection(admitted) == original


def test_invalid_shards_and_modified_projection_fail_closed(monkeypatch):
    admitted = admitted_rows()
    for value in (0, 41, True, 2.5):
        with pytest.raises(ValueError):
            features.feature_only_shards(admitted, shard_size=value)
    original = pipeline.compile_wire_projection(admitted)
    tampered = copy.deepcopy(original)
    tampered[0]["payload"]["state"] = "changed"
    monkeypatch.setattr(pipeline, "compile_wire_projection", lambda _admitted: tampered)
    with pytest.raises(ValueError, match="digest"):
        features.feature_only_shards(admitted, shard_size=2)


def test_no_gpu_invocation_for_unadmitted_input():
    class NoGpu:
        @property
        def runtime(self):
            pytest.fail("provider must not be inspected before admission")

    with pytest.raises(ValueError, match="admission"):
        features.capture_shard(
            object(),
            shard_size=2,
            shard_index=0,
            backend=NoGpu(),
            torch=None,
            source_pins={},
            cache_root=None,
            guard=lambda: None,
            deadline=lambda: None,
        )


def synthetic_prepared():
    from types import SimpleNamespace

    questions = {
        "operation": {"criteria": {"CLICK": "click", "DONE": "done"}},
        "click_target": {"criteria": {"t2": "other", "t1": "target"}},
    }
    prepared = SimpleNamespace(
        errors=[],
        keys=list(questions),
        runnable=list(questions),
        sequences={"operation": [1, 2, 3], "click_target": [4, 5, 6, 7, 8]},
        lengths={"operation": 3, "click_target": 5},
        questions={
            key: SimpleNamespace(keys=list(value["criteria"])) for key, value in questions.items()
        },
        images=[],
        videos=[],
        media=[],
        reversed_sequences=[],
    )
    return prepared, questions


def test_preparation_binds_head_criteria_and_original_sequence_lengths():
    prepared, questions = synthetic_prepared()
    assert features._prepared_contract(prepared, questions) == {"operation": 3, "click_target": 5}
    prepared.errors = {}
    assert features._prepared_contract(prepared, questions) == {"operation": 3, "click_target": 5}
    for change in ("runnable", "keys", "sequences", "lengths", "criteria", "images"):
        prepared, questions = synthetic_prepared()
        if change in ("runnable", "keys"):
            getattr(prepared, change).reverse()
        elif change == "sequences":
            prepared.sequences = dict(reversed(list(prepared.sequences.items())))
        elif change == "lengths":
            prepared.lengths["operation"] = 4
        elif change == "criteria":
            prepared.questions["click_target"].keys.reverse()
        else:
            prepared.images = ["unexpected"]
        with pytest.raises(ValueError):
            features._prepared_contract(prepared, questions)


def synthetic_call():
    native = {"state": "synthetic", "questions": {"operation": {}, "click_target": {}}}
    lengths = {"operation": 3, "click_target": 5}
    preflight = {"lengths": lengths, "errors": [], "project_context_limit": 16384}
    raw = {"model": "d3-edge", "usage": {"input_tokens": 8, "output_tokens": 0}, "answers": {}}
    events = [
        {
            "native_payload": copy.deepcopy(native),
            "preflight": copy.deepcopy(preflight),
            "raw_response": None,
        },
        {
            "native_payload": copy.deepcopy(native),
            "preflight": copy.deepcopy(preflight),
            "raw_response": raw,
        },
    ]
    return native, lengths, events, {"decision3_provider_evidence": events[1]}


def test_provider_events_shapes_and_usage_are_bound():
    native, lengths, events, result = synthetic_call()
    assert (
        features._verify_call_contract(events, native, lengths, (2, 5, 1024), (2, 5, 1024), result)
        is events[1]["raw_response"]
    )
    events[0]["preflight"]["errors"] = {}
    events[1]["preflight"]["errors"] = {}
    features._verify_call_contract(events, native, lengths, (2, 5, 1024), (2, 5, 1024), result)
    for first, norm in (((2, 4, 1024), (2, 5, 1024)), ((2, 5, 1024), (2, 4, 1024))):
        with pytest.raises(ValueError, match="widths"):
            features._verify_call_contract(events, native, lengths, first, norm, result)
    for index in (0, 1):
        native, lengths, events, result = synthetic_call()
        events[index]["preflight"]["lengths"]["operation"] = 4
        with pytest.raises(ValueError, match="preparation"):
            features._verify_call_contract(
                events, native, lengths, (2, 5, 1024), (2, 5, 1024), result
            )
    for bad_usage in (
        {"input_tokens": 7, "output_tokens": 0},
        {"input_tokens": 8, "output_tokens": 1},
        {"input_tokens": True, "output_tokens": 0},
    ):
        native, lengths, events, result = synthetic_call()
        events[1]["raw_response"]["usage"] = bad_usage
        with pytest.raises(ValueError, match="usage"):
            features._verify_call_contract(
                events, native, lengths, (2, 5, 1024), (2, 5, 1024), result
            )


def test_spec_pins_and_cleanup_preserves_predict_error():
    from types import SimpleNamespace

    spec = SimpleNamespace(
        adapter="decision3",
        model_id=features.MODEL_ID,
        revision=features.REVISION,
        device="cuda:0",
        precision="bfloat16",
        context_limit=16384,
    )
    features._validate_provider_spec(spec)
    for key, wrong in (
        ("adapter", "other"),
        ("model_id", "other"),
        ("revision", "other"),
        ("device", "cpu"),
        ("precision", "float32"),
    ):
        bad = copy.copy(spec)
        setattr(bad, key, wrong)
        with pytest.raises(ValueError, match="spec"):
            features._validate_provider_spec(bad)
    calls = []

    class Handle:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        def remove(self):
            calls.append("remove-" + self.name)
            if self.fail:
                raise RuntimeError("remove failure")

    class Hook:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        def register_forward_pre_hook(self, _callback):
            return Handle(self.name, self.fail)

        def register_forward_hook(self, _callback):
            return Handle(self.name, self.fail)

    def fence(_device):
        calls.append("fence")
        raise RuntimeError("fence failure")

    language = SimpleNamespace(layers=[Hook("first")], norm=Hook("norm", True))
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=fence))
    primary = RuntimeError("predict failure")

    def predict():
        raise primary

    with pytest.raises(RuntimeError, match="predict failure") as caught:
        features._call_with_hooks(language, fake_torch, None, None, predict)
    assert caught.value is primary
    assert calls == ["fence", "remove-norm", "remove-first"]
    assert len(primary.__notes__) == 2


def test_owner_source_pins_are_verified(tmp_path):
    source = tmp_path / "provider.py"
    source.write_bytes(b"fixed source")
    pins = {str(source): features._sha(b"fixed source")}
    features._check_sources(pins)
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source pin"):
        features._check_sources(pins)


def test_admitted_capture_preserves_wire_with_synthetic_provider(monkeypatch):
    """Exercise the full capture path using CPU-only fakes, not a GPU proof."""
    import inspect
    from contextlib import nullcontext
    from types import SimpleNamespace

    import numpy as np

    from s1.evaluation.decision3 import native_payload
    from s1.jev_decision3_native import parse_native_payload

    class Tensor:
        def __init__(self, values, *, dtype="float32", device="cuda:0"):
            self.values = np.asarray(values)
            self.dtype, self.device = dtype, device

        @property
        def shape(self):
            return self.values.shape

        @property
        def T(self):
            return Tensor(self.values.T, dtype=self.dtype, device=self.device)

        def numel(self):
            return self.values.size

        def detach(self):
            return self

        def clone(self):
            return Tensor(self.values.copy(), dtype=self.dtype, device=self.device)

        def contiguous(self):
            return self

        def to(self, *, device=None, dtype=None):
            return Tensor(self.values, dtype=dtype or self.dtype, device=device or self.device)

        def cpu(self):
            return self.to(device="cpu")

        def numpy(self):
            return self.values

        def tolist(self):
            return self.values.tolist()

        def all(self):
            return Tensor(np.array(self.values.all()))

        def item(self):
            return self.values.item()

        def masked_fill(self, mask, value):
            return Tensor(np.where(mask.values, value, self.values), device=self.device)

        def __getitem__(self, key):
            return Tensor(self.values[key], dtype=self.dtype, device=self.device)

        def __matmul__(self, other):
            return Tensor(self.values @ other.values, device=self.device)

        def __ge__(self, other):
            return Tensor(self.values >= other.values, device=self.device)

    class Torch:
        float32, bfloat16 = "float32", "bfloat16"
        backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)))
        cuda = SimpleNamespace(synchronize=lambda _device: None)

        @staticmethod
        def get_float32_matmul_precision():
            return "highest"

        @staticmethod
        def is_autocast_enabled(_device):
            return False

        @staticmethod
        def isfinite(tensor):
            return Tensor(np.isfinite(tensor.values))

        @staticmethod
        def inference_mode():
            return nullcontext()

        @staticmethod
        def arange(count, *, device):
            return Tensor(np.arange(count), device=device)

        @staticmethod
        def tensor(values, *, device):
            return Tensor(np.array(values), device=device)

        @staticmethod
        def softmax(tensor, *, dim):
            values = tensor.values - tensor.values.max(axis=dim, keepdims=True)
            exponentials = np.exp(values)
            return Tensor(exponentials / exponentials.sum(axis=dim, keepdims=True))

    class Hook:
        def __init__(self):
            self.callback = None

        def install(self, callback):
            assert self.callback is None
            self.callback = callback
            return SimpleNamespace(remove=self.remove)

        def remove(self):
            self.callback = None

        register_forward_pre_hook = install
        register_forward_hook = install

    class Parameter:
        dtype, device, requires_grad = Torch.bfloat16, "cuda:0", False

        @staticmethod
        def numel():
            return 594582720 - 255 * 1024

    class Runtime:
        temperature, batch_size, max_length = 1.0, 8, None
        attention_mode, permutation_average, fast = "noncausal_full_attention", False, None
        readout_dtype, model_name = "float32", "d3-edge"

        def __init__(self):
            self.readout = Tensor(np.zeros((255, 1024), dtype=np.float32))
            self.language = SimpleNamespace(layers=[Hook()], norm=Hook())
            self.backbone = SimpleNamespace(
                language_model=self.language, training=False, parameters=lambda: [Parameter()]
            )
            self.encoder = SimpleNamespace(language_model=self.language)

        @staticmethod
        def prepare(*, state, questions):
            assert state == {"page": 0}
            heads = list(questions)
            return SimpleNamespace(
                errors=[],
                keys=heads,
                runnable=heads,
                sequences={key: [1, 2, 3] for key in heads},
                lengths={key: 3 for key in heads},
                questions={
                    key: SimpleNamespace(keys=list(questions[key]["criteria"])) for key in heads
                },
                images=[],
                videos=[],
                media=[],
                reversed_sequences=[],
            )

    class Wrapper:
        def __init__(self, runtime):
            self.runtime = runtime

    class Provider:
        def __init__(self):
            self.runtime = Runtime()
            self.model = Wrapper(self.runtime)
            self.identity = {"model_id": features.MODEL_ID}
            self.spec = SimpleNamespace(
                adapter="decision3",
                model_id=features.MODEL_ID,
                revision=features.REVISION,
                device="cuda:0",
                precision="bfloat16",
                context_limit=16384,
            )
            self.received = []

        def predict(self, parsed, *, deadline, evidence_sink):
            deadline()
            payload = native_payload(parsed)
            assert set(payload) == {"state", "questions"}
            assert "teacher_choice" not in repr(payload) and "loss_mask" not in repr(payload)
            self.received.append(copy.deepcopy(payload))
            lengths = {head: 3 for head in payload["questions"]}
            preflight = {"lengths": lengths, "errors": [], "project_context_limit": 16384}
            evidence = {"native_payload": payload, "preflight": preflight, "raw_response": None}
            evidence_sink("native_preflight", evidence)
            data = Tensor(np.ones((2, 3, 1024), dtype=np.float32), dtype=Torch.bfloat16)
            self.runtime.language.layers[0].callback(None, (data,))
            self.runtime.language.norm.callback(None, (), data)
            answers = {}
            for head, question in payload["questions"].items():
                labels = list(question["criteria"])
                answers[head] = {
                    "type": "choice",
                    "choice": labels[0],
                    "probabilities": {label: 1 / len(labels) for label in labels},
                }
            evidence["raw_response"] = {
                "model": "d3-edge",
                "answers": answers,
                "usage": {"input_tokens": 6, "output_tokens": 0},
            }
            evidence_sink("native_raw_returned", evidence)
            return {"decision3_provider_evidence": evidence}

    provider = Provider()
    pins = {
        inspect.getfile(obj): "f" * 64
        for obj in (
            type(provider),
            type(provider.runtime),
            type(provider.model),
            parse_native_payload,
            native_payload,
        )
    }
    monkeypatch.setattr(features, "_check_sources", lambda _pins: None)
    monkeypatch.setattr(features, "_validate_manifest", lambda _runtime, _root: None)
    monkeypatch.setattr(features, "W0_SHA256", features._tensor_sha(provider.runtime.readout))
    admitted = admitted_rows(1)
    output = features.capture_shard(
        admitted,
        shard_size=1,
        shard_index=0,
        backend=provider,
        torch=Torch,
        source_pins=pins,
        cache_root="unused",
        guard=lambda: None,
        deadline=lambda: None,
    )
    assert len(provider.received) == 1
    assert list(provider.received[0]["questions"]) == ["operation", "click_target"]
    assert list(provider.received[0]["questions"]["click_target"]["criteria"]) == ["t2", "t1"]
    case = output["cases"][0]
    assert case["wire_sha256"] == pipeline.compile_wire_projection(admitted)[0]["wire_sha256"]
    assert case["head_order"] == ["operation", "click_target"]
    assert case["criteria_order"] == [["CLICK", "DONE"], ["t2", "t1"]]
    assert case["features"].shape == (2, 1024)
    assert output["status"] == "captured_pending_independent_fp64_audit"
    assert output["fit_eligible"] is False
    assert provider.runtime.language.layers[0].callback is None
    assert provider.runtime.language.norm.callback is None
