"""Pinned 52-row sidecars use real CPU tensors, with no downloaded model weights."""

import copy
import hashlib
import io
import json
from types import SimpleNamespace

import pytest

from s1.browser_readout import load_readout, write_readout

torch = pytest.importorskip("torch")

VARIANT = "grounded_user_question_v1"


@pytest.fixture
def model():
    # A full vocabulary head rather than a preselected layer; its original
    # noncontiguous token rows and bias must survive sidecar construction.
    head = torch.nn.Linear(4, 120)
    with torch.no_grad():
        head.weight.copy_(torch.arange(480).reshape(120, 4).float() / 1000)
        head.bias.copy_(torch.arange(120).float() / 100)
    return SimpleNamespace(
        name="test/pinned-browser-model",
        revision="a" * 40,
        quantization="nf4",
        temperature=2.7830344470383452,
        softcap=3.0,
        head=head,
        letters=torch.arange(1, 105, 2),
    )


def provenance(*, learned=False):
    empty = {"ids": [], "groups": [], "request_fingerprints": []}
    return {
        "source_scope": "development_only",
        "train": (
            {
                "ids": ["dev-row-1"],
                "groups": ["dev-case-1"],
                "request_fingerprints": ["1" * 64],
            }
            if learned
            else copy.deepcopy(empty)
        ),
        "calibration": copy.deepcopy(empty),
        "training_objective": {"name": "cross_entropy", "temperature": 1.0},
    }


def artifact(tmp_path, model, *, learned=False):
    destination = tmp_path / "readout"
    kwargs = {}
    if learned:
        kwargs = {"weight": torch.zeros(52, 4), "bias": torch.arange(52).float()}
    write_readout(destination, model, VARIANT, provenance=provenance(learned=learned), **kwargs)
    return destination


def mutate_manifest(path, mutation):
    filename = path / "manifest.json"
    value = json.loads(filename.read_text())
    mutation(value)
    filename.write_text(json.dumps(value))


def mutate_weights(path, mutation):
    filename = path / "weights.pt"
    data = torch.load(filename, map_location="cpu", weights_only=True)
    mutation(data)
    stream = io.BytesIO()
    torch.save(data, stream)
    payload = stream.getvalue()
    filename.write_bytes(payload)
    mutate_manifest(
        path, lambda manifest: manifest.update(weights_sha256=hashlib.sha256(payload).hexdigest())
    )


def test_untrained_fp32_control_clones_original_token_rows_and_bias_without_model_mutation(
    tmp_path, model
):
    original_weight, original_bias = model.head.weight.clone(), model.head.bias.clone()
    head_id, weight_pointer = id(model.head), model.head.weight.data_ptr()
    path = artifact(tmp_path, model)
    readout = load_readout(path, model, VARIANT)
    hidden = torch.tensor([[0.25, -0.5, 0.1, 0.4]])
    logits = readout.project(hidden, torch.tensor([52]), model.softcap)
    expected = torch.nn.functional.linear(
        hidden.float(), original_weight[model.letters].float(), original_bias[model.letters].float()
    )
    expected = torch.tanh(expected / model.softcap) * model.softcap
    torch.testing.assert_close(logits, expected, atol=0, rtol=0)
    assert logits.dtype == torch.float32
    assert id(model.head) == head_id and model.head.weight.data_ptr() == weight_pointer
    torch.testing.assert_close(model.head.weight, original_weight, atol=0, rtol=0)
    torch.testing.assert_close(model.head.bias, original_bias, atol=0, rtol=0)
    assert readout.telemetry()["kind"] == "untrained_fp32_selected_rows_control"
    assert readout.telemetry()["candidate_ids"] == model.letters.tolist()
    assert readout.temperature == model.temperature


def test_control_supports_biasless_full_head_and_fp32_conversion(tmp_path, model):
    model.head = torch.nn.Linear(4, 120, bias=False, dtype=torch.bfloat16)
    path = artifact(tmp_path, model)
    rows = torch.load(path / "weights.pt", weights_only=True)
    assert rows["weight"].dtype == rows["bias"].dtype == torch.float32
    assert torch.equal(rows["bias"], torch.zeros(52))
    torch.testing.assert_close(
        rows["weight"], model.head.weight[model.letters].float(), atol=0, rtol=0
    )


def test_learned_rows_project_supplied_weights_with_fixed_deployment_temperature(tmp_path, model):
    path = artifact(tmp_path, model, learned=True)
    readout = load_readout(path, model, VARIANT)
    hidden = torch.ones(1, 4)
    logits = readout.project(hidden, torch.tensor([52]), model.softcap)
    expected = torch.tanh(torch.arange(52).float() / model.softcap) * model.softcap
    torch.testing.assert_close(logits[0], expected, atol=0, rtol=0)
    info = readout.telemetry()
    assert info["kind"] == "learned_rows"
    assert info["temperature"] == info["base_temperature"] == model.temperature
    assert info["provenance"]["training_objective"]["temperature"] == 1.0
    assert info["provenance"]["calibration"]["ids"] == []


def test_all_52_candidate_counts_are_masked_exactly_and_outer_autocast_cannot_change_fp32(
    tmp_path, model
):
    readout = load_readout(artifact(tmp_path, model), model, VARIANT)
    hidden = torch.ones(52, 4, dtype=torch.bfloat16)
    counts = torch.arange(1, 53)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = readout.project(hidden, counts, model.softcap)
    assert logits.dtype == torch.float32 and logits.shape == (52, 52)
    for index, count in enumerate(counts):
        assert torch.isfinite(logits[index, :count]).all()
        assert torch.isneginf(logits[index, count:]).all()
        probabilities = (logits[index] / readout.temperature).softmax(0)
        assert probabilities[count:].eq(0).all()
        assert probabilities.sum().item() == pytest.approx(1)


@pytest.mark.parametrize(
    ("key", "wrong"),
    [
        ("base_model_id", "different/model"),
        ("base_revision", "b" * 40),
        ("quantization", "none"),
        ("prompt_variant", "goal_first_user_question_v2"),
        ("candidate_ids", list(range(52))),
        ("candidate_ids", [True] + list(range(3, 105, 2))),
        ("temperature", 1.0),
        ("base_temperature", 1.0),
        ("softcap", None),
        ("hidden_size", 5),
    ],
)
def test_loader_rejects_each_model_or_prompt_pin_mismatch_before_loading_weights(
    tmp_path, model, monkeypatch, key, wrong
):
    path = artifact(tmp_path, model)
    mutate_manifest(path, lambda manifest: manifest.update({key: wrong}))
    monkeypatch.setattr(
        torch, "load", lambda *args, **kwargs: pytest.fail("weights loaded before pins")
    )
    with pytest.raises(ValueError):
        load_readout(path, model, VARIANT)


def test_loader_verifies_exact_bytes_before_weights_only_load(tmp_path, model, monkeypatch):
    path = artifact(tmp_path, model)
    actual_load, calls = torch.load, []

    def safe_load(stream, **kwargs):
        calls.append(kwargs)
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        data = stream.getvalue()
        manifest = json.loads((path / "manifest.json").read_text())
        assert hashlib.sha256(data).hexdigest() == manifest["weights_sha256"]
        return actual_load(stream, **kwargs)

    monkeypatch.setattr(torch, "load", safe_load)
    load_readout(path, model, VARIANT)
    assert len(calls) == 1
    (path / "weights.pt").write_bytes((path / "weights.pt").read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA256"):
        load_readout(path, model, VARIANT)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update(weight=torch.zeros(51, 4)),
        lambda data: data.update(weight=torch.zeros(52, 5)),
        lambda data: data.update(bias=torch.zeros(51)),
        lambda data: data.update(weight=torch.full((52, 4), float("nan"))),
        lambda data: data.update(bias=torch.full((52,), float("inf"))),
        lambda data: data.update(weight=torch.zeros(52, 4, dtype=torch.float64)),
        lambda data: data.update(bias=torch.zeros(52, dtype=torch.int64)),
        lambda data: data.update(gold_labels={"fixture": "CLICK"}),
    ],
)
def test_hashed_artifact_still_rejects_bad_shapes_nonfinite_dtype_or_extra_payload(
    tmp_path, model, mutation
):
    path = artifact(tmp_path, model)
    mutate_weights(path, mutation)
    with pytest.raises(ValueError):
        load_readout(path, model, VARIANT)


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value.update(source_scope="holdout"),
        lambda value: value.update(gold_labels={"fixture": "CLICK"}),
        lambda value: value["train"].update(request_fingerprints=["not-a-hash"]),
        lambda value: value["train"].update(ids=["repeated", "repeated"]),
        lambda value: value["train"].update(groups=[]),
        lambda value: value["calibration"].update(copy.deepcopy(value["train"])),
        lambda value: value.update(training_objective={"loss": float("nan")}),
    ],
)
def test_writer_rejects_invalid_or_leaking_provenance_before_creating_artifact(
    tmp_path, model, change
):
    value = provenance(learned=True)
    change(value)
    destination = tmp_path / "bad"
    with pytest.raises(ValueError):
        write_readout(
            destination,
            model,
            VARIANT,
            weight=torch.zeros(52, 4),
            bias=torch.zeros(52),
            provenance=value,
        )
    assert not destination.exists()


def test_predictions_cannot_depend_on_provenance_ids_or_fingerprints_and_telemetry_is_a_copy(
    tmp_path, model
):
    path = artifact(tmp_path, model, learned=True)
    before = load_readout(path, model, VARIANT)
    mutate_manifest(
        path,
        lambda manifest: manifest["provenance"]["train"].update(
            ids=["a-completely-different-development-row"],
            groups=["another-development-case"],
            request_fingerprints=["f" * 64],
        ),
    )
    after = load_readout(path, model, VARIANT)
    inputs = (torch.arange(8).reshape(2, 4).float(), torch.tensor([2, 52]), model.softcap)
    torch.testing.assert_close(before.project(*inputs), after.project(*inputs), atol=0, rtol=0)
    changed = before.telemetry()
    changed["candidate_ids"][0] = -1
    changed["provenance"]["train"]["ids"].append("fake")
    assert before.telemetry()["candidate_ids"][0] == 1
    assert before.telemetry()["provenance"]["train"]["ids"] == ["dev-row-1"]


@pytest.mark.parametrize(
    "counts",
    [
        torch.tensor([0]),
        torch.tensor([53]),
        torch.tensor([2.0]),
        torch.tensor([True]),
        torch.tensor([[2]]),
        torch.tensor([1, 2]),
    ],
)
def test_projection_rejects_invalid_counts(tmp_path, model, counts):
    readout = load_readout(artifact(tmp_path, model), model, VARIANT)
    with pytest.raises(ValueError, match="counts"):
        readout.project(torch.zeros(1, 4), counts, model.softcap)


@pytest.mark.parametrize("softcap", [0, -1, float("inf"), float("nan"), True, 4.0, None])
def test_projection_rejects_invalid_or_changed_softcap(tmp_path, model, softcap):
    readout = load_readout(artifact(tmp_path, model), model, VARIANT)
    with pytest.raises(ValueError):
        readout.project(torch.zeros(1, 4), torch.tensor([2]), softcap)


@pytest.mark.parametrize(
    "hidden",
    [
        torch.zeros(1, 5),
        torch.zeros(4),
        torch.zeros(0, 4),
        torch.zeros(1, 4, dtype=torch.int64),
        torch.full((1, 4), float("nan")),
        torch.full((1, 4), 1e100, dtype=torch.float64),
    ],
)
def test_projection_rejects_bad_hidden_states(tmp_path, model, hidden):
    readout = load_readout(artifact(tmp_path, model), model, VARIANT)
    with pytest.raises(ValueError, match="hidden"):
        readout.project(hidden, torch.tensor([2]), model.softcap)


def test_writer_rejects_temperature_refit_and_existing_artifact(tmp_path, model):
    with pytest.raises(ValueError, match="temperature"):
        write_readout(
            tmp_path / "changed", model, VARIANT, provenance=provenance(), temperature=1.0
        )
    path = artifact(tmp_path, model)
    original = (path / "weights.pt").read_bytes()
    with pytest.raises(FileExistsError):
        write_readout(path, model, VARIANT, provenance=provenance())
    assert (path / "weights.pt").read_bytes() == original


@pytest.mark.parametrize("key", ["ids", "groups", "request_fingerprints"])
def test_each_provenance_inventory_is_independently_disjoint(tmp_path, model, key):
    value = provenance(learned=True)
    value["calibration"] = {
        "ids": ["calibration-row-2"],
        "groups": ["calibration-case-2"],
        "request_fingerprints": ["2" * 64],
    }
    value["calibration"][key] = value["train"][key].copy()
    with pytest.raises(ValueError, match=f"{key} must be disjoint"):
        write_readout(
            tmp_path / "overlap",
            model,
            VARIANT,
            weight=torch.zeros(52, 4),
            bias=torch.zeros(52),
            provenance=value,
        )


def test_valid_disjoint_calibration_inventory_is_pinned_without_temperature_refit(tmp_path, model):
    value = provenance(learned=True)
    value["calibration"] = {
        "ids": ["calibration-row-2"],
        "groups": ["calibration-case-2"],
        "request_fingerprints": ["2" * 64],
    }
    path = tmp_path / "calibration-pins"
    write_readout(
        path,
        model,
        VARIANT,
        weight=torch.zeros(52, 4),
        bias=torch.zeros(52),
        provenance=value,
        temperature=model.temperature,
    )
    readout = load_readout(path, model, VARIANT)
    assert readout.telemetry()["provenance"] == value
    assert readout.temperature == model.temperature


@pytest.mark.parametrize(
    "change",
    [
        lambda model: setattr(model, "letters", torch.zeros(52, dtype=torch.int64)),
        lambda model: setattr(model, "letters", torch.arange(52).float()),
        lambda model: setattr(model, "letters", torch.arange(51)),
        lambda model: setattr(model, "letters", torch.arange(100, 152)),
        lambda model: setattr(model, "revision", "main"),
        lambda model: setattr(model, "temperature", True),
        lambda model: setattr(model, "softcap", float("nan")),
    ],
)
def test_writer_rejects_unpinned_or_invalid_original_model_metadata(tmp_path, model, change):
    change(model)
    with pytest.raises(ValueError):
        write_readout(tmp_path / "invalid-model", model, VARIANT, provenance=provenance())
    assert not (tmp_path / "invalid-model").exists()


def test_projection_rejects_fp32_overflow_before_softcap_can_hide_it(tmp_path, model):
    path = tmp_path / "large-finite-rows"
    write_readout(
        path,
        model,
        VARIANT,
        weight=torch.full((52, 4), 1e38),
        bias=torch.zeros(52),
        provenance=provenance(learned=True),
    )
    readout = load_readout(path, model, VARIANT)
    with pytest.raises(ValueError, match="nonfinite logits"):
        readout.project(torch.full((1, 4), 1e38), torch.tensor([2]), model.softcap)
