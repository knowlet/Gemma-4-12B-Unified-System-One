"""Worker action-profile selection with fake providers; no network, GPU or model."""

import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/jev_decision3_browser_worker.py"
SPEC = importlib.util.spec_from_file_location("decision3_action_profile_worker_test", SCRIPT)
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)
EXPERIMENTAL = "jev-find-text-experimental-v1"


def experimental_config():
    return {
        "action_profile": EXPERIMENTAL,
        "action_profile_source": {
            "path": str(WORKER.ACTION_PROFILE_PATH),
            "pin": WORKER.pin(WORKER.ACTION_PROFILE_PATH),
        },
    }


@pytest.mark.parametrize("config", [{}, {"action_profile": "standard"}])
def test_legacy_and_explicit_standard_do_not_import_or_authenticate_selector(monkeypatch, config):
    def forbidden(*args, **kwargs):
        raise AssertionError("Standard selection touched experimental code")

    before, state = copy.deepcopy(config), {"existing": True}
    monkeypatch.setattr(WORKER, "pin", forbidden)
    monkeypatch.setattr(WORKER.importlib, "import_module", forbidden)
    assert WORKER.selected_action_profile(config) == "standard"
    assert WORKER.resolve_action_profile(config, state) == (None, None)
    assert WORKER.authenticate_action_profile(config) is None
    assert config == before
    assert state == {"existing": True}


@pytest.mark.parametrize("profile", [None, False, 1, [], {}, "find_text", "STANDARD", ""])
def test_malformed_config_is_refused_before_output_or_backend(monkeypatch, profile):
    def forbidden(*args, **kwargs):
        raise AssertionError("Malformed profile reached output or backend preparation")

    monkeypatch.setattr(WORKER, "validate_output", forbidden)
    monkeypatch.setattr(WORKER, "authenticate_sources", forbidden)
    config = {"action_profile": profile}
    with pytest.raises(ValueError, match="action profile"):
        WORKER.run(config, None)
    with pytest.raises(ValueError, match="action profile"):
        WORKER.initialize(config, {}, None)


def test_cli_selection_requires_consistent_explicit_config():
    legacy = {}
    assert WORKER.selected_action_profile(legacy, EXPERIMENTAL) == EXPERIMENTAL
    assert legacy == {}
    assert (
        WORKER.selected_action_profile({"action_profile": EXPERIMENTAL}, EXPERIMENTAL)
        == EXPERIMENTAL
    )
    with pytest.raises(ValueError, match="CLI/config"):
        WORKER.selected_action_profile({"action_profile": "standard"}, EXPERIMENTAL)
    with pytest.raises(ValueError, match="CLI/config"):
        WORKER.selected_action_profile({"action_profile": EXPERIMENTAL}, "standard")
    with pytest.raises(ValueError, match="override"):
        WORKER.selected_action_profile({}, "automatic")


def test_standard_does_not_silently_accept_an_experimental_source():
    with pytest.raises(ValueError, match="Standard worker"):
        WORKER.selected_action_profile({"action_profile_source": {}})


def fake_selector(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    selector = root / "src/s1/jev_action_profiles.py"
    selector.parent.mkdir(parents=True)
    selector.write_text("# Fake source; imported selector is injected, never executed.\n")
    dependencies = []
    for number in range(7):
        path = root / ("sources/source" + str(number) + ".py")
        path.parent.mkdir(exist_ok=True)
        path.write_text("# Fake selected dependency " + str(number) + "\n")
        dependencies.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    provenance = {
        "action_profile": EXPERIMENTAL,
        "operation_count": 9,
        "implementation": "checked_private_derivatives",
        "native_profile": "jev_decision3_native_find_text_experimental_v1",
        "wire_profile": "jev_decision3_browser_find_text_experimental_v1",
        "source_dependencies": dependencies,
    }
    calls = []

    class Adapter:
        def __init__(self, backend, *, model_identity):
            self.backend, self.identity = backend, model_identity

    def protocol(profile):
        calls.append(("resolve", profile))
        return SimpleNamespace(native_adapter_class=Adapter, provenance=provenance)

    def wire(profile, request, response):
        calls.append(("wire", profile, request, response))
        return response

    module = SimpleNamespace(
        __file__=str(selector),
        ACTION_PROFILES=("standard", EXPERIMENTAL),
        DEFAULT_ACTION_PROFILE="standard",
        resolve_protocol=protocol,
        named_browser_response=wire,
    )

    def import_module(name):
        assert name == "s1.jev_action_profiles"
        calls.append(("import", name))
        return module

    monkeypatch.setattr(WORKER, "ROOT", root)
    monkeypatch.setattr(WORKER, "ACTION_PROFILE_PATH", selector)
    monkeypatch.setattr(WORKER.importlib, "import_module", import_module)
    return experimental_config(), provenance, module, Adapter, calls


def test_same_selection_binds_native_class_and_wire_without_repair(monkeypatch, tmp_path):
    config, provenance, _, adapter_class, calls = fake_selector(monkeypatch, tmp_path)
    state = {}
    selected, wire = WORKER.resolve_action_profile(config, state)
    assert selected is adapter_class
    backend = object()
    adapter = selected(backend, model_identity="fake-alias")
    assert adapter.backend is backend
    request, response = {"ordered": [2, 1]}, {"raw": [3, None, False]}
    before = copy.deepcopy((request, response))
    assert wire.named_browser_response(request, response) is response
    assert calls[-1] == ("wire", EXPERIMENTAL, request, response)
    assert (request, response) == before
    assert state["action_profile"] == EXPERIMENTAL
    assert state["action_profile_source"] == config["action_profile_source"]
    assert state["action_profile_provenance"] == provenance
    provenance["source_dependencies"][0]["sha256"] = "0" * 64
    assert state["action_profile_provenance"] != provenance


def test_changed_selector_is_refused_before_module_or_provider(monkeypatch, tmp_path):
    config, _, _, _, calls = fake_selector(monkeypatch, tmp_path)
    WORKER.ACTION_PROFILE_PATH.write_text("# Different selected source bytes.\n")
    with pytest.raises(ValueError, match="selector (size differs|changed)"):
        WORKER.resolve_action_profile(config, {})
    assert calls == []


def test_changed_dependency_is_refused_before_adapter_is_returned(monkeypatch, tmp_path):
    config, provenance, _, _, _ = fake_selector(monkeypatch, tmp_path)
    dependency = WORKER.ROOT / provenance["source_dependencies"][0]["path"]
    dependency.write_text("# Changed source after selection.\n")
    state = {}
    with pytest.raises(ValueError, match="dependency changed"):
        WORKER.resolve_action_profile(config, state)
    assert state == {}


def test_final_authentication_detects_dependency_change(monkeypatch, tmp_path):
    config, _, _, _, _ = fake_selector(monkeypatch, tmp_path)
    state = {}
    WORKER.resolve_action_profile(config, state)
    provenance = state["action_profile_provenance"]
    dependency = WORKER.ROOT / provenance["source_dependencies"][-1]["path"]
    dependency.write_text("# Changed between the initial and final authentications.\n")
    with pytest.raises(ValueError, match="dependency changed"):
        WORKER.authenticate_action_profile(config, provenance)


@pytest.mark.parametrize(
    "field,value",
    [
        ("operation_count", 8),
        ("native_profile", "jev_decision3_native_v1"),
        ("wire_profile", "jev_decision3_browser_named_answers_v1"),
    ],
)
def test_mixed_protocol_is_refused(monkeypatch, tmp_path, field, value):
    config, provenance, _, _, _ = fake_selector(monkeypatch, tmp_path)
    provenance[field] = value
    with pytest.raises(ValueError, match="native/wire protocol differs"):
        WORKER.resolve_action_profile(config, {})


@pytest.mark.parametrize("kind", ["duplicate", "escape", "absolute", "missing"])
def test_dependency_projection_cannot_escape_or_omit_sources(monkeypatch, tmp_path, kind):
    config, provenance, _, _, _ = fake_selector(monkeypatch, tmp_path)
    rows = provenance["source_dependencies"]
    if kind == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif kind == "escape":
        rows[0]["path"] = "../unselected.py"
    elif kind == "absolute":
        rows[0]["path"] = str(WORKER.ROOT / rows[0]["path"])
    else:
        rows.pop()
    with pytest.raises(ValueError, match="action-profile (source|dependencies)"):
        WORKER.resolve_action_profile(config, {})


def test_wrong_import_origin_is_refused(monkeypatch, tmp_path):
    config, _, module, _, _ = fake_selector(monkeypatch, tmp_path)
    module.__file__ = str(tmp_path / "unselected.py")
    with pytest.raises(ValueError, match="origin/API differs"):
        WORKER.resolve_action_profile(config, {})


def test_real_factory_adapter_and_wire_accept_find_text_only_when_opted_in():
    """Exercise selected DTO/adapter/wire with one fake provider, never a model."""
    from s1.evaluation.decision3 import decision3_spec
    from s1.jev_decision3_native import JevDecision3NativeAdapter

    class Backend:
        def __init__(self):
            self.spec = decision3_spec("d3-edge")
            self.name, self.calls = self.spec.id, []

        def predict(self, request, *, deadline=None, evidence_sink=None):
            self.calls.append(copy.deepcopy(request.model_dump()))
            return {
                "model": self.spec.model_id.rsplit("/", 1)[-1],
                "answers": {
                    question.id: {
                        "type": "choice",
                        "choice": "FIND_TEXT",
                        "probabilities": {
                            label: 0.8 if label == "FIND_TEXT" else 0.1
                            for label in question.labels()
                        },
                        "confidence": 0.2,
                    }
                    for question in request.questions
                },
                "usage": {"input_tokens": 17, "output_tokens": 0},
            }

    payload = {
        "model": "fake-alias",
        "state": {"text": "Invented visible controls; no benchmark input."},
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": "Select an offered operation.",
                "criteria": {"FIND_TEXT": "Find text", "DONE": "Complete", "BLOCKED": "Blocked"},
            }
        },
    }
    standard = Backend()
    with pytest.raises(ValueError):
        JevDecision3NativeAdapter(standard, model_identity="fake-alias").predict(payload)
    assert standard.calls == []
    config, state, backend = experimental_config(), {}, Backend()
    selected, wire = WORKER.resolve_action_profile(config, state)
    original = copy.deepcopy(payload)
    legacy = selected(backend, model_identity="fake-alias").predict(payload)
    before_wire = copy.deepcopy(legacy)
    converted = wire.named_browser_response(payload, legacy)
    assert payload == original and legacy == before_wire
    assert len(backend.calls) == 1
    assert list(backend.calls[0]["questions"][0]["criteria"]) == ["FIND_TEXT", "DONE", "BLOCKED"]
    assert list(converted["answers"]) == ["operation"]
    assert converted["answers"]["operation"] == legacy["answers"][0]
    assert (
        converted["decision3_browser_wire"]["profile"]
        == state["action_profile_provenance"]["wire_profile"]
    )


@pytest.mark.parametrize(
    "label,expected",
    [
        (
            "src/s1/jev_generated.py",
            "c356e0eb3c046cb0787d7be0c5960803006231e8bdb414de644ae2167e72d849",
        ),
        (
            "src/s1/jev_action_grammar.py",
            "399e06e28172c96004b774ba97c04100687340e6b1326d317dbdf10557da75d2",
        ),
        (
            "src/s1/jev_decision3_native.py",
            "67616bb70a8acdbf96b1d2e4c1407cf8cc2a24db51b5ca383e33b4d9a0488de9",
        ),
        (
            "scripts/jev_decision3_browser_wire.py",
            "c048d7779e60fce5d1a56b38e26b9ddf5ed9e1ff9c6adc478786d0267cca3027",
        ),
    ],
)
def test_original_protocol_and_wire_bytes_remain_pinned(label, expected):
    assert hashlib.sha256((ROOT / label).read_bytes()).hexdigest() == expected
