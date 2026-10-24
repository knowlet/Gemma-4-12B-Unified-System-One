"""CFG95 source/design boundary tests; no genuine fit/qualification evidence.

The complete synthetic source trees exercise the real new ART verifier with
independently selected tiny trust anchors. No model, tokenizer, GPU, HTTP or
browser is loaded. Old qualifier/ART/source90/91 remain unchanged.
"""

import ast
import copy
import importlib.util
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import jev_public_action_artifact as OLD_ART
import jev_public_cfg_action_artifact as ART
import jev_qualify_public_actions_bf16 as OLD
import jev_qualify_public_cfg_actions_bf16 as CFG
import pytest
from local_evidence import require_local_evidence

ROOT = Path(__file__).resolve().parents[1]
OLD_PATH = ROOT / "scripts/jev_qualify_public_actions_bf16.py"
NEW_PATH = ROOT / "scripts/jev_qualify_public_cfg_actions_bf16.py"
HISTORICAL = (
    ROOT
    / "artifacts/jev-runtime/public-native-action-bf16-qualification-plan-20261016-01/plan-02.json"
)
HISTORICAL_SHA = "f8b1bf3e3b1cbd063d2bc95d5937394ef6e1a7d89e3acb7f416954a525dcdd82"
OLD_SHA = "d8cc51a8a66a26f95a0ec0f665f8ee1d95f29044ee301d02d24f1495420ee3ea"
CFG_LOSS = "scripts/jev_cfg_action_loss.py"
FORBIDDEN_S1_LOSS = "src/s1/jev_cfg_action_loss.py"


def wrapper_fixture():
    require_local_evidence(HISTORICAL)
    historical = HISTORICAL.read_bytes()
    assert CFG.digest(historical) == HISTORICAL_SHA
    delta = CFG.strict_json(historical)
    base = (ROOT / delta["base_plan"]["path"]).read_bytes()
    return base, {
        "schema_id": "jev-public-cfg-action-bf16-qualification-design-v1",
        "base_plan": copy.deepcopy(delta["base_plan"]),
        "historical_delta_utf8": historical.decode(),
        "extension": CFG.cfg_design_extension(),
    }


def test_new_importer_selects_only_explicit_cfg_artifact(monkeypatch):
    assert CFG._artifact_module() is ART
    assert ART is not OLD_ART
    assert ART.KIND == "jev_public_native_action_cfg_lora_v1"
    assert ART.QUALIFICATION_KIND == CFG.KIND != OLD.KIND
    monkeypatch.setitem(sys.modules, "jev_public_cfg_action_artifact", OLD_ART)
    with pytest.raises(ValueError, match="Foreign public verifier"):
        CFG._artifact_module()


@pytest.mark.local_evidence
def test_cfg_design_composes_authenticated_original_plan_without_changing_inference():
    base, wrapper = wrapper_fixture()
    original_base, original_history = bytes(base), HISTORICAL.read_bytes()
    effective = CFG.apply_design_delta(base, CFG.canonical(wrapper))
    assert effective["historical_inference_design"] == OLD.apply_design_delta(
        base, original_history
    )
    assert effective["cfg_training_extension"] == CFG.cfg_design_extension()
    assert base == original_base and HISTORICAL.read_bytes() == original_history
    assert (
        sum(
            row["count"]
            for row in effective["historical_inference_design"]["prospective_fixed_attempts"][
                "groups"
            ]
        )
        == 41
    )
    effective["cfg_training_extension"]["objectives"].clear()
    assert set(CFG.cfg_design_extension()["objectives"]) == {"A", "C"}


@pytest.mark.parametrize(
    "fault",
    [
        "old-plan02-alone",
        "old-plan01-alone",
        "missing",
        "extra",
        "historical-reformat",
        "historical-replacement",
        "historical-not-string",
        "base-wrapper-drift",
        "base-bytes-drift",
    ],
)
@pytest.mark.local_evidence
def test_cfg_wrapper_cannot_replace_or_skip_historical_authority(fault):
    base, wrapper = wrapper_fixture()
    if fault == "old-plan02-alone":
        body = HISTORICAL.read_bytes()
    elif fault == "old-plan01-alone":
        body = base
    else:
        if fault == "missing":
            del wrapper["extension"]
        elif fault == "extra":
            wrapper["ignore_memory_gate"] = True
        elif fault == "historical-reformat":
            wrapper["historical_delta_utf8"] = json.dumps(
                json.loads(wrapper["historical_delta_utf8"])
            )
        elif fault == "historical-replacement":
            history = json.loads(wrapper["historical_delta_utf8"])
            history["replace_operations"][0]["value"] = "changed"
            wrapper["historical_delta_utf8"] = json.dumps(history)
        elif fault == "historical-not-string":
            wrapper["historical_delta_utf8"] = {}
        elif fault == "base-wrapper-drift":
            wrapper["base_plan"]["path"] = "elsewhere/plan.json"
        else:
            base += b"\n"
        body = CFG.canonical(wrapper)
    with pytest.raises(ValueError):
        CFG.apply_design_delta(base, body)


@pytest.mark.parametrize("field", sorted(CFG.cfg_design_extension()))
@pytest.mark.local_evidence
def test_every_cfg_extension_field_is_exact_and_cannot_self_select_weaker_contract(field):
    base, wrapper = wrapper_fixture()
    value = wrapper["extension"][field]
    wrapper["extension"][field] = (
        value + ["extra"]
        if isinstance(value, list)
        else {"changed": True}
        if isinstance(value, dict)
        else value + 1
        if type(value) is int
        else value + "-changed"
    )
    with pytest.raises(ValueError, match="Exact CFG"):
        CFG.apply_design_delta(base, CFG.canonical(wrapper))


def test_cfg_design_requires_distinct_schemas_complete94_plus95_and_original90_identity():
    extension = CFG.cfg_design_extension()
    assert extension["training_source_count"] == 94 and extension["execution_source_count"] == 95
    assert extension["original_training90_sha256"] == ART.ORIGINAL_SOURCE_HANDOFF_SHA
    assert extension["producer_extensions"] == sorted(ART.EXTENSION_SOURCES)
    assert CFG_LOSS in ART.EXTENSION_SOURCES
    assert all(label.startswith("scripts/") for label in ART.EXTENSION_SOURCES)
    assert FORBIDDEN_S1_LOSS not in ART.EXTENSION_SOURCES
    assert extension["qualification_extension"] == CFG.SELF
    assert extension["artifact_kind"] == ART.KIND
    assert extension["qualification_kind"] == CFG.KIND
    assert extension["execution_schema"] == "jev_public_cfg_action_qualification_execution_v1"
    assert extension["inputs_schema"] == "jev_public_cfg_action_qualification_inputs_v1"
    assert "not whole-action NLL" in extension["comparison_scope"]


@pytest.mark.local_evidence
def test_config_design_rechecks_external_wrapper_sha_before_use(tmp_path):
    _, wrapper = wrapper_fixture()
    path = tmp_path / "plan.json"
    blob = CFG.canonical(wrapper)
    path.write_bytes(blob)
    config = {"plan_path": str(path), "plan_sha256": CFG.digest(blob)}
    assert CFG._design(config)["cfg_training_extension"] == CFG.cfg_design_extension()
    path.write_bytes(blob + b"\n")
    with pytest.raises(ValueError, match="Externally selected prospective plan"):
        CFG._design(config)


@pytest.fixture
def sources95(tmp_path, monkeypatch):
    """Real new94 ART verifier, synthetic84/90 authority and actual95 structure."""
    monkeypatch.setattr(ART, "ROOT", tmp_path)
    monkeypatch.setattr(CFG, "_artifact_module", lambda: ART)
    originals = {f"src/s1/synthetic_{index}.py": b"# synthetic original\n" for index in range(80)}
    originals.update(
        {
            f"scripts/{name}.py": b"# original caller\n"
            for name in [
                "jev_browser_benchmark",
                "jev_local_model",
                "jev_public_web_cases",
                "jev_browser_collect_public_training",
            ]
        }
    )
    assert len(originals) == 84
    assert len([label for label in originals if label.startswith("src/s1/")]) == 80
    assert CFG_LOSS not in originals and FORBIDDEN_S1_LOSS not in originals
    kernels = {label: b"# independently pinned old kernel\n" for label in ART.KERNELS}
    monkeypatch.setattr(ART, "KERNELS", {label: ART.sha(blob) for label, blob in kernels.items()})
    original90 = {
        **originals,
        **kernels,
        **{label: b"# old public artifact source\n" for label in ART.NEW_SOURCES},
    }
    all95 = {
        **original90,
        **{label: b"# new CFG producer source\n" for label in ART.EXTENSION_SOURCES},
        CFG.SELF: b"# distinct95 qualifier\n",
    }
    assert len(original90) == 90 and len(all95) == 95
    for label, blob in all95.items():
        path = tmp_path / label
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    live_s1 = {
        "src/s1/" + path.relative_to(tmp_path / "src/s1").as_posix()
        for path in (tmp_path / "src/s1").rglob("*.py")
    }
    assert live_s1 == {label for label in originals if label.startswith("src/s1/")}
    assert not (tmp_path / FORBIDDEN_S1_LOSS).exists()
    frozen = {
        "git_commit": ART.FROZEN_COMMIT,
        "file_count": 84,
        "files": {label: ART.pin(tmp_path / label) for label in sorted(originals)},
    }
    path = tmp_path / "artifacts/jev-runtime/frozen-public-observation-36/freeze-manifest.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(ART.encode(frozen))
    frozen_sha = ART.sha(path.read_bytes())
    monkeypatch.setattr(ART, "FROZEN_MANIFEST_SHA", frozen_sha)
    monkeypatch.setattr(CFG, "FROZEN_SHA", frozen_sha)
    files90 = {label: ART.pin(tmp_path / label) for label in sorted(original90)}
    old_handoff = {
        "schema_id": "jev_public_action_source_handoff_v1",
        "collection": {
            "frozen_manifest_sha256": frozen_sha,
            "git_commit": ART.FROZEN_COMMIT,
            "source_count": 84,
        },
        "files": files90,
        "identity_sha256": ART.identity(files90),
    }
    old_blob = ART.encode(old_handoff)
    monkeypatch.setattr(ART, "ORIGINAL_SOURCE_HANDOFF_SHA", ART.sha(old_blob))
    training_blob = ART.build_source_handoff(old_blob)
    training = ART.verify_source_handoff(training_blob, expected_sha256=ART.sha(training_blob))
    assert len(training["files"]) == 94
    files95 = {label: ART.pin(tmp_path / label) for label in sorted(all95)}
    value = {
        "schema_id": "jev_public_cfg_action_qualification_execution_v1",
        "training_handoff": {
            "sha256": CFG.digest(training_blob),
            "size_bytes": len(training_blob),
            "identity_sha256": training["identity_sha256"],
        },
        "files": files95,
        "identity_sha256": CFG.digest(CFG.canonical(files95)),
    }
    return tmp_path, value, training_blob, old_blob


def verify_sources(fixture, value=None, training_blob=None, *, ordered=False):
    root, baseline, training, _ = fixture
    value = baseline if value is None else value
    training = training if training_blob is None else training_blob
    blob = CFG.wire_bytes(value) if ordered else CFG.canonical(value)
    return CFG.verify_execution_sources(
        blob,
        expected_sha256=CFG.digest(blob),
        expected_training_sha256=CFG.digest(training),
        root=root,
        training_blob=training,
    )


def test_execution95_returns_detached_immutable_and_freshly_rechecked94_boundary(sources95):
    root, _, _, _ = sources95
    verified = verify_sources(sources95)
    assert len(verified.manifest()["files"]) == 95
    verified.manifest()["files"].clear()
    assert len(verified.manifest()["files"]) == 95
    with pytest.raises(FrozenInstanceError):
        verified.blob = b"{}"
    verified.check()
    (root / CFG_LOSS).write_text("# changed after capture\n")
    with pytest.raises(ValueError):
        verified.check()


@pytest.mark.parametrize(
    "fault",
    [
        "old-schema",
        "old91-population",
        "missing-qualifier",
        "extra",
        "order",
        "identity",
        "bool-size",
        "current-extension",
        "original84-drift",
        "missing-extension-same-count",
        "old90-handoff",
        "frozen-authority",
    ],
)
def test_execution95_refuses_rehashed_legacy_or_incomplete_source_authority(sources95, fault):
    root, baseline, training, old90 = sources95
    value = copy.deepcopy(baseline)
    if fault == "old-schema":
        value["schema_id"] = "jev_public_action_qualification_execution_v1"
    elif fault == "old91-population":
        for label in ART.EXTENSION_SOURCES:
            del value["files"][label]
        assert len(value["files"]) == 91
    elif fault == "missing-qualifier":
        del value["files"][CFG.SELF]
    elif fault == "extra":
        value["files"]["scripts/extra.py"] = {"sha256": "0" * 64, "size_bytes": 1}
    elif fault == "order":
        value["files"] = dict(reversed(list(value["files"].items())))
    elif fault == "identity":
        value["identity_sha256"] = "0" * 64
    elif fault == "bool-size":
        value["files"][CFG.SELF]["size_bytes"] = True
    elif fault == "current-extension":
        (root / CFG_LOSS).write_text("# changed\n")
    elif fault == "original84-drift":
        path = root / "src/s1/synthetic_0.py"
        path.write_text("# changed old source\n")
        value["files"]["src/s1/synthetic_0.py"] = CFG.pin(path)
    elif fault == "missing-extension-same-count":
        absent = "scripts/jev_browser_fit_cfg_actions.py"
        replacement = "scripts/unreviewed_cfg.py"
        (root / replacement).write_bytes((root / absent).read_bytes())
        value["files"][replacement] = value["files"].pop(absent)
        newer = ART.parse(training)
        newer["files"][replacement] = newer["files"].pop(absent)
        newer["files"] = dict(sorted(newer["files"].items()))
        newer["identity_sha256"] = ART.identity(newer["files"])
        training = ART.encode(newer)
        value["training_handoff"]["identity_sha256"] = newer["identity_sha256"]
    elif fault == "old90-handoff":
        training = old90
        value["training_handoff"]["identity_sha256"] = ART.parse(training)["identity_sha256"]
    else:
        (
            root / "artifacts/jev-runtime/frozen-public-observation-36/freeze-manifest.json"
        ).write_bytes(b"{}")
    if fault not in {"identity", "order"}:
        value["files"] = dict(sorted(value["files"].items()))
    if fault != "identity":
        value["identity_sha256"] = CFG.digest(CFG.canonical(value["files"]))
    value["training_handoff"].update(sha256=CFG.digest(training), size_bytes=len(training))
    with pytest.raises(ValueError):
        verify_sources(sources95, value, training, ordered=fault == "order")


@pytest.mark.parametrize("fault", ["extra-live-s1-loss", "rehashed-relocated-loss"])
def test_execution95_rejects_loss_inside_original_s1_boundary(sources95, fault):
    root, baseline, training, _ = sources95
    value = copy.deepcopy(baseline)
    old_path = root / FORBIDDEN_S1_LOSS
    old_path.write_bytes((root / CFG_LOSS).read_bytes())
    if fault == "rehashed-relocated-loss":
        # Even matching94/95 counts, self-consistent identities and a real
        # on-disk file cannot authorize replacing the explicit script source.
        (root / CFG_LOSS).unlink()
        value["files"][FORBIDDEN_S1_LOSS] = value["files"].pop(CFG_LOSS)
        value["files"] = dict(sorted(value["files"].items()))
        value["identity_sha256"] = CFG.digest(CFG.canonical(value["files"]))
        newer = ART.parse(training)
        newer["files"][FORBIDDEN_S1_LOSS] = newer["files"].pop(CFG_LOSS)
        newer["files"] = dict(sorted(newer["files"].items()))
        newer["identity_sha256"] = ART.identity(newer["files"])
        training = ART.encode(newer)
        value["training_handoff"].update(
            sha256=CFG.digest(training),
            size_bytes=len(training),
            identity_sha256=newer["identity_sha256"],
        )
        assert len(value["files"]) == 95 and len(newer["files"]) == 94
    with pytest.raises(ValueError, match="original84|four explicit CFG"):
        verify_sources(sources95, value, training)


def test_execution95_requires_full_external_handoff_and_external_execution_sha(sources95):
    root, value, training, _ = sources95
    blob = CFG.canonical(value)
    with pytest.raises(ValueError, match="training94"):
        CFG.verify_execution_sources(
            blob,
            expected_sha256=CFG.digest(blob),
            expected_training_sha256=CFG.digest(training),
            root=root,
        )
    with pytest.raises(ValueError, match="External execution SHA"):
        CFG.verify_execution_sources(
            blob,
            expected_sha256="0" * 64,
            expected_training_sha256=CFG.digest(training),
            root=root,
            training_blob=training,
        )


@pytest.mark.parametrize(
    "kind", [OLD.KIND, "jev_action_bf16_qualification_v1", "jev_public_native_action_cfg_lora_v1"]
)
def test_old_qualification_kind_is_refused_before_any_candidate_or_model_access(kind):
    value = {
        "schema_version": 1,
        "kind": kind,
        "qualified": True,
        "task_performance_attested": False,
        "public": {},
        "execution_sha256": "1" * 64,
        "runtime_sha256": "2" * 64,
        "plan_sha256": "3" * 64,
        "inputs_sha256": "4" * 64,
        "runtime_profile": CFG.runtime_profile(),
        "inventory": {},
        "run": {},
    }
    with pytest.raises(ValueError, match="Completed distinct technical qualification"):
        CFG.validate_qualification(
            value,
            expected_public={},
            expected_execution_sha256="1" * 64,
            expected_runtime_sha256="2" * 64,
            expected_plan_sha256="3" * 64,
            expected_inputs_sha256="4" * 64,
            captured_files={},
        )


@pytest.mark.parametrize("scope,phase", [("cpu", "startup"), ("load", "case-0-policy-forward-0")])
def test_original_resource_floors_and_no_resident_swap_credit_are_preserved(tmp_path, scope, phase):
    total = 128 * CFG.GIB
    reserve = max(24 * CFG.GIB, (total + 4) // 5)
    required = reserve + (4 * CFG.GIB if scope == "cpu" else 2 * CFG.BASE_BYTES)
    for offset in [0, -1]:
        path = tmp_path / f"gate-{offset}.jsonl"
        gate = CFG.QualificationGate(
            path,
            "synthetic-boot",
            observe=lambda: {
                "MemTotal": total,
                "MemAvailable": required + offset,
                "boot_id": "synthetic-boot",
            },
            clock=lambda: 1.0,
        )
        if offset:
            with pytest.raises(ValueError, match="resource admission refused"):
                gate.check(phase, scope)
        else:
            assert gate.check(phase, scope)["admitted"] is True
        row = CFG.strict_json(path.read_bytes().splitlines()[0])
        assert row["required_bytes"] == required and row["reserve_bytes"] == reserve
        assert row["swap_or_resident_credit_bytes"] == 0 and row["admitted"] is (offset == 0)


def test_original_native_policy_runtime_limits_and41_denominator_are_identical():
    assert CFG.runtime_profile() == OLD.runtime_profile()
    assert CFG.expected_case_ids() == OLD.expected_case_ids()
    assert len(CFG.expected_case_ids()) == 41
    profile = CFG.runtime_profile()
    assert (
        profile["context"],
        profile["candidate_cap"],
        profile["action_tokens"],
        profile["helper_tokens"],
        profile["policy_seconds"],
        profile["http_seconds"],
    ) == (8192, 400, 32, 128, 20, 25)
    assert profile["helper_reasoning"] == {"enabled": False}


def test_new_input_schema_accepts_same_complete41_recipes_and_refuses_legacy_schema():
    # Reuse the existing explicitly synthetic, pure41-input fixture. Importing
    # this test module starts no worker/framework/network; its fixtures are not
    # a genuine qualification for either producer.
    path = ROOT / "tests/test_jev_qualify_public_actions_bf16.py"
    spec = importlib.util.spec_from_file_location("cfg_legacy_pure_input_fixture", path)
    old_fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_fixtures)
    old_inputs = old_fixtures.inputs_fixture()
    arguments = {
        "expected_plan_sha256": old_inputs["plan_sha256"],
        "expected_execution_sha256": old_inputs["execution_sha256"],
        "expected_runtime_sha256": old_inputs["runtime_sha256"],
    }
    with pytest.raises(ValueError, match="Externally selected input identities"):
        CFG.validate_input_manifest(old_inputs, **arguments)
    new_inputs = copy.deepcopy(old_inputs)
    new_inputs["schema_id"] = "jev_public_cfg_action_qualification_inputs_v1"
    assert CFG.validate_input_manifest(new_inputs, **arguments) == new_inputs
    assert new_inputs["cases"] == old_inputs["cases"] and len(new_inputs["cases"]) == 41
    assert new_inputs["materialization"] == old_inputs["materialization"]
    with pytest.raises(ValueError, match="Externally selected input identities"):
        OLD.validate_input_manifest(new_inputs, **arguments)


def test_entire_inherited_qualifier_body_has_only_the_reviewed_semantic_delta():
    assert CFG.digest(OLD_PATH.read_bytes()) == OLD_SHA
    old_tree, new_tree = ast.parse(OLD_PATH.read_bytes()), ast.parse(NEW_PATH.read_bytes())
    old_nodes = {
        node.name: node
        for node in old_tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    new_nodes = {
        node.name: node
        for node in new_tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert set(new_nodes) - set(old_nodes) == {
        "cfg_design_extension",
        "_apply_historical_design_delta",
    }
    changes = {
        "_artifact_module",
        "apply_design_delta",
        "verify_execution_sources",
        "validate_input_manifest",
        "_sources",
        "_source_copy",
        "prepare_qualification_inputs",
    }
    old_text, new_text = OLD_PATH.read_text(), NEW_PATH.read_text()
    for name in set(old_nodes) - changes:
        assert ast.dump(old_nodes[name], include_attributes=False) == ast.dump(
            new_nodes[name], include_attributes=False
        ), name
        assert ast.get_source_segment(old_text, old_nodes[name]) == ast.get_source_segment(
            new_text, new_nodes[name]
        ), name
    historical = copy.deepcopy(new_nodes["_apply_historical_design_delta"])
    historical.name = "apply_design_delta"
    assert isinstance(historical.body[0], ast.Expr) and isinstance(
        historical.body[0].value, ast.Constant
    )
    historical.body.pop(0)
    assert ast.dump(historical, include_attributes=False) == ast.dump(
        old_nodes["apply_design_delta"], include_attributes=False
    )
    substitutions = {
        "jev_public_cfg_action_artifact": "jev_public_action_artifact",
        "scripts/jev_public_cfg_action_artifact.py": "scripts/jev_public_action_artifact.py",
        "Complete external training94 handoff bytes required": "Complete external training90 handoff bytes required",
        "jev_public_cfg_action_qualification_execution_v1": "jev_public_action_qualification_execution_v1",
        "Exact sorted95 source population": "Exact sorted91 source population",
        "Source95 identity": "Source91 identity",
        "Current95 source pin": "Current91 source pin",
        "Separate training94 equality": "Separate training90 equality",
        "jev_public_cfg_action_qualification_inputs_v1": "jev_public_action_qualification_inputs_v1",
        "Project loaded module origin outside source95: ": "Project loaded module origin outside source91: ",
        "Exact full95 copied source closure": "Exact full91 copied source closure",
    }

    class InvertReviewedLiterals(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is str and node.value in substitutions:
                node.value = substitutions[node.value]
            elif type(node.value) is int and node.value == 95:
                node.value = 91
            return node

    for name in changes - {"apply_design_delta"}:
        inverted = InvertReviewedLiterals().visit(copy.deepcopy(new_nodes[name]))
        assert ast.dump(inverted, include_attributes=False) == ast.dump(
            old_nodes[name], include_attributes=False
        ), name
    old_assign = {
        node.targets[0].id: node
        for node in old_tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    new_assign = {
        node.targets[0].id: node
        for node in new_tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    assert set(new_assign) == set(old_assign)
    for name in set(old_assign) - {"SELF", "KIND"}:
        assert ast.dump(old_assign[name], include_attributes=False) == ast.dump(
            new_assign[name], include_attributes=False
        ), name
    assert [
        ast.dump(node, include_attributes=False)
        for node in old_tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ] == [
        ast.dump(node, include_attributes=False)
        for node in new_tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    for name in [
        "QualificationGate",
        "runtime_profile",
        "_make_application",
        "_http",
        "_load_model",
        "attach_qualified_public_bf16_action_adapter",
        "_run_child_body",
        "_owned_controller",
    ]:
        assert name not in changes and name in old_nodes
