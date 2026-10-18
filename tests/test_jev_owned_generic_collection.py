"""Pure saved-evidence seams; no full admission, browser, tokenizer or model."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jev_owned_generic_collection as owned
import jev_public_action_artifact as common
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROOF = ROOT / "artifacts/jev-runtime/generic-menu-semantics-prototype-20261018-01/proof-01"


def saved_episode(name):
    labels = [name + "/" + p.name for p in (PROOF / name).iterdir() if p.is_file()]
    labels.append("profile/runtime/jev_ultrafast/snapshot.js")
    items = {label: (PROOF / label).read_bytes() for label in labels}
    return owned._Snapshot(PROOF, tuple(items.items()), b"{}")


def mutate(snapshot, label, operation):
    items = snapshot.blobs()
    value = common.parse(items[label])
    operation(value)
    items[label] = common.encode(value) + b"\n"
    return replace(snapshot, items=tuple(items.items()))


def verifier():
    return owned._module(owned.PRODUCER), owned._module(owned.PARENT)


@pytest.mark.parametrize(
    "name",
    [
        "semantic-01-menu",
        "semantic-02-menu",
        "semantic-01-aux-global_open-fill",
        "semantic-01-wheel-reference",
    ],
)
def test_saved_single_episode_rederives_original_native_wire_guard_history_and_hits(name):
    snapshot = saved_episode(name)
    semantic, parent = verifier()
    checked = []
    compiler = SimpleNamespace(
        check=lambda payload, compiled: checked.append(compiled["wire_sha256"])
    )
    with pytest.raises(ValueError, match="Exact complete evidence population"):
        owned._verify_evidence(snapshot, semantic, parent, compiler)
    assert len(checked) == len(snapshot.rows(name + "/native-requests.jsonl"))


@pytest.mark.parametrize(
    "change", ["guard", "hit", "native-body", "history", "untrusted", "ownership"]
)
def test_saved_real_episode_mutations_refused_before_population_gate(change):
    name = "semantic-02-menu"
    snapshot = saved_episode(name)
    blobs = snapshot.blobs()
    proof = common.parse(blobs[name + "/proof.json"])
    if change in {"guard", "hit", "ownership"}:
        calls = [common.parse(line) for line in blobs[name + "/native-cdp.jsonl"].splitlines()]
        first = proof["transitions"][1]["inputs"][0]["sequence"]
        if change == "guard":
            matching = [
                c
                for c in calls[:first]
                if c["method"] == "Runtime.evaluate"
                and "c.guard(c.nodes.get" in c["params"].get("expression", "")
            ]
            matching[-1]["response"]["result"]["value"] = None
        elif change == "hit":
            calls[first - 1]["response"]["result"]["value"]["x"] += 1
        else:
            proof["ownership"]["session"] = "other-session"
            blobs[name + "/ownership.jsonl"] = common.encode(proof["ownership"]) + b"\n"
        blobs[name + "/native-cdp.jsonl"] = b"".join(common.encode(row) + b"\n" for row in calls)
    elif change in {"native-body", "history"}:
        requests = proof["requests"]
        if change == "native-body":
            requests[0]["native_payload"]["questions"]["click_target"]["criteria"].pop("1")
        else:
            requests[0]["native_payload"]["state"]["recent_actions"][0]["page_changed"] = False
        blobs[name + "/native-requests.jsonl"] = b"".join(
            common.encode(row) + b"\n" for row in requests
        )
    else:
        proof["final_private"]["events"][0]["trusted"] = False
        blobs[name + "/private-dom.jsonl"] = b"".join(
            common.encode(v) + b"\n" for v in [proof["initial_private"], proof["final_private"]]
        )
    blobs[name + "/proof.json"] = common.encode(proof) + b"\n"
    broken = replace(snapshot, items=tuple(blobs.items()))
    semantic, parent = verifier()
    with pytest.raises(ValueError) as caught:
        owned._verify_evidence(broken, semantic, parent, SimpleNamespace(check=lambda *_: None))
    assert "Exact complete evidence population" not in str(caught.value)


def test_actual_native_constructor_retains_all_heads_candidates_and_only_observed_inputs():
    proof = json.loads((PROOF / "semantic-01-menu/proof.json").read_bytes())
    request = owned.native_constructor()
    payload = request(
        proof["initial_native_page"],
        proof["record"]["payload"]["questions"]["operation"]["instructions"]["goal"],
        proof["record"]["history"],
    )
    assert common.encode(payload) == common.encode(proof["record"]["payload"])
    assert list(payload["questions"]) == ["operation", "click_target", "type_text_target"]
    assert list(payload["questions"]["click_target"]["criteria"]) == ["1", "2", "3"]
    assert "teacher_choice" not in payload and "proof_refs" not in payload
    changed = copy.deepcopy(proof["initial_native_page"])
    changed["actions"][0]["label"] = "Different visible entry"
    assert (
        request(changed, "same goal", [])["state"]["elements"][0]["label"]
        == "Different visible entry"
    )


def test_owned_case_new_namespace_and_detached_fields_refuse_forgery():
    proof = json.loads((PROOF / "semantic-01-menu/proof.json").read_bytes())
    wire = next(
        json.loads(line)
        for line in (PROOF / "compatibility-wire-requests.jsonl").read_bytes().splitlines()
        if json.loads(line)["episode_id"] == "semantic-01-menu"
    )
    scene = owned._module(owned.PRODUCER).scenarios()[0]
    case = owned._case(proof, wire, scene)
    assert case.group_id == "owned-semantic-scene:semantic-01"
    assert case.annotation()["public_registry_admitted"] is False
    assert case.gold == {"operation": "CLICK", "click_target": "3"}
    case.gold["operation"] = "DONE"
    assert case.gold["operation"] == "CLICK"
    assert case.request.wire_bytes == common.encode(wire["native_payload"])
    for bad in [replace(case, _wire=b"{}"), replace(case, _proof=object())]:
        with pytest.raises(ValueError, match="Forged/mutated"):
            bad.check()
    assert not hasattr(case, "state_role")


def test_forged_collection_refused_before_any_files_or_public_coercion():
    value = owned.AdmittedOwnedGenericCollection(None, (), b"{}", b"{}", object(), "bad")
    with pytest.raises(ValueError, match="Forged/mutated"):
        value.check()
    with pytest.raises(ValueError, match="Forged/mutated"):
        _ = value.train_cases


def test_collection_rejects_coordinated_snapshot_and_disk_rewrite(tmp_path, monkeypatch):
    monkeypatch.setattr(owned, "ROOT", tmp_path)
    monkeypatch.setattr(owned, "authenticate_sources", lambda *_: None)
    proof = tmp_path / "proof"
    proof.mkdir()
    leaf = proof / "evidence.json"
    leaf.write_bytes(b'{"original":true}')
    original = common.inventory(proof)
    index_path = tmp_path / "closed-index.json"
    index_path.write_bytes(common.encode({"proof_root": "proof", "files": original}))
    index_bytes = index_path.read_bytes()
    metadata = tmp_path / "source-metadata.json"
    metadata.write_bytes(b"{}")
    plan = common.encode(
        {
            "proof_index": owned._ref(index_path),
            "source_HOLD": owned._ref(metadata),
            "specification": owned._ref(metadata),
            "sources": {},
        }
    )
    snapshot = owned._Snapshot(
        proof, (("evidence.json", leaf.read_bytes()),), common.encode(original)
    )
    value = owned.AdmittedOwnedGenericCollection(
        snapshot,
        (),
        plan,
        b"{}",
        owned._PROOF,
        common.sha(plan + b"\0{}\0" + common.encode([])),
    )
    value.check()
    with pytest.raises(ValueError, match="Exact captured owned snapshot type"):
        replace(value, _snapshot=SimpleNamespace(root=proof, check=lambda: None)).check()
    leaf.write_bytes(b'{"coordinated_rewrite":true}')
    rewritten = replace(
        snapshot,
        items=(("evidence.json", leaf.read_bytes()),),
        inventory_bytes=common.encode(common.inventory(proof)),
    )
    rewritten.check()  # Captured bytes and disk agree, but the original index does not.
    with pytest.raises(ValueError, match="externally pinned closed index"):
        replace(value, _snapshot=rewritten).check()
    assert index_path.read_bytes() == index_bytes


@pytest.mark.parametrize(
    "ref",
    [
        {"path": "../escape", "sha256": "0" * 64, "size_bytes": 1},
        {"path": "unused", "sha256": "0" * 64, "size_bytes": True},
        {"path": "unused", "sha256": "0" * 64, "size_bytes": 1, "extra": 1},
    ],
)
def test_bad_evidence_reference_refused(ref):
    with pytest.raises((ValueError, FileNotFoundError)):
        owned._checked_ref(ref)


def test_cpu_gate_matches_old_floor_and_persists_refusal(tmp_path):
    from jev_browser_fit_public_actions import ResourceGate

    def observe():
        return {
            "MemTotal": 128 * common.GIB,
            "MemAvailable": 120 * common.GIB,
            "boot_id": "owned-boot",
        }

    old = ResourceGate(tmp_path / "old.jsonl", "owned-boot", observe=observe).check(
        "before-public-project-admission", "cpu"
    )
    new = owned.OwnedCPUAdmissionGate(tmp_path / "new.jsonl", "owned-boot", observe=observe).check(
        "before-owned-collection-admission", "cpu"
    )
    assert (
        new["required_bytes"] == old["required_bytes"]
        and new["reserve_bytes"] == old["reserve_bytes"]
    )
    assert new["swap_or_resident_credit_bytes"] == 0
    refused = owned.OwnedCPUAdmissionGate(
        tmp_path / "refused.jsonl",
        "owned-boot",
        observe=lambda: {**observe(), "MemAvailable": common.GIB},
    )
    with pytest.raises(ValueError, match="admission refused"):
        refused.check("before-owned-collection-admission", "cpu")
    row = common.parse((tmp_path / "refused.jsonl").read_bytes())
    assert row["admitted"] is False and row["error"]["type"] == "ValueError"


def test_admission_journal_cannot_change_closed_proof_before_plan_check():
    path = PROOF / "must-not-be-created.json"
    gate = owned.OwnedCPUAdmissionGate(
        path, "unused", observe=lambda: pytest.fail("Overlap entered resource gate")
    )
    with pytest.raises(ValueError, match="cannot mutate closed proof"):
        owned.validate_owned_generic_collection(
            PROOF,
            plan_bytes=b"{}",
            expected_plan_sha256="unused",
            expected_closed_index_sha256="unused",
            expected_producer_sha256="unused",
            reviewed_source_pins={},
            tokenizer_cache="unused",
            gate=gate,
        )
    assert not path.exists()


def test_checked_reference_rejects_bool_size_even_for_real_one_byte_file(tmp_path, monkeypatch):
    monkeypatch.setattr(owned, "ROOT", tmp_path)
    (tmp_path / "one.txt").write_bytes(b"x")
    with pytest.raises(ValueError, match="reference drift"):
        owned._checked_ref({"path": "one.txt", "sha256": common.sha(b"x"), "size_bytes": True})
