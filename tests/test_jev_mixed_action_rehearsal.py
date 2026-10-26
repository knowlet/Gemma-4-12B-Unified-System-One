"""Pure schedules and typed composition seams; no training or actual admission."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import jev_mixed_action_rehearsal as mixed
import jev_public_cfg_action_artifact as public
import pytest


def reference_rows():
    return [
        {
            "component": component,
            "row_id": component + "-" + str(i),
            "wire_sha256": public.sha((component + str(i)).encode()),
        }
        for component, count in (("public", 120), ("owned", 9))
        for i in range(count)
    ]


def test_schedule_complete_stable_seeded_row_equal_no_development_or_labels():
    rows = reference_rows()
    a = mixed.prepare_schedule(rows, seed=42, version=mixed.SAMPLER)
    b = mixed.prepare_schedule(list(reversed(rows)), seed=42, version=mixed.SAMPLER)
    assert a == b and len(a) == 129
    assert (
        sum(r.component == "public" for r in a) == 120
        and sum(r.component == "owned" for r in a) == 9
    )
    assert len({(r.component, r.row_id) for r in a}) == 129
    assert all(r.numerator == r.denominator == 1 for r in a)
    assert set(a[0].value()) == {"component", "row_id", "wire_sha256", "weight"}


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "renamed-corpus", "gold", "weight", "seed-bool", "unknown-sampler"],
)
def test_schedule_refuses_hidden_population_weight_or_version_changes(mutation):
    rows = reference_rows()
    seed = 42
    version = mixed.SAMPLER
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[0] = rows[1]
    elif mutation == "renamed-corpus":
        rows[0]["component"] = "development"
    elif mutation == "gold":
        rows[0]["gold"] = "CLICK"
    elif mutation == "weight":
        rows[0]["weight"] = {"numerator": 2, "denominator": 1}
    elif mutation == "seed-bool":
        seed = True
    else:
        version = "adaptive-best-loss"
    with pytest.raises(ValueError):
        mixed.prepare_schedule(rows, seed=seed, version=version)


def test_source_union_exact_duplicates_conflicts_and_strict_types():
    pin = {"sha256": "1" * 64, "size_bytes": 10}
    assert mixed.merge_source_pins({"scripts/unit.py": pin}, {"scripts/unit.py": dict(pin)}) == {
        "scripts/unit.py": pin
    }
    for changed in ({**pin, "sha256": "2" * 64}, {**pin, "size_bytes": True}):
        with pytest.raises(ValueError):
            mixed.merge_source_pins({"scripts/unit.py": pin}, {"scripts/unit.py": changed})
    with pytest.raises(ValueError):
        mixed.merge_source_pins({"../escape": pin})


@pytest.fixture
def fake_independent_admissions(monkeypatch):
    """Substitute only verified-type boundaries, exercising composition itself."""
    state = {"drift": False}

    def cases(component, split, count):
        return tuple(
            SimpleNamespace(
                id=f"{component}-{split}-{i}",
                split=split,
                group_id=f"{component}-{split}-family",
                request=SimpleNamespace(
                    wire_sha256=public.sha(f"{component}-{split}-{i}".encode())
                ),
            )
            for i in range(count)
        )

    source_public = {"scripts/fake-public.py": {"sha256": "1" * 64, "size_bytes": 10}}
    source_owned = {"scripts/fake-owned.py": {"sha256": "2" * 64, "size_bytes": 20}}

    class Public:
        source_handoff_sha256 = "3" * 64
        source_handoff_bytes = public.encode({"files": source_public})

        def check(self):
            if state["drift"]:
                raise ValueError("Original public admission drift")

        def admitted(self):
            return SimpleNamespace(
                train_cases=cases("public", "train", 120),
                development_cases=cases("public", "development", 22),
            )

        def metadata(self):
            return {
                "public_inventory": {"identity_sha256": "4" * 64, "files": {}},
                "provenance_bytes": "public-original-provenance",
            }

    class Owned:
        source_pins = source_owned
        train_cases = cases("owned", "train", 9)
        development_cases = cases("owned", "development", 5)
        provenance = {
            "origin": "owned_local",
            "public_registry_admitted": False,
            "admission_plan_sha256": "5" * 64,
        }

        def check(self):
            pass

    monkeypatch.setattr(mixed.public, "VerifiedPublicFitInputs", Public)
    monkeypatch.setattr(mixed.owned, "AdmittedOwnedGenericCollection", Owned)
    real_pin = public.pin

    def pin(path, *args, **kwargs):
        label = str(path).removeprefix(str(mixed.ROOT) + "/")
        return (source_public | source_owned).get(label) or real_pin(path, *args, **kwargs)

    monkeypatch.setattr(mixed.public, "pin", pin)
    return Public(), Owned(), state


def test_typed_composition_external_pin_complete_descriptor_and_drift(fake_independent_admissions):
    public_inputs, owned_inputs, state = fake_independent_admissions
    objective = public.objective("C")
    blob = mixed.prepare_rehearsal_manifest(public_inputs, owned_inputs, objective=objective)
    value = public.parse(blob)
    assert len(value["rows"]["train"]) == 129 and len(value["rows"]["development"]) == 27
    assert value["development"] == {"public": 22, "owned": 5, "optimizer_updates": 0}
    assert value["sampler"]["effective_component_mass"] == {"public": [120, 129], "owned": [9, 129]}
    result = mixed.admit_rehearsal(
        public_inputs,
        owned_inputs,
        mix_manifest_bytes=blob,
        expected_mix_manifest_sha256=public.sha(blob),
        objective=objective,
        parent_pins=public.PARENT_PINS,
        source_closure=value["source_closure"],
    )
    assert len(result.updates()) == 129
    changed = result.manifest()
    changed["objective"]["mode"] = "A"
    assert result.manifest()["objective"]["mode"] == "C"
    with pytest.raises(ValueError, match="Forged/mutated"):
        replace(result, _proof=object()).check()
    state["drift"] = True
    with pytest.raises(ValueError, match="Original public admission drift"):
        result.check()


@pytest.mark.parametrize(
    "change",
    [
        "weight",
        "order",
        "objective",
        "dev-update",
        "missing-owned",
        "source-extra",
        "teacher-label",
    ],
)
def test_recomputed_external_sha_does_not_authorize_semantic_manifest_drift(
    fake_independent_admissions, change
):
    public_inputs, owned_inputs, _state = fake_independent_admissions
    objective = public.objective("A")
    original = mixed.prepare_rehearsal_manifest(public_inputs, owned_inputs, objective=objective)
    value = public.parse(original)
    sources = copy.deepcopy(value["source_closure"])
    if change == "weight":
        value["sampler"]["ordered_updates"][0]["weight"]["numerator"] = 2
    elif change == "order":
        value["sampler"]["ordered_updates"].reverse()
    elif change == "objective":
        value["objective"] = public.objective("C")
    elif change == "dev-update":
        value["development"]["optimizer_updates"] = 1
    elif change == "missing-owned":
        value["rows"]["train"].pop()
    elif change == "source-extra":
        value["source_closure"]["scripts/unknown.py"] = {"sha256": "6" * 64, "size_bytes": 3}
    else:
        value["rows"]["train"][0]["gold"] = "CLICK"
    blob = public.encode(value) + b"\n"
    with pytest.raises(ValueError, match="Exact independent components"):
        mixed.admit_rehearsal(
            public_inputs,
            owned_inputs,
            mix_manifest_bytes=blob,
            expected_mix_manifest_sha256=public.sha(blob),
            objective=objective,
            parent_pins=public.PARENT_PINS,
            source_closure=sources,
        )


def test_bare_public_or_owned_shaped_objects_never_coerced():
    with pytest.raises(ValueError, match="Independent exact"):
        mixed.prepare_rehearsal_manifest(
            SimpleNamespace(check=lambda: None),
            SimpleNamespace(check=lambda: None),
            objective=public.objective("C"),
        )
