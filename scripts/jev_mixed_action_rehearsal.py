"""Typed public/owned admission composition and schedules; no trainer or model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import jev_owned_generic_collection as owned
import jev_public_cfg_action_artifact as public

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "mixed-public-owned-rehearsal-plan-v1"
PROFILE = "public120-owned9-row-equal-1pass-v1"
SAMPLER = "sha256-sort-row-references-v1"
SELF = "scripts/jev_mixed_action_rehearsal.py"
_PROOF = object()


def require(value, message):
    public.require(value, message)


def merge_source_pins(*closures):
    merged = {}
    for closure in closures:
        require(type(closure) is dict, "Named source mapping required")
        for label, pin in closure.items():
            public.relative(label)
            require(
                type(pin) is dict
                and set(pin) == {"sha256", "size_bytes"}
                and type(pin["sha256"]) is str
                and len(pin["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in pin["sha256"])
                and type(pin["size_bytes"]) is int
                and pin["size_bytes"] > 0,
                "Typed exact source pin",
            )
            require(
                label not in merged or public.exact(merged[label], pin),
                "Conflicting duplicate source pin: " + label,
            )
            merged[label] = dict(pin)
    return dict(sorted(merged.items()))


@dataclass(frozen=True)
class UpdateRef:
    component: str
    row_id: str
    wire_sha256: str
    numerator: int = 1
    denominator: int = 1

    def value(self):
        return {
            "component": self.component,
            "row_id": self.row_id,
            "wire_sha256": self.wire_sha256,
            "weight": {"numerator": self.numerator, "denominator": self.denominator},
        }


def prepare_schedule(rows, *, seed, version):
    """Stable complete row-reference order; never examine teacher labels."""
    require(
        type(seed) is int and seed == 42 and version == SAMPLER,
        "Explicit sampler version and seed42",
    )
    require(type(rows) is list and len(rows) == 129, "Complete129 unique training row references")
    refs = []
    for row in rows:
        require(
            type(row) is dict
            and set(row) == {"component", "row_id", "wire_sha256"}
            and row["component"] in {"public", "owned"}
            and type(row["row_id"]) is str
            and bool(row["row_id"])
            and type(row["wire_sha256"]) is str
            and len(row["wire_sha256"]) == 64
            and all(c in "0123456789abcdef" for c in row["wire_sha256"]),
            "Exact train row reference, no labels or repaired body",
        )
        refs.append(UpdateRef(**row))
    require(
        len({(r.component, r.row_id) for r in refs}) == 129
        and sum(r.component == "public" for r in refs) == 120
        and sum(r.component == "owned" for r in refs) == 9,
        "Public120/owned9 unique complete exposure",
    )

    def rank(ref):
        return public.sha(
            public.encode([SAMPLER, seed, ref.component, ref.row_id, ref.wire_sha256])
        )

    require(len({rank(r) for r in refs}) == len(refs), "Sampler rank collision")
    return tuple(sorted(refs, key=rank))


def _components(public_inputs, owned_inputs):
    require(
        type(public_inputs) is public.VerifiedPublicFitInputs
        and type(owned_inputs) is owned.AdmittedOwnedGenericCollection,
        "Independent exact public/owned admission types required",
    )
    public_inputs.check()
    owned_inputs.check()
    admitted = public_inputs.admitted()
    components = {
        "public": {"train": admitted.train_cases, "development": admitted.development_cases},
        "owned": {"train": owned_inputs.train_cases, "development": owned_inputs.development_cases},
    }
    rows = {"train": [], "development": []}
    groups = {"train": set(), "development": set()}
    for component, splits in components.items():
        require(
            {s: len(v) for s, v in splits.items()}
            == (
                {"train": 120, "development": 22}
                if component == "public"
                else {"train": 9, "development": 5}
            ),
            "Complete independently admitted component",
        )
        for split, cases in splits.items():
            for case in cases:
                require(case.split == split, "Component split drift")
                rows[split].append(
                    {
                        "component": component,
                        "row_id": case.id,
                        "wire_sha256": case.request.wire_sha256,
                    }
                )
                groups[split].add((component, case.group_id))
    require(not groups["train"] & groups["development"], "Component family/group split leakage")
    require(
        not {r["wire_sha256"] for r in rows["train"]}
        & {r["wire_sha256"] for r in rows["development"]},
        "Global train/development native request leakage",
    )
    require(
        all(len({(r["component"], r["row_id"]) for r in rows[s]}) == len(rows[s]) for s in rows),
        "Duplicate component row identity",
    )
    meta = public_inputs.metadata()
    authorities = {
        "public": {
            "source_handoff_sha256": public_inputs.source_handoff_sha256,
            "public_inventory": meta["public_inventory"],
            "provenance_sha256": public.sha(meta["provenance_bytes"].encode()),
            "counts": {"train": 120, "development": 22},
            "observation_profile": "composed-visible-text-v3",
        },
        "owned": {
            "admission": owned_inputs.provenance,
            "counts": {"train": 9, "development": 5},
            "observation_profile": "composed-visible-hover-entries-v5",
        },
    }
    sources = merge_source_pins(
        public.parse(public_inputs.source_handoff_bytes)["files"],
        owned_inputs.source_pins,
        {SELF: public.pin(ROOT / SELF)},
    )
    return rows, authorities, sources


def prepare_rehearsal_manifest(public_inputs, owned_inputs, *, objective):
    """Prepare a proposal for external pin/review, not training eligibility."""
    public.validate_objective(objective)
    rows, authorities, sources = _components(public_inputs, owned_inputs)
    schedule = prepare_schedule(rows["train"], seed=42, version=SAMPLER)
    ordered = [r.value() for r in schedule]
    manifest = {
        "schema_id": SCHEMA,
        "status": "admitted-data-plan-only-no-fitter",
        "profile": PROFILE,
        "components": authorities,
        "rows": rows,
        "objective": objective,
        "parent": dict(public.PARENT_PINS),
        "base_checkpoint": dict(public.BASE_CHECKPOINT),
        "seed": 42,
        "learning_rate": 5e-5,
        "optimizer": {"name": "AdamW", "weight_decay": 0.0, "gradient_clip_norm": 1.0},
        "sampler": {
            "version": SAMPLER,
            "ordered_updates": ordered,
            "sha256": public.sha(public.encode(ordered)),
            "total_updates": 129,
            "public_updates": 120,
            "owned_updates": 9,
            "per_row_weight": {"numerator": 1, "denominator": 1},
            "effective_component_mass": {"public": [120, 129], "owned": [9, 129]},
        },
        "development": {"public": 22, "owned": 5, "optimizer_updates": 0},
        "source_closure": sources,
        "compilation": "not_performed; new owned reference compiler and artifact contract required",
        "qualification": "new mixed kind/qualification required; current94/95 cannot accept this plan",
        "comparison_control": "Separate public-only129 descriptor with120 unique rows and9 declared repeats, same parent/objective/optimizer; not implemented",
    }
    return public.encode(manifest) + b"\n"


@dataclass(frozen=True)
class AdmittedMixedRehearsal:
    _public: public.VerifiedPublicFitInputs
    _owned: owned.AdmittedOwnedGenericCollection
    _manifest: bytes
    _proof: object = field(repr=False, compare=False)
    _seal: str = field(repr=False, compare=False)

    def check(self):
        require(
            self._proof is _PROOF and public.sha(self._manifest) == self._seal,
            "Forged/mutated mixed admission",
        )
        value = public.parse(self._manifest)
        expected = prepare_rehearsal_manifest(
            self._public, self._owned, objective=value["objective"]
        )
        require(self._manifest == expected, "Mixed component/source/schedule drift")

    def manifest(self):
        self.check()
        return public.parse(self._manifest)

    def updates(self):
        self.check()
        rows = public.parse(self._manifest)["rows"]["train"]
        return prepare_schedule(rows, seed=42, version=SAMPLER)


def admit_rehearsal(
    public_inputs,
    owned_inputs,
    *,
    mix_manifest_bytes,
    expected_mix_manifest_sha256,
    objective,
    parent_pins,
    source_closure,
):
    require(
        type(mix_manifest_bytes) is bytes
        and public.sha(mix_manifest_bytes) == expected_mix_manifest_sha256,
        "Externally reviewed mix manifest SHA",
    )
    require(public.exact(parent_pins, public.PARENT_PINS), "Original parent521 authority required")
    public.validate_objective(objective)
    expected = prepare_rehearsal_manifest(public_inputs, owned_inputs, objective=objective)
    require(
        mix_manifest_bytes == expected,
        "Exact independent components, complete schedule, weights and objective",
    )
    value = public.parse(expected)
    require(
        public.exact(value["source_closure"], source_closure),
        "External exact union source closure required",
    )
    for label, pin in source_closure.items():
        require(
            public.exact(public.pin(ROOT / public.relative(label)), pin),
            "Mixed source drift: " + label,
        )
    result = AdmittedMixedRehearsal(
        public_inputs, owned_inputs, mix_manifest_bytes, _PROOF, public.sha(mix_manifest_bytes)
    )
    result.check()
    return result
