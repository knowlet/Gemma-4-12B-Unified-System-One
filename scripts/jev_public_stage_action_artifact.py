"""Typed local GB10 stage-fit inputs, CPU compilation and closed child contract.

This is a new artifact kind. Original public admission and tensor-only helpers
remain authorities for their own bytes; old ART94/95 completed-child validators
never certify this producer. All framework/tokenizer imports are lazy, after a
fresh durable CPU gate. Importing this module performs no admission or compute.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import jev_stage_memory_profile_v2 as memory

ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA = "jev-local-stage-fit-input-plan-v2"
INPUT_SCHEMA = "jev-local-stage-fit-admitted-inputs-v2"
COMPILE_SCHEMA = "jev-local-stage-fit-CPU-reference-v2"
MANIFEST_SCHEMA = "jev-local-stage-fit-completed-child-v2"
RECEIPT_SCHEMA = "jev-local-stage-fit-closed-controller-v2"
EXTENSIONS = (
    "scripts/jev_browser_fit_public_stage_actions.py",
    "scripts/jev_public_stage_action_artifact.py",
    "scripts/jev_stage_memory_profile_v2.py",
)
PUBLIC_HANDOFF_SHA = "b4d256c64490c4a9668e6ba6a38f8a8227ef46eef628dd091e9ea13aaa127d1e"
CFG142_SHA = "38a863ee8c0234c40fc690688ebd8cba1d7bdcb9e5963bd0e50bbf7d6f8f247e"
OWNED_PLAN_SHA = "1391920f12433241598549c69892619f4c1ace890318f1f9aa54539186ae7243"
MIX_SHA = {
    "A": "624b631a0a89dd9265970ef5602e509aa07b7d0d4ab7b72a2e4219210cb81912",
    "C": "805585db266037f1235f3bacfee5fd32ada6a6d2c9982865d593ec0ec97f331c",
}
RUNTIME_SHA = {
    "native_toolchain": "1ad0a1749e1ab69235cf6356939ef55edf904ba8942794f6241ef9580c676582",
    "allocator": "1d95d5f9250132299901871433850b0a39b86b18cc2b58eed378dabcf5117a61",
    "optional_dependency": "b46f7c03063170992808a0eaddcd0968d9e6f91604430c80d86ad669ab7b48b7",
}
COUNTS = {
    "public120": {"train": 120, "development": 22},
    "mixed129": {"train": 129, "development": 27},
}
KINDS = {"public120": memory.PUBLIC_KIND, "mixed129": memory.MIXED_KIND}
HEAD_SHA = "9619fd670a4074367c72f2cf09d777970ad1eafa511bff605a82c8558dcc24d4"
_PROOF = object()
require, encode, parse, sha = memory.require, memory.encode, memory.parse, memory.sha


def relative(label):
    require(type(label) is str and label and "\\" not in label, "Relative artifact path")
    path = PurePosixPath(label)
    require(
        not path.is_absolute() and all(p not in {"", ".", ".."} for p in path.parts),
        "Confined artifact path",
    )
    require(path.as_posix() == label, "Canonical artifact path")
    return label


def pin(value):
    require(type(value) is dict and set(value) == {"sha256", "size_bytes"}, "Exact file pin")
    require(
        type(value["sha256"]) is str
        and len(value["sha256"]) == 64
        and all(c in "0123456789abcdef" for c in value["sha256"]),
        "SHA256 pin",
    )
    require(
        type(value["size_bytes"]) is int and 0 < value["size_bytes"] <= memory.TOTAL_CAP,
        "Bounded file size",
    )
    return value


def read_ref(ref):
    require(
        type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"},
        "Externally selected reference",
    )
    relative(ref["path"])
    expected = pin({k: ref[k] for k in ("sha256", "size_bytes")})
    path = ROOT / ref["path"]
    require(memory.file_pin(path) == expected, "External reference drift: " + ref["path"])
    return memory.read(path)


def reference(path):
    path = memory.ordinary(path)
    require(path.is_relative_to(ROOT), "Workspace reference required")
    return {"path": path.relative_to(ROOT).as_posix(), **memory.file_pin(path)}


def write_blob(path, blob):
    require(type(blob) is bytes and len(blob) <= memory.FILE_CAP, "Bounded compiled evidence file")
    fd = os.open(memory.ordinary(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        require(stream.write(blob) == len(blob), "Complete evidence write")
        stream.flush()
        os.fsync(stream.fileno())


def check_plan(value):
    require(
        type(value) is dict
        and set(value)
        == {
            "schema_id",
            "regime",
            "mode",
            "public_handoff",
            "original_CFG142_plans",
            "owned_admission_plan",
            "mixed_manifest",
            "source_union",
            "runtime",
        },
        "Exact stage input plan",
    )
    require(
        value["schema_id"] == PLAN_SCHEMA
        and value["regime"] in COUNTS
        and value["mode"] in {"A", "C"},
        "New typed local stage regime",
    )
    require(
        value["public_handoff"]["sha256"] == PUBLIC_HANDOFF_SHA
        and value["original_CFG142_plans"]["sha256"] == CFG142_SHA,
        "Original CFG94 and all142 plans",
    )
    require(
        type(value["runtime"]) is dict and set(value["runtime"]) == set(RUNTIME_SHA),
        "Exact326/22/15 runtime references",
    )
    for key, digest in RUNTIME_SHA.items():
        require(value["runtime"][key]["sha256"] == digest, "Original pinned runtime identity")
    if value["regime"] == "public120":
        require(
            value["owned_admission_plan"] is None and value["mixed_manifest"] is None,
            "Public regime cannot contain owned updates",
        )
    else:
        require(
            value["owned_admission_plan"]["sha256"] == OWNED_PLAN_SHA
            and value["mixed_manifest"]["sha256"] == MIX_SHA[value["mode"]],
            "Externally admitted exact mixed manifest and owned plan",
        )
    for ref in [
        value["public_handoff"],
        value["original_CFG142_plans"],
        value["source_union"],
        *value["runtime"].values(),
    ]:
        read_ref(ref)
    if value["regime"] == "mixed129":
        read_ref(value["owned_admission_plan"])
        read_ref(value["mixed_manifest"])
    return value


def check_source_union(plan):
    """Derive population from exact original closure plus three named additions."""
    union = memory.verify_stage_source_union(
        plan["source_union"], worker_entry=ROOT / EXTENSIONS[0]
    )
    public_files = parse(read_ref(plan["public_handoff"]))["files"]
    base = (
        public_files
        if plan["regime"] == "public120"
        else parse(read_ref(plan["mixed_manifest"]))["source_closure"]
    )
    require(
        all(base.get(k) == v for k, v in public_files.items()),
        "All original CFG94 entries retained exactly",
    )
    require(
        set(union["extensions"]) == set(EXTENSIONS) and not set(base) & set(EXTENSIONS),
        "Exactly v2 capsule, fitter and new artifact contract; v1 excluded",
    )
    require(
        parse(read_ref(union["base_inventory"]))["files"] == base, "Actual base closure authority"
    )
    require(union["files"] == {**base, **union["extensions"]}, "No fabricated source population")
    return {
        "identity": union["identity_sha256"],
        "base_count": len(base),
        "extension_count": len(union["extensions"]),
        "count": len(union["files"]),
    }


class CPUPrepareGate:
    """Fresh original CPU floor; never a resident GPU gate or resource waiver."""

    def __init__(self, output, expected_boot_id, *, observe=memory.host_memory):
        self.output, self.expected_boot_id, self.observe = Path(output), expected_boot_id, observe
        self.sequence = 0
        require(not (self.output / "CPU-gates.jsonl").exists(), "Fresh CPU gate journal")

    def check(self, phase, scope):
        # Original compiler calls use its historical 'train' phase label. This
        # gate authorizes only a CPU processor view, never model construction.
        require(
            scope in {"cpu", "train"} and type(phase) is str and self.sequence < 1024,
            "Bounded CPU compiler phase",
        )
        obs = self.observe()
        require(
            all(type(obs[k]) is int and obs[k] > 0 for k in ("MemTotal", "MemAvailable")),
            "Integer CPU observation",
        )
        reserve = max(24 * memory.GIB, (obs["MemTotal"] + 4) // 5)
        required = reserve + 4 * memory.GIB
        admitted = (
            obs["boot_id"] == self.expected_boot_id
            and required <= obs["MemAvailable"] <= obs["MemTotal"]
        )
        memory.append(
            self.output / "CPU-gates.jsonl",
            {
                "schema_id": "jev-stage-CPU-gate-v2",
                "sequence": self.sequence,
                "phase": phase,
                "requested_label": scope,
                "effective_scope": "CPU-processor-only",
                "observation": obs,
                "required_bytes": required,
                "resident_credit_bytes": 0,
                "admitted": admitted,
            },
            memory.MIB,
        )
        self.sequence += 1
        require(admitted, "Fresh original CPU floor refused")


def public_api():
    import jev_public_cfg_action_artifact as public

    return public


def row_refs(public_inputs, owned_inputs=None):
    admitted = public_inputs.admitted()
    result = {s: [] for s in ("train", "development")}
    components = [("public", admitted)] + (
        [] if owned_inputs is None else [("owned", owned_inputs)]
    )
    for component, collection in components:
        for split in result:
            cases = collection.train_cases if split == "train" else collection.development_cases
            result[split].extend(
                {"component": component, "row_id": case.id, "wire_sha256": case.request.wire_sha256}
                for case in cases
            )
    return result


def schedule(regime, refs, *, mixed_order=None):
    require(
        regime in COUNTS and type(refs) is dict and set(refs) == {"train", "development"},
        "Typed complete split references",
    )
    require(
        {s: len(rows) for s, rows in refs.items()} == COUNTS[regime],
        "All120/22 or129/27 rows required",
    )
    for split, rows in refs.items():
        require(
            all(
                type(r) is dict
                and set(r) == {"component", "row_id", "wire_sha256"}
                and r["component"] in {"public", "owned"}
                and type(r["row_id"]) is str
                and r["row_id"]
                for r in rows
            ),
            "Exact row references",
        )
        require(
            len({(r["component"], r["row_id"]) for r in rows}) == len(rows),
            "Unique component row identities",
        )
    require(
        not {r["wire_sha256"] for r in refs["train"]}
        & {r["wire_sha256"] for r in refs["development"]},
        "No development wire leakage",
    )
    if regime == "public120":
        require(
            mixed_order is None
            and all(r["component"] == "public" for rows in refs.values() for r in rows),
            "Original public-only regime",
        )
        order = list(refs["train"])
        random.Random(42).shuffle(order)
        return [{**r, "weight": {"numerator": 1, "denominator": 1}} for r in order]
    import jev_mixed_action_rehearsal as mixed

    expected = [
        r.value() for r in mixed.prepare_schedule(refs["train"], seed=42, version=mixed.SAMPLER)
    ]
    require(
        mixed_order == expected,
        "Exact independently admitted SHA-ordered UpdateRefs; no shuffle/repeats",
    )
    return expected


def descriptor(plan, public_inputs, owned_inputs, mixed_inputs):
    refs = row_refs(public_inputs, owned_inputs)
    mixed_order = None if mixed_inputs is None else [r.value() for r in mixed_inputs.updates()]
    updates = schedule(plan["regime"], refs, mixed_order=mixed_order)
    meta = public_inputs.metadata()
    return {
        "schema_id": INPUT_SCHEMA,
        "regime": plan["regime"],
        "mode": plan["mode"],
        "kind": KINDS[plan["regime"]],
        "counts": COUNTS[plan["regime"]],
        "rows": refs,
        "ordered_updates": updates,
        "sampler": "random.Random(42).shuffle-original-public-order"
        if mixed_inputs is None
        else "sha256-sort-row-references-v1",
        "ordered_updates_sha256": sha(encode(updates)),
        "public_inventory": meta["public_inventory"],
        "public_provenance_sha256": sha(meta["provenance_bytes"].encode()),
        "owned_provenance": None if owned_inputs is None else owned_inputs.provenance,
        "mixed_manifest": plan["mixed_manifest"],
        "public_handoff": plan["public_handoff"],
        "source_union": plan["source_union"],
        "source_identity": check_source_union(plan),
        "objective": public_api().objective(plan["mode"]),
        "seed": 42,
        "learning_rate": 5e-5,
        "development_updates": 0,
    }


@dataclass(frozen=True)
class StageInputs:
    _plan: bytes
    _public: object
    _owned: object
    _mixed: object
    _descriptor: bytes
    _proof: object = field(repr=False, compare=False)
    _seal: str = field(repr=False, compare=False)

    def check(self):
        require(
            self._proof is _PROOF and sha(self._plan + b"\0" + self._descriptor) == self._seal,
            "Forged or mutated stage admission",
        )
        public = public_api()
        require(
            type(self._public) is public.VerifiedPublicFitInputs,
            "Genuine original public admission required",
        )
        self._public.check()
        plan = check_plan(parse(self._plan))
        if plan["regime"] == "mixed129":
            import jev_mixed_action_rehearsal as mixed
            import jev_owned_generic_collection as owned

            require(
                type(self._owned) is owned.AdmittedOwnedGenericCollection
                and type(self._mixed) is mixed.AdmittedMixedRehearsal,
                "Separate genuine owned and mixed types; no public DTO coercion",
            )
            self._owned.check()
            self._mixed.check()
        else:
            require(self._owned is None and self._mixed is None, "Public-only authority")
        require(
            encode(descriptor(plan, self._public, self._owned, self._mixed)) == self._descriptor,
            "Complete admitted input/schedule/source drift",
        )

    def value(self):
        require(
            self._proof is _PROOF and sha(self._plan + b"\0" + self._descriptor) == self._seal,
            "Sealed stage descriptor",
        )
        return parse(self._descriptor)


def admit_inputs(plan, gate):
    gate.check("before-any-project-framework-or-parent-payload", "cpu")
    check_plan(plan)
    check_source_union(plan)
    import jev_browser_fit_public_cfg_actions as producer

    public = public_api()
    handoff = read_ref(plan["public_handoff"])
    inputs = public.verify_public_fit_inputs(
        **producer.default_paths(),
        source_handoff_bytes=handoff,
        expected_source_handoff=PUBLIC_HANDOFF_SHA,
        expected_admission=public.FIXED_METADATA["admission-receipt.json"],
        expected_preflight=public.FIXED_METADATA["preflight-root-receipt.json"],
        tokenizer_cache=producer.CACHE,
        gate=gate,
    )
    owned_inputs = mixed_inputs = None
    if plan["regime"] == "mixed129":
        import jev_mixed_action_rehearsal as mixed
        import jev_owned_generic_collection as owned

        owned_blob = read_ref(plan["owned_admission_plan"])
        owned_plan = parse(owned_blob)
        index = parse(read_ref(owned_plan["proof_index"]))
        owned_inputs = owned.validate_owned_generic_collection(
            ROOT / relative(index["proof_root"]),
            plan_bytes=owned_blob,
            expected_plan_sha256=OWNED_PLAN_SHA,
            expected_closed_index_sha256=owned_plan["proof_index"]["sha256"],
            expected_producer_sha256=owned.FIXED[owned.PRODUCER],
            reviewed_source_pins=owned_plan["sources"],
            tokenizer_cache=producer.CACHE,
            gate=gate,
        )
        mix_blob = read_ref(plan["mixed_manifest"])
        mixed_inputs = mixed.admit_rehearsal(
            inputs,
            owned_inputs,
            mix_manifest_bytes=mix_blob,
            expected_mix_manifest_sha256=MIX_SHA[plan["mode"]],
            objective=public.objective(plan["mode"]),
            parent_pins=public.PARENT_PINS,
            source_closure=parse(mix_blob)["source_closure"],
        )
    plan_blob = encode(plan)
    desc_blob = encode(descriptor(plan, inputs, owned_inputs, mixed_inputs))
    result = StageInputs(
        plan_blob,
        inputs,
        owned_inputs,
        mixed_inputs,
        desc_blob,
        _PROOF,
        sha(plan_blob + b"\0" + desc_blob),
    )
    result.check()
    return result


def compile_inputs(model, inputs, output, gate):
    """Compile all142 public rows unchanged plus genuine14 owned target rows."""
    require(type(inputs) is StageInputs, "Stage admission type")
    inputs.check()
    public = public_api()
    output = Path(output)
    rows, _, plans, _ = public.compile_public_actions(
        model, inputs._public, output=output, gate=gate
    )
    original = b"".join(
        encode(
            {
                "case_id": row["record"]["case_id"],
                "split": split,
                "plan": plans[row["record"]["case_id"]].evidence(),
            }
        )
        + b"\n"
        for split in ("train", "development")
        for row in rows[split]
    )
    require(
        original == read_ref(parse(inputs._plan)["original_CFG142_plans"]),
        "All142 original CFG plans byteexact",
    )
    components = {"public": rows}
    if inputs._owned is not None:
        import jev_owned_generic_collection as owned

        require(
            type(inputs._owned) is owned.AdmittedOwnedGenericCollection,
            "Genuine owned14 typed cases",
        )
        kernel = public.trusted_module(
            "jev_browser_fit_actions", "scripts/jev_browser_fit_actions.py"
        )
        cfg = public.trusted_module("jev_cfg_action_loss", "scripts/jev_cfg_action_loss.py")
        components["owned"] = {}
        for split in ("train", "development"):
            cases = (
                inputs._owned.train_cases if split == "train" else inputs._owned.development_cases
            )
            projection = inputs._owned.compile_wire_projection(split)
            owned_rows, evidence = [], []
            for ordinal, (case, wire) in enumerate(zip(cases, projection, strict=True)):
                require(
                    type(case) is owned.OwnedNativeActionCase,
                    "Owned teacher never coerced to public",
                )
                gate.check(
                    "before-owned-native-target-compile-" + split + "-" + str(ordinal), "cpu"
                )
                actual = kernel.compile_actions(
                    model,
                    [case],
                    {"blobs": {"wire_requests.jsonl": encode(wire) + b"\n"}},
                    identity=public.IDENTITY,
                )
                require(
                    len(actual) == 1 and case.id not in plans,
                    "Unique genuine owned target compilation",
                )
                row = actual[0]
                plans[case.id] = cfg.compile_action_plan(row, wire["payload"], model.tok)
                evidence.append({"case_id": case.id, "plan": plans[case.id].evidence()})
                owned_rows.append(row)
            require(
                len(owned_rows) == {"train": 9, "development": 5}[split],
                "Exactly14 admitted owned target-appended rows",
            )
            components["owned"][split] = owned_rows
            write_blob(
                output / ("owned-compiled-" + split + ".jsonl"),
                b"".join(encode(public.serialized_row(row)) + b"\n" for row in owned_rows),
            )
            write_blob(
                output / ("owned-cfg-plans-" + split + ".jsonl"),
                b"".join(encode(r) + b"\n" for r in evidence),
            )
    row_map = {}
    for component, splits in components.items():
        for split, values in splits.items():
            for row in values:
                key = (component, row["record"]["case_id"])
                require(key not in row_map, "Unique compiled component identity")
                row_map[key] = row
    value = inputs.value()
    for split, refs in value["rows"].items():
        require(len(refs) == COUNTS[value["regime"]][split], "Full native split")
        for ref in refs:
            row = row_map[(ref["component"], ref["row_id"])]
            require(
                row["record"]["original_wire_sha256"] == ref["wire_sha256"],
                "Compiled row bound to admitted original wire",
            )
    inputs.check()
    return row_map, plans


def compiled_inventory(root):
    result = {}
    for path in sorted(Path(root).iterdir()):
        require(
            not path.is_symlink() and path.is_file() and path.suffix == ".jsonl",
            "Only flat compiled JSONL evidence files",
        )
        result[path.name] = memory.file_pin(path)
    return result


def verify_CPU_reference(ref, plan, *, compiled_root=None, descriptor_value=None):
    value = parse(read_ref(ref))
    require(
        type(value) is dict
        and value.get("schema_id") == COMPILE_SCHEMA
        and value.get("status") == "complete-CPU-reference-not-capacity-or-fit",
        "New completed genuine CPU reference required",
    )
    require(
        value["input_plan_sha256"] == sha(encode(plan))
        and value["source_union"] == plan["source_union"],
        "CPU source/data plan binding",
    )
    require(
        value["counts"] == COUNTS[plan["regime"]]
        and value["owned_target_appended_rows"] == (0 if plan["regime"] == "public120" else 14),
        "Complete native compilation population",
    )
    if descriptor_value is not None:
        require(
            value["admitted_inputs_sha256"] == sha(encode(descriptor_value)),
            "Fresh admission identical to CPU reference",
        )
    root = ROOT / relative(value["output_path"])
    expected_labels = {
        prefix + split + ".jsonl"
        for prefix in ("compiled-", "cfg-plans-")
        for split in ("train", "development")
    }
    if plan["regime"] == "mixed129":
        expected_labels |= {"owned-" + label for label in expected_labels}
    require(
        set(value["compiled_files"]) == expected_labels,
        "Exact complete compiled target/mask/CFG reference population",
    )
    require(
        compiled_inventory(root) == value["compiled_files"], "Closed CPU compiled bytes changed"
    )
    if compiled_root is not None:
        require(
            compiled_inventory(compiled_root) == value["compiled_files"],
            "Actual model compiler prefix/target/masks/CFG differs from independent CPU reference",
        )
    return value


def cpu_prepare(plan_ref, output):
    """Explicit future CPU operation; never constructs a model or touches CUDA."""
    output = memory.ordinary(output)
    require(
        output.is_relative_to(ROOT / "artifacts/jev-runtime") and not output.exists(),
        "Fresh owned CPU output",
    )
    output.mkdir(mode=0o700, parents=True)
    gate = CPUPrepareGate(output, memory.boot_id())
    gate.check("before-any-project-framework-or-parent-payload", "cpu")
    plan = check_plan(parse(read_ref(plan_ref)))
    import jev_browser_fit_public_cfg_actions as producer

    producer._fresh_environment()
    # Verify pinned installed sources/binaries before their first import. This
    # does not execute NativeTools, a compiler, a model, or an allocator probe.
    for label, ref in plan["runtime"].items():
        gate.check("before-CPU-runtime-provenance", "cpu")
        runtime = parse(read_ref(ref))
        files = runtime["files"]
        entries = (
            [{"path": path, "pin": item} for path, item in files.items()]
            if type(files) is dict
            else files
        )
        for spec in entries:
            require(memory.file_pin(spec["path"]) == spec["pin"], "Pinned CPU runtime provenance")
        if label == "native_toolchain":
            authority = runtime["record_authority"]
            require(
                memory.file_pin(authority["path"]) == authority["pin"],
                "Original native wheel RECORD authority",
            )
            memory.verify_python_entrypoint(
                runtime, runtime["python_entrypoint"]["native_file_target"]
            )
        if label == "optional_dependency":
            memory.verify_optional_dependency_files(runtime)
    inputs = admit_inputs(plan, gate)
    public = public_api()
    public.verify_reference_environment(inputs._public.reference_items)
    gate.check("before-native-CPU-processor", "cpu")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        public.MODEL_ID, revision=public.REVISION, local_files_only=True
    )
    reference_items = dict(inputs._public.reference_items)
    model = SimpleNamespace(
        processor=processor,
        tok=processor.tokenizer,
        max_context=8192,
        lm=SimpleNamespace(
            config=SimpleNamespace(**parse(reference_items["processor-metadata/config.json"])),
            generation_config=SimpleNamespace(eos_token_id=[1, 106, 50]),
        ),
    )
    public.verify_reference_environment(inputs._public.reference_items, model=model)
    compiled = output / "compiled"
    compiled.mkdir(mode=0o700)
    rows, _ = compile_inputs(model, inputs, compiled, gate)
    require(len(rows) == sum(COUNTS[plan["regime"]].values()), "Complete actual CPU compilation")
    inputs.check()
    receipt = {
        "schema_id": COMPILE_SCHEMA,
        "status": "complete-CPU-reference-not-capacity-or-fit",
        "input_plan_sha256": sha(encode(plan)),
        "input_plan": plan_ref,
        "source_union": plan["source_union"],
        "counts": COUNTS[plan["regime"]],
        "owned_target_appended_rows": 0 if plan["regime"] == "public120" else 14,
        "admitted_inputs_sha256": sha(encode(inputs.value())),
        "output_path": compiled.relative_to(ROOT).as_posix(),
        "compiled_files": compiled_inventory(compiled),
        "framework_model_constructors": 0,
        "GPU_calls": 0,
    }
    memory.write(output / "CPU-reference.json", receipt)
    return receipt


def expected_counts(regime):
    counts = COUNTS[regime]
    return {
        "updates": counts["train"],
        "forwards": sum(counts.values()),
        "backwards": counts["train"],
        "cache_releases": 2 * counts["train"] + counts["development"],
    }


def verify_worker_receipt(value, *, inputs_value, config):
    counts = expected_counts(inputs_value["regime"])
    require(
        type(value) is dict
        and value.get("schema_id") == memory.RECEIPT_SCHEMA
        and value.get("profile_id") == memory.PROFILE
        and value.get("allocator_policy") == memory.CACHE_POLICY,
        "New v2 worker receipt; old ART/Modal/probe receipts refused",
    )
    require(
        value["kind"] == inputs_value["kind"] == config["artifact_kind"]
        and value["mode"] == inputs_value["mode"] == config["mode"],
        "Exact new child kind and objective",
    )
    require(
        value["stage"] == "complete_stage_fit_unqualified"
        and value["qualified"] is False
        and value["error"] is None,
        "Complete unqualified local fit only",
    )
    for field_name in (
        "completed_updates",
        "synchronized_updates",
        "optimizer_step_attempts",
        "actual_native_backward_calls",
    ):
        require(
            type(value[field_name]) is int and value[field_name] == counts["updates"],
            "Complete returned and synchronized native update counts",
        )
    require(
        value["actual_native_forward_calls"] == counts["forwards"]
        and value["development_updates"] == 0
        and value["actual_cache_releases"] == counts["cache_releases"],
        "All native train/dev calls, zero development updates and fixed cache policy",
    )
    require(
        value["model_loads"]
        == value["verified_model_loads"]
        == value["model_constructor_attempts"]
        == 1,
        "Exactly one verified original model",
    )
    require(
        value["actual_update_refs"] == inputs_value["ordered_updates"]
        and value["admitted_inputs_sha256"] == sha(encode(inputs_value)),
        "All updates in admitted order, no dev/repeats/truncation",
    )
    require(
        value["head_sha256"] == HEAD_SHA
        and value["tensor_metadata"]["tensor_count"] == 176
        and value["tensor_metadata"]["parameters"] == 5193728,
        "Full frozen head and exact FP32 q/v adapter",
    )
    require(
        value["saved_adapter"]["status"] == "verified-complete"
        and value["source_union"] == config["source_inventory"],
        "Guarded complete save and new source union",
    )
    return counts


def verify_closure(arm, worker, config):
    require(
        arm.get("schema_id") == "jev-stage-memory-arm-receipt-v2"
        and arm.get("profile_id") == memory.PROFILE
        and arm.get("success") is True
        and arm.get("closure_confirmed") is True,
        "Complete original owned-unit whole closure required",
    )
    require(
        arm["worker_receipt"] == worker
        and arm["invocation"] == worker["unit_binding"]["InvocationID"],
        "Same actual owned invocation/worker",
    )
    require(
        arm["post_stop"]["MainPID"] == "0"
        and arm["closure_kind"] in {"confirmed-unit-GC", "retained-unit"},
        "Closed owned process and unit",
    )
    require(
        arm.get("retained_lifetime_peaks") is not None
        and worker.get("native_work_end_capture") is not None,
        "Native-end ACK and retained lifetime peaks required",
    )
    require(
        worker["unit_binding"]["ControlGroup"].endswith("/" + arm["unit"]),
        "Exact owned cgroup binding",
    )
    require(
        worker["source_union"] == config["source_inventory"], "Closure bound to source inventory"
    )


def verify_saved_adapter(root, worker, gate):
    public = public_api()
    directory = Path(root) / "adapter"
    require(
        {p.name for p in directory.iterdir()}
        == {"adapter_config.json", "adapter_model.safetensors"},
        "Only complete saved adapter files; partial/marker rejected",
    )
    files = {p.name: memory.file_pin(p) for p in directory.iterdir()}
    require(files == worker["saved_adapter"]["files"], "Saved tensor/config bytes drift")
    metadata = public.tensor_metadata(
        memory.read(directory / "adapter_model.safetensors"),
        worker["tensor_metadata"]["tensor_shapes"],
        gate=gate,
    )
    require(metadata == worker["tensor_metadata"], "Fresh finite176 FP32 tensor verification")
    config = parse(memory.read(directory / "adapter_config.json"))
    require(
        config["base_model_name_or_path"] == public.MODEL_ID
        and config["revision"] == public.REVISION
        and all(config[key] == public.LORA[key] for key in public.LORA),
        "Original saved q/v LoRA config",
    )
    return files


def json_rows(path):
    return [parse(line) for line in memory.read(path).splitlines()]


def validate_native_evidence(root, *, value, worker, arm, config):
    """Independently join saved native calls, original gates, ACK and peaks."""
    root = Path(root)
    counts = expected_counts(value["regime"])
    ready = parse(memory.read(root / "ready-for-readback.json"))
    memory.verify_ready(ready, config)
    identity = memory.native_work_identity(ready, worker["unit_binding"])
    capture_blob = memory.read(root / "native-work-final-cgroup.json")
    capture = parse(capture_blob)
    memory.validate_native_work_capture(capture, identity)
    ack = parse(memory.read(root / "native-work-ack.json"))
    require(
        ack == {"schema_id": "native-work-ack-v1", **identity, "snapshot_sha256": sha(capture_blob)}
        and worker["native_work_end_capture"]["snapshot_sha256"] == sha(capture_blob),
        "Exact saved native-end ACK, live capture and same worker",
    )
    terminal = parse(memory.read(root / "unit-terminal.json"))
    require(
        memory.retained_lifetime_peaks(terminal["properties"], identity, capture)
        == arm["retained_lifetime_peaks"],
        "Original retained lifetime peak/zero swap gate",
    )
    require(
        memory.closed_unit_metadata(arm["post_stop"], arm["invocation"], stopped=True)
        == arm["closure_kind"],
        "Typed retained or authenticated GC closure",
    )
    startup = json_rows(root / "original-startup-gates.jsonl")
    for i, row in enumerate(startup):
        require(
            row["sequence"] == i
            and row["admitted"] is True
            and row["error"] is None
            and row["swap_or_resident_credit_bytes"] == 0
            and row["expected_boot_id"] == config["boot_id"] == row["observation"]["boot_id"],
            "Every original startup gate admitted with no residency credit",
        )
        reserve = max(24 * memory.GIB, (row["observation"]["MemTotal"] + 4) // 5)
        load = 2 * 23919549408 + reserve
        required = {
            "cpu": reserve + 4 * memory.GIB,
            "load": load,
            "train": load + 30 * memory.GIB + 256 * memory.MIB,
        }[row["scope"]]
        require(
            row["required_bytes"] == required <= row["observation"]["MemAvailable"],
            "Original CPU/load/train thresholds preserved",
        )
    for required_phase in (
        "before-any-project-framework-or-parent-payload",
        "before-public-project-admission",
        "immediately-before-legacy-parent-capture-and-Torch",
        "before-checkpoint-payload-hash-and-BF16-loader",
        "before-any-BF16-model-construction",
    ):
        require(
            any(row["phase"] == required_phase for row in startup),
            "Required original startup gate missing",
        )
    phases = json_rows(root / "phases.jsonl")
    intents, complete = [], []
    for ordinal, row in enumerate(phases):
        require(
            row["schema_id"] == memory.PHASE_SCHEMA and row["profile_id"] == memory.PROFILE,
            "Only new v2 resident phase evidence",
        )
        if row["event"] == "stage-gate":
            floor = memory.research_floor(
                row["observation"],
                startup=row["phase"]
                in {"full-original-checkpoint-hash", "full-original-constructor"},
            )
            require(
                row["admitted"] is True
                and row["resident_credit_bytes"] == 0
                and row["required_bytes"]
                == floor["required_bytes"]
                <= row["observation"]["MemAvailable"],
                "Original independent stage host floor",
            )
        else:
            require(row["event"] in {"intent", "complete"}, "No failed/unknown native phase")
            target = intents if row["event"] == "intent" else complete
            require(row["sequence"] == len(target), "Contiguous native phase sequence")
            if row["event"] == "intent":
                require(
                    ordinal > 0
                    and phases[ordinal - 1]["event"] == "stage-gate"
                    and phases[ordinal - 1]["phase"] == row["phase"],
                    "Every native intent follows its original successful entry guard",
                )
            target.append(row)
            cuda = row["cuda"]
            require(
                cuda["backend"] == "native"
                and cuda["fraction"] == worker["allocator"]["fraction"]
                and cuda["reserved"] < memory.STOP,
                "Original allocator backend/fraction/stop",
            )
            cg = row["cgroup"]
            require(
                cg["max"] == memory.CAP
                and cg["swap_max"] == cg["swap"] == 0
                and cg["high"] is None
                and cg["oom_group"] == 1
                and all(cg["events"].get(k, 0) == 0 for k in ("max", "oom", "oom_kill")),
                "Exact owned 40GiB/no-swap/no-pressure phase evidence",
            )
            require(
                cuda["stats"].get("num_ooms", 0) == cuda["stats"].get("num_alloc_retries", 0) == 0,
                "No native OOM or allocator retry",
            )
    require(
        len(intents) == len(complete)
        and all(
            a["phase"] == b["phase"] and b["error"] is None and not b["secondary_errors"]
            for a, b in zip(intents, complete, strict=True)
        ),
        "Every protected native intent completed after its CUDA fence",
    )
    names = [r["phase"] for r in complete]
    phase_counts = {
        "full-original-checkpoint-hash": 1,
        "full-original-constructor": 1,
        "original-model-head-check": 1,
        "original-parent-injection": 1,
        "complete-original-plus-owned-processor-compile": 1,
        "postcompile-original-model-check": 1,
        "optimizer-create-empty": 1,
        "zero-grad-before-forward": counts["updates"],
        "native-train-forward": counts["updates"],
        "native-backward": counts["updates"],
        "native-grad-clip": counts["updates"],
        "native-Adam-step": counts["updates"],
        "native-development-forward": COUNTS[value["regime"]]["development"],
        "final-full-head-hash": 1,
        "final-original-model-check": 1,
        "final-original-parent-check": 1,
        "guarded-save-adapter": 1,
        "guarded-saved-tensor-validation": 1,
        "guarded-final-source-data-validation": 1,
    }
    require(set(names) == set(phase_counts), "No undeclared native phase/retry")
    for name, count in phase_counts.items():
        require(names.count(name) == count, "Exact complete native phase count: " + name)
    trace = parse(memory.read(root / "native-call-trace.json"))
    require(
        trace["decoder_architecture"]["decoder_layers"] == 48
        and trace["decoder_architecture"] == worker["decoder_architecture"],
        "All original48 decoder layers",
    )
    actual = trace["native_forward_calls"]
    expected = [("forward", r["row_id"]) for r in value["ordered_updates"]] + [
        ("development", r["row_id"]) for r in value["rows"]["development"]
    ]
    require(
        [(r["phase"], r["case_id"]) for r in actual] == expected
        and all(r["use_cache"] is False for r in actual),
        "Every saved full native forward bound to admitted train/dev order",
    )
    for phase, count in (
        ("forward", counts["updates"]),
        ("development", COUNTS[value["regime"]]["development"]),
        ("backward", counts["backwards"]),
    ):
        require(
            all(trace["layers"].get(phase + ":" + str(i)) == count for i in range(48)),
            "Every original48 decoder layer visited in each native call",
        )
    operators = json_rows(root / "operators.jsonl")
    require(
        len(operators) == counts["forwards"] + counts["backwards"],
        "Original native SDPA forward/backward operator receipts",
    )
    expected_operator_phases = [
        phase for _ in value["ordered_updates"] for phase in ("forward", "backward")
    ] + ["development"] * COUNTS[value["regime"]]["development"]
    require(
        [r["phase"] for r in operators] == expected_operator_phases,
        "Native operator calls in exact original order",
    )
    for row in operators:
        require(
            any(
                op["name"] == "aten::scaled_dot_product_attention" and op["count"] == 48
                for op in row["operators"]
            ),
            "All48 original SDPA calls including nonreentrant backward recomputation",
        )
        if row["phase"] == "backward":
            require(
                any(
                    "attention_backward" in op["name"] and op["count"] == 48
                    for op in row["operators"]
                ),
                "All48 original attention backward calls",
            )
    updates, development = json_rows(root / "updates.jsonl"), json_rows(root / "development.jsonl")
    require(
        [r["case_id"] for r in updates] == [r["row_id"] for r in value["ordered_updates"]]
        and [r["step"] for r in updates] == list(range(1, counts["updates"] + 1)),
        "No missing/repeated/development updates",
    )
    require(
        [r["case_id"] for r in development] == [r["row_id"] for r in value["rows"]["development"]],
        "Complete independent development pass",
    )
    compiled, plans = {}, {}
    for path in sorted((root / "compiled").iterdir()):
        target = plans if "cfg-plans-" in path.name else compiled
        for row in json_rows(path):
            case_id = row["case_id"] if target is plans else row["record"]["case_id"]
            require(case_id not in target, "Unique actual saved native row/CFG plan")
            target[case_id] = row["plan"] if target is plans else row
    require(
        set(compiled)
        == set(plans)
        == {r["row_id"] for rows in value["rows"].values() for r in rows},
        "Complete original and owned compiled/CFG identity join",
    )
    for row in actual:
        saved = compiled[row["case_id"]]
        require(
            row["input_shape"] == [1, saved["record"]["input_tokens"]]
            and row["positions"] == saved["positions"],
            "Actual native forward used full compiled sequence and all original readout positions",
        )
    public = public_api()
    for row in updates + development:
        public.validate_loss_metrics(
            {**row["metrics"], "loss": row["loss"]},
            SimpleNamespace(**plans[row["case_id"]]),
            value["objective"],
        )
    releases = json_rows(root / "cache-releases.jsonl")
    require(
        len(releases) == 2 * counts["cache_releases"], "All fixed cache release intents/completions"
    )
    expected_releases = [
        (r["row_id"], i, phase)
        for i, r in enumerate(value["ordered_updates"], 1)
        for phase in ("train-forward", "backward")
    ] + [(r["row_id"], counts["updates"], "dev-forward") for r in value["rows"]["development"]]
    for i, wanted in enumerate(expected_releases):
        before, after = releases[2 * i : 2 * i + 2]
        require(
            before["event"] == "intent"
            and after["event"] == "complete"
            and after["success"] is True
            and after["API_returned"] is True
            and after["error"] is None
            and (after["case_id"], after["step"], after["native_phase"]) == wanted,
            "Fixed cache policy never retries/reorders/skips native calls",
        )
        require(
            before["before"] == after["before"]
            and after["policy"] == memory.CACHE_POLICY
            and after["before"]["storage_identity"] == after["after"]["storage_identity"]
            and after["before"]["cuda"]["allocated"] == after["after"]["cuda"]["allocated"],
            "Occupied tensors and storage identities preserved",
        )
        require(
            all(
                (row["case_id"], row["step"], row["native_phase"]) == wanted
                and row["mode"] == value["mode"]
                and row["policy"] == memory.CACHE_POLICY
                for row in (before, after)
            ),
            "Exact cache intent/completion identity join",
        )
        require(
            all(
                after["after"]["cuda"][key] >= after["before"]["cuda"][key]
                for key in ("peak_allocated", "peak_reserved")
            ),
            "Cache release retained high water marks",
        )
    return {
        "native_updates": len(updates),
        "native_forwards": len(actual),
        "native_backwards": counts["backwards"],
        "cache_releases": counts["cache_releases"],
        "phase_completions": len(complete),
    }


def guarded_save_adapter(model, inputs, output, gate, kernel, state):
    """A failed/partial write never becomes a completed candidate marker."""
    inputs.check()
    directory = Path(output) / "adapter"
    require(not directory.exists(), "Fresh adapter destination")
    memory.arm_bound(Path(output), pending=24 * memory.MIB)
    state["saved_adapter"] = {"status": "intent", "files": None}
    memory.replace(Path(output) / "progress.json", state)
    try:
        shapes = {
            name.replace(".lora_A.default.", ".lora_A.").replace(
                ".lora_B.default.", ".lora_B."
            ): list(value.shape)
            for name, value in model.lm.named_parameters()
            if value.requires_grad
        }
        state["saved_adapter"]["files"] = gate.phase(
            "guarded-save-adapter", lambda: kernel._save_adapter(model, directory)
        )
        state["tensor_metadata"] = gate.phase(
            "guarded-saved-tensor-validation",
            lambda: public_api().tensor_metadata(
                memory.read(directory / "adapter_model.safetensors"), shapes, gate=gate
            ),
        )
        state["saved_adapter"]["status"] = "verified-complete"
        gate.phase("guarded-final-source-data-validation", inputs.check)
        verify_saved_adapter(output, state, gate)
    except BaseException as exc:
        state["saved_adapter"].update(
            status="failed-or-partial", error={"type": type(exc).__name__, "message": str(exc)}
        )
        raise
    finally:
        memory.replace(Path(output) / "progress.json", state)
    return state["saved_adapter"]


def seal_closed_child(root, *, config, plan, CPU_ref, gate):
    """Controller-only sealing after native-end ACK and whole owned closure."""
    root = memory.ordinary(root)
    gate.check("before-new-stage-child-tree-or-tensor-payload", "cpu")
    require(not (root / "child-manifest.json").exists(), "No pre-existing valid child marker")
    inputs = admit_inputs(plan, gate)
    value = inputs.value()
    worker = parse(memory.read(root / "worker-receipt.json"))
    arm = parse(memory.read(root / "arm-receipt.json"))
    verify_worker_receipt(worker, inputs_value=value, config=config)
    verify_closure(arm, worker, config)
    validate_native_evidence(root, value=value, worker=worker, arm=arm, config=config)
    verify_CPU_reference(CPU_ref, plan, compiled_root=root / "compiled", descriptor_value=value)
    files = verify_saved_adapter(root, worker, gate)
    inputs.check()
    check_source_union(plan)
    inventory = memory.tree_inventory(root)
    require(
        sum(p["size_bytes"] for p in inventory.values()) + 2 * memory.MIB <= memory.ARM_CAP,
        "Final arm byte budget including marker",
    )
    manifest = {
        "schema_id": MANIFEST_SCHEMA,
        "kind": value["kind"],
        "status": "complete-local-stage-fit-unqualified",
        "qualified": False,
        "profile_id": memory.PROFILE,
        "allocator_policy": memory.CACHE_POLICY,
        "input_plan_sha256": sha(encode(plan)),
        "admitted_inputs": value,
        "CPU_reference": CPU_ref,
        "config_sha256": sha(memory.read(root / "config.json")),
        "source_union": config["source_inventory"],
        "worker_receipt": memory.file_pin(root / "worker-receipt.json"),
        "arm_receipt": memory.file_pin(root / "arm-receipt.json"),
        "tensor_metadata": worker["tensor_metadata"],
        "adapter_files": files,
        "inventory_excluding_manifest": inventory,
        "qualification_required": "Separate new local BF16 native qualification and held-out browser task evaluation; old ART95/Modal90 cannot certify this kind",
    }
    memory.write(root / "child-manifest.json", manifest)
    return manifest


def verify_completed_child(root, *, expected_manifest_sha256, plan, gate):
    root = memory.ordinary(root)
    gate.check("before-new-stage-child-tree-or-tensor-payload", "cpu")
    blob = memory.read(root / "child-manifest.json")
    require(sha(blob) == expected_manifest_sha256, "Externally selected new child manifest")
    manifest = parse(blob)
    require(
        manifest.get("schema_id") == MANIFEST_SCHEMA
        and manifest.get("kind") == KINDS[plan["regime"]]
        and manifest.get("status") == "complete-local-stage-fit-unqualified"
        and manifest.get("qualified") is False,
        "New closed stage child only; old ART95/Modal90/probes refused",
    )
    require(manifest["input_plan_sha256"] == sha(encode(plan)), "Child input plan binding")
    inventory = memory.tree_inventory(root)
    require(
        {k: v for k, v in inventory.items() if k != "child-manifest.json"}
        == manifest["inventory_excluding_manifest"],
        "Complete sealed child inventory, no additional or changed files",
    )
    inputs = admit_inputs(plan, gate)
    require(
        inputs.value() == manifest["admitted_inputs"],
        "Current source/data identity matches completed child",
    )
    config = parse(memory.read(root / "config.json"))
    memory.validate_stage_config(config, root)
    require(
        sha(memory.read(root / "config.json")) == manifest["config_sha256"],
        "Exact child config bytes",
    )
    worker = parse(memory.read(root / "worker-receipt.json"))
    arm = parse(memory.read(root / "arm-receipt.json"))
    require(
        memory.file_pin(root / "worker-receipt.json") == manifest["worker_receipt"]
        and memory.file_pin(root / "arm-receipt.json") == manifest["arm_receipt"],
        "Closed receipts exact bytes",
    )
    verify_worker_receipt(worker, inputs_value=inputs.value(), config=config)
    verify_closure(arm, worker, config)
    validate_native_evidence(root, value=inputs.value(), worker=worker, arm=arm, config=config)
    verify_CPU_reference(
        manifest["CPU_reference"],
        plan,
        compiled_root=root / "compiled",
        descriptor_value=inputs.value(),
    )
    require(
        verify_saved_adapter(root, worker, gate) == manifest["adapter_files"]
        and worker["tensor_metadata"] == manifest["tensor_metadata"],
        "Complete tensor/source/data child validation",
    )
    inputs.check()
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-prepare", action="store_true")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    require(args.cpu_prepare and args.run, "Explicit reviewed CPU preparation required")
    ref = reference(args.plan)
    require(ref["sha256"] == args.plan_sha256, "Externally selected CPU plan SHA")
    cpu_prepare(ref, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
