"""HELD four-call, nontraining CPU5 operation-readout numerical qualification.

Only a future exact owned selection may release a new copy of this source.
This authoring candidate has not imported, checked or executed its dependencies.
V2 compares target response bits against the original publisher's same-forward
respond formatting, retaining its normalization without changing tolerances.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import stat
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE_ONLY = False
PROFILE = "decision3-operation-four-call-numeric-cpu5-4g-nontraining-v1"
NATIVE = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261024-05"
NP02 = ROOT / "artifacts/jev-runtime/decision3-readout-numpy-cpu5-qualification-20261024-02"
BACKEND = (
    ROOT
    / "artifacts/jev-runtime/decision3-operation-head-backend-cpu5-20261024-01/backend_released_v1.py"
)
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
DSL = ROOT / "artifacts/jev-runtime/browser-decision-dsl-20261023-01"
DEPENDENCIES = {
    "native_actor": NATIVE / "actor.py",
    "native_plan": NATIVE / "edge-plan.json",
    "native_helper": NATIVE / "resource_helper.py",
    "np_actor": NP02 / "actor.py",
    "observer": NP02 / "observer.py",
    "numpy_probe": NP02 / "numpy_probe.py",
    "backend": BACKEND,
}
PINS = {
    "native_actor": "9a7e5af918c17f6824a720fb6d112ffa5fdac03493dde3efa176c7d50470a365",
    "native_plan": "095db88b78687d90fbed51aafd736c1b06fb1a582312de8dff97bf6109131e96",
    "native_helper": "4df54e06e8a795ebf5b300a3e2074319e781a278a2a870c21c2bd52897d7eb11",
    "np_actor": "c8a439c794f5176564a29473c524d62f86218818b1f7fb66089c8e007dfd7f82",
    "observer": "455e2c7bf20d1a96f99795566062db15dbe779dd5610e573277c382775bc4f37",
    "numpy_probe": "22ed700ebc9ea49fd8cf4d1a532ca3a571290d349b69053c6bea082631582f7f",
    "backend": "8f356fac4e4291cd2dbb95efa0a5766cd695998a70d864b7d30e1555ae17c7b8",
}
HEADS = ("click_target", "operation", "select_target", "type_text_target")
COUNTS = (1, 8, 2, 3)
CALLS = (("original", 4), ("candidate", 4), ("original", 2), ("candidate", 2))
WALL_SECONDS, CALL_SECONDS = 120, 20
ABS_TOL, REL_TOL = 1e-6, 1e-5
PENDING = "PENDING_NONTRAINING"
MODEL_MODULES = {"torch", "transformers", "safetensors", "numpy", "s1"}
OWNER_CONTRACT = {
    "unit": "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service",
    "profile": "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1",
    "cpu_affinity": [15, 16, 17, 18, 19],
    "cpu_quota_percent": 500,
    "threads": 5,
    "interop_threads": 1,
    "memory_max_bytes": 4294967296,
    "swap_max_bytes": 0,
    "reserve_bytes": 26119172916,
    "startup_available_bytes": 30414140212,
    "owner_runtime_seconds": 10800,
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def pin(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def release_gate():
    require(SOURCE_ONLY is False, "SOURCE_ONLY: numerical qualification is held")
    require(all(pin(value) for value in PINS.values()), "final selected source pins remain pending")


def ordinary(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    require(path.is_relative_to(ROOT) and ".." not in path.parts, "repository path required")
    require(not path.is_relative_to(DSL), "DSL data and authoring are outside this qualification")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink path refused")
    return path


def reference(record):
    require(
        type(record) is dict and {"path", "sha256", "size_bytes"} <= set(record),
        "selected file reference required",
    )
    require(
        pin(record["sha256"])
        and type(record["size_bytes"]) is int
        and 0 < record["size_bytes"] <= 4 * 1024**2,
        "bounded final file pin required",
    )
    return {
        "path": str(ordinary(record["path"])),
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
    }


def read_ref(record, expected_path=None):
    record = reference(record)
    path = Path(record["path"])
    require(expected_path is None or path == expected_path, "selected dependency path differs")
    with path.open("rb") as stream:
        first = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(first.st_mode) and first.st_size == record["size_bytes"],
            "regular exact-size selected file required",
        )
        raw = stream.read(record["size_bytes"] + 1)
        last = os.fstat(stream.fileno())
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(
        len(raw) == record["size_bytes"]
        and all(
            getattr(first, key) == getattr(last, key) == getattr(path.stat(), key) for key in keys
        )
        and hashlib.sha256(raw).hexdigest() == record["sha256"],
        "selected file bytes/identity changed",
    )
    return raw


def source(name, record):
    raw = read_ref(record)
    path = Path(reference(record)["path"])
    require(
        path.suffix == ".py" and len(raw) < 100_000 and name not in sys.modules,
        "fresh bounded Python source required",
    )
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "source loader unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), vars(module))
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def retain_json(helper, path, value):
    helper.write_json(path, value)
    fsync_directory(path.parent)


def finalize_state(helper, output, state, primary):
    try:
        helper.finalize_state(output, state, primary)
        fsync_directory(output)
        fsync_directory(output.parent)
    except BaseException as error:
        if primary is None:
            raise
        primary.add_note("State durability also failed: " + repr(error))


def projection(request, rows):
    require(type(rows) is int and rows in (4, 2), "only original four/two-row projections")
    require(list(request["questions"]) == list(HEADS), "original fixture order differs")
    return {
        "state": copy.deepcopy(request["state"]),
        "questions": {key: copy.deepcopy(request["questions"][key]) for key in HEADS[:rows]},
    }


def fixture_request():
    """Invented numerical input only: no desired answer, DOM, teacher or score.

    The operation is deliberately second; the genuine two-row tail keeps it.
    These source literals are independent of DSL and browser task artifacts.
    """
    return {
        "state": "A fictional laboratory index lists sample folders, a date range and a note box. No action is requested; this input compares numerical projections only.",
        "questions": {
            "click_target": {
                "type": "choice",
                "criteria": {"folder-slate": "Folder Slate"},
            },
            "operation": {
                "type": "choice",
                "criteria": {
                    "CLICK": "Click a control",
                    "TYPE_TEXT": "Type text into a field",
                    "SELECT": "Select an offered option",
                    "SCROLL_UP": "Scroll upward",
                    "SCROLL_DOWN": "Scroll downward",
                    "WAIT": "Wait for a page update",
                    "DONE": "The requested work is complete",
                    "BLOCKED": "The requested work cannot continue",
                },
            },
            "select_target": {
                "type": "choice",
                "criteria": {"range-early": "Early dates", "range-late": "Late dates"},
            },
            "type_text_target": {
                "type": "choice",
                "criteria": {
                    "note-header": "Note heading",
                    "note-body": "Note body",
                    "note-footer": "Note footer",
                },
            },
        },
    }


def authenticate_sources(plan, plan_path, expected):
    require(
        pin(expected) and hashlib.sha256(ordinary(plan_path).read_bytes()).hexdigest() == expected,
        "selected numerical plan changed",
    )
    read_ref(plan["actor_source"], Path(__file__).absolute())
    require(
        set(plan["dependencies"]) == set(DEPENDENCIES), "exact executed dependency set required"
    )
    for key, path in DEPENDENCIES.items():
        require(
            reference(plan["dependencies"][key])["sha256"] == PINS[key],
            "fixed dependency pin differs",
        )
        read_ref(plan["dependencies"][key], path)


def load_plan(path, expected):
    path = ordinary(path)
    raw = path.read_bytes()
    require(
        len(raw) <= 100_000 and pin(expected) and hashlib.sha256(raw).hexdigest() == expected,
        "bounded exact numerical plan required",
    )
    plan = json.loads(raw)
    require(
        plan.get("schema") == "decision3-operation-four-call-numeric-plan-v1"
        and plan.get("profile") == PROFILE
        and plan.get("executable") is True,
        "pending or other plan refused",
    )
    require(
        plan.get("owner") == OWNER_CONTRACT
        and plan.get("call_order") == [list(call) for call in CALLS]
        and plan.get("wall_seconds") == 120
        and plan.get("call_seconds") == 20,
        "fixed CPU5/four-call/120/20 contract required",
    )
    require(
        plan.get("absolute_tolerance") == ABS_TOL
        and plan.get("relative_tolerance") == REL_TOL
        and plan.get("training_eligible") is False
        and plan.get("semantic_expected_answers") is None
        and plan.get("ordered_head_ids") == list(HEADS)
        and plan.get("legal_counts") == list(COUNTS)
        and plan.get("dev_used_for_model_selection") is True
        and plan.get("optimizer_input_dev") is False
        and plan.get("optimizer_input_sealed") is False
        and plan.get("candidate_selection_before_dev_or_sealed") is False,
        "fixed nontraining numerical scope required",
    )
    authenticate_sources(plan, path, expected)
    for key in ("capability_ref", "root_acceptance_ref", "delta_ref"):
        reference(plan[key])  # Bytes are read only in the admitted candidate call20.
    request = fixture_request()
    require(
        list(request["questions"]) == list(HEADS)
        and [len(q["criteria"]) for q in request["questions"].values()] == list(COUNTS)
        and all(q["type"] == "choice" for q in request["questions"].values()),
        "source-defined operation numerical fixture differs",
    )
    return plan, request


def byte_hash(torch, tensor):
    require(
        tensor.device.type == "cpu" and tensor.dtype == torch.float32,
        "CPU FP32 byte witness required",
    )
    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()


def unchanged(shared, observer, runtime, torch):
    require(
        observer.baseline_identity(runtime) == shared["baseline_identity"]
        and torch.equal(runtime.readout.detach(), shared["baseline_values"])
        and torch.equal(
            runtime.readout.detach().contiguous().view(torch.uint8), shared["baseline_bits"]
        )
        and byte_hash(torch, runtime.readout) == shared["baseline_sha256"],
        "original baseline metadata/FP32 values/bit pattern changed",
    )


def numerical_context(shared, probe, native, helper, observer, runtime, torch, state):
    if not shared:
        # After the original genuine loader; never preimport numerical modules.
        np = probe.load_numpy(native, helper, state)
        shared.update(
            np=np,
            baseline_identity=observer.baseline_identity(runtime),
            baseline_values=runtime.readout.detach().clone(),
            baseline_bits=runtime.readout.detach().contiguous().view(torch.uint8).clone(),
            baseline_sha256=byte_hash(torch, runtime.readout),
        )
    return shared["np"]


def validate_training_acceptance(acceptance):
    require(
        type(acceptance) is dict
        and acceptance.get("schema") == "root-accepted-dsl01-operation-trained-readout-v1"
        and acceptance.get("status") == "accepted",
        "new root-selected closed operation-training acceptance required",
    )


def accepted_delta(shared, plan, backend, np, runtime, torch):
    if "delta" not in shared:
        capability = json.loads(read_ref(plan["capability_ref"]))
        backend.validate_capability(capability, plan["model"])
        require(
            reference(capability["delta_ref"]) == reference(plan["delta_ref"])
            and reference(capability["root_acceptance_ref"])
            == reference(plan["root_acceptance_ref"])
            and capability["baseline_fp32_bytes_sha256"] == shared["baseline_sha256"],
            "true selected delta/acceptance/baseline joins differ",
        )
        acceptance = json.loads(read_ref(plan["root_acceptance_ref"]))
        validate_training_acceptance(acceptance)
        # This exact root acceptance is a trusted selected authority, not an
        # independent re-audit of its producer performed by this numeric actor.
        array = np.load(io.BytesIO(read_ref(plan["delta_ref"])), allow_pickle=False)
        require(
            type(array) is np.ndarray
            and array.dtype == np.float32
            and array.shape == (255, 1024)
            and array.flags.c_contiguous
            and bool(np.isfinite(array).all())
            and bool((array != 0).any()),
            "real finite nonzero trained FP32 delta required; zero control is separate",
        )
        delta = torch.from_numpy(array.copy(order="C")).clone()
        require(
            delta.requires_grad is False
            and delta.is_contiguous()
            and byte_hash(torch, delta) == hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            "exact frozen artifact-to-Torch delta copy required",
        )
        shared.update(capability=capability, delta=delta, delta_sha256=byte_hash(torch, delta))
    return shared["capability"], shared["delta"]


def same_forward_replay(torch, features, baseline, delta, counts, ordered_head_ids):
    """Independent explicit FP32 formula, never another encoder invocation."""
    require(
        type(ordered_head_ids) is list
        and len(ordered_head_ids) == len(counts)
        and len(set(ordered_head_ids)) == len(ordered_head_ids)
        and all(key in HEADS for key in ordered_head_ids)
        and all(type(n) is int and 1 <= n <= 255 for n in counts),
        "complete actual ordered head IDs/counts required for independent replay",
    )
    with torch.inference_mode():
        base = features.float() @ baseline.T
        added = torch.zeros_like(base)
        combined = base.clone()
        for index, key in enumerate(ordered_head_ids):
            if key == "operation":
                update = features[index : index + 1].float() @ delta.T
                added[index : index + 1] = update
                combined[index : index + 1] = base[index : index + 1] + update
        limit = torch.as_tensor(list(counts), device="cpu")[:, None]
        invalid = torch.arange(255, device="cpu")[None] >= limit
        require(
            bool(torch.isfinite(base.masked_fill(invalid, 0)).all().item())
            and bool(torch.isfinite(added).all().item()),
            "finite reconstructed legal FP32 logits required",
        )
        masked_base = base.masked_fill(invalid, float("-inf"))
        combined = combined.masked_fill(invalid, float("-inf"))
        preserved = [i for i, key in enumerate(ordered_head_ids) if key != "operation"]
        require(
            all(
                torch.equal(
                    combined[i : i + 1].contiguous().view(torch.uint8),
                    masked_base[i : i + 1].contiguous().view(torch.uint8),
                )
                for i in preserved
            ),
            "independent replay target full-logit bits changed",
        )
        probabilities = torch.softmax(combined / 1.0, dim=-1).cpu().tolist()
        native_probabilities = torch.softmax(masked_base / 1.0, dim=-1).cpu().tolist()
        require(
            all(all(v == 0 for v in row[n:]) for row, n in zip(probabilities, counts))
            and all(probabilities[i] == native_probabilities[i] for i in preserved),
            "illegal reconstructed code mass or changed native target probabilities",
        )
        projection_logits = {
            "ordered_head_ids": list(ordered_head_ids),
            "target_full_logits_bitwise_unchanged": True,
            "target_probabilities_unchanged": True,
            "base_legal_logits": [row[:n] for row, n in zip(base.cpu().tolist(), counts)],
            "delta_legal_logits": [row[:n] for row, n in zip(added.cpu().tolist(), counts)],
            "adapted_legal_logits": [row[:n] for row, n in zip(combined.cpu().tolist(), counts)],
        }
    return (
        masked_base,
        [row[:n] for row, n in zip(probabilities, counts)],
        projection_logits,
    )


def native_response_control(runtime, prepared, keys, lengths, original_replay):
    """Use only the original same-forward publisher response formatting.

    Its normalized Choice values need not have the raw replay float bits. This
    does not encode or call logits, change options, or repair either response.
    """
    require(
        prepared.keys == keys and list(lengths) == keys and len(original_replay) == len(keys),
        "complete same-call original response control inputs required",
    )
    control = runtime.respond(prepared, dict(zip(keys, original_replay)), sum(lengths.values()))
    require(
        list(control["answers"]) == keys
        and all(
            list(control["answers"][key]["probabilities"]) == list(prepared.questions[key].keys)
            for key in keys
        ),
        "original publisher response control order differs",
    )
    return control, [list(control["answers"][key]["probabilities"].values()) for key in keys]


def target_preservation(raw, native_values, keys, counts, projection_logits):
    """Inspect actual legal logits against original publisher-formatted control.

    FP32 packing retains signed zero; probabilities retain their returned
    Python-float bits. No tolerance may hide a changed non-operation row here.
    """
    require(
        len(keys) == len(counts) == len(native_values)
        and projection_logits.get("ordered_head_ids") == keys,
        "complete ordered target preservation evidence required",
    )
    preserved = []
    for i, (key, count) in enumerate(zip(keys, counts)):
        if key == "operation":
            continue
        original = projection_logits["base_legal_logits"][i]
        adapted = projection_logits["adapted_legal_logits"][i]
        added = projection_logits["delta_legal_logits"][i]
        actual = list(raw["answers"][key]["probabilities"].values())
        require(
            len(original)
            == len(adapted)
            == len(added)
            == len(actual)
            == len(native_values[i])
            == count
            and all(
                math.isfinite(v) for v in (*original, *adapted, *added, *actual, *native_values[i])
            )
            and all(v == 0 for v in added)
            and all(struct.pack("<f", a) == struct.pack("<f", b) for a, b in zip(original, adapted))
            and all(
                struct.pack("<d", a) == struct.pack("<d", b)
                for a, b in zip(actual, native_values[i])
            ),
            "actual target logit bits or native target probabilities changed",
        )
        preserved.append(key)
    return {
        "ordered_head_ids": list(keys),
        "preserved_head_ids": preserved,
        "target_legal_logits_bitwise_unchanged": True,
        "target_native_probabilities_bitwise_unchanged": True,
    }


def compare_logits(actual, expected, counts):
    maxima = {}
    for key in ("base_legal_logits", "delta_legal_logits", "adapted_legal_logits"):
        require(
            type(actual.get(key)) is list and len(actual[key]) == len(counts),
            "all actual projection rows required",
        )
        errors = []
        for row, reference_row, count in zip(actual[key], expected[key], counts):
            require(len(row) == len(reference_row) == count, "all legal projection logits required")
            for value, target in zip(row, reference_row):
                require(
                    type(value) in (int, float)
                    and math.isfinite(value)
                    and math.isfinite(target)
                    and abs(value - target) <= ABS_TOL + REL_TOL * abs(target),
                    "same-forward actual/reconstructed logits differ",
                )
                errors.append(abs(value - target))
        maxima[key] = max(errors)
    return {
        "all_legal_logits_matched": True,
        "maximum_absolute_errors": maxima,
        "absolute_tolerance": ABS_TOL,
        "relative_tolerance": REL_TOL,
    }


def entropy_witness(raw, questions):
    rows = []
    for key, question in questions.items():
        answer = raw["answers"][key]
        require(
            list(answer["probabilities"]) == list(question["criteria"]),
            "entropy criteria ordering differs",
        )
        values = list(answer["probabilities"].values())
        require(
            values
            and all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in values),
            "finite legal entropy inputs required",
        )
        entropy = -sum(p * math.log(p) for p in values if p > 0)
        expected = (
            1.0 if len(values) == 1 else max(0.0, min(1.0, 1 - entropy / math.log(len(values))))
        )
        actual = answer["confidence"]
        require(
            type(actual) in (int, float)
            and math.isfinite(actual)
            and abs(actual - expected) <= ABS_TOL + REL_TOL * abs(expected),
            "fixed singleton/entropy comparison differs",
        )
        rows.append(
            {"head": key, "absolute_error": abs(actual - expected), "singleton": len(values) == 1}
        )
    return {
        "all_entropy_confidences_matched": True,
        "absolute_tolerance": ABS_TOL,
        "relative_tolerance": REL_TOL,
        "rows": rows,
    }


def catalog_unchanged(before, after):
    for key in (
        "numpy_version",
        "python",
        "numpy_build_configuration",
        "mapped_gpu_math_candidates_excluded_from_CPU_projection",
    ):
        require(before[key] == after[key], "numerical runtime configuration/GPU metadata changed")
    rows = {row["resolved_path"]: row for row in after["critical_files_and_mapped_math_candidates"]}
    require(
        all(
            rows.get(row["resolved_path"]) == row
            for row in before["critical_files_and_mapped_math_candidates"]
        ),
        "numerical runtime critical file changed",
    )


def perform_call(
    np_actor,
    native,
    helper,
    observer,
    probe,
    backend,
    runtime,
    torch,
    base,
    plan,
    shared,
    request,
    kind,
    rows,
    index,
    output,
    state,
):
    began, handles = time.monotonic(), []
    target = output / f"call-{index:02d}-{kind}-{rows}"
    target.mkdir(mode=0o700)
    fsync_directory(output)
    row = {
        "index": index,
        "kind": kind,
        "actual_rows_required": rows,
        "completed": False,
        "status": PENDING,
        "training_eligible": False,
        "labels": None,
    }
    state["calls"].append(row)
    witness = None

    def deadline():
        helper.ensure_elapsed(began, CALL_SECONDS, "complete four-call numerical prediction")

    def sink(event, data):
        require(
            event
            in {
                "prepared",
                "chunk_returned",
                "raw_returned_before_validation",
                "completed",
                "failed",
            },
            "unknown required backend retention event",
        )
        retain_json(
            helper,
            target / ("backend-" + event + ".json"),
            {"status": PENDING, "training_eligible": False, "event": event, "data": data},
        )

    def demote(error):
        row.update(
            completed=False,
            status="FAILED_NONTRAINING_NO_NUMERIC_CLAIM",
            error={"type": type(error).__name__, "message": str(error)},
        )

    def finalize(primary):
        try:
            if witness is not None:
                row["outer_first_layer_events"] = copy.deepcopy(witness.first_layer_events)
                row["outer_final_norm_events"] = copy.deepcopy(witness.norm_events)
            row["outer_hooks_removed"] = not handles
            row["elapsed_before_final_retention_seconds"] = time.monotonic() - began
            retain_json(helper, target / "evidence-pending.json", row)
            fsync_directory(output)
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note("Numerical call evidence also failed: " + repr(error))
        finalize_state(helper, output, state, primary)

    def operation():
        nonlocal witness
        np = numerical_context(shared, probe, native, helper, observer, runtime, torch, state)
        initial_catalog = probe.catalog(np, native, helper, began)
        native.validate_runtime(runtime, torch, base["model"])
        unchanged(shared, observer, runtime, torch)
        require(
            runtime.encoder.language_model is runtime.backbone.language_model,
            "original encoder/backbone join differs",
        )
        if kind == "original":
            predictor = native.NativeBatch4Backend(
                runtime, torch, base["model"], evidence_sink=sink
            )
            delta = torch.zeros((255, 1024), dtype=torch.float32, device="cpu")
        else:
            capability, delta = accepted_delta(shared, plan, backend, np, runtime, torch)
            # No enclosing inference_mode: the backend clone needs a real
            # version counter for its immutable owned-delta identity witness.
            predictor = backend.DeltaHeadBackend(
                native,
                observer,
                runtime,
                torch,
                base["model"],
                delta,
                root_capability=capability,
                root_capability_sha256=reference(plan["capability_ref"])["sha256"],
                deadline=deadline,
                evidence_sink=sink,
            )
            require(
                torch.equal(predictor.delta.view(torch.uint8), delta.view(torch.uint8)),
                "backend delta differs from true selected artifact",
            )
        prepared, lengths = predictor.prepare(**request)
        keys = list(request["questions"])
        counts = [len(request["questions"][key]["criteria"]) for key in keys]
        require(
            len(keys) == rows
            and prepared.runnable == prepared.keys == keys
            and list(prepared.sequences) == keys
            and list(lengths) == keys
            and not prepared.errors
            and not any(
                (prepared.images, prepared.videos, prepared.media, prepared.reversed_sequences)
            )
            and all(type(n) is int and 0 < n <= 16384 for n in lengths.values()),
            "complete original true partial rows required",
        )
        require(
            all(
                prepared.questions[key].keys == list(request["questions"][key]["criteria"])
                for key in keys
            ),
            "native criteria ordering changed",
        )
        width = max(lengths.values())
        witness = observer.FinalNormObserver(torch, rows, width)
        handles.append(
            runtime.backbone.language_model.layers[0].register_forward_pre_hook(witness.first_layer)
        )
        handles.append(runtime.backbone.language_model.norm.register_forward_hook(witness))
        state["native_request_calls"] += 1
        raw = predictor(**copy.deepcopy(request))
        retain_json(
            helper,
            target / "raw-response-pending.json",
            {"status": PENDING, "training_eligible": False, "raw_response": raw},
        )
        witness.require_single_pass()
        response_model = (
            base["model"] if kind == "original" else {**base["model"], "name": backend.ALIAS}
        )
        row["response_contract"] = probe.validate_answers(
            native, raw, request, lengths, response_model
        )
        row["entropy"] = entropy_witness(raw, request["questions"])
        masked_base, expected, reconstructed = same_forward_replay(
            torch, witness.features, runtime.readout, delta, counts, keys
        )
        row["same_forward_probabilities"] = observer.compare_distributions(
            raw, request["questions"], expected
        )
        original_replay = observer.replay_probabilities(runtime, torch, witness, counts)
        control, native_values = native_response_control(
            runtime, prepared, keys, lengths, original_replay
        )
        row["native_response_control"] = control  # Private, permanently NONTRAINING.
        if kind == "original":
            row["original_torch_replay"] = observer.compare_distributions(
                raw, request["questions"], original_replay
            )
            zero_values, zero_projection = backend.project(
                torch, masked_base, witness.features, delta, counts, keys
            )
            row["zero_primitive_probabilities"] = observer.compare_distributions(
                raw, request["questions"], zero_values
            )
            row["zero_primitive_logits"] = compare_logits(zero_projection, reconstructed, counts)
            row["independent_target_preservation"] = target_preservation(
                raw, native_values, keys, counts, zero_projection
            )
            row["control_provenance"] = (
                "Unmodified NativeBatch4Backend raw; reconstructed same-forward logits; zero primitive only, no trained capability."
            )
        else:
            audit = predictor.last_evidence
            require(
                audit["status"] == "completed"
                and audit["raw_response"] == raw
                and audit["lengths"] == lengths
                and len(audit["chunks"]) == 1,
                "complete same-call candidate evidence required",
            )
            chunk = audit["chunks"][0]
            event = [[rows, width, 1024]]
            require(
                chunk["head_order"] == keys
                and chunk["ordered_criteria_ids"]
                == [list(request["questions"][key]["criteria"]) for key in keys]
                and chunk["rows"] == rows
                and chunk["padded_width"] == width
                and chunk["legal_counts"] == counts
                and chunk["first_layer_events"] == chunk["final_norm_events"] == event,
                "actual candidate inner observer rows/order/one-pass differ",
            )
            row["actual_candidate_logits"] = compare_logits(
                chunk["projection"], reconstructed, counts
            )
            require(
                audit["profile"] == backend.PROFILE
                and audit["alias"] == backend.ALIAS
                and audit["only_adapted_head_id"] == "operation"
                and audit["capability_sha256"] == reference(plan["capability_ref"])["sha256"]
                and chunk["projection"]["ordered_head_ids"] == keys
                and chunk["projection"]["adapted_head_ids"] == ["operation"]
                and chunk["projection"]["preserved_head_ids"]
                == [key for key in keys if key != "operation"]
                and chunk["projection"]["target_logits_bitwise_unchanged"] is True,
                "new operation-only candidate projection provenance differs",
            )
            row["independent_target_preservation"] = target_preservation(
                raw, native_values, keys, counts, chunk["projection"]
            )
            row["candidate_backend_evidence"] = audit
            require(
                byte_hash(torch, delta) == shared["delta_sha256"]
                and torch.equal(predictor.delta.view(torch.uint8), delta.view(torch.uint8)),
                "true/owned trained delta changed",
            )
        row["reconstructed_same_forward_logits"] = reconstructed
        row["private_feature_payload"] = probe.write_pending_array(
            np,
            probe.copy_fp32(np, torch, witness.features, (rows, 1024)),
            target / "h-pending.npy",
            native,
            helper,
            began,
        )
        if "baseline_payload" not in shared:
            shared["baseline_payload"] = probe.write_pending_array(
                np,
                probe.copy_fp32(np, torch, runtime.readout, (255, 1024)),
                output / "W0-pending.npy",
                native,
                helper,
                began,
            )
        row["private_baseline_payload"] = shared["baseline_payload"]
        row["baseline_fp32_bytes_sha256"] = shared["baseline_sha256"]
        final_catalog = probe.catalog(np, native, helper, began)
        catalog_unchanged(initial_catalog, final_catalog)
        row["numpy_catalog_discovery"] = {
            "before": initial_catalog,
            "after": final_catalog,
            "training_eligible": False,
            "preselected_or_full_DSO_qualification": False,
        }
        witness.require_single_pass()
        require(
            witness.first_layer_events == witness.norm_events == [[rows, width, 1024]],
            "actual outer one-pass rows differ; no dummy rows",
        )
        unchanged(shared, observer, runtime, torch)
        native.validate_runtime(runtime, torch, base["model"])
        native.authenticate(helper, base)
        row.update(
            completed=True,
            status="NUMERICAL_CANDIDATE_PENDING_OUTER_CLEANUP_AND_ROOT_ACCEPTANCE",
            original_model_full_authentication_unchanged=True,
            labels=None,
            training_eligible=False,
        )
        return row

    result, elapsed = np_actor.transaction(
        native,
        helper,
        output,
        f"operation-numeric-call-{index}",
        began,
        CALL_SECONDS,
        operation,
        lambda: np_actor.remove_handles(handles),
        finalize,
        demote,
    )
    result["complete_prediction_seconds"] = elapsed  # No success I/O after call gate.
    return result


def execute(plan_path, expected, output):
    release_gate()
    began = time.monotonic()
    require(
        sys.flags.no_user_site
        and sys.dont_write_bytecode
        and not MODEL_MODULES.intersection(sys.modules),
        "fresh -B -s process without numerical/model preimports required",
    )
    plan, fixture_request = load_plan(plan_path, expected)
    native = source("_operation_numeric_native05", plan["dependencies"]["native_actor"])
    helper = native.load_helper()
    identity = helper.process_identity()
    limits = helper.owned_limits(identity)
    native.validate_owner(identity, limits, helper.START_ENV)
    gate = helper.host_gate()
    base = native.load_plan(helper, NATIVE / "edge-plan.json", PINS["native_plan"])
    require(plan["model"] == base["model"], "fixed original model identity differs")
    np_actor = source("_operation_numeric_np02", plan["dependencies"]["np_actor"])
    observer = source("_operation_numeric_observer", plan["dependencies"]["observer"])
    probe = source("_operation_numeric_probe", plan["dependencies"]["numpy_probe"])
    backend = source("_operation_numeric_released_backend", plan["dependencies"]["backend"])
    require(
        backend.SOURCE_ONLY is False
        and backend.PROFILE == "decision3-dsl01-operation-readout-cpu5-4g-candidate-v1"
        and backend.ALIAS == "d3-edge-dsl01-operation-delta-cpu5-candidate"
        and backend.CAPABILITY_SCHEMA == "root-accepted-dsl01-operation-train-delta-v1"
        and backend.CAPABILITY_SHA256 == reference(plan["capability_ref"])["sha256"],
        "new selected operation-only backend/capability binding required",
    )
    output = ordinary(output)
    require(
        output.name == "operation-numeric"
        and output.parent.parent == OWNER
        and re.fullmatch(r"actual-[0-9]{2}", output.parent.name)
        and not output.exists(),
        "fresh original-owner numerical output required",
    )
    helper.ensure_elapsed(began, WALL_SECONDS, "whole numerical preflight")
    output.mkdir(mode=0o700)
    (output / ".gitignore").write_text("*\n")
    fsync_directory(output)
    fsync_directory(output.parent)
    state = {
        "profile": PROFILE,
        "status": PENDING,
        "model_loads": 0,
        "native_request_calls": 0,
        "calls": [],
        "completed": False,
        "training_eligible": False,
        "labels": None,
        "plan_sha256": expected,
        "process": identity,
        "owned_limits": limits,
        "host_startup_admission": gate,
        "benchmark_scores": None,
        "browser_qualified": False,
        "model_qualified": False,
        "no_escape_audited": False,
        "whole_tree_audited": False,
        "semantic_expected_answers": None,
        "only_adapted_head_id": "operation",
        "candidate_alias": backend.ALIAS,
        "dev_used_for_model_selection": True,
        "optimizer_input_dev": False,
        "optimizer_input_sealed": False,
        "candidate_selection_before_dev_or_sealed": False,
    }
    shared = {}

    def demote(error):
        state.update(
            completed=False,
            status="FAILED_NONTRAINING_NO_NUMERIC_CLAIM",
            error={"type": type(error).__name__, "message": str(error)},
        )
        for row in state["calls"]:
            row.update(completed=False, status="FAILED_WHOLE_NONTRAINING_NO_NUMERIC_CLAIM")

    def finalize(primary):
        state["elapsed_before_final_retention_seconds"] = time.monotonic() - began
        finalize_state(helper, output, state, primary)

    def operation():
        runtime, torch = native.load_runtime(helper, base, output, state)
        for index, (kind, rows) in enumerate(CALLS, 1):
            perform_call(
                np_actor,
                native,
                helper,
                observer,
                probe,
                backend,
                runtime,
                torch,
                base,
                plan,
                shared,
                projection(fixture_request, rows),
                kind,
                rows,
                index,
                output,
                state,
            )
        require(
            state["model_loads"] == 1
            and state["native_request_calls"] == len(state["calls"]) == 4
            and all(row["completed"] for row in state["calls"])
            and [(row["kind"], row["actual_rows_required"]) for row in state["calls"]]
            == list(CALLS),
            "one model load and exactly four genuine ordered calls required",
        )
        unchanged(shared, observer, runtime, torch)
        native.validate_runtime(runtime, torch, base["model"])
        native.authenticate(helper, base)
        authenticate_sources(plan, plan_path, expected)
        for key in ("capability_ref", "root_acceptance_ref", "delta_ref"):
            read_ref(plan[key])
        state.update(
            completed=True,
            status="NUMERICAL_CANDIDATE_PENDING_ORIGINAL_ROOT_OWNER_CLOSURE_ACCEPTANCE",
            after_final_auth_owned_counters=native.own_counters(identity),
            baseline_metadata_values_and_bits_unchanged=True,
            original_full_model_runtime_source_authentication=True,
            training_eligible=False,
        )
        return state

    result, elapsed = np_actor.transaction(
        native,
        helper,
        output,
        "whole-operation-numerical-qualification",
        began,
        WALL_SECONDS,
        operation,
        lambda: None,
        finalize,
        demote,
    )
    result["post_cleanup_elapsed_seconds"] = elapsed
    return result


def main(argv=None):
    release_gate()  # Before parsing arguments, source, artifacts or model access.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    execute(args.plan, args.plan_sha256, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
