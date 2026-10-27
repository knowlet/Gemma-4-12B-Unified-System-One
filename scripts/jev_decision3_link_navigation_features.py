"""Feature-only extraction from independently admitted link-navigation wires.

The caller must supply the real, already loaded Decision3 provider, externally
supervise its resources and separately audit/persist the returned evidence.
This module does not start a model, grant fit eligibility, or launch a service.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import jev_decision3_link_navigation_pipeline as pipeline

MODEL_ID = "vllm-sr/d3-edge"
REVISION = "cab71e6ad3461b8086d51ddb572c656babd63b0b"
MANIFEST_SHA256 = "3bae8989a6feb36a66bf4d8c44112dc63199d0dc69f564dc895fe368b1e08e59"
MODEL_SHA256 = "6e00d04abbdbdf6d35c9e12065769d9edaacc3bfe9404583891797718d239dc8"
W0_SHA256 = "d408d94a0ab2915fa85c511dd9b8d62af49903c4e4ca28e2e2d5aff9ef565e0f"
ALIAS = "generated-actions-none"
SCHEMA = "decision3-link-navigation-feature-shard-v1"


def _need(value, message):
    if not value:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _file_sha(path, *, within=None):
    path = Path(path)
    _need(
        path.is_absolute()
        and path.is_file()
        and (
            (within is not None and path.resolve().is_relative_to(within))
            or (within is None and not path.is_symlink())
        ),
        "ordinary absolute pinned file required",
    )
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def feature_only_shards(admitted, *, shard_size):
    """Project admitted wires only; never expose labels to the provider.

    An admitted collection still has fit_eligible=False. The returned shards
    are plans, not GPU evidence; callers must not treat them as accepted data.
    """
    _need(type(shard_size) is int and 1 <= shard_size <= 40, "bounded shard size required")
    rows = pipeline.compile_wire_projection(admitted)
    _need(bool(rows), "nonempty admitted collection required")
    identities = set()
    wires = []
    for ordinal, row in enumerate(rows):
        _need(row["fit_eligible"] is False, "input must remain quarantined")
        payload = copy.deepcopy(row["payload"])
        _need(
            type(payload) is dict
            and set(payload) == {"model", "state", "questions"}
            and payload["model"] == ALIAS,
            "full native wire required",
        )
        _need(_sha(_canonical(payload)) == row["wire_sha256"], "original wire digest differs")
        _need(
            type(row["case_id"]) is str and row["case_id"] not in identities,
            "unique case identities required",
        )
        identities.add(row["case_id"])
        questions = payload["questions"]
        _need(
            type(questions) is dict and 1 <= len(questions) <= 4,
            "one to four ordered native heads required",
        )
        for head, question in questions.items():
            _need(
                type(head) is str
                and type(question) is dict
                and set(question) == {"type", "instructions", "criteria"}
                and question["type"] == "choice",
                "original native choice head required",
            )
            criteria = question["criteria"]
            _need(
                type(criteria) is dict
                and 1 <= len(criteria) <= 255
                and all(type(key) is str and key.strip() for key in criteria),
                "full ordered candidate inventory required",
            )
        wires.append(
            {
                "global_ordinal": ordinal,
                "case_id": row["case_id"],
                "wire_sha256": row["wire_sha256"],
                "payload": payload,
            }
        )
    identity = _sha(
        _canonical(
            [{k: row[k] for k in ("global_ordinal", "case_id", "wire_sha256")} for row in wires]
        )
    )
    return [
        {
            "schema": SCHEMA,
            "index": i // shard_size,
            "start": i,
            "stop": min(i + shard_size, len(wires)),
            "total": len(wires),
            "ordered_identity_sha256": identity,
            "receipt_sha256": admitted.receipt_sha256,
            "plan_sha256": admitted.plan_sha256,
            "wires": wires[i : i + shard_size],
            "status": "planned_not_captured",
            "fit_eligible": False,
        }
        for i in range(0, len(wires), shard_size)
    ]


def _validate_manifest(runtime, cache_root):
    root = Path(runtime.root)
    cache_root = Path(cache_root)
    _need(
        root.is_absolute()
        and root.is_dir()
        and cache_root.is_absolute()
        and cache_root.is_dir()
        and root.resolve().is_relative_to(cache_root.resolve()),
        "pinned model snapshot within selected cache required",
    )
    cache_root = cache_root.resolve()
    manifest_path = root / "MODEL_MANIFEST.json"
    _need(
        _file_sha(manifest_path, within=cache_root) == MANIFEST_SHA256,
        "publisher manifest pin differs",
    )
    manifest = json.loads(manifest_path.read_bytes())
    _need(
        manifest.get("schema") == "d3-package-manifest/1"
        and manifest.get("identity", {}).get("model_sha256") == MODEL_SHA256,
        "publisher checkpoint identity differs",
    )
    files = manifest.get("files_sha256")
    _need(type(files) is dict and files, "manifest file inventory required")
    for name, digest in files.items():
        if (
            name in ("LICENSE", "NOTICE", "requirements.txt", ".gitattributes")
            or name.endswith(".md")
            or name.startswith(("assets/", "LICENSES/"))
        ):
            continue
        relative = Path(name)
        _need(not relative.is_absolute() and ".." not in relative.parts, "unsafe model file path")
        path = root / relative
        _need(
            _file_sha(path, within=cache_root) == digest, "model file differs from pinned manifest"
        )
    _need(runtime.manifest == manifest, "loaded runtime manifest differs")
    return _sha(manifest_path.read_bytes())


def _check_sources(pins):
    _need(type(pins) is dict and bool(pins), "explicit runtime/source pins required")
    for path, digest in pins.items():
        _need(
            type(path) is str
            and type(digest) is str
            and len(digest) == 64
            and _file_sha(path) == digest,
            "runtime source pin differs",
        )


def _tensor_sha(tensor):
    return _sha(tensor.detach().to(device="cpu").contiguous().numpy().tobytes())


def _prepared_contract(prepared, questions):
    """Check the publisher's complete, ordered text-only preparation."""
    heads = list(questions)
    sequences = prepared.sequences
    _need(
        not prepared.errors
        and prepared.keys == prepared.runnable == heads
        and list(sequences) == heads
        and list(prepared.questions) == heads
        and not any(
            (prepared.images, prepared.videos, prepared.media, prepared.reversed_sequences)
        ),
        "complete ordered text-only native preparation required",
    )
    lengths = {head: len(sequence) for head, sequence in sequences.items()}
    lengths.update(prepared.lengths)
    _need(
        list(lengths) == heads
        and all(
            type(length) is int and 0 < length <= 16384 and length == len(sequences[head])
            for head, length in lengths.items()
        ),
        "original full native sequence lengths required",
    )
    _need(
        all(
            list(prepared.questions[head].keys) == list(questions[head]["criteria"])
            for head in heads
        ),
        "original ordered native criteria required",
    )
    return lengths


def _verify_call_contract(events, native, lengths, first_shape, norm_shape, result):
    """Bind both provider events, observed widths and native usage to preparation."""
    _need(type(events) is list and len(events) == 2, "two native events required")
    first_preflight = events[0].get("preflight") if type(events[0]) is dict else None
    _need(
        type(first_preflight) is dict
        and set(first_preflight) == {"lengths", "errors", "project_context_limit"}
        and type(first_preflight["lengths"]) is dict
        and list(first_preflight["lengths"]) == list(lengths)
        and first_preflight["lengths"] == lengths
        and not first_preflight["errors"]
        and first_preflight["project_context_limit"] == 16384,
        "each provider event must bind the original preparation",
    )
    for event in events:
        _need(
            type(event) is dict
            and set(event) == {"native_payload", "preflight", "raw_response"}
            and _canonical(event["native_payload"]) == _canonical(native)
            and _canonical(event["preflight"]) == _canonical(first_preflight),
            "each provider event must bind the original preparation",
        )
    _need(
        events[0]["raw_response"] is None and type(events[1]["raw_response"]) is dict,
        "ordered native preflight and raw return required",
    )
    _need(
        type(result) is dict and result.get("decision3_provider_evidence") == events[1],
        "returned native evidence differs",
    )
    expected_shape = (len(lengths), max(lengths.values()), 1024)
    _need(
        first_shape == norm_shape == expected_shape,
        "first-layer and final-norm widths must match original native preparation",
    )
    raw = events[1]["raw_response"]
    usage = raw.get("usage")
    _need(
        type(usage) is dict
        and set(usage) == {"input_tokens", "output_tokens"}
        and all(type(value) is int for value in usage.values())
        and usage == {"input_tokens": sum(lengths.values()), "output_tokens": 0},
        "original native token usage differs",
    )
    return raw


def _validate_provider_spec(spec):
    _need(
        getattr(spec, "adapter", None) == "decision3"
        and getattr(spec, "model_id", None) == MODEL_ID
        and getattr(spec, "revision", None) == REVISION
        and getattr(spec, "device", None) == "cuda:0"
        and getattr(spec, "precision", None) == "bfloat16"
        and getattr(spec, "context_limit", None) == 16384,
        "pinned native provider spec required",
    )


def _call_with_hooks(language, torch, first_hook, norm_hook, callback):
    """Fence and remove all installed hooks, preserving the primary exception."""
    handles, primary, result = [], None, None
    try:
        handles.append(language.layers[0].register_forward_pre_hook(first_hook))
        handles.append(language.norm.register_forward_hook(norm_hook))
        result = callback()
    except BaseException as error:
        primary = error
    finally:
        cleanup = [lambda: torch.cuda.synchronize(0)]
        cleanup.extend(handle.remove for handle in reversed(handles))
        for action in cleanup:
            try:
                action()
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    primary.add_note("GPU fence or hook removal also failed: " + repr(error))
    if primary is not None:
        raise primary
    return result


def capture_shard(
    admitted, *, shard_size, shard_index, backend, torch, source_pins, cache_root, guard, deadline
):
    """Perform one native CUDA forward per admitted wire under the caller's owner.

    Source pins are supplied by the owner, who must authenticate the pin inventory
    independently; this function verifies those supplied bytes, not their origin.
    Pins must include the publisher runtime and wrapper, provider, parser and
    native serializer. The owner must supervise process limits and cleanup.
    Output is pending independent FP64 replay and fit admission.
    """
    import inspect

    from s1.evaluation.decision3 import native_payload
    from s1.jev_decision3_native import parse_native_payload

    shards = feature_only_shards(admitted, shard_size=shard_size)
    _need(
        type(shard_index) is int and 0 <= shard_index < len(shards),
        "valid admitted shard index required",
    )
    _need(callable(guard) and callable(deadline), "external owner guard and deadline required")
    shard = shards[shard_index]
    runtime = backend.runtime
    _need(
        getattr(getattr(backend, "model", None), "runtime", None) is runtime,
        "provider must invoke the selected native runtime",
    )
    _need(getattr(backend, "identity", {}).get("model_id") == MODEL_ID, "provider model ID differs")
    _validate_provider_spec(backend.spec)
    _need(
        getattr(backend.spec, "context_limit", None) == 16384
        and runtime.temperature == 1.0
        and runtime.batch_size == 8
        and runtime.max_length is None
        and runtime.attention_mode == "noncausal_full_attention"
        and runtime.permutation_average is False
        and runtime.fast is None
        and runtime.readout_dtype == "float32"
        and runtime.model_name == "d3-edge",
        "original provider policy required",
    )
    language = runtime.backbone.language_model
    _need(
        runtime.encoder.language_model is language and not runtime.backbone.training,
        "selected evaluation backbone required",
    )
    _need(
        inspect.getfile(type(runtime)) in source_pins
        and inspect.getfile(type(backend)) in source_pins
        and inspect.getfile(type(backend.model)) in source_pins
        and inspect.getfile(parse_native_payload) in source_pins
        and inspect.getfile(native_payload) in source_pins,
        "actual provider, runtime, parser and serializer source pins required",
    )

    def authenticate(*, checkpoint=False):
        deadline()
        guard()
        _validate_provider_spec(backend.spec)
        _check_sources(source_pins)
        if checkpoint:
            _validate_manifest(runtime, cache_root)
        params = list(runtime.backbone.parameters())
        _need(
            params
            and all(
                str(p.device) == "cuda:0" and p.dtype == torch.bfloat16 and not p.requires_grad
                for p in params
            ),
            "CUDA BF16 frozen backbone required",
        )
        _need(
            sum(p.numel() for p in params) + runtime.readout.numel() == 594582720,
            "model parameter identity differs",
        )
        _need(
            tuple(runtime.readout.shape) == (255, 1024)
            and runtime.readout.dtype == torch.float32
            and str(runtime.readout.device) == "cuda:0"
            and _tensor_sha(runtime.readout) == W0_SHA256,
            "pinned CUDA FP32 W0 required",
        )
        _need(
            torch.get_float32_matmul_precision() == "highest"
            and torch.backends.cuda.matmul.allow_tf32 is False
            and not torch.is_autocast_enabled("cuda"),
            "unmodified FP32 readout policy required",
        )

    authenticate(checkpoint=True)
    baseline = runtime.readout.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    results = []
    for item in shard["wires"]:
        authenticate()
        wire = copy.deepcopy(item["payload"])
        _need(_sha(_canonical(wire)) == item["wire_sha256"], "wire changed before provider call")
        parsed, _, _, _ = parse_native_payload(wire, model_identity=ALIAS)
        native = native_payload(parsed)
        _need(
            _canonical(native["state"]) == _canonical(wire["state"])
            and list(native["questions"]) == list(wire["questions"])
            and all(
                set(native["questions"][head]) == set(wire["questions"][head])
                and all(
                    _canonical(native["questions"][head][key])
                    == _canonical(wire["questions"][head][key])
                    for key in ("type", "instructions", "criteria")
                )
                for head in wire["questions"]
            ),
            "native serialization changed wire order or content",
        )
        heads = list(wire["questions"])
        counts = [len(wire["questions"][key]["criteria"]) for key in heads]
        deadline()
        prepared = runtime.prepare(**copy.deepcopy(native))
        deadline()
        lengths = _prepared_contract(prepared, native["questions"])
        expected_shape = (len(heads), max(lengths.values()), 1024)
        captured, first, norm_shapes, events = [], [], [], []

        def first_hook(_module, args):
            _need(
                not first
                and args
                and args[0].dtype == torch.bfloat16
                and str(args[0].device) == "cuda:0"
                and tuple(args[0].shape) == expected_shape,
                "single CUDA first-layer pass required",
            )
            first.append(tuple(args[0].shape))

        def norm_hook(_module, _args, output):
            _need(
                not captured
                and output.dtype == torch.bfloat16
                and str(output.device) == "cuda:0"
                and tuple(output.shape) == expected_shape,
                "single same-call final-norm output required",
            )
            norm_shapes.append(tuple(output.shape))
            captured.append(output[:, -1, :].detach().to(dtype=torch.float32).clone())

        def sink(event, value):
            _need(
                len(events) < 2
                and event == ("native_preflight", "native_raw_returned")[len(events)]
                and _canonical(value["native_payload"]) == _canonical(native),
                "complete original native provider events required",
            )
            events.append(copy.deepcopy(value))

        result = _call_with_hooks(
            language,
            torch,
            first_hook,
            norm_hook,
            lambda: backend.predict(parsed, deadline=deadline, evidence_sink=sink),
        )
        _need(
            len(first) == len(captured) == len(norm_shapes) == 1
            and len(events) == 2
            and first[0] == norm_shapes[0] == expected_shape,
            "one real original forward required",
        )
        _need(
            tuple(captured[0].shape) == (len(heads), 1024)
            and bool(torch.isfinite(captured[0]).all().item()),
            "finite same-call feature tensor required",
        )
        raw = _verify_call_contract(events, native, lengths, first[0], norm_shapes[0], result)
        preflight = events[0]["preflight"]
        _need(
            type(raw) is dict
            and raw.get("model") == "d3-edge"
            and list(raw.get("answers", {})) == heads,
            "native raw answer inventory differs",
        )
        with torch.inference_mode():
            logits = captured[0] @ runtime.readout.T
            mask = (
                torch.arange(255, device="cuda:0")[None, :]
                >= torch.tensor(counts, device="cuda:0")[:, None]
            )
            probs = torch.softmax(logits.masked_fill(mask, float("-inf")), dim=-1).cpu().tolist()
        for index, head in enumerate(heads):
            answer, labels = raw["answers"][head], list(wire["questions"][head]["criteria"])
            original = answer.get("probabilities", {})
            _need(
                answer.get("type") == "choice"
                and not answer.get("error")
                and list(original) == labels,
                "native ordered choices required",
            )
            values = list(original.values())
            _need(
                all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values)
                and abs(sum(values) - 1) <= 1e-6,
                "finite normalized native probabilities required",
            )
            _need(
                answer.get("choice") == max(original, key=original.__getitem__)
                and max(
                    abs(a - b) for a, b in zip(values, probs[index][: counts[index]], strict=True)
                )
                <= 1e-5,
                "same-call W0 replay differs",
            )
        results.append(
            {
                "global_ordinal": item["global_ordinal"],
                "case_id": item["case_id"],
                "wire_sha256": item["wire_sha256"],
                "native_payload_sha256": _sha(_canonical(native)),
                "head_order": heads,
                "criteria_order": [list(wire["questions"][head]["criteria"]) for head in heads],
                "preflight": preflight,
                "first_layer_shape": first[0],
                "final_norm_shape": norm_shapes[0],
                "features": captured[0].cpu().contiguous(),
                "features_fp32_sha256": _tensor_sha(captured[0]),
                "raw_response": raw,
                "raw_response_sha256": _sha(_canonical(raw)),
            }
        )
        authenticate()
    authenticate(checkpoint=True)
    _need(
        len(results) == shard["stop"] - shard["start"] and _tensor_sha(baseline) == W0_SHA256,
        "complete shard and unchanged W0 required",
    )
    return {
        **{k: v for k, v in shard.items() if k != "wires"},
        "status": "captured_pending_independent_fp64_audit",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "manifest_sha256": MANIFEST_SHA256,
        "model_sha256": MODEL_SHA256,
        "source_pins": dict(source_pins),
        "baseline": baseline,
        "baseline_sha256": W0_SHA256,
        "cases": results,
        "model_score": None,
        "fit_eligible": False,
    }
