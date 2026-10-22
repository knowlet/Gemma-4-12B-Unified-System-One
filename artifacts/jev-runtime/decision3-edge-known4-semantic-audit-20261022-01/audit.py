"""Bounded CPU replay of saved request conversion; no model or network calls."""

import ast
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
HERE = Path(__file__).absolute().parent
PLAN = ROOT / "artifacts/jev-runtime/decision3-edge-browser-retry-plan-20261022-01/plan-02.json"
PLAN_SHA = "852b162a5d2a9b7c5f5516ce2ac913be5510491e6c85dd1603baaeb31d6db6ca"
OUT = ROOT / "artifacts/jev-runtime/decision3-local-execution-retry-20261022-01/edge-02/known4"
NATIVE = ROOT / "artifacts/jev-d3-native/20261022-edge-known4-02"
SNAPSHOT = ROOT / "artifacts/jev-local-hf/hub/models--vllm-sr--d3-edge/snapshots/cab71e6ad3461b8086d51ddb572c656babd63b0b"


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def ref(path):
    blob = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(blob), "size_bytes": len(blob)}


def checked(path, digest):
    blob = path.read_bytes()
    assert sha(blob) == digest
    return blob


def module(path, digest, name):
    blob = checked(path, digest)
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    exec(compile(blob, str(path), "exec", dont_inherit=True), value.__dict__)
    return value


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-14, abs_tol=1e-14)


def main():
    plan = json.loads(checked(PLAN, PLAN_SHA))
    for selected in plan["sources"]:
        assert (ROOT / selected["path"]).stat().st_size == selected["size_bytes"]
        checked(ROOT / selected["path"], selected["sha256"])
    base = ROOT / "artifacts/jev-runtime/decision3-current-source-bootstrap-20261022-01"
    boot = module(base / "current_bootstrap.py", plan["components"]["bootstrap"]["sha256"], "_audit_boot")
    core_ref = next(x for x in plan["sources"] if x["path"].endswith("/current_core.py"))
    core_ast = ast.parse(checked(ROOT / core_ref["path"], core_ref["sha256"]))
    pins = next(ast.literal_eval(n.value) for n in core_ast.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "SOURCE_PINS" for t in n.targets))
    with boot.ProjectSources(pins):
        from s1.evaluation.decision3 import native_payload, normalize_native_response
        from s1.jev_decision3_native import parse_native_payload
    assert not any(n == "torch" or n.startswith("torch.") or n == "transformers"
                   or n.startswith("transformers.") for n in sys.modules)
    cache = json.loads((NATIVE / "initial-cache.json").read_bytes())
    fmt_ref = cache["files"]["d3_format.py"]
    fmt = module(SNAPSHOT / "d3_format.py", fmt_ref["sha256"], "_audit_publisher_format")
    wire_ref = next(x for x in plan["sources"] if x["path"] == "scripts/jev_decision3_browser_wire.py")
    wire = module(ROOT / wire_ref["path"], wire_ref["sha256"], "_audit_named_wire")
    config_ref = cache["files"]["decision_config.json"]
    config = json.loads(checked(SNAPSHOT / "decision_config.json", config_ref["sha256"]))
    assert len(config["codes"]) == len(config["token_ids"]) == 255
    calls = {}
    for path in sorted((OUT / "results/attempts").glob("*/model_calls.json")):
        values = json.loads(path.read_bytes())
        assert len(values) == 1
        call = values[0]
        assert call["kind"] == "policy" and len(call["transport_attempts"]) == 1
        attempt = call["transport_attempts"][0]
        assert attempt["http_status"] == 200 and attempt["request"] == call["request"]
        assert attempt["response"] == call["response"]
        calls[sha(encode(call["request"]))] = (path, call)
    rows = []
    evidence = [ref(PLAN)]
    for path in sorted((NATIVE / "requests").glob("*.json")):
        journal = json.loads(path.read_bytes())
        transport_path = OUT / "transport" / ("request-" + path.name)
        transport = json.loads(transport_path.read_bytes())
        raw_http = bytes.fromhex(transport["raw_http_hex"])
        assert sha(raw_http) == transport["http_sha256"] == journal["http_sha256"]
        request = json.loads(raw_http)
        assert raw_http.decode() == journal["payload_json"]
        assert request == journal["original_request"]
        call_path, call = calls.pop(sha(encode(request)))
        assert request == call["request"]
        parsed, original, source, labels = parse_native_payload(request, model_identity="d3-edge")
        assert original == request and sha(source) == journal["http_sha256"]
        projected = native_payload(parsed)
        events = {x["event"]: x["data"] for x in journal["events"]}
        native = events["native_raw_returned"]
        assert projected == native["native_payload"] == events["native_preflight"]["native_payload"]
        assert projected["state"] == request["state"]
        assert list(projected["questions"]) == list(request["questions"])
        assert native["raw_response"] == journal["raw_response"]
        raw = journal["raw_response"]
        first = normalize_native_response(parsed, raw)
        second = normalize_native_response(parsed, first)
        legacy = journal["legacy_response"]
        assert legacy["answers"] == [second["answers"][q.id] for q in parsed.questions]
        expected_wire = wire.named_browser_response(request, legacy)
        wire_bytes = journal["wire_body_utf8"].encode()
        assert expected_wire == json.loads(wire_bytes) == call["response"]
        assert transport["worker"]["wire_body_utf8"] == journal["wire_body_utf8"]
        assert sha(wire_bytes) == transport["response_sha256"]
        assert transport["worker"]["journal"] == journal
        assert transport["status"] == 200 and journal["success"] is True
        assert journal["stage"] == "complete" and journal["campaign_stop"] is False
        preflight = native["preflight"]
        assert preflight["errors"] == {} and preflight["project_context_limit"] == 8192
        head_rows = []
        for key, question in projected["questions"].items():
            assert question == request["questions"][key]
            keys, texts = fmt.options(question)
            assert keys == labels[key] == list(question["criteria"])
            assert len(texts) == len(keys) <= 255
            answer = raw["answers"][key]
            assert list(answer["probabilities"]) == keys
            replay = fmt.to_answer(question, list(answer["probabilities"].values()))
            assert replay["choice"] == answer["choice"] == expected_wire["answers"][key]["choice"]
            for label in keys:
                close(replay["probabilities"][label], answer["probabilities"][label])
                close(answer["probabilities"][label], expected_wire["answers"][key]["probabilities"][label])
            assert 0 < preflight["lengths"][key] <= 8192
            head_rows.append({"id": key, "candidate_count": len(keys),
                              "candidate_labels_sha256": sha(encode(keys)),
                              "code_to_label_sha256": sha(encode(list(zip(config["codes"], keys)))),
                              "expected_publisher_messages_sha256": sha(encode(fmt.messages(projected["state"], question, config["codes"]))),
                              "observed_tokens": preflight["lengths"][key],
                              "original_criteria_and_instructions_preserved": True,
                              "raw_argmax_choice_preserved": True})
        counts = journal["task_native_calls"]
        assert all(counts[k] == 1 for k in ("backend_predict_entries", "system_one_entries", "probabilities_entries"))
        assert counts["sequence_rows"] == len(head_rows)
        assert counts["input_tokens"] == sum(preflight["lengths"].values())
        operation = raw["answers"]["operation"]
        assert operation["choice"] == "BLOCKED"
        rows.append({"case_id": call_path.parent.name, "request_id": path.stem,
                     "http_sha256": transport["http_sha256"],
                     "raw_operation_choice": operation["choice"],
                     "raw_operation_max_probability": max(operation["probabilities"].values()),
                     "raw_operation_entropy_confidence": operation["confidence"],
                     "wire_operation_confidence": expected_wire["answers"]["operation"]["confidence"],
                     "heads": head_rows, "native_calls": counts,
                     "wire_answer_container_only_replay_equal": True})
        evidence.extend(map(ref, (path, transport_path, call_path)))
    assert len(rows) == 4 and not calls
    summary = json.loads((OUT / "results/summary.json").read_bytes())
    assert summary["attempts"] == 4 and summary["successes"] == 0
    closure = json.loads((OUT / "transport/closure.json").read_bytes())
    assert closure["success"] is False and closure["observed_process_exit_confirmed"] is False
    assert closure["direct_child_exit_confirmed"] is True and closure["returncode"] == 0
    result = {"schema_id": "decision3-edge-known4-saved-semantic-audit-v1", "audit_passed": True,
              "plan_sha256": PLAN_SHA, "selected_sources_checked": len(plan["sources"]),
              "model": "d3-edge", "model_revision": cache["revision"],
              "known4_attempts": 4, "known4_successes": 0, "full_ten_score": None,
              "conversion_or_label_mapping_discrepancy_found": False, "rows": rows,
              "inner_transport_success_preserved": False,
              "inner_observed_process_exit_confirmed_preserved": False,
              "inference_rerun": False, "tokenizer_rerun": False,
              "limitations": ["Actual task prompt/token streams and readout logits were not independently saved; reconstructed publisher message hashes are expected-source replay only.",
                              "No tensor numerical replay, causal model-quality attribution, or cross-model comparison is claimed.",
                              "Publisher entropy confidence is intentionally replaced with project max-probability confidence; choices and label probabilities are preserved within floating-point normalization tolerance."],
              "evidence": evidence + [ref(OUT / "results/summary.json"), ref(OUT / "transport/closure.json"),
                                       ref(SNAPSHOT / "d3_format.py"), ref(SNAPSHOT / "decision_config.json")],
              "audit_source": ref(Path(__file__).absolute())}
    target = HERE / "summary.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"audit_passed": True, "summary": ref(target), "requests": len(rows)}))


if __name__ == "__main__":
    main()
