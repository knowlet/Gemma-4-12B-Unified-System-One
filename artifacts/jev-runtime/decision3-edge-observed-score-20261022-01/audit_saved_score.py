"""Reproduce saved task-score/source/profile/HTTP joins; no model or live HTTP.

This records the ordinary-stdin audit logic as a reusable read-only program.
It does not replay DOM judges or regenerate the past late-closure observation.
"""

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
HERE = Path(__file__).absolute().parent
EXECUTION_SHA = "248687b7dddc5cf7697482e85b2ec909f1c04cbecc5e258d0f23478ecbf3745f"
PLAN_SHA = "852b162a5d2a9b7c5f5516ce2ac913be5510491e6c85dd1603baaeb31d6db6ca"


def digest(value):
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


def audit():
    report = json.loads((HERE / "observed-score.json").read_bytes())
    source = ROOT / "artifacts/jev-runtime/decision3-local-execution-retry-20261022-01/execution.py"
    blob = source.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == EXECUTION_SHA
    spec = importlib.util.spec_from_file_location("_observed_original_execution", source)
    E = importlib.util.module_from_spec(spec)
    exec(compile(blob, str(source), "exec", dont_inherit=True), E.__dict__)
    plan_blob = (ROOT / report["original_plan"]["path"]).read_bytes()
    assert hashlib.sha256(plan_blob).hexdigest() == PLAN_SHA
    plan = json.loads(plan_blob)
    for ref in plan["sources"]:
        E.read_ref(ref)
    v7 = E.load_module(plan["components"]["v7"], "_observed_original_v7")
    groups = {}
    for group in ("known4", "public6"):
        folder = Path(plan["output"]) / group
        score = E.browser_score(folder / "results/summary.json", group)
        assert score["attempts"] == report["observed_browser_score"]["groups"][group]["attempts"]
        assert score["successes"] == report["observed_browser_score"]["groups"][group]["successes"]
        manifest = json.loads((folder / "bundle/bundle-manifest.json").read_bytes())
        v7.verify_bundle(folder / "bundle", manifest)
        clients, gateways = [], []
        for path in sorted((folder / "results/attempts").glob("*/model_calls.json")):
            for call in json.loads(path.read_bytes()):
                for attempt in call.get("transport_attempts", []):
                    if attempt.get("request", {}).get("model") == "d3-edge":
                        clients.append((digest(attempt["request"]), digest(attempt["response"]), attempt["http_status"]))
        for path in sorted((folder / "transport").glob("request-*.json")):
            row = json.loads(path.read_bytes())
            request, worker = bytes.fromhex(row["raw_http_hex"]), row["worker"]
            wire = worker["wire_body_utf8"].encode()
            native = json.loads(Path(worker["journal"]["journal_path"]).read_bytes())
            assert hashlib.sha256(request).hexdigest() == row["http_sha256"] == worker["http_sha256"]
            assert row["response_sha256"] == hashlib.sha256(wire).hexdigest()
            assert row["status"] == 200 and row["gateway_action_authorized"] is True and "error" not in row
            assert worker["success"] is True and worker["campaign_stop"] is False
            assert worker["request_id"] == row["request_id"] and worker["nonce"] == plan["arms"][group]["nonce"]
            assert worker["model"] == plan["model"] and native == worker["journal"]
            assert native["stage"] == "complete" and native["success"] is True and native["campaign_stop"] is False
            assert native["request_id"] == row["request_id"] and native["http_sha256"] == row["http_sha256"]
            assert native["wire_body_utf8"] == worker["wire_body_utf8"]
            gateways.append((digest(json.loads(request)), digest(json.loads(wire)), 200))
        assert Counter(clients) == Counter(gateways)
        assert not list((folder / "transport").glob("delivery-*.json"))
        groups[group] = {"attempts": score["attempts"], "successes": score["successes"], "joined_http_attempts": len(clients)}
    assert sum(g["attempts"] for g in groups.values()) == 10
    assert sum(g["successes"] for g in groups.values()) == 0
    return {"observed_browser_score": {"attempts": 10, "successes": 0, "success_rate": 0.0}, "groups": groups, "source_refs_verified": 39, "execution_completed": False, "model_qualified": False, "live_calls_made": False, "DOM_judges_replayed": False}


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True))
