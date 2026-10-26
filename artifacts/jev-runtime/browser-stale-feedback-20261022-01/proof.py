"""Read-only recorded evidence plus original-source, entirely fake I/O fixtures."""

import copy
import datetime
import hashlib
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
UPSTREAM = ROOT / "artifacts/jev-runtime/upstream/jev_ultrafast"
RAW_PARENT = ROOT / "artifacts/jev-runtime/native-v2-browser-execution-20261022-03"
RAW = RAW_PARENT / "raw-03"
PINS = {
    "agent.py": ("aaf8b271cb0ee6643fe7beb525328ab80de8568c7ac044d02b9c6721161e7c8b", 7880),
    "browser.py": ("ee08dafa93d731bd292aa0263a578a29c8001cf22eef3c59fe44a159649155ee", 9266),
    "model.py": ("a85ada8458d23642c5c8a4095acf0565e6268f81dcda26f6308942b653e9c9f3", 8469),
    "questions.py": ("66a7054b85040b949e1757311257ea0c5a843c635f9a40c99f6d15ad660fb422", 2048),
    "snapshot.js": ("e50473501c8fb8e70f3b21866d987393e3f2315c639d638bd477d170e81ed78d", 6490),
}


def need(condition, label):
    if not condition:
        raise AssertionError(label)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pin(path):
    before = path.stat()
    need(before.st_size <= 32 * 1024**2, "bounded small evidence file")
    data = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    need(
        all(getattr(before, name) == getattr(after, name) for name in fields),
        "stable small evidence file",
    )
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def ref(path):
    return {"path": str(path.relative_to(ROOT)), **pin(path)}


def checked_source(name):
    data = (UPSTREAM / name).read_bytes()
    need((hashlib.sha256(data).hexdigest(), len(data)) == PINS[name], "exact selected " + name)
    return data


def observed_evidence():
    """Publish counts/hash references only, without recorded DOM, goals or answers."""
    plan_path = RAW / "selected-plan.json"
    need(
        pin(plan_path)["sha256"]
        == "e6aab745961255af77258d3cf6bcbbb516d3c8d9bc3507ab327fd99774d187d0",
        "selected raw plan",
    )
    plan = json.loads(plan_path.read_bytes())
    sources = {row["path"]: row for row in plan["sources"]}
    for name in PINS:
        checked_source(name)
        row = sources[str((UPSTREAM / name).relative_to(ROOT))]
        need(
            pin(UPSTREAM / name) == {k: row[k] for k in ("sha256", "size_bytes")},
            "source selected by actual raw plan",
        )
    audit_path = RAW_PARENT / "independent-audit-raw03.json"
    need(
        pin(audit_path)["sha256"]
        == "0edb616db1f8774b5353dd2114667f1d693189a8fadda6284f7ee8aeb590a6dd",
        "selected independent raw audit",
    )
    audit = json.loads(audit_path.read_bytes())
    need(audit["evaluation_complete"] is True, "complete actual run")
    folder = RAW / "browser/public6/results/attempts/heldout-public-003-000"
    states_path = folder / "states.json"
    calls_path = folder / "model_calls.json"
    states = json.loads(states_path.read_bytes())
    calls = json.loads(calls_path.read_bytes())
    need(len(states) == 17 and len(calls) == 16, "actual state/call counts")
    tail = calls[5:16]
    state_tail = states[5:17]
    request = tail[0]["request"]
    need(all(row["request"] == request for row in tail), "identical model requests")
    need(
        all(row["response"]["answers"] == tail[0]["response"]["answers"] for row in tail),
        "identical decoded answers",
    )
    need(
        all(
            row["response"]["answers"]["operation"]["choice"] == "CLICK"
            and row["response"]["answers"]["click_target"]["choice"] == "39"
            for row in tail
        ),
        "observed repeated selected operation/target",
    )
    need(
        all(
            row["page"] == state_tail[0]["page"]
            and row["history"] == state_tail[0]["history"]
            and row["status"] == "ready"
            and row["decision"] is None
            for row in state_tail
        ),
        "unchanged page/history and consumed decisions",
    )
    need(len(state_tail[0]["history"]) == 3, "three recorded actions")
    need(
        state_tail[0]["page"]["fingerprint"]
        == "766bb4ed66f718dbf1af59a37a0b9855eeb886e31fed8a6206aba6a1824e1d9b",
        "selected observed fingerprint",
    )
    matches = []
    for path in (RAW / "browser/http").glob("*/exchange.json"):
        row = json.loads(path.read_bytes())
        if row["path"] != "/v1/systemone":
            continue
        request_path = path.parent / "request.bin"
        if json.loads(request_path.read_bytes()) != request:
            continue
        need(pin(request_path) == row["request"], "stored HTTP request byte pin")
        response_path = path.parent / "response.bin"
        need(pin(response_path) == row["response"], "stored HTTP response byte pin")
        response = json.loads(response_path.read_bytes())
        need(
            response["answers"] == tail[0]["response"]["answers"],
            "wire/call decoded answer equality",
        )
        matches.append((row["sequence"], row["request"], row["response"], ref(path)))
    matches.sort(key=lambda row: row[0])
    need(
        [row[0] for row in matches] == list(range(67, 78)), "exact eleven consecutive HTTP records"
    )
    need(
        all(
            row[1]
            == {
                "sha256": "8dc4610f10a04c852f51a4539a9d195fccb141efcdae53fbb3210a23b4407b39",
                "size_bytes": 16021,
            }
            for row in matches
        ),
        "identical original wire request bytes",
    )
    response_count = len({row[2]["sha256"] for row in matches})
    need(response_count == 11, "response telemetry differs; no entire-body equality claim")
    return {
        "case_id": "heldout-public-003-000",
        "policy_calls": 16,
        "recorded_action_history_length": 3,
        "stable_tail_call_indices_inclusive": [5, 15],
        "stable_tail_state_indices_inclusive": [5, 16],
        "stable_tail_count": 11,
        "original_HTTP_sequence_inclusive": [67, 77],
        "identical_raw_request": matches[0][1],
        "identical_decoded_answers": True,
        "repeated_selected_operation": "CLICK",
        "repeated_selected_target": "39",
        "distinct_full_response_bodies": response_count,
        "unchanged_full_page_and_history": True,
        "observed_page_fingerprint": state_tail[0]["page"]["fingerprint"],
        "physical_guard_branch": "unrecorded; not proven",
        "physical_execution_from_missing_history_alone": "not inferred",
        "refs": {
            "raw_plan": ref(plan_path),
            "prior_independent_audit": ref(audit_path),
            "states": ref(states_path),
            "model_calls": ref(calls_path),
            "HTTP_exchanges": [row[3] for row in matches],
        },
    }


def isolated_fixture(mode="reject", ticks=3):
    """Execute original Agent+choose bytes; fake only browser/network dependencies."""
    prefix = "_jev_stale_feedback_fixture"
    injected = {}
    saved_env = {key: os.environ.get(key) for key in ("TYPESAFE_API_KEY", "TYPESAFE_MODEL")}
    bodies = []
    browsers = []

    def install(name, module):
        injected[name] = sys.modules.get(name)
        sys.modules[name] = module
        return module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("actual CDP/network/daemon operation forbidden")

    def load(short, filename):
        name = prefix + "." + short
        module = install(name, types.ModuleType(name))
        module.__file__ = str(UPSTREAM / filename)
        module.__package__ = prefix
        exec(
            compile(checked_source(filename), module.__file__, "exec", dont_inherit=True),
            module.__dict__,
        )
        return module

    try:
        package = install(prefix, types.ModuleType(prefix))
        package.__path__ = []
        harness = install("browser_harness", types.ModuleType("browser_harness"))
        harness.__path__ = []
        admin = install("browser_harness.admin", types.ModuleType("browser_harness.admin"))
        helpers = install("browser_harness.helpers", types.ModuleType("browser_harness.helpers"))
        admin.ensure_daemon = forbidden
        helpers.cdp = forbidden
        checked_source("snapshot.js")
        browser_definition = load("browser_definition", "browser.py")
        load("questions", "questions.py")

        class Response:
            status_code = 200
            is_error = False

            def __init__(self, body):
                self.body = body

            def json(self):
                answers = {}
                for name, question in self.body["questions"].items():
                    ids = list(question["criteria"])
                    choice = "CLICK" if name == "operation" else ids[0]
                    answers[name] = {
                        "choice": choice,
                        "confidence": 1.0,
                        "probabilities": {key: float(key == choice) for key in ids},
                    }
                return {"model": "fixture-constant-choice", "answers": answers, "usage": {}}

        class Client:
            def __init__(self, **_kwargs):
                pass

            def post(self, _url, *, json, headers):
                need(
                    headers == {"Authorization": "Bearer fixture-no-credential"},
                    "fixture credential only",
                )
                bodies.append(copy.deepcopy(json))
                return Response(json)

        httpx = install("httpx", types.ModuleType("httpx"))
        httpx.Client = Client
        httpx.HTTPError = type("FixtureHTTPError", (Exception,), {})
        load("model", "model.py")

        class FakeBrowser:
            def __init__(self, _url):
                self.act_attempts = 0
                self.executions = 0
                self.observations = 0
                self.post_stale_pending = False
                self.closed = False
                browsers.append(self)

            def observe(self, screenshot=False):
                need(screenshot is False, "no fixture screenshots")
                if self.post_stale_pending:
                    self.post_stale_pending = False
                    raise browser_definition.StalePage("fixture stale post-input observation")
                self.observations += 1
                changed = mode == "changed_page" and self.observations > 1
                return {
                    "url": "https://fixture.invalid/",
                    "title": "CPU fixture",
                    "text": "Changed fixture" if changed else "Unchanged fixture",
                    "fingerprint": "fixture-changed" if changed else "fixture-stable",
                    "scroll": 0,
                    "actions": [
                        {
                            "id": "fixture-click-a",
                            "kind": "click",
                            "node": 1,
                            "label": "Fixture control A",
                        },
                        {
                            "id": "fixture-click-b",
                            "kind": "click",
                            "node": 2,
                            "label": "Fixture control B",
                        },
                    ],
                }

            def fresh(self, _page, _action=None):
                return True

            def act(self, _action, _page, text=None):
                need(text is None, "no text helper in CPU fixture")
                self.act_attempts += 1
                if mode != "post_input_stale":
                    raise browser_definition.StalePage("fixture pre-input rejection")
                self.executions += 1
                self.post_stale_pending = True

            def close(self):
                self.closed = True

        facade = install(prefix + ".browser", types.ModuleType(prefix + ".browser"))
        facade.Browser = FakeBrowser
        facade.StalePage = browser_definition.StalePage
        agent_module = load("agent", "agent.py")
        os.environ["TYPESAFE_API_KEY"] = "fixture-no-credential"
        os.environ["TYPESAFE_MODEL"] = "fixture-constant-choice"
        agent = agent_module.Agent("https://fixture.invalid/", "Exercise the fixture control")
        try:
            initial_history = copy.deepcopy(agent.state["history"])
            snapshots = [copy.deepcopy(agent.command("tick")) for _ in range(ticks)]
            need(len(browsers) == 1, "one fake browser only")
            return {
                "requests": bodies,
                "snapshots": snapshots,
                "initial_history": initial_history,
                "act_attempts": browsers[0].act_attempts,
                "executions": browsers[0].executions,
            }
        finally:
            agent.close()
    finally:
        for name, original in reversed(list(injected.items())):
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        for key, original in saved_env.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


def with_proposed_attempt_context(
    body, *, operation, target, recorded_action_delta, same_page, consecutive_count
):
    """CPU schema illustration, not an installed adapter or a change to Agent."""
    need(type(recorded_action_delta) is int and recorded_action_delta >= 0, "recorded-action delta")
    need(
        type(same_page) is bool and type(consecutive_count) is int and consecutive_count >= 1,
        "observed facts",
    )
    result = copy.deepcopy(body)
    result["state"]["last_attempt"] = {
        "outcome": "stale_page",
        "operation": operation,
        "target": target,
        "recorded_action_delta": recorded_action_delta,
        "same_page": same_page,
        "consecutive_count": consecutive_count,
    }
    return result


def main():
    actual = observed_evidence()
    before = isolated_fixture()
    changed = isolated_fixture("changed_page", ticks=2)
    after = isolated_fixture("post_input_stale", ticks=1)
    need(
        len({canonical(body) for body in before["requests"]}) == 1,
        "three identical next model bodies",
    )
    need(
        all(
            row["history"] == before["initial_history"]
            and row["decision"] is None
            and row["status"] == "ready"
            for row in before["snapshots"]
        ),
        "swallowed stale clears decision without feedback/action history",
    )
    need(before["executions"] == 0 and before["act_attempts"] == 3, "fake pre-input refusal")
    need(
        len({canonical(body) for body in changed["requests"]}) == 2,
        "changed observed page changes next model request",
    )
    need(
        after["executions"] == 1 and len(after["snapshots"][0]["history"]) == 1,
        "post-input stale retains executed history",
    )
    original = before["requests"][0]
    proposed = with_proposed_attempt_context(
        original,
        operation="CLICK",
        target="1",
        recorded_action_delta=0,
        same_page=True,
        consecutive_count=1,
    )
    restored = copy.deepcopy(proposed)
    del restored["state"]["last_attempt"]
    need(
        restored == original and canonical(proposed) != canonical(original),
        "only added observed attempt context",
    )
    receipt = {
        "schema_id": "jev-browser-stale-feedback-proof-v1",
        "UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "actual_recorded_evidence": actual,
        "original_upstream_source_refs": {name: ref(UPSTREAM / name) for name in PINS},
        "CPU_fixture": {
            "original_Agent_model_questions_browser_bytes_compiled": True,
            "actual_browser_or_HTTP_operations": 0,
            "fake_pre_input_rejection": {
                "ticks": 3,
                "act_attempts": before["act_attempts"],
                "executions": before["executions"],
                "distinct_model_request_bodies": 1,
                "history_unchanged": True,
                "next_decision_context_missing_failure": True,
            },
            "changed_page_control": {"ticks": 2, "distinct_model_request_bodies": 2},
            "post_input_stale_control": {
                "ticks": 1,
                "executions": 1,
                "recorded_actions_retained": 1,
            },
            "proposal_only_adds_separate_state_last_attempt": True,
        },
        "limitations": [
            "Recorded physical guard branch and physical execution of unrecorded attempts are not proven.",
            "Original actual responses share decoded answers; complete response bytes differ because telemetry differs.",
            "Fixture uses synthetic constant legal choices and fake browser I/O; it establishes Agent/request behavior, not actual model improvement.",
            "Proposed separate context is not installed in the selected upstream, scorer or model; no scenario score is claimed.",
            "Safety guards, observed candidate inventory, action history and DOM judges remain unchanged.",
        ],
        "research_decision": ref(HERE / "research-decision.json"),
        "proof_source": ref(Path(__file__).resolve()),
    }
    output = HERE / "proof-receipt.json"
    with output.open("x") as stream:
        json.dump(receipt, stream, allow_nan=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": "mechanism_and_original_wire_repeat_verified",
                "receipt": ref(output),
                "original_HTTP_repeat_count": 11,
                "fixture_identical_requests": 3,
                "actual_physical_guard_branch_proven": False,
            }
        )
    )


if __name__ == "__main__":
    main()
