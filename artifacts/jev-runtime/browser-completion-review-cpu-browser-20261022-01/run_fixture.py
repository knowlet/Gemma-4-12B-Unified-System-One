"""One owned CPU browser composition; all policy/helper/reviewer numbers are fake."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import secrets
import sys
import time
import types
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SELECTED = {
    "helper": (
        "artifacts/jev-runtime/browser-find-text-cpu-browser-20261022-01/run_fixture.py",
        "67bfd8a83068c3d0fa6ea8dc3c5f4edd22f3834eb2a1dab21fbb6dcbd2d01bbc",
    ),
    "integration": (
        "artifacts/jev-runtime/browser-find-text-integration-20261022-01/integration.py",
        "6a8fba79c121ddb24858dce455b7b25fdd74df9de08a804aa452738704edcbef",
    ),
    "selection": (
        "artifacts/jev-runtime/browser-find-text-integration-20261022-01/selection.json",
        "063e5afde12bc408f18be446abedac3e66ff064a6e81858c6b7ce4c995719cb0",
    ),
    "completion": (
        "artifacts/jev-runtime/browser-completion-review-profile-20261022-01/completion_review.py",
        "581ebf1d591aa7c2f2b7fb1fc562e028ea4923f10059e93bdd31095ab305e2e6",
    ),
}


def need(value, message):
    if not value:
        raise ValueError(message)


def encoded(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def source(name):
    label, expected = SELECTED[name]
    path = ROOT / label
    need(not any(p.is_symlink() for p in (path, *path.parents)), "Ordinary checked source")
    blob = path.read_bytes()
    need(
        len(blob) <= 1024**2 and hashlib.sha256(blob).hexdigest() == expected,
        "Exact selected component bytes",
    )
    return path, blob


def module(name):
    path, blob = source(name)
    selected = types.ModuleType("_completion_fixture_" + name)
    selected.__file__ = str(path)
    exec(compile(blob, str(path), "exec", dont_inherit=True), vars(selected))
    return selected


def save(output, name, value):
    with (output / name).open("xb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def source_check():
    for name in SELECTED:
        source(name)
    fixture, integration, completion = module("helper"), module("integration"), module("completion")
    for label in fixture.PINS:
        fixture.checked(ROOT / label)
    integration.selected(
        source("selection")[1], SELECTED["selection"][1], profile=integration.PROFILE
    )
    return fixture, integration, completion


def execute(output):
    fixture, integration, completion = source_check()
    from fastapi.testclient import TestClient
    from playwright.sync_api import sync_playwright

    result = {
        "schema_id": "completion-review-owned-cpu-browser-v1",
        "fixture_only": True,
        "completed": False,
        "quality_score": None,
        "model_qualified": False,
        "generic_native_real_server_route_qualified": False,
        "real_model_calls": 0,
        "comfy_calls": 0,
        "chromium": fixture.binary_pin(),
        "launch_flags": fixture.FLAGS,
        "observed_processes": [],
        "closed_observed_processes": False,
        "browser_close_returned": False,
        "server_and_playwright_contexts_closed": False,
        "whole_tree_exit_claimed": False,
    }
    variables = {
        "TYPESAFE_MODEL": "fixture-find-only",
        "TYPESAFE_API_KEY": "fixture-only",
        "TEXT_MODEL": "generated-actions-none",
        "TEXT_MODEL_API_KEY": "fixture-only",
        "TEXT_MODEL_BASE_URL": "http://fixture.invalid",
        "TEXT_MODEL_REASONING": "none",
    }
    prior = {k: os.environ.get(k) for k in variables}
    os.environ.update(variables)
    primary = None
    ledger = {"policy": [], "review": [], "helper": [], "simulated_native_calls": []}
    try:
        with ExitStack() as stack:
            bundle = stack.enter_context(
                integration.bootstrap_selected(
                    source("selection")[1], SELECTED["selection"][1], profile=integration.PROFILE
                )
            )
            native = bundle.protocol["jev_decision3_native"]

            class FixtureIO(fixture.FixtureIO):
                def __init__(self):
                    super().__init__(native, bundle.wire)
                    self.operations = iter(("DONE", "FIND_TEXT", "DONE"))
                    self.reviews = iter(("INCOMPLETE", "COMPLETE"))

                def predict(self, request, *, deadline, evidence_sink):
                    deadline()
                    payload = self.d3.native_payload(request)
                    need(
                        list(payload["questions"]) == [q.id for q in request.questions],
                        "Native head order preserved",
                    )
                    for q in request.questions:
                        need(
                            list(payload["questions"][q.id]["criteria"]) == q.labels(),
                            "Native candidate order preserved",
                        )
                    self.native_calls.append(copy.deepcopy(payload))
                    is_review = [q.id for q in request.questions] == [completion.QUESTION]
                    chosen = next(self.reviews) if is_review else next(self.operations)
                    answers = {}
                    for question in request.questions:
                        labels = question.labels()
                        choice = chosen if is_review or question.id == "operation" else labels[0]
                        need(choice in labels, "Fixture chooses an actually offered native label")
                        answers[question.id] = {
                            "type": "choice",
                            "choice": choice,
                            "confidence": 1.0,
                            "probabilities": {label: float(label == choice) for label in labels},
                        }
                    ledger["simulated_native_calls"].append(
                        {
                            "scope": "review" if is_review else "policy",
                            "native_payload_sha256": digest(payload),
                            "head_ids": [q.id for q in request.questions],
                            "candidate_labels": {q.id: q.labels() for q in request.questions},
                            "chosen": chosen,
                            "fixture_numbers_only": True,
                        }
                    )
                    evidence_sink("fixture_native_forward", {"fixture_only": True})
                    return {
                        "model": self.spec.model_id.rsplit("/", 1)[-1],
                        "answers": answers,
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    }

            io, server_events = FixtureIO(), []
            app = bundle.create_app(
                io,
                provider="decision3-native",
                model_identity=io.name,
                api_key="fixture-only",
                evidence_sink=server_events.append,
            )
            server_client = stack.enter_context(TestClient(app))
            needle = "completion-owned-" + secrets.token_hex(12) + " literal"
            base, requests = stack.enter_context(
                fixture.server({"/owned": fixture.document([needle])})
            )
            playwright = stack.enter_context(sync_playwright())
            browser = playwright.chromium.launch(
                executable_path=str(fixture.CHROMIUM),
                headless=True,
                args=fixture.FLAGS,
                env={"PATH": os.defpath, "LANG": "C.UTF-8"},
                timeout=15000,
            )
            session = None
            try:
                session = browser.new_browser_cdp_session()

                def observe_cohort():
                    seen = {(x["pid"], x["start_ticks"]) for x in result["observed_processes"]}
                    for raw in session.send("SystemInfo.getProcessInfo")["processInfo"]:
                        row = fixture.proc_identity(int(raw["id"]))
                        if row and (row["pid"], row["start_ticks"]) not in seen:
                            result["observed_processes"].append(row)
                            seen.add((row["pid"], row["start_ticks"]))

                observe_cohort()
                blocked, methods = [], Counter()

                class LocalOnly:
                    def new_page(self):
                        page = browser.new_page()

                        def route(request):
                            if request.request.url.startswith(base + "/"):
                                request.continue_()
                            else:
                                blocked.append(request.request.url)
                                request.abort()

                        page.route("**/*", route)
                        return page

                namespace = {"copy": copy, "time": time, "urlsplit": urlsplit}
                fixture.selected_nodes(
                    ROOT / "scripts/jev_browser_benchmark.py", namespace, ["PlaywrightCDP"]
                )

                class Transport(namespace["PlaywrightCDP"]):
                    def cdp(self, method, session_id=None, **params):
                        methods[method] += 1
                        return super().cdp(method, session_id=session_id, **params)

                transport = Transport(LocalOnly())
                original_post = io.post

                def post(url, *, json, headers):
                    if "questions" not in json:
                        return original_post(url, json=json, headers=headers)
                    need(
                        url == "https://api.typesafe.ai/v1/systemone",
                        "Expected fixture policy route",
                    )
                    raw = server_client.post("/v1/systemone", json=json, headers=headers)
                    need(raw.status_code == 200, "Selected native fixture API succeeded")
                    response = bundle.named_response(json, raw.json())
                    need(list(response["answers"]) == list(json["questions"]), "All heads retained")
                    ledger["policy"].append(
                        {
                            "request_sha256": digest(json),
                            "native_raw_sha256": digest(raw.json()),
                            "head_ids": list(json["questions"]),
                            "answers": copy.deepcopy(response["answers"]),
                            "wire": copy.deepcopy(response["decision3_browser_wire"]),
                        }
                    )
                    return types.SimpleNamespace(
                        status_code=200, is_error=False, json=lambda: response
                    )

                io.post = post
                agent_module, model_module, browser_module = fixture.private_upstream(transport, io)
                io.system_prompt, io.query = model_module.TEXT_VALUE, needle

                def review_native(body, deadline):
                    def remaining():
                        need(time.monotonic() < deadline, "Completion fixture absolute deadline")

                    remaining()
                    need(
                        set(body) == {"model", "state", "questions"} and body["model"] == io.name,
                        "Exact generic callback envelope",
                    )
                    need(
                        list(body["questions"]) == [completion.QUESTION],
                        "Exact generic review head",
                    )
                    data = copy.deepcopy(body)
                    data["questions"] = [
                        dict(raw, id=key) for key, raw in body["questions"].items()
                    ]
                    request = native.Decision3NativeRequest.model_validate(data)
                    payload = io.d3.native_payload(request)
                    need(
                        payload == {"state": body["state"], "questions": body["questions"]},
                        "Full generic request reaches publisher payload unchanged",
                    )
                    raw = io.predict(request, deadline=remaining, evidence_sink=lambda *_: None)
                    need(
                        raw["model"] == io.spec.model_id.rsplit("/", 1)[-1],
                        "Fixture native identity",
                    )
                    normalized = io.d3.normalize_native_response(request, raw)
                    need(
                        list(normalized["answers"]) == [completion.QUESTION],
                        "Generic normalize head order",
                    )
                    remaining()
                    response = {**normalized, "model": io.name}
                    ledger["review"].append(
                        {
                            "request_sha256": digest(body),
                            "native_payload_sha256": digest(payload),
                            "dto_roundtrip_sha256": digest(request.model_dump()),
                            "head_ids": [q.id for q in request.questions],
                            "candidate_labels": {q.id: q.labels() for q in request.questions},
                            "answer": copy.deepcopy(response["answers"][completion.QUESTION]),
                            "visible_goal_text_in_request": needle in body["state"]["page"]["text"],
                            "remaining_seconds_at_return": deadline - time.monotonic(),
                            "generic_real_server_route_used": False,
                        }
                    )
                    return response

                find_class = bundle.agent_class(
                    agent_module, model_module, marker_expression=browser_module.MARKER
                )
                cls = completion.make_agent_class(
                    agent_module, model_module, review_native, base_class=find_class
                )
                with cls(
                    base + "/owned", "Make this exact literal text visible: " + needle
                ) as agent:
                    before = fixture.visible(agent.browser)
                    need(
                        before["in_view"] is False and needle not in agent.state["page"]["text"],
                        "Generated goal initially offscreen and absent from visible input",
                    )
                    first = copy.deepcopy(agent.command("tick"))
                    need(
                        first["status"] == "ready"
                        and first["plan_index"] == 0
                        and first["completion_reviews"][0]["outcome"] == "vetoed",
                        "Premature actual DONE proposal vetoed",
                    )
                    second = copy.deepcopy(agent.command("tick"))
                    after = fixture.visible(agent.browser)
                    need(
                        second["status"] == "ready"
                        and after["in_view"] is True
                        and needle in second["page"]["text"]
                        and second["find_text_feedback"][-1]["status"] == "found",
                        "Actual selected Find scroll exposes generated text",
                    )
                    third = copy.deepcopy(agent.command("tick"))
                    need(
                        third["status"] == "done" and third["plan_index"] == 1,
                        "Visible-evidence fixture review approves final DONE",
                    )
                    need(
                        [x["operation"] for x in third["decisions"]]
                        == ["DONE", "FIND_TEXT", "DONE"]
                        and [x["choice"] for x in third["decisions"]]
                        == ["DONE", "find_text", "DONE"],
                        "All three original proposals retained without forged operation",
                    )
                    need(
                        [x["operation"] for x in third["history"]] == ["DONE", "FIND_TEXT", "DONE"]
                        and [x["kind"] for x in third["history"]]
                        == ["completion_review", "find_text", "completion_review"],
                        "Two review records plus exactly one actual navigation",
                    )
                    need(
                        [x["outcome"] for x in third["completion_reviews"]]
                        == ["vetoed", "approved"]
                        and [x["answer"]["choice"] for x in third["completion_reviews"]]
                        == ["INCOMPLETE", "COMPLETE"],
                        "Exact native fixture review results",
                    )
                    need(
                        len(ledger["policy"]) == 3
                        and len(ledger["review"]) == 2
                        and len(io.text) == 1
                        and len(io.native_calls) == 5,
                        "Exactly three policy, two review and one helper substitution",
                    )
                    need(
                        third["completion_policy_attempts"] == 3
                        and len(third["history"]) == 3
                        and len(third["history"]) <= 16
                        and third["completion_policy_attempts"]
                        + sum(x["native_call_attempted"] for x in third["completion_reviews"])
                        == 5,
                        "Shared sixteen-action/thirty-two-logical-slot accounting",
                    )
                    need(
                        [x["visible_goal_text_in_request"] for x in ledger["review"]]
                        == [False, True],
                        "Reviewer receives observed offscreen/visible text only",
                    )
                    need(
                        io.text[0]["schema_valid"]
                        and io.text[0]["query"]
                        == third["find_text_feedback"][-1]["query"]
                        == needle,
                        "Strict TEXT_VALUE helper equals actual Find query",
                    )
                    need(
                        not any(k.startswith("Input.") for k in methods),
                        "No forged physical control input",
                    )
                    result.update(
                        completed=True,
                        before=before,
                        after=after,
                        decision_operations=[x["operation"] for x in third["decisions"]],
                        history_kinds=[x["kind"] for x in third["history"]],
                        final_status=third["status"],
                        policy_attempt_slots=3,
                        review_attempt_slots=2,
                        total_native_logical_slots=5,
                        helper_calls=1,
                        history_slots=3,
                        review_outcomes=[x["outcome"] for x in third["completion_reviews"]],
                        task_success_claimed=False,
                        generated_query_sha256=hashlib.sha256(needle.encode()).hexdigest(),
                        cdp_method_counts=dict(methods),
                        local_document_requests=len(requests),
                        blocked_nonlocal_requests=len(blocked),
                        native_api_server_returned_events=sum(
                            x["event"] == "server_returned" for x in server_events
                        ),
                    )
                    ledger["helper"] = [
                        {
                            "request_sha256": x["request_sha256"],
                            "context_sha256": x["context_sha256"],
                            "schema_valid": x["schema_valid"],
                            "query_sha256": hashlib.sha256(x["query"].encode()).hexdigest(),
                        }
                        for x in io.text
                    ]
                need(
                    "torch" not in sys.modules and "transformers" not in sys.modules,
                    "No framework/model imports",
                )
            finally:
                errors = []
                if session is not None:
                    try:
                        observe_cohort()
                    except Exception as error:
                        errors.append({"stage": "final_cohort", "type": type(error).__name__})
                try:
                    browser.close()
                    result["browser_close_returned"] = True
                except Exception as error:
                    errors.append({"stage": "browser_close", "type": type(error).__name__})
                result["cleanup_errors"] = errors
                if errors and sys.exc_info()[0] is None:
                    raise ValueError("Owned browser cleanup incomplete")
        result["server_and_playwright_contexts_closed"] = True
    except BaseException as error:
        primary = error
        result.update(completed=False, error_type=type(error).__name__, error_message=str(error))
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        until = time.monotonic() + 2.0
        while True:
            closure = []
            for row in result["observed_processes"]:
                after = fixture.proc_identity(row["pid"])
                closure.append(
                    {
                        "before": row,
                        "after": after,
                        "same_process_absent": after is None
                        or after["start_ticks"] != row["start_ticks"],
                    }
                )
            if all(x["same_process_absent"] for x in closure) or time.monotonic() >= until:
                break
            time.sleep(0.02)
        result["process_closure"] = closure
        result["closed_observed_processes"] = bool(closure) and all(
            x["same_process_absent"] for x in closure
        )
        save(output, "join-ledger.json", ledger)
        save(output, "receipt.json", result)
    if primary:
        raise primary
    need(result["closed_observed_processes"], "Observed Chromium cohort exit unproven")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "actual-01")
    args = parser.parse_args(argv)
    source_check()
    if not args.execute:
        print(
            json.dumps(
                {
                    "fixture_only": True,
                    "actual_browser_started": False,
                    "quality_score": None,
                    "cases": 1,
                }
            )
        )
        return 0
    output = args.output.absolute()
    need(
        output.parent == HERE
        and not output.exists()
        and not any(x.is_symlink() for x in (output, *output.parents)),
        "Fresh own ordinary output",
    )
    output.mkdir(mode=0o700)
    result = execute(output)
    print(
        json.dumps(
            {
                "completed": result["completed"],
                "closed_observed_processes": result["closed_observed_processes"],
                "quality_score": None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
