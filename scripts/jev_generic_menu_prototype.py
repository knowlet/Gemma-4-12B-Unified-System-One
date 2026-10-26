"""Small owned-browser menu/scroll proof; no model, fit or public admission.

Native payloads come from the unchanged Jev choose() request constructor. An
in-process capture stops before its HTTP client. Labels are published only
after the selected native input reaches its independent fixture postcondition.
The legacy public DTO/projection is checked for serialization compatibility;
these loopback fixtures are not an admitted public collection or fit bundle.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import importlib.abc
import importlib.util
import json
import os
import re
import sys
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from s1.browser_public_training import NativeWireRequest, public_training_case  # noqa: E402
from s1.browser_public_training_validation import public_compile_wire_projection  # noqa: E402

PROFILE = "composed-visible-hover-entries-v5"
PROFILE_SHA = "7847b2f9eecb3f9d492ccfdf8415c52632b99ac6ead6df560c49c3caf004e141"
COLLECTOR_SHA = "5cdfa3ffb8b1077eb5cd67c4910fb8b1759005258faf11b0c2012ed73fac0f4b"
CHROMIUM = Path(
    "/home/albert_hsieh/.cache/ms-playwright/chromium_headless_shell-1208/chrome-linux/headless_shell"
)
MODEL_ALIAS = "generated-actions-none"
FORBIDDEN = {"torch", "transformers", "peft", "bitsandbytes", "tensorflow", "jax", "safetensors"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def write_rows(path, rows):
    with Path(path).open("xb") as stream:
        for row in rows:
            stream.write(encode(row) + b"\n")


def scenarios():
    """Eight independently authored scenes, not imported evaluation tasks."""
    leaves = [
        "Maproom",
        "Folio tray",
        "Waypoint shelf",
        "Chapter desk",
        "Reading cabinet",
        "Trail ledger",
        "Plain journal",
        "Quiet workbook",
    ]
    far = [
        "Copper meadow",
        "Amber quay",
        "Lilac ridge",
        "Silver marsh",
        "Ivory grove",
        "Cobalt glade",
        "Ochre coast",
        "Pearl orchard",
    ]
    near = [
        "Basil notes",
        "Saffron ledger",
        "Juniper sketches",
        "Cedar diary",
        "Hazel pages",
        "Willow entries",
        "Birch records",
        "Spruce notes",
    ]
    kinds = ["hover", "button", "hover", "button", "hover", "button", "none", "none"]
    prior = [0, 1, 3, 0, 1, 3, 1, 0]
    return [
        {
            "id": f"scene-{i + 1:02d}",
            "split": "train" if i in {0, 1, 2, 3, 6} else "development",
            "layout": i,
            "menu_kind": kinds[i],
            "prior_scrolls": prior[i],
            "menu_label": leaves[i],
            "link_label": "Visit " + far[i],
            "near_label": near[i],
            "far_label": far[i],
        }
        for i in range(8)
    ]


def render_fixture(spec):
    require(spec in scenarios(), "Only the eight declared fixture scenes are allowed")
    esc = html.escape
    i, kind = spec["layout"], spec["menu_kind"]
    near_y = spec["prior_scrolls"] * 560 + 900
    far_y = near_y + 4000
    left, top = [
        (32, 18),
        (720, 30),
        (64, 88),
        (600, 100),
        (360, 20),
        (790, 92),
        (50, 20),
        (50, 20),
    ][i]
    menu = ""
    if kind != "none":
        trigger = (
            f'<span id="entry">{esc(spec["menu_label"])}</span>'
            if kind == "hover"
            else f"<button id=\"entry\" aria-expanded=\"false\" onclick=\"document.getElementById('popup').hidden=false;this.setAttribute('aria-expanded','true')\">{esc(spec['menu_label'])}</button>"
        )
        hidden = "" if kind == "hover" else " hidden"
        menu = f'<div class="navbox">{trigger}<div class="popup" id="popup"{hidden}><a id="destination-link" href="#far">{esc(spec["link_label"])}</a></div></div>'
    hover = ".popup{display:none}.navbox:hover .popup{display:block}" if kind == "hover" else ""
    colors = [
        "#f2e7dc",
        "#e4edf6",
        "#eee8f5",
        "#e1ede7",
        "#f5ecd8",
        "#e4e9f2",
        "#edf0e0",
        "#f3e5ea",
    ]
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Owned reading scene {i + 1}</title>
<style>html{{scroll-behavior:auto}}body{{margin:0;font:18px sans-serif;background:{colors[i]}}}
.navbox{{position:fixed;left:{left}px;top:{top}px;z-index:5;width:235px;height:48px;background:white;border:2px solid #42546b}}
#entry{{display:inline-block;padding:12px;font:18px sans-serif;cursor:pointer}}
.popup{{position:absolute;top:46px;left:0;width:270px;background:white;border:2px solid #42546b;padding:14px}}
.popup a{{display:block;padding:10px;color:#153b67}}{hover}
main{{position:relative;height:{far_y + 1400}px}}h1{{position:absolute;top:165px;left:{40 + i * 7}px}}
section{{position:absolute;left:{55 + i * 9}px;width:750px;padding:20px;border:2px solid #56616c;background:white;scroll-margin-top:330px}}
#near{{top:{near_y}px}}#far{{top:{far_y}px}}aside{{position:fixed;right:12px;bottom:12px;background:white;padding:8px}}
</style></head><body>{menu}<main><h1>Reading collection {i + 1}</h1>
<section id="near"><h2>{esc(spec["near_label"])}</h2><p>A nearby page about everyday materials and changing light.</p></section>
<section id="far"><h2>{esc(spec["far_label"])}</h2><p>A distant chapter with a separate heading and a complete reading paragraph.</p></section>
</main><aside><label for="note">Reader scratchpad {i + 1}</label><input id="note" value="unchanged"></aside>
<script>window.ownedEvents=[];for(const type of ['click','input','change'])document.addEventListener(type,e=>ownedEvents.push({{type,target:e.target.id,trusted:e.isTrusted,hash:location.hash,scrollY}}));</script>
</body></html>"""


def normalized_pair_payload(payload):
    wire = NativeWireRequest.model_validate(payload)
    result = wire.model_dump(mode="json")
    goals = []
    for question in result["questions"].values():
        instructions = question.get("instructions")
        require(type(instructions) is dict, "Every native head must have instructions")
        goal = instructions.get("goal")
        require(type(goal) is str and bool(goal.strip()), "Every native head must have a goal")
        goals.append(goal)
        instructions["goal"] = "<independent-goal>"
    require(len(set(goals)) == 1, "Native head goals must agree")
    return encode(result)


def validate_pair(left, right):
    require(
        left["branch"] == "menu" and right["branch"] == "scroll",
        "Only menu/scroll contrasts form a pair",
    )
    for key in ("scenario_id", "split", "html_sha256", "history"):
        require(encode(left[key]) == encode(right[key]), "Pair initialization differs: " + key)
    require(re.fullmatch(r"[0-9a-f]{64}", left["html_sha256"]), "Actual initial HTML SHA required")
    require(
        left["reachability_verified"] is True and right["reachability_verified"] is True,
        "Both native branches must be proved",
    )
    require(
        left["selected"]["operation"] == "CLICK"
        and right["selected"] == {"operation": "SCROLL_DOWN", "target": None},
        "Pair must contrast actual menu CLICK and SCROLL_DOWN",
    )
    first, second = (
        normalized_pair_payload(left["payload"]),
        normalized_pair_payload(right["payload"]),
    )
    require(
        first == second and encode(left["payload"]) != encode(right["payload"]),
        "Full native payload must differ only by goal",
    )
    labels = NativeWireRequest.model_validate(left["payload"]).candidate_labels
    require("SCROLL_DOWN" in labels["operation"], "Pair scroll operation was not offered")
    for record in (left, right):
        require(
            record["payload"]["state"]["recent_actions"] == record["history"][-10:],
            "Pair history differs from actual native request",
        )
    require(left["selected"]["target"] in labels["click_target"], "Menu target was not offered")
    return {
        "scenario_id": left["scenario_id"],
        "normalized_payload_sha256": sha(first),
        "goals_differ": True,
    }


def build_case_and_wire(record, proof_refs):
    require(
        record["reachability_verified"] is True,
        "A real reached postcondition is required before annotation",
    )
    selected = record["selected"]
    case_id = record["episode_id"] + "-initial"
    pair_id = "owned-generic-menu-" + record["scenario_id"]
    operation, target = selected.get("operation"), selected.get("target")
    case = public_training_case(
        record["payload"],
        case_id=case_id,
        group_id="public-native-site:owned-generic-menu-" + record["scenario_id"],
        task_id=record["scenario_id"],
        split=record["split"],
        selected=selected,
        valid_targets={
            operation: [target] if operation in {"CLICK", "TYPE_TEXT", "SELECT"} else []
        },
        wire_row_id=case_id + "-wire",
        episode_id=record["episode_id"],
        pair_id=pair_id,
        state_role="ordinary_incomplete",
        proof_refs=proof_refs,
    )
    wire = {
        "schema_id": "jev-public-native-action-wire-v1",
        "wire_row_id": case.wire_row_id,
        "case_id": case.id,
        "episode_id": case.episode_id,
        "pair_id": pair_id,
        "native_payload": copy.deepcopy(record["payload"]),
        "native_wire_sha256": case.request.wire_sha256,
        "teacher_choice": copy.deepcopy(selected),
        "valid_next_targets": copy.deepcopy(case.valid_next_targets),
        "native_policy_request_sequence": 0,
        "compiled_input_sequence": 0,
        "proof_refs": {k: v.model_dump(mode="json") for k, v in case.proof_refs.items()},
        "label_source": "Owned local native reachability prototype; serialization compatibility only; not public admission",
    }
    return case, wire


class RejectModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise RuntimeError("No model/weight package is allowed: " + fullname)


def memory_admission():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            amount, unit = value.split()
            require(unit == "kB", "Unexpected memory unit")
            values[key] = int(amount) * 1024
    reserve = max(24 * 1024**3, (values["MemTotal"] + 4) // 5)
    require(
        values["MemAvailable"] >= reserve + 3 * 1024**3,
        "Fresh host reserve plus3GiB for owned browser/tokenizer required",
    )
    return {
        **values,
        "reserve_bytes": reserve,
        "owned_browser_tokenizer_bytes": 3 * 1024**3,
        "swap_credit": 0,
    }


PRIVATE = """(() => {
 const info=id=>{const e=document.getElementById(id);if(!e)return null;const r=e.getBoundingClientRect();
 return {id,text:e.innerText,visible:e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true}),
 rect:{x:r.x,y:r.y,w:r.width,h:r.height},unoccluded_center:e.contains(document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)),in_view:r.top>=0&&r.bottom<=innerHeight&&r.left>=0&&r.right<=innerWidth};};
 return {url:location.href,title:document.title,ready:document.readyState,scroll_y:scrollY,width:innerWidth,height:innerHeight,
 entry:info('entry'),link:info('destination-link'),near:info('near'),far:info('far'),note:document.getElementById('note').value,
 events:structuredClone(window.ownedEvents),html:document.documentElement.outerHTML};})()"""


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), FixtureHandler)
        self.documents = {"/" + spec["id"]: render_fixture(spec).encode() for spec in scenarios()}
        self.ledger = []


class FixtureHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.respond()

    def do_HEAD(self):
        self.respond(head_only=True)

    def respond(self, head_only=False):
        parsed = urlsplit(self.path)
        blob = self.server.documents.get(parsed.path)
        status = 200 if blob is not None and not parsed.query else 404
        self.server.ledger.append({"method": self.command, "path": self.path, "status": status})
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(blob) if status == 200 else 0))
        self.end_headers()
        if status == 200 and not head_only:
            self.wfile.write(blob)

    def log_message(self, *_args):
        pass


class PayloadCaptured(BaseException):
    pass


def capture_request(model, page, goal, history):
    payloads = []

    def capture(_url, _key, body):
        payloads.append(copy.deepcopy(body))
        raise PayloadCaptured()

    previous = model.post_json
    previous_model = os.environ.get("TYPESAFE_MODEL")
    previous_key = os.environ.get("TYPESAFE_API_KEY")
    os.environ["TYPESAFE_MODEL"] = MODEL_ALIAS
    os.environ["TYPESAFE_API_KEY"] = "owned-local-capture-no-http"
    model.post_json = capture
    try:
        try:
            model.choose(page, goal, history)
        except PayloadCaptured:
            pass
        require(len(payloads) == 1, "Exactly one original native request must be captured")
        wire = NativeWireRequest.model_validate(payloads[0])
        require(wire.wire_bytes == encode(payloads[0]), "Native wire serialization changed")
        return payloads[0]
    finally:
        model.post_json = previous
        if previous_model is None:
            os.environ.pop("TYPESAFE_MODEL", None)
        else:
            os.environ["TYPESAFE_MODEL"] = previous_model
        if previous_key is None:
            os.environ.pop("TYPESAFE_API_KEY", None)
        else:
            os.environ["TYPESAFE_API_KEY"] = previous_key


def selected_for(model, page, action):
    _, targets, controls = model.action_space(page["actions"])
    if action["kind"] == "click":
        matches = [key for key, value in targets["CLICK"].items() if value == action]
        require(len(matches) == 1, "Unique actual native CLICK target required")
        return {"operation": "CLICK", "target": matches[0]}
    require(
        action["id"] == "scroll_down" and controls["SCROLL_DOWN"] == action,
        "Original offered scroll required",
    )
    return {"operation": "SCROLL_DOWN", "target": None}


def find_action(page, *, label=None, action_id=None):
    values = [
        a
        for a in page["actions"]
        if (a["label"] == label if label is not None else a["id"] == action_id)
    ]
    require(len(values) == 1, "One genuinely offered action required")
    return values[0]


def actual_step(browser, transport, page, action, history):
    before = browser.evaluate(PRIVATE)
    require(browser.fresh(page, action) is True, "Action must be fresh before original executor")
    first_input = len(transport.inputs)
    result = browser.act(action, page)
    require(
        result == {"executed": action["id"]},
        "Original executor must confirm the actual offered action",
    )
    observed = browser.observe(screenshot=False)
    after = browser.evaluate(PRIVATE)
    require(after["width"] == 1120 and after["height"] == 780, "Viewport changed")
    require(after["note"] == "unchanged", "Unused input must be preserved")
    inputs = copy.deepcopy(transport.inputs[first_input:])
    expected = ["mousePressed", "mouseReleased"] if action["kind"] == "click" else ["mouseWheel"]
    require(
        [row["params"]["type"] for row in inputs] == expected, "Unexpected or replayed native input"
    )
    history.append(
        {
            "action": action["label"],
            "kind": action["kind"],
            "text": None,
            "page_changed": observed["fingerprint"] != page["fingerprint"],
        }
    )
    return observed, {
        "action": copy.deepcopy(action),
        "before": before,
        "after": after,
        "fresh": True,
        "executor_result": result,
        "inputs": inputs,
    }


def run_episode(
    spec, branch, origin, chromium, harness, verified, upstream, compiler, checker, folder
):
    folder.mkdir()
    calls, requests, transitions, raw, inputs, blocked, posts, network = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    html_blob = render_fixture(spec).encode()
    try:
        with chromium.new_context(
            viewport={"width": 1120, "height": 780}, service_workers="block"
        ) as context:

            def route_owned(route):
                request = route.request
                url = urlsplit(request.url)
                valid = (
                    request.method in {"GET", "HEAD"}
                    and url.scheme == "http"
                    and url.hostname == "127.0.0.1"
                    and url.port == urlsplit(origin).port
                    and url.username is None
                    and url.password is None
                )
                if valid:
                    route.continue_()
                else:
                    blocked.append({"method": request.method, "url": request.url})
                    route.abort("blockedbyclient")

            context.route("**/*", route_owned)
            context.on(
                "response",
                lambda response: network.append(
                    {
                        "url": response.url,
                        "method": response.request.method,
                        "status": response.status,
                    }
                ),
            )

            class Transport(harness.PlaywrightCDP):
                def cdp(self, method, session_id=None, **params):
                    row = {
                        "sequence": len(calls),
                        "method": method,
                        "session": session_id,
                        "params": copy.deepcopy(params),
                    }
                    calls.append(row)
                    try:
                        result = super().cdp(method, session_id=session_id, **params)
                        row["response"] = copy.deepcopy(result)
                        if method.startswith("Input."):
                            inputs.append(copy.deepcopy(row))
                        return result
                    except BaseException as exc:
                        row["error"] = {"type": type(exc).__name__, "message": str(exc)}
                        raise

            transport = Transport(context)
            transport.inputs = inputs
            transport.observation_expression = verified.read_state
            transport.observation_sink = lambda event: raw.append(copy.deepcopy(event))
            with harness.load_upstream(upstream, transport, observation_runtime=verified) as (
                _,
                model,
            ):
                native = sys.modules["jev_ultrafast.browser"]

                def deny(*_args, **_kwargs):
                    posts.append(True)
                    raise AssertionError("No policy/helper HTTP POST or field helper is allowed")

                model.CLIENT.post = deny
                model.field_text = deny
                browser = native.Browser(origin + "/" + spec["id"])
                history = []
                try:
                    page = browser.observe(screenshot=False)
                    for _ in range(spec["prior_scrolls"]):
                        page, step = actual_step(
                            browser,
                            transport,
                            page,
                            find_action(page, action_id="scroll_down"),
                            history,
                        )
                        step["phase"] = "real_prior_scroll"
                        transitions.append(step)
                    require(
                        page["scroll"]["y"] == spec["prior_scrolls"] * 560,
                        "Actual prior scroll position differs",
                    )
                    initial = browser.evaluate(PRIVATE)
                    initial_raw_sequence = len(raw) - 1
                    require(
                        initial_raw_sequence >= 0,
                        "Initial page needs its original raw READ_STATE response",
                    )
                    raw_initial = raw[initial_raw_sequence]["response"]["result"]["value"]
                    require(
                        all(page.get(key) == value for key, value in raw_initial.items()),
                        "Initial native page differs from raw observer response",
                    )
                    require(
                        initial["near"]["in_view"] is False and initial["far"]["in_view"] is False,
                        "Target must not already be reached",
                    )
                    require(
                        (initial["entry"] is None) == (spec["menu_kind"] == "none"),
                        "Fixture menu state differs",
                    )
                    goal = (
                        "Use the reading directory to bring the "
                        + spec["far_label"]
                        + " chapter and paragraph into view."
                        if branch == "menu"
                        else "Continue down the document until the upcoming "
                        + spec["near_label"]
                        + " section and paragraph are in view."
                    )
                    payload = capture_request(model, page, goal, history)
                    action = (
                        find_action(page, label=spec["menu_label"])
                        if branch == "menu"
                        else find_action(page, action_id="scroll_down")
                    )
                    selected = selected_for(model, page, action)
                    initial_history = copy.deepcopy(history)
                    initial_page = copy.deepcopy(page)

                    def record_request(current, phase):
                        compiled = {"sequence": len(requests), **compiler.compile(current)}
                        checker.check(current, compiled)
                        requests.append(
                            {
                                "sequence": len(requests),
                                "phase": phase,
                                "native_payload": copy.deepcopy(current),
                                "native_wire_sha256": sha(encode(current)),
                                "compiled": compiled,
                            }
                        )

                    record_request(payload, "initial")
                    page, step = actual_step(browser, transport, page, action, history)
                    step["phase"] = "initial_selected_input"
                    transitions.append(step)
                    if branch == "menu":
                        require(
                            step["after"]["link"]["visible"] and step["after"]["link"]["in_view"],
                            "Entry input must reveal a real visible link",
                        )
                        require(
                            any(
                                e["target"] == "entry" and e["trusted"]
                                for e in step["after"]["events"]
                            ),
                            "Entry must receive a trusted native click",
                        )
                        link = find_action(page, label=spec["link_label"])
                        require(link["role"] == "link", "Second action must be a real link")
                        record_request(
                            capture_request(model, page, goal, history), "reachability_visible_link"
                        )
                        page, step = actual_step(browser, transport, page, link, history)
                        step["phase"] = "visible_link_input"
                        transitions.append(step)
                        require(
                            step["after"]["url"].endswith("#far")
                            and step["after"]["far"]["visible"]
                            and step["after"]["far"]["in_view"]
                            and step["after"]["far"]["unoccluded_center"],
                            "Native visible link must reach a fresh visible anchor section",
                        )
                        require(
                            any(
                                e["target"] == "destination-link" and e["trusted"]
                                for e in step["after"]["events"]
                            ),
                            "Anchor link must receive a trusted native click",
                        )
                    else:
                        require(
                            step["after"]["near"]["visible"]
                            and step["after"]["near"]["in_view"]
                            and step["after"]["near"]["unoccluded_center"],
                            "Original scroll must reveal the near section",
                        )
                        require(
                            not step["after"]["url"].endswith("#far"),
                            "Scroll branch must not navigate to a hidden anchor",
                        )
                        require(
                            not any(
                                e["target"] in {"entry", "destination-link"}
                                for e in step["after"]["events"]
                            ),
                            "Scroll branch must not open unrelated menu",
                        )
                    require(
                        browser.fresh(page) is True, "Reached final native page must still be fresh"
                    )
                    record = {
                        "scenario_id": spec["id"],
                        "split": spec["split"],
                        "episode_id": spec["id"] + "-" + branch,
                        "branch": branch,
                        "html_sha256": sha(html_blob),
                        "history": initial_history,
                        "payload": payload,
                        "selected": selected,
                        "reachability_verified": True,
                    }
                    proof = {
                        "schema": "owned-generic-native-branch-proof-v1",
                        "record": record,
                        "initial_private": initial,
                        "initial_native_page": initial_page,
                        "initial_raw_sequence": initial_raw_sequence,
                        "final_native_page": page,
                        "transitions": transitions,
                        "requests": requests,
                        "inputs": inputs,
                        "raw_observations": raw,
                        "model_http_posts": len(posts),
                        "blocked_requests": blocked,
                        "ownership": {
                            "target": browser.target,
                            "session": browser.session,
                            "native_page_key": initial_page["page_key"],
                            "url": initial_page["url"],
                        },
                    }
                finally:
                    browser.close()
                    model.CLIENT.close()
                require(
                    not posts and not blocked,
                    "No policy/helper POST or out-of-scope request allowed",
                )
                require(not transport.observation_profile_errors, "Profile integrity failure")
        (folder / "initial.html").write_bytes(html_blob)
        write(folder / "proof.json", proof)
        write_rows(folder / "native-cdp.jsonl", calls)
        write_rows(folder / "observations.jsonl", raw)
        write_rows(folder / "private-dom.jsonl", [initial, transitions[-1]["after"]])
        write_rows(folder / "transitions.jsonl", transitions)
        write_rows(folder / "ownership.jsonl", [proof["ownership"]])
        write_rows(folder / "native-requests.jsonl", requests)
        write(
            folder / "local-request-ledger.json",
            {
                "scope": "owned loopback only; not a public-site ledger",
                "responses": network,
                "blocked": blocked,
            },
        )
        return record, proof
    except BaseException as exc:
        write(
            folder / "failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "positive_annotation_published": False,
            },
        )
        write_rows(folder / "failed-native-cdp.jsonl", calls)
        write_rows(folder / "failed-observations.jsonl", raw)
        write_rows(folder / "failed-requests.jsonl", requests)
        write_rows(folder / "failed-transitions.jsonl", transitions)
        raise


def run_prototype(output, *, expected_source_sha256, upstream=None, chromium_executable=CHROMIUM):
    require(
        sha(Path(__file__).read_bytes()) == expected_source_sha256,
        "Reviewed prototype source changed",
    )
    require(
        not any(name.split(".")[0] in FORBIDDEN for name in sys.modules),
        "No model package may be resident in this proof interpreter",
    )
    output = Path(output).absolute()
    require(
        not output.exists() and not any(p.is_symlink() for p in (output, *output.parents)),
        "Fresh ordinary output required",
    )
    upstream = Path(upstream or ROOT / "artifacts/jev-runtime/upstream")
    output.mkdir(parents=True)
    write(
        output / "started.json",
        {
            "scope": "owned local proof only",
            "source_sha256": expected_source_sha256,
            "memory": memory_admission(),
        },
    )
    reject = RejectModels()
    sys.meta_path.insert(0, reject)
    before_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    primary = None
    server = thread = None
    try:
        import jev_browser_collect_public_training as collector
        import jev_browser_profile_v5_benchmark as wrapper
        import jev_observation_profile_v5 as profile
        from playwright.sync_api import sync_playwright

        from s1.browser_public_training_validation import _CachedNativeCompiler

        require(
            sha(Path(collector.__file__).read_bytes()) == COLLECTOR_SHA
            and sha(Path(profile.__file__).read_bytes()) == PROFILE_SHA,
            "Pinned collector/profile sources changed",
        )
        metadata = {
            name: {
                "sha256": sha((collector.CACHE / name).read_bytes()),
                "size_bytes": (collector.CACHE / name).stat().st_size,
            }
            for name in collector.ASSET_SHA
        }
        require(
            {name: value["sha256"] for name, value in metadata.items()} == collector.ASSET_SHA,
            "Existing offline compiler metadata changed",
        )
        compiler, checker = (
            collector.NativeCompiler(),
            _CachedNativeCompiler(collector.CACHE, metadata),
        )
        bundle, bundle_manifest = wrapper.prepare_bundle(output / "runner", case_module="original")
        prepared = profile.materialize_profile(
            upstream, output / "profile", expected_source_sha256=PROFILE_SHA
        )
        verified = profile.verify_materialized_profile(
            output / "profile",
            expected_manifest_sha256=prepared["manifest_sha256"],
            expected_source_sha256=PROFILE_SHA,
        )
        spec = importlib.util.spec_from_file_location(
            "owned_generic_v5_runner", bundle / "runner.py"
        )
        harness = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = harness
        spec.loader.exec_module(harness)
        require(
            harness.upstream_manifest(upstream)["source_sha256"] == dict(profile.CANONICAL_SHA256),
            "Original canonical Git source differs",
        )
        server = FixtureServer()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        origin = f"http://127.0.0.1:{server.server_port}"
        records, proofs, pairs, cases, wires = [], [], [], [], []
        with sync_playwright() as runtime:
            owned = runtime.chromium.launch(
                executable_path=str(chromium_executable),
                headless=True,
                args=[
                    "--disable-gpu",
                    "--disable-software-rasterizer",
                    "--disable-dev-shm-usage",
                    "--disable-background-networking",
                ],
            )
            try:
                for scene in scenarios():
                    memory_admission()
                    branches = ["menu", "scroll"] if scene["menu_kind"] != "none" else ["control"]
                    current = []
                    for branch in branches:
                        folder = output / (scene["id"] + "-" + branch)
                        record, proof = run_episode(
                            scene,
                            branch,
                            origin,
                            owned,
                            harness,
                            verified,
                            upstream,
                            compiler,
                            checker,
                            folder,
                        )
                        records.append(record)
                        proofs.append(proof)
                        current.append(record)

                        def reference(name, sequence):
                            return {
                                "artifact_path": folder.name + "/" + name,
                                "sha256": sha((folder / name).read_bytes()),
                                "sequence": sequence,
                            }

                        refs = {
                            "raw_observation": reference(
                                "observations.jsonl", proof["initial_raw_sequence"]
                            ),
                            "private_precondition": reference("private-dom.jsonl", 0),
                            "guarded_transition": reference(
                                "transitions.jsonl", scene["prior_scrolls"]
                            ),
                            "private_postcondition": reference("private-dom.jsonl", 1),
                            "ownership": reference("ownership.jsonl", 0),
                            "public_ledger": reference("local-request-ledger.json", 0),
                        }
                        case, wire = build_case_and_wire(record, refs)
                        cases.append(case)
                        wires.append(wire)
                    if len(current) == 2:
                        pair = validate_pair(*current)
                        first_html, second_html = (
                            proof["initial_private"]["html"] for proof in proofs[-2:]
                        )
                        require(
                            first_html.encode() == second_html.encode(),
                            "Actual initial DOM differs between goal-only branches",
                        )
                        pair.update(
                            actual_initial_DOM_bytes_equal=True,
                            actual_initial_DOM_sha256=sha(first_html.encode()),
                        )
                        pairs.append(pair)
            finally:
                owned.close()
        projection = public_compile_wire_projection(cases, wires)
        require(
            len(projection) == 14 and len(pairs) == 6,
            "The fixed small prototype population changed",
        )
        for case, wire, projected in zip(cases, wires, projection, strict=True):
            require(
                encode(case.request.model_dump(mode="json"))
                == encode(wire["native_payload"])
                == encode(projected["payload"]),
                "Compiler projection changed original wire",
            )
        write_rows(
            output / "compatibility-cases.jsonl", [case.model_dump(mode="json") for case in cases]
        )
        write_rows(output / "compatibility-wire-requests.jsonl", wires)
        write(output / "pair-proof.json", pairs)
        write(output / "local-request-ledger.json", server.ledger)
        counts = {}
        for split in ("train", "development"):
            subset = [case for case in cases if case.split == split]
            request_rows = [
                r
                for record, proof in zip(records, proofs, strict=True)
                if record["split"] == split
                for r in proof["requests"]
            ]
            counts[split] = {
                "initial_DTO_requests": len(subset),
                "all_actual_native_requests": len(request_rows),
                "offered_heads_initial": sum(len(case.request.candidate_labels) for case in subset),
                "offered_heads_all_actual_requests": sum(
                    len(r["native_payload"]["questions"]) for r in request_rows
                ),
                "supervised_gold_heads_initial": sum(len(case.gold) for case in subset),
                "initial_teacher_operations": dict(
                    Counter(case.gold["operation"] for case in subset)
                ),
            }
        wrapper.verify_bundle(bundle, bundle_manifest)
        profile.verify_materialized_profile(
            output / "profile",
            expected_manifest_sha256=prepared["manifest_sha256"],
            expected_source_sha256=PROFILE_SHA,
        )
        require(
            sha(Path(__file__).read_bytes()) == expected_source_sha256,
            "Prototype source changed during proof",
        )
        receipt = {
            "schema": "owned-generic-menu-prototype-receipt-v1",
            "status": "complete_local_proof_not_public_admission",
            "source_sha256": expected_source_sha256,
            "profile": PROFILE,
            "profile_source_sha256": PROFILE_SHA,
            "profile_manifest_sha256": prepared["manifest_sha256"],
            "scenarios": scenarios(),
            "scene_count": 8,
            "menu_pair_count": 6,
            "no_menu_control_count": 2,
            "episode_count": 14,
            "request_and_head_grain": counts,
            "zero_policy_helper_HTTP_POSTs": True,
            "model_or_training_calls": 0,
            "external_network_calls": 0,
            "public_collection_admitted": False,
            "fit_bundle_created": False,
            "unchanged_DTO_and_native_compile_projection": True,
            "actual_offline_original_compiler_and_independent_cached_check": True,
            "compiler_metadata": metadata,
            "goal_only_full_native_payload_pairs": pairs,
            "limits": [
                "Compatibility DTO proof reference names are legacy syntax only; private postconditions and owned loopback ledger are retained in proof.json, not public registry proof.",
                "Only initial paired/control annotations are emitted as compatibility rows; all20 actual native requests including second visible-link proof steps remain in their full original episode journals.",
                "Accepted demonstrated teacher actions are not an exhaustive proof that every alternative operation is invalid.",
                "No heldout website traces, goals, target labels, selectors or judges were loaded by this prototype.",
            ],
        }
        write(output / "receipt.json", receipt)
        return receipt
    except BaseException as exc:
        primary = exc
        write(
            output / "failure.json",
            {"type": type(exc).__name__, "message": str(exc), "positive_fit_rows_published": False},
        )
        raise
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2)
            if thread.is_alive() and primary is None:
                raise RuntimeError("Owned loopback server did not exit")
        sys.dont_write_bytecode = before_bytecode
        sys.meta_path.remove(reject)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--chromium-executable", type=Path, default=CHROMIUM)
    args = parser.parse_args(argv)
    result = run_prototype(
        args.output,
        expected_source_sha256=args.source_sha256,
        upstream=args.upstream,
        chromium_executable=args.chromium_executable,
    )
    print(
        json.dumps(
            {"status": result["status"], "episodes": result["episode_count"], "model_calls": 0}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
