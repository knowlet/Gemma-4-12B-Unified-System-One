"""Owned semantic proof admission only; never public admission or fitting.

New project sources stay outside src/s1 so original source84 remains exact.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import jev_public_action_artifact as common

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "owned-semantic-menu-collection-admission-v1"
ORIGINAL_SOURCE_HANDOFF_SHA = "c4601d28519da5563586a84f5d17fc08620a64ce3e80baec529b43f27f5f20d8"
PRODUCER = "scripts/jev_generic_menu_semantics_prototype.py"
PARENT = "scripts/jev_generic_menu_prototype.py"
EXTRAS = {
    PRODUCER,
    PARENT,
    "scripts/jev_observation_profile_v5.py",
    "scripts/jev_browser_profile_v5_benchmark.py",
    "scripts/jev_owned_generic_collection.py",
}
UPSTREAM = {
    "artifacts/jev-runtime/upstream/jev_ultrafast/model.py": "a85ada8458d23642c5c8a4095acf0565e6268f81dcda26f6308942b653e9c9f3",
    "artifacts/jev-runtime/upstream/jev_ultrafast/questions.py": "66a7054b85040b949e1757311257ea0c5a843c635f9a40c99f6d15ad660fb422",
}
FIXED = {
    PRODUCER: "7dedcce509fd8e2acb0c2c0a9dc7d0b712a929f9bc90ec11ed6fb9b630e44495",
    PARENT: "97230a6b741d82a54236191edd8236083ab46fa1d7612dd0934b94abdcde34ee",
    "scripts/jev_observation_profile_v5.py": "7847b2f9eecb3f9d492ccfdf8415c52632b99ac6ead6df560c49c3caf004e141",
    "scripts/jev_browser_profile_v5_benchmark.py": "c84c722d080b1aba8c4225f6e47e42baf33cb19f15545c25b167c0105c89f5d6",
}
COUNTS = {"train": 9, "development": 5}
DISPOSITIONS = {
    "supervised_primary_initial": 14,
    "proof_only_primary_leaf": 6,
    "proof_only_auxiliary": 192,
    "proof_only_wheel_reference": 24,
}
_PROOF = object()


def require(value, message):
    common.require(value, message)


def _digest(value):
    return common.sha(common.encode(value, canonical=True))


def _module(label):
    path = ROOT / common.relative(label)
    name = "_owned_admission_" + common.sha(path.read_bytes())
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == path, "Canonical pure verifier origin")
    return module


def authenticate_sources(plan, reviewed_source_pins):
    original_blob = plan["original_source_handoff_utf8"].encode()
    require(
        common.sha(original_blob) == ORIGINAL_SOURCE_HANDOFF_SHA,
        "Original source90 authority",
    )
    original = common.parse(original_blob)
    expected = set(original["files"]) | EXTRAS | set(UPSTREAM)
    pins = plan["sources"]
    require(
        type(pins) is dict
        and set(pins) == expected
        and list(pins) == sorted(pins)
        and common.exact(pins, reviewed_source_pins),
        "Exact reviewed named owned source closure",
    )
    for label, pin in pins.items():
        require(
            common.exact(common.pin(ROOT / common.relative(label)), pin),
            "Owned source drift: " + label,
        )
        if label in original["files"]:
            require(common.exact(pin, original["files"][label]), "Original90 source changed")
        if label in FIXED | UPSTREAM:
            require(
                pin["sha256"] == (FIXED | UPSTREAM)[label], "Pinned semantic/native source changed"
            )
    # Exercise the original nested inventory contract; no extra s1 module waiver.
    from s1.browser_public_training_validation import _authenticate_complete_source_inventory

    frozen = common.parse(
        common.read(
            ROOT / "artifacts/jev-runtime/frozen-public-observation-36/freeze-manifest.json"
        )
    )["files"]
    _authenticate_complete_source_inventory({k.removeprefix("src/"): v for k, v in frozen.items()})
    return pins


def native_constructor():
    """Compile only the pinned upstream pure request prefix; no CLIENT or HTTP."""
    for label, digest in UPSTREAM.items():
        require(
            common.pin(ROOT / label)["sha256"] == digest, "Pinned upstream pure constructor source"
        )
    model = ast.parse(common.read(ROOT / next(iter(UPSTREAM))))
    functions = {n.name: n for n in model.body if isinstance(n, ast.FunctionDef)}
    choose = functions["choose"]
    stop = next(
        i
        for i, n in enumerate(choose.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "started" for t in n.targets)
    )
    choose.body = choose.body[:stop] + [ast.Return(ast.Name("body", ast.Load()))]
    questions = ast.parse(common.read(ROOT / list(UPSTREAM)[1]))
    constants = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in questions.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    }
    namespace = {
        "os": SimpleNamespace(environ={"TYPESAFE_MODEL": "generated-actions-none"}),
        "NEXT_ACTION": constants["NEXT_ACTION"],
        "TARGET": constants["TARGET"],
    }
    selected = ast.fix_missing_locations(
        ast.Module(body=[functions["action_space"], choose], type_ignores=[])
    )
    exec(compile(selected, "<pinned-native-request-prefix>", "exec"), namespace)
    return namespace["choose"]


@dataclass(frozen=True)
class _Snapshot:
    root: Path
    items: tuple[tuple[str, bytes], ...]
    inventory_bytes: bytes

    def blobs(self):
        return dict(self.items)

    def read(self, label):
        return self.blobs()[common.relative(label)]

    def json(self, label):
        return common.parse(self.read(label))

    def rows(self, label):
        return [common.parse(line) for line in self.read(label).splitlines()]

    def check(self):
        expected = common.parse(self.inventory_bytes)
        require(
            common.exact(
                {k: {"sha256": common.sha(v), "size_bytes": len(v)} for k, v in self.items},
                expected,
            ),
            "Captured owned proof bytes changed",
        )
        common.check_inventory(self.root, expected)


def capture_snapshot(root, index):
    root = common.ordinary(root)
    require(
        type(index) is dict
        and index.get("status") == "complete_readonly_audit"
        and index.get("proof_root") == root.relative_to(ROOT).as_posix(),
        "Externally pinned closed proof root",
    )
    files = index["files"]
    require(
        type(files) is dict
        and len(files) == index["file_count"] == 1079
        and sum(v["size_bytes"] for v in files.values()) == index["total_bytes"] == 56923332,
        "Exact complete owned proof population",
    )
    common.check_inventory(root, files)
    snapshot = _Snapshot(
        root,
        tuple((label, common.read(root / common.relative(label), 64 * 1024**2)) for label in files),
        common.encode(files),
    )
    snapshot.check()
    return snapshot


@dataclass(frozen=True)
class OwnedNativeActionCase:
    """A demonstrated owned route; not a public case or unique-correct action."""

    _annotation: bytes
    _wire: bytes
    _proof: object = field(repr=False, compare=False)
    _seal: str = field(repr=False, compare=False)

    def check(self):
        require(
            self._proof is _PROOF
            and self._seal == common.sha(self._annotation + b"\0" + self._wire),
            "Forged/mutated owned case",
        )

    def annotation(self):
        self.check()
        return common.parse(self._annotation)

    @property
    def id(self):
        return self.annotation()["id"]

    @property
    def split(self):
        return self.annotation()["split"]

    @property
    def group_id(self):
        return self.annotation()["group_id"]

    @property
    def gold(self):
        return self.annotation()["gold"]

    @property
    def request(self):
        from s1.browser_public_training import NativeWireRequest

        self.check()
        return NativeWireRequest.model_validate_json(self._wire)

    @property
    def wire_sha256(self):
        self.check()
        return common.sha(self._wire)


def _case(proof, wire, scene):
    from s1.jev_generated import TARGETS

    selected = proof["record"]["selected"]
    head = TARGETS.get(selected["operation"])
    annotation = {
        "schema_id": "owned-semantic-native-action-case-v1",
        "origin": "owned_local",
        "public_registry_admitted": False,
        "id": wire["case_id"],
        "split": scene["split"],
        "group_id": "owned-semantic-scene:" + scene["id"],
        "goal_family": scene["family"],
        "episode_id": proof["record"]["episode_id"],
        "teacher_choice": selected,
        "gold": {
            "operation": selected["operation"],
            **({head: selected["target"]} if head else {}),
        },
        "proof_refs": wire["proof_refs"],
        "teacher_policy": "useful-demonstrated-route",
        "unique_correct_action": False,
        "global_shortest_path": False,
        "alternative_actions": "not exhaustively classified",
    }
    body = common.encode(wire["native_payload"])
    blob = common.encode(annotation)
    return OwnedNativeActionCase(blob, body, _PROOF, common.sha(blob + b"\0" + body))


@dataclass(frozen=True)
class AdmittedOwnedGenericCollection:
    _snapshot: _Snapshot
    _cases: tuple[OwnedNativeActionCase, ...]
    _plan: bytes
    _provenance: bytes
    _proof: object = field(repr=False, compare=False)
    _seal: str = field(repr=False, compare=False)

    def _identity_check(self):
        require(
            self._proof is _PROOF
            and self._seal
            == common.sha(
                self._plan
                + b"\0"
                + self._provenance
                + b"\0"
                + common.encode([case._seal for case in self._cases])
            ),
            "Forged/mutated owned collection",
        )
        for case in self._cases:
            case.check()

    def check(self):
        self._identity_check()
        require(type(self._snapshot) is _Snapshot, "Exact captured owned snapshot type")
        self._snapshot.check()
        plan = common.parse(self._plan)
        index = common.parse(_checked_ref(plan["proof_index"]))
        require(
            self._snapshot.inventory_bytes == common.encode(index["files"]),
            "Captured owned inventory differs from externally pinned closed index",
        )
        require(
            self._snapshot.root == ROOT / common.relative(index["proof_root"]), "Owned root drift"
        )
        _checked_ref(plan["source_HOLD"])
        _checked_ref(plan["specification"])
        authenticate_sources(plan, plan["sources"])

    @property
    def train_cases(self):
        self._identity_check()
        return tuple(c for c in self._cases if c.split == "train")

    @property
    def development_cases(self):
        self._identity_check()
        return tuple(c for c in self._cases if c.split == "development")

    @property
    def provenance(self):
        self.check()
        return common.parse(self._provenance)

    @property
    def source_pins(self):
        self.check()
        return common.parse(self._plan)["sources"]

    def wire_rows(self, split):
        require(split in COUNTS, "Owned split")
        self.check()
        return [common.parse(c._wire) for c in self._cases if c.split == split]

    def compile_wire_projection(self, split, *, model_alias=None):
        require(model_alias in (None, "generated-actions-none"), "Owned alias cannot be rewritten")
        self.check()
        rows = []
        for case in self._cases:
            if case.split != split:
                continue
            teacher = case.annotation()["teacher_choice"]
            rows.append(
                {
                    "case_id": case.id,
                    "payload": common.parse(case._wire),
                    "teacher_choice": teacher,
                    "valid_next_targets": {
                        teacher["operation"]: []
                        if teacher["target"] is None
                        else [teacher["target"]]
                    },
                }
            )
        require(len(rows) == COUNTS[split], "Owned complete split projection")
        return rows


def _verify_evidence(snapshot, semantic, parent, compiler):
    read, rows = snapshot.json, snapshot.rows
    native_request = native_constructor()
    specs = {scene["id"]: scene for scene in semantic.scenarios()}
    proofs, native_captures, all_inputs = [], 0, 0
    marker = (
        "(() => { const state="
        + snapshot.read("profile/runtime/jev_ultrafast/snapshot.js").decode()
        + "; return state?.marker ?? null; })()"
    )
    for label in sorted(
        k for k in snapshot.blobs() if k.count("/") == 1 and k.endswith("/proof.json")
    ):
        folder = label.split("/")[0]
        proof = snapshot.json(label)
        record = proof["record"]
        require(
            record["episode_id"] == folder and record["reachability_verified"] is True,
            "Actual episode identity",
        )
        require(
            proof["zero_policy_helper_POSTs"] is True and proof["blocked_requests"] == [],
            "No policy/helper/external calls",
        )
        calls = rows(folder + "/native-cdp.jsonl")
        observations = rows(folder + "/observations.jsonl")
        requests = rows(folder + "/native-requests.jsonl")
        transitions = rows(folder + "/transitions.jsonl")
        private = rows(folder + "/private-dom.jsonl")
        ownership = rows(folder + "/ownership.jsonl")
        require(
            [row["sequence"] for row in calls] == list(range(len(calls))),
            "Full ordered native CDP journal",
        )
        require(
            common.encode(transitions) == common.encode(proof["transitions"])
            and common.encode(requests) == common.encode(proof["requests"]),
            "Embedded/journal equality",
        )
        require(
            common.encode(private)
            == common.encode([proof["initial_private"], proof["final_private"]])
            and ownership == [proof["ownership"]],
            "Private/ownership joins",
        )
        inputs = [call for call in calls if call["method"].startswith("Input.")]
        require(
            common.encode(inputs) == common.encode(proof["inputs"]), "Every actual Input accounted"
        )
        owner = proof["ownership"]
        url = urlsplit(owner["url"])
        require(
            url.scheme == "http"
            and url.hostname == "127.0.0.1"
            and url.port
            and not url.username
            and not url.password
            and url.path == "/" + record["scenario_id"]
            and not url.query,
            "Actual owned initial document origin",
        )
        require(
            calls[-1]["method"] == "Target.closeTarget"
            and calls[-1]["params"] == {"targetId": owner["target"]}
            and calls[-1].get("response") == {"success": True},
            "Owned target has actual terminal close",
        )
        require(
            type(proof["positive_supervision"]) is bool
            and proof["positive_supervision"] is (proof["mode"] in {"menu", "scroll", "control"}),
            "Primary versus proof-only annotation disposition",
        )
        require(
            record["html_sha256"]
            == common.sha(semantic.render_fixture(specs[record["scenario_id"]]).encode()),
            "Original authored fixture source bytes",
        )
        require(
            all(call["session"] == owner["session"] for call in inputs),
            "Owned native Input session",
        )
        require(
            any(
                call["method"] == "Target.attachToTarget"
                and call["params"]["targetId"] == owner["target"]
                and call["response"]["sessionId"] == owner["session"]
                for call in calls
            ),
            "Target/session provenance",
        )
        require(
            common.encode(observations) == common.encode(proof["raw_observations"]),
            "Embedded/raw observation journal equality",
        )
        for observation in observations:
            call = calls[observation["protocol_call"] - 1]
            require(
                call["method"] == "Runtime.evaluate"
                and call["session"] == observation["session"] == owner["session"]
                and common.encode(call["response"]) == common.encode(observation["response"]),
                "Raw observation joins actual CDP result",
            )
        for key, index in (
            ("initial_native_page", proof["initial_raw_sequence"]),
            ("final_native_page", len(observations) - 1),
        ):
            page = proof[key]
            raw = observations[index]["response"]["result"]["value"]
            require(
                set(page) == set(raw) | {"fingerprint"}
                and common.encode({k: v for k, v in page.items() if k != "fingerprint"})
                == common.encode(raw),
                "Browser.observe adds only fingerprint",
            )
            content = {k: raw[k] for k in ("url", "text", "actions", "scroll")}
            require(
                page["fingerprint"] == common.sha(json.dumps(content, sort_keys=True).encode()),
                "Exact pinned Browser fingerprint",
            )
        semantic.assert_initial(
            specs[record["scenario_id"]], proof["initial_native_page"], proof["initial_private"]
        )
        require(
            len(observations) == len(transitions) + 1,
            "Every native transition has an exact raw before/after observation",
        )
        history = []

        def fingerprint(state):
            return common.sha(
                json.dumps(
                    {k: state[k] for k in ("url", "text", "actions", "scroll")}, sort_keys=True
                ).encode()
            )

        for j, transition in enumerate(transitions):
            before = observations[j]["response"]["result"]["value"]
            after = observations[j + 1]["response"]["result"]["value"]
            history.append(
                {
                    "action": transition["action"]["label"],
                    "kind": transition["action"]["kind"],
                    "text": transition["text"],
                    "page_changed": fingerprint(before) != fingerprint(after),
                }
            )
        require(
            common.exact(record["history"], history[: proof["initial_raw_sequence"]]),
            "Actual prior native input history",
        )
        scene = specs[record["scenario_id"]]
        goal = scene["goals"][
            proof["mode"]
            if proof["mode"] in scene["goals"]
            else "menu"
            if scene["target_family"] != "control"
            else "control"
        ]
        for index, request in enumerate(requests):
            payload = request["native_payload"]
            at = proof["initial_raw_sequence"] + index
            observed = observations[at]["response"]["result"]["value"]
            require(
                common.encode(native_request(observed, goal, history[:at]))
                == common.encode(payload),
                "Complete original native request differs from actual observation/history/goal",
            )
            compiler.check(payload, request["compiled"])
            require(
                request["sequence"] == request["compiled"]["sequence"] == index
                and common.sha(common.encode(payload))
                == request["native_wire_sha256"]
                == request["compiled"]["wire_sha256"],
                "Every native request/compiler join",
            )
            require(
                request["compiled"]["no_alias_or_payload_rewrite"] is True, "Original alias/payload"
            )
            require(
                request["compiled"]["labels"]
                == {
                    name: list(question["criteria"])
                    for name, question in payload["questions"].items()
                },
                "Every original head/candidate order",
            )
        require(
            common.encode(requests[0]["native_payload"]) == common.encode(record["payload"]),
            "Original initial request",
        )
        for transition in transitions:
            require(
                transition["fresh"] is True
                and transition["executor_result"] == {"executed": transition["action"]["id"]},
                "Fresh canonical action",
            )
            require(
                all(
                    common.encode(row) == common.encode(calls[row["sequence"]])
                    for row in transition["inputs"]
                ),
                "Input bytes join raw native CDP",
            )
            action = transition["action"]
            actual = [(row["method"], row["params"].get("type")) for row in transition["inputs"]]
            expected = (
                [("Input.dispatchMouseEvent", "mouseWheel")]
                if action["kind"] == "scroll"
                else [
                    ("Input.dispatchMouseEvent", "mousePressed"),
                    ("Input.dispatchMouseEvent", "mouseReleased"),
                ]
            )
            if action["kind"] == "fill":
                expected += [
                    ("Input.dispatchKeyEvent", "keyDown"),
                    ("Input.dispatchKeyEvent", "keyUp"),
                    ("Input.insertText", None),
                ]
            require(actual == expected, "Canonical Input sequence only")
            first = transition["inputs"][0]["sequence"]
            before_observation = max(
                (o for o in observations if o["protocol_call"] - 1 < first),
                key=lambda o: o["protocol_call"],
            )
            observed = before_observation["response"]["result"]["value"]
            expression = (
                (
                    "(() => { const c=window.__jevFast; "
                    + f"return c ? [c.pageKey(),c.guard(c.nodes.get({action['node']}))] : null; }})()"
                )
                if action["kind"] in {"click", "select"}
                else marker
            )
            desired = (
                [observed["page_key"], observed["guards"].get(str(action["node"]))]
                if action["kind"] in {"click", "select"}
                else observed["marker"]
            )
            guards = [
                c
                for c in calls[before_observation["protocol_call"] : first]
                if c["method"] == "Runtime.evaluate" and c["params"].get("expression") == expression
            ]
            require(
                len(guards) == 2
                and all(common.exact(c["response"]["result"]["value"], desired) for c in guards),
                "Both actual canonical fresh guards must match observed identity",
            )
            if action["kind"] != "scroll":
                hit = calls[first - 1]
                target = hit["response"]["result"]["value"]
                require(
                    hit["method"] == "Runtime.evaluate"
                    and type(target) is dict
                    and set(target) == {"x", "y"}
                    and all(
                        row["params"][k] == target[k]
                        for row in transition["inputs"][:2]
                        for k in ("x", "y")
                    ),
                    "Actual native hit-test coordinate binds pressed/released target",
                )
            require(
                transition["after"]["width"] == 1120 and transition["after"]["height"] == 780,
                "Fixed actual viewport",
            )
        require(
            all(event["trusted"] is True for event in proof["final_private"]["events"]),
            "Trusted actual DOM events",
        )
        ledger = read(folder + "/local-request-ledger.json")
        require(
            ledger["blocked"] == []
            and all(
                row["method"] in {"GET", "HEAD"}
                and row["status"] == 200
                and urlsplit(row["url"]).netloc == urlsplit(owner["url"]).netloc
                for row in ledger["responses"]
            ),
            "Owned loopback GET/HEAD only",
        )
        spec = specs[record["scenario_id"]]
        if proof["mode"] == "menu":
            semantic.assert_destination(
                spec, spec["target_family"], proof["final_private"], owner["url"]
            )
        elif proof["mode"] in {"scroll", "control"}:
            for key in ("near-heading", "near-paragraph"):
                require(
                    semantic.complete(proof["final_private"]["elements"][key]),
                    "Current-document near passage actually visible",
                )
        if proof["mode"] == "auxiliary":
            if proof["probe"] == "leaf":
                semantic.assert_destination(
                    spec, proof["stage"].removesuffix("_open"), proof["final_private"], owner["url"]
                )
            elif proof["probe"] in {"local", "global"}:
                require(
                    semantic.complete(proof["final_private"]["elements"]["leaf-" + proof["probe"]]),
                    "Auxiliary selected trigger reveals actual leaf",
                )
            elif proof["probe"] == "fill":
                require(
                    proof["final_private"]["note"] == "owned auxiliary focus proof",
                    "Actual auxiliary native edit value",
                )
            else:
                require(
                    proof["probe"] == "focus" and proof["final_private"]["focus"] == "note",
                    "Actual auxiliary focus",
                )
        if proof["mode"] == "wheel_reference":
            require(
                semantic.complete(proof["final_private"]["elements"]["far-heading"])
                and semantic.complete(proof["final_private"]["elements"]["far-paragraph"]),
                "Actual wheel-only same passage",
            )
            require(
                proof["final_private"]["title"] == spec["title"]
                and proof["final_private"]["document_scope"] == "record"
                and proof["final_private"]["elements"]["far-paragraph"]["text"]
                == spec["local_detail"],
                "Reference wheel reaches exact current-document passage",
            )
        if proof["mode"] != "wheel_reference":
            state = (
                observations[proof["stage_raw_sequence"]]["response"]["result"]["value"]
                if proof["mode"] == "auxiliary"
                else proof["initial_native_page"]
            )
            _elements, targets, controls = native_request.__globals__["action_space"](
                state["actions"]
            )
            chosen = record["selected"]
            action = (
                targets[chosen["operation"]][chosen["target"]]
                if chosen["operation"] in targets
                else controls[chosen["operation"]]
            )
            actual = (
                transitions[-1]
                if proof["mode"] == "auxiliary"
                else next(t for t in transitions if t["phase"] == "initial_selected_input")
            )
            require(
                common.encode(action) == common.encode(actual["action"]),
                "Teacher target binds actual offered native Input",
            )
        proofs.append(proof)
        native_captures += len(requests)
        all_inputs += len(inputs)
    require(len(proofs) == 129 and native_captures == 236, "Exact complete evidence population")
    require(
        Counter(p["mode"] for p in proofs)
        == {"auxiliary": 112, "menu": 6, "scroll": 6, "control": 2, "wheel_reference": 3},
        "Every declared episode disposition",
    )
    dispositions = {
        "supervised_primary_initial": sum(bool(p["positive_supervision"]) for p in proofs),
        "proof_only_primary_leaf": sum(
            len(p["requests"]) - 1 for p in proofs if p["positive_supervision"]
        ),
        "proof_only_auxiliary": sum(len(p["requests"]) for p in proofs if p["mode"] == "auxiliary"),
        "proof_only_wheel_reference": sum(
            len(p["requests"]) for p in proofs if p["mode"] == "wheel_reference"
        ),
    }
    require(common.exact(dispositions, DISPOSITIONS), "All236 capture dispositions rederived")
    auxiliary = [proof for proof in proofs if proof["mode"] == "auxiliary"]
    task_order = {
        (task["stage"], task["probe"]): index
        for index, task in enumerate(semantic.auxiliary_tasks())
    }
    auxiliary.sort(key=lambda p: (p["record"]["scenario_id"], task_order[(p["stage"], p["probe"])]))
    require(
        semantic.coverage_certificate(auxiliary) == read("coverage-certificate.json"),
        "Full112 frontier certificate in declared task order",
    )
    primary = [proof for proof in proofs if proof["positive_supervision"]]
    require(
        len(primary) == 14
        and len(auxiliary) == 112
        and len(proofs) - len(primary) - len(auxiliary) == 3,
        "14 primary112 auxiliary3 wheel references",
    )
    pairs = []
    for scene in semantic.scenarios():
        group = [p for p in primary if p["record"]["scenario_id"] == scene["id"]]
        if len(group) == 2:
            menu = next(p for p in group if p["mode"] == "menu")
            scroll = next(p for p in group if p["mode"] == "scroll")
            pair = parent.validate_pair(menu["record"], scroll["record"])
            require(
                menu["initial_private"]["html"] == scroll["initial_private"]["html"],
                "Actual pair initial DOM equal",
            )
            pair.update(
                actual_initial_DOM_bytes_equal=True,
                actual_initial_DOM_sha256=common.sha(menu["initial_private"]["html"].encode()),
            )
            pairs.append(pair)
    require(
        pairs == read("pair-proof.json") and len(pairs) == 6, "Six full native goal-only contrasts"
    )
    wires, cases = rows("compatibility-wire-requests.jsonl"), rows("compatibility-cases.jsonl")
    require(len(wires) == len(cases) == 14, "Fourteen initial requests only")
    for wire, case in zip(wires, cases, strict=True):
        proof = next(p for p in primary if p["record"]["episode_id"] == wire["episode_id"])
        require(
            common.encode(wire["native_payload"])
            == common.encode(case["request"])
            == common.encode(proof["record"]["payload"]),
            "Compatibility body is full original initial native request",
        )
        for reference in wire["proof_refs"].values():
            blob = snapshot.read(reference["artifact_path"])
            require(common.sha(blob) == reference["sha256"], "Every DTO proof ref SHA")
    result = []
    for wire, compatibility in zip(wires, cases, strict=True):
        proof = next(p for p in primary if p["record"]["episode_id"] == wire["episode_id"])
        scene = specs[proof["record"]["scenario_id"]]
        owned = _case(proof, wire, scene)
        require(
            common.exact(wire["teacher_choice"], owned.annotation()["teacher_choice"])
            and common.exact(compatibility["gold"], owned.gold)
            and common.exact(compatibility["proof_refs"], wire["proof_refs"]),
            "Compatibility annotation differs from proved native teacher",
        )
        require(
            wire["native_policy_request_sequence"] == wire["compiled_input_sequence"] == 0
            and wire["case_id"] == proof["record"]["episode_id"] + "-initial",
            "Only original initial supervision",
        )
        require(len(owned.request.candidate_labels) == 3, "All initial heads retained")
        require(
            set(wire["proof_refs"])
            == {
                "raw_observation",
                "private_precondition",
                "guarded_transition",
                "private_postcondition",
                "ownership",
                "public_ledger",
            },
            "Complete actual evidence references",
        )
        for name, ref in wire["proof_refs"].items():
            require(
                type(ref) is dict
                and set(ref) == {"artifact_path", "sha256", "sequence"}
                and type(ref["sequence"]) is int
                and ref["sequence"] >= 0,
                "Typed owned evidence reference",
            )
            expected_sequences = {
                "raw_observation": proof["initial_raw_sequence"],
                "private_precondition": 0,
                "guarded_transition": next(
                    i
                    for i, t in enumerate(proof["transitions"])
                    if t["phase"] == "initial_selected_input"
                ),
                "private_postcondition": 1,
                "ownership": 0,
                "public_ledger": 0,
            }
            require(
                ref["sequence"] == expected_sequences[name]
                and ref["artifact_path"]
                == wire["episode_id"]
                + "/"
                + {
                    "raw_observation": "observations.jsonl",
                    "private_precondition": "private-dom.jsonl",
                    "guarded_transition": "transitions.jsonl",
                    "private_postcondition": "private-dom.jsonl",
                    "ownership": "ownership.jsonl",
                    "public_ledger": "local-request-ledger.json",
                }[name],
                "Actual owned reference sequence/episode",
            )
        result.append(owned)
    require(
        {split: sum(c.split == split for c in result) for split in COUNTS} == COUNTS,
        "Exact local9 train5 development",
    )
    require(
        semantic.lexical_and_split_audit() == read("lexical-and-split-audit.json"),
        "Actual frozen family split",
    )
    return tuple(result), {
        "episodes": len(proofs),
        "captures": native_captures,
        "native_inputs": all_inputs,
        "goal_only_pairs": len(pairs),
        "dispositions": dispositions,
    }


PLAN_KEYS = {
    "schema_id",
    "origin",
    "public_registry_admitted",
    "original_source_handoff_utf8",
    "sources",
    "proof_index",
    "source_HOLD",
    "specification",
    "counts",
    "capture_dispositions",
    "compiler_metadata",
    "profile",
    "fit_status",
}


def _ref(path):
    path = common.ordinary(path)
    require(path.is_relative_to(ROOT), "Named owned input under project ROOT")
    return {"path": path.relative_to(ROOT).as_posix(), **common.pin(path)}


def _checked_ref(value):
    require(
        type(value) is dict and set(value) == {"path", "sha256", "size_bytes"},
        "Strict owned input reference",
    )
    path = ROOT / common.relative(value["path"])
    blob = common.read(path, 64 * 1024**2)
    require(
        common.exact(
            {"sha256": common.sha(blob), "size_bytes": len(blob)},
            {k: value[k] for k in ("sha256", "size_bytes")},
        ),
        "Owned reference drift",
    )
    return blob


def prepare_owned_admission_plan(closed_index_path):
    """Create a reviewable draft only; does not admit or read the proof payload."""
    index = common.parse(common.read(closed_index_path))
    original = common.read(
        ROOT
        / "artifacts/jev-runtime/public-native-action-actual07-fit-20261016-01/source-handoff-03.json"
    )
    require(common.sha(original) == ORIGINAL_SOURCE_HANDOFF_SHA, "Original90 handoff")
    proof = ROOT / common.relative(index["proof_root"])
    receipt = common.parse(common.read(proof / "receipt.json"))
    plan = {
        "schema_id": SCHEMA,
        "origin": "owned_local",
        "public_registry_admitted": False,
        "original_source_handoff_utf8": original.decode(),
        "sources": {
            k: common.pin(ROOT / k)
            for k in sorted(set(common.parse(original)["files"]) | EXTRAS | set(UPSTREAM))
        },
        "proof_index": _ref(Path(closed_index_path)),
        "source_HOLD": _ref(proof.parent / "source-HOLD-01/HOLD.json"),
        "specification": _ref(proof.parent / "specification-02.json"),
        "counts": {
            "scenes": 8,
            "primary": 14,
            "train": 9,
            "development": 5,
            "auxiliary": 112,
            "wheel_references": 3,
            "episodes": 129,
            "native_captures": 236,
        },
        "capture_dispositions": dict(DISPOSITIONS),
        "compiler_metadata": receipt["compiler_metadata"],
        "profile": {
            "id": receipt["profile"],
            "source_sha256": receipt["profile_source_sha256"],
            "manifest_sha256": receipt["profile_manifest_sha256"],
        },
        "fit_status": "not_fit_eligible_without_separate_mixed_producer_and_qualification",
    }
    return common.encode(plan) + b"\n"


class OwnedCPUAdmissionGate:
    """Single durable same-boot gate with the unchanged public CPU reserve floor."""

    def __init__(self, path, expected_boot_id, *, observe=None):
        from jev_browser_fit_public_actions import memory_observation

        self.path = common.ordinary(path)
        self.boot, self.observe = expected_boot_id, observe or memory_observation
        require(
            type(expected_boot_id) is str and bool(expected_boot_id) and not self.path.exists(),
            "Fresh owned CPU journal",
        )

    def check(self, phase, scope):
        import os
        import time

        require(
            phase == "before-owned-collection-admission"
            and scope == "cpu"
            and not self.path.exists(),
            "Single explicit owned CPU admission phase",
        )
        row = {
            "phase": phase,
            "scope": scope,
            "monotonic": time.monotonic(),
            "expected_boot_id": self.boot,
            "admitted": False,
            "swap_or_resident_credit_bytes": 0,
        }
        primary = None
        try:
            observed = self.observe()
            require(
                all(
                    type(observed[k]) is int and observed[k] > 0
                    for k in ("MemTotal", "MemAvailable")
                ),
                "Positive integer CPU memory observations",
            )
            reserve = max(24 * common.GIB, (observed["MemTotal"] + 4) // 5)
            required = reserve + 4 * common.GIB
            row.update(observation=observed, reserve_bytes=reserve, required_bytes=required)
            require(
                observed["boot_id"] == self.boot
                and 0 < required <= observed["MemAvailable"] <= observed["MemTotal"],
                "Fresh same-boot owned CPU admission refused",
            )
            row["admitted"] = True
        except BaseException as exc:
            primary = exc
        row["error"] = (
            None if primary is None else {"type": type(primary).__name__, "message": str(primary)}
        )
        try:
            common.create(self.path, common.encode(row) + b"\n")
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except BaseException as secondary:
            if primary is not None:
                primary.add_note("Owned CPU refusal persistence failed: " + repr(secondary))
                raise primary from secondary
            raise
        if primary is not None:
            raise primary
        return row


def validate_owned_generic_collection(
    proof_root,
    *,
    plan_bytes,
    expected_plan_sha256,
    expected_closed_index_sha256,
    expected_producer_sha256,
    reviewed_source_pins,
    tokenizer_cache,
    gate,
):
    """Authenticate complete owned evidence; never converts to a public DTO."""
    root = common.ordinary(proof_root)
    if isinstance(gate, OwnedCPUAdmissionGate):
        require(
            not gate.path.is_relative_to(root), "Admission journal cannot mutate closed proof tree"
        )
    gate.check("before-owned-collection-admission", "cpu")
    require(
        type(plan_bytes) is bytes and common.sha(plan_bytes) == expected_plan_sha256,
        "Externally reviewed owned admission plan SHA",
    )
    plan = common.parse(plan_bytes)
    require(
        type(plan) is dict
        and set(plan) == PLAN_KEYS
        and plan["schema_id"] == SCHEMA
        and plan["origin"] == "owned_local"
        and plan["public_registry_admitted"] is False,
        "Strict separate owned admission plan",
    )
    require(
        expected_producer_sha256 == FIXED[PRODUCER]
        and plan["sources"][PRODUCER]["sha256"] == expected_producer_sha256,
        "Externally pinned semantic producer",
    )
    require(
        common.exact(
            plan["counts"],
            {
                "scenes": 8,
                "primary": 14,
                "train": 9,
                "development": 5,
                "auxiliary": 112,
                "wheel_references": 3,
                "episodes": 129,
                "native_captures": 236,
            },
        )
        and common.exact(plan["capture_dispositions"], DISPOSITIONS),
        "Complete owned grain/disposition contract",
    )
    require(
        plan["fit_status"] == "not_fit_eligible_without_separate_mixed_producer_and_qualification",
        "Owned admission is not deployment or fit eligibility",
    )
    authenticate_sources(plan, reviewed_source_pins)
    index_blob = _checked_ref(plan["proof_index"])
    require(
        common.sha(index_blob) == expected_closed_index_sha256, "External closed proof index SHA"
    )
    source_hold = common.parse(_checked_ref(plan["source_HOLD"]))
    require(
        source_hold["source"]["sha256"] == FIXED[PRODUCER], "Original reviewed semantic source HOLD"
    )
    specification = common.parse(_checked_ref(plan["specification"]))
    semantic, parent = _module(PRODUCER), _module(PARENT)
    require(
        common.exact(semantic.specification(), specification),
        "Reviewed authored semantic specification",
    )
    snapshot = capture_snapshot(proof_root, common.parse(index_blob))
    receipt = snapshot.json("receipt.json")
    require(
        receipt["source_sha256"] == FIXED[PRODUCER]
        and receipt["parent_sha256"] == FIXED[PARENT]
        and receipt["status"] == "complete_local_proof_not_public_admission"
        and receipt["model_or_fit_calls"] == receipt["external_network_calls"] == 0
        and receipt["zero_policy_helper_POSTs"] is True
        and receipt["public_admitted"] is False
        and receipt["fit_bundle_created"] is False,
        "Completed genuinely local no-model source collection",
    )
    require(
        common.exact(plan["compiler_metadata"], receipt["compiler_metadata"])
        and common.exact(
            plan["profile"],
            {
                "id": "composed-visible-hover-entries-v5",
                "source_sha256": FIXED["scripts/jev_observation_profile_v5.py"],
                "manifest_sha256": common.sha(snapshot.read("profile/profile-manifest.json")),
            },
        )
        and plan["profile"]["manifest_sha256"] == receipt["profile_manifest_sha256"],
        "Pinned full native metadata/profile",
    )
    from s1.browser_public_training_validation import _CachedNativeCompiler

    compiler = _CachedNativeCompiler(tokenizer_cache, plan["compiler_metadata"])
    cases, counts = _verify_evidence(snapshot, semantic, parent, compiler)
    provenance = common.encode(
        {
            "schema_id": SCHEMA,
            "origin": "owned_local",
            "public_registry_admitted": False,
            "admission_plan_sha256": expected_plan_sha256,
            "closed_index_sha256": expected_closed_index_sha256,
            "source_identity_sha256": common.identity(plan["sources"]),
            "counts": counts,
            "splits": {split: [c.id for c in cases if c.split == split] for split in COUNTS},
            "teacher_policy": "useful-demonstrated-route; no unique action or global shortest-path assertion",
            "no_training": True,
        }
    )
    seal = common.sha(
        plan_bytes + b"\0" + provenance + b"\0" + common.encode([case._seal for case in cases])
    )
    admitted = AdmittedOwnedGenericCollection(snapshot, cases, plan_bytes, provenance, _PROOF, seal)
    admitted.check()
    return admitted
