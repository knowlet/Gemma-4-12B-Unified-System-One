"""Separate admission of closed authored form routes; no tokenizer at import.

This type is neither the public registry nor OwnedNativeActionCase. Admission
checks saved evidence only. Actual processor/CFG reference preparation, fitting
and model qualification require separate reviewed execution and receipts.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import jev_public_action_artifact as common

ROOT = Path(__file__).resolve().parents[1]
BASE = "artifacts/jev-runtime/generic-form-progress-native-proof-20261019-01"
CANDIDATES = "artifacts/jev-runtime/generic-form-progress-candidates-20261019-01"
MATRIX = "artifacts/jev-runtime/public-affordance-matrix-harness-20261019-01"
SCHEMA = "owned-form-progress-native-collection-admission-v1"
CASE_SCHEMA = "owned-form-progress-native-action-case-v1"
SOURCE_SCHEMA = "owned-form-progress-source-union-v1"
REFERENCE_SCHEMA = "owned-form-progress-native-CPU-reference-v1"
COUNTS = {"train": 13, "development": 10}
ORIGINAL = (
    "artifacts/jev-runtime/public-native-action-actual07-fit-20261016-01/source-handoff-03.json"
)
ORIGINAL_SHA = "c4601d28519da5563586a84f5d17fc08620a64ce3e80baec529b43f27f5f20d8"
EXTRAS = {
    "scripts/jev_owned_form_progress.py",
    "scripts/jev_cfg_action_loss.py",
    *(
        BASE + "/" + x
        for x in ("collector_core.py", "form_common.py", "form_worker.py", "form_controller.py")
    ),
    CANDIDATES + "/candidate_sources.py",
    *(
        MATRIX + "/" + x
        for x in (
            "matrix_common.py",
            "matrix_worker.py",
            "matrix_controller.py",
            "native-request-source/model.py",
            "native-request-source/questions.py",
            "runtime/v5/browser.py",
            "runtime/v5/snapshot.js",
        )
    ),
}
ANCHORS = {
    "source_HOLD": (
        BASE + "/source-HOLD-03/HOLD.json",
        "36783a4231f18e1831985797f0b8f9b9755e656da2a86e8692c7e140844100de",
    ),
    "collection_plan": (
        BASE + "/plan-03.json",
        "871b3a33729277974839aa02a6a05f286c8440c8d1483f152173d67b32695191",
    ),
    "closed_index": (
        BASE + "/root-actual01-closed-index-01.json",
        "00eed5a57f169145cd661f91c65db3ea547b847123ee2d2719f9b3cdc9c098eb",
    ),
    "root_closure": (
        BASE + "/root-actual01-exit-review-01.json",
        "c3bed7e854a796cc5833194e74f7c3f5c659dd472877b4e64d35644c761ed36b",
    ),
    "independent_review": (
        BASE + "/independent-actual01-closed-review-01.json",
        "1e4828c5ca5ca5f67dacea7892c66879c1d9912f0b25475da9826315d1eedc7c",
    ),
}
_PROOF = object()
PLAN_KEYS = {
    "schema_id",
    "source_schema_id",
    "origin",
    "public_registry_admitted",
    "original_source_handoff_utf8",
    "sources",
    "anchors",
    "proof_root",
    "counts",
    "compiled_input_sequence",
    "fit_status",
}


def require(value, message):
    common.require(value, message)


def _ref(label):
    return {"path": label, **common.pin(ROOT / common.relative(label))}


def _checked(ref):
    require(
        type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"}, "Typed form reference"
    )
    blob = common.read(ROOT / common.relative(ref["path"]), 64 * 1024**2)
    require(
        common.exact(
            ref, {"path": ref["path"], "sha256": common.sha(blob), "size_bytes": len(blob)}
        ),
        "Form evidence/source reference drift",
    )
    return blob


def prepare_admission_plan():
    """Small source/metadata draft only; does not read/admit the proof tree."""
    original = common.read(ROOT / ORIGINAL)
    require(common.sha(original) == ORIGINAL_SHA, "Original90 source authority")
    names = set(common.parse(original)["files"]) | EXTRAS
    return {
        "schema_id": SCHEMA,
        "source_schema_id": SOURCE_SCHEMA,
        "origin": "owned_local",
        "public_registry_admitted": False,
        "original_source_handoff_utf8": original.decode(),
        "sources": {k: common.pin(ROOT / k) for k in sorted(names)},
        "anchors": {k: _ref(v[0]) for k, v in ANCHORS.items()},
        "proof_root": BASE + "/actual-01",
        "counts": dict(COUNTS),
        "compiled_input_sequence": None,
        "fit_status": "NOT_COMPILED_NOT_FIT_NOT_QUALIFIED",
    }


def authenticate_sources(plan, reviewed_source_pins):
    original = plan["original_source_handoff_utf8"].encode()
    require(common.sha(original) == ORIGINAL_SHA, "Original90 source authority")
    original = common.parse(original)["files"]
    sources = plan["sources"]
    require(
        type(sources) is dict
        and list(sources) == sorted(sources)
        and set(sources) == set(original) | EXTRAS
        and common.exact(sources, reviewed_source_pins),
        "Exact externally reviewed form source union; no legacy-source waiver",
    )
    for label, pin in sources.items():
        require(
            common.exact(common.pin(ROOT / common.relative(label)), pin),
            "Form source drift: " + label,
        )
        if label in original:
            require(common.exact(pin, original[label]), "Original90 source changed")
    authenticate_original_package_population(original)
    return sources


def authenticate_original_package_population(original):
    """Retain the original closed src/s1 population, without a new module waiver."""
    expected = {k for k in original if k.startswith("src/s1/") and k.endswith(".py")}
    actual = {
        p.relative_to(ROOT).as_posix() for p in (ROOT / "src/s1").rglob("*.py") if p.is_file()
    }
    require(
        expected == actual and not any((ROOT / k).is_symlink() for k in actual),
        "Original installed s1 source population changed",
    )


def _core():
    path = ROOT / BASE / "collector_core.py"
    name = "_admitted_form_core_" + common.sha(path.read_bytes())
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def native_constructor():
    """Only original pure action_space/choose prefix; never CLIENT or POST."""
    model_blob = common.read(ROOT / MATRIX / "native-request-source/model.py")
    questions_blob = common.read(ROOT / MATRIX / "native-request-source/questions.py")
    require(
        common.sha(model_blob) == "a85ada8458d23642c5c8a4095acf0565e6268f81dcda26f6308942b653e9c9f3"
        and common.sha(questions_blob)
        == "66a7054b85040b949e1757311257ea0c5a843c635f9a40c99f6d15ad660fb422",
        "Exact original pure request source",
    )
    model = ast.parse(model_blob)
    functions = {n.name: n for n in model.body if isinstance(n, ast.FunctionDef)}
    choose = functions["choose"]
    stop = next(
        i
        for i, n in enumerate(choose.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "started" for t in n.targets)
    )
    choose.body = choose.body[:stop] + [ast.Return(ast.Name("body", ast.Load()))]
    questions = ast.parse(questions_blob)
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
    exec(compile(selected, "<pinned-form-native-request-prefix>", "exec"), namespace)
    return namespace["choose"], namespace["action_space"]


def _targets():
    """Read the original TARGETS literal, without importing model/framework code."""
    tree = ast.parse(common.read(ROOT / "src/s1/jev_generated.py"))
    return next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "TARGETS" for t in n.targets)
    )


@dataclass(frozen=True)
class FormNativeActionCase:
    _annotation: bytes
    _wire: bytes
    _proof: object = field(repr=False, compare=False)
    _seal: str = field(repr=False, compare=False)

    def check(self):
        require(
            self._proof is _PROOF
            and self._seal == common.sha(self._annotation + b"\0" + self._wire),
            "Forged/mutated form case",
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

    @property
    def wire_bytes(self):
        self.check()
        return self._wire


def _case(decision, wire, spec):
    selected = decision["selected"]
    head = _targets()[selected["operation"]]
    raw = wire["native_wire_utf8"].encode()
    require(
        common.sha(raw) == wire["native_wire_sha256"]
        and len(raw) == wire["native_wire_size_bytes"],
        "Original ordered form body authority",
    )
    annotation = common.encode(
        {
            "schema_id": CASE_SCHEMA,
            "id": decision["id"],
            "split": spec["split"],
            "group_id": "owned-form-family:" + spec["family"],
            "family": spec["family"],
            "episode_id": spec["id"],
            "origin": "owned_local",
            "public_registry_admitted": False,
            "gold": {"operation": selected["operation"], head: selected["target"]},
            "teacher_choice": selected,
            "proof_refs": wire["proof_refs"],
            "compiled_input_sequence": None,
            "compilation_status": "NOT_PERFORMED",
            "teacher_policy": "useful native demonstrated route; not unique correct/global shortest",
            "field_text_scope": "TYPE_TEXT value belongs to the separate transition proof; no field value in action gold",
        }
    )
    return FormNativeActionCase(annotation, raw, _PROOF, common.sha(annotation + b"\0" + raw))


@dataclass(frozen=True)
class _Snapshot:
    items: tuple[tuple[str, bytes], ...]

    def read(self, label):
        return dict(self.items)[common.relative(label)]

    def json(self, label):
        return common.parse(self.read(label))

    def rows(self, label):
        return [common.parse(line) for line in self.read(label).splitlines()]

    def digest(self):
        return common.sha(
            common.encode(
                {k: {"sha256": common.sha(v), "size_bytes": len(v)} for k, v in self.items}
            )
        )


def capture_snapshot(index):
    root = common.ordinary(ROOT / BASE / "actual-01")
    refs = index["files"]
    require(
        index["schema"] == "root-closed-owned-form-native-proof-index-v1"
        and index["ordinary_file_count"] == len(refs) == 236
        and index["total_bytes"] == sum(r["size_bytes"] for r in refs) == 11438498,
        "Complete fixed closed proof population",
    )
    paths = [r["path"] for r in refs]
    # The frozen root index sorts Path objects, whose component ordering differs
    # from whole-string ordering for e.g. episodes/00/trace and episodes/00.json.
    # Keep its authenticated order; compare the full population in one domain.
    require(len(set(paths)) == len(paths), "Unique complete index")
    entries = list(root.rglob("*"))
    require(not any(p.is_symlink() for p in entries), "No symlink in closed proof population")
    actual = sorted(p.relative_to(ROOT).as_posix() for p in entries if p.is_file())
    require(actual == sorted(paths), "Missing/extra closed proof leaf; no selection")
    items = []
    for ref in refs:
        p = ROOT / common.relative(ref["path"])
        require(p.is_relative_to(root), "Confined form proof")
        items.append((p.relative_to(root).as_posix(), _checked(ref)))
    return _Snapshot(tuple(items))


def _verify_episode(snapshot, ordinal, spec, counter, core, constructor, action_space):
    prefix = f"episodes/{ordinal:02d}"
    summary = snapshot.json(prefix + ".json")
    row = snapshot.json(prefix + "/proof.json")
    proof = row["proof"]
    require(
        summary["ordinal"] == ordinal
        and summary["case_id"] == spec["id"]
        and summary["counterexample"] is counter
        and summary["proof"]
        == {
            "path": prefix + "/proof.json",
            "sha256": common.sha(snapshot.read(prefix + "/proof.json")),
            "size_bytes": len(snapshot.read(prefix + "/proof.json")),
        },
        "Complete episode summary/raw proof identity",
    )
    require(
        row["case_id"] == spec["id"]
        and row["counterexample"] is counter
        and not row["cleanup_errors"]
        and not row["blocked"]
        and not row["page_errors"]
        and row["admitted"] is False,
        "Complete exact episode, no dropped failure",
    )
    expected_status = (
        "COMPLETE_WRONG_DOCUMENT_COUNTEREXAMPLE" if counter else "COMPLETE_NATIVE_ROUTE_UNADMITTED"
    )
    require(
        row["status"] == summary["status"] == expected_status
        and proof["schema_id"] == core.PROOF_KIND
        and proof["case_id"] == spec["id"]
        and proof["goal"] == spec["goal"]
        and proof["admitted"] is proof["compiled"] is False,
        "Typed original native episode",
    )
    require(
        all(n["method"] in {"GET", "HEAD"} and n["status"] == 200 for n in row["network"])
        and not any(c["type"] == "error" for c in row["console"]),
        "Owned non-error route evidence",
    )
    arrays = {
        "native-cdp.jsonl": row["calls"],
        "observations.jsonl": row["observations"],
        "ownership.jsonl": [row["ownership"]],
        "local-request-ledger.jsonl": [{"responses": row["network"], "blocked": row["blocked"]}],
        "progress-events.jsonl": row["events"],
        "native-requests.jsonl": proof["requests"],
        "transitions.jsonl": proof["transitions"],
        "private-dom.jsonl": proof["private_states"],
    }
    for label, expected in arrays.items():
        require(
            common.exact(snapshot.rows(prefix + "/" + label), expected),
            "Full raw journal/proof join: " + label,
        )
        ref = summary["journals"][label]
        require(
            ref["path"] == prefix + "/" + label
            and ref["sha256"] == common.sha(snapshot.read(ref["path"])),
            "Saved journal reference",
        )
    steps = (
        [{"document": "index.html", "operation": "CLICK", "private_dom_id": "detail-tab"}]
        if counter
        else spec["teacher_steps"]
    )
    n = len(steps)
    require(
        len(proof["requests"])
        == len(proof["native_pages"])
        == len(proof["private_states"])
        == n + 1
        and len(proof["transitions"]) == len(proof["history"]) == n
        and len(proof["decisions"]) == (0 if counter else n),
        "All teacher/final/clone states retained",
    )
    origin = "http://" + urlsplit(proof["private_states"][0]["url"]).netloc
    require(urlsplit(origin).hostname == "127.0.0.1", "Owned loopback proof")
    core.assert_initial(spec, proof["native_pages"][0], proof["private_states"][0], origin)
    require(
        not any("error" in c for c in row["calls"])
        and common.exact(
            row["inputs"], [c for c in row["calls"] if c["method"].startswith("Input.")]
        ),
        "All attempted Input returned and retained in completed proof",
    )
    private_values = [
        c["response"]["result"]["value"]
        for c in row["calls"]
        if c["method"] == "Runtime.evaluate" and c["params"].get("expression") == core.PRIVATE
    ]
    require(
        common.exact(private_values, proof["private_states"]),
        "Raw private DOM evaluation/state join",
    )
    snapshot_expression = common.read(ROOT / MATRIX / "runtime/v5/snapshot.js").decode()
    history = []
    for j, request in enumerate(proof["requests"]):
        page = proof["native_pages"][j]
        raw = request["native_wire_utf8"].encode()
        body = snapshot.read(prefix + f"/native-request-bodies/{j:02d}.json")
        require(
            raw == body == core.encode(constructor(page, spec["goal"], history))
            and common.sha(raw) == request["native_wire_sha256"]
            and common.exact(common.parse(raw), request["native_payload"]),
            "Original43 complete ordered bodies; no prompt/head pruning",
        )
        require(
            request["sequence"] == j
            and request["compiled"] is None
            and request["compilation_status"] == "NOT_PERFORMED"
            and request["phase"]
            == ("final_proof_only" if j == n else "counterexample" if counter else "teacher"),
            "Proof-only final versus teacher request disposition",
        )
        observation = row["observations"][request["raw_observation_sequence"]]
        call = row["calls"][observation["protocol_call"] - 1]
        require(
            observation["session"] == call["session"] == "owned-matrix-fixture"
            and call["method"] == "Runtime.evaluate"
            and call["params"]["expression"] == snapshot_expression
            and common.exact(call["response"], observation["response"]),
            "Accepted raw snapshot is an actual pinned READ_STATE return",
        )
        require(
            common.exact(
                observation["response"]["result"]["value"],
                {k: v for k, v in page.items() if k != "fingerprint"},
            )
            and page["fingerprint"] == core.fingerprint(page),
            "Original accepted raw observation/fingerprint",
        )
        require(
            summary["native_wire_bodies"][j]
            == {
                "path": prefix + f"/native-request-bodies/{j:02d}.json",
                "sha256": common.sha(body),
                "size_bytes": len(body),
            },
            "Original saved ordered body reference",
        )
        if j == n:
            continue
        step = steps[j]
        before, after = proof["private_states"][j : j + 2]
        core.document(before, spec, origin, step["document"])
        resolution = core.resolve_teacher(
            page, before, step, action_space, request["native_payload"]
        )
        transition = proof["transitions"][j]
        require(
            transition["text"] == (step.get("value") if step["operation"] == "TYPE_TEXT" else None)
            and transition["before_fingerprint"] == page["fingerprint"]
            and transition["after_fingerprint"] == proof["native_pages"][j + 1]["fingerprint"],
            "Original text provenance/history fingerprints",
        )
        require(
            common.exact(transition["resolution"], resolution)
            and common.exact(transition["before"], before)
            and common.exact(transition["after"], after)
            and transition["fresh"] is True
            and transition["executor_result"] == {"executed": resolution["action"]["id"]},
            "Fresh offered native target/executor proof",
        )
        start, end = transition["call_range"]
        require(
            type(start) is type(end) is int and 0 <= start < end <= len(row["calls"]),
            "Original bounded native call range",
        )
        calls = row["calls"][start:end]
        require(
            common.exact(
                [c for c in calls if c["method"].startswith("Input.")], transition["inputs"]
            ),
            "Every attempted/returned Input retained",
        )
        core.verify_native_transition(
            step, resolution, before, after, calls, transition["inputs"], "owned-matrix-fixture"
        )
        history.append(
            {
                "action": resolution["action"]["label"],
                "kind": resolution["action"]["kind"],
                "text": transition["text"],
                "page_changed": page["fingerprint"] != proof["native_pages"][j + 1]["fingerprint"],
            }
        )
    require(
        common.exact(history, proof["history"])
        and core.semantic_complete(spec, proof["private_states"][-1], origin) is (not counter)
        and proof["semantic_complete"] is (not counter),
        "Correct full document/visible semantic final; negatives not training states",
    )
    if not counter:
        core.assert_demonstration(spec, proof["private_states"][-1])
    return row, summary


def _validate_split(cases, specs):
    require(Counter(c.split for c in cases) == COUNTS, "All23 decision states13train10development")
    families = {s["family"]: s["split"] for s in specs}
    require(
        len(specs) == 12
        and len({s["id"] for s in specs}) == 12
        and len(families) == 6
        and Counter(s["family"] for s in specs) == {f: 2 for f in families}
        and Counter(families.values()) == {"train": 4, "development": 2}
        and all(s["split"] == families[s["family"]] for s in specs),
        "Whole-family4/2 split; no scene or successful-state selection",
    )
    require(
        len({c.id for c in cases}) == 23
        and all(
            c.annotation()["family"] in families and c.split == families[c.annotation()["family"]]
            for c in cases
        ),
        "Exact unique case family membership",
    )


@dataclass(frozen=True)
class AdmittedFormProgressCollection:
    _snapshot: _Snapshot
    _cases: tuple[FormNativeActionCase, ...]
    _plan: bytes
    _provenance: bytes
    _proof: object = field(repr=False, compare=False)
    _seal: str = field(repr=False, compare=False)

    def check(self):
        require(
            type(self._snapshot) is _Snapshot
            and self._proof is _PROOF
            and self._seal
            == common.sha(
                self._plan
                + b"\0"
                + self._provenance
                + b"\0"
                + common.encode([c._seal for c in self._cases])
                + self._snapshot.digest().encode()
            ),
            "Forged/mutated admitted form collection",
        )
        for case in self._cases:
            require(type(case) is FormNativeActionCase, "Exact new form case type")
            case.check()

    @property
    def train_cases(self):
        self.check()
        return tuple(c for c in self._cases if c.split == "train")

    @property
    def development_cases(self):
        self.check()
        return tuple(c for c in self._cases if c.split == "development")

    @property
    def provenance(self):
        self.check()
        return common.parse(self._provenance)

    @property
    def source_pins(self):
        self.check()
        return common.parse(self._plan)["sources"]

    def compile_wire_projection(self, split):
        require(split in COUNTS, "Exact form split")
        self.check()
        rows = [
            {
                "case_id": c.id,
                "payload": common.parse(c._wire),
                "teacher_choice": c.annotation()["teacher_choice"],
                "valid_next_targets": {
                    c.gold["operation"]: [c.annotation()["teacher_choice"]["target"]]
                },
            }
            for c in self._cases
            if c.split == split
        ]
        require(len(rows) == COUNTS[split], "Complete ordered form compile projection")
        return rows


def validate_form_progress_collection(
    plan_bytes, *, expected_plan_sha256, reviewed_source_pins, gate
):
    """Authenticate every saved state before constructing the separate type.

    The caller supplies a durable fresh CPU gate. This function does not load a
    processor or manufacture a CPU compilation receipt.
    """
    require(
        type(plan_bytes) is bytes and common.sha(plan_bytes) == expected_plan_sha256,
        "Externally selected form admission plan",
    )
    gate.check("before-any-project-framework-or-parent-payload", "cpu")
    plan = common.parse(plan_bytes)
    require(
        type(plan) is dict
        and set(plan) == PLAN_KEYS
        and plan["schema_id"] == SCHEMA
        and plan["source_schema_id"] == SOURCE_SCHEMA
        and plan["origin"] == "owned_local"
        and plan["public_registry_admitted"] is False
        and common.exact(plan["counts"], COUNTS)
        and plan["compiled_input_sequence"] is None
        and plan["fit_status"] == "NOT_COMPILED_NOT_FIT_NOT_QUALIFIED",
        "Strict new form admission kind, never legacy/public coercion",
    )
    authenticate_sources(plan, reviewed_source_pins)
    require(
        plan["proof_root"] == BASE + "/actual-01" and set(plan["anchors"]) == set(ANCHORS),
        "Exact closed form proof namespace",
    )
    anchors = {}
    for key, (label, sha) in ANCHORS.items():
        ref = plan["anchors"][key]
        require(ref["path"] == label and ref["sha256"] == sha, "Externally frozen anchor: " + key)
        anchors[key] = common.parse(_checked(ref))
    closure, independent = anchors["root_closure"], anchors["independent_review"]
    require(
        closure["status"] == "WHOLE_CLOSED_COMPLETE_UNADMITTED_NATIVE_PROOF"
        and closure["exec_exit"] == 0
        and closure["fresh_same_identities_or_owned_groups_alive"] == 0
        and closure["controller_and_original_group_absent"]
        is closure["local_HTTP_listener_absent"]
        is True
        and independent["status"] == "COMPLETE_CLOSED_NATIVE_PROOF_CONSISTENT_STILL_UNADMITTED"
        and independent["audit_blockers"] == [],
        "Root-owned whole closure and independent saved-proof review",
    )
    collection_plan = anchors["collection_plan"]
    require(
        common.exact(anchors["source_HOLD"]["implementation"], collection_plan["implementation"])
        and anchors["source_HOLD"]["plan"]["sha256"]
        == plan["anchors"]["collection_plan"]["sha256"],
        "Reviewed source HOLD/plan/implementation binding",
    )
    manifest = common.parse(_checked(collection_plan["candidate_manifest"]))
    specs = collection_plan["cases"]
    require(common.exact(specs, manifest["cases"]), "Exact authored12case manifest")
    for ref in collection_plan["implementation"].values():
        _checked(ref)
    snapshot = capture_snapshot(anchors["closed_index"])
    terminal = snapshot.json("worker-terminal.json")
    controller = snapshot.json("controller-terminal.json")
    require(
        controller["actual_returncode"] == 0
        and controller["closure_confirmed"] is True
        and controller["primary_error"] is None
        and terminal["primary_error"] is None
        and terminal["cleanup_errors"] == terminal["secondary_errors"] == []
        and terminal["status"] == "COMPLETE_UNADMITTED_NATIVE_PROOF_PENDING_EXTERNAL_OWNED_EXIT"
        and terminal["candidate_wire_export_complete"] is True
        and terminal["candidate_wires"] == 23
        and terminal["policy_helper_POST_calls"]
        == terminal["compiled_rows"]
        == terminal["admitted_rows"]
        == 0,
        "Complete genuine no-model uncompiled proof",
    )
    core, (constructor, space) = _core(), native_constructor()
    tasks = [(s, False) for s in specs] + [(s, True) for s in specs if s["kind"] != "table"]
    verified = [
        _verify_episode(snapshot, i, spec, counter, core, constructor, space)
        for i, (spec, counter) in enumerate(tasks)
    ]
    require(
        common.exact(terminal["rows"], [summary for _, summary in verified])
        and terminal["source_plan_sha256"]
        == controller["plan_sha256"]
        == plan["anchors"]["collection_plan"]["sha256"],
        "Whole terminal population/source plan binding",
    )
    require(
        len(tasks) == 16
        and sum(len(row["proof"]["requests"]) for row, _ in verified) == 43
        and sum(len(row["proof"]["transitions"]) for row, _ in verified) == 27
        and sum(len(row["inputs"]) for row, _ in verified) == 63,
        "Complete16/43/27/63 denominator, not only successful labels",
    )
    wires = snapshot.rows("candidate-native-wire-requests.jsonl")
    expected, cases = [], []
    for (spec, counter), (row, summary) in zip(tasks, verified, strict=True):
        if counter:
            continue
        for j, decision in enumerate(row["proof"]["decisions"]):
            resolution = row["proof"]["transitions"][j]["resolution"]
            actual_decision = {
                "id": spec["id"] + f"-decision-{j:02d}",
                "origin": "owned_local",
                "public_registry_admitted": False,
                "split": spec["split"],
                "family": spec["family"],
                "episode_id": spec["id"],
                "request_sequence": j,
                "selected": resolution["selected"],
                "proof_indices": {
                    "request": j,
                    "transition": j,
                    "private_before": j,
                    "private_after": j + 1,
                },
            }
            require(
                common.exact(decision, actual_decision),
                "Original teacher decision and exact family metadata",
            )
            request = row["proof"]["requests"][j]
            refs = {}
            for key, label, seq in [
                ("raw_observation", "observations.jsonl", request["raw_observation_sequence"]),
                ("private_precondition", "private-dom.jsonl", j),
                ("guarded_transition", "transitions.jsonl", j),
                ("private_postcondition", "private-dom.jsonl", j + 1),
                ("ownership", "ownership.jsonl", 0),
                ("public_ledger", "local-request-ledger.jsonl", 0),
            ]:
                ref = summary["journals"][label]
                refs[key] = {"artifact_path": ref["path"], "sha256": ref["sha256"], "sequence": seq}
            ref = summary["native_wire_bodies"][j]
            refs["native_wire_body"] = {
                "artifact_path": ref["path"],
                "sha256": ref["sha256"],
                "sequence": 0,
            }
            wire = core.UnadmittedDecision(
                core.encode(decision), request["native_wire_utf8"].encode()
            ).wire(refs)
            expected.append(wire)
            cases.append(_case(decision, wire, spec))
    require(
        common.exact(wires, expected),
        "All23 exact candidate rows/body/head/proof refs; final/clones proof-only",
    )
    _validate_split(cases, specs)
    provenance = common.encode(
        {
            "schema_id": SCHEMA,
            "origin": "owned_local",
            "public_registry_admitted": False,
            "admission_plan_sha256": expected_plan_sha256,
            "source_identity_sha256": common.identity(plan["sources"]),
            "proof_index_sha256": plan["anchors"]["closed_index"]["sha256"],
            "counts": dict(COUNTS),
            "admitted_form_rows": 23,
            "final_primary_states_proof_only": 12,
            "wronglocal_clones_proof_only": 4,
            "all_ordered_wire_bodies": 43,
            "compiled_input_sequence": None,
            "compiled_rows": 0,
            "fit_updates": 0,
            "qualified": False,
            "source_proof_limitations": independent["limitations"],
            "admission_scope": "New typed form evidence only; the preserved independent review's UNADMITTED wording describes immutable original proof leaves. No compilation, fitting, deployment qualification or benchmark claim.",
        }
    )
    cases = tuple(cases)
    seal = common.sha(
        plan_bytes
        + b"\0"
        + provenance
        + b"\0"
        + common.encode([c._seal for c in cases])
        + snapshot.digest().encode()
    )
    result = AdmittedFormProgressCollection(snapshot, cases, plan_bytes, provenance, _PROOF, seal)
    result.check()
    return result


def compile_form_references(model, collection, reference_items, *, gate):
    """Lazy original full-processor/CFG adapter; never loads a model itself.

    A reviewed owned CPU driver must gate framework/processor loading before
    calling this function, preserve the authenticated51 reference items, and
    durably seal returned rows/plans plus whole-child exit. No saved result is
    certified merely by constructing this source or passing pure fake tests.
    """
    require(
        type(collection) is AdmittedFormProgressCollection, "Exact new form collection required"
    )
    collection.check()
    plan = common.parse(collection._plan)
    authenticate_sources(plan, plan["sources"])
    common.verify_reference_environment(reference_items, model=model)
    kernel = common.trusted_module("jev_browser_fit_actions", "scripts/jev_browser_fit_actions.py")
    cfg = common.trusted_module("jev_cfg_action_loss", "scripts/jev_cfg_action_loss.py")

    rows, plans = {}, {}
    for split in COUNTS:
        cases = collection.train_cases if split == "train" else collection.development_cases
        wires = collection.compile_wire_projection(split)
        for index, (case, wire) in enumerate(zip(cases, wires, strict=True)):
            # These are existing original ResourceGate registry phases, not new
            # unregistered phase labels. The driver must retain its CPU gates.
            gate.check("before-BF16-processor-compile-" + split + "-" + str(index), "train")
            actual = kernel.compile_actions(
                model,
                [case],
                {"blobs": {"wire_requests.jsonl": common.encode(wire) + b"\n"}},
                identity=common.IDENTITY,
            )
            require(
                len(actual) == 1
                and actual[0]["record"]["original_wire_sha256"] == case.wire_sha256,
                "Full original ordered form request/processor reference join",
            )
            row = actual[0]
            rows[case.id] = row
            plans[case.id] = cfg.compile_action_plan(row, wire["payload"], model.tok)
    require(
        len(rows) == len(plans) == 23
        and list(rows) == [c.id for c in collection.train_cases + collection.development_cases],
        "All23 full targets/masks/prefix/EOS/request-bound CFG plans; no selection",
    )
    return rows, plans
