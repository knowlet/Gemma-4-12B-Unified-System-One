"""Explicit source-selected FIND arm over the unchanged native/Q95 control.

Dry by default. No helper seed is discovered, and no production plan is created.
Only a future independently owned run may enter the selected runtime context.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

from checked_source import module

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
PROFILE = "jev-current-nf4-v7-q95-find-browser-v1"
OWNER_PROFILE = "jev-current-nf4-v7-q95-find-browser-owned-v1"
CONTROL03 = "artifacts/jev-runtime/native-v2-q95-control-20261023-03"
CONTROL03_UNION_SHA = "b397f5f0f0ad1e17b1f37a6861ac1a2b7e6c82d8e33b510c957b375e9ed7ce7a"
PARENT = "artifacts/jev-runtime/native-v2-q95-browser-control-20261022-01/control_driver.py"
PARENT_SHA = "00295d7110522dd346320e598298f923bda56fa4a73dba427035371801c73543"
SERVICE = "artifacts/jev-runtime/browser-find-text-s1-nativev2-integration-20261022-01"
SERVICE_SHA = "c3712f8d29851a521a3e9b5a1691d3fb8cf7ae9f82aeb1051c27aa8c64b04c1a"
SERVICE_SELECTION_SHA = "b54a709873c6c3a6ed41894c5fb12aca0ba806791afa718820578346e0ee5d14"
EXPANDED_SELECTION_SHA = "ff78514e79baf8ea2a39b72a7718e990dd7f3e00a0bd6da94929ebc304900fb0"
PROJECT_SHA = "0c1681c735c8b0ef078dbd57f41cd5d55d5ca6fca5e880c3ed94225fdaa385c5"
BASE_PLAN = "artifacts/jev-runtime/native-v2-browser-execution-20261022-04/v2-01/selected-plan.json"
BASE_PLAN_SHA = "73bff770e242b93d5b7b53d24ef933926bd63897a4707d6d2e2a0f06f4438b6d"
SERVICE_PROFILE = "jev-native-v2-find-text-source-integration-v1"
CLIENT_PARENT = "artifacts/jev-runtime/browser-find-text-v7-client-integration-20261023-01/find_client_selector.py"
CLIENT_PARENT_SHA = "0c663f03f9900c5b7c1d0d3e7c64ad51908133eedfea53bd2baf4b5ba3253150"
CLIENT = (
    "artifacts/jev-runtime/browser-find-text-v7-client-budget-20261023-02/find_client_selector.py"
)
CLIENT_SHA = "f56f0800a981370f5a0e74d8722677721cb2043b03596c7779227502d4565ee1"
CLIENT_PROFILE = "jev-v7-find-text-client-budget-source-v2"
NATIVE = "scripts/jev_native_v2_browser.py"
NATIVE_SHA = "c513dc5bcb091de4fd0d4bd7a0776078e82be16106328b6de9d8424e32f2ee11"
SOURCE_NAMES = (
    "launch_browser.py",
    "browser_entry.py",
    "control_driver.py",
    "gemma_helper.py",
    "policy_boundary.py",
    "prepare_control.py",
    "checked_source.py",
)


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def read(path, maximum=1024**2):
    path = Path(path).absolute()
    need(
        path.is_relative_to(ROOT)
        and ".." not in path.parts
        and not any(p.is_symlink() for p in (path, *path.parents))
        and path.is_file(),
        "Ordinary confined FIND source/evidence",
    )
    with path.open("rb") as stream:
        blob = stream.read(maximum + 1)
    need(len(blob) <= maximum, "Bounded FIND source/evidence")
    return blob


def ref(path):
    path = Path(path).absolute()
    blob = read(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(blob), "size_bytes": len(blob)}


def checked(reference):
    need(
        type(reference) is dict and set(reference) == {"path", "sha256", "size_bytes"},
        "Exact FIND source reference",
    )
    relative = Path(reference["path"])
    need(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.as_posix() == reference["path"],
        "Repo-relative FIND source",
    )
    blob = read(ROOT / relative)
    need(
        len(blob) == reference["size_bytes"] and sha(blob) == reference["sha256"],
        "Selected FIND source bytes differ",
    )
    return blob


def fixed(path, digest):
    value = ref(ROOT / path)
    need(value["sha256"] == digest, "Frozen FIND composition source changed")
    return value


def source_refs():
    """Small source-only union; no seed discovery, tensor or runtime import."""
    union_ref = fixed(CONTROL03 + "/source-union.json", CONTROL03_UNION_SHA)
    control = json.loads(checked(union_ref))
    need(len(control["sources"]) == 7, "Frozen control03 seven sources")
    service_ref = fixed(SERVICE + "/selection.py", SERVICE_SHA)
    selection_ref = fixed(SERVICE + "/selection.json", SERVICE_SELECTION_SHA)
    selection = json.loads(checked(selection_ref))
    need(selection["profile"] == SERVICE_PROFILE, "Actual service selector profile")
    client_ref = fixed(CLIENT, CLIENT_SHA)
    descriptor = selection["project"]["descriptor"]
    need(len(descriptor["project_sources"]) == 22, "Full selected FIND project22")
    expanded_ref = fixed(
        (HERE / "service-selection.json").relative_to(ROOT).as_posix(), EXPANDED_SELECTION_SHA
    )
    expanded = json.loads(checked(expanded_ref))
    project = expanded["project"]
    base_ref = fixed(BASE_PLAN, BASE_PLAN_SHA)
    original = json.loads(checked(base_ref))["sources"]
    original_by_path = {row["path"]: row for row in original}
    need(
        project["sha256"] == PROJECT_SHA
        and sha(json.dumps(project["descriptor"]).encode()) == PROJECT_SHA
        and len(project["descriptor"]["project_sources"]) == 84
        and len(original) == 785
        and all(row in original for row in project["descriptor"]["project_sources"])
        and all(
            row in project["descriptor"]["project_sources"] for row in descriptor["project_sources"]
        )
        and {k: v for k, v in expanded.items() if k != "project"}
        == {k: v for k, v in selection.items() if k != "project"}
        and {k: v for k, v in project["descriptor"].items() if k != "project_sources"}
        == {k: v for k, v in descriptor.items() if k != "project_sources"},
        "Only explicit original-authority project84 extension",
    )
    rows = [
        union_ref,
        base_ref,
        *control["sources"],
        service_ref,
        selection_ref,
        expanded_ref,
        selection["integration"],
        selection["driver"],
        *selection["scripts"].values(),
        *descriptor["components"].values(),
        *descriptor["project_sources"],
        *project["descriptor"]["project_sources"],
        client_ref,
        fixed(CLIENT_PARENT, CLIENT_PARENT_SHA),
        fixed(PARENT, PARENT_SHA),
        *(
            ref(ROOT / "artifacts/jev-runtime/native-v2-q95-browser-control-20261022-01" / name)
            for name in (
                "browser_entry.py",
                "policy_boundary.py",
                "launch_browser.py",
                "gemma_helper.py",
            )
        ),
        *(
            original_by_path[path]
            for path in (
                "artifacts/jev-runtime/native-v2-browser-execution-20261022-04/launch_browser.py",
                "scripts/jev_browser_profile_v7_benchmark.py",
                "scripts/jev_browser_benchmark.py",
                "scripts/jev_public_web_cases.py",
                "scripts/jev_heldout_public_cases.py",
                "scripts/jev_observation_profile_v7.py",
                "artifacts/jev-runtime/upstream/jev_ultrafast/browser.py",
                "artifacts/jev-runtime/upstream/jev_ultrafast/__init__.py",
                "artifacts/jev-runtime/upstream/jev_ultrafast/questions.py",
                "artifacts/jev-runtime/upstream/jev_ultrafast/snapshot.js",
                "artifacts/jev-runtime/upstream/jev_ultrafast/static/fixture.html",
            )
        ),
        fixed(
            "scripts/jev_decision3_browser_wire.py",
            "c048d7779e60fce5d1a56b38e26b9ddf5ed9e1ff9c6adc478786d0267cca3027",
        ),
        fixed(
            "artifacts/jev-runtime/upstream/jev_ultrafast/agent.py",
            "aaf8b271cb0ee6643fe7beb525328ab80de8568c7ac044d02b9c6721161e7c8b",
        ),
        fixed(
            "artifacts/jev-runtime/upstream/jev_ultrafast/model.py",
            "a85ada8458d23642c5c8a4095acf0565e6268f81dcda26f6308942b653e9c9f3",
        ),
        *(ref(HERE / name) for name in SOURCE_NAMES),
    ]
    result = {}
    for row in rows:
        checked(row)
        need(
            row["path"] not in result or result[row["path"]] == row, "FIND source identity conflict"
        )
        result[row["path"]] = row
    return [result[key] for key in sorted(result)]


def binding():
    return {
        "profile": PROFILE,
        "owner_profile": OWNER_PROFILE,
        "service_profile": SERVICE_PROFILE,
        "service_source_sha256": SERVICE_SHA,
        "service_selection_sha256": SERVICE_SELECTION_SHA,
        "expanded_service_selection_sha256": EXPANDED_SELECTION_SHA,
        "project_descriptor_sha256": PROJECT_SHA,
        "project_source_count": 84,
        "client_profile": CLIENT_PROFILE,
        "client_source_sha256": CLIENT_SHA,
        "client_parent_source_sha256": CLIENT_PARENT_SHA,
        "original_native_sha256": NATIVE_SHA,
        "observation_profile": "composed-visible-text-settled-v7",
        "observation_helper_sha256": "d9f4236d1d46c9472c3d6c1551342639ecb870e6c0dd419246927645410ea868",
        "max_steps": 16,
        "seed": 20261014,
        "helper_seconds": 20,
        "helper_tokens": 128,
        "helper_context": 8192,
    }


def authenticate_plan(plan):
    need(plan.get("find_binding") == binding(), "Explicit exact FIND arm binding")
    selected = {row["path"]: row for row in plan["sources"]}
    need(len(selected) == len(plan["sources"]), "Unique selected FIND source rows")
    rows = source_refs()
    need(all(selected.get(row["path"]) == row for row in rows), "Complete exact FIND source union")
    return rows


def derive_function(blob, name, before, after):
    tree = ast.parse(blob)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    original = b"".join(blob.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno])
    need(
        original.count(before) == 1 and after not in original, "Single declared FIND function seam"
    )
    derived = original.replace(before, after)
    inverse = derived.replace(after, before)
    need(
        inverse == original
        and ast.dump(ast.parse(inverse), include_attributes=False)
        == ast.dump(ast.parse(original), include_attributes=False),
        "Exact FIND function inverse bytes/AST",
    )
    return derived, {
        "function": name,
        "original_function_sha256": sha(original),
        "derived_function_sha256": sha(derived),
        "declared_edits": 1,
        "inverse_bytes_AST": True,
    }


def derivations():
    parent_blob = checked(fixed(PARENT, PARENT_SHA))
    native_blob = checked(fixed(NATIVE, NATIVE_SHA))
    execute, execute_proof = derive_function(
        parent_blob,
        "execute",
        b'    native = selected_module(NATIVE, NATIVE_SHA, "_native_q95_original_driver")\n',
        b"    native = _find_selected_native\n",
    )
    groups, groups_proof = derive_function(
        native_blob,
        "run_groups",
        b"    import jev_browser_profile_v7_benchmark as v7\n",
        b"    v7 = _find_selected_v7\n",
    )
    return execute, groups, {"control_execute": execute_proof, "native_run_groups": groups_proof}


class SelectedV7:
    def __init__(self, client, output):
        self.client, self.output, self.groups = client, Path(output), []

    def prepare_bundle(self, directory, *, case_module):
        expected = (("known4", "original"), ("public6", "heldout"))
        need(len(self.groups) < 2, "Only the original two FIND groups")
        group, mode = expected[len(self.groups)]
        need(
            case_module == mode and Path(directory) == self.output / group / "runtime-bundle",
            "Exact FIND group order and isolated output",
        )
        bundle, manifest = self.client.prepare_bundle(
            directory, case_module=case_module, profile=CLIENT_PROFILE
        )
        self.client.verify_bundle(bundle, manifest)
        self.groups.append(
            {"group": group, "case_module": mode, "manifest": ref(bundle / "bundle-manifest.json")}
        )
        return bundle, manifest

    def verify_bundle(self, bundle, manifest):
        return self.client.verify_bundle(bundle, manifest)


def expected_service_proof():
    return {
        "original_sha256": NATIVE_SHA,
        "derived_sha256": "349413e78b4cb117e3dc1e2e100fbadb2f039264174ba8acb3840f0479ca688f",
        "declared_edits": 1,
        "inverse_exact": True,
        "constructor_LoRA_HTTP_helper_resource_math_unchanged": True,
    }


@contextmanager
def selected_native(plan):
    initial = set(sys.modules)
    rows = authenticate_plan(plan)
    execute, groups, proofs = derivations()
    primary = None
    try:
        need(
            not initial.intersection(
                {"_find04_service_selector", "_find04_client_selector", "_find04_control_parent"}
            ),
            "Fresh private FIND arm namespaces",
        )
        service = module(SERVICE + "/selection.py", SERVICE_SHA, "_find04_service_selector")
        client = module(CLIENT, CLIENT_SHA, "_find04_client_selector")
        need(
            service.PROFILE == SERVICE_PROFILE and client.PROFILE == CLIENT_PROFILE,
            "Loaded selector profiles",
        )
        with service.selected(
            checked(
                fixed(
                    (HERE / "service-selection.json").relative_to(ROOT).as_posix(),
                    EXPANDED_SELECTION_SHA,
                )
            ),
            EXPANDED_SELECTION_SHA,
            profile=SERVICE_PROFILE,
        ) as (_, native, service_proof):
            need(
                service_proof == expected_service_proof(), "Actual selected FIND service derivation"
            )
            selected_v7 = SelectedV7(client, ROOT / plan["output"])
            native._find_selected_v7 = selected_v7
            exec(
                compile(
                    groups, str(ROOT / NATIVE) + "::find04-run-groups", "exec", dont_inherit=True
                ),
                vars(native),
            )
            need(
                native.run_groups.__globals__ is vars(native),
                "Same private native globals required",
            )
            parent = module(PARENT, PARENT_SHA, "_find04_control_parent")
            parent.HERE, parent.PROFILE = HERE, PROFILE
            parent._find_selected_native = native
            exec(
                compile(
                    execute, str(ROOT / PARENT) + "::find04-execute", "exec", dont_inherit=True
                ),
                vars(parent),
            )
            original_metadata = native.metadata_plan

            def metadata(*args, **kwargs):
                return {**original_metadata(*args, **kwargs), "find_binding": binding()}

            native.metadata_plan = metadata
            try:
                yield parent, native, selected_v7, {**proofs, "service": service_proof}
            except BaseException as error:
                primary = error
                raise
            finally:
                try:
                    need(authenticate_plan(plan) == rows, "Final FIND source union unchanged")
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note("Final FIND arm source verification failed: " + str(error))
    finally:
        for name in (
            "_find04_service_selector",
            "_find04_client_selector",
            "_find04_control_parent",
        ):
            if name not in initial:
                sys.modules.pop(name, None)


def evidence_json(path):
    return json.loads(read(path, 16 * 1024**2))


def verify_outputs(plan, output):
    """Require actual emitted experiment bundles, task results and Q95 joins."""
    authenticate_plan(plan)
    output = Path(output)
    need(output == ROOT / plan["output"], "Exact owned FIND output")
    saved = evidence_json(output.parent / "find-selection-receipt.json")
    _, _, proofs = derivations()
    need(
        set(saved) == {"profile", "binding", "proofs", "groups", "model_quality_qualified"}
        and saved["profile"] == PROFILE
        and saved["binding"] == binding()
        and saved["proofs"] == {**proofs, "service": expected_service_proof()}
        and saved["model_quality_qualified"] is False,
        "Exact actual FIND selector receipt",
    )
    native = evidence_json(output / "receipt.json")
    need(
        native["profile"] == PROFILE
        and native["find_binding"] == binding()
        and native["stage"] == "complete_pending_external_closure"
        and native["attempts"] == native["planned_denominator"] == 10,
        "Complete actual selected FIND ten-task receipt",
    )
    client = module(CLIENT, CLIENT_SHA, "_find04_closure_client")
    try:
        expected = [("known4", "original", client.KNOWN4), ("public6", "heldout", client.PUBLIC6)]
        need(len(saved["groups"]) == len(native["groups"]) == 2, "Both FIND groups required")
        control = evidence_json(output.parent / "q95-control/receipt.json")
        need(
            control["profile"] == PROFILE
            and control["primary_error"] is None
            and control["stage"] == "complete_pending_external_closure"
            and control["selection_sha256"] == plan["helper_selection"]["sha256"]
            and control["native_receipt"]
            == {
                "sha256": sha(read(output / "receipt.json", 16 * 1024**2)),
                "size_bytes": len(read(output / "receipt.json", 16 * 1024**2)),
            }
            and control["native_journal_covers_external_helper"] is False
            and control["external_owned_closure_required"] is True
            and len(control["group_indexes"]) == 2,
            "Actual FIND control/helper receipt and two-field native pin",
        )
        for i, (group, mode, ids) in enumerate(expected):
            bundle = output / group / "runtime-bundle"
            row = saved["groups"][i]
            manifest_path = bundle / "bundle-manifest.json"
            need(
                row == {"group": group, "case_module": mode, "manifest": ref(manifest_path)},
                "Actual source-bound FIND bundle join",
            )
            manifest = evidence_json(manifest_path)
            client.verify_bundle(bundle, manifest)
            need(manifest["fixed_case_ids"] == list(ids), "Complete ordered FIND task IDs")
            result = evidence_json(output / group / "results/summary.json")
            need(
                native["groups"][i] == {"group": group, "denominator": len(ids), "summary": result}
                and result["attempts"] == len(ids)
                and not result.get("observation_profile_failed"),
                "Actual complete FIND group summary join",
            )
            index_row = control["group_indexes"][i]
            need(
                Path(index_row["group_directory"]) == output / group,
                "Exact external Q95 group directory",
            )
            index = evidence_json(output / group / "external-q95-request-index.json")
            need(
                index["pruned"] is False
                and len(index["request_ids"]) == len(set(index["request_ids"]))
                and set(index["request_ids"]) == set(index["requests"])
                and index_row["external_q95_requests"] == len(index["requests"]),
                "Complete unpruned external Q95 index join",
            )
            for request_id, request in index["requests"].items():
                need(
                    len(request_id) == 32 and all(c in "0123456789abcdef" for c in request_id),
                    "Exact Q95 request identifier",
                )
                paths = [Path(request[key]["path"]) for key in ("raw", "outer_completion")]
                need(
                    paths[0].name == "raw.json"
                    and paths[1].name == "outer-completion.json"
                    and paths[0].parent == paths[1].parent
                    and paths[0].parent.name == request_id,
                    "Actual Q95 raw/final request paths",
                )
                for key, path in zip(("raw", "outer_completion"), paths):
                    blob = read(path, 16 * 1024**2)
                    need(
                        request[key]
                        == {"path": str(path), "sha256": sha(blob), "size_bytes": len(blob)},
                        "Actual Q95 raw/final pins",
                    )
                raw, final = (evidence_json(path) for path in paths)
                need(
                    raw["request_id"] == final["request_id"] == request_id
                    and raw["guard_restored"] is True
                    and raw["server_terminal"] is True
                    and raw["error"] is None
                    and final["completed"] is True
                    and final["final_gate_measured"] is True
                    and final["raw_sha256"] == request["raw"]["sha256"]
                    and final["selected_status"] == request["selected_status"],
                    "Actual terminal Q95 raw/final evidence join",
                )
    finally:
        sys.modules.pop("_find04_closure_client", None)


def execute(plan, descriptor, helper, boundary, rest):
    with selected_native(plan) as (parent, native, v7, proofs):
        original_paths = native.composition_paths
        rows = authenticate_plan(plan)
        native.composition_paths = lambda: {
            **original_paths(),
            **{r["path"]: ROOT / r["path"] for r in rows},
        }
        result = parent.execute(plan, descriptor, helper, boundary, rest)
        native.write(
            (ROOT / plan["output"]).parent / "find-selection-receipt.json",
            {
                "profile": PROFILE,
                "binding": binding(),
                "proofs": proofs,
                "groups": v7.groups,
                "model_quality_qualified": False,
            },
        )
        verify_outputs(plan, ROOT / plan["output"])
        return result


def main(argv=None):
    parent = module(PARENT, PARENT_SHA, "_find04_main_parent")
    parent.HERE, parent.PROFILE, parent.execute = HERE, PROFILE, execute
    try:
        return parent.main(argv)
    finally:
        sys.modules.pop("_find04_main_parent", None)


if __name__ == "__main__":
    raise SystemExit(main())
