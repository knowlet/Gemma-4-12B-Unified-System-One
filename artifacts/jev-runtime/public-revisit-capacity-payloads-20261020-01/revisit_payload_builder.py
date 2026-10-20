"""Prospective ordered payloads only: no tokenizer, admission or browser claim.

Actual closed snapshots remain unretagged. Trace sequence replay is explicitly
counterfactual because the old journals may omit intermediate predict refreshes.
The maximum-window recipe is technical synthetic repetition, never live data.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

KIND = "jev_revisit_prospective_ordered_payload_cohort_v1"
PROFILE = "composed-visible-hover-entries-settled-revisit-v1"
PROFILE_MANIFEST_SHA = "f3e65535836518581deab8487f6c20c837bcecfa03173db85f953cc02975ce88"
COHORT_SHA = "b736506ccd403b176b9e4b7e7e74770890e8da786b40b59fcfe277a29ddb3469"
REPLAY_SHA = "e8603635f57d0b401e4414ccc408226e0249a8b8116bb8567573c8f258e01e6b"
MODEL_SHA = "d4437f50784e4bfb7905006c1852a7eb128a3b8a69d62d01daf43dcfab987002"
OLD_MODEL_SHA = "a85ada8458d23642c5c8a4095acf0565e6268f81dcda26f6308942b653e9c9f3"
QUESTIONS_SHA = "66a7054b85040b949e1757311257ea0c5a843c635f9a40c99f6d15ad660fb422"
CORE_SHA = "80249cdd386619b9121160c94d3a071f586cfa40d75977a7e751138175a6b5d0"
MAX_INPUT = 8 * 1024**2
MAX_OUTPUT = 16 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def pin(path, blob):
    return {"path": str(path), "sha256": sha(blob), "size_bytes": len(blob)}


def checked(path, expected, size=None):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary exact input",
    )
    require(path.is_file() and 0 < path.stat().st_size <= MAX_INPUT, "Bounded input")
    before = path.stat()

    def identity(row):
        return (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)

    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        opened = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(opened.st_mode) and identity(opened) == identity(before),
            "Opened input changed",
        )
        blob = stream.read(MAX_INPUT + 1)
        require(
            identity(os.fstat(stream.fileno())) == identity(opened), "Input changed while reading"
        )
    require(
        len(blob) == before.st_size
        and identity(path.stat()) == identity(before)
        and sha(blob) == expected
        and (size is None or len(blob) == size),
        "Input changed or pin differs",
    )
    return blob


def question_builder(model_blob, questions_blob, *, baseline=False):
    """Pinned choose AST through the hint, stopping before timer or HTTP code."""
    require(sha(model_blob) == (OLD_MODEL_SHA if baseline else MODEL_SHA), "Exact model source")
    require(sha(questions_blob) == QUESTIONS_SHA, "Exact question source")
    constants = {}
    for node in ast.parse(questions_blob).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"NEXT_ACTION", "TARGET"}:
                    constants[target.id] = ast.literal_eval(node.value)
    require(set(constants) == {"NEXT_ACTION", "TARGET"}, "Native instruction constants")
    tree = ast.parse(model_blob)
    action = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "action_space"
    )
    choose = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "choose")
    positions = [
        i
        for i, node in enumerate(choose.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "started" for t in node.targets)
    ]
    require(len(positions) == 1, "Exact pre-timer boundary")
    prefix = choose.body[: positions[0]]
    require(
        not any(
            isinstance(n, ast.Name) and n.id in {"post_json", "time", "CLIENT"}
            for n in ast.walk(ast.Module(body=prefix, type_ignores=[]))
        ),
        "No timer, HTTP or runtime in prefix",
    )
    choose.body = prefix + [ast.Return(value=ast.Name(id="body", ctx=ast.Load()))]
    choose.name = "prospective_body"
    namespace = {
        **constants,
        "copy": copy,
        "os": SimpleNamespace(environ={"TYPESAFE_MODEL": "fixture-no-LLM"}),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[action, choose], type_ignores=[])),
            "<pinned-native-body-and-hint-only>",
            "exec",
        ),
        namespace,
    )
    return namespace["prospective_body"]


def core_module(blob):
    require(sha(blob) == CORE_SHA, "Exact pure tracker source")
    namespace = {"__name__": "pinned_pure_capacity_revisit_core"}
    exec(compile(blob, "<pinned-pure-revisit-core>", "exec"), namespace)
    return SimpleNamespace(**namespace)


def payload_row(case_id, scope, page, goal, history, metadata, new, old, provenance):
    before = canonical(page)
    original = old(page, goal, history)
    body = new(page, goal, history, observation_revisit=metadata)
    removed = copy.deepcopy(body)
    hint = removed["state"].pop("observation_revisit", None)
    require(wire(removed) == wire(original), "New request changed beyond explicit state hint")
    require(bool(hint) == bool(metadata["observation_notes"]), "Hint signal omitted or invented")
    body["model"] = "generated-actions-none"
    require(canonical(page) == before, "Raw page/guards mutated")
    return {
        "case_id": case_id,
        "scope": scope,
        "provenance": provenance,
        "raw_page_sha256": sha(before),
        "raw_observation_profile": copy.deepcopy(page["observation_profile"]),
        "guards_sha256": sha(canonical(page["guards"])),
        "metadata": metadata,
        "baseline_payload": original,
        "payload": body,
        "same_historical_HTTP_wire_claimed": False,
        "capacity_admission_measured": False,
    }


def matrix_rows(cohort, cells, new, old, core):
    require(cohort["fixed_denominator"] == len(cohort["rows"]) == 16, "Complete original16 phases")
    rows, trackers = [], {}
    goal = "Inspect the currently offered local fixture controls."
    for ordinal, selected in enumerate(cohort["rows"]):
        require(selected["ordinal"] == ordinal, "Original phase order")
        cell_id, phase = selected["cell_ordinal"], selected["phase"]
        cell = cells[cell_id]
        require(cell["variant"] == "v8" and cell["ordinal"] == cell_id, "Original genuine V8 cell")
        page, saved = cell[phase], cell[phase + "_question_payload"]
        require(
            sha(json.dumps(page, sort_keys=True).encode()) == selected["saved_snapshot_sha256"],
            "Original snapshot authority",
        )
        original = old(page, goal, [])

        def authority(value):
            return (
                json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            ).encode()

        require(
            authority(original) == authority(saved)
            and sha(authority(saved)) == selected["saved_canonical_question_payload_sha256"],
            "Original matrix body authority",
        )
        if cell_id not in trackers:
            tracker = core.ObservationRevisitTracker(include_projection=True)
            trackers[cell_id] = tracker
            initial = tracker.observe(cell["before"], observation_sequence=0)
        else:
            tracker, initial = trackers[cell_id], None
        metadata = initial if phase == "before" else tracker.observe(page, observation_sequence=1)
        require(metadata is not None, "No duplicate before phase")
        rows.append(
            payload_row(
                f"matrix-{cell_id}-{phase}",
                "GENUINE_V8_CELL_PHASE_REPLAY_EACH_CELL_INDEPENDENT",
                page,
                goal,
                [],
                metadata,
                new,
                old,
                selected,
            )
        )
    return rows


def trace_rows(name, snapshots, source_ref, new, old, core):
    require(type(snapshots) is list and len(snapshots) == 17, "All17 saved snapshots per trace")
    tracker = core.ObservationRevisitTracker(include_projection=True)
    return [
        payload_row(
            f"trace-{name}-{i:02d}",
            "COUNTERFACTUAL_V7_SAVED_SNAPSHOT_SEQUENCE_NOT_COMPLETE_AGENT_OBSERVE_JOURNAL",
            state["page"],
            state["goal"],
            state["history"],
            tracker.observe(state["page"], observation_sequence=i),
            new,
            old,
            {
                "source_states": source_ref,
                "saved_snapshot_index": i,
                "missing_intermediate_predict_refreshes_possible": True,
            },
        )
        for i, state in enumerate(snapshots)
    ]


def technical_row(seed, new, old, core):
    tracker = core.ObservationRevisitTracker(include_projection=True)
    for sequence in range(core.MAX_SEQUENCE - 32, core.MAX_SEQUENCE + 1):
        metadata = tracker.observe(seed, observation_sequence=sequence)
    require(len(metadata["exact"]["previous_observation_sequences"]) == 32, "Full bounded window")
    return payload_row(
        "technical-max32-window",
        "TECHNICAL_33_DUPLICATE_REPLAYS_MAX_SEQUENCE_DIGITS_NOT_NEW_OBSERVATIONS",
        seed,
        "Inspect the currently offered local fixture controls.",
        [],
        metadata,
        new,
        old,
        {
            "recipe": "First actual matrix before-page is reused33 times; no additional live observation is claimed.",
            "repetitions": 33,
            "window": 32,
            "last_sequence": core.MAX_SEQUENCE,
        },
    )


def prepare(root, profile_root, cohort_path, replay_path, output, *, expected_source_sha256):
    """Save declared prospective bodies; imports no processor or model package."""
    root, profile_root, output = (
        Path(root).absolute(),
        Path(profile_root).absolute(),
        Path(output).absolute(),
    )
    self_path = Path(__file__).absolute()
    self_blob = checked(self_path, expected_source_sha256)
    manifest_blob = checked(profile_root / "profile-manifest.json", PROFILE_MANIFEST_SHA)
    manifest = json.loads(manifest_blob)
    require(
        manifest["profile"] == PROFILE and manifest["runtime_file_count"] == 8,
        "New materialized profile identity",
    )
    sources = {
        label: checked(profile_root / label, row["sha256"], row["size_bytes"])
        for label, row in manifest["artifact_files"].items()
    }
    new_model, questions, core_blob = (
        sources["runtime/jev_ultrafast/" + name]
        for name in ("model.py", "questions.py", "observation_revisit.py")
    )
    old_model = sources["parent_v8/runtime/jev_ultrafast/model.py"]
    new, old, core = (
        question_builder(new_model, questions),
        question_builder(old_model, questions, baseline=True),
        core_module(core_blob),
    )
    cohort_blob, replay_blob = checked(cohort_path, COHORT_SHA), checked(replay_path, REPLAY_SHA)
    cohort, replay = json.loads(cohort_blob), json.loads(replay_blob)
    cells = {}
    for selected in cohort["rows"]:
        ref = selected["source_cell"]
        if selected["cell_ordinal"] not in cells:
            cells[selected["cell_ordinal"]] = json.loads(
                checked(root / ref["path"], ref["sha256"], ref["size_bytes"])
            )
    rows = matrix_rows(cohort, cells, new, old, core)
    traces = []
    for selected in replay["results"]:
        ref = selected["states"]
        snapshots = json.loads(checked(root / ref["path"], ref["sha256"], ref["size_bytes"]))
        prepared = trace_rows(selected["case"], snapshots, ref, new, old, core)
        rows.extend(prepared)
        traces.append(
            {
                "case": selected["case"],
                "rows": len(prepared),
                "exact_revisits": sum(
                    bool(r["metadata"]["exact"]["previous_observation_sequences"]) for r in prepared
                ),
                "projection_revisits": sum(
                    bool(r["metadata"]["projection"]["previous_observation_sequences"])
                    for r in prepared
                ),
            }
        )
    rows.append(technical_row(cells[cohort["rows"][0]["cell_ordinal"]]["before"], new, old, core))
    require(
        len(rows) == 68 and len({r["case_id"] for r in rows}) == 68,
        "Complete16+51+1 prospective cohort",
    )
    require(
        not output.exists()
        and not any(p.is_symlink() for p in (output, *output.parents))
        and not output.is_relative_to(profile_root)
        and not profile_root.is_relative_to(output),
        "Fresh disjoint ordinary payload output",
    )
    source_union = {
        str(profile_root / label): manifest["artifact_files"][label]
        for label in manifest["artifact_files"]
        if label.startswith(("runtime/", "sources/"))
    }
    source_union[str(profile_root / "parent_v8/runtime/jev_ultrafast/model.py")] = manifest[
        "artifact_files"
    ]["parent_v8/runtime/jev_ultrafast/model.py"]
    source_union[str(self_path)] = {"sha256": sha(self_blob), "size_bytes": len(self_blob)}
    output.mkdir(parents=True, exist_ok=False)
    total = 0

    def save(name, blob):
        nonlocal total
        total += len(blob)
        require(total <= MAX_OUTPUT, "Prospective payload output bound; no truncation permitted")
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(blob)
        return pin(name, blob)

    for ordinal, row in enumerate(rows):
        row["payload_ref"] = save(f"payloads/{ordinal:02d}.json", wire(row.pop("payload")))
        row["baseline_ref"] = save(
            f"baseline/{ordinal:02d}.json", wire(row.pop("baseline_payload"))
        )
    result = {
        "schema_id": KIND,
        "status": "ORDERED_PAYLOADS_ONLY_NO_PROCESSOR_OR_CAPACITY_ADMISSION",
        "profile": PROFILE,
        "profile_manifest": pin(profile_root / "profile-manifest.json", manifest_blob),
        "matrix_cohort": pin(cohort_path, cohort_blob),
        "trace_replay_authority": pin(replay_path, replay_blob),
        "source_union": source_union,
        "source_union_files": len(source_union),
        "rows": rows,
        "counts": {
            "genuine_V8_cell_phase_replays": 16,
            "counterfactual_V7_saved_snapshots": 51,
            "technical_repetition": 1,
        },
        "trace_counts": traces,
        "old_technical95_or_other_authority_modified": False,
        "genuine_new_profile_browser_observation_claimed": False,
        "original_HTTP_wire_identity_claimed": False,
        "processor_tokenizer_model_GPU_browser_calls": 0,
    }
    save("payload-cohort.json", canonical(result) + b"\n")
    require(checked(self_path, expected_source_sha256) == self_blob, "Builder source changed")
    require(
        checked(profile_root / "profile-manifest.json", PROFILE_MANIFEST_SHA) == manifest_blob,
        "Profile identity changed",
    )
    return result
