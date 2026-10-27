"""Pinned, independently replayed owned link-navigation collection pipeline.

Planning does not launch a browser or read benchmark examples. Collection uses
only fresh authored TRAIN fixtures and records the original protocol journal.
Every output remains quarantined until explicit, independent evidence replay.
An admitted projection is still not automatically eligible for model fitting.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import jev_browser_benchmark as benchmark
import jev_decision3_link_navigation_collection as links
import jev_decision3_link_navigation_scenarios as scenarios
import jev_decision3_text_history_collection as base

SCHEMA = "decision3-owned-link-navigation-pipeline-v1"
ROOT = base.ROOT
PROFILE = base.SETTLED_V7_PROFILE
_ADMISSION_PROOF = object()


def _source_inventory():
    result = base._source_inventory()
    for path in (Path(__file__), Path(links.__file__), Path(scenarios.__file__)):
        path = path.absolute()
        if path.is_symlink() or not path.is_file():
            raise ValueError("Pipeline source must be an ordinary file")
        raw = path.read_bytes()
        result[path.relative_to(ROOT).as_posix()] = {"sha256": base._sha(raw), "bytes": len(raw)}
    return dict(sorted(result.items()))


def _episodes(seed, collection_id, exclusions):
    specs = scenarios.scenarios(seed, set(exclusions["goal_sha256"]))
    if not specs or len({spec["id"] for spec in specs}) != len(specs):
        raise ValueError("Unique authored TRAIN scenarios are required")
    result = []
    for spec in specs:
        scenarios._check_spec(spec)
        if (
            spec["id"] in exclusions["scenario_ids"]
            or spec["ancestor_id"] in exclusions["ancestor_ids"]
            or base._goal_hash(spec["goal"]) in exclusions["goal_sha256"]
        ):
            raise ValueError("Excluded goal or identity cannot enter training")
        result.append(
            {
                "episode_id": collection_id + "." + spec["id"],
                "scenario": copy.deepcopy(spec),
                "scenario_sha256": base._digest(spec),
                "documents_sha256": base._digest(scenarios.render_documents(spec)),
            }
        )
    return result


def _make_plan(
    *,
    output_dir,
    upstream,
    collection_id,
    seed,
    exclusion_manifest_path,
    chromium,
    observation_helper_sha256,
    fresh,
):
    if type(collection_id) is not str or not base._IDENTITY.fullmatch(collection_id):
        raise ValueError("Use a bounded new collection identity")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Use a bounded explicit scenario seed")
    if chromium is None:
        raise ValueError("A pinned Chromium executable is required")
    output = base._prospective_output(output_dir) if fresh else Path(output_dir).absolute()
    exclusions, exclusions_sha = base.exclusion_manifest(exclusion_manifest_path)
    upstream_path = Path(upstream).expanduser().resolve(strict=True)
    upstream_pin = benchmark.upstream_manifest(upstream_path)
    bundle = base._prepare_observation_profile(upstream_path, PROFILE, observation_helper_sha256)
    sources = _source_inventory()
    manifest = bundle.materialized_manifest()
    manifest_bytes = (
        json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode()
    return {
        "schema_id": SCHEMA,
        "schema_version": 1,
        "split": "train",
        "collection_id": collection_id,
        "seed": seed,
        "output_dir": str(output),
        "upstream_path": str(upstream_path),
        "upstream": upstream_pin,
        "sources": sources,
        "collector_sha256": sources[Path(__file__).absolute().relative_to(ROOT).as_posix()][
            "sha256"
        ],
        "exclusion_manifest_path": str(Path(exclusion_manifest_path).resolve()),
        "exclusion_manifest_sha256": exclusions_sha,
        "exclusions": exclusions,
        "observation_profile": PROFILE,
        "observation_helper_sha256": observation_helper_sha256,
        "observation_profile_manifest": manifest,
        "observation_profile_manifest_sha256": base._sha(manifest_bytes),
        "chromium": base._executable_pin(chromium),
        "model_alias": base.MODEL_ALIAS,
        "episodes": _episodes(seed, collection_id, exclusions),
        "model_score": None,
        "fit_eligible": False,
        "admission_status": "not_collected",
    }


def collection_plan(
    *,
    output_dir,
    upstream,
    collection_id,
    seed,
    exclusion_manifest_path,
    chromium,
    observation_helper_sha256,
):
    """Construct the reviewed deterministic plan without browser or filesystem writes."""
    return _make_plan(
        output_dir=output_dir,
        upstream=upstream,
        collection_id=collection_id,
        seed=seed,
        exclusion_manifest_path=exclusion_manifest_path,
        chromium=chromium,
        observation_helper_sha256=observation_helper_sha256,
        fresh=True,
    )


def _verify_plan(plan, *, fresh):
    if type(plan) is not dict or plan.get("schema_id") != SCHEMA:
        raise ValueError("Foreign plan schema")
    expected = _make_plan(
        output_dir=plan["output_dir"],
        upstream=plan["upstream_path"],
        collection_id=plan["collection_id"],
        seed=plan["seed"],
        exclusion_manifest_path=plan["exclusion_manifest_path"],
        chromium=plan["chromium"]["path"],
        observation_helper_sha256=plan["observation_helper_sha256"],
        fresh=fresh,
    )
    if base._encode(plan) != base._encode(expected):
        raise ValueError("Plan, sources, exclusions, Chromium, upstream or V7 manifest changed")


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    base._write_json(path, value)
    return base._sha(path.read_bytes())


def _observe_request(events, request):
    try:
        main = request.is_navigation_request() and request.frame.parent_frame is None
        events.append(
            {
                "url": request.url,
                "method": request.method,
                "redirected": request.redirected_from is not None,
                "main_document": main,
            }
        )
    except Exception:
        events.append({"observer_error": True})


def _verify_network(spec, trace, events):
    values = trace.get("network")
    if type(values) is not list or not values:
        raise ValueError("Original authored document requests are missing")
    first = values[0]
    if type(first) is not str or not first.endswith(spec["start_path"]):
        raise ValueError("Initial request does not match the episode")
    origin = first[: -len(spec["start_path"])]
    guard = links.EpisodeNetworkGuard(spec, origin)
    if type(events) is not list or not events:
        raise ValueError("Independent browser request evidence is missing")
    for event in events:
        if (
            type(event) is not dict
            or set(event) != {"url", "method", "redirected", "main_document"}
            or type(event["redirected"]) is not bool
            or type(event["main_document"]) is not bool
            or not guard.check(
                event["url"],
                event["method"],
                redirected=event["redirected"],
                main_document=event["main_document"],
            )
        ):
            raise ValueError("Unowned, redirected or unexpected browser request")
    guard.verify()
    if guard.requests != values:
        raise ValueError("Independent request evidence differs from native collector")
    return origin


def collect(plan, *, plan_sha256, collector_sha256):
    """Execute an explicitly pinned plan; outputs always remain pending admission."""
    if base._hash_pin(plan_sha256) != base._digest(plan):
        raise ValueError("Reviewed plan SHA256 differs")
    if base._hash_pin(collector_sha256) != base._sha(Path(__file__).read_bytes()):
        raise ValueError("Reviewed pipeline source SHA256 differs")
    _verify_plan(plan, fresh=True)
    output = Path(plan["output_dir"])
    output.mkdir(mode=0o700)
    base._write_json(output / "plan.json", plan)
    receipt = {
        "schema_id": SCHEMA,
        "plan_sha256": plan_sha256,
        "completed": False,
        "status": "pending",
        "model_score": None,
        "fit_eligible": False,
        "model_calls": 0,
        "episodes": [],
        "errors": [],
        "browser_closed": False,
        "sources_unchanged": False,
    }
    _write(output / "receipt.pending.json", receipt)
    browser = None
    try:
        for name, pin in plan["sources"].items():
            raw = (ROOT / name).read_bytes()
            if base._sha(raw) != pin["sha256"] or len(raw) != pin["bytes"]:
                raise ValueError("Reviewed source changed before capture")
            target = output / "sources" / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
        runtime = base._materialize_profile(plan, output)
        if (
            runtime.manifest() != plan["observation_profile_manifest"]
            or runtime.manifest_sha256 != plan["observation_profile_manifest_sha256"]
        ):
            raise ValueError("Materialized V7 manifest differs from reviewed plan")
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True, executable_path=plan["chromium"]["resolved_path"]
            )
            receipt["browser_version"] = browser.version
            for item in plan["episodes"]:
                context = browser.new_context(
                    viewport={"width": 1120, "height": 780}, service_workers="block"
                )
                transport = base.RecordingCDP(context)
                context_closed = False
                events = []
                context.on("request", lambda request: _observe_request(events, request))
                try:
                    with (
                        base._publisher_environment(),
                        base._load_runtime(plan, transport, runtime) as (agent, model),
                    ):
                        read_state = sys.modules["jev_ultrafast.browser"].READ_STATE
                        if read_state != runtime.read_state:
                            raise ValueError("V7 observer differs from reviewed manifest")
                        trace = links.collect_owned_episode(
                            spec=item["scenario"],
                            episode_id=item["episode_id"],
                            transport=transport,
                            agent_module=agent,
                            model_module=model,
                            upstream=plan["upstream_path"],
                            read_state=read_state,
                        )
                    if transport.observation_profile_errors or transport.targets or context.pages:
                        raise ValueError("Runtime or target closure failed")
                    _verify_network(item["scenario"], trace, events)
                    relative = "episodes/" + item["episode_id"]
                    summary = {
                        "episode_id": item["episode_id"],
                        "trace": relative + ".trace.json",
                        "journal": relative + ".journal.json",
                        "network": relative + ".network.json",
                    }
                    summary["trace_sha256"] = _write(output / summary["trace"], trace)
                    summary["journal_sha256"] = _write(
                        output / summary["journal"], transport.journal
                    )
                    summary["network_sha256"] = _write(output / summary["network"], events)
                    context.close()
                    context_closed = True
                    summary["context_closed"] = True
                    receipt["episodes"].append(summary)
                finally:
                    if not context_closed:
                        context.close()
            browser.close()
            browser = None
            receipt["browser_closed"] = True
        _verify_plan(plan, fresh=False)
        receipt["sources_unchanged"] = True
        receipt["completed"] = len(receipt["episodes"]) == len(plan["episodes"])
        receipt["status"] = (
            "completed_pending_admission" if receipt["completed"] else "failed_quarantined"
        )
    except Exception as exc:
        receipt["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        receipt["status"] = "failed_quarantined"
    finally:
        if browser is not None:
            try:
                browser.close()
                receipt["browser_closed"] = True
            except Exception as exc:
                receipt["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        if receipt["errors"]:
            receipt["completed"] = False
            receipt["status"] = "failed_quarantined"
        base._write_json(output / "receipt.json", receipt)
    return receipt


@dataclass(frozen=True)
class AdmittedLinkCollection:
    _rows: tuple[bytes, ...]
    receipt_sha256: str
    plan_sha256: str
    _seal: str = field(repr=False)
    _proof: object = field(repr=False, compare=False)


def _read_evidence(output, path, digest):
    value, actual = base._plain_json(base._confined_file(output, path))
    if actual != base._hash_pin(digest):
        raise ValueError("Captured episode evidence changed")
    return value


def admit_training_collection(output_dir, *, expected_receipt_sha256, expected_plan_sha256):
    """Replay all original protocol calls and native wires under explicit evidence pins."""
    output = Path(output_dir).absolute()
    plan, plan_sha = base._plain_json(base._confined_file(output, "plan.json"))
    receipt, receipt_sha = base._plain_json(base._confined_file(output, "receipt.json"))
    if plan_sha != base._hash_pin(expected_plan_sha256) or receipt_sha != base._hash_pin(
        expected_receipt_sha256
    ):
        raise ValueError("Explicit plan or receipt SHA256 differs")
    if (
        plan.get("output_dir") != str(output)
        or receipt.get("schema_id") != SCHEMA
        or receipt.get("plan_sha256") != plan_sha
        or receipt.get("status") != "completed_pending_admission"
        or receipt.get("completed") is not True
        or receipt.get("errors")
        or receipt.get("browser_closed") is not True
        or receipt.get("sources_unchanged") is not True
        or receipt.get("model_calls") != 0
        or receipt.get("model_score") is not None
        or receipt.get("fit_eligible") is not False
    ):
        raise ValueError("Incomplete or foreign collection is quarantined")
    _verify_plan(plan, fresh=False)
    for name, pin in plan["sources"].items():
        raw = base._confined_file(output, "sources/" + name).read_bytes()
        if base._sha(raw) != pin["sha256"] or len(raw) != pin["bytes"]:
            raise ValueError("Captured source changed")
    profile = base._profile_module(PROFILE)
    manifest_path = base._confined_file(output, "observation-profile/profile-manifest.json")
    manifest_sha = base._sha(manifest_path.read_bytes())
    if manifest_sha != plan["observation_profile_manifest_sha256"]:
        raise ValueError("V7 manifest SHA256 differs from the reviewed plan")
    runtime = profile.verify_materialized_profile(
        output / "observation-profile",
        expected_manifest_sha256=manifest_sha,
        expected_source_sha256=plan["observation_helper_sha256"],
        profile=PROFILE,
    )
    if runtime.manifest() != plan["observation_profile_manifest"]:
        raise ValueError("V7 runtime differs from the reviewed manifest")
    summaries = receipt.get("episodes")
    if type(summaries) is not list or len(summaries) != len(plan["episodes"]):
        raise ValueError("Every planned episode must be complete")
    encoded_rows = []
    for item, summary in zip(plan["episodes"], summaries, strict=True):
        spec, episode_id = item["scenario"], item["episode_id"]
        prefix = "episodes/" + episode_id
        if (
            summary.get("episode_id") != episode_id
            or summary.get("context_closed") is not True
            or any(
                summary.get(key) != prefix + suffix
                for key, suffix in (
                    ("trace", ".trace.json"),
                    ("journal", ".journal.json"),
                    ("network", ".network.json"),
                )
            )
        ):
            raise ValueError("Episode identity or context closure changed")
        trace = _read_evidence(output, summary["trace"], summary["trace_sha256"])
        journal = _read_evidence(output, summary["journal"], summary["journal_sha256"])
        events = _read_evidence(output, summary["network"], summary["network_sha256"])
        if (
            type(journal) is not list
            or any(call.get("sequence") != index for index, call in enumerate(journal))
            or trace.get("scenario") != spec
            or trace.get("episode_id") != episode_id
            or trace.get("schema_id") != links.SCHEMA
            or trace.get("scenario_profile") != links.DEFAULT_SCENARIO_PROFILE
            or trace.get("fit_eligible") is not False
            or trace.get("model_score") is not None
            or trace.get("status") != "completed"
            or trace.get("completed") is not True
            or trace.get("errors")
            or trace.get("agent_closed") is not True
            or trace.get("cdp_start") != 0
            or trace.get("cdp_end") != len(journal)
        ):
            raise ValueError("Incomplete, unowned or foreign native episode")
        settling = trace.get("observation_settling", {})
        if settling.get("fault") is not False or not settling.get("records"):
            raise ValueError("Settled V7 observation evidence is missing or faulted")
        origin = _verify_network(spec, trace, events)
        links._verify_ownership(trace, journal, origin, runtime.read_state)
        steps, rows = trace["steps"], trace["rows"]
        if not steps or len(steps) != len(rows) or len(steps) > links.MAX_TICKS:
            raise ValueError("Incomplete native row/step population")
        previous = None
        for index, (step, row) in enumerate(zip(steps, rows, strict=True)):
            if (
                step["index"] != index
                or step["scenario"] != spec
                or (previous is None and step["history_before"])
                or (
                    previous is not None
                    and (
                        step["before"] != previous["after"]
                        or step["history_before"] != previous["history_after"]
                    )
                )
            ):
                raise ValueError("Native episode state or history is discontinuous")
            links._verify_document_urls(step, origin)
            proof = links._verify_step(step, journal, plan["upstream_path"], runtime.read_state)
            if proof != step.get("proof"):
                raise ValueError("Native transition or executor proof changed")
            rebuilt = links._verified_training_row(
                step["payload"],
                row_id=episode_id + f".step-{index:02d}",
                scenario_id=spec["id"],
                ancestor_id=spec["ancestor_id"],
                episode_id=episode_id,
                step=index,
                operation=step["teacher"]["operation"],
                target=step["target"],
                proof_refs={"transition": episode_id + f"#step-{index}"},
            )
            if base._encode(row) != base._encode(rebuilt):
                raise ValueError("Native wire, offered target, loss mask or row identity changed")
            encoded_rows.append(base._encode(row))
            previous = step
        if steps[-1]["teacher"]["operation"] != "DONE" or not scenarios.verify_goal(
            spec, steps[-1]["after"]
        ):
            raise ValueError("Verified completion is required before DONE")
    seal = base._digest(
        {"rows": [blob.decode() for blob in encoded_rows], "receipt": receipt_sha, "plan": plan_sha}
    )
    return AdmittedLinkCollection(
        tuple(encoded_rows), receipt_sha, plan_sha, seal, _ADMISSION_PROOF
    )


def compile_wire_projection(admitted, *, model_alias=None):
    if type(admitted) is not AdmittedLinkCollection or admitted._proof is not _ADMISSION_PROOF:
        raise ValueError("Independent admission is required")
    if model_alias not in (None, base.MODEL_ALIAS):
        raise ValueError("Original native model alias cannot be rewritten")
    if admitted._seal != base._digest(
        {
            "rows": [row.decode() for row in admitted._rows],
            "receipt": admitted.receipt_sha256,
            "plan": admitted.plan_sha256,
        }
    ):
        raise ValueError("Admitted row bytes or identity changed")
    result = []
    for blob in admitted._rows:
        row = json.loads(blob)
        if row["schema_id"] != base.ROW_SCHEMA or row["split"] != "train":
            raise ValueError("Foreign or nontraining row")
        result.append(
            {
                "case_id": row["id"],
                "payload": row["payload"],
                "teacher_choice": row["teacher_choice"],
                "valid_next_targets": row["valid_next_targets"],
                "loss_mask": row["loss_mask"],
                "wire_sha256": row["wire_sha256"],
                "fit_eligible": False,
            }
        )
    return result
