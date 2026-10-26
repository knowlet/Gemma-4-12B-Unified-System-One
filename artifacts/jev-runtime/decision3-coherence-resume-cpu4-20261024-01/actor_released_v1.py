"""Held coherence-only continuation using original owner07/native03/full06.

No model, scorer, input or process IO is authorized until root releases the
source with an independently accepted input-manifest SHA. Cached input data is
authenticated transitively; it is not claimed as controller static-source
coverage. The failed full06 invocation is never rewritten or qualified here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import time
from pathlib import Path

SOURCE_ONLY = False
ACCEPTED_INPUT_SHA256 = '6176290eabee29a33d996ffeebdd6e1ece9345e605292864d8b391ffeee6ca6c'
ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = ROOT / "artifacts/jev-runtime/decision3-coherence-resume-cpu4-20261024-01"
FULL06 = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-both-20261024-06"
NATIVE03 = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261024-03"
OWNER07 = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-07"
CAMPAIGN_SHA = "81113f1e2e1f1342c547b156e307ef78844c5947a9fa47f85fb054579eb4d966"
NATIVE_SHA = "4a08be7703d8559f5dc86593f0e3138e9c9546ea1cc4613a951ed796a13d9d3f"
CONTROLLER_SHA = "685a9f809f429f4e9658083dc8530b563740625f9ba7816908fdc58002de256d"
ORIGINAL_PLAN_SHA = "351500bcdfdd0b1c3c6096b21bd767321e761a0af79a64ed080dad518d9d575c"
BENCHMARK_SHA = "b5caa794d864917a637034bd1cebe0487c9eec0e5c3c423835d24fca4f397021"
BACKEND = "decision3-edge-gx10-publisher-cpu2g-coexist-batch4-both-stage-completions-ece-schema-20261024-v1"
PROFILE = "decision3-edge-original-cpu4-coherence-cross-invocation-resume-v1"
INPUT_SCHEMA = "root-accepted-decision3-cpu4-coherence-resume-inputs-v1"
ACCEPTED = "ROOT_ACCEPTED_ORIGINAL_CPU4_CACHE_PREFIX"
MAX_BYTES, TOTAL, WALL_SECONDS = 1024**2, 1248, 10000
SETTINGS = {
    "suite": "jevbench-mini",
    "seed": 0,
    "repeats": 1,
    "trials": 1,
    "concurrency": 1,
    "diagnostics": False,
    "tolerance": 0.05,
    "min_n": 5,
    "bootstrap": 1000,
}
SCORE_SETTINGS = {"tolerance": 0.05, "min_n": 5, "bootstrap": 1000, "min_coverage": 0.95}
OUTCOME_KEYS = (
    "jevbench_version",
    "backend",
    "seed",
    "repeats",
    "trials",
    "noise_multiplier",
    "noise_floor",
    "relations",
    "cases",
    "planned_tests",
    "truncated_by_budget",
    "backend_down",
    "bases",
    "outcomes",
    "errors",
    "suite",
    "case_domains",
    "selection",
)


class ResumeAbort(BaseException):
    """Input/admission/native failures must escape scorer Exception handlers."""


def require(value, message):
    if not value:
        raise ResumeAbort(message)


def is_sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def released():
    require(SOURCE_ONLY is False, "SOURCE_ONLY: coherence resume is held before IO/imports")
    require(is_sha(ACCEPTED_INPUT_SHA256), "root-accepted input SHA remains pending")


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate input JSON key")
            value[key] = item
        return value

    return json.loads(
        raw,
        object_pairs_hook=unique,
        parse_constant=lambda _value: require(False, "nonfinite input JSON"),
    )


def ordinary(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_relative_to(ROOT) and ".." not in path.parts, "repository input path only")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "ordinary input paths only")
    return path


def read_small(path):
    path = ordinary(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_BYTES, "small regular input"
        )
        raw = stream.read(MAX_BYTES + 1)
        after = os.fstat(stream.fileno())
    require(
        len(raw) == before.st_size
        and all(
            getattr(before, k) == getattr(after, k) == getattr(path.stat(), k)
            for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ),
        "input changed during authenticated read",
    )
    return raw


def read_ref(record):
    require(
        type(record) is dict
        and set(record) == {"path", "sha256", "size_bytes"}
        and is_sha(record["sha256"])
        and type(record["size_bytes"]) is int
        and 0 < record["size_bytes"] <= MAX_BYTES,
        "exact bounded input reference",
    )
    raw = read_small(record["path"])
    require(
        len(raw) == record["size_bytes"] and digest(raw) == record["sha256"], "input pin differs"
    )
    return raw


def reference(path):
    raw = read_small(path)
    return {"path": str(ordinary(path)), "sha256": digest(raw), "size_bytes": len(raw)}


def import_selected(path, sha, name):
    raw = read_small(path)
    require(Path(path).suffix == ".py" and digest(raw) == sha, "original source pin differs")
    require(name not in sys.modules, "fresh selected module required")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), vars(module))
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def request_identity(state, questions):
    payload = compact({"state": state, "questions": questions})
    key = digest(compact({"b": BACKEND, "s": state, "q": questions, "r": 0}))
    return key, digest(payload)


def probability_sha(answers):
    require(type(answers) is dict and answers, "named original answers required")
    distributions = {}
    for key, answer in answers.items():
        require(type(answer) is dict, "native answer object required")
        if "probabilities" in answer:
            distributions[key] = answer["probabilities"]
        else:
            require("noul" in answer, "native probabilities/noul required")
            distributions[key] = answer["noul"]
    return digest(compact(distributions))


def validate_manifest(manifest, requests):
    require(
        manifest.get("schema") == INPUT_SCHEMA
        and manifest.get("status") == ACCEPTED
        and manifest.get("original_backend_identity") == BACKEND
        and manifest.get("settings") == SETTINGS,
        "explicit accepted original backend/settings required",
    )
    count = manifest.get("accepted_prior_count")
    require(
        type(count) is int and 0 < count < TOTAL, "audited cache count, never a progress estimate"
    )
    require(
        requests.get("schema") == "decision3-original-coherence1248-request-manifest-v1"
        and type(requests.get("requests")) is list
        and len(requests["requests"]) == TOTAL,
        "all original1248 request bindings required",
    )
    rows = requests["requests"]
    require(
        all(
            type(row) is dict
            and set(row) == {"request_fingerprint", "payload_sha256"}
            and is_sha(row["request_fingerprint"])
            and is_sha(row["payload_sha256"])
            for row in rows
        )
        and len({row["request_fingerprint"] for row in rows}) == TOTAL,
        "unique ordered original request/payload hashes required",
    )
    require(
        type(manifest.get("accepted_entry_shards")) is list
        and 1 <= len(manifest["accepted_entry_shards"]) <= 32,
        "bounded root-accepted cache shards required",
    )
    require(
        ordinary(manifest["original_full06_plan_ref"]["path"]) == FULL06 / "edge-plan.json"
        and manifest["original_full06_plan_ref"]["sha256"] == ORIGINAL_PLAN_SHA
        and manifest["benchmark_source_plan_ref"]["sha256"] == BENCHMARK_SHA,
        "unchanged full06/native03/benchmark source plans required",
    )
    return rows


def authenticated_prefix(manifest, rows, cache=None):
    """Stream verified prefix data; keep raw answers out of resident metadata."""
    audit = decode(read_ref(manifest["root_prefix_audit_ref"]))
    require(
        audit.get("schema") == "root-original-cpu4-cache-prefix-audit-v1"
        and audit.get("status") == "ROOT_ACCEPTED_CACHE_PREFIX_ONLY"
        and audit.get("accepted_prior_count") == manifest["accepted_prior_count"]
        and audit.get("planned_requests") == TOTAL
        and audit.get("missing_requests") == TOTAL - manifest["accepted_prior_count"]
        and audit.get("original_scope_closure_positive") is True
        and audit.get("original_campaign_success") is False
        and audit.get("original_model_qualified") is False,
        "separate accepted prefix audit, without repairing original failure",
    )
    for key in (
        "request_manifest_ref",
        "accepted_entry_shards",
        "prior_closure_ref",
        "prior_source_union_ref",
        "original_full06_plan_ref",
        "benchmark_source_plan_ref",
    ):
        require(audit.get(key) == manifest[key], "accepted prefix audit input binding differs")
    closure = decode(read_ref(manifest["prior_closure_ref"]))
    require(
        closure["selection"]["unit"] == "jev-d3-llama-cpp-cpu4-2g-coexist-20261024-07.service"
        and closure["selection"]["memory_max_bytes"] == 2 * 1024**3
        and closure["closure_mode"] == "authenticated_stop"
        and not closure["closure_errors"]
        and all(
            closure[k] is True
            for k in (
                "whole_unit_exit_confirmed",
                "observed_cgroup_cohort_exit_confirmed",
                "post_stop_terminal_verified",
                "cgroup_absent",
                "sources_unchanged",
            )
        )
        and len(closure["two_process_scans"]) == 2
        and all(not s["matches"] and not s["unknown"] for s in closure["two_process_scans"]),
        "original positive CPU4 unit/cohort/source closure required",
    )
    union = decode(read_ref(manifest["prior_source_union_ref"]))
    require(
        union.get("schema") == "original-cpu4-selected-source-union-v1"
        and type(union.get("sources")) is list
        and len(union["sources"]) == 59,
        "complete original54 selected plus5 base source union required",
    )
    original_cfg = decode(read_ref(union["sealed_config_ref"]))
    require(
        original_cfg["sources"] == union["sources"]
        and original_cfg["profile"] == "llama-cpp-cpu4-2g-coexist-command-owned-v1"
        and original_cfg["nonce"] == closure["selection"]["nonce"]
        and original_cfg["boot_id"] == closure["selection"]["boot_id"],
        "original sealed config/source union/closure binding required",
    )
    require(len({str(ordinary(r["path"])) for r in union["sources"]}) == 59, "unique prior sources")
    for record in union["sources"]:
        read_ref(record)
    seen = []
    for shard_ref in manifest["accepted_entry_shards"]:
        shard = decode(read_ref(shard_ref))
        require(
            shard.get("schema") == "root-accepted-decision3-coherence-cache-shard-v1"
            and type(shard.get("entries")) is list
            and shard["entries"],
            "accepted cache shard",
        )
        for entry in shard["entries"]:
            index = len(seen)
            require(index < manifest["accepted_prior_count"], "no extra accepted cache entry")
            require(
                entry["request_fingerprint"] == rows[index]["request_fingerprint"]
                and entry["payload_sha256"] == rows[index]["payload_sha256"]
                and entry["original_call_index"] == 231 + index,
                "exact original coherence prefix/callback binding required",
            )
            key = entry["request_fingerprint"]
            source = ordinary(entry["cache_ref"]["path"])
            previous = ordinary(original_cfg["output"]) / "edge-both"
            require(
                source.name == key + ".json" and source.parent == previous / "coherence-cache",
                "original fingerprint-named cache entry required",
            )
            raw = read_ref(entry["cache_ref"])
            answers = decode(raw)
            returned = decode(read_ref(entry["raw_return_ref"]))
            validated = decode(read_ref(entry["validated_ref"]))
            prepared = decode(read_ref(entry["prepared_ref"]))
            for field, suffix in (
                ("raw_return_ref", "raw_returned_before_validation"),
                ("validated_ref", "validated"),
                ("prepared_ref", "prepared"),
            ):
                require(
                    ordinary(entry[field]["path"])
                    == previous
                    / "native-evidence"
                    / f"{entry['original_call_index']:05d}-{suffix}.json",
                    "exact original native callback evidence path required",
                )
            lengths = prepared["value"]["lengths"]
            require(
                returned["index"]
                == validated["index"]
                == prepared["index"]
                == entry["original_call_index"]
                and returned["stage"] == validated["stage"] == prepared["stage"] == "jevbench-mini"
                and returned["event"] == "raw_returned_before_validation"
                and prepared["event"] == "prepared"
                and type(lengths) is dict
                and lengths
                and all(type(n) is int and 0 < n <= 16384 for n in lengths.values())
                and list(answers) == prepared["value"]["original_question_order"]
                and set(answers) == set(lengths)
                and returned["value"]["model"] == "d3-edge"
                and returned["value"]["usage"]
                == {"input_tokens": sum(lengths.values()), "output_tokens": 0}
                and validated["status"] == "validated_pending_durable_evidence"
                and compact(returned["value"]["answers"]) == compact(answers)
                and digest(compact(answers)) == entry["answers_sha256"]
                and probability_sha(answers) == entry["probabilities_sha256"],
                "accepted cache/native return/probability/validated binding differs",
            )
            if cache is not None:
                destination = cache / (key + ".json")
                with destination.open("xb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
            seen.append(key)
    require(
        len(seen) == manifest["accepted_prior_count"], "exact audited accepted cache count required"
    )
    return union["sources"]


def owned_selection(controller, batch, helper, owner_output, manifest_path, manifest_sha):
    """Reuse the original owner07 selection and current admitted wrapper binding."""
    cfg_raw = read_small(owner_output / "sealed-config.json")
    cfg = decode(cfg_raw)
    admission = decode(read_small(owner_output / "admission.json"))
    controller.sources_match(cfg["sources"])
    require(
        cfg["profile"] == controller.PROFILE
        and cfg["unit"] == batch.UNIT
        and cfg["output"] == str(owner_output)
        and cfg["runtime_seconds"] == 10800
        and cfg["nonce"] == os.environ.get("JEV_COMMAND_NONCE") == admission["nonce"]
        and admission["config_sha256"] == digest(cfg_raw),
        "original owner07 admission/config binding",
    )
    pins = {str(ordinary(r["path"])): r["sha256"] for r in cfg["sources"]}
    require(
        pins.get(str(Path(__file__).absolute())) == digest(read_small(__file__))
        and pins.get(str(manifest_path)) == manifest_sha,
        "actor/input manifest must be selected sources",
    )
    boundary = controller.load_fixed("boundary")
    current = controller.process_identity(boundary, os.getpid())
    require(
        all(
            current["process"][k] == admission["actor"][k] for k in ("pgid", "sid", "uid", "cgroup")
        ),
        "same admitted CPU4 wrapper session/cgroup",
    )
    spec = controller.command_selection(
        cfg["selection"], cfg["selection_sha256"], cfg["base_sources"]
    )
    require(
        any(
            [arg.replace("{OUTPUT}", str(owner_output)) for arg in row["argv"]] == current["argv"]
            and row["cwd"].replace("{OUTPUT}", str(owner_output)) == current["cwd"]
            for row in spec["commands"]
        ),
        "actual worker argv must equal original selected command",
    )
    identity = helper.process_identity()
    batch.validate_owner(identity, helper.owned_limits(identity), helper.START_ENV)
    return identity


def populations(report, requests):
    result = report.result
    require(report.settings == SCORE_SETTINGS, "original official score settings required")
    require(
        result["requests"] == requests
        and not result["truncated_by_budget"]
        and not result["backend_down"]
        and len(result["cases"]) == 240
        and len(result["case_domains"]) == 240
        and len(result["relations"]) == 50
        and len(result["planned_tests"]) == 50
        and sum(result["planned_tests"].values()) == 1200,
        "complete original official coherence populations required",
    )


def execute(manifest_path, manifest_sha, owner_output):
    released()
    began = time.monotonic()
    require(
        manifest_sha == ACCEPTED_INPUT_SHA256 and digest(read_small(manifest_path)) == manifest_sha,
        "root-selected accepted input manifest SHA required",
    )
    require(
        sys.flags.no_user_site
        and sys.dont_write_bytecode
        and not {
            "torch",
            "transformers",
            "numpy",
            "safetensors",
            "d3_runtime",
            "jevbench",
        }.intersection(sys.modules),
        "fresh -B -s process without framework or scorer imports required",
    )
    manifest_path, owner_output = ordinary(manifest_path), ordinary(owner_output)
    require(
        owner_output.parent == OWNER07 and re.fullmatch(r"actual-[0-9]{2}", owner_output.name),
        "fresh original owner07 output namespace required",
    )
    manifest = decode(read_small(manifest_path))
    requests = decode(read_ref(manifest["request_manifest_ref"]))
    rows = validate_manifest(manifest, requests)
    read_ref(manifest["original_full06_plan_ref"])
    read_ref(manifest["benchmark_source_plan_ref"])
    campaign = import_selected(FULL06 / "campaign.py", CAMPAIGN_SHA, "_resume_original_full06")
    original = campaign.load_plan(manifest["original_full06_plan_ref"]["path"], ORIGINAL_PLAN_SHA)
    batch = import_selected(NATIVE03 / "actor.py", NATIVE_SHA, "_resume_original_native03")
    helper = batch.load_helper()
    native = batch.load_plan(helper, NATIVE03 / "edge-plan.json", campaign.NATIVE_PLAN_SHA["edge"])
    require(
        original["profile"] == BACKEND and original["model"] == native["model"],
        "original full06/backend/native03 model binding required",
    )
    controller = import_selected(
        OWNER07 / "controller.py", CONTROLLER_SHA, "_resume_original_owner07"
    )
    identity = owned_selection(controller, batch, helper, owner_output, manifest_path, manifest_sha)
    helper.host_gate()
    output = owner_output / "coherence-resume"
    require(not output.exists(), "new resume output required; never mutate original full06")
    os.umask(0o077)
    output.mkdir(mode=0o700)
    cache = output / "coherence-cache"
    cache.mkdir(mode=0o700)
    state = {
        "profile": PROFILE,
        "stage": "authenticating_accepted_prefix",
        "model_loads": 0,
        "cross_invocation_resume": True,
        "original_full06_remains_failed": True,
        "manifest_ref": reference(manifest_path),
        "accepted_prior_count": manifest["accepted_prior_count"],
        "new_completed_count": 0,
        "official_score": None,
        "external_original_owner07_closure_required": True,
    }
    primary = None
    try:
        remaining = WALL_SECONDS - (time.monotonic() - began)
        require(remaining > 0, "resume preflight exhausted separate whole10000 deadline")
        with campaign.DeadlineScope(helper.Deadline(remaining, output, "resume-campaign")):
            prior_sources = authenticated_prefix(manifest, rows, cache)
            for record in prior_sources:
                if ordinary(record["path"]) == FULL06 / "root-edge-both-command-spec-01.json":
                    old_spec = decode(read_ref(record))
                    break
            else:
                raise ResumeAbort("original command-spec provenance missing")
            argv = old_spec["commands"][0]["argv"]

            def option(name):
                return argv[argv.index(name) + 1]

            campaign.check_smoke(
                option("--smoke-receipt"), option("--smoke-receipt-sha256"), native
            )
            campaign.check_audit(
                option("--smoke-audit"),
                option("--smoke-audit-sha256"),
                option("--smoke-receipt"),
                option("--smoke-receipt-sha256"),
                native,
            )
            benchmark = decode(read_ref(manifest["benchmark_source_plan_ref"]))
            for record in benchmark["source_files"] + benchmark["existing_dataset_metadata"]:
                helper.check_record(record, full=True)
            helper.write_json(output / "state.json", state)
            runtime, torch = batch.load_runtime(helper, native, output, state)
            require(state["model_loads"] == 1, "one original CPU4 constructor required")
            backend = batch.NativeBatch4Backend(
                runtime, torch, native["model"], context_limit=16384
            )
            predictor = campaign.TimedPredictor(
                backend, helper, output / "native-evidence", lambda: batch.own_counters(identity)
            )
            predictor.stage = "jevbench-mini"
            missing = {
                r["request_fingerprint"]: r["payload_sha256"]
                for r in rows[manifest["accepted_prior_count"] :]
            }
            completed = set()

            def infer_missing(state, questions):
                key, payload_sha = request_identity(state, questions)
                require(
                    key in missing and key not in completed and payload_sha == missing[key],
                    "only exact uncached original request may invoke native CPU4",
                )
                try:
                    result = predictor.answers(state, questions)
                except BaseException as error:
                    raise ResumeAbort(
                        "original native missing callback failed; no complete resume"
                    ) from error
                completed.add(key)
                return result

            coherence = ordinary(benchmark["coherence"]["root"])
            sys.path.insert(0, str(coherence / "src"))
            import jevbench as jb

            require(
                Path(jb.__file__).resolve() == coherence / "src/jevbench/__init__.py",
                "pinned official scorer origin",
            )
            options = {**SETTINGS, "minimize": False}
            filled = jb.evaluate(
                jb.from_callable(infer_missing, name=BACKEND), cache=str(cache), **options
            )
            require(
                completed == set(missing)
                and predictor.started
                == predictor.completed
                == predictor.raw_returned
                == len(missing),
                "exact missing native callback population required",
            )
            populations(filled, len(missing))
            require(
                {p.name for p in cache.iterdir()}
                == {r["request_fingerprint"] + ".json" for r in rows},
                "complete exact1248 cache inventory required",
            )
            cache_hashes = {
                r["request_fingerprint"]: reference(cache / (r["request_fingerprint"] + ".json"))
                for r in rows
            }

            def never_infer(_state, _questions):
                raise ResumeAbort("complete-cache official recomputation must make no model call")

            final = jb.evaluate(
                jb.from_callable(never_infer, name=BACKEND), cache=str(cache), **options
            )
            populations(final, 0)
            require(
                all(compact(filled.result[k]) == compact(final.result[k]) for k in OUTCOME_KEYS),
                "official complete-cache recomputation changed original outcomes",
            )
            require(
                compact(filled.scores) == compact(final.scores), "official recomputed scores differ"
            )
            for record in cache_hashes.values():
                read_ref(record)
            authenticated_prefix(manifest, rows)
            require(digest(read_small(manifest_path)) == manifest_sha, "accepted manifest changed")
            for record in benchmark["source_files"] + benchmark["existing_dataset_metadata"]:
                helper.check_record(record, full=True)
            batch.validate_runtime(runtime, torch, native["model"])
            batch.authenticate(helper, native)
            owned_selection(controller, batch, helper, owner_output, manifest_path, manifest_sha)
            helper.ensure_elapsed(began, WALL_SECONDS, "complete resumed coherence authentication")
            final.save(output / "coherence-mini.json")
            helper.write_json(
                output / "cache-inventory.json",
                {"requests": [cache_hashes[r["request_fingerprint"]] for r in rows]},
            )
            state.update(
                stage="complete_pending_original_owner07_external_closure",
                new_completed_count=len(completed),
                request_population=TOTAL,
                official_score=final.overall,
                official_interval=final.interval,
                final_full_native_and_input_authentication=True,
                cuda_initialized=torch.cuda.is_initialized(),
                new_latency=predictor.summary(),
                fresh_full06_success=False,
                heaven_accuracy=None,
                browser_success=None,
            )
            helper.write_json(output / "state.json", state)
            helper.ensure_elapsed(began, WALL_SECONDS, "durable resumed coherence result")
    except BaseException as error:
        primary = error
        state.update(
            stage="failed_no_complete_resume_claim",
            official_score=None,
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        if primary is not None:
            helper.finalize_state(output, state, primary)
    return 0


def main(argv=None):
    released()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--inputs-sha256")
    parser.add_argument("--owner-output", type=Path)
    args = parser.parse_args(argv)
    require(
        args.execute and args.inputs and args.owner_output, "explicit original-owned resume only"
    )
    return execute(args.inputs, args.inputs_sha256, args.owner_output)


if __name__ == "__main__":
    raise SystemExit(main())
