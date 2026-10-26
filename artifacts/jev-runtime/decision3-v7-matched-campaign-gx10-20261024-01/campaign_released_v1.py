"""Held original-V7 matched campaign; original NF4 driver owns the model/helper.

The selected external pump owns CPU controller processes. This actor only runs
the unchanged V7 clients and exchanges bounded, private fixed-file handoffs.
Task success and complete infrastructure are separate. No training is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

SOURCE_ONLY = False
NATIVE_SOURCE_SHA256 = 'f4cfdbb1fc33fa03fa07c1a871b43107a241de5da9a865414bc30e5163a57dee'
CALLER_SOURCE_SHA256 = '0e8f0196b7f3fb87a52fa649b87894c957adc55d2600a83c05344e024e8708da'
SERIAL_SOURCE_SHA256 = '17d58567652a9b2de0fdc6f5e41acef63bf344e233545444953b4595b32c64e6'
CAPABILITY_SHA256 = '4dbbb9470d011931854f6473c0840ef7f0746c4194c719cde6ab454c3e79d463'
ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = Path(__file__).absolute().parent
PROFILE = "jev-nf4-d3-matched-v7-owned-v1"
SCHEMA = "jev-nf4-d3-matched-v7-campaign-v1"
ALIAS = "d3-edge-dsl01-operation-delta-cpu5-candidate"
OWNER_MODULE = "_selected_matched_nf4_d3_outer_owner"
KNOWN4 = (
    "public-arxiv-title-filter-results-v1",
    "public-arxiv-original-paper-open-v1",
    "public-python-counter-most-common-v1",
    "public-nasa-mars-facts-moons-v1",
)
PUBLIC6 = tuple(f"heldout-public-{index:03d}" for index in range(6))
CASE_IDS = KNOWN4 + PUBLIC6


def require(value, message):
    if not value:
        raise RuntimeError(message)


def released():
    require(SOURCE_ONLY is False, "SOURCE_ONLY: campaign held before arguments/IO")
    require(
        all(
            type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in (
                NATIVE_SOURCE_SHA256,
                CALLER_SOURCE_SHA256,
                SERIAL_SOURCE_SHA256,
                CAPABILITY_SHA256,
            )
        ),
        "Native/caller/serial/capability source pins remain pending",
    )


def ordinary(path):
    path = Path(path).absolute()
    require(
        path.is_relative_to(ROOT) and ".." not in path.parts,
        "Fixed ordinary repository path required",
    )
    require(not any(p.is_symlink() for p in (path, *path.parents)), "No linked campaign path")
    return path


def raw(path, maximum=16 * 1024**2):
    path = ordinary(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        blob = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(
        0 < len(blob) == before.st_size <= maximum
        and all(
            getattr(before, key) == getattr(after, key) == getattr(path.stat(), key) for key in keys
        ),
        "Bounded unchanged campaign bytes required",
    )
    return blob


def decode(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate campaign JSON key")
            result[key] = value
        return result

    return json.loads(
        blob,
        object_pairs_hook=pairs,
        parse_constant=lambda _value: require(False, "Nonfinite campaign JSON"),
    )


def reference(path):
    blob = raw(path)
    return {
        "path": str(Path(path).absolute()),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_bytes": len(blob),
    }


def authenticated(record):
    require(
        type(record) is dict and set(record) == {"path", "sha256", "size_bytes"},
        "Exact campaign reference required",
    )
    path = ordinary(ROOT / record["path"])
    blob = raw(path)
    require(
        len(blob) == record["size_bytes"] and hashlib.sha256(blob).hexdigest() == record["sha256"],
        "Selected campaign evidence changed",
    )
    return blob


def write(path, value):
    path = ordinary(path)
    blob = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        + b"\n"
    )
    require(len(blob) <= 16 * 1024**2, "Bounded campaign publication")
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".pending")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    primary = None
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        # Exclusive publication: observers see the complete durable object, or
        # no object. Existing handoffs are never replaced or partially exposed.
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except BaseException as error:
            if primary is None:
                raise
            if primary is not error:
                primary.add_note("Pending campaign publication cleanup also failed: " + repr(error))
    return reference(path)


def import_source(name, record, expected):
    require(
        name not in sys.modules and record["sha256"] == expected, "Fresh selected source required"
    )
    path = ordinary(ROOT / record["path"])
    blob = authenticated(record)
    require(path.suffix == ".py" and len(blob) <= 1024**2, "Small Python source required")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Source loader required")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(blob, str(path), "exec", dont_inherit=True), vars(module))
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def process_identity():
    pid = os.getpid()
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text[text.rfind(")") + 2 :].split()
    return {"pid": pid, "pgid": os.getpgrp(), "sid": os.getsid(0), "start_ticks": int(fields[19])}


def wait_json(path, deadline, *, seconds):
    end = time.monotonic() + seconds
    while True:
        deadline()
        if path.exists():
            return decode(raw(path))
        require(time.monotonic() < end, "Selected campaign handoff deadline exceeded")
        time.sleep(0.05)


def case_binding(value, *, schema, root_sha, index, case_id):
    require(
        type(value) is dict
        and value.get("schema") == schema
        and value.get("root_plan_sha256") == root_sha
        and value.get("case_index") == index
        and value.get("case_id") == case_id,
        "Current selected case handoff binding differs",
    )


def combined_summary(summaries):
    require(
        summaries and all(item["attempts"] == 1 for item in summaries),
        "One complete original outcome per case required",
    )
    outcomes = [item["outcomes"][0] for item in summaries]
    result = {
        name: sum(item.get(name, 0) for item in summaries)
        for name in (
            "attempts",
            "successes",
            "goals_satisfied",
            "premature_done_count",
            "policy_calls",
            "text_calls",
        )
    }
    result.update(
        outcomes=outcomes,
        success_rate=result["successes"] / result["attempts"],
        observation_profile_failed=any(
            item.get("observation_profile_failed", False) for item in summaries
        ),
        scope="Original unchanged single-case summaries; fresh browser per case in both arms",
    )
    return result


def retain_finalization_failure(output, primary):
    """Keep a late failure beside an immutable earlier receipt; never hide it."""
    try:
        path = output / "receipt.json"
        write(
            output / "finalization-failure.json",
            {
                "schema": "jev-v7-matched-finalization-failure-v1",
                "stage": "failed_finalization_no_matched_score",
                "error": {"type": type(primary).__name__, "message": str(primary)},
                "earlier_receipt_ref": reference(path) if path.exists() else None,
                "complete_matched_score_authorized": False,
                "external_owner_closure_required": True,
            },
        )
    except BaseException as secondary:
        if secondary is not primary:
            primary.add_note(
                "Campaign failure finalization evidence also failed: " + repr(secondary)
            )


class MatchedCases:
    def __init__(
        self, *, owner, plan, root_sha, directory, output, caller, lease, verify_owned, deadline
    ):
        self.owner, self.plan, self.root_sha = owner, plan, root_sha
        self.directory, self.output, self.caller = directory, output, caller
        self.lease, self.verify_owned, self.deadline = lease, verify_owned, deadline
        self.records = {"baseline": [], "candidate": []}
        self.children, self.baseline_summaries = [], []
        self.helper_endpoint = None

    def _call(self, case_id, output, policy_endpoint, policy_model, helper_endpoint, token):
        self.verify_owned()
        record = self.caller.call_case(
            case_id,
            output,
            policy_endpoint=policy_endpoint,
            policy_model=policy_model,
            text_endpoint=helper_endpoint,
            text_model="generated-actions-nf4",
            upstream=ROOT / self.plan["upstream"],
            chromium=self.plan["chromium"],
            capability_sha256=CAPABILITY_SHA256,
            verify_owned=self.verify_owned,
            policy_api_key=token if policy_model == "generated-actions-nf4" else None,
            text_api_key=token,
        )
        self.verify_owned()
        return record

    def run(self, native_output, upstream, endpoint, token, chromium, *, completed):
        require(
            str(upstream) == str(ROOT / self.plan["upstream"])
            and chromium == self.plan["chromium"],
            "Original native callback inputs differ",
        )
        self.helper_endpoint = endpoint
        for group, case_ids in (("known4", KNOWN4), ("public6", PUBLIC6)):
            summaries = []
            for case_id in case_ids:
                record = self._call(
                    case_id,
                    self.output / "baseline" / case_id,
                    endpoint + "/v1/systemone",
                    "generated-actions-nf4",
                    endpoint,
                    token,
                )
                self.records["baseline"].append(record)
                summaries.append(decode(authenticated(record["summary_ref"])))
                write(
                    self.output
                    / ("baseline-case-" + str(len(self.records["baseline"])).zfill(2) + ".json"),
                    record,
                )
            summary = combined_summary(summaries)
            self.baseline_summaries.extend(summaries)
            completed(
                {
                    "group": group,
                    "denominator": len(case_ids),
                    "summary": summary,
                    "single_case_summary_refs": [
                        record["summary_ref"]
                        for record in self.records["baseline"]
                        if record["group"] == group
                    ],
                }
            )
        require(len(self.records["baseline"]) == 10, "Complete baseline must precede candidate")
        handoff = ordinary(self.plan["matched"]["handoff_root"])
        for index, case_id in enumerate(CASE_IDS, 1):
            self.verify_owned()
            case = ordinary(handoff / f"case-{index:02d}")
            require(case.is_dir(), "External selected pump must create case handoff directory")
            request = {
                "schema": "jev-v7-matched-case-request-v1",
                "root_plan_sha256": self.root_sha,
                "case_index": index,
                "case_id": case_id,
                "group": "known4" if index <= 4 else "public6",
                "parent_identity": process_identity(),
            }
            write(case / "request.json", request)
            ready = wait_json(case / "ready.json", self.deadline, seconds=120)
            case_binding(
                ready,
                schema="jev-v7-matched-case-ready-v1",
                root_sha=self.root_sha,
                index=index,
                case_id=case_id,
            )
            require(
                ready["model"] == ALIAS
                and ready["endpoint"].startswith("http://127.0.0.1:")
                and ready["endpoint"].endswith("/v1/systemone"),
                "Owned candidate endpoint required",
            )
            record, primary = None, None
            try:
                record = self._call(
                    case_id,
                    self.output / "candidate" / case_id,
                    ready["endpoint"],
                    ALIAS,
                    endpoint,
                    token,
                )
                self.records["candidate"].append(record)
            except BaseException as error:
                primary = error
                record = getattr(error, "jev_v7_case_record", None)
                if record is not None:
                    self.records["candidate"].append(record)
            finally:
                try:
                    write(
                        case / "stop.json",
                        {
                            "schema": "jev-v7-matched-case-stop-v1",
                            "root_plan_sha256": self.root_sha,
                            "case_index": index,
                            "case_id": case_id,
                            "plan_sha256": ready["plan_sha256"],
                            "nonce": ready["nonce"],
                            "caller_record_ref": None
                            if record is None
                            else record.get("receipt_ref"),
                        },
                    )
                    closed = wait_json(case / "closed.json", self.deadline, seconds=90)
                    case_binding(
                        closed,
                        schema="jev-v7-matched-case-closed-v1",
                        root_sha=self.root_sha,
                        index=index,
                        case_id=case_id,
                    )
                    require(
                        closed["plan_sha256"] == ready["plan_sha256"]
                        and closed["nonce"] == ready["nonce"]
                        and closed["owner_output"] == ready["owner_output"],
                        "Current candidate plan/nonce/output closure differs",
                    )
                    self.children.append(closed)
                    require(
                        closed["controller_exit_code"] == 0
                        and closed["owner_completed"] is True
                        and closed["original_scoped_closure_positive"] is True,
                        "Independent original CPU positive completed closure required",
                    )
                except BaseException as error:
                    if primary is None:
                        primary = error
                    elif primary is not error:
                        primary.add_note(
                            "Candidate independent closure also failed: " + repr(error)
                        )
            if primary is not None:
                raise primary
            write(self.output / f"candidate-case-{index:02d}.json", record)
        self.verify_owned()

    def arms(self):
        result = []
        for arm in ("baseline", "candidate"):
            records = self.records[arm]
            summary = self.caller.aggregate_cases(records)
            value = {
                "arm": arm,
                "planned_denominator": 10,
                "attempted_cases": len(records),
                "known4_attempts": sum(row["group"] == "known4" for row in records),
                "public6_attempts": sum(row["group"] == "public6" for row in records),
                "records": records,
                **summary,
            }
            if arm == "candidate":
                value["children"] = list(self.children)
            result.append(value)
        return result


def run(plan_path, root_sha):
    released()
    owner = sys.modules.get(OWNER_MODULE)
    require(
        owner is not None and owner.PROFILE == PROFILE,
        "Authenticated selected entry owner required",
    )
    owner.released()
    plan = owner.load_plan(plan_path, root_sha)
    directory, output = owner.run_paths(plan)
    require(
        owner.DRIVER == str(Path(__file__).absolute().relative_to(ROOT)),
        "Exact selected campaign source entry required",
    )
    settings = plan["matched"]
    require(
        settings["schema"] == "jev-v7-matched-settings-v1"
        and settings["candidate_alias"] == ALIAS
        and settings["capability_sha256"] == CAPABILITY_SHA256,
        "Exact selected matched candidate settings required",
    )
    identity = process_identity()
    began = time.monotonic()

    def deadline():
        require(
            time.monotonic() - began < 10740, "Integrated campaign cooperative deadline exceeded"
        )

    def verify_owned():
        deadline()
        require(
            process_identity() == identity
            and identity["pid"] == identity["pgid"] == identity["sid"],
            "Original owned campaign actor changed",
        )
        intent = decode(raw(directory / "actor-intent.json"))
        require(
            intent["actor"] == identity
            and intent["plan_sha256"] == root_sha
            and intent["unit"] == plan["unit"]
            and intent["nonce"] == plan["nonce"],
            "Original actor intent/plan/unit/nonce binding required",
        )
        require(
            Path("/proc/self/cgroup").read_text().strip() == "0::" + plan["cgroup"]
            and Path("/proc/sys/kernel/random/boot_id").read_text().strip() == plan["boot_id"],
            "Same actual cgroup and boot required",
        )
        if lease is not None:
            lease.check()
        if models:
            require(len(models) == 1, "Exactly one original NF4 model")
            models[0].action_adapter.check(full=True)
        for key in ("caller_source", "serial_source", "native_driver_source"):
            authenticated(settings[key])
        require(
            hashlib.sha256(raw(plan_path)).hexdigest() == root_sha, "Selected root plan changed"
        )

    models, lease = [], None
    verify_owned()
    require(not output.exists(), "Fresh campaign output required")
    output.mkdir(mode=0o700)
    write(output / ".placeholder.json", {"schema": SCHEMA, "status": "setup_not_qualified"})
    for name in ("baseline", "candidate"):
        (output / name).mkdir(mode=0o700)
    serial = import_source(
        "_selected_matched_forward_serial", settings["serial_source"], SERIAL_SOURCE_SHA256
    )
    descriptor = wait_json(ordinary(settings["serialization_path"]), deadline, seconds=30)
    lease = serial.ForwardLease(
        descriptor["lock_path"],
        descriptor["journal_path"],
        descriptor["expected_stat"],
        nonce=descriptor["nonce"],
    )
    caller = import_source(
        "_selected_matched_v7_caller", settings["caller_source"], CALLER_SOURCE_SHA256
    )
    caller.released()
    native = import_source(
        "_selected_matched_original_nf4", settings["native_driver_source"], NATIVE_SOURCE_SHA256
    )
    require(
        tuple(native.KNOWN4 + native.PUBLIC6) == CASE_IDS, "Original exact ten source IDs required"
    )
    original_register, original_groups = native.register_native_gate, native.run_groups

    def register(model, gate):
        import torch

        models.append(model)
        return serial.install_nf4_hooks(
            model,
            torch,
            native,
            lease,
            resource_gate=gate,
            campaign_deadline=deadline,
            register_original=original_register,
        )

    cases = MatchedCases(
        owner=owner,
        plan=plan,
        root_sha=root_sha,
        directory=directory,
        output=output,
        caller=caller,
        lease=lease,
        verify_owned=verify_owned,
        deadline=deadline,
    )
    native.register_native_gate, native.run_groups = register, cases.run
    receipt = {
        "schema": SCHEMA,
        "profile": PROFILE,
        "root_plan_sha256": root_sha,
        "stage": "setup",
        "arms": [],
        "helper_runtime_qualification_claimed": False,
        "outer_external_closure_required": True,
        "official_score_claim": False,
        "dev_used_for_model_selection": True,
        "optimizer_input_dev": False,
        "optimizer_input_sealed": False,
        "new_unseen_evaluation_claim": False,
    }
    primary = None
    try:
        native_plan = native.metadata_plan(
            ROOT / plan["adapter"], output / "native", ROOT / plan["upstream"], "shared-elements-v2"
        )
        native_result = native.run(
            native_plan, expected_boot_id=plan["boot_id"], chromium=plan["chromium"]
        )
        verify_owned()
        receipt["arms"] = cases.arms()
        require(
            all(arm["infrastructure_complete"] for arm in receipt["arms"]),
            "Both original ten-case arms must be complete",
        )
        receipt.update(
            stage="complete_pending_external_closure",
            native_receipt_ref=reference(output / "native/receipt.json"),
            native_original_stage=native_result["stage"],
            helper_prepared_ref=reference(output / "native/helper-prepared.json"),
            nf4_model_loaded_ref=reference(output / "native/model-loaded.json"),
            helper_runtime_evidence_retained=True,
            helper_scope="Original same NF4 base, adapter disabled, original grammar/EOS/128/20s; complete HTTP/closure root audit still required",
        )
    except BaseException as error:
        primary = error
        receipt.update(
            stage="failed_or_incomplete_no_matched_score",
            partial_records=cases.records,
            partial_children=cases.children,
        )
    finally:
        native.register_native_gate, native.run_groups = original_register, original_groups
        receipt["error"] = (
            None if primary is None else {"type": type(primary).__name__, "message": str(primary)}
        )
        try:
            write(output / "receipt.json", receipt)
            deadline()
            if primary is None:
                files = native.output_inventory(output)
                write(
                    output / "output-index.json",
                    {
                        "scope": "Complete campaign output after original NF4 service drain; external pump handoffs and both owners still require independent closure joins",
                        "files": files,
                        "file_count": len(files),
                        "total_bytes": sum(item["size_bytes"] for item in files.values()),
                    },
                )
                deadline()
        except BaseException as error:
            if primary is None:
                primary = error
            elif primary is not error:
                primary.add_note("Campaign terminal receipt also failed: " + repr(error))
            retain_finalization_failure(output, primary)
    if primary is not None:
        raise primary
    return receipt


def main(argv=None):
    released()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args(argv)
    require(args.run, "Explicit selected campaign execution required")
    run(args.plan, args.plan_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
