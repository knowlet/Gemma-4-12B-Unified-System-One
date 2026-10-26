"""Stdlib-only native Q8 CPU cache arithmetic smoke; outer unit owns cleanup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import selectors
import subprocess
import sys
import time
from pathlib import Path

from smoke_common import (
    HERE,
    SCORING_SECONDS,
    compare,
    encoded,
    error_row,
    file_stat,
    need,
    ordinary,
    payload_check,
    read,
    ref,
    response_check,
    save,
    selection,
    sha,
)

MAX_RESPONSE = 1024**2
STDERR_LIMIT = 4 * 1024**2


def map_metadata(maps, selected_model):
    libraries, vmas, count, virtual_bytes = set(), [], 0, 0
    for line in maps.decode().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6:
            continue
        if re.search(r"\.so(?:\.[0-9]+)*$", parts[5]):
            libraries.add(parts[5])
        if selected_model is not None and parts[5] == selected_model:
            start, end = (int(value, 16) for value in parts[0].split("-"))
            need(end >= start, "Valid owned-child VMA addresses")
            count += 1
            virtual_bytes += end - start
            if len(vmas) < 64:
                vmas.append(
                    {
                        "start_hex": hex(start),
                        "end_hex": hex(end),
                        "virtual_bytes": end - start,
                        "permissions": parts[1],
                        "offset_bytes": int(parts[2], 16),
                        "device": parts[3],
                        "inode": int(parts[4]),
                        "path": parts[5],
                    }
                )
    return libraries, {
        "selected_path": selected_model,
        "observed_VMA_count": count,
        "observed_virtual_bytes": virtual_bytes,
        "retained_VMAs": vmas,
        "truncated": count > len(vmas),
        "scope": "Selected model path VMAs in this bounded maps observation only; virtual mappings are not resident memory, all-weight backing, absence of anonymous buffers, or a universal mmap proof",
    }


def process_identity(pid, selected_model=None):
    base = Path("/proc") / str(pid)

    def process():
        fields = (base / "stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "start_ticks": int(fields[19]),
            "pgid": int(fields[2]),
            "sid": int(fields[3]),
            "uid": base.stat().st_uid,
            "cgroup": (base / "cgroup").read_text().strip(),
        }

    before = process()
    result = {
        "process": before,
        "exe": str((base / "exe").resolve(strict=True)),
        "argv": [p.decode() for p in (base / "cmdline").read_bytes().split(b"\0") if p],
        "cwd": str((base / "cwd").resolve(strict=True)),
        "loaded_library_stats": [],
    }
    with (base / "maps").open("rb") as stream:
        maps = stream.read(MAX_RESPONSE + 1)
    need(len(maps) <= MAX_RESPONSE, "Bounded owned-child maps observation")
    paths, result["selected_model_file_mappings"] = map_metadata(maps, selected_model)
    need(len(paths) <= 128, "Bounded actual library population")
    for path in sorted(paths):
        result["loaded_library_stats"].append({"path": path, "stat": file_stat(path)})
    need(process() == before, "Stable owned child PID/start/session/cgroup during observation")
    return result


class NativeRPC:
    def __init__(self, argv, directory, context, full):
        self.directory = ordinary(directory)
        self.context, self.full = context, full
        self.selector = selectors.DefaultSelector()
        self.process = None
        self.stderr = None
        self.stderr_hash = hashlib.sha256()
        self.stderr_seen = self.stderr_saved = 0
        self.stdout = bytearray()
        self.rows = []
        try:
            self.stderr = (self.directory / "native.stderr").open("xb")
            self.launch_monotonic = time.monotonic()
            self.process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
                shell=False,
            )
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                os.set_blocking(stream.fileno(), False)
            self.selector.register(self.process.stdout, selectors.EVENT_READ, "stdout")
            self.selector.register(self.process.stderr, selectors.EVENT_READ, "stderr")
        except BaseException:
            self.detach()
            raise

    def consume(self, event):
        blob = os.read(event.fileobj.fileno(), 65536)
        if not blob:
            self.selector.unregister(event.fileobj)
            return
        if event.data == "stderr":
            self.stderr_hash.update(blob)
            self.stderr_seen += len(blob)
            retained = blob[: max(0, STDERR_LIMIT - self.stderr_saved)]
            self.stderr.write(retained)
            self.stderr.flush()
            self.stderr_saved += len(retained)
        else:
            self.stdout.extend(blob)
            need(len(self.stdout) <= MAX_RESPONSE, "Bounded native response line")

    def drain_pending(self):
        # Preserve ready diagnostics on failure, without waiting for a running child.
        for _ in range(64):
            events = self.selector.select(timeout=0)
            if not events:
                break
            for event, _ in events:
                if event.data != "stdin":
                    self.consume(event)

    def score(self, payload, timeout, name):
        wire = encoded(payload)
        need(len(wire) <= MAX_RESPONSE, "Bounded native request")
        save(self.directory / (name + "-request.json"), payload)
        submitted = time.monotonic()
        timing_started = self.launch_monotonic if name == "warm" else submitted
        deadline = timing_started + timeout
        sent, response, primary, response_wire = 0, None, None, None
        response_valid = False
        validated_at = None
        notes = []
        try:
            self.selector.register(self.process.stdin, selectors.EVENT_WRITE, "stdin")
            while True:
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError("Native research RPC deadline")
                for event, _ in self.selector.select(timeout=min(0.2, max(0, deadline - now))):
                    if event.data == "stdin":
                        written = os.write(event.fileobj.fileno(), wire[sent : sent + 65536])
                        need(written > 0, "Native request stdin remains writable")
                        sent += written
                        if sent == len(wire):
                            self.selector.unregister(event.fileobj)
                    else:
                        self.consume(event)
                if b"\n" in self.stdout:
                    line, extra = bytes(self.stdout).split(b"\n", 1)
                    need(
                        not extra and sent == len(wire),
                        "Exactly one response for submitted request",
                    )
                    response_wire = line + b"\n"
                    response = json.loads(line)
                    response_check(response, payload, self.context, self.full)
                    validated_at = time.monotonic()
                    if validated_at >= deadline:
                        raise TimeoutError(
                            "Native validated response exceeded research RPC deadline"
                        )
                    response_valid = True
                    self.stdout.clear()
                    break
                need(
                    self.process.poll() is None, "Native child remains live until complete response"
                )
        except BaseException as error:
            primary = error
        finally:
            try:
                self.selector.unregister(self.process.stdin)
            except KeyError:
                pass
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    notes.append({"phase": "stdin_unregister", **error_row(error)})
            if primary is not None:
                try:
                    self.drain_pending()
                except BaseException as error:
                    notes.append({"phase": "failure_diagnostic_drain", **error_row(error)})
            finished = time.monotonic()
            rpc_finished = validated_at if response_valid else finished
            elapsed = rpc_finished - timing_started
            row = {
                "name": name,
                "request_sha256": sha(wire),
                "request_size_bytes": len(wire),
                "launch_monotonic": self.launch_monotonic,
                "submitted_monotonic": submitted,
                "timing_started_monotonic": timing_started,
                "validated_response_monotonic": validated_at,
                "rpc_finished_monotonic": rpc_finished,
                "finalization_monotonic": finished,
                "local_finalization_seconds": finished - rpc_finished,
                "submission_elapsed_seconds": rpc_finished - submitted,
                "pre_submission_elapsed_seconds": submitted - timing_started,
                "deadline_monotonic": deadline,
                "deadline_seconds": timeout,
                "elapsed_seconds": elapsed,
                "meets20s_observed": response_valid and primary is None and elapsed <= 20,
                "elapsed_le20s": elapsed <= 20,
                "response_contract_validated": response_valid,
                "stdin_bytes_sent": sent,
                "child_returncode_at_exchange_final": self.process.poll(),
                "complete_response": response is not None,
                "response": response,
                "response_wire_sha256": sha(response_wire) if response_wire is not None else None,
                "response_wire_size_bytes": len(response_wire)
                if response_wire is not None
                else None,
                "partial_stdout_hex": bytes(self.stdout[:4096]).hex(),
                "partial_stdout_bytes": len(self.stdout),
                "primary_error": error_row(primary) if primary else None,
                "secondary_errors": notes,
                "startup_scope": "First short warm request measured from before Popen through response, including launch and identity observation; startup versus short scoring is not independently separated"
                if name == "warm"
                else "Post-warm request elapsed from payload submission including write/read/JSON overhead",
            }
            self.rows.append(row)
            try:
                raw = response_wire if response_wire is not None else bytes(self.stdout)
                with ordinary(
                    self.directory
                    / (
                        name
                        + ("-response.raw" if response_wire is not None else "-response.partial")
                    )
                ).open("xb") as stream:
                    stream.write(raw)
                save(self.directory / (name + "-exchange.json"), row)
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    notes.append({"phase": "exchange_save", **error_row(error)})
        if primary is not None:
            primary.evidence = row
            raise primary
        return response

    def finish(self):
        self.process.stdin.close()  # Normal EOF requests native main-loop exit.
        deadline = time.monotonic() + 5
        while self.process.poll() is None or self.selector.get_map():
            need(
                time.monotonic() < deadline,
                "Native normal EOF exit deadline; outer unit retains ownership",
            )
            for event, _ in self.selector.select(timeout=0.1):
                self.consume(event)
        need(
            not self.stdout and self.process.wait(timeout=0.1) == 0,
            "Native clean EOF exit with no unsolicited output",
        )
        return {"returncode": self.process.returncode, "normal_EOF_exit": True}

    def detach(self):
        notes = []
        operations = [("selector", self.selector.close)]
        if self.process is not None:
            operations.extend(
                ("pipe", stream.close)
                for stream in (self.process.stdin, self.process.stdout, self.process.stderr)
                if stream is not None
            )
        for name, operation in operations:
            try:
                operation()
            except BaseException as error:
                notes.append({"phase": name, **error_row(error)})
        if self.stderr is not None:
            try:
                self.stderr.close()
            except BaseException as error:
                notes.append({"phase": "stderr", **error_row(error)})
        return notes

    def evidence(self):
        return {
            "pid": self.process.pid if self.process else None,
            "poll": self.process.poll() if self.process else None,
            "stderr_observed_bytes": self.stderr_seen,
            "stderr_observed_sha256": self.stderr_hash.hexdigest(),
            "stderr_retained_bytes": self.stderr_saved,
            "stderr_truncated": self.stderr_seen > self.stderr_saved,
            "stderr_scope": "First4MiB retained; hash/count cover bytes drained before detach, not unread bytes after failure",
            "direct_kill_or_terminate_called": False,
        }


def run(args):
    need("torch" not in sys.modules, "Fresh stdlib scoring process must not import Torch")
    need(
        args.expected_entry_sha256 == sha(read(HERE / "run_smoke.py")), "Explicit scoring entry SHA"
    )
    cfg = selection(args.selection, args.selection_sha256, HERE / "run_smoke.py", runtime=True)
    need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("ACCELERATE_USE_CPU") == "true",
        "Explicit CPU-only scoring environment",
    )
    prepared = ordinary(args.payload_root)
    receipt = json.loads(read(prepared / "prepare-receipt.json"))
    need(
        receipt["completed"] is True
        and receipt["selection_sha256"] == args.selection_sha256
        and receipt["entry_sha256"] == ref(HERE / "prepare_smoke.py")["sha256"],
        "Successful original preparation bound to same selected sources",
    )
    payloads = {}
    for name, long in (("warm", False), ("long", True)):
        record = receipt["state"]["generated"][name]
        path = prepared / (name + "-native.json")
        need(ref(path) == record["payload"], "Prepared payload exact byte hash")
        payloads[name] = payload_check(json.loads(read(path)), long=long)
    out = ordinary(args.output)
    need(not out.exists(), "Fresh scoring output")
    out.mkdir(mode=0o700)
    state = {
        "prepare_receipt": ref(prepared / "prepare-receipt.json"),
        "cases": [],
        "comparisons": [],
    }
    primary, notes, active = None, [], None
    try:
        baseline = None
        for name, context, full in (
            ("full2048", 2048, True),
            ("compact2048", 2048, False),
            ("compact8192", 8192, False),
        ):
            directory = out / name
            directory.mkdir(mode=0o700)
            native = cfg["native"]
            argv = [
                native["path"],
                "--model",
                cfg["package"]["model"]["path"],
                "--mmproj",
                cfg["package"]["mmproj"]["path"],
                "--ctx-size",
                str(context),
                "--batch-size",
                "512",
                "--threads",
                "1",
                "--gpu-layers",
                "0",
                "--swa-full",
                "1" if full else "0",
                "--cpu-mmap-only",
                "1",
            ]
            row = {
                "name": name,
                "argv": argv,
                "context_requested": context,
                "swa_full_requested": full,
                "input_tokens": payloads["long"]["source_input_token_count"],
                "completed": False,
            }
            state["cases"].append(row)
            save(directory / "intent.json", row)
            active = NativeRPC(argv, directory, context, full)
            try:
                row["initial_identity"] = process_identity(
                    active.process.pid, cfg["package"]["model"]["path"]
                )
                need(
                    row["initial_identity"]["exe"] == native["path"],
                    "Actual native child executable",
                )
            except (FileNotFoundError, ProcessLookupError) as error:
                row["initial_identity_observation_error"] = error_row(error)
            # A fast startup failure is diagnosed through the bounded RPC/stderr path.
            warm = active.score(payloads["warm"], 60, "warm")
            response_check(warm, payloads["warm"], context, full)
            row["postwarm_identity"] = process_identity(
                active.process.pid, cfg["package"]["model"]["path"]
            )
            need(
                row["postwarm_identity"]["exe"] == native["path"],
                "Actual postwarm native executable",
            )
            if "initial_identity" not in row:
                row["initial_identity"] = row["postwarm_identity"]
            loaded = {
                r["path"]: r["stat"] for r in row["postwarm_identity"]["loaded_library_stats"]
            }
            need(
                all(loaded.get(lib["path"]) == lib["stat"] for lib in native["libraries"]),
                "Selected native libraries actually mapped with pinned stats after warm response",
            )
            long = active.score(payloads["long"], SCORING_SECONDS, "long")
            logits = response_check(long, payloads["long"], context, full)
            row["postscore_identity"] = process_identity(
                active.process.pid, cfg["package"]["model"]["path"]
            )
            need(
                row["postscore_identity"]["process"] == row["initial_identity"]["process"],
                "Same native PID/start/session/cgroup through scores",
            )
            row["normal_exit"] = active.finish()
            row["RPCs"] = active.rows
            row["native_observation"] = active.evidence()
            cleanup = active.detach()
            active = None
            need(not cleanup, "Native stdlib descriptor cleanup")
            row["completed"] = True
            if baseline is None:
                baseline = logits
            else:
                comparison = {
                    "reference": "full2048",
                    "candidate": name,
                    **compare(baseline, logits),
                }
                state["comparisons"].append(comparison)
                need(comparison["passed"], "Preselected cache arithmetic threshold and argmax")
            save(directory / "case-receipt.json", row)
    except BaseException as error:
        primary = error_row(error)
    finally:
        if active is not None:
            state["failed_child"] = active.evidence()
            state["failed_RPCs"] = active.rows
            notes.extend(active.detach())
            state["failed_child_left_to_outer_authenticated_unit_close"] = True
        try:
            selection(args.selection, args.selection_sha256, HERE / "run_smoke.py", runtime=True)
            need("torch" not in sys.modules, "Scoring remains stdlib-only")
            state["source_and_stat_integrity_final"] = True
        except BaseException as error:
            row = {"phase": "final_source_or_stat", **error_row(error)}
            if primary is None:
                primary = row
            else:
                notes.append(row)
        save(
            out / "smoke-receipt.json",
            {
                "profile": "llama-cpp-q8-cpu-cache-smoke-v1",
                "completed": primary is None and not notes,
                "selection_sha256": args.selection_sha256,
                "entry_sha256": args.expected_entry_sha256,
                "research_scoring_deadline_seconds": SCORING_SECONDS,
                "warm_including_startup_deadline_seconds": 60,
                "browser_RPC_contract_seconds": 20,
                "browser_RPC_contract_changed": False,
                "state": state,
                "primary_error": primary,
                "secondary_errors": notes,
                "Torch_imported": "torch" in sys.modules,
                "quality_N3_Q95_or_20s_qualified": False,
                "scope": "Synthetic CPU1 cache arithmetic; 8192 is actual allocated context for same <=2048-token input, not an8192-length benchmark",
            },
        )
    return 0 if primary is None and not notes else 1


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--expected-entry-sha256")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-sha256")
    parser.add_argument("--payload-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.run:
        print(
            json.dumps(
                {"mode": "DRY_SOURCE_ONLY", "Torch_imported": False, "native_started": False}
            )
        )
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
