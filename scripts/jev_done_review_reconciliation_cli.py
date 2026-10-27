"""Load a single caller-selected run and print only its DONE reconciliation aggregate.

This reads private artifacts: choose paths only after authorization. Source pins,
origin, closure provenance, numerical witnesses and full run completeness remain
external. Paths are opened beneath an absolute root using no-follow dirfds.
"""

import argparse
import hashlib
import json
import os
import re
import stat
import sys

import jev_done_review_reconciliation as audit


def parts(path, *, absolute=False):
    if type(path) is not str or not path or path.startswith("/") != absolute:
        raise ValueError("invalid path")
    value = path[1:] if absolute else path
    result = value.split("/")
    if any(p in ("", ".", "..") for p in result):
        raise ValueError("invalid path component")
    return result


def directory(parent, name):
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
    if not stat.S_ISDIR(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("directory required")
    return fd


def walk(start, path):
    fd = os.dup(start)
    try:
        for name in path:
            nxt = directory(fd, name)
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def read(root, path, limit):
    names = parts(path)
    parent = walk(root, names[:-1])
    try:
        fd = os.open(names[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
                raise ValueError("bounded regular file required")
            chunks, size = [], 0
            while True:
                chunk = os.read(fd, min(65536, limit + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise ValueError("file too large")
                chunks.append(chunk)
            after = os.fstat(fd)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("file changed")
            return b"".join(chunks)
        finally:
            os.close(fd)
    finally:
        os.close(parent)


def names(root, path, pattern, template):
    fd = walk(root, parts(path))
    try:
        values = sorted(os.listdir(fd))
    finally:
        os.close(fd)
    if len(values) > 1024 or any(re.fullmatch(pattern, value) is None for value in values):
        raise ValueError("unexpected directory contents")
    if values != [template.format(i) for i in range(1, len(values) + 1)]:
        raise ValueError("noncontiguous sequence")
    return values


def value(root, path, limit=4 * 1024 * 1024):
    return audit._parse(read(root, path, limit))


def load(root_path, *, runner_records, events_dir, capture_dir, service_dir):
    start = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        root = walk(start, parts(root_path, absolute=True))
    finally:
        os.close(start)
    try:
        record_parts = parts(runner_records)
        if (
            len(record_parts) < 4
            or record_parts[-1] != "model_calls.json"
            or record_parts[-3] != "attempts"
            or record_parts[-4] != "results"
        ):
            raise ValueError("runner records must belong to results/attempts")
        attempt_dir = "/".join(record_parts[:-2])
        attempt_fd = walk(root, parts(attempt_dir))
        try:
            if os.listdir(attempt_fd) != [record_parts[-2]]:
                raise ValueError("exactly one attempt required")
        finally:
            os.close(attempt_fd)
        records = value(root, runner_records)
        events = [
            value(root, events_dir + "/" + name, 1024 * 1024)
            for name in names(root, events_dir, r"event-[0-9]{4}\.json", "event-{:04d}.json")
        ]
        captures = []
        for name in names(root, capture_dir, r"attempt-[0-9]{4}", "attempt-{:04d}"):
            base = capture_dir + "/" + name
            record = value(root, base + "/record.json", 1024 * 1024)
            item = {"record": record}
            if record.get("status") == "response":
                item["request_bytes"] = read(root, base + "/request_bytes.bin", 1_000_000)
                item["response_bytes"] = read(root, base + "/response_bytes.bin", 16 * 1024**2)
            captures.append(item)
        capture_parts = parts(capture_dir)
        if len(capture_parts) < 2:
            raise ValueError("capture parent required")
        closure = value(
            root, "/".join(capture_parts[:-1]) + "/owned-client-closure.json", 1024 * 1024
        )
        terminal = value(root, service_dir + "/terminal.json")
        refs = terminal["service"]["private_bindings"]
        if type(refs) is not list or len(refs) > 1024:
            raise ValueError("invalid binding references")
        bindings = []
        for i, ref in enumerate(refs, 1):
            if (
                type(ref) is not dict
                or set(ref) != {"path", "sha256", "size_bytes"}
                or ref["path"] != f"binding-{i:04d}.json"
                or not audit._digest(ref["sha256"])
                or type(ref["size_bytes"]) is not int
            ):
                raise ValueError("invalid binding reference")
            raw = read(root, service_dir + "/" + ref["path"], 64 * 1024**2)
            if len(raw) != ref["size_bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
                raise ValueError("binding differs from reference")
            bindings.append(audit._parse(raw))
        raw_events = read(root, service_dir + "/http-events.jsonl", 8 * 1024**2)
        service_events = [audit._parse(line) for line in raw_events.splitlines()]
        return audit.reconcile(
            events=events,
            runner_records=records,
            captures=captures,
            client_closure=closure,
            bindings=bindings,
            service_events=service_events,
        )
    finally:
        os.close(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="absolute run directory")
    parser.add_argument("--runner-records", required=True, help="relative model_calls.json path")
    parser.add_argument("--events-dir", required=True, help="relative DONE events directory")
    parser.add_argument("--capture-dir", required=True, help="relative client capture directory")
    parser.add_argument("--service-dir", required=True, help="relative candidate service directory")
    args = parser.parse_args()
    try:
        result = load(
            args.run_root,
            runner_records=args.runner_records,
            events_dir=args.events_dir,
            capture_dir=args.capture_dir,
            service_dir=args.service_dir,
        )
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        print(
            "Private evidence could not be loaded or reconciled; no result was produced.",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "supplied_evidence_consistent" else 1


if __name__ == "__main__":
    raise SystemExit(main())
