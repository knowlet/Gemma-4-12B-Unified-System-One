"""CPU integration fixture only; no native model/backend/qualification claims."""

import argparse
import json
import os
import subprocess
import sys
import time

from local_transport import Selection, encode, parse, parse_envelope, serve_worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--mode", default="valid")
    args = parser.parse_args()
    selection = Selection(**json.loads(args.selection))
    descendants = []
    escaped = None

    class Control:
        def write(self, blob):
            original = bytes(blob)
            value = parse(original)
            if args.mode == "truncated-ready" and value["event"] == "ready":
                sys.stdout.buffer.write(b'{"event":')
                sys.stdout.buffer.flush()
                os._exit(3)
            if args.mode == "missing-closed" and value["event"] == "closed":
                return len(original)
            replacements = {
                "wrong-ready-nonce": ("ready", "nonce", "another-owner"),
                "wrong-ready-pid": ("ready", "pid", os.getppid()),
                "wrong-ready-start": ("ready", "start_ticks", 1),
                "wrong-ready-uuid": ("ready", "gpu_uuid", "another-device"),
                "wrong-ready-total": ("ready", "mem_total_bytes", selection.expected_memtotal - 4096),
                "wrong-response-model": ("response", "model", "d3-lite"),
                "wrong-response-sha": ("response", "http_sha256", "0" * 64),
                "wrong-response-request": ("response", "request_id", "different-request"),
                "wrong-closed-start": ("closed", "start_ticks", 1),
                "closed-with-error": ("closed", "error", {"type": "SyntheticFinalError"}),
            }
            if args.mode in replacements:
                event, key, replacement = replacements[args.mode]
                if value["event"] == event:
                    value[key] = replacement
            sys.stdout.buffer.write(encode(value) + b"\n")
            return len(original)

        def flush(self):
            sys.stdout.buffer.flush()

    def initialize():
        nonlocal escaped
        if args.mode == "partial-startup":
            raise RuntimeError("Fake constructor entered, then failed")
        if args.mode == "startup-timeout":
            time.sleep(2)
        if args.mode == "detached-descendant":
            descendants.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                                start_new_session=True))
        if args.mode == "unmarked-double-fork":
            reader, writer = os.pipe()
            intermediate = os.fork()
            if intermediate == 0:
                os.close(reader)
                if os.fork() == 0:
                    os.setsid()
                    os.set_inheritable(writer, True)
                    code = (
                        "import json,os,time; "
                        "raw=open('/proc/self/stat').read().rsplit(')',1)[1].split(); "
                        "value={'pid':os.getpid(),'start_ticks':int(raw[19])}; "
                        f"os.write({writer},json.dumps(value).encode());os.close({writer}); "
                        "time.sleep(30)"
                    )
                    os.execve(sys.executable, [sys.executable, "-c", code], {})
                os._exit(0)
            os.close(writer)
            os.waitpid(intermediate, 0)
            with os.fdopen(reader, "rb") as stream:
                escaped = json.loads(stream.read(4096))
        return {"cpu_fixture": True, "model_loaded": False, "qualified": False,
                "escaped_fixture": escaped}

    def process_request(line):
        envelope = parse_envelope(line)
        if args.mode == "request-timeout":
            time.sleep(0.15)
        refused = args.mode == "refused"
        return {
            "event": "response", "request_id": envelope["request_id"],
            "http_sha256": envelope["http_sha256"], "nonce": selection.nonce,
            "model": selection.model, "gpu_uuid": selection.expected_gpu_uuid,
            "success": not refused, "campaign_stop": refused,
            "wire_body_utf8": None if refused else encode({"fixture": True, "answer": "WAIT"}).decode(),
            "journal": {"cpu_fixture": True, "original_envelope": envelope,
                        "raw_envelope_utf8": line.decode(), "native_calls": 0},
        }

    def finalize():
        for child in descendants:
            child.terminate()
            child.wait(timeout=1)
        if args.mode == "failed-finalize":
            raise RuntimeError("Fake final source/cache check failed")
        return {"success": True, "cpu_fixture": True, "model_loaded": False}

    result = serve_worker(selection, initialize=initialize, process_request=process_request,
                          finalize=finalize, control=Control())
    if args.mode == "linger-after-closed":
        time.sleep(30)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
