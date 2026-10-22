"""Owned CPU fixture only. No model imports, credential arguments or HTTP logs."""

import hashlib
import json
import os
import secrets
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--internal-child":
        return 2
    signal.alarm(15)
    config_path = Path(sys.argv[2])
    config = json.loads(config_path.read_bytes())
    directory = config_path.parent

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            control_path = directory / "fake-control.json"
            control = json.loads(control_path.read_bytes()) if control_path.exists() else {}
            selected = self.headers.get("Authorization") == "Bearer " + config["token"]
            if control.get("ignore_credentials"):
                selected = True
            status = (
                (200 if control.get("generation_instead_of_schema") else 422) if selected else 500
            )
            blob = (
                json.dumps({"error": {"type": "helper_schema"}}).encode()
                if selected
                else b"Internal Server Error"
            )
            request_id = None
            if selected or control.get("unauthorized_creates_trace"):
                request_id = secrets.token_hex(16)
                request_root = directory / "requests" / request_id
                request_root.mkdir(parents=True)
                raw = {
                    "schema_id": "jev-public-field-syntax-raw-candidate-v1",
                    "request_id": request_id,
                    "api_request_id": None,
                    "payload_sha256": hashlib.sha256(b"{}").hexdigest(),
                    "prefix": None,
                    "prepared_prompt": None,
                    "helper_phases": [],
                    "generation": {},
                    "forward_events": [],
                    "guard_restored": True,
                    "server_terminal": True,
                    "error": None,
                    "candidate_response": {"error": {"type": "helper_schema"}},
                }
                if not control.get("schema_without_handler_evidence"):
                    (request_root / "raw.json").write_bytes(json.dumps(raw).encode())
            self.send_response(status)
            if request_id:
                self.send_header("X-Public-Request-ID", request_id)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    endpoint = {
        "policy": f"http://127.0.0.1:{port}/v1/systemone",
        "helper": f"http://127.0.0.1:{port}",
        "model": "generated-actions-none",
        "qualified": True,
        "TEXT_MODEL_REASONING": "none",
    }
    pending = directory / ".endpoint-pending"
    descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(endpoint).encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, directory / "endpoint.json")
    pending.unlink()
    server.serve_forever(poll_interval=0.02)


if __name__ == "__main__":
    raise SystemExit(main())
