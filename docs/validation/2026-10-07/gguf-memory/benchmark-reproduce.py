"""Serial native experiments with identical CPU-prepared, gold-free inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import selectors
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(ROOT / "scripts"))
from gguf_adapter import make_native_payload  # noqa: E402

from s1.contracts import DecisionRequest  # noqa: E402
from s1.unified import UnifiedDecisionModel, candidate_ids  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exchange(process, payload, timeout=300):
    start = time.perf_counter()
    process.stdin.write(json.dumps(payload) + "\n")
    process.stdin.flush()
    selector = selectors.DefaultSelector()
    try:
        selector.register(process.stdout, selectors.EVENT_READ)
        if not selector.select(timeout):
            raise TimeoutError(f"native request timed out: {payload['case_id']}")
        line = process.stdout.readline()
    finally:
        selector.close()
    if not line:
        raise RuntimeError(f"native exited: {process.poll()}")
    response = json.loads(line)
    elapsed = (time.perf_counter() - start) * 1000
    if response.get("case_id") != payload["case_id"]:
        raise ValueError("response case identity differs")
    if response.get("status") == "ok":
        for key in ("question_ids", "slots"):
            if response[key] != payload[key]:
                raise ValueError(f"response {key} differs")
        if response["runtime_slots"] != payload["slots"]:
            raise ValueError("runtime slots differ")
        if response["native_token_count"] != payload["source_input_token_count"]:
            raise ValueError("native token count differs")
        rows = response["raw_logits"]
        if len(rows) != len(payload["slots"]):
            raise ValueError("missing output rows")
        for row, count in zip(rows, payload["nopts"]):
            if len(row) != count or not np.isfinite(row).all():
                raise ValueError("invalid legal candidate logits")
    return {"wall_ms": elapsed, "response": response}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--boundary-repeat", type=int, default=1)
    args = parser.parse_args()
    if min(args.repeat, args.boundary_repeat) < 1:
        parser.error("repeat counts must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    package = ROOT / "artifacts/exports/s1-boolq-gguf"
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(package, local_files_only=True)
    context = SimpleNamespace(
        processor=processor, tok=processor.tokenizer, device="cpu", max_context=16384
    )
    letters = candidate_ids(context.tok)
    temperature = json.loads((package / "s1_config.json").read_text())["temperature"]
    records = []
    for filename in ("gguf-runtime-smoke-inputs.jsonl", "gguf-boundary-smoke-inputs.jsonl"):
        records.extend(
            (json.loads(line), "boundary" in filename)
            for line in (
                ROOT
                / "docs/validation/2026-10-07/gguf-memory"
                / ("boundary-inputs.jsonl" if "boundary" in filename else "runtime-inputs.jsonl")
            )
            .read_text()
            .splitlines()
        )
    request64 = {
        "state": "The number two is even.",
        "questions": [
            {"id": f"q{i}", "type": "noul", "instructions": "Is two even?"} for i in range(64)
        ],
    }
    records.append(({"case_id": "slots-64", "request": request64}, True))
    results = {}
    with tempfile.TemporaryDirectory(prefix="s1-research-", dir=args.output.parent) as tmp:
        payloads = []
        for index, (envelope, boundary) in enumerate(records):
            request = DecisionRequest.model_validate(envelope["request"])
            inputs, slots, counts = UnifiedDecisionModel.prepare(context, request)
            directory = Path(tmp) / str(index)
            directory.mkdir()
            payload = make_native_payload(
                envelope if "input_ids" in envelope else None,
                request,
                inputs,
                slots.tolist(),
                counts.tolist(),
                letters,
                directory,
            )
            payload["case_id"] = envelope["case_id"]
            payloads.append((payload, boundary))
        payload_receipt = [
            {
                "case_id": p["case_id"],
                "tokens": p["source_input_token_count"],
                "slots": p["slots"],
                "nopts": p["nopts"],
                "sha256": hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest(),
            }
            for p, _ in payloads
        ]
        for specification in args.variant:
            name, binary = specification.split("=", 1)
            binary = Path(binary).resolve()
            stderr_path = args.output.parent / f"{name}.stderr.log"
            command = [
                str(binary),
                "--model",
                str(package / "s1-boolq-Q8_0.gguf"),
                "--mmproj",
                str(package / "mmproj-s1-boolq-f16.gguf"),
                "--ctx-size",
                "16384",
                "--batch-size",
                "8192",
                "--gpu-layers",
                "99",
                "--threads",
                "8",
                "--image-max-tokens",
                "280",
            ]
            started = time.perf_counter()
            with stderr_path.open("w") as log:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                    bufsize=1,
                    cwd=ROOT,
                )
                try:
                    first = exchange(process, payloads[0][0])
                    startup_first_ms = (time.perf_counter() - started) * 1000
                    if first["response"]["status"] != "ok":
                        raise ValueError(first)
                    rows = []
                    for payload, boundary in payloads:
                        warmup = exchange(process, payload)
                        if warmup["response"]["status"] != "ok":
                            raise ValueError(warmup)
                        measured = [
                            exchange(process, payload)
                            for _ in range(args.boundary_repeat if boundary else args.repeat)
                        ]
                        if any(r["response"]["status"] != "ok" for r in measured):
                            raise ValueError(measured)
                        rows.append(
                            {"case_id": payload["case_id"], "warmup": warmup, "measured": measured}
                        )
                        print(
                            json.dumps(
                                {
                                    "event": "case",
                                    "variant": name,
                                    "case_id": payload["case_id"],
                                    "native_ms": [r["response"]["latency_ms"] for r in measured],
                                }
                            ),
                            flush=True,
                        )
                    invalid = json.loads(json.dumps(payloads[-1][0]))
                    invalid["case_id"] = "slots-65-rejected"
                    invalid["slots"].insert(1, invalid["slots"][0] + 1)
                    invalid["nopts"].insert(1, 2)
                    invalid["question_ids"].insert(1, "extra-question")
                    rejected = exchange(process, invalid)
                    if (
                        rejected["response"]["status"] != "error"
                        or rejected["response"].get("error") != "inconsistent question metadata"
                    ):
                        raise ValueError("65 slots must be rejected")
                    recovery = exchange(process, payloads[0][0])
                    if recovery["response"]["status"] != "ok":
                        raise ValueError("error recovery failed")
                    rss = subprocess.check_output(
                        ["ps", "-o", "rss=", "-p", str(process.pid)], text=True
                    ).strip()
                    results[name] = {
                        "command": command,
                        "binary_sha256": sha(binary),
                        "startup_including_first_request_ms": startup_first_ms,
                        "first": first,
                        "cases": rows,
                        "invalid65": rejected,
                        "recovery": recovery,
                        "rss_kib_at_end": int(rss),
                        "rss_scope": "one process RSS sample after all requests; not peak or total unified-memory use",
                        "stderr_path": str(stderr_path),
                    }
                finally:
                    process.stdin.close()
                    try:
                        code = process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        code = process.wait(timeout=10)
                    if code:
                        raise RuntimeError(f"{name} exited {code}")
            results[name]["stderr_sha256"] = sha(stderr_path)
            checkpoint = args.output.with_suffix(".checkpoint.json")
            checkpoint.write_text(
                json.dumps({"status": "in_progress", "variants": results}, indent=2) + "\n"
            )
    baseline_name = next(iter(results))
    baseline = results[baseline_name]
    comparison = {}

    def probabilities(row):
        values = np.asarray(row, dtype=np.float64) / temperature
        values -= values.max()
        return np.exp(values) / np.exp(values).sum()

    for name, variant in results.items():
        raw_max = probability_max = 0.0
        count = agreements = 0
        for original, candidate in zip(baseline["cases"], variant["cases"]):
            for a, b in zip(original["measured"], candidate["measured"]):
                for ar, br in zip(a["response"]["raw_logits"], b["response"]["raw_logits"]):
                    raw_max = max(raw_max, float(np.max(np.abs(np.asarray(ar) - br))))
                    probability_max = max(
                        probability_max,
                        float(np.max(np.abs(probabilities(ar) - probabilities(br)))),
                    )
                    count += 1
                    agreements += int(np.argmax(ar) == np.argmax(br))
        original = baseline["cases"][0]["measured"][0]["response"]["raw_logits"]
        recovery = variant["recovery"]["response"]["raw_logits"]
        recovery_max = max(
            float(np.max(np.abs(np.asarray(a) - b))) for a, b in zip(original, recovery)
        )
        comparison[name] = {
            "decision_comparisons": count,
            "argmax_agreements": agreements,
            "max_abs_raw_logit_difference": raw_max,
            "max_abs_calibrated_probability_difference": probability_max,
            "recovery_max_abs_raw_difference": recovery_max,
        }
    report = {
        "status": "measured",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "harness_sha256": sha(__file__),
        "temperature": temperature,
        "timing_scope": "native persistent scoring including media projection; CPU HF preparation excluded; one per-case warmup",
        "repeat": args.repeat,
        "boundary_repeat": args.boundary_repeat,
        "payloads": payload_receipt,
        "variants": results,
        "comparison": comparison,
    }
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"event": "complete", "comparison": comparison}), flush=True)


if __name__ == "__main__":
    main()
