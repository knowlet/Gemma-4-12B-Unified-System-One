"""Prospective CPU-only public Decision3 cache download and closed profile.

Only an exact18 named-source inventory is baked. No701/717 collection, model
load or GPU request. Preparation/child dispatch exits before importing Modal.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import os
import re
import secrets
import select
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
SCRIPT = ROOT / "apps/modal/decision3_public_weights.py"
SERVICE = ROOT / "apps/modal/decision3_browser_service.py"
SERVICE_SHA = "55229ac0215b9d070c26535e5e5e781f8e1eff8b0f322112a0466023d56e4023"
INTERPRETER = "/opt/decision3/bin/python"
SCHEMA = "decision3-public-cpu-weight-source18-v1"
MAX_LOG, DEADLINE = 16 * 1024**2, 14400


def load_service():
    # Prefix execution supplies reviewed stdlib staging/ownership composition;
    # neither the SDK/image graph nor any model/backend constructor is evaluated.
    raw = SERVICE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SERVICE_SHA:
        raise ValueError("Reviewed browser service source differs")
    tree = ast.parse(raw)
    stop = next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Import) and any(alias.name == "modal" for alias in node.names)
    )
    module = types.ModuleType("reviewed_decision3_weight_service")
    module.__file__ = str(SERVICE)
    exec(
        compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(SERVICE), "exec"),
        module.__dict__,
    )
    module.ordinary(SERVICE)
    return module


def inventory():
    service = load_service()
    original = service.source_inventory()
    rows = [
        *original["rows"],
        {
            "path": str(SCRIPT),
            "pin": service.pin(SCRIPT),
            "classification": "public_cpu_weight_launcher",
        },
    ]
    rows.sort(key=lambda row: row["path"])
    service.require(
        len(rows) == len({row["path"] for row in rows}) == 18,
        "Exactly18 named public sources required",
    )
    return {
        "schema_id": SCHEMA,
        "files": 18,
        "rows": rows,
        "service_source_inventory_sha256": hashlib.sha256(service.encode(original)).hexdigest(),
        "weights_uploaded": False,
        "credentials_uploaded": False,
        "collection_or_parent_uploaded": False,
    }


def authenticate_manifest(blob):
    service = load_service()
    supplied = service.parse(blob)
    service.require(
        len(blob) <= 65536 and supplied == inventory() and blob == service.encode(supplied),
        "Reviewed exact18 source inventory required",
    )
    return supplied


def setup(blob):
    service = load_service()
    authenticate_manifest(blob)
    staging = service.load_staging()
    # Every named source is baked before startup; this also handles an SDK
    # importer that previously observed a missing scripts directory.
    importlib.invalidate_caches()
    core = staging.load_core()
    identities = core.authenticate_source(service.CORE_SHA)
    return service, staging, core, identities


def verify_geometry(core, staging, binding, identity, manifest, files):
    """Authenticate full-file hashes first, then publisher index/header geometry."""

    def file_at(label):
        return staging.provider_cache_file(binding, identity, label)

    layout = core.backbone_layout(files, lambda label: core.read(file_at(label)))
    shards, weight_map = layout["shards"], layout["weight_map"]
    union, parameters = {}, 0
    for shard in shards:
        entries, count = core.tensor_header(file_at(shard))
        core.require(
            bool(entries)
            and not set(union).intersection(entries)
            and all(row["dtype"] == "BF16" for row in entries.values()),
            "Empty/duplicate/non-BF16 backbone tensor",
        )
        if weight_map is not None:
            core.require(
                {name for name, filename in weight_map.items() if filename == shard}
                == set(entries),
                "Shard header/index keys differ",
            )
        union.update(entries)
        parameters += count
    head, head_parameters = core.tensor_header(file_at("readout.safetensors"))
    width = identity["readout_tensor_bytes"] // (255 * 4)
    core.require(
        head
        == {
            "weight": {
                "dtype": "F32",
                "shape": [255, width],
                "data_offsets": [0, identity["readout_tensor_bytes"]],
            }
        },
        "Fixed255 FP32 readout differs",
    )
    core.require(
        (weight_map is None or set(union) == set(weight_map))
        and head_parameters == 255 * width
        and parameters + head_parameters == identity["loaded_parameters"]
        and sum(files[name]["size_bytes"] for name in [*shards, "readout.safetensors"])
        == identity["resident_file_bytes"],
        "All-shard parameter/file byte count differs",
    )
    core.require(
        set(files) == set(staging.runtime_files(manifest)),
        "Complete publisher runtime files required",
    )
    return {
        "all_shards": shards,
        "backbone_layout": layout["layout"],
        "index_file": layout["index_file"],
        "backbone_parameters": parameters,
        "readout_shape": [255, width],
        "full_payload_sha256_authenticated": True,
    }


def download_models(service, staging, core, identities, binding, output, snapshot_download):
    remote_hub = staging.mount_root(binding)
    report = {
        "models": {},
        "token": False,
        "model_loads": 0,
        "GPU_requested": False,
        "source_inventory_sha256": hashlib.sha256(service.encode(inventory())).hexdigest(),
        "downloader_source": service.pin(SCRIPT),
        "provider_mount": binding,
        "resources_before": core.resource_snapshot(),
        "cache_disk_capacity": {
            "observed": False,
            "usable_capacity_evidence": False,
            "scope": "Provider statfs is a placeholder; actual I/O errors are retained",
        },
    }
    for name in service.MODELS:
        identity = identities[name]
        arguments = {
            "repo_id": identity["model_id"],
            "revision": identity["revision"],
            "cache_dir": str(remote_hub),
            "token": False,
            "max_workers": 1,
        }
        actual = Path(snapshot_download(**arguments, allow_patterns=["MODEL_MANIFEST.json"]))
        expected = staging.provider_snapshot(binding, identity)
        service.require(actual == expected, "Exact immutable public cache path required")
        manifest = core.read(
            staging.provider_cache_file(binding, identity, "MODEL_MANIFEST.json"),
            identity["manifest_sha256"],
        )
        service.require(
            manifest["schema"] == "d3-package-manifest/1"
            and manifest["model_name"] == name
            and manifest["repo_id"] == identity["model_id"],
            "Pinned public publisher identity differs",
        )
        names = staging.runtime_files(manifest)
        snapshot_download(**arguments, allow_patterns=["MODEL_MANIFEST.json", *names])
        files = {}
        for label in names:
            files[label] = core.pin(staging.provider_cache_file(binding, identity, label))
            service.require(
                files[label]
                == {
                    "sha256": manifest["files_sha256"][label],
                    "size_bytes": manifest["files_bytes"][label],
                },
                "Full public runtime SHA/size differs",
            )
        geometry = verify_geometry(core, staging, binding, identity, manifest, files)
        report["models"][name] = {
            "revision": identity["revision"],
            "manifest_sha256": identity["manifest_sha256"],
            "snapshot": str(expected),
            "files": files,
            "geometry": geometry,
        }
        core.write(output / (name + "-download.json"), report["models"][name])
        staging.mount_root(binding)
    report["resources_after"] = core.resource_snapshot()
    core.authenticate_source(service.CORE_SHA)
    authenticate_manifest(service.encode(inventory()))
    core.write(output / "download-result.json", report)
    return report


def download_child(config_path):
    service = load_service()
    config_path = service.ordinary(config_path)
    service.require(config_path.stat().st_size <= 65536, "Bounded child configuration required")
    config = service.parse(config_path.read_bytes())
    output = service.ordinary(config["output"])
    try:
        service.require(
            sys.executable == INTERPRETER
            and sys.version_info[:2] == (3, 13)
            and sys.flags.no_user_site
            and sys.dont_write_bytecode,
            "Dedicated Python3.13 -B -s CPU child required",
        )
        service.require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only child visibility required"
        )
        service, staging, core, identities = setup(service.encode(config["source_inventory"]))
        service.require(
            output.is_relative_to(core.OUTPUT_ROOT)
            and output.name == "public-weights-" + config["run_id"],
            "Unique ordinary CPU output required",
        )
        forbidden = {
            "torch",
            "transformers",
            "fla",
            "causal_conv1d",
            "triton",
            "tensorflow",
            "jax",
            "tokenizers",
            "PIL",
        }
        service.require(
            not any(name.split(".")[0] in forbidden for name in sys.modules),
            "No framework/processor may precede public CPU download",
        )

        def no_framework(event, arguments):
            if event == "import" and arguments[0].split(".")[0] in forbidden:
                raise RuntimeError(
                    "Framework/processor import forbidden in public CPU download: " + arguments[0]
                )

        sys.addaudithook(no_framework)
        from huggingface_hub import snapshot_download

        download_models(
            service,
            staging,
            core,
            identities,
            config["mounts"]["/weights"],
            output,
            snapshot_download,
        )
        return 0
    except BaseException as exc:
        try:
            service.save(
                output / "child-failure.json",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "notes": list(getattr(exc, "__notes__", [])),
                },
            )
        except BaseException as secondary:
            exc.add_note("CPU child failure persistence also failed: " + repr(secondary))
        raise


def closed_profile(directory, result, identities):
    service = load_service()
    service.require(
        result["success"] is True
        and result["owned_exit_confirmed"] is True
        and result["export_index"] is not None,
        "Actual successful closed CPU export required",
    )
    directory = service.ordinary(directory)
    originals = {}
    for label in ("export-index.json", "controller-receipt.json", "download-result.json"):
        path = service.ordinary(directory / label)
        service.require(
            path.stat().st_size <= 8 * 1024**2, "Bounded actual closed profile inputs required"
        )
        originals[label] = path.read_bytes()
    service.require(
        service.pin(directory / "export-index.json") == result["export_index"],
        "Externally selected received closure index differs",
    )
    value = {
        "schema_id": service.WEIGHT_SCHEMA,
        "wrapper_sha256": service.WRAPPER_SHA,
        "cpu_downloader_source": service.pin(SCRIPT),
        "closed_download": {
            "index_json": originals["export-index.json"].decode(),
            "controller_json": originals["controller-receipt.json"].decode(),
            "result_json": originals["download-result.json"].decode(),
        },
    }
    blob = service.encode(value)
    sha = hashlib.sha256(blob).hexdigest()
    service.validate_weight_profile(blob, sha, identities)
    return blob, sha


def owned(run_id, manifest_blob):
    service, staging, core, _identities = setup(manifest_blob)
    service.require(
        re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", run_id), "Fresh bounded CPU run ID required"
    )
    output = service.ordinary(core.OUTPUT_ROOT / ("public-weights-" + run_id))
    service.require(not output.exists(), "Fresh CPU download output required")
    output.mkdir(parents=True)
    mounts = {alias: staging.capture_mount(alias) for alias in ("/weights", "/export")}
    config = {
        "run_id": run_id,
        "output": str(output),
        "source_inventory": service.parse(manifest_blob),
        "mounts": mounts,
    }
    core.write(output / "config.json", config)
    core.write(output / "source-inventory.json", service.parse(manifest_blob))
    owner = object.__new__(service.Session)
    owner.core, owner.staging, owner.output = core, staging, output
    owner.marker, owner.blocked, owner.observations = "cpu-" + secrets.token_hex(24), False, []
    owner.process, owner.start_ticks = None, None
    owner.stderr = output / "stderr.log"
    stdout = output / "stdout.log"
    environment = staging.child_environment("download")
    environment["JEV_D3_BROWSER_OWNER"] = owner.marker
    primary, confirmed, export_pin = None, False, None
    started = time.monotonic()
    try:
        with owner.stderr.open("xb") as error_log, stdout.open("xb") as output_log:
            owner.process = subprocess.Popen(
                [
                    INTERPRETER,
                    "-B",
                    "-s",
                    str(SCRIPT),
                    "--download-child",
                    str(output / "config.json"),
                ],
                cwd=ROOT,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=error_log,
                start_new_session=True,
                bufsize=0,
            )
            owner.process.stdin.close()
            owner.start_ticks = staging.proc_record(Path("/proc") / str(owner.process.pid))[
                "start_ticks"
            ]
            while True:
                ready, _, _ = select.select([owner.process.stdout], [], [], 0.1)
                if ready:
                    chunk = os.read(owner.process.stdout.fileno(), 65536)
                    if chunk:
                        output_log.write(chunk)
                        output_log.flush()
                    elif owner.process.poll() is not None:
                        break
                service.require(
                    owner.stderr.stat().st_size <= MAX_LOG and stdout.stat().st_size <= MAX_LOG,
                    "Bounded CPU child logs exceeded",
                )
                service.require(
                    time.monotonic() - started < DEADLINE, "CPU public download watchdog exceeded"
                )
                if owner.process.poll() is not None and not ready:
                    break
            service.require(owner.process.wait(timeout=5) == 0, "CPU public download failed")
            service.require(
                owner.stderr.stat().st_size <= MAX_LOG and stdout.stat().st_size <= MAX_LOG,
                "Final CPU child log bound exceeded",
            )
    except BaseException as exc:
        primary = exc
    finally:
        try:
            confirmed = owner.reap() if owner.process is not None else True
            service.require(
                not owner.blocked, "Unknown owned CPU process was observed; profile refused"
            )
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note("Whole CPU process closure also failed: " + repr(exc))
        try:
            authenticate_manifest(manifest_blob)
            core.authenticate_source(service.CORE_SHA)
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note("Final source authentication also failed: " + repr(exc))
        try:
            weights.commit()
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note("Public cache publication also failed: " + repr(exc))
        controller = {
            "child_success": primary is None and confirmed,
            "owned_exit_confirmed": confirmed,
            "returncode": getattr(owner, "last_returncode", None),
            "source_inventory_sha256": hashlib.sha256(manifest_blob).hexdigest(),
            "downloader_source": service.pin(SCRIPT),
            "provider_mounts": mounts,
            "elapsed_seconds": time.monotonic() - started,
            "GPU_requested": False,
            "model_loads": 0,
            "error": None
            if primary is None
            else {
                "type": type(primary).__name__,
                "message": str(primary),
                "notes": list(getattr(primary, "__notes__", [])),
            },
        }
        try:
            core.write(output / "controller-receipt.json", controller, terminal=True)
            if confirmed:
                core.write(
                    output / "export-index.json",
                    {
                        "root": output.name,
                        "files": staging.tree_inventory(output),
                        "complete_owned_exit": True,
                    },
                    terminal=True,
                )
            export_pin = staging.publish_output(output, mounts["/export"], confirmed)
            results.commit()
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note("CPU full output export also failed: " + repr(exc))
    result = {
        "success": primary is None and confirmed,
        "owned_exit_confirmed": confirmed,
        "root": output.name,
        "export_index": export_pin,
        "GPU_requested": False,
        "model_loads": 0,
        "error": None
        if primary is None
        else {
            "type": type(primary).__name__,
            "message": str(primary),
            "notes": list(getattr(primary, "__notes__", [])),
        },
    }
    if primary is not None and not isinstance(primary, Exception):
        raise primary
    return result


def prepare(destination):
    service = load_service()
    destination = service.ordinary(destination)
    service.require(not destination.exists(), "Fresh CPU source-only preparation required")
    destination.mkdir(parents=True)
    service.save(destination / "source-inventory.json", inventory())
    service.save(
        destination / "plan.json",
        {
            "stage": "prospective_not_executed",
            "source_files": 18,
            "service_source_files": 17,
            "GPU_requested": False,
            "model_loads": 0,
            "token": False,
            "public_models": list(service.MODELS),
            "weights_uploaded": False,
            "collection_or_parent_uploaded": False,
            "actual_profile": "Generated only after a successful actual owned CPU download and SHA-verified full export",
        },
    )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--download-child":
        raise SystemExit(download_child(sys.argv[2]))
    if sys.argv[1:2] == ["prepare"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("prepare")
        parser.add_argument("--output", required=True)
        prepare(parser.parse_args().output)
        raise SystemExit(0)

import modal  # noqa: E402 -- stdlib-only preparation/CPU child exits before SDK.

from apps.modal import decision3_public_replay as reviewed  # noqa: E402

app = modal.App("decision3-public-source18-cpu-weights", include_source=False)
weights, results = reviewed.weights, reviewed.results
image = reviewed.image.env({"PYTHONPATH": str(SCRIPT.parent) + os.pathsep + str(ROOT)})
for row in inventory()["rows"]:
    image = image.add_local_file(row["path"], row["path"], copy=True)


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    ephemeral_disk=524288,
    timeout=15000,
    max_containers=1,
    retries=0,
    single_use_containers=True,
    include_source=False,
    volumes={"/weights": weights, "/export": results},
)
def download(run_id: str, manifest_blob: bytes):
    return owned(run_id, manifest_blob)


@app.local_entrypoint()
def main(output: str, manifest: str, run_id: str, run: bool = False):
    service = load_service()
    service.require(run, "Explicit --run after source18/public-download review required")
    destination = service.ordinary(output)
    service.require(not destination.exists(), "Fresh full CPU download export required")
    manifest_path = service.ordinary(manifest)
    service.require(
        manifest_path.stat().st_size <= 65536, "Bounded named-source18 inventory required"
    )
    manifest_blob = manifest_path.read_bytes()
    authenticate_manifest(manifest_blob)
    result = download.remote(run_id, manifest_blob)
    destination.mkdir(parents=True)
    service.save(destination / "rpc-result.json", result)
    service.require(
        result["export_index"] is not None, "Unknown CPU exit; partial provider evidence only"
    )
    reviewed.export(result, destination / "closed-download")
    service.require(result["success"], "Public CPU staging failed; complete raw export retained")
    _service, _staging, _core, identities = setup(manifest_blob)
    blob, sha = closed_profile(destination / "closed-download", result, identities)
    with (destination / "complete-weight-profile.json").open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    service.save(
        destination / "complete-weight-profile-pin.json",
        {
            "sha256": sha,
            "size_bytes": len(blob),
            "stage": "actual_closed_cpu_download",
            "GPU_requested": False,
            "model_loads": 0,
        },
    )
