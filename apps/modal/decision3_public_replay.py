"""Named public07 inputs and six pinned Decision3 models on one H200.

Review source and a local exact inventory first (no Modal import/upload):
  .venv/bin/python apps/modal/decision3_public_replay.py --inventory-only NEW.json
Only after review, use the Modal entrypoint with --run and that manifest.
The input collection is an authorized local derived artifact, not a public HF
model. No Gemma image, overlay, helper, training or browser rollout is used.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
SCRIPT = ROOT / "apps/modal/decision3_public_replay.py"
CORE = ROOT / "scripts/jev_decision3_public_dev22_gpu_replay.py"
CORE_SHA = "69baedec7efeb7b361ef0b9f8d3c7fdcc70aac30cc19ffba3a74b2a88f54a029"
SPEC = ROOT / "src/s1/evaluation/decision3.py"
SPEC_SHA = "aaa21ab6f2ecf5ae0e00694b43755541dc75fc0310ba10d945ade2a4511add52"
PUBLIC = ROOT / "artifacts/jev-browser/20261016-public-native-training-collection-07"
ADMISSION = ROOT / "artifacts/jev-browser/20261016-public-native-training-07-admission-01"
ANCHORS = {
    str(CORE): CORE_SHA,
    str(
        ADMISSION / "receipt.json"
    ): "bb799a48e8bbf1fb0af5afaf23cb38fb09bb67c3d26c833e96ba33bc34b0fcdb",
    str(
        ADMISSION / "admission-provenance.json"
    ): "d80e5106ab67375403f1b3b5ab9132ae7f212a134532d0d0d422ac044a648c80",
}
HUB = ROOT / "artifacts/jev-local-hf/hub"
DECLARED_MOUNTS = ("/input", "/weights", "/export")
INTERPRETER = "/opt/decision3/bin/python"
MODELS = ("d3", "d3-flash", "d3-mini", "d3-nano", "d3-lite", "d3-edge")
GIB, MAX_FILES, MAX_LOG = 1024**3, 4096, 16 * 1024**2
UPLOAD_LIMIT = 128 * 1024**2
PACKAGES = {
    "torch": "2.10.0+cu130",
    "torchvision": "0.25.0+cu130",
    "transformers": "5.17.0",
    "tokenizers": "0.23.3",
    "accelerate": "1.15.0",
    "safetensors": "0.8.0",
    "huggingface_hub": "1.33.0",
    "pillow": "12.3.0",
    "einops": "0.8.2",
    "pydantic": "2.13.4",
    "httpx": "0.28.1",
    "numpy": "2.5.2",
    "flash-linear-attention": "0.5.2",
    "fla-core": "0.5.2",
    "causal-conv1d": "1.7.0",
    "triton": "3.6.0",
}
WHEELS = {
    "torch": "https://download-r2.pytorch.org/whl/cu130/torch-2.10.0%2Bcu130-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=7e0d9922e9e91f780b2761a0c5ebac3c15c9740bab042e1b59149afa6d6474eb",
    "torchvision": "https://download-r2.pytorch.org/whl/cu130/torchvision-0.25.0%2Bcu130-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=e85337d59bdbf7006cd0c76012da3663ded5606dadc68f72280d9f2ab1e9191a",
    "causal-conv1d": "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl#sha256=e03db18b70006574a8f4206297cdeedd55d2ae1284034cb60ef9d02c4c014696",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def parse(blob):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate JSON key")
            value[key] = item
        return value

    return json.loads(
        blob,
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )


def ordinary(path):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary confined path required",
    )
    return path


def capture_mount(alias):
    # Only these declared provider boundaries may resolve symlinks. Descendant
    # data/output paths still pass the unchanged ordinary-path checks.
    require(alias in DECLARED_MOUNTS, "Declared provider mount required")
    root = ordinary(Path(alias).resolve(strict=True))
    require(root.is_dir(), "Provider mount directory required")
    before = root.stat()
    binding = {"alias": alias, "root": str(root), "device": before.st_dev, "inode": before.st_ino}
    require(
        Path(alias).resolve(strict=True) == root
        and (root.stat().st_dev, root.stat().st_ino) == (before.st_dev, before.st_ino),
        "Provider mount changed during capture",
    )
    return binding


def mount_root(binding):
    require(capture_mount(binding["alias"]) == binding, "Provider mount alias/identity changed")
    return ordinary(binding["root"])


def relative(name):
    require(type(name) is str and bool(name), "Named relative path required")
    path = PurePosixPath(name)
    require(
        str(path) == name
        and not path.is_absolute()
        and ".." not in path.parts
        and not any(
            part in {".git", ".aws", ".ssh", ".codex", ".agents"} or part.startswith(".env")
            for part in path.parts
        ),
        "Confined named task path required",
    )
    return path


def read(path, limit=16 * 1024**2):
    path = ordinary(path)
    require(path.is_file() and path.stat().st_size <= limit, "Bounded ordinary input required")
    blob = path.read_bytes()
    require(len(blob) <= limit, "Input byte bound")
    return blob


def pin(path, limit=64 * 1024**2):
    path = ordinary(path)
    before, count, digest = path.stat(), 0, hashlib.sha256()
    require(stat.S_ISREG(before.st_mode) and before.st_size <= limit, "Bounded named file required")
    with path.open("rb") as stream:
        while chunk := stream.read(1024**2):
            count += len(chunk)
            require(count <= before.st_size, "Input grew while hashing")
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        and count == before.st_size,
        "Input changed while hashing",
    )
    return {"sha256": digest.hexdigest(), "size_bytes": count}


def create(path, blob):
    path = ordinary(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())


def stream_copy(source, destination, expected):
    source, destination = ordinary(source), ordinary(destination)
    before, count, digest = source.stat(), 0, hashlib.sha256()
    require(
        stat.S_ISREG(before.st_mode) and before.st_size == expected["size_bytes"],
        "Exact ordinary copy source required: " + str(source),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        while chunk := incoming.read(1024**2):
            count += len(chunk)
            require(count <= expected["size_bytes"], "Copy source grew")
            outgoing.write(chunk)
            digest.update(chunk)
        after = os.fstat(incoming.fileno())
        outgoing.flush()
        os.fsync(outgoing.fileno())
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        and {"sha256": digest.hexdigest(), "size_bytes": count} == expected,
        "Streamed ordinary copy identity/SHA/size differs: " + str(source),
    )


def authority(path, sha):
    blob = read(path)
    require(hashlib.sha256(blob).hexdigest() == sha, "Fixed named authority differs: " + str(path))
    return parse(blob)


def literal(path, name):
    values = [
        ast.literal_eval(node.value)
        for node in ast.parse(read(path)).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    require(len(values) == 1, "One fixed literal declaration required")
    return values[0]


def source_pins():
    require(pin(CORE)["sha256"] == CORE_SHA, "Reviewed replay source differs")
    pins = literal(CORE, "SOURCE_PINS")
    require(
        len(pins) == 12 and pins["src/s1/evaluation/decision3.py"] == SPEC_SHA,
        "Fixed source12 required",
    )
    return pins


def expected_upload_manifest():
    rows = {}

    def add(path, expected, classification):
        path = ordinary(path)
        require(
            path.is_relative_to(ROOT) and path.relative_to(ROOT).parts,
            "Named ROOT destination required",
        )
        label = path.relative_to(ROOT).as_posix()
        relative(label)
        require(
            label not in rows
            and set(expected) == {"sha256", "size_bytes"}
            and type(expected["size_bytes"]) is int
            and 0 <= expected["size_bytes"] <= 64 * 1024**2,
            "Unique bounded named upload required",
        )
        rows[label] = {"path": str(path), "pin": expected, "classification": classification}

    for label, sha in source_pins().items():
        actual = pin(ROOT / relative(label))
        require(actual["sha256"] == sha, "Production source12 differs")
        add(ROOT / label, actual, "reviewed_public_github_source12")
    add(CORE, pin(CORE), "reviewed_public_github_replay_source")
    add(SCRIPT, pin(SCRIPT), "reviewed_new_modal_wrapper_source")
    for path, sha in ANCHORS.items():
        if Path(path) != CORE:
            authority(Path(path), sha)
            add(Path(path), pin(path), "authorized_local_derived_admission_metadata")
    provenance = authority(
        ADMISSION / "admission-provenance.json",
        ANCHORS[str(ADMISSION / "admission-provenance.json")],
    )
    closed = provenance["closed_inventory"]
    require(
        len(closed["files"]) == 701
        and sum(p["size_bytes"] for p in closed["files"].values()) == 105415940
        and closed["identity_sha256"]
        == "d74f729460c648b8c88cf677bc379d1ae34fe7530a9651810e8af2825d449742",
        "Original authorized public701 required",
    )
    for label, expected in closed["files"].items():
        add(PUBLIC / relative(label), expected, "authorized_local_derived_public_page_collection")
    ordered = [rows[label] for label in sorted(rows)]
    require(
        len(ordered) == 717 and sum(row["pin"]["size_bytes"] for row in ordered) <= UPLOAD_LIMIT,
        "Exact717 named upload byte/file bound",
    )
    return {
        "schema_id": "decision3-public07-modal-named-upload-v1",
        "rows": ordered,
        "files": len(ordered),
        "bytes": sum(row["pin"]["size_bytes"] for row in ordered),
        "replay_sha256": CORE_SHA,
        "credentials_uploaded": False,
        "weights_uploaded": False,
        "scope": "Authorized public701/admission/source12 subset plus reviewed replay/wrapper; no repository recursion, parent adapter, Gemma reference or local weights",
    }


def load_core():
    require(pin(CORE)["sha256"] == CORE_SHA, "Reviewed replay before import")
    sys.path.insert(0, str(ROOT / "scripts"))
    module = importlib.import_module("jev_decision3_public_dev22_gpu_replay")
    require(
        Path(module.__file__).resolve() == CORE and module.MODEL_ORDER == MODELS,
        "Exact replay module origin/order",
    )
    return module


def materialize(config):
    binding = config["mounts"]["/input"]
    base = mount_root(binding) / ("d3-" + config["run_id"])
    blob = read(base / "manifest.json", 8 * 1024**2)
    require(
        hashlib.sha256(blob).hexdigest() == config["manifest_sha256"],
        "Reviewed external manifest SHA",
    )
    manifest = parse(blob)
    rows = manifest["rows"]
    require(
        type(rows) is list
        and len(rows) == 717
        and sum(row["pin"]["size_bytes"] for row in rows) <= UPLOAD_LIMIT,
        "Bounded named manifest",
    )
    mapping = {row["path"]: (index, row) for index, row in enumerate(rows)}
    require(len(mapping) == len(rows), "Unique named manifest paths")

    def put(path, sha):
        index, row = mapping[str(path)]
        source = base / "files" / f"{index:05d}"
        require(
            row["pin"]["sha256"] == sha and pin(source) == row["pin"],
            "Named uploaded authority differs",
        )
        target = ordinary(path)
        require(target.is_relative_to(ROOT), "Named ROOT authority destination")
        if target.exists():
            require(pin(target) == row["pin"], "Existing image authority differs")
        else:
            create(target, read(source, 64 * 1024**2))

    for path, sha in ANCHORS.items():
        put(Path(path), sha)
    for label, sha in source_pins().items():
        put(ROOT / relative(label), sha)
    require(
        manifest == expected_upload_manifest() and blob == encode(manifest),
        "Remote manifest differs from fixed authorities",
    )
    for index, row in enumerate(rows):
        source, target = base / "files" / f"{index:05d}", ordinary(row["path"])
        require(
            target.is_relative_to(ROOT) and pin(source) == row["pin"],
            "Named upload bytes/destination differ",
        )
        if target.exists():
            require(pin(target) == row["pin"], "Never overwrite differing installed/image input")
        else:
            create(target, read(source, 64 * 1024**2))
    core = load_core()
    identities = core.authenticate_source(CORE_SHA)
    core.authenticate_inputs()
    mount_root(binding)
    return core, identities


def dependency_commands(constraints):
    command = [
        INTERPRETER,
        "-m",
        "pip",
        "--isolated",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--only-binary=:all:",
        "--timeout",
        "60",
        "--retries",
        "1",
        "--index-url",
        "https://pypi.org/simple",
        "--constraint",
        str(constraints),
    ]
    return [
        [
            *command,
            *WHEELS.values(),
            *[name + "==" + version for name, version in PACKAGES.items() if name not in WHEELS],
        ],
        [INTERPRETER, "-m", "pip", "check"],
    ]


def install_dependencies():
    require(
        sys.executable == INTERPRETER and sys.version_info[:2] == (3, 13),
        "Dedicated Python3.13 installer",
    )
    with tempfile.TemporaryDirectory() as temporary:
        constraints = Path(temporary) / "constraints.txt"
        constraints.write_text(
            "".join(name + "==" + version + "\n" for name, version in PACKAGES.items())
        )
        environment = {**os.environ, "PYTHONPATH": "", "PYTHONNOUSERSITE": "1"}
        for command in dependency_commands(constraints):
            subprocess.run(command, env=environment, check=True, timeout=2400)
    for name, version in PACKAGES.items():
        actual = importlib.metadata.version(name)
        require(
            actual == version or name == "causal-conv1d" and actual.split("+")[0] == version,
            "Pinned D3 package version differs: " + name,
        )
    return 0


def child_environment(mode):
    environment = {
        "PATH": "/opt/decision3/bin:/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "PYTHONPATH": str(ROOT / "src") + ":" + str(ROOT / "scripts"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "HF_HUB_CACHE": str(HUB),
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_TOKEN_PATH": "/nonexistent-d3-public-token",
        "HF_HOME": "/tmp/decision3-public-hf",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
        if name in os.environ:
            environment[name] = os.environ[name]
    if mode == "replay":
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    else:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


def replay_output(run_id):
    return ROOT / "artifacts/jev-d3-native" / ("modal-d3-" + run_id)


def provider_snapshot(binding, identity):
    return ordinary(
        mount_root(binding)
        / ("models--" + identity["model_id"].replace("/", "--"))
        / "snapshots"
        / identity["revision"]
    )


def provider_cache_file(binding, identity, label):
    require(len(relative(label).parts) == 1, "Flat immutable cache filename required")
    hub, snapshot = mount_root(binding), provider_snapshot(binding, identity)
    source = ordinary((snapshot / label).resolve(strict=True))
    require(
        source.is_relative_to(snapshot.parent.parent) or source.is_relative_to(hub / "blobs"),
        "Pinned runtime file escaped its provider repository/shared blobs",
    )
    return source


def runtime_files(manifest):
    names = sorted(
        label
        for label in manifest["files_sha256"]
        if label not in {"LICENSE", "NOTICE", "requirements.txt", ".gitattributes"}
        and not label.endswith(".md")
        and not label.startswith(("assets/", "LICENSES/"))
    )
    require(len(names) <= 512, "Bounded publisher runtime file set")
    for label in names:
        require(len(relative(label).parts) == 1, "Flat publisher runtime file required")
    return names


def stage_cache(config, core, identities):
    # The core's fixed HUB must remain an ordinary directory. Copy only exact
    # pinned runtime files, never relaxing its symlink or RAM admission rules.
    binding, output = config["mounts"]["/weights"], Path(config["output"])
    ordinary(HUB)
    require(not HUB.exists() or not any(HUB.iterdir()), "Fresh ordinary fixed HF cache required")
    report = {"provider_mount": binding, "fixed_hub": str(HUB), "models": {}}
    create(
        output / "cache-staging-plan.json",
        encode({"models": identities, "provider_mount": binding, "fixed_hub": str(HUB)}),
    )
    for name in MODELS:
        identity = identities[name]
        source = provider_cache_file(binding, identity, "MODEL_MANIFEST.json")
        manifest = core.read(source, identity["manifest_sha256"])
        require(
            manifest["model_name"] == name and manifest["repo_id"] == identity["model_id"],
            "Pinned provider manifest identity differs",
        )
        target = (
            HUB
            / ("models--" + identity["model_id"].replace("/", "--"))
            / "snapshots"
            / identity["revision"]
        )
        manifest_pin = core.pin(source)
        require(manifest_pin["sha256"] == identity["manifest_sha256"], "Publisher manifest SHA")
        files = {"MODEL_MANIFEST.json": manifest_pin}
        stream_copy(source, target / "MODEL_MANIFEST.json", manifest_pin)
        for label in runtime_files(manifest):
            expected = {
                "sha256": manifest["files_sha256"][label],
                "size_bytes": manifest["files_bytes"][label],
            }
            source = provider_cache_file(binding, identity, label)
            stream_copy(source, target / label, expected)
            files[label] = expected
        mount_root(binding)
        report["models"][name] = {"snapshot": str(target), "files": files}
        create(output / (name + "-cache-staging.json"), encode(report["models"][name]))
    create(output / "cache-staging-result.json", encode(report))


def download_child(config):
    core, identities = materialize(config)
    from huggingface_hub import snapshot_download

    output, binding = Path(config["output"]), config["mounts"]["/weights"]
    remote_hub = mount_root(binding)
    report = {
        "models": {},
        "token": False,
        "model_loads": 0,
        "GPU_requested": False,
        "provider_mount": binding,
        "resources_before": core.resource_snapshot(),
        "cache_disk_capacity": {
            "observed": False,
            "usable_capacity_evidence": False,
            "scope": "Modal Volume statfs is a placeholder; download/write failures are retained",
        },
    }
    for name in MODELS:
        identity = identities[name]
        snapshot = Path(
            snapshot_download(
                repo_id=identity["model_id"],
                revision=identity["revision"],
                cache_dir=str(remote_hub),
                token=False,
                allow_patterns=["MODEL_MANIFEST.json"],
                max_workers=1,
            )
        )
        expected = provider_snapshot(binding, identity)
        require(snapshot == expected, "Exact immutable public HF cache path")
        manifest = core.read(
            provider_cache_file(binding, identity, "MODEL_MANIFEST.json"),
            identity["manifest_sha256"],
        )
        require(
            manifest["model_name"] == name and manifest["repo_id"] == identity["model_id"],
            "Publisher manifest identity",
        )
        names = runtime_files(manifest)
        for label in names:
            if (snapshot / label).exists():
                provider_cache_file(binding, identity, label)
        snapshot_download(
            repo_id=identity["model_id"],
            revision=identity["revision"],
            cache_dir=str(remote_hub),
            token=False,
            allow_patterns=["MODEL_MANIFEST.json", *names],
            max_workers=1,
        )
        files = {}
        # Streaming authentication only: no framework/tensor load or new memory
        # admission policy. The unchanged core performs its full gates/headers on GPU.
        for label in names:
            files[label] = core.pin(provider_cache_file(binding, identity, label))
            require(
                files[label]
                == {
                    "sha256": manifest["files_sha256"][label],
                    "size_bytes": manifest["files_bytes"][label],
                },
                "Pinned public cache bytes differ",
            )
        report["models"][name] = {
            "revision": identity["revision"],
            "manifest_sha256": identity["manifest_sha256"],
            "snapshot": str(snapshot),
            "files": files,
        }
        create(output / (name + "-download.json"), encode(report["models"][name]))
        mount_root(binding)
    report["resources_after"] = core.resource_snapshot()
    create(output / "download-result.json", encode(report))
    return 0


def replay_child(config):
    core, identities = materialize(config)
    require(sys.executable == INTERPRETER, "Actual dedicated D3 interpreter required")
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,memory.free,memory.used,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    require(
        len(gpu.stdout) <= 65536
        and b"H200" in gpu.stdout
        and len(gpu.stdout.decode().strip().splitlines()) == 1,
        "Actual single H200 required",
    )
    target = replay_output(config["run_id"])
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), "Fresh fixed-root replay output")
    create(
        Path(config["output"]) / "replay-preflight.json",
        encode(
            {
                "gpu": gpu.stdout.decode(),
                "interpreter": sys.executable,
                "pythonpath": os.environ.get("PYTHONPATH"),
                "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
                "resources": core.resource_snapshot(),
                "replay_sha256": CORE_SHA,
            }
        ),
    )
    stage_cache(config, core, identities)

    # Let the core's finally reap its model session before the outer watchdog escalates.
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("Owned Modal D3 replay termination")

    signal.signal(signal.SIGTERM, interrupted)
    result = core.controller(SimpleNamespace(source_sha256=CORE_SHA, output_dir=target))
    create(
        Path(config["output"]) / "replay-child-result.json",
        encode(
            {
                "returncode": result,
                "receipt": pin(target / "receipt.json"),
                "closed_inventory": pin(target / "closed-inventory.json"),
            }
        ),
    )
    return result


def child(mode, config_path):
    config = parse(read(config_path))
    try:
        require(sys.executable == INTERPRETER, "Dedicated D3 Python required")
        return download_child(config) if mode == "download" else replay_child(config)
    except BaseException as exc:
        try:
            create(
                Path(config["output"]) / "child-failure.json",
                encode(
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "notes": list(getattr(exc, "__notes__", [])),
                    }
                ),
            )
        except BaseException as secondary:
            exc.add_note("Child failure persistence: " + repr(secondary))
        raise


def group_alive(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    return True


def remember_model_groups(run_id, groups, *, force=False):
    # The frozen controller permits one model session at a time. Once its
    # group is known, avoid repeatedly parsing the growing raw-result receipt
    # while that arm is running on the same two allocated CPUs.
    if not force and any(
        value.get("model") is not None and group_alive(pid) for pid, value in groups.items()
    ):
        return
    receipt = replay_output(run_id) / "receipt.json"
    if not receipt.exists():
        return
    report = parse(read(receipt))
    require(report.get("source_sha256") == CORE_SHA, "Owned core receipt identity")
    for arm in report.get("arms", []):
        pid, command = arm.get("pid"), arm.get("command")
        if pid is None or pid in groups:
            continue
        require(
            type(pid) is int
            and pid > 1
            and arm["model"] in MODELS
            and command[:4] == [INTERPRETER, "-B", "-s", str(CORE)]
            and command[command.index("--worker") + 1] == arm["model"]
            and command[command.index("--output-dir") + 1]
            == str(replay_output(run_id) / arm["model"]),
            "Only this unique replay's declared model sessions may be tracked",
        )
        if not group_alive(pid):
            continue
        path = Path("/proc") / str(pid) / "cmdline"
        try:
            actual = path.read_bytes().rstrip(b"\x00").split(b"\x00")
            require(
                actual == [item.encode() for item in command] and os.getpgid(pid) == pid,
                "Actual owned model command/session differs",
            )
        except (FileNotFoundError, ProcessLookupError):
            if not group_alive(pid):
                continue
            # A reaped leader can leave compiler descendants in its declared
            # group. The exact source-pinned controller receipt owns that PID.
        groups[pid] = {"model": arm["model"], "command": command}


def proc_read(path, limit):
    with path.open("rb") as stream:
        blob = stream.read(limit + 1)
    require(len(blob) <= limit, "Bounded process observation required")
    return blob


def proc_record(path):
    raw = proc_read(path / "stat", 65536)
    fields = raw[raw.rindex(b")") + 2 :].split()
    return {
        "parent": int(fields[1]),
        "group": int(fields[2]),
        "session": int(fields[3]),
        "start_ticks": int(fields[19]),
    }


def discover_owned_processes(run_id, groups):
    # The marker survives exec and reparenting. This closes the fork-to-receipt
    # window without trusting arbitrary processes with similar executable names.
    marker = b"JEV_DECISION3_RUN_OUTPUT=" + str(replay_output(run_id)).encode()
    records = {}
    entries = list(Path("/proc").iterdir())
    require(len(entries) <= 65536, "Bounded process observation required")
    for path in entries:
        if not path.name.isdecimal():
            continue
        try:
            records[int(path.name)] = proc_record(path)
        except (FileNotFoundError, ProcessLookupError):
            continue
    starts = [value.get("start_ticks") for value in groups.values()]
    floor = min((value for value in starts if value is not None), default=None)
    owned = {pid for pid, row in records.items() if row["group"] in groups}
    changed = True
    while changed:
        added = {pid for pid, row in records.items() if row["parent"] in owned} - owned
        changed = bool(added)
        owned.update(added)
    uncertain = []
    for pid, row in records.items():
        if pid not in owned and floor is not None and row["start_ticks"] < floor:
            continue
        try:
            environment = proc_read(Path("/proc") / str(pid) / "environ", 1024**2)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # A recent unreadable process could be an unjournaled orphan.
            uncertain.append({"pid": pid, "reason": "Process ownership unreadable"})
            continue
        if marker in environment.split(b"\x00"):
            owned.add(pid)
    require(len(owned) + len(uncertain) <= 4096, "Bounded owned process set required")
    unknown = list(uncertain)
    for pid in sorted(owned):
        row = records[pid]
        if row["group"] in groups:
            continue
        try:
            raw = proc_read(Path("/proc") / str(pid) / "cmdline", 1024**2)
            command = [item.decode() for item in raw.rstrip(b"\x00").split(b"\x00")]
        except (FileNotFoundError, ProcessLookupError):
            continue
        model = next(
            (
                name
                for name in MODELS
                if len(command) == 14
                and command[:10]
                == [
                    INTERPRETER,
                    "-B",
                    "-s",
                    str(CORE),
                    "--source-sha256",
                    CORE_SHA,
                    "--worker",
                    name,
                    "--output-dir",
                    str(replay_output(run_id) / name),
                ]
                and command[10] == "--expected-boot-id"
                and re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", command[11])
                and command[12] == "--expected-memtotal"
                and command[13].isdecimal()
                and int(command[13]) > 0
            ),
            None,
        )
        if model is not None and row["group"] == row["session"] == pid:
            groups[pid] = {
                "model": model,
                "command": command,
                "start_ticks": row["start_ticks"],
                "source": "actual_owned_process_before_or_after_receipt",
            }
        else:
            unknown.append({"pid": pid, "group": row["group"], "reason": "Unknown owned session"})
    return {"observed_processes": len(records), "owned_pids": sorted(owned), "unknown": unknown}


def stop_owned(process, groups, run_id=None):
    # New model sessions can appear during TERM handling. Refresh both the receipt
    # and actual processes throughout cleanup, and again after all creators exit.
    failure, observation = None, None
    observations = {"passes": 0, "history_limit": 16, "history_truncated": False, "history": []}

    def refresh():
        nonlocal failure, observation
        if run_id is None:
            return
        try:
            remember_model_groups(run_id, groups, force=True)
        except BaseException as exc:
            if failure is None:
                failure = exc
            else:
                failure.add_note("Further owned receipt discovery: " + repr(exc))
        try:
            observation = discover_owned_processes(run_id, groups)
            observations["passes"] += 1
            history = observations["history"]
            if observation != (history[-1] if history else None):
                if len(history) < observations["history_limit"]:
                    history.append(observation)
                else:
                    observations["history_truncated"] = True
                    history[-1] = observation
        except BaseException as exc:
            if failure is None:
                failure = exc
            else:
                failure.add_note("Further owned process discovery: " + repr(exc))

    for sig in (signal.SIGTERM, signal.SIGKILL):
        sent, until = set(), time.monotonic() + 10
        while True:
            refresh()
            for pid in list(groups):
                if pid not in sent and group_alive(pid):
                    try:
                        os.killpg(pid, sig)
                    except ProcessLookupError:
                        pass
                    sent.add(pid)
            process.poll()
            active = any(group_alive(pid) for pid in groups)
            if not active and process.returncode is not None:
                # No tracked process can create another child after this point.
                refresh()
                if not any(group_alive(pid) for pid in groups) and (
                    failure is not None or not (observation and observation["unknown"])
                ):
                    break
            if time.monotonic() >= until:
                break
            time.sleep(0.05)
    process.poll()
    refresh()
    try:
        if failure is not None:
            raise failure
        require(
            process.returncode is not None
            and not any(group_alive(pid) for pid in groups)
            and not (observation and observation["unknown"]),
            "Owned wrapper/model process-group exit unconfirmed; no next stage",
        )
    except BaseException as exc:
        exc.owned_process_observations = observations
        raise
    return observations


def tree_inventory(root):
    found, pending, total, entries = {}, [ordinary(root)], 0, 0
    while pending:
        with os.scandir(pending.pop()) as children:
            for entry in children:
                entries += 1
                require(entries <= MAX_FILES, "Output traversal cap")
                mode = entry.stat(follow_symlinks=False).st_mode
                require(not stat.S_ISLNK(mode), "Output links rejected")
                if stat.S_ISDIR(mode):
                    pending.append(Path(entry.path))
                else:
                    require(stat.S_ISREG(mode), "Ordinary output files only")
                    size = entry.stat(follow_symlinks=False).st_size
                    require(total + size <= GIB, "Whole export byte cap")
                    found[Path(entry.path).relative_to(root).as_posix()] = pin(entry.path, GIB)
                    total += size
    return dict(sorted(found.items()))


def preserve_replay(run_id, output):
    source = replay_output(run_id)
    require(source.exists(), "Replay child produced no raw output tree")
    before = tree_inventory(source)
    destination = output / "replay"
    require(not destination.exists(), "Fresh complete replay export")
    destination.mkdir()
    for name, expected in before.items():
        target = destination / relative(name)
        create(target, read(source / name, GIB))
        require(pin(target, GIB) == expected, "Copied raw replay evidence SHA/size differs")
    require(
        tree_inventory(source) == before and tree_inventory(destination) == before,
        "Complete raw replay tree changed during preservation",
    )


def publish_output(output, binding, confirmed):
    # The provider alias is resolved only at its declared boundary. All ordinary
    # local evidence is copied and verified before the closure index is visible.
    destination = ordinary(mount_root(binding) / output.name)
    require(not destination.exists(), "Fresh provider result directory required")
    destination.mkdir()
    before = tree_inventory(output)
    index = before.get("export-index.json")
    require(index is None or confirmed, "Closure index requires actual owned exit")
    payload = {name: value for name, value in before.items() if name != "export-index.json"}
    for name, expected in payload.items():
        stream_copy(output / relative(name), destination / relative(name), expected)
    require(
        tree_inventory(output) == before and tree_inventory(destination) == payload,
        "Complete provider export payload differs",
    )
    mount_root(binding)
    if index is not None:
        stream_copy(output / "export-index.json", destination / "export-index.json", index)
        require(tree_inventory(destination) == before, "Final provider closure index differs")
    mount_root(binding)
    return index


def owned(mode, run_id, manifest_sha256):
    require(
        mode in {"download", "replay"} and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", run_id),
        "Fresh bounded run ID/mode",
    )
    output = ordinary(ROOT / "artifacts/jev-d3-modal-owned" / ("decision3-" + run_id + "-" + mode))
    require(not output.exists(), "Fresh remote result directory")
    output.mkdir(parents=True)
    process, primary, groups, confirmed, export_pin, report = None, None, {}, False, None, None
    log_sizes, process_observations, mounts = {}, [], {}
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            mounts = {alias: capture_mount(alias) for alias in DECLARED_MOUNTS}
            create(
                output / "child-config.json",
                encode(
                    {
                        "run_id": run_id,
                        "output": str(output),
                        "manifest_sha256": manifest_sha256,
                        "mounts": mounts,
                    }
                ),
            )
            process = subprocess.Popen(
                [
                    INTERPRETER,
                    "-B",
                    "-s",
                    str(SCRIPT),
                    "--" + mode + "-child",
                    str(output / "child-config.json"),
                ],
                env={
                    **child_environment(mode),
                    "JEV_DECISION3_RUN_OUTPUT": str(replay_output(run_id)),
                },
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            groups[process.pid] = {"model": None, "source": "owned_wrapper_child"}
            try:
                groups[process.pid]["start_ticks"] = proc_record(Path("/proc") / str(process.pid))[
                    "start_ticks"
                ]
            except (FileNotFoundError, ProcessLookupError):
                pass
            started = time.monotonic()
            while process.poll() is None:
                if mode == "replay":
                    remember_model_groups(run_id, groups)
                require(
                    max(os.fstat(stdout.fileno()).st_size, os.fstat(stderr.fileno()).st_size)
                    <= MAX_LOG,
                    "Owned logs byte cap",
                )
                require(
                    time.monotonic() - started < (13800 if mode == "download" else 24000),
                    "Owned outer watchdog",
                )
                time.sleep(0.1)
            require(process.returncode == 0, "Owned child failed: " + str(process.returncode))
        except BaseException as exc:
            primary = exc
        finally:
            if process is not None:
                discovered = True
                try:
                    if mode == "replay":
                        remember_model_groups(run_id, groups)
                except BaseException as secondary:
                    discovered = False
                    if primary is None:
                        primary = secondary
                    else:
                        primary.add_note("Owned model-group discovery: " + repr(secondary))
                try:
                    process_observations = stop_owned(
                        process, groups, run_id if mode == "replay" else None
                    )
                    confirmed = discovered
                except BaseException as secondary:
                    process_observations = getattr(secondary, "owned_process_observations", [])
                    if primary is None:
                        primary = secondary
                    else:
                        primary.add_note("Owned group cleanup: " + repr(secondary))
            else:
                confirmed = True  # Setup failed before any worker was created.
            for name, stream in (("stdout.log", stdout), ("stderr.log", stderr)):
                try:
                    size = os.fstat(stream.fileno()).st_size
                    stream.seek(0)
                    create(output / name, stream.read(MAX_LOG))
                    log_sizes[name] = {
                        "actual_bytes": size,
                        "retained_bytes": min(size, MAX_LOG),
                        "truncated": size > MAX_LOG,
                    }
                    require(size <= MAX_LOG, "Final owned logs cap")
                except BaseException as secondary:
                    if primary is None:
                        primary = secondary
                    else:
                        primary.add_note("Bounded log preservation: " + repr(secondary))
    preservation_complete = mode == "download"
    if confirmed and mode == "replay":
        try:
            preserve_replay(run_id, output)
            preservation_complete = replay_output(run_id).exists()
        except BaseException as secondary:
            if primary is None:
                primary = secondary
            else:
                primary.add_note("Full replay preservation: " + repr(secondary))
    if mode == "download":
        try:
            weights.commit()
        except BaseException as secondary:
            if primary is None:
                primary = secondary
            else:
                primary.add_note("Public HF cache publication: " + repr(secondary))
    try:
        report = {
            "child_success": primary is None and confirmed,
            "owned_exit_confirmed": confirmed,
            "replay_preservation_complete": preservation_complete,
            "owned_groups": groups,
            "owned_process_observations": process_observations,
            "provider_mounts": mounts,
            "ordinary_output": str(output),
            "returncode": process.returncode if process is not None else None,
            "scope": "Child exit and complete raw tree preservation; export/volume publication still pending",
            "error": None
            if primary is None
            else {
                "type": type(primary).__name__,
                "message": str(primary),
                "notes": list(getattr(primary, "__notes__", [])),
            },
            "logs": log_sizes,
            "modal_app_id": app.app_id,
            "modal_call_id": modal.current_function_call_id(),
        }
        create(output / "controller-receipt.json", encode(report))
        if confirmed:
            create(
                output / "export-index.json",
                encode(
                    {
                        "root": output.name,
                        "files": tree_inventory(output),
                        "complete_owned_exit": True,
                    }
                ),
            )
    except BaseException as secondary:
        if primary is None:
            primary = secondary
        else:
            primary.add_note("Closed full-tree export: " + repr(secondary))
    try:
        binding = mounts.get("/export") or capture_mount("/export")
        export_pin = publish_output(output, binding, confirmed)
        results.commit()
    except BaseException as secondary:
        if primary is None:
            primary = secondary
        else:
            primary.add_note("Results volume publication: " + repr(secondary))
    if primary is not None and not isinstance(primary, Exception):
        raise primary
    return {
        "success": primary is None and report is not None and report["child_success"],
        "root": output.name,
        "export_index": export_pin,
        "error": None
        if primary is None
        else {
            "type": type(primary).__name__,
            "message": str(primary),
            "notes": list(getattr(primary, "__notes__", [])),
        },
    }


if __name__ == "__main__":
    if sys.argv[1:] == ["--install-dependencies"]:
        raise SystemExit(install_dependencies())
    if len(sys.argv) == 3 and sys.argv[1] in {"--download-child", "--replay-child"}:
        raise SystemExit(child(sys.argv[1][2:-6], sys.argv[2]))
    if len(sys.argv) == 3 and sys.argv[1] == "--inventory-only":
        manifest = expected_upload_manifest()
        require(
            all(pin(row["path"]) == row["pin"] for row in manifest["rows"]),
            "Exact local named upload bytes",
        )
        create(Path(sys.argv[2]), encode(manifest))
        raise SystemExit(0)

import modal  # noqa: E402 -- stdlib inventory/build/worker children exit before SDK import.

app = modal.App("decision3-public07-dev22-replay", include_source=False)
inputs = modal.Volume.from_name("jev-public07-fit-named-inputs-01", create_if_missing=False)
weights = modal.Volume.from_name("jev-public07-fit-public-hf-a66-01", create_if_missing=False)
results = modal.Volume.from_name("jev-public07-fit-results-01", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("build-essential")
    .run_commands("python -m venv /opt/decision3")
    .add_local_file(str(SCRIPT), str(SCRIPT), copy=True)
    .env(
        {"PYTHONPATH": str(SCRIPT.parent), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
    )
    .run_commands(shlex.join([INTERPRETER, "-B", "-s", str(SCRIPT), "--install-dependencies"]))
)


@app.function(
    image=image,
    cpu=2,
    memory=262144,
    ephemeral_disk=524288,
    timeout=14400,
    max_containers=1,
    retries=0,
    single_use_containers=True,
    include_source=False,
    volumes={
        "/weights": weights,
        "/input": inputs.with_mount_options(read_only=True),
        "/export": results,
    },
)
def download(run_id: str, manifest_sha256: str):
    return owned("download", run_id, manifest_sha256)


@app.function(
    image=image,
    gpu="H200",
    cpu=2,
    memory=262144,
    ephemeral_disk=524288,
    timeout=25200,
    max_containers=1,
    retries=0,
    single_use_containers=True,
    include_source=False,
    block_network=True,
    volumes={
        "/weights": weights.with_mount_options(read_only=True),
        "/input": inputs.with_mount_options(read_only=True),
        "/export": results,
    },
)
def replay(run_id: str, manifest_sha256: str):
    return owned("replay", run_id, manifest_sha256)


def receive(remote, destination, expected, limit=GIB):
    require(
        expected["size_bytes"] <= limit and not destination.exists(),
        "Fresh bounded streamed export",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest, count = hashlib.sha256(), 0
    with destination.open("xb") as stream:
        for blob in results.read_file(remote):
            count += len(blob)
            require(count <= expected["size_bytes"], "Stream exceeded declared output size")
            stream.write(blob)
            digest.update(blob)
        stream.flush()
        os.fsync(stream.fileno())
    require(
        {"sha256": digest.hexdigest(), "size_bytes": count} == expected,
        "Streamed export SHA/size differs",
    )


def export(result, destination):
    require(result["export_index"] is not None, "Remote tree not closed; no next worker")
    require(len(relative(result["root"]).parts) == 1, "One confined export root basename")
    index_path = destination / "export-index.json"
    receive(result["root"] + "/export-index.json", index_path, result["export_index"], 8 * 1024**2)
    index = parse(read(index_path, 8 * 1024**2))
    require(
        index["root"] == result["root"]
        and index["complete_owned_exit"] is True
        and len(index["files"]) <= MAX_FILES
        and sum(p["size_bytes"] for p in index["files"].values()) <= GIB,
        "Closed bounded full export index",
    )
    for name, expected in index["files"].items():
        receive(result["root"] + "/" + str(relative(name)), destination / relative(name), expected)


@app.local_entrypoint()
def main(output: str, manifest: str, run_id: str, run: bool = False):
    require(
        run and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", run_id),
        "Explicit --run and unique bounded run ID after source/manifest review",
    )
    destination = ordinary(output)
    require(not destination.exists(), "Fresh local full export destination")
    supplied, expected = read(manifest, 8 * 1024**2), expected_upload_manifest()
    require(supplied == encode(expected), "Reviewed exact named manifest bytes")
    require(
        all(pin(row["path"]) == row["pin"] for row in expected["rows"]),
        "Named upload bytes changed",
    )
    destination.mkdir(parents=True)
    with inputs.batch_upload(force=False) as batch:
        for index, row in enumerate(expected["rows"]):
            batch.put_file(row["path"], f"/d3-{run_id}/files/{index:05d}")
        batch.put_file(manifest, f"/d3-{run_id}/manifest.json")
    require(
        all(pin(row["path"]) == row["pin"] for row in expected["rows"]),
        "Named upload changed during transfer",
    )
    sha = hashlib.sha256(supplied).hexdigest()
    downloaded = download.remote(run_id, sha)
    export(downloaded, destination / "download")
    require(downloaded["success"], "Public HF staging failed; no GPU worker started")
    replayed = replay.remote(run_id, sha)
    export(replayed, destination / "replay-job")
    require(replayed["success"], "GPU replay failed; full raw failure evidence retained")
