"""Mapped-only native Gemma decoder CPU experiment; dry by default."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import time
from collections import UserDict
from pathlib import Path

PROFILE = "bf16-real-layer-cpu-fixture-v1"
MAX_FILE = 1024**2
PLANNED_CASES = [
    {"layer_index": layer, "tokens": tokens, "repeats": 2} for layer in (0, 5) for tokens in (1, 32)
]


def need(condition, message):
    if not condition:
        raise ValueError(message)


def ordinary(path):
    path = Path(path).absolute()
    need(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary unlinked path required",
    )
    return path


def read(path):
    path = ordinary(path)
    need(path.is_file(), "Regular small selected file required")
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE + 1)
    need(len(data) <= MAX_FILE, "Small source/fixture bound")
    return data


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    need(len(data) <= MAX_FILE, "Small evidence bound")
    with ordinary(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def binary_stat(path):
    s = ordinary(path).stat()
    return {key: getattr(s, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns")}


def selection(path, digest, entry_digest):
    blob = read(path)
    need(sha(blob) == digest, "Independent selected manifest SHA")
    value = json.loads(blob)
    need(
        set(value) == {"profile", "sources", "versions", "binary_stats"}
        and value["profile"] == PROFILE,
        "Exact CPU fixture selection",
    )
    for row in value["sources"].values():
        data = read(row["path"])
        need(
            len(data) == row["size_bytes"] and sha(data) == row["sha256"], "Selected source changed"
        )
    own = value["sources"]["entry"]
    need(
        ordinary(own["path"]) == ordinary(__file__) and own["sha256"] == entry_digest,
        "Independent entry SHA/path",
    )
    for row in value["binary_stats"]:
        need(binary_stat(row["path"]) == row["stat"], "Selected binary stat identity changed")
    return value


def identity():
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": os.getpid(),
        "start_ticks": int(fields[19]),
        "pgid": os.getpgrp(),
        "sid": os.getsid(0),
    }


def exception(error):
    return {
        "type": type(error).__name__,
        "message": str(error)[:2000],
        "notes": getattr(error, "__notes__", []),
    }


class DeadlinePartial(RuntimeError):
    pass


def check_deadline(deadline, minimum_seconds=0.25):
    if deadline - time.monotonic() < minimum_seconds:
        raise DeadlinePartial("Insufficient parent-sealed deadline for the next stage")


def strict_stat(path):
    s = ordinary(path).stat()
    return {
        key: getattr(s, key)
        for key in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    }


def checked_header(header_selection):
    base, expected = header_selection["base"], header_selection["header"]
    path = ordinary(base["canonical_blob_path"])
    need(strict_stat(path) == base["strict_stat_before"], "Strict base stat identity")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        prefix = os.pread(fd, 8, 0)
        need(len(prefix) == 8, "Exact header length prefix")
        length = int.from_bytes(prefix, "little")
        need(
            length == expected["json_bytes"] and 0 < length <= 2 * MAX_FILE,
            "Bounded selected header",
        )
        body = os.pread(fd, length, 8)
        need(
            len(body) == length
            and sha(body) == expected["json_sha256"]
            and sha(prefix + body) == expected["prefix_and_json_sha256"],
            "Exact selected header SHA",
        )
        parsed = json.loads(body)
        for layer in header_selection["layers"]:
            for row in layer["states"]:
                actual = parsed[row["checkpoint_key"]]
                need(
                    actual
                    == {
                        "dtype": row["dtype"],
                        "shape": row["shape"],
                        "data_offsets": row["payload_offsets_half_open"],
                    },
                    "Selected range must match actual header",
                )
    finally:
        os.close(fd)
    need(strict_stat(path) == base["strict_stat_before"], "Base stat after bounded header")
    return path


def memory_snapshot():
    started = time.monotonic()
    status = {}
    for line in read(f"/proc/{os.getpid()}/status").decode().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap"):
            amount, unit = value.split()
            need(unit == "kB", "Linux process memory units")
            status[key + "_bytes"] = int(amount) * 1024
    rows = read(f"/proc/{os.getpid()}/cgroup").decode().splitlines()
    groups = [row[3:] for row in rows if row.startswith("0::")]
    need(
        len(groups) == 1 and groups[0].startswith("/") and ".." not in Path(groups[0]).parts,
        "Own unified cgroup path",
    )
    cgroup = ordinary(Path("/sys/fs/cgroup") / groups[0].lstrip("/"))
    memory = {
        "path": groups[0],
        "memory_current_bytes": int(read(cgroup / "memory.current")),
        "memory_peak_bytes": int(read(cgroup / "memory.peak")),
    }
    memory["memory_stat"] = {
        key: int(value)
        for key, value in (
            line.split() for line in read(cgroup / "memory.stat").decode().splitlines()
        )
        if key in ("anon", "file", "shmem", "kernel", "pgfault", "pgmajfault")
    }
    need(
        all(key in status for key in ("VmRSS_bytes", "RssAnon_bytes", "RssFile_bytes")),
        "Required process RSS breakdown",
    )
    return {
        "process": status,
        "own_cgroup": memory,
        "monotonic_before": started,
        "monotonic_after": time.monotonic(),
        "scope": "Sequential non-atomic own-process/own-cgroup snapshot; page cache release and disk IO bytes are not inferred.",
    }


def stage(output, result, name, details=None):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    record = {
        "stage": name,
        "monotonic": time.monotonic(),
        "elapsed_seconds": time.monotonic() - result["started_monotonic"],
        "peak_RSS_bytes": usage.ru_maxrss * 1024,
        "memory": memory_snapshot(),
        "completed_cases": result["completed_cases"],
        "planned_cases": PLANNED_CASES,
        "ru_minflt": usage.ru_minflt,
        "ru_majflt": usage.ru_majflt,
        "details": details,
    }
    path = output / ("stage-%02d.json" % len(result["stages"]))
    write(path, record)
    result["stages"].append(
        {"path": str(path), "sha256": sha(read(path)), "size_bytes": path.stat().st_size}
    )


def range_hashes(path, rows, deadline, progress):
    started = time.monotonic()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        for selected in rows:
            check_deadline(deadline)
            start, stop = selected["file_offsets_half_open"]
            row = {
                "key": selected["checkpoint_key"],
                "file_offsets_half_open": [start, stop],
                "bytes_hashed": 0,
                "complete": False,
            }
            progress["ranges"].append(row)
            digest = hashlib.sha256()
            while start < stop:
                check_deadline(deadline)
                data = os.pread(fd, min(MAX_FILE, stop - start), start)
                need(
                    len(data) == min(MAX_FILE, stop - start), "Exact bounded selected payload read"
                )
                digest.update(data)
                start += len(data)
                row["bytes_hashed"] += len(data)
            row.update(sha256=digest.hexdigest(), complete=True)
    finally:
        os.close(fd)
        progress["elapsed_seconds"] = time.monotonic() - started
        progress["explicit_selected_payload_hash_bytes"] = sum(
            row["bytes_hashed"] for row in progress["ranges"]
        )


def run_torch(output, selected, result, deadline, header_selection, blob):
    check_deadline(deadline, 2)
    import torch

    need(vars(torch.cuda).get("_initialized") is False, "CUDA must remain uninitialized")
    attempts, queries, originals = [], [], []
    result["CUDA"] = {
        "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "attempts": attempts,
        "queries": queries,
        "initialized_after_torch_import": False,
        "guard_scope": "Guard starts after Torch import; that interval and unwrapped native calls are not monitored. Real is_available/device_count queries require False/0; own operations are CPU.",
        "GPU_transfer_measured": False,
    }

    def deny(name):
        def refused(*_args, **_kwargs):
            attempts.append(name)
            raise RuntimeError("CUDA work forbidden: " + name)

        return refused

    def query(name, original, expected):
        def observed(*args, **kwargs):
            row = {"name": name}
            queries.append(row)
            try:
                value = original(*args, **kwargs)
                row["result"] = value
                need(
                    type(value) is type(expected) and value == expected,
                    "Actual CPU-only CUDA query",
                )
                return value
            except BaseException as error:
                row["error"] = exception(error)
                raise

        return observed

    for name, expected in (("is_available", False), ("device_count", 0)):
        original = getattr(torch.cuda, name)
        originals.append((torch.cuda, name, original))
        setattr(torch.cuda, name, query(name, original, expected))
    for name in (
        "current_device",
        "set_device",
        "get_device_properties",
        "mem_get_info",
        "init",
        "_lazy_init",
        "synchronize",
        "empty_cache",
        "manual_seed",
        "manual_seed_all",
        "current_stream",
        "stream",
    ):
        if hasattr(torch.cuda, name):
            originals.append((torch.cuda, name, getattr(torch.cuda, name)))
            setattr(torch.cuda, name, deny(name))
    for owner in (torch.Tensor, torch.nn.Module):
        originals.append((owner, "cuda", owner.cuda))
        owner.cuda = deny(owner.__name__ + ".cuda")
    interface, primary = None, None
    try:
        import safetensors
        import transformers
        from safetensors import safe_open
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        from transformers.models.gemma4_unified import modeling_gemma4_unified as native
        from transformers.models.gemma4_unified.configuration_gemma4_unified import (
            Gemma4UnifiedConfig,
        )

        packages = {"torch": torch, "transformers": transformers, "safetensors": safetensors}
        result["versions"] = {key: str(value.__version__) for key, value in packages.items()}
        need(result["versions"] == selected["versions"], "Selected actual versions")
        for key, package in packages.items():
            need(
                ordinary(package.__file__) == ordinary(selected["sources"][key]["path"]),
                "Selected actual package origin",
            )
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        check_deadline(deadline, 2)
        raw_config = json.loads(read(header_selection["config"]["path"]))
        config = Gemma4UnifiedConfig.from_dict(raw_config).text_config
        config._attn_implementation = "sdpa"
        need(
            config.num_hidden_layers == 48 and config.num_kv_shared_layers == 0,
            "Original 48-layer nonsharing configuration",
        )
        rotary = native.Gemma4UnifiedTextRotaryEmbedding(config=config, device="cpu")
        result["configuration"] = {
            "num_hidden_layers": config.num_hidden_layers,
            "num_kv_shared_layers": config.num_kv_shared_layers,
            "attention_implementation": config._attn_implementation,
            "cache_used": False,
            "synthetic_inputs": True,
            "cloned_weight_baseline": False,
        }
        result["attention_trace"] = []
        interface = native.ALL_ATTENTION_FUNCTIONS
        local_before, global_before = (
            dict(interface._local_mapping),
            dict(interface._global_mapping),
        )
        original_sdpa = interface["sdpa"]
        need(original_sdpa is sdpa_attention_forward, "Exact original native SDPA API route")

        def sdpa_observed(*args, **kwargs):
            module, q, k, v, mask = args[:5]
            row = {
                "layer_index": module.layer_idx,
                "q_shape": list(q.shape),
                "k_shape": list(k.shape),
                "v_shape": list(v.shape),
                "q_dtype": str(q.dtype),
                "k_dtype": str(k.dtype),
                "v_dtype": str(v.dtype),
                "devices": [str(t.device) for t in (q, k, v)],
                "mask": None
                if mask is None
                else {
                    "shape": list(mask.shape),
                    "dtype": str(mask.dtype),
                    "device": str(mask.device),
                },
                "dropout": kwargs.get("dropout"),
                "scaling": kwargs.get("scaling"),
                "sliding_window": kwargs.get("sliding_window"),
                "module_is_causal": module.is_causal,
                "native_sdpa_rule_expected_is_causal": q.shape[2] > 1
                and mask is None
                and module.is_causal,
                "num_key_value_groups": module.num_key_value_groups,
                "observed_route": "original transformers.integrations.sdpa_attention.sdpa_attention_forward; no profiler/kernel-kind claim",
            }
            result["attention_trace"].append(row)
            need(
                row["devices"] == ["cpu"] * 3
                and row["q_dtype"] == row["k_dtype"] == row["v_dtype"] == "torch.bfloat16"
                and row["scaling"] == 1.0
                and row["dropout"] == 0.0,
                "Unchanged native CPU attention arguments",
            )
            return original_sdpa(*args, **kwargs)

        interface["sdpa"] = sdpa_observed
        stage(output, result, "imports-native-config-rotary-ready", result["configuration"])

        def tensor_info(tensor):
            try:
                version, version_error = tensor._version, None
            except RuntimeError as error:
                version, version_error = None, str(error)
            return {
                "class": type(tensor).__name__,
                "object_id": id(tensor),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "device": str(tensor.device),
                "requires_grad": tensor.requires_grad,
                "version": version,
                "version_error": version_error,
                "data_ptr": None if tensor.device.type == "meta" else tensor.data_ptr(),
                "storage_offset": tensor.storage_offset(),
                "storage_bytes": None
                if tensor.device.type == "meta"
                else tensor.untyped_storage().nbytes(),
            }

        def mapped_state(layer, descriptor):
            tensors = {
                **dict(layer.named_parameters(remove_duplicate=False)),
                **dict(layer.named_buffers(remove_duplicate=False)),
            }
            need(
                set(tensors) == {row["decoder_state_key"] for row in descriptor["states"]},
                "Exact selected parameter and buffer set",
            )
            maps = [
                line.split(None, 5)
                for line in read(f"/proc/{os.getpid()}/maps").decode().splitlines()
            ]
            proof = {}
            for row in descriptor["states"]:
                tensor = tensors[row["decoder_state_key"]]
                info = tensor_info(tensor)
                need(
                    info["device"] == "cpu"
                    and info["dtype"] == "torch.bfloat16"
                    and info["shape"] == row["shape"]
                    and not info["requires_grad"],
                    "Actual assigned BF16 frozen state",
                )
                joined = []
                for fields in maps:
                    lo, hi = (int(x, 16) for x in fields[0].split("-"))
                    if lo <= info["data_ptr"] and info["data_ptr"] + row["size_bytes"] <= hi:
                        if (
                            int(fields[4])
                            != header_selection["base"]["strict_stat_before"]["st_ino"]
                        ):
                            continue
                        need(
                            tuple(int(x, 16) for x in fields[3].split(":"))
                            == (
                                os.major(header_selection["base"]["strict_stat_before"]["st_dev"]),
                                os.minor(header_selection["base"]["strict_stat_before"]["st_dev"]),
                            ),
                            "Mapped base device",
                        )
                        offset = int(fields[2], 16) + info["data_ptr"] - lo
                        need(
                            offset == row["file_offsets_half_open"][0],
                            "Exact tensor pointer to file offset join",
                        )
                        joined.append(
                            {
                                "range": fields[0],
                                "permissions": fields[1],
                                "file_mapping_offset": fields[2],
                                "inode": fields[4],
                                "device": fields[3],
                                "path": fields[5] if len(fields) > 5 else None,
                                "tensor_file_offset": offset,
                            }
                        )
                need(len(joined) == 1, "One exact file-backed VMA join per tensor")
                info["VMA"] = joined[0]
                proof[row["decoder_state_key"]] = info
            return proof

        def tensor_digest(tensor):
            need(tensor.device.type == "cpu", "CPU-only output hashing")
            return sha(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes())

        result["layers"] = []
        result["range_hashes"] = {}
        result["cache_state"] = (
            "Selected-range prehash prefaults/warms pages. No cold latency claim or cache eviction. OS page-level read-ahead may touch adjacent bytes; only explicit application range requests are restricted."
        )
        for descriptor in header_selection["layers"]:
            check_deadline(deadline, 2)
            idx = descriptor["layer_index"]
            hashes = result["range_hashes"][str(idx)] = {
                "before": {"ranges": []},
                "after": {"ranges": []},
            }
            range_hashes(blob, descriptor["states"], deadline, hashes["before"])
            stage(output, result, "layer-%d-selected-range-prehash" % idx, hashes["before"])
            check_deadline(deadline, 2)
            stage(output, result, "layer-%d-before-meta-construction" % idx)
            with torch.device("meta"):
                layer = native.Gemma4UnifiedTextDecoderLayer(config, idx)
            layer.eval().requires_grad_(False)
            need(
                all(
                    t.device.type == "meta"
                    for t in list(layer.parameters()) + list(layer.buffers())
                ),
                "Native layer constructed entirely meta",
            )
            stage(
                output,
                result,
                "layer-%d-meta-constructed" % idx,
                {
                    "parameters": len(list(layer.parameters())),
                    "buffers": len(list(layer.buffers())),
                },
            )
            with safe_open(blob, framework="pt", device="cpu", backend="mmap") as handle:
                weights = {
                    row["decoder_state_key"]: handle.get_tensor(row["checkpoint_key"])
                    for row in descriptor["states"]
                }
            loading = layer.load_state_dict(weights, strict=True, assign=True)
            need(
                not loading.missing_keys and not loading.unexpected_keys,
                "Strict native layer assign",
            )
            record = {
                "layer_index": idx,
                "mapped_before": mapped_state(layer, descriptor),
                "cases": [],
            }
            result["layers"].append(record)
            need(
                all(
                    record["mapped_before"][key]["data_ptr"] == value.data_ptr()
                    for key, value in weights.items()
                ),
                "Assign must preserve every selected mmap pointer",
            )
            stage(output, result, "layer-%d-meta-strict-assign-VMA" % idx, record["mapped_before"])
            for tokens in (1, 32):
                check_deadline(deadline, 2)
                hidden = (
                    torch.arange(tokens * 3840, dtype=torch.float32, device="cpu")
                    .remainder(97)
                    .sub(48)
                    .div(64)
                    .to(torch.bfloat16)
                    .reshape(1, tokens, 3840)
                )
                position_ids = torch.arange(tokens, dtype=torch.long, device="cpu").unsqueeze(0)
                input_hash = tensor_digest(hidden)
                builder = (
                    native.create_sliding_window_causal_mask
                    if descriptor["attention_type"] == "sliding_attention"
                    else native.create_causal_mask
                )
                mask = builder(
                    config=config,
                    inputs_embeds=hidden,
                    attention_mask=None,
                    past_key_values=None,
                    position_ids=position_ids,
                )
                with torch.inference_mode():
                    positions = rotary(hidden, position_ids, descriptor["attention_type"])
                case = {
                    "layer_index": idx,
                    "tokens": tokens,
                    "repeats": [],
                    "input_sha256": input_hash,
                    "input_object_id": id(hidden),
                    "mask_builder": builder.__name__,
                    "mask": None if mask is None else tensor_info(mask),
                    "RoPE_shapes": [list(t.shape) for t in positions],
                    "prefetched_warm_pages": True,
                }
                record["cases"].append(case)
                for repeat in range(2):
                    check_deadline(deadline, 2)
                    started = time.monotonic()
                    before_calls = len(result["attention_trace"])
                    with torch.inference_mode():
                        actual = layer(
                            hidden,
                            shared_kv_states=UserDict(),
                            position_embeddings=positions,
                            attention_mask=mask,
                            position_ids=position_ids,
                            past_key_values=None,
                            use_cache=False,
                        )
                    elapsed = time.monotonic() - started
                    need(
                        len(result["attention_trace"]) == before_calls + 1,
                        "One observed native SDPA dispatch per repeat",
                    )
                    need(
                        actual.device.type == "cpu"
                        and actual.dtype == torch.bfloat16
                        and list(actual.shape) == [1, tokens, 3840]
                        and bool(torch.isfinite(actual).all()),
                        "Finite native BF16 CPU output",
                    )
                    case["repeats"].append(
                        {
                            "repeat": repeat,
                            "forward_seconds": elapsed,
                            "output_sha256": tensor_digest(actual),
                            "output_dtype": str(actual.dtype),
                            "output_shape": list(actual.shape),
                            "finite": True,
                            "absolute_max": float(actual.float().abs().max()),
                        }
                    )
                    del actual
                    need(tensor_digest(hidden) == input_hash, "Same unmodified synthetic input")
                    stage(
                        output, result, "layer-%d-tokens-%d-repeat-%d" % (idx, tokens, repeat), case
                    )
                need(
                    case["repeats"][0]["output_sha256"] == case["repeats"][1]["output_sha256"],
                    "Two repeat output hashes must match exactly",
                )
                case["repeat_determinism_only"] = True
                result["completed_cases"].append(
                    {"layer_index": idx, "tokens": tokens, "repeats": 2}
                )
                del hidden, positions, position_ids, mask
            record["mapped_after"] = mapped_state(layer, descriptor)
            need(
                all(
                    record["mapped_before"][key] == value
                    for key, value in record["mapped_after"].items()
                ),
                "Native state identity, dtype, grad flags, version and mapping unchanged",
            )
            range_hashes(blob, descriptor["states"], deadline, hashes["after"])
            need(
                hashes["before"]["ranges"] == hashes["after"]["ranges"],
                "Every selected tensor payload range hash unchanged",
            )
            record["selected_range_integrity_final"] = True
            stage(
                output,
                result,
                "layer-%d-after-repeat-state-and-range-integrity" % idx,
                {"mapped_after": record["mapped_after"], "range_hashes_after": hashes["after"]},
            )
            del weights, layer, handle
            gc.collect()
            stage(
                output,
                result,
                "layer-%d-model-and-handle-released-gc" % idx,
                {
                    "references_deleted": ["selected weights", "decoder layer", "safe_open handle"],
                    "page_cache_reclaimed_claim": False,
                },
            )
        check_deadline(deadline)
        need(result["completed_cases"] == PLANNED_CASES, "All four planned cases must complete")
        need(
            not attempts
            and all("error" not in q for q in queries)
            and vars(torch.cuda).get("_initialized") is False,
            "No CUDA work or swallowed invalid query",
        )
        result["selected_range_integrity_final"] = True
        result["mechanism_checks_completed"] = True
    except BaseException as error:
        primary = error
        raise
    finally:
        cleanup_errors = []
        if interface is not None and "local_before" in locals():
            try:
                if "sdpa" in local_before:
                    interface["sdpa"] = local_before["sdpa"]
                elif "sdpa" in interface._local_mapping:
                    del interface["sdpa"]
                result["attention_interface_restored"] = (
                    interface._local_mapping == local_before
                    and interface._global_mapping == global_before
                )
                need(
                    result["attention_interface_restored"], "Original attention interface restored"
                )
            except BaseException as error:
                result["attention_interface_restored"] = False
                cleanup_errors.append(exception(error))
        result["CUDA"]["initialized_final"] = vars(torch.cuda).get("_initialized")
        for owner, name, original in reversed(originals):
            try:
                setattr(owner, name, original)
            except BaseException as error:
                cleanup_errors.append(exception(error))
        result["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            if primary is not None:
                primary.add_note(
                    "Cleanup errors retained separately: " + json.dumps(cleanup_errors)
                )
            else:
                raise RuntimeError("Explicit API restoration failed: " + json.dumps(cleanup_errors))


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--profile")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-sha256")
    parser.add_argument("--entry-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.run:
        print(
            json.dumps(
                {
                    "profile": PROFILE,
                    "mode": "DRY_NO_TORCH_PAYLOAD_MMAP_OR_WRITES",
                    "planned_cases": PLANNED_CASES,
                }
            )
        )
        return 0
    need(
        args.profile == PROFILE
        and args.selection
        and args.selection_sha256
        and args.entry_sha256
        and args.output,
        "Explicit source/profile/output selection",
    )
    need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and "CUDA_VISIBLE_DEVICES" in os.environ
        and os.environ.get("ACCELERATE_USE_CPU") == "true",
        "Explicit CPU-only environment",
    )
    need(
        all(
            os.environ.get(k) == "1"
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        )
        and os.environ.get("USE_TF") == "0"
        and os.environ.get("USE_FLAX") == "0",
        "One CPU thread and no unrelated frameworks",
    )
    deadline = float(os.environ["JEV_LAYER_DEADLINE_MONOTONIC"])
    need(math.isfinite(deadline) and deadline > 0, "Explicit finite parent deadline")
    selected = selection(args.selection, args.selection_sha256, args.entry_sha256)
    header_selection = json.loads(read(selected["sources"]["header_selection"]["path"]))
    blob = checked_header(header_selection)
    output = ordinary(args.output)
    need(not output.exists(), "Fresh nonexistent output required")
    output.mkdir(mode=0o700)
    started = time.monotonic()
    result = {
        "profile": PROFILE,
        "CPU_ONLY": True,
        "GPU_transfer_measured": False,
        "large_model_runtime_or_quality_qualified": False,
        "passed": False,
        "completed": False,
        "mechanism_checks_completed": False,
        "planned_cases": PLANNED_CASES,
        "completed_cases": [],
        "stages": [],
        "identity": identity(),
        "started_monotonic": started,
        "entry_sha256": args.entry_sha256,
        "source_selection_sha256": args.selection_sha256,
        "parent_deadline_monotonic": deadline,
        "status": "started",
        "limits": "Mapped native single-layer synthetic CPU determinism only; no clone-equivalence,GPU,UMA,12B,8192,20s,N3,Q95,browser or score qualification",
    }
    write(output / "intent.json", result)
    primary = None
    try:
        run_torch(output, selected, result, deadline, header_selection, blob)
        need(
            result["mechanism_checks_completed"]
            and result["completed_cases"] == PLANNED_CASES
            and result.get("attention_interface_restored") is True,
            "All four cases and reversible interface cleanup required",
        )
    except BaseException as error:
        primary = error
    finally:
        try:
            need(
                selection(args.selection, args.selection_sha256, args.entry_sha256) == selected,
                "Final selected small sources and binary stats",
            )
            checked_header(header_selection)
            result["source_integrity_final"] = True
        except BaseException as error:
            result["source_integrity_final"] = False
            if primary is None or isinstance(primary, DeadlinePartial):
                if primary is not None:
                    error.add_note("Earlier deadline partial: " + str(primary))
                primary = error
            else:
                primary.add_note("Final source/header refusal: " + str(error))
        partial = isinstance(primary, DeadlinePartial) and not result.get("cleanup_errors")
        result.update(
            status="partial_deadline"
            if partial
            else "failed"
            if primary is not None
            else "completed_mechanism_observations",
            passed=primary is None,
            completed=primary is None,
            elapsed_seconds=time.monotonic() - started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            identity_after=identity(),
            error=None if primary is None else exception(primary),
        )
        if primary is not None:
            result["mechanism_checks_completed"] = False
        write(output / "receipt.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_cases": len(result["completed_cases"]),
                "qualified": False,
            }
        )
    )
    return 2 if partial else 1 if primary is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
