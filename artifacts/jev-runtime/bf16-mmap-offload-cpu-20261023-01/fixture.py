"""Tiny CPU mechanism observations; dry by default, never GPU qualification."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import time
from pathlib import Path

PROFILE = "bf16-mmap-offload-cpu-fixture-v1"
MAX_FILE = 1024**2


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


def run_torch(output, selected, result):
    # No heavy import occurs on dry/static paths. The owner supplies the CPU-only
    # environment and admits its own process before ever reaching this function.
    import torch

    need(vars(torch.cuda).get("_initialized") is False, "CUDA must not be initialized")
    cuda_attempts, cuda_queries, originals = [], [], []

    def denied(name):
        def refuse(*_args, **_kwargs):
            cuda_attempts.append(name)
            raise RuntimeError("CUDA API forbidden in CPU fixture: " + name)

        return refuse

    def query(name, original, expected):
        def observe(*args, **kwargs):
            row = {"name": name}
            cuda_queries.append(row)
            try:
                actual = original(*args, **kwargs)
                row["result"] = actual
                need(
                    type(actual) is type(expected) and actual == expected,
                    "CPU-only CUDA query result",
                )
                return actual
            except BaseException as error:
                row["error"] = exception(error)
                raise

        return observe

    for name, expected in (("is_available", False), ("device_count", 0)):
        original = getattr(torch.cuda, name)
        originals.append((torch.cuda, name, original))
        setattr(torch.cuda, name, query("torch.cuda." + name, original, expected))
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
            setattr(torch.cuda, name, denied("torch.cuda." + name))
    for owner, name in ((torch.Tensor, "cuda"), (torch.nn.Module, "cuda")):
        originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, denied(owner.__name__ + ".cuda"))
    result["CUDA"] = {
        "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "initialized_after_torch_import": False,
        "attempts": cuda_attempts,
        "queries": cuda_queries,
        "guard_scope": "Installed after torch import; that import interval and unwrapped native calls are not monitored. Real availability queries require False/0. Own operations use explicit CPU devices.",
        "GPU_transfer_measured": False,
    }
    try:
        import accelerate
        import peft
        import safetensors
        from safetensors import safe_open
        from safetensors.torch import save_file

        packages = {
            "torch": torch,
            "accelerate": accelerate,
            "peft": peft,
            "safetensors": safetensors,
        }
        result["versions"] = {key: str(value.__version__) for key, value in packages.items()}
        need(result["versions"] == selected["versions"], "Selected actual package versions")
        for key, package in packages.items():
            need(
                ordinary(package.__file__) == ordinary(selected["sources"][key]["path"]),
                "Actual selected package origin",
            )
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        nn, trace, control = torch.nn, [], {"raise_in": None}

        def tensor_spec(tensor):
            meta = tensor.device.type == "meta"
            try:
                version, version_error = tensor._version, None
            except RuntimeError as error:
                version, version_error = None, str(error)
            return {
                "object_id": id(tensor),
                "class": type(tensor).__name__,
                "version": version,
                "version_error": version_error,
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "storage_offset": tensor.storage_offset(),
                "storage_bytes": None if meta else tensor.untyped_storage().nbytes(),
                "storage_ptr": None if meta else tensor.untyped_storage().data_ptr(),
                "data_ptr": None if meta else tensor.data_ptr(),
                "requires_grad": tensor.requires_grad,
            }

        def snapshot(model):
            return {
                name: tensor_spec(p) for name, p in model.named_parameters(remove_duplicate=False)
            }

        class BodyError(RuntimeError):
            pass

        class Linear(nn.Linear):
            def forward(self, x):
                trace.append({"body": self.label, "weight": tensor_spec(self.weight)})
                if control["raise_in"] == self.label:
                    raise BodyError("controlled original proj forward body failure")
                return super().forward(x)

        class Embedding(nn.Embedding):
            def forward(self, x):
                trace.append({"body": "embed", "weight": tensor_spec(self.weight)})
                return super().forward(x)

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = Embedding(8, 4, device="cpu", dtype=torch.bfloat16)
                self.proj = Linear(4, 4, bias=False, device="cpu", dtype=torch.bfloat16)
                self.head = Linear(4, 8, bias=False, device="cpu", dtype=torch.bfloat16)
                self.proj.label, self.head.label = "proj", "head"
                self.head.weight = self.embed.weight

            def forward(self, ids):
                return self.head(self.proj(self.embed(ids)))

        model = Tiny().eval().requires_grad_(False)
        with torch.no_grad():
            model.embed.weight.copy_(
                torch.arange(32, device="cpu", dtype=torch.float32)
                .reshape(8, 4)
                .div(32)
                .to(torch.bfloat16)
            )
            model.proj.weight.copy_(torch.eye(4, device="cpu", dtype=torch.bfloat16).div(2))
        weights = output / "tiny-weights.safetensors"
        save_file(
            {"embed": model.embed.weight.detach(), "proj": model.proj.weight.detach()}, weights
        )
        need(weights.stat().st_size <= MAX_FILE, "Tiny generated safetensors bound")
        weights.chmod(0o444)
        file_sha = sha(read(weights))
        result["generated_file"] = {
            "path": str(weights),
            "sha256": file_sha,
            "size_bytes": weights.stat().st_size,
        }
        with safe_open(weights, framework="pt", device="cpu", backend="mmap") as handle:
            embedding, projection = handle.get_tensor("embed"), handle.get_tensor("proj")
            same = embedding.to(device="cpu", dtype=torch.bfloat16)
            need(same.data_ptr() == embedding.data_ptr(), "Same-dtype CPU to must preserve pointer")
            mmap_before = tensor_spec(embedding)
        gc.collect()
        need(
            torch.equal(embedding, model.embed.weight),
            "Mapped values must survive safe_open context",
        )
        maps = []
        for line in read(f"/proc/{os.getpid()}/maps").decode().splitlines():
            fields = line.split(None, 5)
            lo, hi = (int(x, 16) for x in fields[0].split("-"))
            if lo <= embedding.data_ptr() < hi:
                maps.append(
                    {
                        "range": fields[0],
                        "permissions": fields[1],
                        "offset": fields[2],
                        "tensor_file_offset": int(fields[2], 16) + embedding.data_ptr() - lo,
                        "device": fields[3],
                        "inode": fields[4],
                        "path": fields[5] if len(fields) > 5 else None,
                    }
                )
        need(
            any(row["path"] == str(weights) for row in maps),
            "Mapped tensor address must belong to own tiny file",
        )
        result["mmap"] = {
            "file_sha256": file_sha,
            "file_stat": binary_stat(weights),
            "size_bytes": weights.stat().st_size,
            "before_context_close": mmap_before,
            "after_context_close": tensor_spec(embedding),
            "same_to_object": same is embedding,
            "maps": maps,
            "lifetime_values_equal": True,
        }
        model.embed.weight = nn.Parameter(embedding, requires_grad=False)
        model.proj.weight = nn.Parameter(projection, requires_grad=False)
        model.head.weight = model.embed.weight
        ids = torch.tensor([[0, 2, 5]], dtype=torch.long, device="cpu")
        with torch.inference_mode():
            baseline = model(ids).clone()

        def offloaded_forward(module, reference, label):
            external = {key: value.detach() for key, value in module.state_dict().items()}
            row = {
                "before": snapshot(module),
                "external": {k: tensor_spec(v) for k, v in external.items()},
                "execution_device": "cpu",
                "body_trace_start": len(trace),
            }
            primary = None
            try:
                accelerate.cpu_offload(module, execution_device="cpu", state_dict=external)
                installed = snapshot(module)
                row["installed"] = installed
                row["hook_chains"] = {
                    name: {
                        "class": type(child._hf_hook).__name__,
                        "nested": [type(h).__name__ for h in getattr(child._hf_hook, "hooks", ())],
                    }
                    for name, child in module.named_modules()
                    if hasattr(child, "_hf_hook")
                }
                row["external_map_references"] = {
                    name: child._hf_hook.weights_map.dataset is external
                    for name, child in module.named_modules()
                    if hasattr(getattr(child, "_hf_hook", None), "weights_map")
                    and getattr(child._hf_hook, "weights_map") is not None
                }
                need(
                    bool(row["external_map_references"])
                    and all(row["external_map_references"].values()),
                    "Actual hooks must retain supplied external state_dict",
                )
                need(
                    all(v["device"] == "meta" for v in installed.values()),
                    "All offloaded parameters must start meta",
                )
                for _ in range(2):
                    with torch.inference_mode():
                        actual = module(ids)
                    need(torch.equal(actual, reference), label + " exact CPU baseline parity")
                row["after_two_forwards"] = snapshot(module)
                need(
                    all(v["device"] == "meta" for v in row["after_two_forwards"].values()),
                    "Normal hook post_forward must restore meta",
                )
            except BaseException as error:
                primary = error
                row["error"] = exception(error)
            finally:
                try:
                    row["before_explicit_detach"] = snapshot(module)
                except BaseException as error:
                    if primary is None:
                        primary = error
                    else:
                        primary.add_note("Pre-detach snapshot failed: " + str(error))
                try:
                    accelerate.hooks.remove_hook_from_module(module, recurse=True)
                except BaseException as error:
                    if primary is None:
                        primary = error
                    else:
                        primary.add_note("Explicit detach also failed: " + str(error))
                try:
                    row["after_explicit_detach"] = snapshot(module)
                except BaseException as error:
                    if primary is None:
                        primary = error
                    else:
                        primary.add_note("Post-detach snapshot failed: " + str(error))
                if primary is not None:
                    row["error"] = exception(primary)
                result[label] = row
            if primary is not None:
                raise primary
            need(
                all(v["device"] == "cpu" for v in row["after_explicit_detach"].values()),
                "Explicit detach must restore CPU weights",
            )
            row["two_forwards_equal"] = True

        result["tie_before"] = {
            "same_parameter": model.embed.weight is model.head.weight,
            "embed": tensor_spec(model.embed.weight),
            "head": tensor_spec(model.head.weight),
        }
        offloaded_forward(model, baseline, "base_offload")
        result["tie_after_base_detach"] = {
            "same_parameter": model.embed.weight is model.head.weight,
            "same_nonmeta_storage": model.embed.weight.data_ptr() == model.head.weight.data_ptr(),
            "no_retie_performed": True,
        }

        result["LoRA_starting_model"] = {
            "fresh_model": False,
            "prior_base_offload_and_explicit_detach": True,
            "same_parameter": model.embed.weight is model.head.weight,
            "same_nonmeta_storage": model.embed.weight.data_ptr() == model.head.weight.data_ptr(),
            "no_retie_performed": True,
        }
        model = peft.get_peft_model(
            model,
            peft.LoraConfig(
                r=2, lora_alpha=2, lora_dropout=0, target_modules=["proj"], bias="none"
            ),
        )
        model.eval().requires_grad_(False)
        layer = model.base_model.model.proj
        with torch.no_grad():
            layer.lora_A["default"].weight.fill_(0.125)
            layer.lora_B["default"].weight.fill_(0.25)
        with torch.inference_mode():
            enabled = model(ids).clone()
            with model.disable_adapter():
                disabled = model(ids).clone()
        need(
            torch.equal(disabled, baseline) and not torch.equal(enabled, baseline),
            "Nonzero LoRA enable/disable must distinguish exact baseline",
        )
        result["LoRA_before_offload"] = {
            "enabled": enabled.float().tolist(),
            "baseline": baseline.float().tolist(),
            "parameters": snapshot(model),
            "active": list(layer.active_adapters),
            "merged": list(layer.merged_adapters),
            "B_nonzero": bool(torch.count_nonzero(layer.lora_B["default"].weight)),
        }
        offloaded_forward(model, enabled, "LoRA_offload")
        external = {key: value.detach() for key, value in model.state_dict().items()}
        primary = None
        expected_body_error = False
        try:
            accelerate.cpu_offload(model, execution_device="cpu", state_dict=external)
            with torch.inference_mode():
                with model.disable_adapter():
                    need(
                        torch.equal(model(ids), baseline),
                        "Offloaded disabled adapter must equal baseline",
                    )
                need(
                    torch.equal(model(ids), enabled),
                    "Offloaded re-enabled adapter must equal enabled reference",
                )
            control["raise_in"] = "proj"
            try:
                with torch.inference_mode():
                    model(ids)
            except BodyError as error:
                primary = error
                expected_body_error = True
                result["controlled_body_error"] = {
                    "original_error": exception(error),
                    "residual_before_detach": snapshot(model),
                    "automatic_post_forward_safe": all(
                        p.device.type == "meta" for p in model.parameters()
                    ),
                    "classification": "MECHANISM_OBSERVATION_NOT_GPU_OR_IDENTITY_PARITY",
                }
            need(primary is not None, "Controlled original body error must occur")
        except BaseException as error:
            primary = error
            expected_body_error = False
        finally:
            control["raise_in"] = None
            result["error_phase"] = {"expected_body_error_observed": expected_body_error}
            try:
                result["error_phase"]["before_explicit_detach"] = snapshot(model)
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    primary.add_note("Pre-detach snapshot failed: " + str(error))
                expected_body_error = False
            try:
                accelerate.hooks.remove_hook_from_module(model, recurse=True)
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    primary.add_note("Explicit detach also failed: " + str(error))
                expected_body_error = False
            try:
                result["after_error_explicit_detach"] = snapshot(model)
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    primary.add_note("Post-detach snapshot failed: " + str(error))
                expected_body_error = False
            if primary is not None:
                result["error_phase"]["error"] = exception(primary)
                if "controlled_body_error" in result:
                    result["controlled_body_error"]["original_error"] = exception(primary)
            if primary is not None and not expected_body_error:
                raise primary
        need(
            all(p.device.type == "cpu" for p in model.parameters()),
            "After error explicit detach must restore CPU",
        )
        result["LoRA_final_state"] = {
            "active": list(layer.active_adapters),
            "merged": list(layer.merged_adapters),
            "disable_adapters": layer.disable_adapters,
        }
        result["body_trace"] = trace
        need(
            all(row["weight"]["device"] == "cpu" for row in trace),
            "Actual toy forward bodies must observe CPU weights",
        )
        expected_pointers = {
            "embed": embedding.data_ptr(),
            "head": embedding.data_ptr(),
            "proj": projection.data_ptr(),
        }
        need(
            all(row["weight"]["data_ptr"] == expected_pointers[row["body"]] for row in trace),
            "Actual CPU body weights must preserve external mapped storage",
        )
        need(sha(read(weights)) == file_sha, "Own mapped tiny file must remain unchanged")
        need(
            not cuda_attempts and vars(torch.cuda).get("_initialized") is False,
            "No blocked CUDA work attempt or initialization permitted",
        )
        need(
            all("error" not in row for row in cuda_queries),
            "No failed or unexpected real CUDA availability query may be swallowed",
        )
        result["mechanism_checks_completed"] = True
    finally:
        if "trace" in locals():
            result["body_trace"] = trace
        result.setdefault("CUDA", {})["attempts"] = cuda_attempts
        result["CUDA"]["initialized_final"] = vars(torch.cuda).get("_initialized")
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


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
        print(json.dumps({"profile": PROFILE, "mode": "DRY_NO_TORCH_MMAP_OR_WRITES"}))
        return 0
    need(
        args.profile == PROFILE
        and args.selection
        and args.selection_sha256
        and args.entry_sha256
        and args.output,
        "Explicit profile/source/output selection required",
    )
    need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "" and "CUDA_VISIBLE_DEVICES" in os.environ,
        "Explicit empty CUDA_VISIBLE_DEVICES required before import",
    )
    need(os.environ.get("ACCELERATE_USE_CPU") == "true", "Explicit CPU-only Accelerate environment")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        need(os.environ.get(key) == "1", "Single CPU thread environment required")
    need(
        os.environ.get("USE_TF") == "0" and os.environ.get("USE_FLAX") == "0",
        "Disable unrelated framework imports",
    )
    selected = selection(args.selection, args.selection_sha256, args.entry_sha256)
    output = ordinary(args.output)
    need(not output.exists(), "Fresh nonexistent output required")
    output.mkdir(mode=0o700)
    started = time.monotonic()
    result = {
        "profile": PROFILE,
        "identity": identity(),
        "source_selection_sha256": args.selection_sha256,
        "entry_sha256": args.entry_sha256,
        "CPU_ONLY": True,
        "passed": False,
        "GPU_transfer_measured": False,
        "large_model_runtime_or_quality_qualified": False,
        "CPU_only_limits": "Tiny CPU hooks cannot qualify GPU transfer, UMA,12B,8192context,20s,N3 or Q95",
        "status": "started",
    }
    write(output / "intent.json", result)
    primary = None
    try:
        run_torch(output, selected, result)
        need(
            result.get("mechanism_checks_completed") is True,
            "All mechanism observations must complete",
        )
    except BaseException as error:
        primary = error
    finally:
        try:
            need(
                selection(args.selection, args.selection_sha256, args.entry_sha256) == selected,
                "Final selected sources unchanged",
            )
            result["source_integrity_final"] = True
        except BaseException as error:
            result["source_integrity_final"] = False
            if primary is None:
                primary = error
            else:
                primary.add_note("Final source check failed: " + str(error))
        if "generated_file" in result:
            try:
                row = result["generated_file"]
                data = read(row["path"])
                need(
                    len(data) == row["size_bytes"] and sha(data) == row["sha256"],
                    "Own generated tensor file changed",
                )
                result["ownfile_integrity_final"] = True
            except BaseException as error:
                result["ownfile_integrity_final"] = False
                if primary is None:
                    primary = error
                else:
                    primary.add_note("Final own-file check failed: " + str(error))
        result.update(
            status="failed" if primary is not None else "completed_mechanism_observations",
            passed=primary is None and result.get("mechanism_checks_completed") is True,
            elapsed_seconds=time.monotonic() - started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            error=None if primary is None else exception(primary),
            identity_after=identity(),
        )
        write(output / "receipt.json", result)
    print(json.dumps({"status": result["status"], "output": str(output), "qualified": False}))
    return 1 if primary is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
