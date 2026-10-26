"""Pure source/argument/closure tests; no frameworks, units or real helper."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).absolute().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    exec(compile((HERE / file).read_bytes(), str(spec.origin), "exec"), module.__dict__)
    return module


d = load("_native_q95_driver_test", "control_driver.py")
o = load("_native_q95_owner_test", "launch_browser.py")
p = load("_native_q95_prepare_test", "prepare_control.py")


def test_only_text_connection_changes_policy_channel_and_noncredential_arguments():
    argv = [
        "python",
        "client.py",
        "--endpoint",
        "http://own/v1/systemone",
        "--model",
        "generated-actions-nf4",
        "--text-endpoint",
        "http://old",
        "--text-model",
        "old",
        "--text-api-key-env",
        "S1_TEXT_API_KEY",
        "--max-steps",
        "16",
    ]
    env = {
        "S1_POLICY_API_KEY": "synthetic-policy",
        "S1_TEXT_API_KEY": "synthetic-old",
        "other": "same",
    }
    original = copy.deepcopy((argv, env))
    changed, selected_env = d.bind_text_client(
        argv,
        env,
        {
            "endpoint": "http://selected",
            "model": "generated-actions-none",
            "api_key": "synthetic-helper",
        },
    )
    assert (argv, env) == original
    inverse = changed.copy()
    inverse[inverse.index("--text-endpoint") + 1] = "http://old"
    inverse[inverse.index("--text-model") + 1] = "old"
    assert inverse == argv
    assert selected_env == {**env, "S1_TEXT_API_KEY": "synthetic-helper"}
    assert "synthetic-helper" not in " ".join(changed)


@pytest.mark.parametrize(
    "argv",
    [
        ["--text-endpoint", "x"],
        [
            "--text-endpoint",
            "a",
            "--text-endpoint",
            "b",
            "--text-model",
            "m",
            "--text-api-key-env",
            "S1_TEXT_API_KEY",
        ],
        ["--text-endpoint", "a", "--text-model", "m", "--text-api-key-env", "S1_POLICY_API_KEY"],
    ],
)
def test_missing_ambiguous_or_wrong_text_credential_channel_refuses(argv):
    with pytest.raises((ValueError, IndexError)):
        d.bind_text_client(
            argv, {}, {"endpoint": "http://selected", "model": "m", "api_key": "synthetic"}
        )


def test_driver_default_dry_never_loads_parent_or_components(monkeypatch, capsys):
    def forbidden(*_args):
        raise AssertionError("Dry must not import selected actor components")

    monkeypatch.setattr(d, "selected_module", forbidden)
    monkeypatch.setattr(d, "selected_local", forbidden)
    assert d.main([]) == 0
    assert "DRY_NO_MODEL_OR_BROWSER" in capsys.readouterr().out


def test_owner_default_dry_never_loads_probe_model_plan_or_guard(monkeypatch, capsys):
    def forbidden(*_args):
        raise AssertionError("No actual owner path in dry mode")

    monkeypatch.setattr(o.owner, "load_plan", forbidden)
    monkeypatch.setattr(o.owner, "load_observer", forbidden)
    monkeypatch.setattr(o.owner, "load_guard", forbidden)
    assert o.main([]) == 0
    assert "DRY_NO_ACTOR_OR_MODEL" in capsys.readouterr().out


def test_plan_preparer_default_dry_has_no_metadata_writes(monkeypatch, capsys):
    monkeypatch.setattr(p, "prepare", lambda *_: pytest.fail("Dry must not prepare"))
    assert p.main([]) == 0
    assert "DRY_NO_METADATA_OR_ACTOR_WRITES" in capsys.readouterr().out


def test_checked_source_ignores_a_stale_bytecode_population_and_rejects_drift(tmp_path):
    path = tmp_path / "selected.py"
    blob = b"answer = 42\n"
    path.write_bytes(blob)
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__/selected.cpython-312.pyc").write_bytes(b"stale synthetic cache")
    assert (
        d.selected_module(path, hashlib.sha256(blob).hexdigest(), "_gemma_selected_test").answer
        == 42
    )
    path.write_text("answer = 99\n")
    with pytest.raises(ValueError):
        d.selected_module(path, hashlib.sha256(blob).hexdigest(), "_gemma_selected_drift")


def test_source_symlink_rejected_before_compile(tmp_path):
    source = tmp_path / "real.py"
    source.write_text("raise AssertionError('must not execute')\n")
    alias = tmp_path / "alias.py"
    alias.symlink_to(source)
    with pytest.raises(ValueError):
        d.selected_module(alias, hashlib.sha256(source.read_bytes()).hexdigest(), "_gemma_alias")


def test_original_owner_reused_exactly_with_only_explicit_private_bindings():
    assert hashlib.sha256(o.PARENT.read_bytes()).hexdigest() == o.PARENT_SHA
    assert o.owner.CAP == 40 * 1024**3
    assert o.owner.FORMATS == {"v2-q95": "shared-elements-v2"}
    assert o.owner.OBSERVER_SOURCE == o.OBSERVER
    assert (
        o.owner.HERE == HERE
        and o.owner.DRIVER == (HERE / "control_driver.py").relative_to(d.ROOT).as_posix()
    )
    assert o.owner.verify_closed is o.verify_closed and o.owner.validate_plan is o.validate_plan
    assert o.owner.execute.__code__.co_filename == str(o.PARENT)
    assert o.owner.readback.__code__.co_filename == str(o.PARENT)


@pytest.fixture
def selected_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(o, "ROOT", tmp_path)
    monkeypatch.setattr(o.owner, "ROOT", tmp_path)

    def ref(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(value).encode())
        return o.owner.pin(path)

    adapter = ref("adapter/manifest.json", {"synthetic": "adapter"})
    seed = {
        "schema_id": "decision3-q95-live-helper-selection-v1",
        "binding": {
            "profile": "field-syntax-preserving-base-helper-v1",
            "qualification_sha256": o.Q95_SHA,
            "source_count": 95,
            "seconds": 20,
            "tokens": 128,
            "context": 8192,
            "boot_id": "synthetic-boot",
        },
    }
    seed.update({key: {"sha256": digest} for key, digest in o.Q95_MANIFESTS.items()})
    seed_ref = ref("seed.json", seed)
    plan = {
        "sources": [adapter, seed_ref],
        "adapter": "adapter",
        "unit": "selected.service",
        "cgroup": "/selected",
        "boot_id": "synthetic-boot",
    }
    descriptor = {
        "schema_id": "gemma-native-nf4-v2-q95-helper-selection-v1",
        "helper_seed_ref": seed_ref,
        "native_co_resident": {
            "profile": "gemma-native-nf4-v2-with-q95-v1",
            "unit": plan["unit"],
            "cgroup": plan["cgroup"],
            "memory_max_bytes": o.owner.CAP,
            "adapter_manifest": adapter,
        },
    }
    selected = ref("descriptor.json", descriptor)
    plan["sources"].append(selected)
    plan["helper_selection"] = selected
    return plan, descriptor, seed, ref


def test_owner_requires_exact_descriptor_and_same_helper_binding(selected_owner):
    plan, descriptor, seed, _ = selected_owner
    assert o.descriptor(plan) == (descriptor, seed)


@pytest.mark.parametrize(
    "problem",
    [
        "descriptor_unpinned",
        "seed_unpinned",
        "wrong_unit",
        "wrong_cap",
        "wrong_adapter",
        "wrong_q95",
        "wrong_boot",
    ],
)
def test_owner_wrong_selection_or_resource_binding_refuses(selected_owner, problem):
    plan, descriptor, seed, ref = selected_owner
    if problem == "descriptor_unpinned":
        plan["sources"].remove(plan["helper_selection"])
    elif problem == "seed_unpinned":
        plan["sources"].remove(descriptor["helper_seed_ref"])
    else:
        if problem in {"wrong_q95", "wrong_boot"}:
            seed["binding"]["qualification_sha256" if problem == "wrong_q95" else "boot_id"] = (
                "different"
            )
            seed_ref = ref("seed2.json", seed)
            plan["sources"].append(seed_ref)
            descriptor["helper_seed_ref"] = seed_ref
        else:
            descriptor["native_co_resident"][
                {
                    "wrong_unit": "unit",
                    "wrong_cap": "memory_max_bytes",
                    "wrong_adapter": "adapter_manifest",
                }[problem]
            ] = "different"
        descriptor_ref = ref("descriptor2.json", descriptor)
        plan["sources"].append(descriptor_ref)
        plan["helper_selection"] = descriptor_ref
    with pytest.raises(ValueError):
        o.descriptor(plan)


def test_q95_installed_source_manifest_hash_is_checked_without_import(tmp_path, monkeypatch):
    monkeypatch.setattr(o, "ROOT", tmp_path)
    monkeypatch.setattr(o.owner, "ROOT", tmp_path)
    source, runtime = tmp_path / "q95", tmp_path / "installed"
    source.mkdir()
    runtime.mkdir()
    seed = {"source_root": str(source)}
    for kind, count, root in (("execution", 95, source), ("runtime", 348, runtime)):
        files = {}
        for number in range(count):
            path = root / f"file{number}.py"
            path.write_text(
                "raise AssertionError('selected bytes must never be imported in owner')\n"
            )
            identity = o.owner.pin(path)
            files[path.name if kind == "execution" else str(path)] = {
                k: identity[k] for k in ("sha256", "size_bytes")
            }
        manifest = tmp_path / (kind + ".json")
        manifest.write_text(json.dumps({"files": files}))
        seed[kind] = o.owner.pin(manifest)
    o.helper_static(seed)
    (runtime / "file347.py").write_text("changed\n")
    with pytest.raises(ValueError):
        o.helper_static(seed)


@pytest.fixture
def closed_control(tmp_path, monkeypatch):
    output = tmp_path / "browser"
    output.mkdir()
    (output / "receipt.json").write_text('{"stage":"complete_pending_external_closure"}')
    control_directory = tmp_path / "q95-control"
    control_directory.mkdir()
    monkeypatch.setattr(o, "ROOT", tmp_path)
    monkeypatch.setattr(o.owner, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(o, "original_closed", lambda *_: calls.append("original-whole-closure"))
    monkeypatch.setattr(o, "descriptor", lambda _: ({}, {"binding": {"same": "Q95"}}))
    monkeypatch.setattr(o, "helper_static", lambda _: calls.append("Q95-static-closure"))
    monkeypatch.setattr(o.owner, "run_paths", lambda _: (tmp_path, output))
    groups = []
    for name in ("known4", "public6"):
        folder = output / name
        folder.mkdir()
        (folder / "external-q95-request-index.json").write_text("{}")
        groups.append({"group_directory": str(folder), "external_q95_requests": 1})
    receipt = {
        "profile": "jev-current-nf4-v7-q95-browser-v1",
        "stage": "complete_pending_external_closure",
        "primary_error": None,
        "selection_sha256": "a" * 64,
        "helper_binding": {"same": "Q95"},
        "native_receipt": o.owner.pin(output / "receipt.json"),
        "group_indexes": groups,
        "native_journal_covers_external_helper": False,
        "external_owned_closure_required": True,
    }
    path = control_directory / "receipt.json"
    path.write_text(json.dumps(receipt))
    return {
        "path": path,
        "receipt": receipt,
        "output": output,
        "calls": calls,
        "plan": {"helper_selection": {"sha256": "a" * 64}},
    }


def test_original_whole_closure_and_same_helper_control_both_required(closed_control):
    f = closed_control
    o.verify_closed({}, {}, f["plan"], f["output"])
    assert f["calls"] == ["original-whole-closure", "Q95-static-closure"]


@pytest.mark.parametrize(
    "problem",
    ["missing_control", "primary", "wrong_binding", "wrong_receipt", "incomplete", "missing_index"],
)
def test_partial_or_unbound_control_cannot_be_reported_complete(closed_control, problem):
    f = closed_control
    if problem == "missing_control":
        f["path"].unlink()
    elif problem == "missing_index":
        (f["output"] / "public6/external-q95-request-index.json").unlink()
    else:
        key, value = {
            "primary": ("primary_error", {"type": "refusal"}),
            "wrong_binding": ("helper_binding", {"foreign": True}),
            "wrong_receipt": ("native_receipt", {}),
            "incomplete": ("group_indexes", f["receipt"]["group_indexes"][:1]),
        }[problem]
        f["receipt"][key] = value
        f["path"].write_text(json.dumps(f["receipt"]))
    with pytest.raises((ValueError, FileNotFoundError)):
        o.verify_closed({}, {}, f["plan"], f["output"])


def test_original_whole_closure_refusal_is_never_bypassed(closed_control, monkeypatch):
    def refused(*_):
        raise ValueError("Original unit/cgroup closure unknown")

    monkeypatch.setattr(o, "original_closed", refused)
    with pytest.raises(ValueError, match="closure unknown"):
        o.verify_closed({}, {}, closed_control["plan"], closed_control["output"])
    assert not closed_control["calls"]


def test_external_request_join_preserves_hashes_and_honest_population_scope(tmp_path):
    service, output = tmp_path / "service", tmp_path / "browser"
    directory = service / "requests" / ("a" * 32)
    directory.mkdir(parents=True)
    output.mkdir()

    def pin(path):
        blob = path.read_bytes()
        return {
            "path": str(path),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "size_bytes": len(blob),
        }

    raw_path = directory / "raw.json"
    raw_path.write_text(
        json.dumps(
            {"request_id": "a" * 32, "guard_restored": True, "server_terminal": True, "error": None}
        )
    )
    final = {
        "request_id": "a" * 32,
        "completed": True,
        "final_gate_measured": True,
        "raw_sha256": pin(raw_path)["sha256"],
        "selected_status": 422,
    }
    (directory / "outer-completion.json").write_text(json.dumps(final))
    native = SimpleNamespace(pin=pin, write=lambda path, value: path.write_text(json.dumps(value)))
    assert d.index_external_requests(service, set(), native, output) == 1
    evidence = json.loads((output / "external-q95-request-index.json").read_bytes())
    assert evidence["request_ids"] == ["a" * 32] and evidence["pruned"] is False
    assert "not a fabricated per-client" in evidence["requests"]["a" * 32]["scope"]
    final["raw_sha256"] = "f" * 64
    (directory / "outer-completion.json").write_text(json.dumps(final))
    with pytest.raises(ValueError):
        d.index_external_requests(service, set(), native, output)
