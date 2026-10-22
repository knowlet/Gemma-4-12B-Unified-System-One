"""CPU source composition, with tmp-only impossible-live metadata and fake I/O."""

import ast
import copy
import hashlib
import importlib
import json
import shutil
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import checked_source  # noqa: E402


def local(path, name):
    value = types.ModuleType(name)
    value.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), vars(value))
    return value


driver = local(HERE / "control_driver.py", "_find04_test_driver")


@pytest.fixture
def selected(tmp_path, monkeypatch):
    """Copy small selected source bytes only; never a tensor or live seed."""
    for row in driver.source_refs():
        target = tmp_path / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / row["path"], target)
    new_here = tmp_path / HERE.relative_to(ROOT)
    d = local(new_here / "control_driver.py", "_find04_test_tmp_driver")
    monkeypatch.setattr(checked_source, "ROOT", tmp_path)
    run_directory = new_here / "SYNTHETIC-NOT-LIVE"
    run_directory.mkdir()
    plan = {
        "find_binding": d.binding(),
        "sources": d.source_refs(),
        "run_directory": run_directory.relative_to(tmp_path).as_posix(),
        "output": (run_directory / "browser").relative_to(tmp_path).as_posix(),
        "helper_selection": {"sha256": "0" * 64},
    }
    return d, plan, run_directory


def test_default_dry_has_no_model_helper_or_metadata_writes(capsys):
    before = sorted(path.name for path in HERE.iterdir())
    for name in ("control_driver.py", "prepare_control.py", "browser_entry.py"):
        assert local(HERE / name, "_find04_dry_" + name).main([]) == 0
    assert "DRY_NO_MODEL_OR_BROWSER" in capsys.readouterr().out
    assert sorted(path.name for path in HERE.iterdir()) == before
    assert not any(
        (HERE / name).exists()
        for name in ("launch-plan-01.json", "helper-selection-01.json", "v2-q95-find-04")
    )
    assert "torch" not in sys.modules and "transformers" not in sys.modules


def test_retained_control03_and_all84_original_project_rows():
    rows = driver.source_refs()
    original = json.loads((ROOT / driver.BASE_PLAN).read_bytes())["sources"]
    old = json.loads((ROOT / driver.SERVICE / "selection.json").read_bytes())
    new = json.loads((HERE / "service-selection.json").read_bytes())
    project = new["project"]["descriptor"]["project_sources"]
    assert len(original) == 785 and len(project) == 84
    assert all(row in original and row in rows for row in project)
    assert all(row in project for row in old["project"]["descriptor"]["project_sources"])
    control = json.loads((ROOT / driver.CONTROL03 / "source-union.json").read_bytes())
    assert all(row in rows for row in control["sources"])
    assert new["project"]["sha256"] == driver.PROJECT_SHA
    assert (
        driver.sha((HERE / "service-selection.json").read_bytes()) == driver.EXPANDED_SELECTION_SHA
    )


def test_original22_refuses_actual_whole_main_dependency_new84_imports_it(selected):
    d, _, _ = selected
    service = checked_source.module(
        d.SERVICE + "/selection.py", d.SERVICE_SHA, "_find04_test_service"
    )
    old = (d.ROOT / d.SERVICE / "selection.json").read_bytes()
    new = (d.HERE / "service-selection.json").read_bytes()
    try:
        with service.selected(old, d.SERVICE_SELECTION_SHA, profile=d.SERVICE_PROFILE):
            with pytest.raises(
                ValueError, match="Unselected project module import refused: s1.browser_evidence"
            ):
                importlib.import_module("s1.browser_evidence")
        assert not any(name == "s1" or name.startswith("s1.") for name in sys.modules)
        with service.selected(new, d.EXPANDED_SELECTION_SHA, profile=d.SERVICE_PROFILE):
            for name in ("browser_evidence", "browser_action_adapter", "quantization", "resources"):
                value = importlib.import_module("s1." + name)
                assert value.__loader__.__class__.__name__ == "ProjectSources"
                # The checked a39e importer deliberately fixes the real repo
                # root. Metadata/output fixtures are tmp-only; these are the
                # actual authenticated real source bytes, never model imports.
                assert value.__file__ == str(ROOT / "src/s1" / (name + ".py"))
                assert (
                    Path(value.__file__).read_bytes()
                    == (d.ROOT / "src/s1" / (name + ".py")).read_bytes()
                )
            assert "torch" not in sys.modules and "transformers" not in sys.modules
        assert not any(name == "s1" or name.startswith("s1.") for name in sys.modules)
    finally:
        sys.modules.pop("_find04_test_service", None)


def test_actual_selectors_private_derivations_and_original_run_groups(selected):
    d, plan, output = selected
    calls = []
    with d.selected_native(plan) as (parent, native, v7, proofs):
        assert parent._find_selected_native is native
        assert native.run_groups.__globals__ is vars(native)
        assert "s1._experimental_find_api" in native.make_application.__code__.co_names
        assert proofs["service"] == d.expected_service_proof()
        assert (
            proofs["control_execute"]["declared_edits"]
            == proofs["native_run_groups"]["declared_edits"]
            == 1
        )
        native.run_client = fake_client(native, calls)
        groups = native.run_groups(
            output / "browser",
            output / "FAKE-UPSTREAM",
            "http://127.0.0.1:1",
            "FAKE-NOT-SECRET",
            "/FAKE-CHROMIUM",
        )
        assert [(row["group"], row["denominator"]) for row in groups] == [
            ("known4", 4),
            ("public6", 6),
        ]
        assert [row["case_module"] for row in v7.groups] == ["original", "heldout"]
        assert len(calls) == 2
    assert not any(name == "s1" or name.startswith("s1.") for name in sys.modules)
    assert "jev_browser_profile_v7_benchmark" not in sys.modules


def test_find04_genuine_canonical60_constructor_hook_uses_private16_without_globals_change(
    selected,
):
    """Actual selected bundle hook and canonical loader; no CDP/HTTP/Browser."""
    d, plan, run_directory = selected
    calls = []

    def no_cdp(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("Budget-only source regression must never create Browser or issue CDP")

    with d.selected_native(plan) as (_, _, v7, _):
        bundle, manifest = v7.prepare_bundle(
            run_directory / "browser/known4/runtime-bundle", case_module="original"
        )
        assert manifest["agent_profile"] == d.CLIENT_PROFILE
        assert d.sha((bundle / "find_client_selector.py").read_bytes()) == d.CLIENT_SHA
        assert any(
            row["path"] == d.CLIENT_PARENT and row["sha256"] == d.CLIENT_PARENT_SHA
            for row in plan["sources"]
        )
        runner = types.ModuleType("_find04_genuine_runner")
        runner.__file__ = str(bundle / "runner.py")
        sys.modules[runner.__name__] = runner
        sys.path.insert(0, str(bundle))
        try:
            exec(
                compile(
                    (bundle / "runner.py").read_bytes(), runner.__file__, "exec", dont_inherit=True
                ),
                vars(runner),
            )
            with runner.load_upstream(
                d.ROOT / "artifacts/jev-runtime/upstream", types.SimpleNamespace(cdp=no_cdp)
            ) as (agent_module, model_module):
                original_agent, original_choose = agent_module.Agent, agent_module.choose
                assert (
                    agent_module.MAX_STEPS == original_agent.command.__globals__["MAX_STEPS"] == 60
                )
                try:
                    with runner.model_adapter(
                        agent_module,
                        model_module,
                        endpoint="http://fixture.invalid/v1/systemone",
                        model="FAKE-POLICY",
                        api_key="FAKE-ONLY",
                        text_endpoint="http://fixture.invalid",
                        text_model="FAKE-Q95",
                        text_api_key="FAKE-ONLY",
                    ) as (records, _):
                        installed_choose = agent_module.choose
                        cls = runner._selected_find_agent_class(agent_module, model_module)
                        assert issubclass(cls, original_agent)
                        assert (
                            agent_module.Agent is original_agent
                            and agent_module.choose is installed_choose
                        )
                        assert (
                            agent_module.MAX_STEPS
                            == original_agent.command.__globals__["MAX_STEPS"]
                            == 60
                        )
                        selected_agent = object.__new__(cls)
                        selected_agent.state = {
                            "decision": {
                                "operation": "FIND_TEXT",
                                "choice": "find_text",
                                "target": None,
                            },
                            "page": {"fingerprint": "FAKE-MARKER"},
                            "history": [{}] * 16,
                            "status": "ready",
                        }
                        with pytest.raises(ValueError, match="sixteen-action demo budget"):
                            selected_agent.command("act", {"fingerprint": "FAKE-MARKER"})
                        assert (
                            selected_agent.state["status"] == "blocked"
                            and selected_agent.state["decision"] is None
                        )
                        selected_agent.state = {
                            "browser": types.SimpleNamespace(fresh=lambda _: True),
                            "page": {"fingerprint": "FAKE-MARKER"},
                            "started_at": 0,
                            "status": "ready",
                            "decision": None,
                            "decisions": [{}] * 32,
                        }
                        with pytest.raises(ValueError, match="model-call budget"):
                            selected_agent.command("predict")
                        assert not records and not calls
                    assert (
                        agent_module.choose is original_choose
                        and agent_module.Agent is original_agent
                    )
                    assert agent_module.MAX_STEPS == 60
                finally:
                    model_module.CLIENT.close()
        finally:
            sys.path.remove(str(bundle))
            sys.modules.pop(runner.__name__, None)
            sys.modules.pop("find_client_selector", None)
        v7.verify_bundle(bundle, manifest)


def fake_client(native, calls):
    def client(argv, env, output):
        """No process/browser/HTTP; only record actual unmodified native argv."""
        ids = argv[argv.index("--public-cases") + 1 :]
        expected = native.KNOWN4 if len(calls) == 0 else native.PUBLIC6
        assert ids == list(expected)
        assert argv[argv.index("--max-steps") + 1] == "16"
        assert argv[argv.index("--seed") + 1] == "20261014"
        assert argv[argv.index("--observation-profile-helper-sha256") + 1] == native.V7_SHA
        result = Path(argv[argv.index("--output-dir") + 1])
        result.mkdir(parents=True)
        (result / "summary.json").write_text(
            json.dumps({"attempts": len(ids), "successes": 0, "observation_profile_failed": False})
        )
        calls.append((argv, output))

    return client


@pytest.fixture
def completed(selected, monkeypatch):
    """Actual selected composition; native main body is explicitly a CPU fake."""
    d, plan, run_directory = selected
    calls, events = [], []
    helper_service = run_directory / "SYNTHETIC-Q95-NOT-LIVE"
    helper_service.mkdir()
    synthetic_binding = {
        "profile": "field-syntax-preserving-base-helper-v1",
        "pid": 0,
        "start_ticks": 0,
        "boot_id": "SYNTHETIC-NOT-LIVE",
    }

    class Checker:
        def __init__(self, descriptor, selected_sha, output):
            assert descriptor == b"SYNTHETIC-NOT-LIVE" and selected_sha == "0" * 64
            self.binding, self._value = (
                synthetic_binding,
                {"service_directory": str(helper_service)},
            )

        def __call__(self, binding, phase):
            assert binding == synthetic_binding
            events.append(phase)

        def text_connection(self):
            return {
                "endpoint": "http://127.0.0.1:1",
                "model": "SYNTHETIC-Q95",
                "api_key": "FAKE-NOT-SECRET",
            }

    class Control:
        def __init__(self):
            self.requests, self.failure = [], None

        def healthy(self):
            assert self.failure is None

        def fail(self, error):
            self.failure = type(error).__name__

    real_selected = d.selected_native
    index_hash = []

    def selected_harness(plan):
        from contextlib import contextmanager

        @contextmanager
        def entered():
            with real_selected(plan) as value:
                parent, native, _, _ = value
                parent.selected_module = lambda *_: pytest.fail(
                    "Original unmodified native reload must never occur"
                )
                native.run_client = fake_client(native, calls)
                adapter = d.ROOT / "artifacts/synthetic-adapter/adapter"
                adapter.mkdir(parents=True)
                original_manifest = (
                    ROOT
                    / "artifacts/jev-runtime/native-v2-training-20261022-03/fit-01/adapter/manifest.json"
                )
                shutil.copyfile(original_manifest, adapter / "manifest.json")
                upstream = d.ROOT / "artifacts/synthetic-upstream"
                upstream.mkdir()

                def fake_run(metadata, **kwargs):
                    # Actual original metadata_plan plus both wrappers must have
                    # put this binding into the very value used by run.
                    assert metadata["find_binding"] == d.binding()
                    assert metadata["external_text_helper"]["binding"] == synthetic_binding
                    assert metadata["profile"] == d.PROFILE
                    assert len(native.composition_paths()) >= len(d.source_refs())
                    out = Path(metadata["output"])
                    out.mkdir()
                    groups = native.run_groups(
                        out, upstream, "http://127.0.0.1:1", "FAKE-NOT-SECRET", "/FAKE-CHROMIUM"
                    )
                    receipt = {
                        **metadata,
                        "stage": "complete_pending_external_closure",
                        "planned_denominator": 10,
                        "attempts": 10,
                        "groups": groups,
                    }
                    native.write(out / "receipt.json", receipt)
                    native.write(out / "output-index.json", {"files": native.output_inventory(out)})
                    index_hash.append(native.pin(out / "output-index.json"))
                    return receipt

                native.run = fake_run

                def fake_main(rest):
                    metadata = native.metadata_plan(
                        adapter, d.ROOT / plan["output"], upstream, "shared-elements-v2"
                    )
                    native.run(
                        metadata, expected_boot_id="SYNTHETIC-NOT-LIVE", chromium="/FAKE-CHROMIUM"
                    )
                    return 0

                native.main = fake_main
                yield value

        return entered()

    monkeypatch.setattr(d, "selected_native", selected_harness)
    helper = types.SimpleNamespace(GemmaHelperChecker=Checker, PARENT=d.ROOT / "FAKE-PARENT.py")
    boundary = types.SimpleNamespace(ControlState=Control)
    assert d.execute(plan, b"SYNTHETIC-NOT-LIVE", helper, boundary, ["--run"]) == 0
    assert len(calls) == 2
    assert events == [
        "before-native-driver-imports",
        "before-browser-group-0",
        "after-browser-group-0",
        "before-browser-group-1",
        "after-browser-group-1",
        "after-native-service-drain",
    ]
    output = d.ROOT / plan["output"]
    assert not (output / "find-selection-receipt.json").exists()
    assert (run_directory / "find-selection-receipt.json").is_file()
    assert {
        "sha256": hashlib.sha256((output / "output-index.json").read_bytes()).hexdigest(),
        "size_bytes": (output / "output-index.json").stat().st_size,
    } == index_hash[0]
    return d, plan, output


def test_actual_composition_metadata_binding_ten_tasks_and_external_indexes(completed):
    d, plan, output = completed
    d.verify_outputs(plan, output)
    saved = json.loads((output.parent / "find-selection-receipt.json").read_bytes())
    assert saved["binding"] == d.binding() and saved["model_quality_qualified"] is False


def test_actual_external_index_raw_final_join_and_tamper_refusal(completed):
    d, plan, output = completed
    parent = checked_source.module(d.PARENT, d.PARENT_SHA, "_find04_test_index_parent")
    native = checked_source.module(d.NATIVE, d.NATIVE_SHA, "_find04_test_pin_native")
    service = output.parent / "SYNTHETIC-Q95-NOT-LIVE"
    request_id = "a" * 32
    request = service / "requests" / request_id
    request.mkdir(parents=True)
    raw = {"request_id": request_id, "guard_restored": True, "server_terminal": True, "error": None}
    native.write(request / "raw.json", raw)
    final = {
        "request_id": request_id,
        "completed": True,
        "final_gate_measured": True,
        "raw_sha256": native.pin(request / "raw.json")["sha256"],
        "selected_status": 200,
    }
    native.write(request / "outer-completion.json", final)
    index_out = output.parent / "SYNTHETIC-INDEX"
    index_out.mkdir()
    assert parent.index_external_requests(service, set(), native, index_out) == 1
    shutil.copyfile(
        index_out / "external-q95-request-index.json",
        output / "known4/external-q95-request-index.json",
    )
    control_path = output.parent / "q95-control/receipt.json"
    control = json.loads(control_path.read_bytes())
    control["group_indexes"][0]["external_q95_requests"] = 1
    control_path.write_text(json.dumps(control))
    d.verify_outputs(plan, output)
    final["completed"] = False
    (request / "outer-completion.json").write_text(json.dumps(final))
    with pytest.raises(ValueError, match="raw/final pins"):
        d.verify_outputs(plan, output)


def test_owner_closed_delegates_original_authority_then_actual_find_verifier(
    completed, monkeypatch
):
    d, plan, output = completed
    owner = local(d.HERE / "launch_browser.py", "_find04_actual_closed_owner")
    control = json.loads((output.parent / "q95-control/receipt.json").read_bytes())
    calls = []
    monkeypatch.setattr(
        owner, "original_closed", lambda *_: calls.append("original-whole-unit-closure")
    )
    monkeypatch.setattr(owner, "descriptor", lambda _: ({}, {"binding": control["helper_binding"]}))
    monkeypatch.setattr(owner, "helper_static", lambda _: calls.append("Q95-static-authority"))
    monkeypatch.setattr(owner.owner, "run_paths", lambda _: (output.parent, output))
    owner.verify_closed({}, {}, plan, output)
    assert calls == ["original-whole-unit-closure", "Q95-static-authority"]
    (output.parent / "find-selection-receipt.json").unlink()
    with pytest.raises(ValueError, match="source/evidence"):
        owner.verify_closed({}, {}, plan, output)


@pytest.mark.parametrize(
    "mutation",
    [
        "client_profile",
        "service_proof",
        "native_binding",
        "denominator",
        "missing_bundle",
        "bundle_bytes",
        "external_count",
        "wrong_group",
        "native_pin_path",
    ],
)
def test_actual_output_closure_refuses_wrong_branch_incomplete_or_tampered_evidence(
    completed, mutation
):
    d, plan, output = completed
    saved_path = output.parent / "find-selection-receipt.json"
    saved = json.loads(saved_path.read_bytes())
    if mutation in {"client_profile", "service_proof"}:
        if mutation == "client_profile":
            saved["binding"]["client_profile"] = "standard-v7"
        else:
            saved["proofs"]["service"]["derived_sha256"] = d.NATIVE_SHA
        saved_path.write_text(json.dumps(saved))
    elif mutation in {"native_binding", "denominator"}:
        path = output / "receipt.json"
        native = json.loads(path.read_bytes())
        if mutation == "native_binding":
            del native["find_binding"]
        else:
            native["attempts"] = 9
        path.write_text(json.dumps(native))
    elif mutation == "missing_bundle":
        (output / "public6/runtime-bundle/bundle-manifest.json").unlink()
    elif mutation == "bundle_bytes":
        (output / "public6/runtime-bundle/runner.py").write_text("# forged standard runner\n")
    else:
        path = output.parent / "q95-control/receipt.json"
        control = json.loads(path.read_bytes())
        if mutation == "external_count":
            control["group_indexes"][0]["external_q95_requests"] = 1
        elif mutation == "wrong_group":
            control["group_indexes"].reverse()
        else:
            control["native_receipt"]["path"] = "invented"
        path.write_text(json.dumps(control))
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        d.verify_outputs(plan, output)


def test_actual_selected_scope_final_checks_run_on_exception_and_preserve_primary(
    selected, monkeypatch
):
    d, plan, _ = selected
    original = d.authenticate_plan
    calls = []

    def checked(plan):
        calls.append(1)
        if len(calls) > 1:
            raise ValueError("simulated final source refusal")
        return original(plan)

    monkeypatch.setattr(d, "authenticate_plan", checked)
    with pytest.raises(RuntimeError, match="original body failure") as caught:
        with d.selected_native(plan):
            raise RuntimeError("original body failure")
    assert len(calls) == 2
    assert any("simulated final source refusal" in note for note in caught.value.__notes__)
    assert not any(name == "s1" or name.startswith("s1.") for name in sys.modules)


def test_owner_three_reused_sources_and_helper_single_unit_inverse():
    old = ROOT / driver.CONTROL03
    for name in ("browser_entry.py", "checked_source.py", "policy_boundary.py"):
        assert (HERE / name).read_bytes() == (old / name).read_bytes()
    assert (HERE / "gemma_helper.py").read_bytes().replace(
        b"jev-current-nf4-v2-q95-find-20261023-04.service",
        b"jev-current-nf4-v2-q95-20261023-03.service",
    ) == (old / "gemma_helper.py").read_bytes()
    owner = local(HERE / "launch_browser.py", "_find04_test_owner")
    assert owner.owner.FORMATS == {"v2-q95-find": "shared-elements-v2"}
    assert owner.owner.CAP == 40 * 1024**3
    assert owner.owner.execute.__code__.co_filename == str(owner.PARENT)
    assert owner.owner.readback.__code__.co_filename == str(owner.PARENT)


def test_explicit_binding_and_source_row_conflicts_refuse(selected):
    d, plan, _ = selected
    wrong = copy.deepcopy(plan)
    wrong["find_binding"]["client_profile"] = "standard"
    with pytest.raises(ValueError, match="binding"):
        d.authenticate_plan(wrong)
    wrong = copy.deepcopy(plan)
    wrong["sources"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="union"):
        d.authenticate_plan(wrong)
    wrong = copy.deepcopy(plan)
    wrong["sources"].append(copy.deepcopy(wrong["sources"][0]))
    with pytest.raises(ValueError, match="Unique"):
        d.authenticate_plan(wrong)


def metadata_fixture(tmp_path, monkeypatch, *, conflict=False):
    """Reuse the frozen control03 impossible-live fixture body, no old seed."""
    path = ROOT / driver.CONTROL03 / "test_control03.py"
    blob = path.read_bytes()
    assert (
        hashlib.sha256(blob).hexdigest()
        == "8fd54013a7c5aeda4165f7a94d55a8d850b2455e424a85e8da04e3e56f9efe53"
    )
    node = next(
        n
        for n in ast.parse(blob).body
        if isinstance(n, ast.FunctionDef) and n.name == "metadata_factory"
    )
    node.decorator_list = []
    owner = local(HERE / "launch_browser.py", "_find04_metadata_owner")

    def copy_find_sources(prepare):
        for row in driver.source_refs():
            target = tmp_path / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / row["path"], target)
        if conflict:
            base = json.loads(prepare.BASE_PLAN.read_bytes())
            row = next(row for row in base["sources"] if row["path"] == driver.NATIVE)
            row["sha256"] = "0" * 64
            data = json.dumps(base).encode()
            prepare.BASE_PLAN.write_bytes(data)
            prepare.BASE_SHA = hashlib.sha256(data).hexdigest()

    # This one fixture-only insertion copies the additional frozen small FIND
    # sources immediately before calling the actual new metadata factory.
    index = next(
        i
        for i, item in enumerate(node.body)
        if isinstance(item, ast.Expr)
        and isinstance(item.value, ast.Call)
        and isinstance(item.value.func, ast.Attribute)
        and item.value.func.attr == "prepare"
    )
    node.body.insert(index, ast.parse("copy_find_sources(prepare)").body[0])
    namespace = {
        "Path": Path,
        "json": json,
        "hashlib": hashlib,
        "shutil": shutil,
        "ROOT": ROOT,
        "HERE": HERE,
        "o": owner,
        "local": lambda name: local(HERE / name, "_find04_metadata_" + name),
        "copy_find_sources": copy_find_sources,
    }
    monkeypatch.setattr(checked_source, "ROOT", tmp_path)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path) + "::find04-fixture",
            "exec",
        ),
        namespace,
    )
    plan, new_here = namespace["metadata_factory"](tmp_path, monkeypatch)
    return owner, plan, new_here


def test_actual_new_metadata_factory_retains785_and_passes_real_owner_validation(
    tmp_path, monkeypatch
):
    owner, plan, new_here = metadata_fixture(tmp_path, monkeypatch)
    assert plan["source_population_counts"]["retained_original_sources"] == 785
    assert plan["source_population_counts"]["additional_installed_runtime_population"] == 348
    assert plan["find_binding"] == driver.binding()
    assert plan["arm"] == "v2-q95-find" and plan["attempt"] == "04"
    assert plan["unit"] == "jev-current-nf4-v2-q95-find-20261023-04.service"
    assert owner.owner.run_paths(plan) == (
        new_here / "v2-q95-find-04",
        new_here / "v2-q95-find-04/browser",
    )
    assert owner.validate_plan(plan) == plan
    descriptor = json.loads((tmp_path / plan["helper_selection"]["path"]).read_bytes())
    seed = json.loads((tmp_path / descriptor["helper_seed_ref"]["path"]).read_bytes())
    assert seed["binding"]["pid"] == seed["binding"]["start_ticks"] == 0
    assert seed["expected_memtotal"] == 1 and seed["worker"]["cgroup"] == "/SYNTHETIC-NOT-LIVE"
    assert plan["preparation_status"] == "METADATA_ONLY_NOT_ACTUAL_ADMISSION"


def test_actual_new_metadata_factory_refuses_original_source_identity_conflict(
    tmp_path, monkeypatch
):
    with pytest.raises(ValueError, match="contradictory original source identity"):
        metadata_fixture(tmp_path, monkeypatch, conflict=True)
    new_here = tmp_path / HERE.relative_to(ROOT)
    assert not (new_here / "launch-plan-01.json").exists()
    assert not (new_here / "helper-selection-01.json").exists()
