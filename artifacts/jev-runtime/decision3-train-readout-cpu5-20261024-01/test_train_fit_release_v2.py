"""Source-authored fake inventory negatives; no actual exports or NumPy.

These tests are intentionally not an execution/qualification receipt. They may
be run later, after the root explicitly opens the execution boundary.
"""

import copy
import types
import unittest
from pathlib import Path
from unittest import mock

import fit_head_released_v2 as fit_head
import train_loader_released_v1 as loader


def fake_inventory():
    requests, labels = [], []
    for family in loader.FAMILIES:
        for ancestor_index in (1, 2, 3):
            ancestor = family + "/workflow-" + str(ancestor_index)
            for variant in range(8):
                heads = ["operation", "click_target", "type_text_target", "select_target"]
                # Real criterion/head order is a join key, never a fixed zip.
                if variant % 2:
                    heads = ["operation", "select_target", "click_target", "type_text_target"]
                questions = {
                    head: {
                        "type": "choice",
                        "criteria": {head + ":second": "Second", head + ":first": "First"},
                    }
                    for head in heads
                }
                request = {"state": {"fake": family + str(variant)}, "questions": questions}
                identity = {
                    "case_id": family + "-" + str(ancestor_index) + "-" + str(variant),
                    "dsl_version": loader.DSL_VERSION,
                    "family_id": family,
                    "ancestor_id": ancestor,
                    "request_sha256": loader.digest(request),
                    "native_payload_sha256": loader.digest(
                        {key: request[key] for key in ("state", "questions")}
                    ),
                }
                supervised = {}
                for head in ("operation", heads[1]):
                    criteria = list(questions[head]["criteria"])
                    supervised[head] = {
                        "criteria_ids": criteria,
                        "valid_ids": criteria[1:],
                        "valid_positions": [1],
                        "soft_target": [0.0, 1.0],
                    }
                requests.append({**identity, "request": request})
                labels.append({**identity, "heads": supervised})
    summary = {
        "dsl_version": loader.DSL_VERSION,
        "split": "train",
        "case_count": 96,
        "family_ids": sorted(loader.FAMILIES),
        "ancestor_ids": sorted(loader.ANCESTORS),
        "is_authored_synthetic_not_NN": True,
    }
    return requests, labels, summary


def fake_acceptance():
    # In-memory schema fixture only. It has no accepted SHA, payloads, proof refs
    # or capability and can never be consumed by load_training under the hold.
    selection = {
        key: {"fake": key}
        for key in (
            "train_inputs",
            "model",
            "source_records",
            "collector_source",
            "root_np5_prerequisite_ref",
        )
    }
    selection.update(projection=loader.PROJECTION, baseline={"sha256": "1" * 64})
    entry = {
        "ancestor_id": loader.ANCESTORS[0],
        "family_id": loader.FAMILIES[0],
        "chunk": {"path": "fake", "sha256": "2" * 64, "size_bytes": 1},
    }
    acceptance = {
        "schema": "root-closed-dsl01-train-chunk-acceptance-v1",
        "status": "COMMITTED",
        "training_eligible": True,
        "split": "train",
        "dsl_version": loader.DSL_VERSION,
        "ancestor_id": entry["ancestor_id"],
        "family_id": entry["family_id"],
        "producer_chunk": entry["chunk"],
        "baseline_sha256": "1" * 64,
        "transaction": {
            "case_count": 8,
            "call_budget_seconds": 20,
            "whole_budget_seconds": 120,
            "complete_call_seconds": [1.0] * 8,
            "whole_seconds": 10.0,
        },
    }
    acceptance.update(
        {
            key: selection[key]
            for key in (
                "train_inputs",
                "model",
                "projection",
                "source_records",
                "collector_source",
                "root_np5_prerequisite_ref",
            )
        }
    )
    return acceptance, entry, selection


def fake_np_header():
    selection = {
        "source_records": {
            "np5_actor": {
                "path": str(loader.HERE / "invented-np5.py"),
                "sha256": "a" * 64,
                "size_bytes": 100,
            }
        },
        "root_np5_prerequisite_ref": {
            "path": str(loader.HERE / "invented-accepted.json"),
            "sha256": "b" * 64,
            "size_bytes": 100,
        },
    }
    prerequisite = {
        "schema": "root-accepted-np5-qualification-v1",
        "status": "accepted",
        "independent_actual_review_claimed": False,
        "checks": {key: True for key in loader.NP_CHECKS},
        "source_refs": copy.deepcopy(selection["source_records"]),
        "owner_unit": loader.UNIT,
        "owner_profile": loader.OWNER_PROFILE,
        "proof_refs": {
            key: {"path": "invented-" + key, "sha256": "c" * 64, "size_bytes": 1}
            for key in ("root_execution", "owner_receipt", "current_closure")
        },
    }
    return prerequisite, selection


def fake_producer():
    """Only in-memory join schema; not files, real proof or accepted payload."""
    collector = {"path": str(loader.HERE / "collector.py"), "sha256": "a" * 64, "size_bytes": 10}
    dependency = {
        "path": str(loader.HERE / "invented-dependency.py"),
        "sha256": "b" * 64,
        "size_bytes": 20,
    }
    plan_ref = {
        "path": str(loader.HERE / "invented-plan.json"),
        "sha256": "c" * 64,
        "size_bytes": 30,
    }
    output = loader.OWNER / "actual-99"
    parent = output / "train-chunk"
    group = "/invented/" + loader.UNIT
    wrapper = {"pid": 10, "start_ticks": 100, "pgid": 10, "sid": 10, "uid": 1001, "cgroup": group}
    child = {**wrapper, "pid": 21, "start_ticks": 500}
    argv = [
        "/home/albert_hsieh/ComfyUI/.venv/bin/python",
        "-B",
        "-s",
        collector["path"],
        "--plan",
        plan_ref["path"],
        "--plan-sha256",
        plan_ref["sha256"],
        "--output",
        str(parent),
    ]
    observed = {"process": child, "exe": "/invented/python", "argv": argv, "cwd": str(loader.ROOT)}
    command = {
        "pid": 21,
        "returncode": 0,
        "observed_identity": observed,
        "resolved_executable": observed["exe"],
        "argv": argv,
        "cwd": str(loader.ROOT),
    }
    config = {
        "output": str(output),
        "command_count": 1,
        "sources": [
            {**ref, "path": str(Path(ref["path"]).relative_to(loader.ROOT))}
            for ref in (collector, dependency, plan_ref)
        ],
    }
    proof = {
        "sealed_config": config,
        "admission": {"actor": wrapper, "config_sha256": "d" * 64},
        "owner_receipt": {
            "state": {
                "actor_exit": {
                    "config_sha256": "d" * 64,
                    "entry_exit_code": 0,
                    "error": None,
                    "planned_command_count": 1,
                    "command_results": [command],
                }
            }
        },
    }
    selection = {
        "source_records": {"np5_actor": dependency},
        "collector_source": collector,
        "root_np5_prerequisite_ref": {"invented": True},
    }
    chunk = {
        "profile": "invented-collector-profile",
        "ancestor_id": loader.ANCESTORS[0],
        "selected_plan_sha256": plan_ref["sha256"],
        "owner": {
            "unit": loader.UNIT,
            "profile": loader.OWNER_PROFILE,
            "process": {
                "pid": 21,
                "starttime_ticks": 500,
                "pgid": 10,
                "sid": 10,
                "cgroup": "0::" + group,
                "executable": observed["exe"],
            },
        },
        "train_inputs": {"requests": {"invented": True}, "summary": {"invented": True}},
    }
    plan = {
        "schema": "decision3-dsl01-train-collector-plan-v1",
        "profile": chunk["profile"],
        "status": "selected_after_separate_review",
        "ancestor_id": chunk["ancestor_id"],
        "dependencies": selection["source_records"],
        "collector_source": collector,
        "train_inputs": chunk["train_inputs"],
        "np5_root_acceptance": selection["root_np5_prerequisite_ref"],
    }
    return chunk, parent, proof, selection, plan, plan_ref


class TrainInventoryTests(unittest.TestCase):
    def test_inventory_keeps_actual_order_and_skips_unlabelled_heads(self):
        requests, labels, summary = fake_inventory()
        grouped = loader.validate_train_inventory(requests, labels, summary)
        self.assertEqual([len(rows) for rows in grouped.values()], [8] * 12)
        self.assertNotEqual(
            list(requests[0]["request"]["questions"]), list(requests[1]["request"]["questions"])
        )
        for label in labels:
            self.assertEqual(len(label["heads"]), 2)
            self.assertNotIn("unused", label["heads"])
            for target in label["heads"].values():
                vector = loader.target_vector(target, target["criteria_ids"])
                self.assertEqual(vector[:2], [0.0, 1.0])
                self.assertEqual(vector[2:], [0.0] * 253)

    def test_rejects_foreign_family_and_changed_request_projection(self):
        for mutation in ("foreign-family", "changed-request", "changed-native-hash"):
            with self.subTest(mutation=mutation):
                requests, labels, summary = fake_inventory()
                if mutation == "foreign-family":
                    requests[0]["family_id"] = labels[0]["family_id"] = "foreign_dev_family"
                elif mutation == "changed-request":
                    requests[0]["request"]["state"]["fake"] = "changed-after-freeze"
                else:
                    requests[0]["native_payload_sha256"] = labels[0]["native_payload_sha256"] = (
                        "0" * 64
                    )
                with self.assertRaises(ValueError):
                    loader.validate_train_inventory(requests, labels, summary)

    def test_rejects_criterion_reordering_and_all_zero_targets(self):
        for mutation in ("reordered", "all-zero", "empty-valid"):
            with self.subTest(mutation=mutation):
                requests, labels, summary = fake_inventory()
                target = labels[0]["heads"]["operation"]
                if mutation == "reordered":
                    target["criteria_ids"].reverse()
                elif mutation == "all-zero":
                    target["soft_target"] = [0.0, 0.0]
                else:
                    target["valid_ids"], target["valid_positions"] = [], []
                with self.assertRaises(ValueError):
                    loader.validate_train_inventory(requests, labels, summary)

    def test_pending_marker_changed_root_pin_and_incomplete_call_are_rejected(self):
        for mutation in ("pending", "source", "prerequisite", "whole", "call"):
            with self.subTest(mutation=mutation):
                acceptance, entry, selection = fake_acceptance()
                acceptance = copy.deepcopy(acceptance)
                if mutation == "pending":
                    acceptance["status"] = "PENDING_TRAIN_ONLY"
                elif mutation == "source":
                    acceptance["source_records"] = {"fake": "different"}
                elif mutation == "prerequisite":
                    acceptance["root_np5_prerequisite_ref"] = {"fake": "different"}
                elif mutation == "whole":
                    acceptance["transaction"]["whole_seconds"] = 120.01
                else:
                    acceptance["transaction"]["complete_call_seconds"][-1] = 20.01
                with self.assertRaises(ValueError):
                    loader.validate_commit_header(acceptance, entry, selection)

    def test_holds_precede_argument_io_and_array_access(self):
        class Poison:
            def __getitem__(self, key):
                raise AssertionError("held loader touched selection")

            def __getattr__(self, name):
                raise AssertionError("held loader touched numerical runtime")

        with (
            mock.patch.object(loader, "SOURCE_ONLY", True),
            mock.patch.object(loader, "read_ref") as reader,
        ):
            with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
                loader.load_training(Poison(), Poison(), Poison(), check_deadline=lambda: None)
            reader.assert_not_called()
        with (
            mock.patch.object(fit_head, "SOURCE_ONLY", True),
            mock.patch.object(fit_head.argparse, "ArgumentParser") as parser,
            mock.patch.object(fit_head, "source") as importer,
        ):
            with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
                fit_head.main(["--selection", "/must-not-read"])
            parser.assert_not_called()
            importer.assert_not_called()
        with (
            mock.patch.object(fit_head, "SOURCE_ONLY", True),
            mock.patch.object(fit_head, "owned") as admission,
        ):
            with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
                fit_head.execute(Poison(), Poison())
            admission.assert_not_called()


class AcceptanceBindingTests(unittest.TestCase):
    def test_np_header_requires_true_checks_accepted_status_and_exact_sources(self):
        prerequisite, selection = fake_np_header()
        loader.validate_np_header(prerequisite, selection)
        for mutation in (
            "pending",
            "check_false",
            "missing_check",
            "independent",
            "source",
            "owner",
            "proof",
        ):
            changed = copy.deepcopy(prerequisite)
            if mutation == "pending":
                changed["status"] = "failed"
            elif mutation == "check_false":
                changed["checks"]["FP32_copy_exact"] = False
            elif mutation == "missing_check":
                changed["checks"].pop("root_exit0")
            elif mutation == "independent":
                changed["independent_actual_review_claimed"] = True
            elif mutation == "source":
                changed["source_refs"]["np5_actor"]["sha256"] = "0" * 64
            elif mutation == "owner":
                changed["owner_unit"] = "invented-other-owner"
            else:
                changed["proof_refs"].pop("current_closure")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                loader.validate_np_header(changed, selection)

    def test_rejected_np_prerequisite_fails_before_numerical_import(self):
        prerequisite, selection = fake_np_header()
        prerequisite["status"] = "rejected"
        with (
            mock.patch.object(loader, "read_ref", return_value=loader.canonical(prerequisite)),
            mock.patch.object(
                fit_head.importlib,
                "import_module",
                side_effect=AssertionError("must not import NumPy"),
            ) as importer,
        ):
            with self.assertRaisesRegex(ValueError, "accepted honest-root"):
                fit_head.numpy_runtime(loader, selection, lambda: None)
            importer.assert_not_called()

    def test_np_acceptance_cannot_attach_a_different_original_closure(self):
        prerequisite, selection = fake_np_header()
        config = {
            "profile": loader.OWNER_PROFILE,
            "unit": loader.UNIT,
            "output": str(loader.OWNER / "actual-99"),
        }
        documents = {
            "root_execution": {"exit_code": 0},
            "owner_receipt": {
                "profile": loader.OWNER_PROFILE,
                "completed": True,
                "primary_error": None,
                "secondary_errors": [],
                "sources_unchanged": True,
                "selected_config": config,
                "closure": {"invented": "original closure"},
            },
            "current_closure": {"invented": "another closure"},
        }
        lookup = {
            prerequisite["proof_refs"][key]["path"]: loader.canonical(value)
            for key, value in documents.items()
        }
        with mock.patch.object(
            loader, "read_ref", side_effect=lambda ref, *_args, **_kwargs: lookup[ref["path"]]
        ):
            with self.assertRaisesRegex(ValueError, "owner/closure belongs"):
                loader.validate_np_acceptance(prerequisite, selection, lambda: None)

    def test_producer_joins_actual_child_config_source_and_plan(self):
        chunk, parent, proof, selection, plan, expected_ref = fake_producer()
        with mock.patch.object(loader, "read_ref", return_value=loader.canonical(plan)) as reader:
            self.assertEqual(
                loader.validate_producer_binding(chunk, parent, proof, selection, lambda: None),
                expected_ref,
            )
            self.assertEqual(reader.call_args.args[0], expected_ref)

    def test_other_successful_run_pid_start_or_source_cannot_qualify_chunk(self):
        for mutation in ("run", "pid", "start", "group", "exit", "source", "plan_pin", "ancestor"):
            chunk, parent, proof, selection, plan, _ref = fake_producer()
            if mutation == "run":
                parent = loader.OWNER / "actual-98/train-chunk"
            elif mutation == "pid":
                chunk["owner"]["process"]["pid"] = 22
            elif mutation == "start":
                chunk["owner"]["process"]["starttime_ticks"] = 501
            elif mutation == "group":
                chunk["owner"]["process"]["cgroup"] = "0::/different"
            elif mutation == "exit":
                proof["owner_receipt"]["state"]["actor_exit"]["command_results"][0][
                    "returncode"
                ] = 1
            elif mutation == "source":
                proof["sealed_config"]["sources"][0]["sha256"] = "0" * 64
            elif mutation == "plan_pin":
                proof["sealed_config"]["sources"][-1]["sha256"] = "0" * 64
            else:
                plan["ancestor_id"] = loader.ANCESTORS[1]
            with (
                self.subTest(mutation=mutation),
                mock.patch.object(loader, "read_ref", return_value=loader.canonical(plan)),
                self.assertRaises(ValueError),
            ):
                loader.validate_producer_binding(chunk, parent, proof, selection, lambda: None)


class GPUStatProjectionTests(unittest.TestCase):
    def test_gpu_blas_is_separate_and_nvpl_remains_cpu(self):
        names = (
            "libcublas.so.13",
            "libcublasLt.so.13",
            "libnvblas.so.13",
            "libnvpl_blas_core.so.0",
            "libscipy_openblas64_example.so",
        )
        raw = b"\n".join(("1000-2000 r--p 0 00:00 1 /invented/" + name).encode() for name in names)
        with mock.patch.object(Path, "resolve", lambda path, **_kwargs: path):
            cpu, gpu = fit_head.mapped_math(raw)
        self.assertEqual({Path(path).name for path in cpu}, set(names[3:]))
        self.assertEqual({Path(path).name for path in gpu}, set(names[:3]))

    def test_large_gpu_candidate_uses_only_stat_not_cpu_hash_or_open(self):
        row = {
            "path": "/invented/libcublasLt.so.13",
            "size_bytes": 629945952,
            "stat": {"st_dev": 1, "st_ino": 2, "st_size": 629945952, "st_mtime_ns": 3},
            "hashed": False,
            "included_in_CPU_projection": False,
            "execution_or_symbol_binding_proven": False,
        }
        info = types.SimpleNamespace(st_mode=0o100600, **row["stat"])
        catalog = {"critical_files_and_mapped_math_candidates": [], fit_head.GPU_CATALOG_KEY: [row]}
        with (
            mock.patch.object(Path, "resolve", lambda path, **_kwargs: path),
            mock.patch.object(Path, "stat", return_value=info),
            mock.patch.object(
                Path, "open", side_effect=AssertionError("GPU payload must never open")
            ),
        ):
            self.assertEqual(fit_head.gpu_runtime_files(catalog, lambda: None), {row["path"]})
            changed = copy.deepcopy(catalog)
            changed[fit_head.GPU_CATALOG_KEY][0]["stat"]["st_mtime_ns"] = 4
            with self.assertRaisesRegex(ValueError, "stat changed"):
                fit_head.gpu_runtime_files(changed, lambda: None)

    def test_gpu_metadata_still_counts_toward_original_24file_limit(self):
        catalog = {
            "critical_files_and_mapped_math_candidates": [{}] * 24,
            fit_head.GPU_CATALOG_KEY: [{}],
        }
        with mock.patch.object(Path, "stat", side_effect=AssertionError("must refuse before stat")):
            with self.assertRaisesRegex(ValueError, "combined CPU/GPU"):
                fit_head.gpu_runtime_files(catalog, lambda: None)


class ChildKernelResourceTests(unittest.TestCase):
    def test_child_resource_checks_need_no_user_bus(self):
        group = Path("/invented/cgroup/owned")
        selected = types.SimpleNamespace(cgroup="/owned")
        boundary = types.SimpleNamespace(
            CGROUP_ROOT=Path("/invented/cgroup"),
            command=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no user bus allowed")),
        )
        controller = types.SimpleNamespace(
            main_affinity=lambda *a: {"checked": "original"},
            memory_events=lambda *a: {"counters": {"oom": 0, "oom_kill": 0}},
        )
        files = {
            group / "cpu.max": "500000 100000",
            group / "memory.peak": "12345",
            Path(
                "/proc/123/limits"
            ): "Max core file size         0                    0                    bytes\nMax file size              268435456            268435456            bytes\nMax open files            512                  512                  files\n",
        }
        with (
            mock.patch.object(Path, "read_text", lambda path, *a, **k: files[path]),
            mock.patch.object(Path, "exists", return_value=False),
        ):
            result = fit_head.child_properties(controller, boundary, selected, 123)
            self.assertEqual(result["kernel_cpu_max"], ["500000", "100000"])
            self.assertEqual(result["kernel_memory_peak_bytes"], 12345)
        for mutation in ("quota", "process_limit"):
            wrong = dict(files)
            if mutation == "quota":
                wrong[group / "cpu.max"] = "400000 100000"
            else:
                wrong[Path("/proc/123/limits")] = files[Path("/proc/123/limits")].replace(
                    "512", "1024"
                )
            with (
                self.subTest(mutation=mutation),
                mock.patch.object(Path, "read_text", lambda path, *a, **k: wrong[path]),
                self.assertRaises(ValueError),
            ):
                fit_head.child_properties(controller, boundary, selected, 123)


if __name__ == "__main__":
    unittest.main()
