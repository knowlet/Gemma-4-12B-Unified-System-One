import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "first_edge_preparation", Path(__file__).with_name("prepare_edge_plan.py")
)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


class PreparedPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.value = P.selection()

    def seed(self):
        value = self.value
        seed = dict.fromkeys(P.SEED_FIELDS)
        seed.update(
            schema_id="decision3-q95-live-helper-selection-v1",
            binding={"boot_id": value["boot_id"], "synthetic_fixture_only": True},
            worker={"cgroup": value["helper_seed"]["cgroup"]},
            source_root=value["helper_seed"]["source_root"],
            service_directory=value["helper_seed"]["service_directory"],
            interpreter=value["browser"]["interpreter"],
            expected_memtotal=value["mem_total_bytes"],
        )
        return seed

    def test_static_selection_keeps_both_fresh_original_groups(self):
        self.assertEqual(len(self.value["expected_base_source_refs"]), 37)
        self.assertEqual(self.value["model"], "d3-edge")
        self.assertEqual(self.value["policy"]["fixed_denominator"], 10)
        self.assertEqual(self.value["policy"]["known4_envelopes"], 64)
        self.assertEqual(self.value["policy"]["public6_envelopes"], 96)
        self.assertEqual(len(set(self.value["units"].values())), 2)
        self.assertEqual(len(set(self.value["nonces"].values())), 2)

    def test_public_seed_is_copied_with_exact_per_group_cgroup(self):
        seed = self.seed()
        before = copy.deepcopy(seed)
        descriptors = P.derive_descriptors(self.value, seed)
        self.assertEqual(seed, before)
        self.assertEqual(descriptors["known4"]["binding"], descriptors["public6"]["binding"])
        for group, descriptor in descriptors.items():
            self.assertEqual(
                descriptor["co_resident"],
                {
                    "model": "d3-edge",
                    "resident_file_bytes": 1189722528,
                    "cgroup": self.value["cgroups"][group],
                },
            )

    def test_private_or_extra_fields_cannot_be_published_in_descriptor(self):
        seed = self.seed()
        seed["api_key"] = "synthetic-forbidden-field"
        with self.assertRaisesRegex(ValueError, "thirteen-field"):
            P.derive_descriptors(self.value, seed)

    def test_wrong_host_helper_cgroup_or_source_root_refuses(self):
        for field in ["boot", "memtotal", "cgroup", "source"]:
            with self.subTest(field=field):
                seed = self.seed()
                if field == "boot":
                    seed["binding"]["boot_id"] = "different-boot"
                elif field == "memtotal":
                    seed["expected_memtotal"] += 4096
                elif field == "cgroup":
                    seed["worker"]["cgroup"] = "/unselected/helper"
                else:
                    seed["source_root"] += "-current"
                with self.assertRaises(ValueError):
                    P.derive_descriptors(self.value, seed)

    def test_checked_bytes_refuse_stale_or_substituted_authority(self):
        with tempfile.TemporaryDirectory(dir=P.HERE) as directory:
            path = Path(directory) / "metadata.json"
            path.write_bytes(b"{}\n")
            digest = P.reference(path)["sha256"]
            path.write_bytes(b"[]\n")
            with self.assertRaisesRegex(ValueError, "SHA differs"):
                P.checked(path, digest)

    def test_full_preparation_selects_real_sources_but_does_not_admit_fake_worker(self):
        value = copy.deepcopy(self.value)
        source = Path(value["helper_seed"]["source_root"])
        service = Path(value["helper_seed"]["service_directory"])
        interpreter = value["browser"]["interpreter"]
        binding = {
            "profile": "field-syntax-preserving-base-helper-v1",
            "qualification_sha256": "d33bfcac49f90c47b626c3915baf3d5fcbfb2e32522bfaccbbfec59dce35d8db",
            "source_count": 95,
            "boot_id": value["boot_id"],
            "pid": 999999,
            "start_ticks": 1,
            "pgid": 999999,
            "sid": 999999,
            "endpoint": "http://127.0.0.1:9",
            "seconds": 20,
            "tokens": 128,
            "context": 8192,
        }
        seed = {
            "schema_id": "decision3-q95-live-helper-selection-v1",
            "binding": binding,
            "worker": {k: binding[k] for k in ["pid", "start_ticks", "pgid", "sid", "boot_id"]}
            | {
                "exe": str(Path(interpreter).resolve(strict=True)),
                "argv": [
                    interpreter,
                    str(source / "scripts/jev_qualify_public_field_syntax_bf16.py"),
                    "--internal-child",
                    str(service / "controller-config.json"),
                ],
                "cgroup": value["helper_seed"]["cgroup"],
            },
            "listener": {"pid": 999999, "address": "127.0.0.1", "port": 9, "inode": "1"},
            "service_directory": str(service),
            "private_config": str(service / "controller-config.json"),
            "interpreter": interpreter,
            "source_root": str(source),
            "qualification": P.reference(
                P.ROOT
                / "artifacts/jev-runtime/public-field-syntax-helper-20261018-01/qualify-01/qualification.json"
            ),
            "execution": P.reference(source / "support/execution95.json"),
            "runtime": P.reference(source / "support/runtime348.json"),
            "registry": P.reference(P.ROOT / "src/s1/evaluation/decision3.py"),
            "expected_memtotal": value["mem_total_bytes"],
        }
        # Fake public identities exist only in this temporary preparation test.
        # Neither real checker callback nor any controller/actor is called.
        with tempfile.TemporaryDirectory(dir=P.HERE) as directory:
            folder = Path(directory)
            seed_path = folder / "fixture-seed.json"
            P.exclusive_json(seed_path, seed)
            value["helper_seed"]["path"] = str(seed_path)
            value["descriptor_paths"] = {
                g: str(folder / (g + ".json")) for g in ["known4", "public6"]
            }
            value["plan_path"] = str(folder / "plan.json")
            with mock.patch.object(P, "HERE", folder):
                result = P.prepare(
                    value, seed_path, P.reference(seed_path)["sha256"], expires_after_seconds=90
                )
            self.assertEqual(result["source_count"], 39)
            self.assertTrue(result["selected_plan_validator_passed"])
            self.assertFalse(result["actual_live_admission_performed"])
            self.assertFalse(result["actual_model_browser_GPU_or_unit_started"])
            self.assertFalse(result["model_qualified"])
            self.assertFalse(Path(value["output"]).exists())


if __name__ == "__main__":
    unittest.main()
