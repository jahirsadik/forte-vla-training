"""Synthetic H5 audit tests only: no simulated task-success or replay claim."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.contracts import robot_definition_digest
from utils.coverage_plan import load_plan, select_collection_cases
from utils.coverage_recordings import audit_recordings, expected_instruction
from utils.research_protocol import load_manifest, manifest_digest
from scripts.audit_sorting_collection import main as audit_main

SOURCE = "a" * 64
CONTRACT = dict(id="forte-joint-reference-v2", shape=[8], arm_joint_count=7, policy_hz=20,
                cameras=["front", "wrist"], image_size_hw=[480, 640], privileged_fields_are_policy_inputs=False,
                robot_definition_sha256=robot_definition_digest(), environment_protocol="forte-manipulation-v3",
                representation="normalized_joint_reference_velocity_and_absolute_jaw_width",
                dtype="float32", normalized_range=[-1, 1], joint_reference_speed_rad_s=.25)


def fixture(path, case, manifest, *, success=True):
    """Minimal compressed reader fixture; its claimed scoring is synthetic."""
    n = 2
    metadata = dict(case, manifest_sha256=manifest_digest(manifest), policy="expert", custom_horizon=False,
                    execution_horizon_steps=1, constructor=dict(asset_set=case["asset_set"], observation_mode="rgb",
                    image_size=[480, 640], cameras=["front", "wrist"]))
    metrics = dict(success=success, within_goal=success, supported=success, released=success,
                   stable=True, placement_valid=success, failure_reason=None, timeout=not success,
                   dropped_objects=[], linear_speed_m_s=0., angular_speed_rad_s=0.,
                   distractor_translation_m=0., distractor_rotation_rad=0., reward_sparse=float(success))
    summary = dict(task_id=case["task_id"], protocol_version="forte-manipulation-v3", model_fingerprint="fixture-model",
                   is_success=success, metrics=metrics, episode=dict(steps=n, done=True, ever_grasped=True,
                   ever_lifted=True, wrong_grasp_steps=0, dwell=.5 if success else 0.))
    action = np.zeros((n, 8), np.float32)
    with h5py.File(path, "w") as f:
        f.attrs.update(schema="forte-demonstrations-v1", metadata=json.dumps(metadata), action_contract=json.dumps(CONTRACT),
                       implementation=json.dumps(dict(source_sha256=SOURCE)), complete=True, success=success,
                       environment_done=True, collector_error="", events="[]", instruction=expected_instruction(case),
                       model_fingerprint="fixture-model", transition_count=n, summary=json.dumps(summary))
        f.create_dataset("observations/proprio", data=np.zeros((n + 1, 23), np.float32))
        for camera in CONTRACT["cameras"]:
            f.create_dataset(f"observations/rgb/{camera}", shape=(n + 1, 480, 640, 3), dtype=np.uint8,
                             fillvalue=0, chunks=(1, 480, 640, 3), compression="gzip", compression_opts=1)
        f.create_dataset("observations/time_s", data=np.arange(n + 1) * .05)
        f.create_dataset("privileged/integration_state", data=np.zeros((n + 1, 3)))
        f.create_dataset("privileged/metrics_json", data=np.array([json.dumps(metrics)] * n, dtype=object), dtype=h5py.string_dtype())
        f.create_dataset("transitions/action", data=action)
        f.create_dataset("transitions/proposed_action", data=action)
        f.create_dataset("transitions/source", data=np.array(["expert"] * n, dtype=object), dtype=h5py.string_dtype())
        f.create_dataset("transitions/intervention", data=np.zeros(n, bool))
        f.create_dataset("transitions/expert_action_valid", data=np.zeros(n, bool))
        f.create_dataset("transitions/expert_action", data=np.full_like(action, np.nan))
        f.create_dataset("transitions/reward", data=[0., float(success)])
        f.create_dataset("transitions/terminated", data=[False, success])
        f.create_dataset("transitions/truncated", data=[False, not success])


def mutate_json(path, attribute, change):
    with h5py.File(path, "r+") as f:
        value = json.loads(f.attrs[attribute])
        change(value)
        f.attrs[attribute] = json.dumps(value)


class CoverageRecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest()
        cls.plan = load_plan(manifest=cls.manifest)
        cls.cases = select_collection_cases(cls.plan, cls.manifest, "nominal_seed0")
        cls.baseline_directory = tempfile.TemporaryDirectory()
        cls.baseline = Path(cls.baseline_directory.name)
        for case in cls.cases:
            fixture(cls.baseline / f"{case['case_id']}.h5", case, cls.manifest)

    @classmethod
    def tearDownClass(cls):
        cls.baseline_directory.cleanup()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.paths = []
        for case in self.cases:
            target = self.root / f"{case['case_id']}.h5"
            shutil.copyfile(self.baseline / target.name, target)
            self.paths.append(target)

    def tearDown(self):
        self.directory.cleanup()

    def audit(self, paths=None, **kwargs):
        return audit_recordings(self.paths if paths is None else paths, self.plan, self.manifest,
                                source_sha256=SOURCE, **kwargs)

    def assertBlocked(self, report):
        self.assertFalse(report["ready_for_balanced_nominal_fit"])
        self.assertEqual(report["admitted_paths"], [])

    def test_complete_declared_balanced_collection_is_structurally_ready(self):
        result = self.audit()
        self.assertTrue(result["ready_for_balanced_nominal_fit"])
        self.assertEqual(result["expected_episodes"], 24)
        self.assertEqual(result["eligible_episodes"], 24)
        self.assertEqual(result["complete_destination_pairs"], 12)
        self.assertEqual(len(result["admitted_paths"]), 24)
        self.assertEqual(result["missing_case_ids"], [])
        self.assertEqual(result["errors"], [])
        self.assertEqual(len(result["counterfactual_scenes"]), 2)
        self.assertIn("Not performed", result["physics_replay"])

    def test_failed_or_missing_member_blocks_full_collection_without_replacement(self):
        fixture(self.paths[0], self.cases[0], self.manifest, success=False)
        failed = self.audit()
        self.assertBlocked(failed)
        self.assertEqual(failed["eligible_episodes"], 23)
        self.assertEqual(failed["missing_case_ids"], [])
        self.assertEqual(failed["complete_destination_pairs"], 11)
        self.assertEqual(failed["errors"], [])
        missing = self.audit(self.paths[1:])
        self.assertBlocked(missing)
        self.assertEqual(missing["missing_case_ids"], [self.cases[0]["case_id"]])

    def test_duplicate_paths_and_duplicate_case_copies_are_rejected(self):
        for duplicate in (self.paths[0], self.root / "copied.h5"):
            if duplicate != self.paths[0]:
                shutil.copyfile(self.paths[0], duplicate)
            result = self.audit(self.paths + [duplicate])
            self.assertBlocked(result)
            self.assertTrue(any("Duplicate" in item["error"] for item in result["errors"]))

    def test_outside_subset_and_test_members_are_rejected(self):
        for split in ("train", "test"):
            case = next(case for case in self.manifest["cases"] if case["environment"] == "sorting"
                        and case["split"] == split and case["recovery"] == "none"
                        and (case["seed"] == 1 if split == "train" else True))
            extra = self.root / "outside.h5"
            fixture(extra, case, self.manifest)
            result = self.audit(self.paths + [extra])
            self.assertBlocked(result)
            self.assertTrue(any("outside" in item["error"] for item in result["errors"]))

    def test_frozen_source_and_contract_identity_must_match(self):
        mutations = (("implementation", lambda value: value.update(source_sha256="b" * 64)),
                     ("action_contract", lambda value: value.update(id="wrong")),
                     ("action_contract", lambda value: value.update(robot_definition_sha256="c" * 64)),
                     ("action_contract", lambda value: value.update(joint_reference_speed_rad_s=.5)))
        for attr, change in mutations:
            shutil.copyfile(self.baseline / self.paths[0].name, self.paths[0])
            mutate_json(self.paths[0], attr, change)
            with self.subTest(attribute=attr):
                result = self.audit()
                self.assertBlocked(result)
                self.assertTrue(result["errors"])
        for invalid in ("x" * 64, "a" * 63, None):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                audit_recordings(self.paths, self.plan, self.manifest, source_sha256=invalid)

    def test_same_scene_physics_proprio_and_rgb_mismatches_block_readiness(self):
        for field, index, delta in (("privileged/integration_state", (0, 0), .001),
                                    ("observations/proprio", (0, 0), .001),
                                    ("observations/rgb/front", (0, 0, 0, 0), 2)):
            shutil.copyfile(self.baseline / self.paths[1].name, self.paths[1])
            with h5py.File(self.paths[1], "r+") as f:
                f[field][index] = delta
            with self.subTest(field=field):
                result = self.audit()
                self.assertBlocked(result)
                self.assertTrue(any("Counterfactual" in item["error"] for item in result["errors"]))
        shutil.copyfile(self.baseline / self.paths[1].name, self.paths[1])
        with h5py.File(self.paths[1], "r+") as f:
            f["observations/rgb/front"][0, 0, 0, 0] = 1
        self.assertTrue(self.audit()["ready_for_balanced_nominal_fit"])

    def test_distinct_wrong_language_and_hidden_override_are_rejected(self):
        with h5py.File(self.paths[0], "r+") as f:
            f.attrs["instruction"] = "Pick up a wrong object: unique but invalid instruction."
        result = self.audit()
        self.assertBlocked(result)
        self.assertTrue(any("language" in item["error"] for item in result["errors"]))
        shutil.copyfile(self.baseline / self.paths[0].name, self.paths[0])
        mutate_json(self.paths[0], "metadata", lambda value: value.update(effective_instruction="wrong"))
        self.assertBlocked(self.audit())

    def test_nominal_protocol_rejects_hidden_action_horizon_and_fault_changes(self):
        for change in (lambda value: value.update(execution_horizon_steps=True),
                       lambda value: value.update(instruction_override="wrong"),
                       lambda value: value["constructor"].update(horizon_steps=5000)):
            shutil.copyfile(self.baseline / self.paths[0].name, self.paths[0])
            mutate_json(self.paths[0], "metadata", change)
            self.assertBlocked(self.audit())
        for dataset, index, value in (("transitions/proposed_action", (0, 0), .1),
                                       ("transitions/source", 0, "policy"),
                                       ("transitions/intervention", 0, True)):
            shutil.copyfile(self.baseline / self.paths[0].name, self.paths[0])
            with h5py.File(self.paths[0], "r+") as f:
                f[dataset][index] = value
            self.assertBlocked(self.audit())
        shutil.copyfile(self.baseline / self.paths[0].name, self.paths[0])
        with h5py.File(self.paths[0], "r+") as f:
            f.attrs["events"] = json.dumps([dict(kind="missed_grasp", step=1)])
        self.assertBlocked(self.audit())

    def test_final_summary_terminal_and_physical_scores_must_agree(self):
        for change in (lambda summary: summary.update(is_success=False),
                       lambda summary: summary["episode"].update(steps=3),
                       lambda summary: summary["episode"].update(dwell=.1),
                       lambda summary: summary["metrics"].update(released=False)):
            shutil.copyfile(self.baseline / self.paths[0].name, self.paths[0])
            mutate_json(self.paths[0], "summary", change)
            self.assertBlocked(self.audit())
        shutil.copyfile(self.baseline / self.paths[0].name, self.paths[0])
        with h5py.File(self.paths[0], "r+") as f:
            f["transitions/terminated"][-1] = False
        self.assertBlocked(self.audit())

    def test_malformed_flags_nonfinite_state_and_missing_required_fields_fail_closed(self):
        for field, replacement in (("transitions/intervention", np.zeros(2, np.uint8)),
                                    ("transitions/source", np.array([["expert"], ["expert"]], dtype=object))):
            shutil.copyfile(self.baseline / self.paths[0].name, self.paths[0])
            with h5py.File(self.paths[0], "r+") as f:
                del f[field]
                f.create_dataset(field, data=replacement, **({"dtype": h5py.string_dtype()} if replacement.dtype == object else {}))
            self.assertBlocked(self.audit())
        for field in ("observations/proprio", "privileged/integration_state"):
            shutil.copyfile(self.baseline / self.paths[0].name, self.paths[0])
            with h5py.File(self.paths[0], "r+") as f:
                f[field][-1, 0] = np.inf
            self.assertBlocked(self.audit())
        shutil.copyfile(self.baseline / self.paths[0].name, self.paths[0])
        with h5py.File(self.paths[0], "r+") as f:
            del f.attrs["summary"]
        self.assertBlocked(self.audit())

    def test_recovery_subset_requires_a_separate_label_aware_audit(self):
        with self.assertRaisesRegex(ValueError, "does not admit recovery"):
            self.audit(subset="recovery_seed0")

    def test_cli_publishes_audit_and_preserves_existing_output(self):
        output = self.root / "audit.json"
        arguments = ["audit_sorting_collection", "--episodes", str(self.root), "--source-sha256", SOURCE,
                     "--output", str(output)]
        with patch.object(sys, "argv", arguments), patch("builtins.print"):
            self.assertEqual(audit_main(), 0)
        self.assertTrue(json.loads(output.read_text())["ready_for_balanced_nominal_fit"])
        original = output.read_bytes()
        with patch.object(sys, "argv", arguments), patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            audit_main()
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(output.read_bytes(), original)

    def test_cli_empty_collection_returns_failure_with_all_cases_missing(self):
        output = self.root / "empty_audit.json"
        empty = self.root / "empty"
        empty.mkdir()
        arguments = ["audit_sorting_collection", "--episodes", str(empty), "--source-sha256", SOURCE,
                     "--output", str(output)]
        with patch.object(sys, "argv", arguments), patch("builtins.print"):
            self.assertEqual(audit_main(), 1)
        report = json.loads(output.read_text())
        self.assertBlocked(report)
        self.assertEqual(len(report["missing_case_ids"]), 24)


if __name__ == "__main__":
    unittest.main()
