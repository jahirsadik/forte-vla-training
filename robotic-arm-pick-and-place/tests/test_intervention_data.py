"""Strict supervision boundaries for actual learner-prefix/expert-suffix trials."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import h5py
import numpy as np
try:
    import torch
except ImportError:
    raise unittest.SkipTest("Install optional requirements-learning.txt")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.data import (CompactChunkDataset, check_data_separation, inspect_episodes,
                           intervention_rollout_id, load_reference_targets, INTERVENTION_VERSION_BY_EXPERT)
from learning.diffusion_policy import ModelConfig, Tokenizer
from utils.demonstrations import training_chunks
from utils.research_protocol import load_manifest, manifest_digest


CONTRACT = dict(id="forte-joint-reference-v2", shape=[8], arm_joint_count=7, policy_hz=20,
                cameras=["front", "wrist"], image_size_hw=[24, 32], privileged_fields_are_policy_inputs=False)


def fixture(path, *, kind="expert_intervention", seed=0, takeover=3, success=True,
            split="train", source="fixture-current", offset=None, steps=8,
            version="forte-release-intervention-v1", expert="supported_goal_release_v1",
            parent_checkpoint_sha256="a" * 64):
    """Synthetic reader fixture only; no physical demonstration claim."""
    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if c["split"] == split and c["environment"] == "sorting"
                and c["appearance"] == "nominal" and c["recovery"] == "none")
    descriptor = dict(version=version, parent_checkpoint_sha256=parent_checkpoint_sha256,
                      policy_seed=seed, minimum_policy_steps=2, confirmation_steps=1,
                      actual_takeover_step=takeover, expert=expert)
    meta = dict(case, manifest_sha256=manifest_digest(manifest), policy=kind, custom_horizon=False)
    if kind == "expert_intervention":
        meta.update(intervention=descriptor, rollout_id=intervention_rollout_id(case["case_id"], descriptor),
                    collection_plan_sha256="b" * 64)
    action = np.zeros((steps, 8), np.float32)
    action[:, 0] = (.25 if kind == "expert" else .01 * (seed + 1)) if offset is None else offset
    action[:, -1] = np.where(np.arange(steps) < 5, -1., 1.)
    policy_prefix = 0 if kind == "expert" else (len(action) if takeover is None else takeover)
    is_intervention = np.arange(steps) >= policy_prefix if kind == "expert_intervention" else np.zeros(steps, bool)
    sources = np.where(np.arange(steps) < policy_prefix, "policy", "expert")
    valid = is_intervention.copy()
    labels = np.where(valid[:, None], action, np.nan).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.attrs.update(metadata=json.dumps(meta), action_contract=json.dumps(CONTRACT),
                       implementation=json.dumps(dict(source_sha256=source)), complete=True, success=success,
                       environment_done=True, collector_error="", events="[]", instruction="move the red cube")
        f.create_dataset("observations/proprio", data=np.repeat(np.arange(steps + 1, dtype=np.float32)[:, None], 23, 1))
        for camera in CONTRACT["cameras"]:
            f.create_dataset(f"observations/rgb/{camera}", data=np.zeros((steps + 1, 24, 32, 3), np.uint8))
        f.create_dataset("observations/time_s", data=np.arange(steps + 1) * .05)
        f.create_dataset("privileged/integration_state", data=np.zeros((steps + 1, 3)))
        f.create_dataset("transitions/action", data=action)
        f.create_dataset("transitions/proposed_action", data=action)
        f.create_dataset("transitions/source", data=sources.astype(object), dtype=h5py.string_dtype())
        f.create_dataset("transitions/intervention", data=is_intervention)
        f.create_dataset("transitions/expert_action_valid", data=valid)
        f.create_dataset("transitions/expert_action", data=labels)
    return meta


class InterventionDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.config = ModelConfig(image_height=24, image_width=32, hidden_dim=32,
                                  diffusion_steps=8, inference_steps=4)
        self.tokenizer = Tokenizer.fit(["move the red cube"])

    def tearDown(self):
        self.directory.cleanup()


    def test_carry_protocol_binds_expert_and_preserves_known_release_rollout_id(self):
        descriptor = dict(version="forte-release-intervention-v1", expert="supported_goal_release_v1",
                          parent_checkpoint_sha256="19e29d68d71802800fbbf484a555a80ecc22ec452e39115c87583cb0299a8715",
                          policy_seed=0, minimum_policy_steps=650, confirmation_steps=3, actual_takeover_step=None)
        case_id = "0fc510ebb88105fe8404"
        self.assertEqual(intervention_rollout_id(case_id, descriptor), "762330dbcdbd98ffa89d")
        self.assertEqual(intervention_rollout_id(case_id, dict(descriptor, actual_takeover_step=652)), "762330dbcdbd98ffa89d")
        carry = dict(descriptor, version="forte-carry-intervention-v1", expert="grasped_carry_recovery_v1")
        self.assertNotEqual(intervention_rollout_id(case_id, carry), "762330dbcdbd98ffa89d")
        self.assertEqual(INTERVENTION_VERSION_BY_EXPERT[carry["expert"]], carry["version"])
        for invalid in (dict(descriptor, expert=carry["expert"]), dict(carry, expert=descriptor["expert"]),
                        dict(carry, version="unknown"), dict(carry, expert="unknown"),
                        dict(carry, expert=[]), dict(carry, version=[])):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "Unsupported"):
                intervention_rollout_id(case_id, invalid)

    def test_mixed_release_carry_supervision_and_parent_preservation_stay_separate(self):
        from learning.preservation_data import PreservationDataset
        nominal, release, carry = [self.root / name for name in ("nominal.h5", "release.h5", "carry.h5")]
        fixture(nominal, kind="expert", steps=16)
        fixture(release, steps=16, takeover=8, offset=.1)
        fixture(carry, steps=16, takeover=8, offset=.2,
                version="forte-carry-intervention-v1", expert="grasped_carry_recovery_v1")
        records, skipped = inspect_episodes([nominal, release, carry], "train", allow_interventions=True)
        self.assertEqual(skipped, [])
        self.assertEqual([r["supervised_transitions"] for r in records], [16, 8, 8])
        samples = list(training_chunks([carry]))
        self.assertEqual([sample["transition_index"] for sample in samples], list(range(8, 16)))
        self.assertTrue(all(sample["source"] == "expert" and sample["is_intervention"] for sample in samples))
        with h5py.File(carry, "r") as episode:
            np.testing.assert_array_equal(samples[0]["action"], episode["transitions/action"][8:16])
            self.assertFalse(episode["transitions/expert_action_valid"][:8].any())
        dataset = CompactChunkDataset(records, self.config, self.tokenizer, split="train")
        self.assertEqual(len(dataset), 32)
        self.assertEqual(int(dataset.is_intervention.sum()), 16)
        normalization = dict(split="train", mean=[0.] * 23, std=[1.] * 23, count=16)
        preserved = PreservationDataset(records[1:], self.config, self.tokenizer, normalization,
                                        prefix_steps=8, anchor_stride=8, parent_checkpoint_sha256="a" * 64)
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved.metadata["duplicate_anchor_count"], 1)
        self.assertEqual(preserved.metadata["anchors"][0]["transition_index"], 0)
        self.assertEqual(len(preserved[0]), 3)
        self.assertFalse(preserved.metadata["action_labels"])
        self.assertEqual(preserved.normalization, normalization)
        with self.assertRaisesRegex(ValueError, "parent checkpoint mismatch"):
            PreservationDataset(records[1:], self.config, self.tokenizer, normalization,
                                prefix_steps=8, parent_checkpoint_sha256="c" * 64)

    def test_cross_protocol_relabeling_cannot_evade_physical_duplicate_check(self):
        release, carry = self.root / "release.h5", self.root / "carry.h5"
        release_meta = fixture(release)
        carry_meta = fixture(carry, version="forte-carry-intervention-v1", expert="grasped_carry_recovery_v1")
        self.assertNotEqual(release_meta["rollout_id"], carry_meta["rollout_id"])
        with self.assertRaisesRegex(ValueError, "physical trajectory"):
            inspect_episodes([release, carry], "train", allow_interventions=True)

    def test_carry_retains_label_boundary_outcome_source_split_and_contract_checks(self):
        nominal, carry = self.root / "nominal.h5", self.root / "carry.h5"
        fixture(nominal, kind="expert")
        protocol = dict(version="forte-carry-intervention-v1", expert="grasped_carry_recovery_v1")
        for mutation in ("prefix_label", "suffix_label", "early_takeover", "late_takeover", "source", "contract", "fault"):
            with self.subTest(mutation=mutation):
                fixture(carry, **protocol)
                with h5py.File(carry, "r+") as f:
                    meta = json.loads(f.attrs["metadata"])
                    if mutation == "prefix_label": f["transitions/expert_action_valid"][0] = True
                    elif mutation == "suffix_label": f["transitions/expert_action"][4, 0] = .9
                    elif mutation == "early_takeover": meta["intervention"]["actual_takeover_step"] = 1
                    elif mutation == "late_takeover": meta["intervention"]["actual_takeover_step"] = 8
                    elif mutation == "source": f.attrs["implementation"] = json.dumps(dict(source_sha256="different"))
                    elif mutation == "contract":
                        contract = dict(CONTRACT, policy_hz=10)
                        f.attrs["action_contract"] = json.dumps(contract)
                    elif mutation == "fault": f.attrs["events"] = json.dumps([dict(kind="missed_grasp", step=2)])
                    f.attrs["metadata"] = json.dumps(meta)
                with self.assertRaises((ValueError, AssertionError)):
                    inspect_episodes([nominal, carry], "train", allow_interventions=True)
        for success, takeover, reason in ((False, 3, "incomplete_or_unsuccessful"),
                                          (True, None, "no_expert_suffix")):
            fixture(carry, success=success, takeover=takeover, **protocol)
            records, skipped = inspect_episodes([nominal, carry], "train", allow_interventions=True)
            self.assertEqual(len(records), 1)
            self.assertEqual(skipped[0]["reason"], reason)
        for split in ("validation", "test"):
            fixture(carry, split=split, **protocol)
            with self.assertRaisesRegex(ValueError, f"{split} episode"):
                inspect_episodes([nominal, carry], "train", allow_interventions=True)
        fixture(carry, **protocol)
        with self.assertRaisesRegex(ValueError, "nominal expert"):
            load_reference_targets(carry, CONTRACT)

    def test_explicit_optin_and_verified_multiple_rollouts_share_frozen_case(self):
        expert, first, second = [self.root / name for name in ("expert.h5", "first.h5", "second.h5")]
        fixture(expert, kind="expert")
        a = fixture(first, seed=0)
        b = fixture(second, seed=1, takeover=4)
        self.assertEqual(a["case_id"], b["case_id"])
        self.assertNotEqual(a["rollout_id"], b["rollout_id"])
        with self.assertRaises(ValueError):
            inspect_episodes([first], "train")
        with self.assertRaises(ValueError):
            inspect_episodes([expert, first], "train")
        records, skipped = inspect_episodes([expert, first, second], "train", allow_interventions=True)
        self.assertEqual(skipped, [])
        self.assertEqual([r["supervised_transitions"] for r in records], [8, 5, 4])
        self.assertEqual([r["is_intervention"] for r in records], [False, True, True])

    def test_rollout_id_ignores_outcome_but_binds_immutable_plan_fields(self):
        path = self.root / "episode.h5"
        meta = fixture(path)
        d = meta["intervention"]
        expected = intervention_rollout_id(meta["case_id"], d)
        self.assertEqual(expected, intervention_rollout_id(meta["case_id"], dict(d, actual_takeover_step=None)))
        for changed in (dict(d, policy_seed=4), dict(d, minimum_policy_steps=4),
                        dict(d, confirmation_steps=2), dict(d, parent_checkpoint_sha256="c" * 64)):
            self.assertNotEqual(expected, intervention_rollout_id(meta["case_id"], changed))
        for changed in (dict(d, policy_seed=True), dict(d, minimum_policy_steps=0),
                        dict(d, parent_checkpoint_sha256="missing"), dict(d, unknown=True)):
            with self.assertRaises(ValueError):
                intervention_rollout_id(meta["case_id"], changed)
        with h5py.File(path, "r+") as f:
            meta["rollout_id"] = "0" * 20
            f.attrs["metadata"] = json.dumps(meta)
        with self.assertRaisesRegex(ValueError, "identity"):
            inspect_episodes([path], "train", allow_interventions=True)

    def test_duplicates_rejected_even_after_metadata_is_reidentified(self):
        first, second = self.root / "first.h5", self.root / "second.h5"
        fixture(first)
        shutil.copy2(first, second)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            inspect_episodes([first, second], "train", allow_interventions=True)
        with h5py.File(second, "r+") as f:
            meta = json.loads(f.attrs["metadata"])
            meta["intervention"]["policy_seed"] = 99
            meta["rollout_id"] = intervention_rollout_id(meta["case_id"], meta["intervention"])
            f.attrs["metadata"] = json.dumps(meta)
            # Pixel rounding is not a new physical trajectory.
            f["observations/rgb/front"][0, 0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "physical trajectory"):
            inspect_episodes([first, second], "train", allow_interventions=True)

    def test_no_policy_prefix_or_future_observations_enter_supervision(self):
        path = self.root / "episode.h5"; fixture(path)
        records, _ = inspect_episodes([path], "train", allow_interventions=True)
        samples = list(training_chunks([path]))
        self.assertEqual([s["transition_index"] for s in samples], [3, 4, 5, 6, 7])
        self.assertTrue(all(s["source"] == "expert" and s["is_intervention"] for s in samples))
        dataset = CompactChunkDataset(records, self.config, self.tokenizer, split="train")
        self.assertEqual(len(dataset), 5)
        np.testing.assert_array_equal(dataset.is_intervention, True)
        np.testing.assert_array_equal(dataset.normalization()["mean"], np.full(23, 5.))
        first_obs, first_action = samples[0]["observation"]["proprio"].copy(), samples[0]["action"].copy()
        with h5py.File(path, "r+") as f:
            f["observations/proprio"][:3] = -999.
            f["observations/proprio"][4:] = 999.
            f["privileged/integration_state"][:] = np.nan
        first = next(training_chunks([path]))
        np.testing.assert_array_equal(first["observation"]["proprio"], first_obs)
        np.testing.assert_array_equal(first["action"], first_action)
        self.assertEqual(set(first["observation"]), {"rgb", "proprio", "instruction"})

    def test_chunk_targets_stop_at_source_and_intervention_boundaries(self):
        path = self.root / "episode.h5"; fixture(path, kind="expert")
        with h5py.File(path, "r+") as f:
            f["transitions/source"][:] = ["expert", "expert", "policy", "policy", "expert", "expert", "expert", "expert"]
            f["transitions/intervention"][:] = [False] * 5 + [True] * 3
            f["transitions/expert_action_valid"][:] = True
            f["transitions/expert_action"][:] = f["transitions/action"][:]
        samples = list(training_chunks([path], horizon=8))
        self.assertEqual([int(s["action_mask"].sum()) for s in samples], [2, 1, 2, 1, 1, 3, 2, 1])
        for sample in samples:
            np.testing.assert_array_equal(sample["action"][~sample["action_mask"]], 0.)

    def test_malformed_prefix_suffix_faults_and_metadata_are_rejected(self):
        path = self.root / "episode.h5"
        for kind in ("prefix_label", "prefix_flag", "suffix_missing_label", "different_label", "source_switch",
                     "override", "custom_horizon", "fault", "hidden_command", "wrong_takeover", "missing_plan", "not_done"):
            with self.subTest(kind=kind):
                fixture(path)
                with h5py.File(path, "r+") as f:
                    meta = json.loads(f.attrs["metadata"])
                    if kind == "prefix_label": f["transitions/expert_action_valid"][1] = True
                    elif kind == "prefix_flag": f["transitions/intervention"][1] = True
                    elif kind == "suffix_missing_label": f["transitions/expert_action_valid"][5] = False
                    elif kind == "different_label": f["transitions/expert_action"][5, 0] = .5
                    elif kind == "source_switch": f["transitions/source"][5] = "policy"
                    elif kind == "override": meta["instruction_override"] = "move differently"
                    elif kind == "custom_horizon": meta["custom_horizon"] = True
                    elif kind == "fault": f.attrs["events"] = json.dumps([dict(kind="missed_grasp", step=2)])
                    elif kind == "hidden_command": f["transitions/proposed_action"][2, 0] = .2
                    elif kind == "wrong_takeover": meta["intervention"]["actual_takeover_step"] = 4
                    elif kind == "missing_plan": meta.pop("collection_plan_sha256")
                    elif kind == "not_done": f.attrs["environment_done"] = False
                    f.attrs["metadata"] = json.dumps(meta)
                with self.assertRaises(ValueError):
                    inspect_episodes([path], "train", allow_interventions=True)

    def test_failures_are_preserved_but_excluded_and_reference_mode_rejects_interventions(self):
        expert, failed, never, success = [self.root / name for name in ("expert.h5", "failed.h5", "never.h5", "success.h5")]
        fixture(expert, kind="expert")
        fixture(failed, seed=0, success=False)
        fixture(never, seed=1, takeover=None, success=False)
        fixture(success, seed=2)
        records, skipped = inspect_episodes([expert, failed, never, success], "train", allow_interventions=True)
        self.assertEqual(len(records), 2)
        self.assertEqual([r["reason"] for r in skipped], ["incomplete_or_unsuccessful"] * 2)
        self.assertTrue(failed.exists() and never.exists())
        with self.assertRaisesRegex(ValueError, "nominal expert"):
            load_reference_targets(success, CONTRACT)

    def test_autonomous_success_without_takeover_is_retained_but_not_expert_data(self):
        expert, autonomous = self.root / "expert.h5", self.root / "autonomous.h5"
        fixture(expert, kind="expert")
        fixture(autonomous, takeover=None, success=True)
        records, skipped = inspect_episodes([expert, autonomous], "train", allow_interventions=True)
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["is_intervention"])
        self.assertEqual(skipped[0]["reason"], "no_expert_suffix")
        self.assertEqual(list(training_chunks([autonomous])), [])
        self.assertTrue(autonomous.exists())

    def test_split_source_and_ordinary_expert_duplicate_protections_remain(self):
        expert, intervention, validation = self.root / "expert.h5", self.root / "intervention.h5", self.root / "validation.h5"
        fixture(expert, kind="expert")
        fixture(intervention, source="different-current")
        with self.assertRaisesRegex(ValueError, "mixed"):
            inspect_episodes([expert, intervention], "train", allow_interventions=True)
        fixture(validation, split="validation")
        with self.assertRaisesRegex(ValueError, "train split"):
            inspect_episodes([validation], "validation", allow_interventions=True)
        with self.assertRaises(ValueError):
            inspect_episodes([validation], "train", allow_interventions=True)
        fixture(intervention, kind="expert", offset=.4)
        with self.assertRaisesRegex(ValueError, "Duplicate frozen expert"):
            inspect_episodes([expert, intervention], "train", allow_interventions=True)
        records, _ = inspect_episodes([expert], "train")
        with self.assertRaisesRegex(ValueError, "overlap"):
            check_data_separation(records, records)


if __name__ == "__main__":
    unittest.main()
