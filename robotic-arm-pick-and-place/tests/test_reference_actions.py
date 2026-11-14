"""Causal absolute-reference targets and bounded execution through the old API."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.reference_actions import ReferenceActionCodec, normalize_joint_references, reference_targets
from utils.actions import decode_action
from utils.demonstrations import training_chunks
from utils.research_protocol import load_manifest, manifest_digest

HAS_TORCH = importlib.util.find_spec("torch") is not None


CONTRACT = dict(id="forte-joint-reference-v2", shape=[8], arm_joint_count=7, policy_hz=20,
                cameras=["front", "wrist"], image_size_hw=[24, 32], privileged_fields_are_policy_inputs=False,
                joint_lower_limits_rad=[-.17, -3.14, -2.6, -7.9, -6.42, -5.0, -6.42],
                joint_upper_limits_rad=[.17, 0., 2.09, 4.64, 6.14, 7.55, 6.14],
                joint_reference_speed_rad_s=.25, gripper_closed_m=0., gripper_open_m=.074,
                gripper_speed_limit_m_s=.1)


def initial_observation():
    values = np.zeros(23, np.float32)
    values[:7] = [-.01, -1.57, .1, -1.57, .1, 1.57, -.1]
    values[14] = .074
    return values


def commands():
    values = np.random.default_rng(21).uniform(-.75, .75, (6, 8)).astype(np.float32)
    values[:, 7] = [1., .02702703, -.08108108, -1., -1., 1.]
    return values


def fixture(path, *, fault=False):
    """Small synthetic reader fixture; it makes no physical success claim."""
    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if c["split"] == "train" and c["environment"] == "sorting"
                and c["appearance"] == "nominal" and c["recovery"] == "none")
    actions = commands()
    with h5py.File(path, "w") as f:
        f.attrs.update(metadata=json.dumps(dict(case, manifest_sha256=manifest_digest(manifest), policy="expert")),
                       action_contract=json.dumps(CONTRACT), implementation=json.dumps(dict(source_sha256="fixture")),
                       complete=True, success=True, instruction="pick the red cube", events="[]", collector_error="")
        f.create_dataset("observations/proprio", data=np.stack([initial_observation()] * 7))
        for camera in CONTRACT["cameras"]:
            f.create_dataset(f"observations/rgb/{camera}", data=np.zeros((7, 24, 32, 3), np.uint8))
        f.create_dataset("observations/time_s", data=np.arange(7) * .05)
        f.create_dataset("privileged/integration_state", data=np.zeros((7, 3)))
        f.create_dataset("privileged/joint_reference_rad", data=np.full((7, 7), 99.))
        f.create_dataset("privileged/jaw_reference_m", data=np.full(7, 99.))
        f.create_dataset("transitions/action", data=actions)
        f.create_dataset("transitions/proposed_action", data=actions)
        f.create_dataset("transitions/expert_action", data=np.full((6, 8), np.nan, np.float32))
        f.create_dataset("transitions/expert_action_valid", data=np.zeros(6, bool))
        sources = ["expert"] * 6
        if fault:
            sources[2] = "perturbed"
        f.create_dataset("transitions/source", data=sources, dtype=h5py.string_dtype())
        f.create_dataset("transitions/intervention", data=np.zeros(6, bool))


class ReferenceActionTests(unittest.TestCase):
    def test_reconstructed_targets_decode_to_original_references_and_jaw_commands(self):
        initial, actions = initial_observation(), commands()
        targets = reference_targets(initial, actions, CONTRACT)
        codec = ReferenceActionCodec(CONTRACT)
        codec.initialize(initial)
        decoded = codec.decode_chunk(targets, initial)
        np.testing.assert_allclose(decoded[:, :7], actions[:, :7], atol=5e-5, rtol=0)
        np.testing.assert_array_equal(decoded[:, 7], actions[:, 7])
        intended = initial[:7].astype(float)
        jaw = float(initial[14])
        for original, issued in zip(actions, decoded):
            intended, jaw = decode_action(original, intended, jaw, CONTRACT)
            codec.observe_action(issued)
            np.testing.assert_allclose(codec.arm_reference, intended, atol=5e-7, rtol=0)
            self.assertAlmostEqual(codec.jaw_reference, jaw, places=9)

    def test_prediction_does_not_consume_commands_and_reset_discards_history(self):
        initial = initial_observation()
        codec = ReferenceActionCodec(CONTRACT)
        with self.assertRaisesRegex(ValueError, "Initialize"):
            codec.observe_action(commands()[0])
        codec.initialize(initial)
        before, jaw = codec.arm_reference.copy(), codec.jaw_reference
        targets = reference_targets(initial, commands(), CONTRACT)
        first = codec.decode_chunk(targets, initial)
        np.testing.assert_array_equal(codec.decode_chunk(targets, initial), first)
        np.testing.assert_array_equal(codec.arm_reference, before)
        self.assertEqual(codec.jaw_reference, jaw)
        codec.observe_action(first[0])
        expected, expected_jaw = decode_action(first[0], before, jaw, CONTRACT)
        np.testing.assert_array_equal(codec.arm_reference, expected)
        self.assertEqual(codec.jaw_reference, expected_jaw)
        # A measured pose does not silently overwrite causal command state.
        changed = initial.copy(); changed[:7] += .01
        codec.initialize(changed)
        np.testing.assert_array_equal(codec.arm_reference, expected)
        codec.reset()
        self.assertIsNone(codec.arm_reference)
        self.assertIsNone(codec.jaw_reference)
        np.testing.assert_array_equal(codec.decode_chunk(targets, initial), first)

    def test_unreachable_targets_are_rate_limited_with_original_jaw_slew(self):
        initial = initial_observation()
        codec = ReferenceActionCodec(CONTRACT)
        targets = np.ones((40, 8), np.float32); targets[:, 7] = -1
        commands_out = codec.decode_chunk(targets, initial)
        self.assertEqual(commands_out.dtype, np.float32)
        self.assertTrue(np.isfinite(commands_out).all())
        self.assertTrue((np.abs(commands_out) <= 1).all())
        for command in commands_out:
            before, jaw = codec.arm_reference.copy(), codec.jaw_reference
            codec.observe_action(command)
            self.assertTrue(np.all(np.abs(codec.arm_reference - before) <= .0125 + 1e-12))
            self.assertLessEqual(abs(codec.jaw_reference - jaw), .005 + 1e-12)
            self.assertTrue(np.all(codec.arm_reference >= codec.lower))
            self.assertTrue(np.all(codec.arm_reference <= codec.upper))
        self.assertAlmostEqual(codec.arm_reference[0], CONTRACT["joint_upper_limits_rad"][0])
        self.assertEqual(codec.jaw_reference, 0.)
        for malformed in [np.zeros(8), np.zeros((0, 8)), np.zeros((2, 7)), np.full((2, 8), np.nan), np.full((2, 8), 1.01)]:
            with self.subTest(shape=malformed.shape), self.assertRaises(ValueError):
                codec.decode_chunk(malformed, initial)

    def test_public_operating_limits_define_normalization(self):
        np.testing.assert_array_equal(normalize_joint_references(CONTRACT["joint_lower_limits_rad"], CONTRACT), -np.ones(7))
        np.testing.assert_array_equal(normalize_joint_references(CONTRACT["joint_upper_limits_rad"], CONTRACT), np.ones(7))
        midpoint = (np.asarray(CONTRACT["joint_lower_limits_rad"]) + CONTRACT["joint_upper_limits_rad"]) / 2
        np.testing.assert_allclose(normalize_joint_references(midpoint, CONTRACT), 0, atol=1e-7)
        with self.assertRaises(ValueError):
            normalize_joint_references(np.full(7, 100.), CONTRACT)
        with self.assertRaises(ValueError):
            ReferenceActionCodec(dict(CONTRACT, joint_reference_speed_rad_s=0))

    def test_training_chunk_indices_remain_raw_across_unlabeled_faults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.h5"; fixture(path, fault=True)
            samples = list(training_chunks([path], horizon=4))
            self.assertEqual([s["transition_index"] for s in samples], [0, 1, 3, 4, 5])
            self.assertEqual([int(s["action_mask"].sum()) for s in samples], [2, 1, 3, 2, 1])

    @unittest.skipUnless(HAS_TORCH, "Dataset tests require optional learning dependencies")
    def test_reference_dataset_has_causal_post_action_targets_and_masked_tail(self):
        from learning.data import CompactChunkDataset, inspect_episodes
        from learning.diffusion_policy import ModelConfig, Tokenizer
        config = ModelConfig(image_height=24, image_width=32, hidden_dim=32,
                             diffusion_steps=8, inference_steps=4, action_representation="joint_reference")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.h5"; fixture(path)
            records, _ = inspect_episodes([path], "train")
            dataset = CompactChunkDataset(records, config, Tokenizer.fit(["pick the red cube"]), split="train")
            expected = reference_targets(initial_observation(), commands(), CONTRACT)
            self.assertEqual(len(dataset), 6)
            np.testing.assert_array_equal(dataset.actions[0, :6], expected)
            np.testing.assert_array_equal(dataset.actions[3, :3], expected[3:])
            np.testing.assert_array_equal(dataset.actions[5, 0], expected[5])
            np.testing.assert_array_equal(dataset.actions[5, 1:], 0)
            np.testing.assert_array_equal(dataset.masks[5], [True, False, False, False, False, False, False, False])

    @unittest.skipUnless(HAS_TORCH, "Dataset tests require optional learning dependencies")
    def test_reference_targets_ignore_privileged_and_future_observation_values(self):
        from learning.data import load_reference_targets
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.h5"; fixture(path)
            expected = load_reference_targets(path, CONTRACT)
            with h5py.File(path, "r+") as f:
                for dataset in f["privileged"].values():
                    dataset[:] = np.nan
                f["observations/proprio"][1:] = 1234.
            np.testing.assert_array_equal(load_reference_targets(path, CONTRACT), expected)

    @unittest.skipUnless(HAS_TORCH, "Dataset tests require optional learning dependencies")
    def test_reference_targets_reject_non_nominal_or_counterfactual_commands(self):
        from learning.data import load_reference_targets
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.h5"
            for mode in ("source", "proposal", "intervention", "recovery", "appearance", "events", "correction", "missing_proposal"):
                with self.subTest(mode=mode):
                    fixture(path)
                    with h5py.File(path, "r+") as f:
                        if mode == "source": f["transitions/source"][2] = "policy"
                        elif mode == "proposal": f["transitions/proposed_action"][2, 0] = 1.
                        elif mode == "intervention": f["transitions/intervention"][2] = True
                        elif mode in ("recovery", "appearance"):
                            meta = json.loads(f.attrs["metadata"]); meta[mode] = "perturbed"; f.attrs["metadata"] = json.dumps(meta)
                        elif mode == "events": f.attrs["events"] = '[{"fault": "injected"}]'
                        elif mode == "correction":
                            f["transitions/expert_action_valid"][2] = True
                            f["transitions/expert_action"][2] = 0.
                        else: del f["transitions/proposed_action"]
                    with self.assertRaises(ValueError):
                        load_reference_targets(path, CONTRACT)


if __name__ == "__main__":
    unittest.main()
