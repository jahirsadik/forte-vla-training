"""Reference-target policy integration with the issued-action lifecycle."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
try:
    import torch
except ImportError:
    raise unittest.SkipTest("Install optional requirements-learning.txt")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.diffusion_policy import ActionDiffusion, DiffusionPolicy, ModelConfig, Tokenizer
from utils.policy import PolicySession


CONTRACT = dict(id="forte-joint-reference-v2", shape=[8], arm_joint_count=7,
                policy_hz=20., joint_reference_speed_rad_s=.25,
                joint_lower_limits_rad=[-2.] * 7, joint_upper_limits_rad=[2.] * 7,
                gripper_closed_m=0., gripper_open_m=.074, gripper_speed_limit_m_s=.1,
                cameras=["front", "wrist"], image_size_hw=[24, 32],
                privileged_fields_are_policy_inputs=False)


class ReferencePolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = ModelConfig(image_height=24, image_width=32, hidden_dim=32,
                                  diffusion_steps=8, inference_steps=4,
                                  action_representation="joint_reference", gripper_mode="binary")
        tokenizer = Tokenizer.fit(["move the cube"])
        self.model = ActionDiffusion(self.config, len(tokenizer.words))
        self.policy = DiffusionPolicy(self.model, tokenizer,
                                      dict(mean=[0.] * 23, std=[1.] * 23), CONTRACT, {}, "cpu")
        self.observation = dict(proprio=np.zeros(23, np.float32), instruction="move the cube",
                                rgb={c: np.zeros((24, 32, 3), np.uint8) for c in CONTRACT["cameras"]})

    def test_reference_mode_requires_feedback_and_reset_clears_it(self):
        targets = torch.full((1, 8, 8), .1)
        with patch.object(self.model, "sample", return_value=targets):
            first = self.policy.predict(self.observation)
            np.testing.assert_array_equal(first, np.ones((8, 8), np.float32))
            np.testing.assert_array_equal(self.policy.reference_codec.arm_reference, np.zeros(7))
            with self.assertRaisesRegex(RuntimeError, "feedback"):
                self.policy.predict(self.observation)
            self.policy.observe_action(first[0])
            np.testing.assert_allclose(self.policy.reference_codec.arm_reference, .0125)
            self.policy.predict(self.observation)
            self.policy.reset(seed=3)
            np.testing.assert_array_equal(first, self.policy.predict(self.observation))

    def test_cached_actions_update_only_issued_references(self):
        targets = torch.full((1, 8, 8), .1)
        space = gym.spaces.Box(-1., 1., shape=(8,), dtype=np.float32)
        session = PolicySession(self.policy, space, execute_actions=4)
        session.reset(seed=0)
        with patch.object(self.model, "sample", return_value=targets) as sample:
            for step in range(1, 5):
                session(self.observation)
                np.testing.assert_allclose(self.policy.reference_codec.arm_reference, step * .0125)
            self.assertEqual(sample.call_count, 1)
            session(self.observation)
            self.assertEqual(sample.call_count, 2)
            np.testing.assert_allclose(self.policy.reference_codec.arm_reference, .0625)
            session.reset(seed=0)
            session(self.observation)
            np.testing.assert_allclose(self.policy.reference_codec.arm_reference, .0125)

    def test_unknown_target_representation_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.config, action_representation="simulator_qpos")


if __name__ == "__main__":
    unittest.main()
