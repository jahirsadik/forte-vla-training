"""Physical action units, closed-loop chunk execution, and episode/fault boundaries."""

import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environments.sorting import SortingEnv
from utils.actions import decode_action, encode_reference_step, LEARNING_INTERFACE
from utils.policy import PolicySession
from utils.demonstrations import EpisodeWriter, training_chunks
from utils.research_protocol import load_manifest, manifest_digest


class LearningInterfaceTests(unittest.TestCase):
    def test_saturated_reference_roundoff_stays_within_strict_limits(self):
        with SortingEnv() as env:
            env.reset(seed=0)
            start = env.robot.target_q.copy()
            # Last recorded reference before the red-cube policy rollout failed
            # at step 1690. Its valid next endpoint is shoulder_pitch == 0.
            start[1] = -0.0012901752779725943
            env.robot.set_arm_target(start)
            action = np.zeros(8, dtype=np.float32)
            action[1], action[-1] = 1., 1.
            target, _ = decode_action(action, start, env.robot.gripper_target,
                                      env.action_contract)
            raw_last = start + (target - start) * env.substeps / env.substeps
            self.assertGreater(raw_last[1], env.robot.upper[1])
            self.assertEqual(target[1], env.robot.upper[1])
            setter = env.robot.set_arm_target
            with patch.object(env.robot, 'set_arm_target', wraps=setter) as calls:
                env.step(action)
            self.assertEqual(calls.call_count, env.substeps)
            for call in calls.call_args_list:
                reference = call.args[0]
                self.assertTrue(np.all(reference >= env.robot.lower))
                self.assertTrue(np.all(reference <= env.robot.upper))
            self.assertEqual(env.robot.target_q[1], env.robot.upper[1])

    def test_repeated_float32_saturation_and_strict_direct_api(self):
        with SortingEnv() as env:
            for sign in (-1., 1.):
                with self.subTest(sign=sign):
                    env.reset(seed=0)
                    bound = env.robot.lower[0] if sign < 0 else env.robot.upper[0]
                    reference = env.robot.target_q.copy()
                    reference[0] = bound - sign * .001
                    env.robot.set_arm_target(reference)
                    action = np.zeros(8, dtype=np.float32)
                    action[0], action[-1] = sign, 1.
                    for _ in range(5):
                        env.step(action)
                        self.assertEqual(env.robot.target_q[0], bound)
                    invalid = env.robot.target_q.copy()
                    invalid[0] = bound + sign * 1e-6
                    with self.assertRaisesRegex(ValueError, 'Arm target exceeds joint limits'):
                        env.robot.set_arm_target(invalid)

    def test_interior_substep_reference_arithmetic_is_unchanged(self):
        with SortingEnv() as env:
            env.reset(seed=0)
            start = env.robot.target_q.copy()
            action = np.array([.2, -.3, .4, -.2, .1, -.1, .3, 1], dtype=np.float32)
            target, _ = decode_action(action, start, env.robot.gripper_target,
                                      env.action_contract)
            setter = env.robot.set_arm_target
            with patch.object(env.robot, 'set_arm_target', wraps=setter) as calls:
                env.step(action)
            for index, call in enumerate(calls.call_args_list):
                expected = start + (target - start) * (index + 1) / env.substeps
                np.testing.assert_array_equal(call.args[0], expected)

    def test_decoding_matches_live_reference_integration(self):
        with SortingEnv() as env:
            env.reset(seed=0)
            start, width = env.robot.target_q.copy(), env.robot.gripper_target
            action = np.array([.2, -.3, .4, -.2, .1, -.1, .3, -1], dtype=np.float32)
            target, jaw = decode_action(action, start, width, env.action_contract)
            np.testing.assert_allclose(target - start, action[:-1] * .0125, atol=1e-9)
            self.assertAlmostEqual(width - jaw, .005)
            env.step(action)
            np.testing.assert_allclose(env.robot.target_q, target, atol=1e-12)
            self.assertAlmostEqual(env.robot.gripper_target, jaw)
            reconstructed = encode_reference_step(target, start, 0., env.action_contract)
            np.testing.assert_allclose(reconstructed, action, atol=1e-6)
            with self.assertRaises(ValueError):
                encode_reference_step(start + .1, start, 0., env.action_contract)
            with self.assertRaises(ValueError):
                decode_action(np.zeros(7), start, width, env.action_contract)

    def test_replanning_discards_unexecuted_prediction_tail(self):
        class Policy:
            calls = 0
            def __call__(self, observation):
                self.calls += 1
                return np.array([[self.calls / 10, 1], [.9, 1], [.8, 1]], dtype=np.float32)
        policy = Policy()
        session = PolicySession(policy, gym.spaces.Box(-1, 1, (2,), dtype=np.float32), execute_actions=1)
        np.testing.assert_allclose(session({}), [.1, 1])
        np.testing.assert_allclose(session({}), [.2, 1])
        self.assertEqual(policy.calls, 2)
        self.assertEqual(LEARNING_INTERFACE["prediction_horizon_steps"], 8)
        with self.assertRaises(ValueError):
            PolicySession(policy, session.action_space, execute_actions=0)

    def test_training_chunks_stop_at_faults_and_episode_end(self):
        manifest = load_manifest()
        case = next(c for c in manifest["cases"] if c["environment"] == "sorting" and c["split"] == "train")
        with tempfile.TemporaryDirectory() as folder, SortingEnv(observation_mode="rgb", image_size=(60, 80), horizon_steps=5) as env:
            obs, info = env.reset(seed=case["seed"], options=dict(task_id=case["task_id"], layout=case["layout"]))
            path = Path(folder) / "correction_fixture.h5"
            meta = dict(case, manifest_sha256=manifest_digest(manifest),
                        constructor=dict(observation_mode="rgb", image_size=[60, 80], horizon_steps=5))
            writer = EpisodeWriter(path, env, obs, info, metadata=meta)
            action = np.r_[np.zeros(7), 1].astype(np.float32)
            for t in range(5):
                obs, reward, done, truncated, info = env.step(action)
                writer.append(action, obs, reward, done, truncated, info,
                              source="perturbed" if t == 2 else "policy",
                              expert_action=None if t == 2 else action)
            writer.finish(info)
            self.assertEqual(list(training_chunks([path])), [])
            samples = list(training_chunks([path], horizon=4, include_failures=True))
            self.assertEqual(len(samples), 4)
            self.assertEqual([s["action_mask"].sum() for s in samples], [2, 1, 2, 1])
            for sample in samples:
                self.assertEqual(sample["action"].shape, (4, 8))
                np.testing.assert_array_equal(sample["action"][~sample["action_mask"]], 0)
                self.assertEqual(set(sample["observation"]), {"rgb", "proprio", "instruction"})


if __name__ == "__main__":
    unittest.main()
