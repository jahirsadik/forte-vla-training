"""Conservative takeover gates, scratch isolation, and real release continuation."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controllers.benchmark_expert import BenchmarkExpert
from controllers.placement_correction import PlacementReleaseExpert
from environments.sorting import SortingEnv


class PlacementCorrectionTests(unittest.TestCase):
    def test_gate_requires_correct_supported_active_goal_and_grasp_history(self):
        with SortingEnv(observation_mode="proprio") as env:
            expert = PlacementReleaseExpert(env)
            self.assertFalse(expert.eligible())
            env.reset(seed=0, options=dict(task_id="red_cube_to_red", layout="shuffled", clutter_level=1))
            self.assertFalse(expert.eligible())
            with self.assertRaisesRegex(ValueError, "supported"):
                expert.reset()
            with self.assertRaisesRegex(RuntimeError, "Reset"):
                expert.predict(env._observe())
            # Synthetic gate states only; they are not physical rollouts/data.
            episode, metrics = deepcopy(env._episode), deepcopy(env._last_metrics)
            eligible_episode = dict(episode, ever_grasped=True, ever_lifted=True, done=False)
            eligible_metrics = dict(metrics, within_goal=True, supported=True, stable=True,
                                    released=False, failure_reason=None, dropped_objects=[])
            with patch.object(env, "_episode", eligible_episode), patch.object(env, "_last_metrics", eligible_metrics):
                self.assertTrue(expert.eligible())
                for field in ("ever_grasped", "ever_lifted"):
                    with patch.dict(env._episode, {field: False}):
                        self.assertFalse(expert.eligible())
                with patch.dict(env._episode, {"done": True}):
                    self.assertFalse(expert.eligible())
                for field in ("within_goal", "supported", "stable"):
                    with patch.dict(env._last_metrics, {field: False}):
                        self.assertFalse(expert.eligible())
                for change in ({"released": True}, {"failure_reason": "distractor_disturbed"},
                               {"dropped_objects": [env.target]}):
                    with patch.dict(env._last_metrics, change):
                        self.assertFalse(expert.eligible())
                with patch.object(expert, "_other_collisions", return_value={(1, 2): -.001}):
                    self.assertFalse(expert.eligible())
                contacts, _, _ = env._contacts()
                distractor = next(name for name in contacts if name != env.target)
                contacts[distractor]["l"] = contacts[distractor]["r"] = True
                with patch.object(env, "_contacts", return_value=(contacts, 0., 0.)):
                    self.assertFalse(expert.eligible())
                with patch.object(env, "scene_config", {}):
                    self.assertFalse(expert.eligible())

    def test_prediction_refuses_domain_loss_without_changing_physics_or_history(self):
        with SortingEnv(observation_mode="proprio") as env:
            observation, _ = env.reset(seed=0)
            expert = PlacementReleaseExpert(env)
            episode = dict(env._episode, ever_grasped=True, ever_lifted=True)
            metrics = dict(env._last_metrics, within_goal=True, supported=True, stable=True,
                           released=False, failure_reason=None)
            with patch.object(env, "_episode", episode), patch.object(env, "_last_metrics", metrics):
                state, history = env._physics_state().copy(), json.dumps(env._episode, sort_keys=True)
                expert.reset(seed=9)
                for _ in range(3):
                    action = expert.predict(observation)
                    np.testing.assert_array_equal(action, np.r_[np.zeros(7), 1.].astype(np.float32))
                np.testing.assert_array_equal(env._physics_state(), state)
                self.assertEqual(json.dumps(env._episode, sort_keys=True), history)
                with patch.dict(env._last_metrics, {"within_goal": False}):
                    action = expert.predict(observation)
                    self.assertEqual(expert.stage, "failed")
                    self.assertIn("left", expert.error)
                    self.assertTrue(env.action_space.contains(action))
                self.assertTrue(np.array_equal(state, env._physics_state()))

    def test_physical_goal_takeover_opens_withdraws_and_releases(self):
        with SortingEnv(observation_mode="proprio") as env:
            observation, info = env.reset(seed=0, options=dict(task_id="red_cube_to_red", layout="shuffled", clutter_level=1))
            nominal = BenchmarkExpert(env)
            nominal.reset()
            correction = PlacementReleaseExpert(env)
            for _ in range(1000):
                if correction.eligible():
                    break
                action = nominal(observation)
                self.assertIsNone(nominal.error)
                observation, _, terminated, truncated, info = env.step(action)
                self.assertFalse(terminated or truncated)
            else:
                self.fail("Physical expert never reached a supported-goal takeover state")
            self.assertTrue(info["episode"]["ever_grasped"] and info["episode"]["ever_lifted"])
            self.assertFalse(info["metrics"]["released"])
            steps_before = env._episode["steps"]
            checkpoint = env.get_state()
            correction.reset()
            self.assertEqual(env.get_state(), checkpoint)
            for _ in range(250):
                state, history = env._physics_state().copy(), json.dumps(env._episode, sort_keys=True)
                action = correction.predict(observation)
                self.assertIsNone(correction.error)
                self.assertTrue(env.action_space.contains(action))
                np.testing.assert_array_equal(env._physics_state(), state)
                self.assertEqual(json.dumps(env._episode, sort_keys=True), history)
                # The existing robot backend owns gravity compensation. The
                # correction must not apply any force to manipulated objects.
                for spec in env.objects.values():
                    np.testing.assert_array_equal(env.data.qfrc_applied[spec["dof"]:spec["dof"] + 6], 0)
                self.assertTrue(np.all(env.data.xfrc_applied == 0))
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            self.assertTrue(info["is_success"], info)
            self.assertGreater(info["episode"]["steps"], steps_before)
            for field in ("within_goal", "supported", "stable", "released"):
                self.assertTrue(info["metrics"][field], field)
            self.assertFalse(info["metrics"]["grasped"])


if __name__ == "__main__":
    unittest.main()
