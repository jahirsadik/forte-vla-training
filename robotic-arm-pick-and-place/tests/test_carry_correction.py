"""Carry takeover boundaries, observed-history gate, and honest physical continuation."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controllers.benchmark_expert import BenchmarkExpert
from controllers.carry_correction import CarryRecoveryExpert
from environments.sorting import SortingEnv


class CarryCorrectionTests(unittest.TestCase):
    def test_physical_gate_rejects_wrong_domain_and_wrong_object(self):
        with SortingEnv(observation_mode="proprio") as env:
            expert = CarryRecoveryExpert(env)
            self.assertFalse(expert.eligible())
            obs, _ = env.reset(seed=0)
            self.assertFalse(expert.eligible())
            with self.assertRaisesRegex(ValueError, "lifted target"):
                expert.reset()
            with self.assertRaisesRegex(RuntimeError, "Reset"):
                expert.predict(obs)
            episode = dict(env._episode, ever_grasped=True, ever_lifted=True)
            metrics = dict(env._last_metrics, grasped=True, within_goal=False, supported=False,
                           failure_reason=None, dropped_objects=[])
            # Synthetic gate fixtures, not rollouts or training data.
            with patch.object(env, "_episode", episode), patch.object(env, "_last_metrics", metrics):
                self.assertTrue(expert.eligible())
                for change in ({"done": True}, {"ever_grasped": False}, {"ever_lifted": False}):
                    with patch.dict(episode, change):
                        self.assertFalse(expert.eligible())
                for change in ({"grasped": False}, {"failure_reason": "distractor_disturbed"},
                               {"dropped_objects": [env.target]}, {"within_goal": True, "supported": True}):
                    with patch.dict(metrics, change):
                        self.assertFalse(expert.eligible())
                contacts, _, _ = env._contacts()
                other = next(name for name in contacts if name != env.target)
                contacts[other]["l"] = contacts[other]["r"] = True
                with patch.object(env, "_contacts", return_value=(contacts, 0., 0.)):
                    self.assertFalse(expert.eligible())
                with patch.object(expert, "_forbidden_contact", return_value=True):
                    self.assertFalse(expert.eligible())
                with patch.object(env, "scene_config", {}):
                    self.assertFalse(expert.eligible())

    def test_gate_requires_full_history_excludes_goal_entry_and_preserves_history_on_reset(self):
        with SortingEnv(observation_mode="proprio") as env:
            _, info = env.reset(seed=0)
            expert = CarryRecoveryExpert(env)
            episode = dict(env._episode, ever_grasped=True, ever_lifted=True)
            metrics = dict(env._last_metrics, grasped=True, within_goal=False, supported=False,
                           failure_reason=None, dropped_objects=[])
            public_robot = SimpleNamespace(target_q=np.ones(7) * .06, arm_position=np.zeros(7))

            def observation(step, goal=False, distance=.2):
                return dict(episode=dict(info["episode"], steps=step),
                            metrics=dict(info["metrics"], within_goal=goal, goal_distance_m=distance))

            with patch.object(env, "_episode", episode), patch.object(env, "_last_metrics", metrics):
                with patch.object(expert, "robot", public_robot):
                    self.assertFalse(expert.should_intervene())
                    for step in range(10, 60):
                        expert.observe(observation(step))
                    self.assertFalse(expert.should_intervene(), "Late observer must not invent prehistory")
                    for step in range(41):
                        expert.observe(observation(step))
                    self.assertTrue(expert.should_intervene())
                    expert.observe(observation(40))
                    self.assertEqual(len(expert.history), 41)
                history = deepcopy(expert.history)
                diagnostics = deepcopy(expert.diagnostics)
                state = env.get_state()
                expert.reset(seed=23)
                self.assertEqual(expert.history, history)
                self.assertTrue(expert.history_complete)
                self.assertEqual(expert.diagnostics["gate"], diagnostics["gate"])
                self.assertEqual(env.get_state(), state)
                with patch.object(expert, "robot", public_robot):
                    for step in range(81):
                        expert.observe(observation(step, goal=step == 5))
                    self.assertFalse(expert.should_intervene(), "Historical goal entry survives the rolling window")
                    for step in range(41):
                        expert.observe(observation(step, distance=.4 - step * .001))
                    self.assertFalse(expert.should_intervene(), "A progressing carry is not stalled")
                    for step in range(41):
                        expert.observe(observation(step))
                    public_robot.target_q[:] = .01
                    for step in range(41, 51):
                        expert.observe(observation(step))
                    self.assertFalse(expert.should_intervene(), "Grasp force without reference lag is not the gate")
                json.dumps(expert.diagnostics)

    def test_physical_lifted_grasp_continues_to_stable_release(self):
        with SortingEnv(observation_mode="proprio") as env:
            obs, info = env.reset(seed=0, options=dict(task_id="red_cube_to_red", layout="shuffled", clutter_level=1))
            nominal, correction = BenchmarkExpert(env), CarryRecoveryExpert(env)
            nominal.reset()
            correction.observe(info)
            for _ in range(1000):
                if correction.eligible():
                    break
                action = nominal(obs)
                self.assertIsNone(nominal.error)
                obs, _, terminated, truncated, info = env.step(action)
                correction.observe(info)
                self.assertFalse(terminated or truncated)
            else:
                self.fail("Nominal expert never made a lifted grasp")
            self.assertFalse(correction.should_intervene(), "Proactive test continuation must not be called a stall")
            before = env.get_state()
            correction.reset()
            self.assertEqual(env.get_state(), before)
            for _ in range(env.horizon - env._episode["steps"]):
                state, history = env._physics_state().copy(), json.dumps(env._episode, sort_keys=True)
                action = correction.predict(obs)
                self.assertIsNone(correction.error)
                self.assertTrue(env.action_space.contains(action))
                self.assertIn(float(action[-1]), (-1., 1.))
                np.testing.assert_array_equal(state, env._physics_state())
                self.assertEqual(history, json.dumps(env._episode, sort_keys=True))
                for spec in env.objects.values():
                    np.testing.assert_array_equal(env.data.qfrc_applied[spec["dof"]:spec["dof"] + 6], 0.)
                self.assertTrue(np.all(env.data.xfrc_applied == 0))
                obs, _, terminated, truncated, info = env.step(action)
                correction.observe(info)
                if terminated or truncated:
                    break
            self.assertTrue(info["is_success"], info)
            self.assertTrue(info["metrics"]["supported"] and info["metrics"]["released"] and info["metrics"]["stable"])
            self.assertFalse(info["metrics"]["grasped"])
            self.assertEqual(info["episode"]["wrong_grasp_steps"], 0)
            json.dumps(correction.diagnostics)

    def test_intentional_release_allows_grasp_loss_but_bounds_settling(self):
        with SortingEnv(observation_mode="proprio") as env:
            obs, _ = env.reset(seed=0)
            correction = CarryRecoveryExpert(env)
            correction.stage = "release"
            correction.commanded_release = True
            correction.expert = SimpleNamespace()
            metrics = dict(env._last_metrics, grasped=False, within_goal=True, supported=False,
                           stable=False, released=False, failure_reason=None, dropped_objects=[])
            before = env.get_state()
            with patch.object(env, "_last_metrics", metrics), patch.object(correction, "_carry_action", return_value=np.r_[np.zeros(7), 1.].astype(np.float32)):
                for _ in range(4):
                    action = correction.predict(obs)
                    self.assertIsNone(correction.error)
                    self.assertEqual(action[-1], 1.)
                self.assertEqual(correction.grasp_loss_steps, 0)
                for _ in range(37):
                    action = correction.predict(obs)
                self.assertIn("settle", correction.error)
                self.assertEqual(action[-1], 1., "Failure must not reclose after intentional opening")
            self.assertEqual(env.get_state(), before)


if __name__ == "__main__":
    unittest.main()
