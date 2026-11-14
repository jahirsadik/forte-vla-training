"""Grasp-before-lift takeover boundaries and physical continuation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controllers.benchmark_expert import BenchmarkExpert
from controllers.carry_correction import CarryRecoveryExpert
from controllers.grasp_correction import GraspToGoalExpert
from environments.sorting import SortingEnv
from learning.data import intervention_rollout_id
from scripts.collect_corrections import descriptor, make_correction_expert


class GraspCorrectionTests(unittest.TestCase):
    def test_gate_accepts_before_lift_but_rejects_unsafe_or_wrong_grasps(self):
        with SortingEnv(observation_mode="proprio") as env:
            obs, info = env.reset(seed=0)
            teacher = GraspToGoalExpert(env)
            self.assertFalse(teacher.eligible())
            with self.assertRaisesRegex(ValueError, "active target grasp"):
                teacher.reset()
            episode = dict(env._episode, ever_grasped=True, ever_lifted=False)
            metrics = dict(env._last_metrics, grasped=True, within_goal=False, supported=False,
                           failure_reason=None, dropped_objects=[])
            with patch.object(env, "_episode", episode), patch.object(env, "_last_metrics", metrics):
                self.assertTrue(teacher.eligible())
                self.assertFalse(CarryRecoveryExpert(env).eligible())
                for change in ({"done": True}, {"ever_grasped": False}):
                    with patch.dict(episode, change): self.assertFalse(teacher.eligible())
                for change in ({"grasped": False}, {"failure_reason": "distractor_disturbed"},
                               {"dropped_objects": [env.target]}, {"within_goal": True, "supported": True}):
                    with patch.dict(metrics, change): self.assertFalse(teacher.eligible())
                contacts, _, _ = env._contacts()
                other = next(name for name in contacts if name != env.target)
                contacts[other]["l"] = contacts[other]["r"] = True
                with patch.object(env, "_contacts", return_value=(contacts, 0., 0.)):
                    self.assertFalse(teacher.eligible())
                with patch.object(teacher, "_forbidden_contact", return_value=True):
                    self.assertFalse(teacher.eligible())

    def test_gate_requires_contiguous_reset_history_and_sustained_grasp(self):
        with SortingEnv(observation_mode="proprio") as env:
            _, info = env.reset(seed=0)
            teacher = GraspToGoalExpert(env)
            def observe(step, grasped=True):
                teacher.observe(dict(episode=dict(info['episode'], steps=step),
                                     metrics=dict(info['metrics'], grasped=grasped)))
            with patch.object(teacher, "eligible", return_value=True):
                for step in range(10, 20): observe(step)
                self.assertFalse(teacher.should_intervene())
                observe(0, False)
                for step in range(1, 6): observe(step)
                for _ in range(10): observe(5)
                self.assertFalse(teacher.should_intervene(), "duplicate reads cannot manufacture stability")
                observe(6)
                self.assertTrue(teacher.should_intervene())
                observe(7, False)
                self.assertFalse(teacher.should_intervene())
                for step in range(8, 14): observe(step)
                self.assertTrue(teacher.should_intervene())
                observe(15)
                self.assertFalse(teacher.should_intervene(), "missing control step invalidates history")

    def test_expert_identity_is_distinct_and_factory_explicit(self):
        plan = dict(version="forte-correction-pilot-v2", confirmation_steps=3)
        rollout = dict(policy_seed=0, minimum_policy_steps=100, expert=GraspToGoalExpert.expert_id)
        grasp = descriptor(plan, rollout, "a" * 64)
        carry = descriptor(plan, dict(rollout, expert=CarryRecoveryExpert.expert_id), "a" * 64)
        self.assertEqual(grasp['version'], 'forte-grasp-intervention-v1')
        self.assertNotEqual(intervention_rollout_id('case', grasp), intervention_rollout_id('case', carry))
        with patch('controllers.grasp_correction.GraspToGoalExpert') as cls:
            env = object()
            self.assertIs(make_correction_expert(GraspToGoalExpert.expert_id, env), cls.return_value)

    def test_physical_unlifted_grasp_continues_without_live_state_edits(self):
        with SortingEnv(observation_mode="proprio") as env:
            obs, info = env.reset(seed=0, options=dict(task_id="red_cube_to_red", layout="shuffled", clutter_level=1))
            nominal, teacher = BenchmarkExpert(env), GraspToGoalExpert(env)
            nominal.reset(); teacher.observe(info)
            for _ in range(1000):
                if teacher.should_intervene(): break
                action = nominal(obs)
                self.assertIsNone(nominal.error)
                obs, _, terminated, truncated, info = env.step(action)
                teacher.observe(info)
                self.assertFalse(terminated or truncated)
            else: self.fail("No sustained nominal grasp")
            self.assertFalse(info['episode']['ever_lifted'])
            before = env.get_state(); history = deepcopy(teacher.grasp_history)
            teacher.reset()
            self.assertEqual(env.get_state(), before)
            self.assertEqual(teacher.grasp_history, history)
            for _ in range(env.horizon - env._episode['steps']):
                before = env._physics_state().copy(); episode = json.dumps(env._episode, sort_keys=True)
                action = teacher(obs)
                self.assertIsNone(teacher.error)
                self.assertTrue(env.action_space.contains(action))
                np.testing.assert_array_equal(before, env._physics_state())
                self.assertEqual(episode, json.dumps(env._episode, sort_keys=True))
                self.assertTrue(np.all(env.data.xfrc_applied == 0))
                for spec in env.objects.values():
                    np.testing.assert_array_equal(env.data.qfrc_applied[spec['dof']:spec['dof']+6], 0.)
                obs, _, terminated, truncated, info = env.step(action)
                teacher.observe(info)
                if terminated or truncated: break
            self.assertTrue(info['is_success'], info)
            self.assertTrue(all(info['metrics'][k] for k in ('within_goal','released','supported','stable')))


if __name__ == '__main__': unittest.main()
