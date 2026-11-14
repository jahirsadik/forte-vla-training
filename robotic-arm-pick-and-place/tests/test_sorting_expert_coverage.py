"""Sorting cadence scope and physical action-boundary regressions."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controllers.benchmark_expert import BenchmarkExpert
from environments.bookshelf_task import BookshelfEnv
from environments.sorting import SortingEnv


class SortingExpertCoverageTests(unittest.TestCase):
    def test_sorting_reset_selects_cadence_and_clears_failure_without_moving(self):
        with SortingEnv() as env:
            env.reset(seed=0, options={"task_id": "red_cube_to_red", "layout": "shuffled"})
            expert = BenchmarkExpert(env)
            self.assertEqual(expert.cartesian_minimum_steps, 10)
            before = env._physics_state().copy()
            expert.reset()
            self.assertEqual(expert.cartesian_minimum_steps, 5)
            expert.error, expert.failure_stage = "prior failure", "transfer"
            expert.cartesian_minimum_steps = 99
            expert.reset()
            self.assertIsNone(expert.error)
            self.assertIsNone(expert.failure_stage)
            self.assertEqual(expert.cartesian_minimum_steps, 5)
            np.testing.assert_array_equal(before, env._physics_state())

    def test_books_and_manual_carry_keep_original_cadence(self):
        with BookshelfEnv() as env:
            env.reset(seed=0, options={"task_id": "extract_red", "layout": "shuffled"})
            expert = BenchmarkExpert(env)
            # Isolate dispatch and reset lifecycle from book-route feasibility.
            expert._build_book_approach = Mock()
            expert.cartesian_minimum_steps = 5
            expert.reset()
            expert._build_book_approach.assert_called_once()
            self.assertEqual(expert.cartesian_minimum_steps, 10)
        with SortingEnv() as env:
            env.reset(seed=0)
            expert = BenchmarkExpert(env)
            # CarryRecoveryExpert creates a fresh planner and sets its kind
            # without rebuilding a table approach from the lifted grasp.
            expert.kind, expert.phase = "sorting", 2
            self.assertEqual(expert.cartesian_minimum_steps, 10)

    def test_faster_cartesian_queue_retains_speed_bounds_and_scratch_planning(self):
        with SortingEnv() as env:
            obs, _ = env.reset(seed=0)
            expert = BenchmarkExpert(env)
            expert.cartesian_minimum_steps = 5
            expert.width = env.robot.open_width
            quaternion = np.zeros(4)
            mujoco.mju_mat2Quat(quaternion, env.data.site_xmat[env.robot.site_id])
            before = env._physics_state().copy()
            expert._cartesian_segment(env.robot.tool_position + [0., 0., .02], quaternion, expert.width)
            np.testing.assert_array_equal(before, env._physics_state())
            previous = env.robot.target_q.copy()
            bound = env.protocol["joint_reference_speed_rad_s"] * env.control_dt
            for reference, _, geometry in expert.commands:
                self.assertLessEqual(float(np.max(np.abs(reference - previous))), bound + 1e-10)
                self.assertTrue(expert.ik.screen(geometry)[0])
                previous = reference
            count = len(expert.commands)
            self.assertGreater(count, 0)
            for _ in range(count):
                before = env._physics_state().copy()
                action = expert(obs)
                self.assertIsNone(expert.error)
                self.assertTrue(env.action_space.contains(action))
                np.testing.assert_array_equal(before, env._physics_state())
                obs, _, _, _, _ = env.step(action)
            self.assertIsNone(env.robot.payload_body)

    def test_slip_failure_retains_the_stage_that_failed(self):
        with SortingEnv() as env:
            obs, _ = env.reset(seed=0)
            expert = BenchmarkExpert(env)
            expert.width = env.robot.closed_width
            expert.phase, expert.stage, expert.grasp_loss_steps = 3, "transfer", 2
            before = env._physics_state().copy()
            action = expert(obs)
            self.assertTrue(env.action_space.contains(action))
            self.assertEqual(expert.stage, "failed")
            self.assertEqual(expert.failure_stage, "transfer")
            self.assertIn("slipped before commanded release", expert.error)
            np.testing.assert_array_equal(before, env._physics_state())


if __name__ == "__main__":
    unittest.main()
