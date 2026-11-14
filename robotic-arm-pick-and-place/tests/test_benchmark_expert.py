"""Expert planning boundaries and real contact-driven manipulation regressions."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controllers.benchmark_expert import BenchmarkExpert
from environments.sorting import SortingEnv
from environments.bookshelf_task import BookshelfEnv


class ExpertPlanningTests(unittest.TestCase):
    def test_planning_starts_from_measured_pose_under_reference_error(self):
        with SortingEnv() as env:
            env.reset(seed=0)
            expert = BenchmarkExpert(env)
            measured = env.robot.arm_position.copy()
            reference = measured.copy()
            reference[-1] += 0.04
            env.robot.set_arm_target(reference)
            np.testing.assert_array_equal(expert._planned_geometry(), measured)
            self.assertFalse(np.array_equal(measured, env.robot.target_q))

    def test_payload_reference_planning_preserves_live_physics(self):
        with SortingEnv() as env:
            env.reset(seed=0, options={"task_id": "red_cube_to_blue"})
            expert = BenchmarkExpert(env)
            before = env._physics_state().copy()
            expert.ik.set_payload(env.target)
            bias = expert.ik.payload_reference_bias(env.robot.arm_position)
            self.assertTrue(np.isfinite(bias).all())
            self.assertGreater(np.linalg.norm(bias), 1e-5)
            np.testing.assert_array_equal(before, env._physics_state())
            self.assertIsNone(env.robot.payload_body)

    def test_lost_grasp_aborts_before_an_accidental_success(self):
        with SortingEnv() as env:
            obs, _ = env.reset(seed=0, options={"task_id": "red_cube_to_blue"})
            expert = BenchmarkExpert(env)
            expert.reset()
            # Lifecycle fixture, not a trajectory or demonstration.
            expert.phase = 3
            env._last_metrics["grasped"] = False
            before = env._physics_state().copy()
            for _ in range(3):
                expert(obs)
            self.assertIn("slipped before commanded release", expert.error)
            np.testing.assert_array_equal(before, env._physics_state())


class PhysicalExpertTests(unittest.TestCase):
    def run_transfer(self, task, constructor=SortingEnv):
        with constructor() as env:
            obs, info = env.reset(
                seed=0, options={"task_id": task, "layout": "shuffled"}
            )
            expert = BenchmarkExpert(env)
            expert.reset(seed=0)
            for _ in range(env.horizon):
                before = env._physics_state().copy()
                action = expert(obs)
                np.testing.assert_array_equal(before, env._physics_state())
                self.assertIsNone(expert.error, expert.error)
                self.assertTrue(env.action_space.contains(action))
                obs, _, done, truncated, info = env.step(action)
                self.assertIsNone(env.robot.payload_body)
                for spec in env.objects.values():
                    np.testing.assert_array_equal(
                        env.data.qfrc_applied[spec["dof"] : spec["dof"] + 6], 0
                    )
                if done or truncated:
                    break
            self.assertTrue(info["is_success"], info)
            self.assertIn(expert.stage, ("release", "retreat", "settle"))
            self.assertTrue(info["episode"]["ever_grasped"])
            self.assertTrue(info["episode"]["ever_lifted"])

    def test_wedge_reaches_commanded_release(self):
        self.run_transfer("yellow_wedge_to_red")

    def test_low_puck_clears_table_and_is_placed(self):
        self.run_transfer("orange_puck_to_green")

    def test_book_insertion_releases_upright_in_slot(self):
        self.run_transfer("insert_orange", BookshelfEnv)

    def test_book_relocation_routes_around_table_book(self):
        self.run_transfer("relocate_teal", BookshelfEnv)

    def test_book_extraction_releases_flat_on_mat(self):
        self.run_transfer("extract_yellow", BookshelfEnv)


if __name__ == "__main__":
    unittest.main()
