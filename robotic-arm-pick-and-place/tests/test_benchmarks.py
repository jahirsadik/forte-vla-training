"""Task contract tests; goal fixtures validate scoring, not a successful policy."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environments.bookshelf_task import BookshelfEnv
from environments.sorting import SortingEnv


class BookshelfBenchmarkTests(unittest.TestCase):
    env_class = BookshelfEnv
    task = "insert_orange"
    alternate_task = "extract_blue"

    def setUp(self):
        self.env = self.env_class(horizon_steps=40)
        self.env.reset(seed=4, options={"task_id": self.task})
        self.hold = np.r_[np.zeros(len(self.env.robot.home)), 1].astype(np.float32)

    def tearDown(self):
        self.env.close()

    def test_gym_contract_seeded_resets_and_instruction_counterfactual(self):
        e = self.env
        check_env(e, skip_render_check=True)
        first, info = e.reset(seed=9, options={"task_id": self.task})
        physics = e._physics_state().copy()
        other, _ = e.reset(seed=9, options={"task_id": self.alternate_task})
        np.testing.assert_array_equal(physics, e._physics_state())
        self.assertNotEqual(first["instruction"], other["instruction"])
        self.assertEqual(set(first), {"instruction", "proprio"})
        self.assertFalse(info["is_success"])
        e.reset(seed=10, options={"task_id": self.task})
        self.assertFalse(np.array_equal(physics, e._physics_state()))

    def test_invalid_action_is_rejected_without_mutating_physics(self):
        state = self.env._physics_state().copy()
        for action in (np.zeros(2), self.hold * 2, np.full_like(self.hold, np.nan)):
            with self.assertRaises(ValueError):
                self.env.step(action)
            np.testing.assert_array_equal(state, self.env._physics_state())

    def test_failed_layout_reset_invalidates_previous_episode(self):
        with self.assertRaises(ValueError):
            self.env.reset(seed=0, options={"clutter_level": 99})
        with self.assertRaises(gym.error.ResetNeeded):
            self.env.step(self.hold)
        with self.assertRaises(gym.error.ResetNeeded):
            self.env.expert_state()
        with self.assertRaises(gym.error.ResetNeeded):
            self.env.get_state()
        self.env.reset(seed=0)
        self.env.step(self.hold)

    def test_replay_includes_reference_and_scoring_history(self):
        e = self.env
        action = self.hold.copy()
        action[0] = -0.2
        action[-1] = 0.5
        e.step(action)
        checkpoint = json.loads(json.dumps(e.get_state()))
        expected = [e.step(action) for _ in range(3)]
        expected_state = e._physics_state().copy()
        e.set_state(checkpoint)
        for outcome in expected:
            actual = e.step(action)
            np.testing.assert_array_equal(outcome[0]["proprio"], actual[0]["proprio"])
            self.assertEqual(outcome[1:], actual[1:])
        np.testing.assert_array_equal(expected_state, e._physics_state())
        with self.env_class(horizon_steps=41) as other:
            with self.assertRaises(ValueError):
                other.set_state(checkpoint)
        with self.env_class(horizon_steps=40) as other:
            other.set_state(checkpoint)
            outcome = other.step(action)
            np.testing.assert_array_equal(
                outcome[0]["proprio"], expected[0][0]["proprio"]
            )

    def _put_target_at_goal(self, *, history=False, height=0, quaternion=(1, 0, 0, 0)):
        e = self.env
        position = np.asarray(e.goal["position"]).copy()
        position[2] += height
        e._set_object_pose(e.target, position, quaternion)
        mujoco.mj_forward(e.model, e.data)
        # Explicit constructed prior-event fixture. No controller or physical
        # grasp success is claimed by this scoring test.
        e._episode["ever_grasped"] = history
        e._episode["ever_lifted"] = history
        e._previous_potential = e._potential()

    def test_placement_requires_grasp_lift_release_support_and_dwell(self):
        e = self.env
        self._put_target_at_goal(height=0.02)
        self.assertFalse(e._measure()["supported"])
        self.assertFalse(e._measure()["placement_valid"])
        self._put_target_at_goal()
        for _ in range(12):
            _, reward, terminated, _, info = e.step(self.hold)
        self.assertTrue(info["metrics"]["placement_valid"])
        self.assertFalse(
            terminated
        )  # Merely teleporting an object is not a pick-place task.
        self.assertEqual(reward, 0)
        e._episode["ever_grasped"] = e._episode["ever_lifted"] = True
        for _ in range(9):
            _, _, terminated, _, _ = e.step(self.hold)
            self.assertFalse(terminated)
        _, reward, terminated, truncated, info = e.step(self.hold)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["is_success"])
        self.assertEqual(reward, 1)
        with self.assertRaises(gym.error.ResetNeeded):
            e.step(self.hold)

    def test_wrong_orientation_and_wrong_object_do_not_succeed(self):
        e = self.env
        self._put_target_at_goal(quaternion=(np.sqrt(0.5), np.sqrt(0.5), 0, 0))
        self.assertFalse(e._measure()["placement_valid"])
        e.reset(seed=4, options={"task_id": self.task})
        other = next(n for n in e.objects if n != e.target)
        pos = e.data.xpos[e.objects[other]["body"]].copy()
        pos[0] -= 0.10
        e._set_object_pose(other, pos)
        mujoco.mj_forward(e.model, e.data)
        _, reward, terminated, truncated, info = e.step(self.hold)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["is_success"])
        self.assertEqual(reward, -1)
        self.assertEqual(info["metrics"]["failure_reason"], "distractor_disturbed")

    def test_timeout_is_truncation_drop_is_failure(self):
        e = self.env
        for _ in range(e.horizon):
            _, reward, terminated, truncated, info = e.step(self.hold)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, 0)
        self.assertFalse(info["is_success"])
        e.reset(seed=4, options={"task_id": self.task})
        e._set_object_pose(e.target, [1.4, 0, 0.4])
        mujoco.mj_forward(e.model, e.data)
        _, reward, terminated, _, info = e.step(self.hold)
        self.assertTrue(terminated)
        self.assertEqual(reward, -1)
        self.assertEqual(info["metrics"]["failure_reason"], "object_dropped")

    def test_dense_reward_telescopes_and_passive_books_receive_no_control(self):
        with self.env_class(horizon_steps=4, reward_mode="dense") as e:
            e.reset(seed=5, options={"task_id": self.task})
            initial = e._previous_potential
            reward_sum = 0.0
            for _ in range(4):
                _, reward, _, _, info = e.step(self.hold)
                reward_sum += reward
                for spec in e.objects.values():
                    np.testing.assert_array_equal(
                        e.data.qfrc_applied[spec["dof"] : spec["dof"] + 6], 0
                    )
            self.assertAlmostEqual(reward_sum, e._potential() - initial)
            self.assertEqual(info["metrics"]["reward_sparse"], 0)

    def test_finger_contact_is_detected_as_not_released(self):
        e = self.env
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, e.data.site_xmat[e.robot.site_id])
        e._set_object_pose(e.target, e.robot.tool_position, quat)
        width = 2 * e.objects[e.target]["half_extents"][1] - 0.004
        e.robot.gripper.initialize(e.data, width)
        mujoco.mj_forward(e.model, e.data)
        contacts, _, _ = e._contacts()
        self.assertTrue(contacts[e.target]["l"])
        self.assertTrue(contacts[e.target]["r"])
        self.assertFalse(e._measure()["released"])

    def test_extraction_goal(self):
        # This tests the other bookshelf operation as a physical scoring fixture.
        if self.env_class is not BookshelfEnv:
            return
        e = self.env
        e.reset(seed=4, options={"task_id": "extract_blue"})
        self._put_target_at_goal(
            history=True, quaternion=(np.sqrt(0.5), np.sqrt(0.5), 0, 0)
        )
        success = False
        for _ in range(25):
            _, _, terminated, _, info = e.step(self.hold)
            if terminated:
                success = info["is_success"]
                break
        self.assertTrue(success)


class SortingBenchmarkTests(BookshelfBenchmarkTests):
    env_class = SortingEnv
    task = "red_cube_to_blue"
    alternate_task = "red_cube_to_green"

    def test_wrong_orientation_and_wrong_object_do_not_succeed(self):
        e = self.env
        other_goal = e._make_goal(e.tasks[self.alternate_task])
        e._set_object_pose(e.target, other_goal["position"])
        mujoco.mj_forward(e.model, e.data)
        self.assertFalse(e._measure()["placement_valid"])
        self.assertFalse(e._measure()["within_goal"])
        other = next(n for n in e.objects if n != e.target)
        e._set_object_pose(other, e.goal["position"])
        mujoco.mj_forward(e.model, e.data)
        _, reward, terminated, _, info = e.step(self.hold)
        self.assertTrue(terminated)
        self.assertEqual(reward, -1)
        self.assertEqual(info["metrics"]["failure_reason"], "distractor_disturbed")

    def test_all_clutter_levels_reset_stably_and_trays_have_open_cavities(self):
        e = self.env
        for level in range(3):
            for seed in range(5):
                e.reset(
                    seed=seed, options={"task_id": self.task, "clutter_level": level}
                )
                _, _, terminated, _, info = e.step(self.hold)
                self.assertFalse(terminated)
                self.assertEqual(info["metrics"]["robot_penetration_m"], 0)
        with self.assertRaises(ValueError):
            e.reset(options={"clutter_level": 3})
        e.reset(seed=4, options={"task_id": self.task})
        self._put_target_at_goal(height=0.01)
        target_geoms = e.objects[e.target]["geoms"]
        self.assertFalse(
            any(
                c.dist < 0 and (c.geom1 in target_geoms or c.geom2 in target_geoms)
                for c in e.data.contact
            )
        )
        pos = np.array(e.goal["position"])
        pos[0] += e.scene_config["tray"]["size_xy_m"][0] / 2 - 0.005
        e._set_object_pose(e.target, pos)
        mujoco.mj_forward(e.model, e.data)
        self.assertTrue(
            any(
                c.dist < 0 and (c.geom1 in target_geoms or c.geom2 in target_geoms)
                for c in e.data.contact
            )
        )

    def test_extraction_goal(self):
        # Sorting has no extraction operation; verify its generated source instead.
        from tools.build_sorting_scene import build

        with tempfile.TemporaryDirectory() as tmp:
            scene = build(destination=Path(tmp))
            standalone = mujoco.MjModel.from_xml_path(
                str(scene.parent / "environment.xml")
            )
            complete = mujoco.MjModel.from_xml_path(str(scene))
            self.assertEqual(standalone.njnt, 6)
            self.assertEqual(standalone.neq, 0)
            self.assertEqual(complete.neq, 1)
            for obj in self.env.scene_config["objects"]:
                self.assertAlmostEqual(
                    float(standalone.body("object_" + obj["name"]).mass[0]),
                    obj["mass_kg"],
                )
            first = scene.read_bytes()
            build(destination=Path(tmp))
            self.assertEqual(first, scene.read_bytes())


if __name__ == "__main__":
    unittest.main()
