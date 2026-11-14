"""Migration boundaries: source geometry, independent wrist, and policy contracts."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from controllers.ik import rotation_error
from environments.bookshelf_task import BookshelfEnv
from environments.sorting import SortingEnv
from robots.factory import load_robot
from tools.build_forte_scene import check
from utils.research_protocol import load_manifest, validate_manifest


class SevenJointModelTests(unittest.TestCase):
    def test_source_geometry_and_inertias_are_preserved(self):
        model, _ = check()
        self.assertEqual(model.nu, 0)
        self.assertEqual(model.neq, 1)

    def test_all_seven_columns_match_finite_difference_kinematics(self):
        robot = load_robot("forte")
        model, data = robot.model, robot.data
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, robot.site_id)
        initial = data.qpos.copy()
        position = robot.tool_position
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, data.site_xmat[robot.site_id])
        for index, dof in zip(robot.qpos_indices, robot.dof_indices):
            data.qpos[:] = initial
            data.qpos[index] += 1e-6
            mujoco.mj_forward(model, data)
            moved_quat = np.zeros(4)
            mujoco.mju_mat2Quat(moved_quat, data.site_xmat[robot.site_id])
            np.testing.assert_allclose((robot.tool_position - position) / 1e-6, jp[:, dof], atol=1e-6)
            np.testing.assert_allclose(rotation_error(moved_quat, quaternion) / 1e-6, jr[:, dof], atol=1e-6)
            self.assertGreater(np.linalg.norm(jr[:, dof]), .99)

    def test_wrist_moves_gripper_camera_but_not_upstream_or_fixed_cameras(self):
        with SortingEnv() as env:
            env.reset(seed=0)
            model, data = env.model, env.data
            joint = model.joint("wrist_roll").id
            parent = model.body_parentid[model.jnt_bodyid[joint]]
            before = data.xmat[parent].copy()
            camera = data.cam_xmat[model.camera("wrist").id].copy()
            fixed = data.cam_xmat[model.camera("overview").id].copy()
            tool = data.site_xmat[env.robot.site_id].copy()
            data.qpos[model.jnt_qposadr[joint]] += .2
            mujoco.mj_forward(model, data)
            np.testing.assert_array_equal(before, data.xmat[parent])
            np.testing.assert_array_equal(fixed, data.cam_xmat[model.camera("overview").id])
            self.assertGreater(np.linalg.norm(tool - data.site_xmat[env.robot.site_id]), .1)
            self.assertGreater(np.linalg.norm(camera - data.cam_xmat[model.camera("wrist").id]), .1)

    def test_unlimited_hinges_have_explicit_operating_bounds(self):
        robot = load_robot("forte")
        for j in robot.joint_ids[3:]:
            self.assertFalse(robot.model.jnt_limited[j])
        self.assertTrue(np.all(robot.upper[3:] - robot.lower[3:] > 2 * np.pi))
        target = robot.home.copy()
        target[-1] = .2
        robot.set_arm_target(target)
        with self.assertRaises(ValueError):
            robot.set_arm_target(robot.upper + .01)

    def test_jaw_command_and_physical_gap_are_distinct_and_monotonic(self):
        robot = load_robot("forte")
        grip = robot.gripper
        for width in (0., .03, .074):
            grip.initialize(robot.data, width)
            mujoco.mj_forward(robot.model, robot.data)
            self.assertAlmostEqual(grip.width_from_position(robot.data.qpos[grip.driver_qpos]), width)
            self.assertEqual(robot.data.qpos[grip.driver_qpos], robot.data.qpos[grip.follower_qpos])
            gap, center = grip.geometry(robot.data)
            self.assertAlmostEqual(gap, width + grip.config["pad_gap_offset_m"], places=7)
            self.assertLess(center, 1e-5)
        closing_force = grip.apply(robot.data, 0)
        self.assertEqual(closing_force, -grip.config["effort_limit_n"])

    def test_environments_expose_eight_actions_and_twenty_three_proprio_values(self):
        for constructor in (BookshelfEnv, SortingEnv):
            with self.subTest(environment=constructor.__name__), constructor() as env:
                obs, _ = env.reset(seed=0)
                self.assertEqual(env.action_space.shape, (8,))
                self.assertEqual(obs["proprio"].shape, (23,))
                self.assertEqual(env.action_contract["joint_names"][-1], "wrist_roll")
                self.assertEqual(env.action_contract["id"], "forte-joint-reference-v2")
                before = env._physics_state().copy()
                with self.assertRaises(ValueError):
                    env.step(np.zeros(7, dtype=np.float32))
                np.testing.assert_array_equal(before, env._physics_state())
                action = np.zeros(8, dtype=np.float32)
                action[-1] = 1
                obs, _, _, _, _ = env.step(action)
                self.assertEqual(obs["proprio"].shape, (23,))
                self.assertEqual(set(obs), {"proprio", "instruction"})

    def test_old_or_mismatched_robot_manifests_fail_closed(self):
        manifest = load_manifest()
        for field, value in (("version", "forte-research-v1"), ("robot_definition_sha256", "wrong")):
            altered = deepcopy(manifest)
            altered[field] = value
            with self.assertRaises(ValueError):
                validate_manifest(altered)


if __name__ == "__main__":
    unittest.main()
