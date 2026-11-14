"""Control limits, real kinematics, lifecycle and human provenance regressions."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import mujoco
import numpy as np

from controllers.gamepad import GamepadControl, PadState, TeleopSettings, deadzone
from environments.sorting import SortingEnv
from environments.bookshelf_task import BookshelfEnv
from environments.bookshelf_hard import BookshelfHardEnv
from scripts.inspect_teleop import inspect_episode
from utils.actions import decode_action
from utils.demonstrations import replay_episode, training_samples
from utils.teleoperation import TeleopSession, available_cases


class GamepadControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = SortingEnv()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset(seed=0, options={"task_id": "red_cube_to_red"})

    def test_input_validation_and_neutral_interlock(self):
        for value in (float("nan"), float("inf"), 1.1):
            with self.assertRaises(ValueError):
                PadState(lx=value)
        with self.assertRaises(ValueError):
            PadState(lt=-.1)
        with self.assertRaises(ValueError):
            TeleopSettings(deadzone=1.)
        self.assertTrue(PadState().neutral())
        self.assertFalse(PadState(buttons=("rb",)).neutral())
        self.assertFalse(PadState(connected=False).neutral())
        self.assertEqual(deadzone(.10, .12), 0.)

    def test_deadman_disconnect_and_jaw_hold(self):
        for mode in GamepadControl.modes:
            control = GamepadControl(self.env, mode)
            for pad in (PadState(lx=1, lt=1), PadState(lx=1, lt=1, buttons=("rb",), connected=False)):
                action, _ = control.action(pad)
                q, jaw = decode_action(action, self.env.robot.target_q, self.env.robot.gripper_target, self.env.action_contract)
                np.testing.assert_array_equal(q, self.env.robot.target_q)
                self.assertAlmostEqual(jaw, self.env.robot.gripper_target)
            action, _ = control.action(PadState(lt=1, buttons=("rb",)))
            _, jaw = decode_action(action, self.env.robot.target_q, self.env.robot.gripper_target, self.env.action_contract)
            self.assertLess(jaw, self.env.robot.gripper_target)

    def test_all_seven_joints_selection_edges_precision_and_limits(self):
        c = GamepadControl(self.env, "joint")
        for j in range(7):
            self.assertEqual(c.selected_joint, j)
            # Negative input stays inside the shoulder-pitch upper limit.
            action, _ = c.action(PadState(ly=1, buttons=("rb",)))
            self.assertLess(action[j], 0)
            self.assertEqual(np.count_nonzero(action[:-1]), 1)
            slow, _ = c.action(PadState(ly=1, buttons=("rb", "lb")))
            np.testing.assert_allclose(slow[:-1], .25 * action[:-1], atol=1e-7)
            c.action(PadState(buttons=("right",)))
            c.action(PadState(buttons=("right",)))
            self.assertEqual(c.selected_joint, (j+1) % 7)
            c.action(PadState())
        # Excess lag blocks forward accumulation, but still allows backing out.
        self.env.robot.target_q[0] = self.env.robot.arm_position[0] + .14
        action, diag = c.action(PadState(ly=-1, buttons=("rb",)))
        self.assertEqual(action[0], 0)
        self.assertTrue(diag["reference_limited"])
        action, _ = c.action(PadState(ly=1, buttons=("rb",)))
        self.assertLess(action[0], 0)

    def test_cartesian_direction_and_no_live_state_changes_in_every_scene(self):
        for cls in (SortingEnv, BookshelfEnv, BookshelfHardEnv):
            with cls() as env:
                env.reset(seed=0)
                c = GamepadControl(env)
                state = env._physics_state().copy()
                q = env.robot.target_q.copy()
                forces = env.data.xfrc_applied.copy()
                for pad in (PadState(ly=-1, buttons=("rb",)), PadState(ry=-1, buttons=("rb",)),
                            PadState(rx=.5, ly=.3, buttons=("rb", "y")), PadState(buttons=("rb", "right"))):
                    action, diag = c.action(pad)
                    self.assertTrue(env.action_space.contains(action))
                    np.testing.assert_array_equal(env._physics_state(), state)
                    np.testing.assert_array_equal(env.robot.target_q, q)
                    np.testing.assert_array_equal(env.data.xfrc_applied, forces)
                    if pad.ly == -1 or pad.ry == -1:
                        jac = c.jp[:, env.robot.dof_indices]
                        velocity = action[:-1] * env.action_contract["joint_reference_speed_rad_s"]
                        axis = 0 if pad.ly == -1 else 2
                        self.assertGreater((jac @ velocity)[axis], 0)

    def test_real_motion_is_bounded_and_finite(self):
        c = GamepadControl(self.env)
        initial = self.env.robot.tool_position.copy()
        for _ in range(25):
            action, _ = c.action(PadState(ry=-.5, buttons=("rb",)))
            _, _, done, truncated, _ = self.env.step(action)
            self.assertFalse(done or truncated)
        self.assertGreater(self.env.robot.tool_position[2], initial[2] + .005)
        self.assertTrue(np.isfinite(self.env._physics_state()).all())


class TeleopRecordingTests(unittest.TestCase):
    def test_training_only_selection(self):
        for env in ("sorting", "bookshelf", "bookshelf-hard"):
            _, cases = available_cases(env)
            self.assertTrue(all(c["split"] == "train" for c in cases))
            with self.assertRaises(ValueError):
                available_cases(env, seed=10000)

    def test_record_pause_disconnect_modes_failure_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            s = TeleopSession(output_root=directory, image_size=(64, 80), horizon_steps=4)
            try:
                self.assertFalse(s.toggle_record(PadState(buttons=("rb",))))
                self.assertFalse(s.toggle_record(PadState(connected=False)))
                self.assertTrue(s.toggle_record(PadState()))
                s.step(PadState(ry=-.3, buttons=("rb",)))
                s.pause("test_pause")
                physics = s.env._physics_state().copy()
                self.assertFalse(s.step(PadState(ry=-1, buttons=("rb",))))
                np.testing.assert_array_equal(physics, s.env._physics_state())
                s.toggle_mode()
                self.assertEqual(s.mode, "cartesian")
                s.toggle_record(PadState())
                self.assertFalse(s.step(PadState(connected=False)))
                self.assertTrue(s.paused)
                s.toggle_record(PadState())
                s.step(PadState())
                first = s.finish()
                self.assertFalse(first["success"])
                self.assertEqual(first["transitions"], 2)
                self.assertEqual(first["pause_count"], 2)
                self.assertTrue(inspect_episode(first["path"])["passed"])
                self.assertEqual(replay_episode(first["path"])["transitions"], 2)
                self.assertEqual(list(training_samples([first["path"]], include_failures=True)), [])
                with h5py.File(first["path"]) as f:
                    self.assertFalse(f.attrs["environment_done"])
                    self.assertEqual(f["observations/proprio"].shape[0], 3)
                s.toggle_mode()
                self.assertEqual(s.mode, "joint")
                s.toggle_record(PadState())
                for _ in range(4):
                    s.step(PadState(ly=.3, buttons=("rb",)))
                self.assertIsNone(s.writer)
                second = s.report["episodes"][-1]
                self.assertEqual(second["reason"], "environment_done")
                with h5py.File(second["path"]) as f:
                    self.assertTrue(f.attrs["environment_done"])
                    self.assertTrue(f["transitions/truncated"][-1])
                self.assertEqual(replay_episode(second["path"])["transitions"], 4)
                s.toggle_record(PadState())
                s.finish("empty_trial")
                self.assertEqual(s.report["episodes"][-1]["transitions"], 0)
                s.close()
                self.assertEqual(json.loads((s.path / "session.json").read_text())["status"], "closed")
            finally:
                s.env.close()


if __name__ == "__main__":
    unittest.main()
