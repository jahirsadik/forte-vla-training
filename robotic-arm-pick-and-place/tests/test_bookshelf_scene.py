"""Physical scene contracts for future bookshelf controllers and data collection."""

from pathlib import Path
import json
import sys
import tempfile
import unittest

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from environments.bookshelf import SCENE, load_bookshelf, validate_hold
from tools.build_bookshelf_scene import build


class BookshelfSceneTests(unittest.TestCase):
    def setUp(self):
        self.robot = load_bookshelf()
        self.model, self.data = self.robot.model, self.robot.data
        self.config = json.loads((PROJECT / "configs/bookshelf.json").read_text())

    def test_furniture_loads_without_robot_and_books_have_physical_mass(self):
        environment = mujoco.MjModel.from_xml_path(
            str(SCENE.parent / "environment.xml")
        )
        self.assertEqual(environment.neq, 0)
        self.assertEqual(environment.nu, 0)
        self.assertEqual(environment.njnt, len(self.config["books"]))
        for book in self.config["books"]:
            body = environment.body("book_" + book["name"])
            joint = environment.body_jntadr[body.id]
            self.assertEqual(environment.jnt_type[joint], mujoco.mjtJoint.mjJNT_FREE)
            self.assertAlmostEqual(float(body.mass[0]), book["mass_kg"])
            d, t, h = (book[k] for k in ("depth_m", "thickness_m", "height_m"))
            expected = (
                book["mass_kg"]
                / 12
                * np.array([t * t + h * h, d * d + h * h, d * d + t * t])
            )
            np.testing.assert_allclose(body.inertia, expected, rtol=1e-8)
        self.assertEqual(self.model.neq, 1)  # Original gripper mimic only.
        self.assertEqual(self.model.nu, 0)

    def test_books_settle_and_robot_holds_without_collision(self):
        report = validate_hold(self.robot, 2.0)
        self.assertEqual(report["warning_count"], 0)
        self.assertLess(report["book_max_translation_m"], 0.003)
        self.assertLess(report["book_max_velocity_norm"], 0.01)
        self.assertLess(report["peak_robot_penetration_m"], 1e-5)
        self.assertLess(report["peak_book_penetration_m"], 0.001)
        self.assertLess(report["arm_max_hold_error_rad"], 0.001)
        controlled = np.r_[self.robot.dof_indices, self.robot.gripper.driver_dof]
        np.testing.assert_array_equal(
            self.data.qfrc_applied[np.setdiff1d(np.arange(self.model.nv), controlled)],
            0,
        )
        self.assertTrue(
            np.all(np.abs(self.robot.last_effort) <= self.robot.effort_limits)
        )

    def _move_orange(self, position):
        self.robot.reset()
        adr = self.model.joint("book_orange_joint").qposadr[0]
        self.data.qpos[adr : adr + 7] = [*position, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        book = self.model.geom("book_orange_collision").id
        return {
            self.model.geom(c.geom2 if c.geom1 == book else c.geom1).name
            for c in self.data.contact
            if book in (c.geom1, c.geom2) and c.dist < 0
        }

    def test_shelf_is_open_but_each_board_collides(self):
        goal = self.data.site("insertion_target").xpos.copy()
        position = goal.copy()
        position[2] += 0.002
        self.assertEqual(self._move_orange(position), set())
        shelf = self.config["shelf"]
        book = next(b for b in self.config["books"] if b["name"] == "orange")
        sx, sy = shelf["center_xy_m"]
        board = shelf["board_m"]
        top = self.config["table"]["surface_z_m"]
        tests = {
            "shelf_back": [
                sx + shelf["depth_m"] / 2 - board - book["depth_m"] / 2 + 0.001,
                goal[1],
                goal[2],
            ],
            "shelf_left": [
                goal[0],
                sy + shelf["width_m"] / 2 - board - book["thickness_m"] / 2 + 0.001,
                goal[2],
            ],
            "shelf_right": [
                goal[0],
                sy - shelf["width_m"] / 2 + board + book["thickness_m"] / 2 - 0.001,
                goal[2],
            ],
            "shelf_bottom": [
                goal[0],
                goal[1],
                top + shelf["bottom_surface_m"] + book["height_m"] / 2 - 0.001,
            ],
            "shelf_upper": [
                goal[0],
                goal[1],
                top + shelf["upper_surface_m"] - board - book["height_m"] / 2 + 0.001,
            ],
        }
        for expected, point in tests.items():
            with self.subTest(board=expected):
                self.assertIn(expected, self._move_orange(point))

    def test_book_to_book_collision_and_deterministic_reset(self):
        blue = self.data.body("book_blue").xpos.copy()
        self.assertIn("book_blue_collision", self._move_orange(blue))
        self.robot.reset()
        initial = self.data.qpos.copy()
        for _ in range(50):
            self.robot.step()
        self.robot.reset()
        np.testing.assert_array_equal(self.data.qpos, initial)
        self.assertEqual(self.data.time, 0)
        for camera in ("overview", "front", "overhead", "wrist"):
            self.assertGreaterEqual(self.model.camera(camera).id, 0)

    def test_rebuild_and_relocation_use_repo_assets_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = build(destination=Path(tmp))
            rebuilt = load_bookshelf(scene)
            np.testing.assert_allclose(rebuilt.data.qpos, self.data.qpos)
            np.testing.assert_allclose(rebuilt.model.body_mass, self.model.body_mass)
            first = (Path(tmp) / "scene.xml").read_bytes()
            build(destination=Path(tmp))
            self.assertEqual((Path(tmp) / "scene.xml").read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
