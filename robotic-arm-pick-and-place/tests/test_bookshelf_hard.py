"""Dense packing, physical neighbour contacts, and independent benchmark contracts."""

import sys
from pathlib import Path
import unittest
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environments.bookshelf_hard import BookshelfHardEnv
from environments.bookshelf_task import BookshelfEnv
from utils.research_protocol import HARD_MANIFEST, load_manifest, validate_manifest


class DenseBookshelfTests(unittest.TestCase):
    def test_packing_is_seeded_touching_and_stable(self):
        with BookshelfHardEnv() as env:
            for layout, count in (("narrow", 0), ("mixed", 6), ("touching", 10)):
                first, info = env.reset(seed=0, options=dict(task_id="extract_red", layout=layout))
                physical = env._physics_state().copy()
                self.assertEqual(len(info["metrics"]["touching_book_pairs"]), count)
                other, _ = env.reset(seed=0, options=dict(task_id="extract_blue", layout=layout))
                np.testing.assert_array_equal(physical, env._physics_state())
                self.assertNotEqual(first["instruction"], other["instruction"])
                for _ in range(20):
                    _, _, done, _, info = env.step(np.r_[np.zeros(7), 1])
                self.assertFalse(done, info)
                self.assertLess(info["metrics"]["distractor_translation_m"], .001)
                for spec in env.objects.values():
                    np.testing.assert_array_equal(env.data.qfrc_applied[spec["dof"]:spec["dof"]+6], 0)

    def test_touching_books_transmit_a_physical_impulse(self):
        with BookshelfHardEnv() as env:
            env.reset(seed=0, options=dict(task_id="extract_red", layout="touching"))
            names = sorted((n for n in env.objects if n != "book_orange"
                            and env.data.xpos[env.objects[n]["body"], 1] > 0),
                           key=lambda n: env.data.xpos[env.objects[n]["body"], 1])
            struck, neighbour = env.objects[names[2]], env.objects[names[3]]
            initial = env.data.xpos[neighbour["body"]].copy()
            # Explicit collision-test impulse, never an expert demonstration.
            env.data.qvel[struck["dof"]+1] = .2
            peak_force = 0.
            for _ in range(200):
                env.robot.step()
                for index, contact in enumerate(env.data.contact):
                    if {int(contact.geom1), int(contact.geom2)} == {struck["geoms"][0], neighbour["geoms"][0]}:
                        force = np.zeros(6)
                        mujoco.mj_contactForce(env.model, env.data, index, force)
                        peak_force = max(peak_force, abs(force[0]))
            # Shelf friction can arrest translation while contact transmits force.
            self.assertGreater(peak_force, .1)
            self.assertGreater(np.linalg.norm(env.data.xpos[neighbour["body"]] - initial), .000001)

    def test_tight_goal_and_final_neighbour_disturbance(self):
        with BookshelfHardEnv() as env:
            env.reset(seed=0, options=dict(task_id="insert_orange", layout="mixed"))
            self.assertAlmostEqual(env.goal["high"][1] - env.goal["low"][1], .043)
            # Constructed scoring fixture; no collection or expert success claim.
            env._set_object_pose(env.target, env.goal["position"])
            env._episode["ever_grasped"] = env._episode["ever_lifted"] = True
            other = "book_reference_1"
            position = env.data.xpos[env.objects[other]["body"]].copy()
            position[0] -= .025
            env._set_object_pose(other, position)
            mujoco.mj_forward(env.model, env.data)
            for _ in range(15):
                _, _, done, _, info = env.step(np.r_[np.zeros(7), 1])
            self.assertFalse(info["metrics"]["neighbors_within_final_tolerance"])
            self.assertFalse(info["is_success"])
            self.assertFalse(done)  # Recoverable displacement below the transient limit.

    def test_original_suite_and_hard_suite_are_distinct(self):
        original = load_manifest()
        hard = load_manifest(HARD_MANIFEST)
        validate_manifest(hard)
        self.assertEqual(len(original["cases"]), 829)
        self.assertTrue({c["family_id"] for c in original["cases"]}.isdisjoint(
                        {c["family_id"] for c in hard["cases"]}))
        with BookshelfEnv() as easy, BookshelfHardEnv() as dense:
            self.assertEqual(len(easy.objects), 7)
            self.assertEqual(len(dense.objects), 13)
            self.assertEqual(set(easy.tasks), set(dense.tasks))
            self.assertNotEqual(easy.action_contract["environment_protocol"], dense.action_contract["environment_protocol"])
            self.assertEqual(dense.action_space.shape, (8,))


if __name__ == "__main__":
    unittest.main()
