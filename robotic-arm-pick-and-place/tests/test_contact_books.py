"""Physical contact extraction, isolation, and preservation of the nominal path."""

from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controllers.benchmark_expert import BenchmarkExpert
from controllers.contact_books import ContactBookExpert
from environments.bookshelf_task import BookshelfEnv
from environments.bookshelf_hard import BookshelfHardEnv


class ContactBookPlanningTests(unittest.TestCase):
    def test_dense_press_preflight_is_scratch_only_and_restores_contact_flags(self):
        with BookshelfHardEnv() as env:
            env.reset(seed=0, options=dict(task_id='extract_teal', layout='mixed'))
            before = env._physics_state().copy()
            expert = ContactBookExpert(env)
            expert.reset(seed=0)
            np.testing.assert_array_equal(before, env._physics_state())
            self.assertFalse(expert.ik.target_contacts)
            prep = expert.preparation
            self.assertIsNotNone(prep)
            self.assertTrue(any('Contact approach preflight:' in item['error']
                                for item in prep.report['planning_attempts']))
            self.assertIn(prep.force_target_n, (3., 8.))
            self.assertEqual(prep.report['press_preflight_m'], .020)
            # The accepted approach must leave a screened pressure corridor.
            prep._screen_press_approach(prep.press_position, prep.quaternion)
            self.assertFalse(expert.ik.target_contacts)
            np.testing.assert_array_equal(before, env._physics_state())

    def test_pressure_selection_distinguishes_tight_and_roomy_approaches(self):
        with BookshelfHardEnv() as env:
            expert = ContactBookExpert(env)
            for task, seed, expected_force in (('extract_blue', 0, 3.),
                                               ('extract_yellow', 1, 8.)):
                env.reset(seed=seed, options=dict(task_id=task, layout='mixed'))
                before = env._physics_state().copy()
                expert.reset(seed=seed)
                np.testing.assert_array_equal(before, env._physics_state())
                self.assertEqual(expert.preparation.force_target_n, expected_force)
                self.assertFalse(expert.ik.target_contacts)
                reserve = expert.contact_summary['pressure_reserve_m']
                self.assertGreaterEqual(reserve, .020)
                self.assertEqual(reserve >= .049, expected_force == 8.)

    def test_preparation_is_scratch_only_force_guard_and_reset(self):
        with BookshelfEnv() as env:
            observation, _ = env.reset(seed=1, options=dict(task_id='extract_blue', layout='shuffled'))
            before = env._physics_state().copy()
            expert = ContactBookExpert(env)
            expert.reset(seed=1)
            self.assertIsNotNone(expert.preparation)
            np.testing.assert_array_equal(before, env._physics_state())
            # A sensor-abort fixture: no force is applied to the simulation.
            expert.preparation.contact_force = lambda: 16.
            action = expert(observation)
            self.assertIn('abort limit', expert.error)
            np.testing.assert_array_equal(action[:-1], 0)
            np.testing.assert_array_equal(before, env._physics_state())
            env.reset(seed=0, options=dict(task_id='extract_red', layout='shuffled'))
            expert.reset(seed=0)
            self.assertIsNone(expert.preparation)
            self.assertIsNone(expert.contact_summary)
            self.assertIsNone(expert.error)
            self.assertFalse(expert.prepared)
            self.assertFalse(expert.ik.sliding_geoms)

    def test_accessible_book_keeps_the_nominal_controller_path(self):
        with BookshelfEnv() as env:
            env.reset(seed=0, options=dict(task_id='extract_red', layout='shuffled'))
            direct, contact = BenchmarkExpert(env), ContactBookExpert(env)
            direct.reset(seed=0)
            contact.reset(seed=0)
            self.assertIsNone(contact.preparation)
            self.assertEqual(len(direct.commands), len(contact.commands))
            for a, b in zip(direct.commands, contact.commands):
                for x, y in zip(a, b):
                    np.testing.assert_array_equal(x, y)


class PhysicalContactBookTests(unittest.TestCase):
    def run_extraction(self, cls, task, seed, layout):
        with cls() as env:
            observation, info = env.reset(seed=seed, options=dict(task_id=task, layout=layout))
            expert = ContactBookExpert(env)
            expert.reset(seed=seed)
            self.assertIsNotNone(expert.preparation)
            for _ in range(env.horizon):
                before = env._physics_state().copy()
                action = expert(observation)
                np.testing.assert_array_equal(before, env._physics_state())
                self.assertIsNone(expert.error, expert.error)
                self.assertTrue(env.action_space.contains(action))
                observation, _, done, truncated, info = env.step(action)
                for spec in env.objects.values():
                    np.testing.assert_array_equal(env.data.qfrc_applied[spec['dof']:spec['dof']+6], 0)
                self.assertIsNone(env.robot.payload_body)
                if done or truncated:
                    break
            self.assertTrue(info['is_success'], info)
            self.assertTrue(info['episode']['ever_grasped'])
            self.assertTrue(info['episode']['ever_lifted'])
            self.assertTrue(expert.contact_summary['completed'])
            self.assertGreater(expert.contact_summary['withdrawal_m'], .04)
            self.assertLessEqual(expert.contact_summary['peak_force_n'], 15.)
            self.assertTrue(info['metrics']['released'])
            self.assertTrue(info['metrics']['stable'])
            return expert.planning_attempts

    def test_wall_blocked_blue_book(self):
        self.run_extraction(BookshelfEnv, 'extract_blue', 1, 'shuffled')

    def test_flush_red_book_in_touching_mixed_row(self):
        self.run_extraction(BookshelfHardEnv, 'extract_red', 0, 'mixed')

    def test_teal_book_previously_blocked_before_pulling(self):
        self.run_extraction(BookshelfHardEnv, 'extract_teal', 0, 'mixed')

    def test_yellow_book_retains_previously_successful_pressure(self):
        self.run_extraction(BookshelfHardEnv, 'extract_yellow', 1, 'mixed')

    def test_carried_book_uses_screened_alternate_posture_and_releases(self):
        attempts = self.run_extraction(BookshelfHardEnv, 'extract_green', 1, 'mixed')
        self.assertTrue(any(item.get('route') == 'object_aware_joint_transfer'
                            and item.get('success') for item in attempts))


if __name__ == '__main__':
    unittest.main()
