"""Long demonstrations must not silently determine instruction sampling mass."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_policy import training_sampler


class Anchors:
    split = "train"
    records = [dict(instruction="red cube to red"), dict(instruction="red cube to red"),
               dict(instruction="red cube to blue")]

    def __init__(self):
        self.episode_index = np.array([0, 0, 1, 1, 1, 1, 1, 1, 2], np.int32)
        self.is_intervention = np.zeros(9, bool)

    def __len__(self):
        return len(self.episode_index)


class InstructionSamplingTests(unittest.TestCase):
    def test_balances_languages_across_unequal_episode_lengths(self):
        sampler, metadata = training_sampler(Anchors(), None, seed=3, instruction_balanced=True)
        self.assertAlmostEqual(float(sampler.weights[:8].sum()), .5)
        self.assertAlmostEqual(float(sampler.weights[8]), .5)
        self.assertEqual(metadata["kind"], "instruction_balanced_anchors")
        self.assertEqual(sorted(row["anchors"] for row in metadata["instructions"]), [1, 8])
        self.assertTrue(all(row["expected_probability"] == .5 for row in metadata["instructions"]))

    def test_sampling_reproducible_without_using_global_rng(self):
        before = torch.random.get_rng_state().clone()
        first, _ = training_sampler(Anchors(), None, seed=7, instruction_balanced=True)
        second, _ = training_sampler(Anchors(), None, seed=7, instruction_balanced=True)
        self.assertEqual(list(first), list(second))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_existing_uniform_and_correction_paths_are_unchanged(self):
        self.assertEqual(training_sampler(Anchors(), None, seed=3),
                         (None, dict(kind="uniform_shuffled_anchors", anchors=9)))
        data = Anchors()
        data.is_intervention[-1] = True
        sampler, metadata = training_sampler(data, .25, seed=3)
        self.assertAlmostEqual(float(sampler.weights[-1]), .25)
        self.assertAlmostEqual(float(sampler.weights[:-1].sum()), .75)
        self.assertEqual(metadata["kind"], "weighted_supervised_anchors")

    def test_ambiguous_weighting_and_nontraining_data_rejected(self):
        with self.assertRaisesRegex(ValueError, "combined"):
            training_sampler(Anchors(), .25, seed=3, instruction_balanced=True)
        data = Anchors()
        data.split = "validation"
        with self.assertRaisesRegex(ValueError, "training data"):
            training_sampler(data, None, seed=3, instruction_balanced=True)
        with self.assertRaisesRegex(ValueError, "boolean"):
            training_sampler(Anchors(), None, seed=3, instruction_balanced=1)

    def test_invalid_anchor_membership_and_missing_language_rejected(self):
        for indices in (np.array([], int), np.array([-1, 2]), np.array([3]), np.array([0.0])):
            data = Anchors()
            data.episode_index = indices
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "membership"):
                training_sampler(data, None, seed=3, instruction_balanced=True)
        data = Anchors()
        data.records = [dict(instruction=" ") for _ in data.records]
        with self.assertRaisesRegex(ValueError, "nonempty language"):
            training_sampler(data, None, seed=3, instruction_balanced=True)


if __name__ == "__main__":
    unittest.main()
