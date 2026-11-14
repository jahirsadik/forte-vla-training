"""Behavior preservation conditions stay distinct from expert action targets."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
try:
    import torch
except ImportError:
    raise unittest.SkipTest("Install optional requirements-learning.txt")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.data import inspect_episodes
from learning.diffusion_policy import ModelConfig, Tokenizer
from learning.preservation_data import PreservationDataset
from test_intervention_data import fixture as base_fixture


def fixture(path, **kwargs):
    """Extend the shared reader fixture to fit full 8-action teacher chunks."""
    takeover = kwargs.pop("takeover", 16)
    meta = base_fixture(path, takeover=3, **kwargs)
    n = 32
    kind = meta["policy"]
    if kind == "expert_intervention":
        meta["intervention"]["actual_takeover_step"] = takeover
    with h5py.File(path, "r+") as f:
        first_action = f["transitions/action"][0]
        actions = np.repeat(first_action[None], n, 0)
        prefix = 0 if kind == "expert" else takeover
        flags = np.arange(n) >= prefix if kind == "expert_intervention" else np.zeros(n, bool)
        data = {
            "observations/proprio": np.repeat(np.arange(n+1, dtype=np.float32)[:, None], 23, 1),
            "observations/time_s": np.arange(n+1) * .05,
            "observations/rgb/front": np.zeros((n+1,24,32,3), np.uint8),
            "observations/rgb/wrist": np.zeros((n+1,24,32,3), np.uint8),
            "privileged/integration_state": np.zeros((n+1,3)),
            "transitions/action": actions, "transitions/proposed_action": actions,
            "transitions/source": np.where(np.arange(n)<prefix,"policy","expert").astype(object),
            "transitions/intervention": flags, "transitions/expert_action_valid": flags,
            "transitions/expert_action": np.where(flags[:,None], actions, np.nan).astype(np.float32),
        }
        for key, value in data.items():
            del f[key]
            f.create_dataset(key, data=value, **({"dtype": h5py.string_dtype()} if key=="transitions/source" else {}))
        f.attrs["metadata"] = json.dumps(meta)
    return meta


class PreservationDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.config = ModelConfig(image_height=24, image_width=32, hidden_dim=32, diffusion_steps=8, inference_steps=4)
        self.tokenizer = Tokenizer.fit(["move the red cube"])
        self.normalization = dict(split="train", mean=[1.] * 23, std=[2.] * 23, count=999)

    def tearDown(self):
        self.directory.cleanup()

    def records(self, *paths):
        return inspect_episodes(paths, "train", allow_interventions=True)[0]

    def dataset(self, records, **kwargs):
        return PreservationDataset(records, self.config, self.tokenizer, self.normalization,
                                   parent_checkpoint_sha256=kwargs.pop("parent_checkpoint_sha256", "a" * 64),
                                   prefix_steps=kwargs.pop("prefix_steps", 16), **kwargs)

    def test_cutoff_and_takeover_are_exclusive_no_action_labels(self):
        path = self.root / "episode.h5"
        fixture(path, takeover=24)
        dataset = self.dataset(self.records(path), prefix_steps=16, anchor_stride=8)
        self.assertEqual([a["transition_index"] for a in dataset.metadata["anchors"]], [0, 8])
        partial = self.dataset(self.records(path), prefix_steps=15, anchor_stride=8)
        self.assertEqual([a["transition_index"] for a in partial.metadata["anchors"]], [0])
        early = self.root / "early.h5"; fixture(early, takeover=19)
        bounded_by_takeover = self.dataset(self.records(early), prefix_steps=32, anchor_stride=8)
        self.assertEqual([a["transition_index"] for a in bounded_by_takeover.metadata["anchors"]], [0, 8])
        self.assertEqual(len(dataset[0]), 3)
        self.assertEqual(dataset[0][0].dtype, torch.uint8)
        self.assertEqual(dataset[0][1].shape, (23,))
        self.assertFalse(hasattr(dataset, "actions"))
        self.assertFalse(dataset.metadata["action_labels"])
        bounded = self.dataset(self.records(path), prefix_steps=999, anchor_stride=8)
        self.assertEqual([a["transition_index"] for a in bounded.metadata["anchors"]], [0, 8, 16])
        self.assertGreater(dataset.cache_bytes, 0)
        repeated = self.dataset(self.records(path), prefix_steps=16, anchor_stride=8)
        self.assertEqual(dataset.metadata["selection_sha256"], repeated.metadata["selection_sha256"])
        self.assertNotEqual(dataset.metadata["selection_sha256"], bounded.metadata["selection_sha256"])

    def test_exact_public_observations_deduplicate_across_distinct_rollouts(self):
        a, b = self.root / "a.h5", self.root / "b.h5"
        fixture(a, seed=0, takeover=24)
        fixture(b, seed=1, takeover=24)
        # Different executed actions make distinct valid trajectories; common
        # public prefix observations are still only one distillation condition.
        records = self.records(a, b)
        dataset = self.dataset(records, prefix_steps=24, anchor_stride=8)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.metadata["candidate_anchor_count"], 6)
        self.assertEqual(dataset.metadata["duplicate_anchor_count"], 3)
        self.assertEqual([a["duplicate_of_index"] for a in dataset.metadata["duplicate_anchors"]], [0, 1, 2])
        with h5py.File(b, "r+") as f:
            f["observations/rgb/wrist"][8, 0, 0, 0] = 1
        changed = self.dataset(self.records(a, b), prefix_steps=24, anchor_stride=8)
        self.assertEqual(len(changed), 4)
        self.assertEqual(changed.metadata["duplicate_anchor_count"], 2)

    def test_public_inputs_only_ignore_privileged_state_and_future_suffix(self):
        a, b = self.root / "a.h5", self.root / "b.h5"
        fixture(a, seed=0, takeover=16)
        fixture(b, seed=1, takeover=16)
        with h5py.File(b, "r+") as f:
            f["privileged/integration_state"][:] = np.nan
            f["observations/proprio"][16:] = -999
            f["observations/rgb/front"][16:] = 255
            # Labels are different valid expert actions; never distillation inputs.
            f["transitions/action"][16:, 0] = .9
            f["transitions/proposed_action"][16:, 0] = .9
            f["transitions/expert_action"][16:, 0] = .9
        first = self.dataset(self.records(a), anchor_stride=1)
        second = self.dataset(self.records(b), anchor_stride=1)
        for i in range(len(first)):
            for x, y in zip(first[i], second[i]):
                torch.testing.assert_close(x, y)
        self.assertEqual(first.metadata["input_fields"], ["rgb", "proprio", "instruction"])

    def test_parent_normalization_is_copied_and_never_refit(self):
        path = self.root / "episode.h5"; fixture(path)
        original = copy.deepcopy(self.normalization)
        dataset = self.dataset(self.records(path), anchor_stride=1)
        self.assertEqual(dataset.normalization, original)
        self.assertFalse(dataset.metadata["normalization_refit"])
        np.testing.assert_array_equal(dataset[0][1], np.full(23, -.5, np.float32))
        np.testing.assert_array_equal(dataset[2][1], np.full(23, .5, np.float32))
        self.normalization["mean"][0] = 500.
        np.testing.assert_array_equal(dataset[0][1], np.full(23, -.5, np.float32))
        self.assertEqual(dataset.normalization, original)

    def test_wrong_parent_nontrain_and_stale_records_rejected(self):
        path = self.root / "episode.h5"; fixture(path)
        records = self.records(path)
        with self.assertRaisesRegex(ValueError, "parent checkpoint mismatch"):
            self.dataset(records, parent_checkpoint_sha256="c" * 64)
        for split in ("validation", "test"):
            modified = copy.deepcopy(records); modified[0]["split"] = split
            with self.assertRaisesRegex(ValueError, "train records"):
                self.dataset(modified)
        with h5py.File(path, "r+") as f:
            f["observations/proprio"][0, 0] = 5
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            self.dataset(records)
        # An actual frozen test episode cannot bypass record metadata checks.
        fixture(path, split="test")
        records[0]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "test episode"):
            self.dataset(records)

    def test_corrupt_takeover_failed_and_nominal_records_rejected(self):
        path = self.root / "episode.h5"; fixture(path)
        records = self.records(path)
        with h5py.File(path, "r+") as f:
            f["transitions/expert_action_valid"][0] = True
        with self.assertRaisesRegex(ValueError, "unlabeled policy prefix"):
            self.dataset(records)
        fixture(path, success=False)
        with self.assertRaisesRegex(ValueError, "No complete successful"):
            self.dataset(records)
        fixture(path, kind="expert")
        with self.assertRaisesRegex(ValueError, "intervention-only"):
            self.dataset(self.records(path))

    def test_invalid_selection_normalization_and_reference_mode_rejected(self):
        path = self.root / "episode.h5"; fixture(path)
        records = self.records(path)
        for value in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                self.dataset(records, prefix_steps=value)
            with self.assertRaises(ValueError):
                self.dataset(records, anchor_stride=value)
        with self.assertRaises(ValueError):
            self.dataset([])
        with self.assertRaisesRegex(ValueError, "full model horizon"):
            self.dataset(records, prefix_steps=7)
        short = self.root / "short.h5"; fixture(short, takeover=3)
        with self.assertRaisesRegex(ValueError, "zero full-chunk"):
            self.dataset(self.records(short))
        self.config = replace(self.config, action_representation="joint_reference")
        with self.assertRaisesRegex(ValueError, "velocity"):
            self.dataset(records)
        self.config = replace(self.config, action_representation="velocity")
        for bad in (dict(split="test", mean=[0.] * 23, std=[1.] * 23),
                    dict(split="train", mean=[0.] * 23, std=[0.] * 23),
                    dict(split="train", mean=[float("nan")] * 23, std=[1.] * 23)):
            self.normalization = bad
            with self.assertRaises(ValueError):
                self.dataset(records)


if __name__ == "__main__":
    unittest.main()
