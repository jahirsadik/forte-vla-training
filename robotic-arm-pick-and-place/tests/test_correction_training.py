"""Warm-start provenance, frozen preprocessing and correction replay sampling."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
try:
    import torch
except ImportError:
    raise unittest.SkipTest("Install optional requirements-learning.txt")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.diffusion_policy import ActionDiffusion, CHECKPOINT_SCHEMA, ModelConfig, Tokenizer
from scripts import train_policy


CONTRACT = dict(id="forte-joint-reference-v2", shape=[8], arm_joint_count=7, policy_hz=20,
                cameras=["front", "wrist"], image_size_hw=[24, 32], privileged_fields_are_policy_inputs=False)


def record(case="parent", *, split="train", family=None, instruction="pick the red cube"):
    return dict(path=f"/{case}.h5", sha256=case + "-sha", case_id=case, family_id=family or case + "-family",
                split=split, instruction=instruction, action_contract=deepcopy(CONTRACT),
                collection_implementation=dict(source_sha256="fixture-collection"), transitions=6)


class AnchorFixture:
    """Synthetic supervised reader fixture; never a physical demonstration."""

    def __init__(self, flags, tokenizer=None):
        self.is_intervention = np.array(flags, dtype=bool)
        self.cache_bytes = 1024
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.is_intervention)

    def normalization(self):
        raise AssertionError("Warm start must not refit normalization")

    def set_normalization(self, normalization):
        self.applied_normalization = deepcopy(normalization)

    def __getitem__(self, index):
        return (torch.zeros(2, 3, 24, 32, dtype=torch.uint8), torch.zeros(23),
                torch.from_numpy(self.tokenizer.encode("pick the red cube")),
                torch.zeros(8, 8), torch.ones(8, dtype=torch.bool))


class CorrectionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = ModelConfig(image_height=24, image_width=32, hidden_dim=32,
                                  diffusion_steps=8, inference_steps=4)
        self.tokenizer = Tokenizer.fit(["pick the red cube"])
        torch.manual_seed(12)
        self.model = ActionDiffusion(self.config, len(self.tokenizer.words))
        self.normalization = dict(mean=[.1] * 23, std=[.2] * 23, split="train", count=100, std_floor=.001)
        self.parent = dict(schema=CHECKPOINT_SCHEMA, model_config=asdict(self.config),
                           vocabulary=self.tokenizer.words, normalization=self.normalization,
                           action_contract=CONTRACT, step=73,
                           metadata=dict(training_episodes=[record()], validation_episodes=[],
                                         learner_source_sha256="parent-learner", training_source_sha256="parent-collection"),
                           model_state=deepcopy(self.model.state_dict()),
                           training_model_state={k: torch.zeros_like(v) for k, v in self.model.state_dict().items()},
                           optimizer_state=dict(state={0: {"step": torch.tensor(91.)}}, param_groups=[]))
        self.path = self.root / "parent.pt"

    def save(self, parent=None):
        torch.save(self.parent if parent is None else parent, self.path)
        return self.path

    def initialize(self, *, config=None, train=None, validation=None):
        return train_policy.load_initialization(self.path, self.config if config is None else config,
                                                [record("new")] if train is None else train,
                                                [] if validation is None else validation, device=torch.device("cpu"))

    def test_exact_ema_weights_and_frozen_preprocessing_are_reused(self):
        self.save()
        model, tokenizer, normalization, metadata = self.initialize()
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(parameter, self.parent["model_state"][name], rtol=0, atol=0)
        self.assertEqual(tokenizer.words, self.tokenizer.words)
        self.assertEqual(normalization, self.normalization)
        self.assertEqual(metadata["parent_step"], 73)
        self.assertEqual(metadata["parent_learner_source_sha256"], "parent-learner")
        self.assertIn("fresh", metadata["optimizer"])
        normalization["mean"][0] = 99
        self.assertEqual(self.parent["normalization"]["mean"][0], .1)

    def test_config_contract_unknown_training_words_and_bad_normalization_rejected(self):
        self.save()
        with self.assertRaisesRegex(ValueError, "ModelConfig"):
            self.initialize(config=replace(self.config, prediction_type="sample"))
        changed = record("new")
        changed["action_contract"]["image_size_hw"] = [48, 64]
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            self.initialize(train=[changed])
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            self.initialize(train=[record("new", instruction="pick the blue cube")])
        for normalization in (dict(self.normalization, split="validation"),
                              dict(self.normalization, count=0),
                              dict(self.normalization, std=[0.] * 23)):
            with self.subTest(normalization=normalization):
                parent = deepcopy(self.parent)
                parent["normalization"] = normalization
                self.save(parent)
                with self.assertRaises(ValueError):
                    self.initialize()

    def test_test_derived_parent_and_unexplained_vocabulary_rejected(self):
        for key in ("training_episodes", "validation_episodes"):
            parent = deepcopy(self.parent)
            parent["metadata"][key] = [record("test-data", split="test")]
            self.save(parent)
            with self.assertRaisesRegex(ValueError, "provenance"):
                self.initialize()
        parent = deepcopy(self.parent)
        parent["metadata"]["training_episodes"] = []
        self.save(parent)
        with self.assertRaisesRegex(ValueError, "declared parent"):
            self.initialize()
        parent = deepcopy(self.parent)
        parent["metadata"]["training_episodes"][0]["instruction"] = "pick red cube"
        self.save(parent)
        with self.assertRaisesRegex(ValueError, "not explained"):
            self.initialize()

    def test_parent_ancestry_cannot_cross_new_train_validation_boundary(self):
        self.save()
        with self.assertRaisesRegex(ValueError, "Validation overlaps parent training ancestry"):
            self.initialize(validation=[record("new-val", split="validation", family="parent-family")])
        parent = deepcopy(self.parent)
        parent["metadata"]["validation_episodes"] = [record("parent-val", split="validation")]
        self.save(parent)
        with self.assertRaisesRegex(ValueError, "Training overlaps parent validation ancestry"):
            self.initialize(train=[record("new", family="parent-val-family")])
        parent = deepcopy(self.parent)
        parent["metadata"]["initialization"] = dict(ancestral_training_episodes=[record("ancestor")])
        self.save(parent)
        with self.assertRaisesRegex(ValueError, "ancestry"):
            self.initialize(validation=[record("new-val", split="validation", family="ancestor-family")])
        parent["metadata"]["initialization"]["ancestral_training_episodes"][0]["split"] = "test"
        self.save(parent)
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.initialize()

    def test_correction_sampler_allocates_mass_independently_of_group_size(self):
        dataset = AnchorFixture([False] * 90 + [True] * 10)
        sampler, metadata = train_policy.training_sampler(dataset, .25, seed=8)
        self.assertAlmostEqual(float(sampler.weights[:90].sum()), .75)
        self.assertAlmostEqual(float(sampler.weights[90:].sum()), .25)
        self.assertEqual(metadata["nominal_anchors"], 90)
        self.assertEqual(metadata["intervention_anchors"], 10)
        again, _ = train_policy.training_sampler(dataset, .25, seed=8)
        self.assertEqual(list(sampler), list(again))
        default, summary = train_policy.training_sampler(dataset, None, seed=8)
        self.assertIsNone(default)
        self.assertEqual(summary["kind"], "uniform_shuffled_anchors")
        for flags in ([False] * 10, [True] * 10):
            with self.assertRaisesRegex(ValueError, "both nominal"):
                train_policy.training_sampler(AnchorFixture(flags), .25, seed=8)

    def test_intervention_sampling_is_explicit_and_velocity_only(self):
        self.assertIsNone(train_policy.correction_fraction(False, None, "velocity"))
        self.assertEqual(train_policy.correction_fraction(True, None, "velocity"), .25)
        with self.assertRaisesRegex(ValueError, "requires --allow"):
            train_policy.correction_fraction(False, .25, "velocity")
        with self.assertRaisesRegex(ValueError, "velocity"):
            train_policy.correction_fraction(True, None, "joint_reference")
        for fraction in (0, 1, -.1, np.nan, np.inf, True):
            with self.assertRaises(ValueError):
                train_policy.correction_fraction(True, fraction, "velocity")

    def test_main_one_update_uses_fresh_optimizer_and_preserves_parent_statistics(self):
        self.save()
        output = self.root / "fit"
        anchor_fixture = AnchorFixture([False] * 4 + [True] * 2, self.tokenizer)
        argv = ["train_policy.py", "--train", "fixture.h5", "--output", str(output),
                "--init-checkpoint", str(self.path), "--allow-interventions", "--device", "cpu",
                "--steps", "1", "--batch-size", "6", "--threads", "1", "--image-size", "24", "32",
                "--hidden-dim", "32", "--diffusion-steps", "8", "--inference-steps", "4", "--eval-every", "1"]
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            with patch.object(sys, "argv", argv), patch.object(train_policy, "expand_paths", return_value=["fixture.h5"]), \
                    patch.object(train_policy, "inspect_episodes", return_value=([record("new")], [])) as inspect, \
                    patch.object(train_policy, "CompactChunkDataset", return_value=anchor_fixture):
                train_policy.main()
            inspect.assert_called_once_with(["fixture.h5"], "train", allow_interventions=True)
            actual = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(actual["step"], 1)
            self.assertEqual(actual["normalization"], self.normalization)
            self.assertEqual(anchor_fixture.applied_normalization, self.normalization)
            self.assertEqual(actual["vocabulary"], self.tokenizer.words)
            self.assertEqual(actual["metadata"]["initialization"]["parent_step"], 73)
            self.assertEqual(actual["metadata"]["sampling"]["expected_intervention_probability"], .25)
            self.assertTrue(actual["optimizer_state"]["state"])
            self.assertTrue(all(int(state["step"]) == 1 for state in actual["optimizer_state"]["state"].values()))
            self.assertTrue(any(not torch.equal(actual["training_model_state"][k], self.parent["model_state"][k])
                                for k in self.parent["model_state"]))
        finally:
            torch.use_deterministic_algorithms(previous)


if __name__ == "__main__":
    unittest.main()
