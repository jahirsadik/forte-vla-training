"""Parent-trace parity and opt-in behavior preservation; synthetic CPU fixtures."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Install optional requirements-learning.txt")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.diffusion_policy import ActionDiffusion, CHECKPOINT_SCHEMA, ModelConfig, Tokenizer
from learning.preservation import PreservationCache, preservation_options, sampling_trace
from scripts import train_policy
from test_correction_training import AnchorFixture, CONTRACT, record


class PublicFixture:
    def __init__(self, tokenizer, count=3):
        self.tokenizer = tokenizer
        self.count = count
        self.cache_bytes = count * 2 * 3 * 24 * 32
        self.metadata = dict(scope="synthetic public-only fixture", anchors=list(range(count)))

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return (torch.full((2, 3, 24, 32), index * 17, dtype=torch.uint8), torch.full((23,), index / 10),
                torch.from_numpy(self.tokenizer.encode("pick the red cube")))


class BehaviorPreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(71)
        self.config = ModelConfig(image_height=24, image_width=32, hidden_dim=32,
                                  diffusion_steps=8, inference_steps=4)
        self.tokenizer = Tokenizer.fit(["pick the red cube"])
        self.model = ActionDiffusion(self.config, len(self.tokenizer.words)).eval()
        self.dataset = PublicFixture(self.tokenizer)

    def batch(self):
        return [torch.stack(values) for values in zip(*(self.dataset[index] for index in range(len(self.dataset))))]

    def cache(self, **kwargs):
        return PreservationCache(deepcopy(self.model), self.dataset, noise_seeds=[201, 202],
                                 batch_size=kwargs.get("batch_size", 3), device=torch.device("cpu"))

    def test_trace_matches_public_sampler_exactly_including_clipped_ddim(self):
        batch = self.batch()
        trace = sampling_trace(self.model, *batch, generator=torch.Generator().manual_seed(501))
        sampled = self.model.sample(*batch, generator=torch.Generator().manual_seed(501))
        torch.testing.assert_close(trace["sample"], sampled, rtol=0, atol=0)
        self.assertEqual(trace["timesteps"].tolist(), [7, 5, 2, 0])
        alpha = self.model.alpha_bar[7]
        raw_clean = (trace["noisy"][:, 0] - (1 - alpha).sqrt() * trace["targets"][:, 0]) / alpha.sqrt()
        self.assertTrue(bool((raw_clean.abs() > 1).any()))
        clean = raw_clean.clamp(-1, 1)
        corrected_noise = (trace["noisy"][:, 0] - alpha.sqrt() * clean) / (1 - alpha).sqrt()
        previous = self.model.alpha_bar[5]
        expected_next = previous.sqrt() * clean + (1 - previous).sqrt() * corrected_noise
        torch.testing.assert_close(trace["noisy"][:, 1], expected_next, rtol=0, atol=0)
        self.assertFalse(trace["targets"].requires_grad)

    def test_equal_student_zero_loss_and_perturbed_student_has_gradient_teacher_frozen(self):
        teacher = deepcopy(self.model)
        cache = PreservationCache(teacher, self.dataset, noise_seeds=[20], batch_size=3, device=torch.device("cpu"))
        student = deepcopy(self.model).train()
        loss = cache.loss(student, 6, generator=torch.Generator().manual_seed(13))
        self.assertLess(float(loss.detach()), 1e-12)
        with torch.no_grad():
            student.output.bias.add_(.2)
        changed_loss = cache.loss(student, 6, generator=torch.Generator().manual_seed(13))
        self.assertGreater(float(changed_loss.detach()), .01)
        changed_loss.backward()
        self.assertGreater(float(student.output.bias.grad.abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None and not parameter.requires_grad for parameter in teacher.parameters()))
        torch.testing.assert_close(teacher.output.bias, self.model.output.bias, rtol=0, atol=0)

    def test_cache_deterministic_and_global_training_rng_unchanged(self):
        state = torch.random.get_rng_state().clone()
        cache = self.cache()
        torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
        again = self.cache()
        self.assertEqual(cache.metadata["cache_sha256"], again.metadata["cache_sha256"])
        torch.testing.assert_close(cache.noisy, again.noisy, rtol=0, atol=0)
        torch.testing.assert_close(cache.targets, again.targets, rtol=0, atol=0)
        split = self.cache(batch_size=2)
        torch.testing.assert_close(cache.noisy[:, :, 0], split.noisy[:, :, 0], rtol=0, atol=0)
        torch.testing.assert_close(cache.samples, split.samples, rtol=1e-5, atol=1e-5)
        generator = torch.Generator().manual_seed(53)
        cache.loss(self.model, 3, generator=generator)
        torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)

    def test_flags_are_explicit_positive_and_compatible(self):
        options = dict(init_checkpoint="parent.pt", allow_interventions=True,
                       prediction_type="epsilon", action_representation="velocity")
        self.assertFalse(preservation_options(None, None, 8, 2, **options))
        self.assertTrue(preservation_options(10., 400, 8, 4, **options))
        for weight in (0, -1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                preservation_options(weight, 400, 8, 2, **options)
        for values in ((None, 400, 8, 2), (10., None, 8, 2), (10., 400, 0, 2), (10., 400, 8, 0)):
            with self.assertRaises(ValueError):
                preservation_options(*values, **options)
        for key, value in (("init_checkpoint", None), ("allow_interventions", False),
                           ("prediction_type", "sample"), ("action_representation", "joint_reference")):
            with self.assertRaises(ValueError):
                preservation_options(10., 400, 8, 2, **dict(options, **{key: value}))

    def test_cache_rejects_bad_seeds_configuration_and_action_bearing_samples(self):
        for seeds in ([], [1, 1], [-1], [True]):
            with self.assertRaises(ValueError):
                PreservationCache(deepcopy(self.model), self.dataset, noise_seeds=seeds,
                                  batch_size=3, device=torch.device("cpu"))
        incompatible = ActionDiffusion(replace(self.config, prediction_type="sample"), len(self.tokenizer.words))
        with self.assertRaisesRegex(ValueError, "epsilon"):
            PreservationCache(incompatible, self.dataset, noise_seeds=[1], device=torch.device("cpu"))
        class ActionFixture(PublicFixture):
            def __getitem__(self, index):
                return (*super().__getitem__(index), torch.zeros(8, 8))
        with self.assertRaisesRegex(ValueError, "only images"):
            PreservationCache(deepcopy(self.model), ActionFixture(self.tokenizer), noise_seeds=[1], device=torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_trace_parity_and_deterministic_student_backward(self):
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            teacher = deepcopy(self.model).cuda().eval()
            batch = [value.cuda() for value in self.batch()]
            trace = sampling_trace(teacher, *batch, generator=torch.Generator(device="cuda").manual_seed(91))
            sample = teacher.sample(*batch, generator=torch.Generator(device="cuda").manual_seed(91))
            torch.testing.assert_close(trace["sample"], sample, rtol=0, atol=0)
            cache = PreservationCache(teacher, self.dataset, noise_seeds=[91, 92], batch_size=3, device=torch.device("cuda"))
            student = deepcopy(self.model).cuda().train()
            with torch.no_grad():
                student.output.bias.add_(.1)
            loss = cache.loss(student, 6, generator=torch.Generator().manual_seed(92))
            loss.backward()
            self.assertTrue(bool(torch.isfinite(student.output.bias.grad).all()))
            self.assertGreater(float(student.output.bias.grad.abs().sum()), 0)
            self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_tiny_training_preserves_parent_provenance_and_separates_objectives(self):
        normalization = dict(mean=[.1] * 23, std=[.2] * 23, split="train", count=100, std_floor=.001)
        parent = dict(schema=CHECKPOINT_SCHEMA, model_config=asdict(self.config), vocabulary=self.tokenizer.words,
                      normalization=normalization, action_contract=CONTRACT, step=73,
                      metadata=dict(training_episodes=[record()], validation_episodes=[]),
                      model_state=deepcopy(self.model.state_dict()))
        nominal = dict(record("new-nominal"), is_intervention=False)
        intervention = dict(record("new-correction"), is_intervention=True)
        anchor_fixture = AnchorFixture([False] * 4 + [True] * 2, self.tokenizer)
        previous = torch.are_deterministic_algorithms_enabled()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "parent.pt"
            torch.save(parent, checkpoint)
            output = root / "fit"
            argv = ["train_policy.py", "--train", "fixture.h5", "--output", str(output),
                    "--init-checkpoint", str(checkpoint), "--allow-interventions", "--device", "cpu",
                    "--steps", "2", "--batch-size", "6", "--threads", "1", "--image-size", "24", "32",
                    "--hidden-dim", "32", "--diffusion-steps", "8", "--inference-steps", "4", "--eval-every", "2",
                    "--preservation-weight", "10", "--preserve-prefix-steps", "400", "--preservation-noise-seeds", "2"]
            try:
                with patch.object(sys, "argv", argv), patch.object(train_policy, "expand_paths", return_value=["fixture.h5"]), \
                        patch.object(train_policy, "inspect_episodes", return_value=([nominal, intervention], [])), \
                        patch.object(train_policy, "CompactChunkDataset", return_value=anchor_fixture), \
                        patch.object(train_policy, "PreservationDataset", return_value=self.dataset) as dataset:
                    train_policy.main()
                self.assertEqual(dataset.call_args.args[0], [intervention])
                self.assertEqual(dataset.call_args.args[3], normalization)
                self.assertEqual(dataset.call_args.kwargs["parent_checkpoint_sha256"], train_policy.file_sha256(checkpoint))
                actual = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
                self.assertEqual(actual["normalization"], normalization)
                self.assertEqual(actual["vocabulary"], self.tokenizer.words)
                preservation = actual["metadata"]["preservation"]
                self.assertEqual(preservation["weight"], 10)
                self.assertEqual(preservation["noise_seeds"], [200000, 200001])
                self.assertIn("not expert labels", preservation["targets"])
                metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
                for item in metrics:
                    self.assertAlmostEqual(item["train_total_loss"], item["train_noise_mse"] + 10 * item["train_preservation_epsilon_mse"], places=6)
                self.assertTrue(any(not torch.equal(value, parent["model_state"][key])
                                    for key, value in actual["training_model_state"].items()))
            finally:
                torch.use_deterministic_algorithms(previous)


if __name__ == "__main__":
    unittest.main()
