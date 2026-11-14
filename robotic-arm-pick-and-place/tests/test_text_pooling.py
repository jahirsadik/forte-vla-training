"""Opt-in recurrent-state pooling preserves legacy checkpoint behavior."""

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
from learning.diffusion_policy import ActionDiffusion, CHECKPOINT_SCHEMA, DiffusionPolicy, ModelConfig, Tokenizer
from scripts import train_policy

CONTRACT = dict(id="forte-joint-reference-v2", shape=[8], arm_joint_count=7, policy_hz=20,
                cameras=["front", "wrist"], image_size_hw=[24, 32], privileged_fields_are_policy_inputs=False)


def fixture_record(name="parent", instruction="pick red cube into blue tray"):
    return dict(path=f"/{name}.h5", sha256=name + "-sha", case_id=name, family_id=name + "-family",
                split="train", instruction=instruction, action_contract=deepcopy(CONTRACT),
                collection_implementation=dict(source_sha256="synthetic-fixture"), transitions=4)


class TinyDataset:
    """Synthetic reader fixture for one optimizer step, never a demonstration."""

    cache_bytes = 1024

    def __init__(self, records, config, tokenizer, **kwargs):
        self.tokenizer = tokenizer

    def __len__(self):
        return 4

    def normalization(self):
        return dict(mean=[0.] * 23, std=[1.] * 23, split="train", count=4)

    def set_normalization(self, normalization):
        self.normalization_used = normalization

    def __getitem__(self, index):
        return (torch.zeros(2, 3, 24, 32, dtype=torch.uint8), torch.zeros(23),
                torch.from_numpy(self.tokenizer.encode("pick red cube into blue tray")),
                torch.zeros(8, 8), torch.ones(8, dtype=torch.bool))


class TextPoolingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.instructions = ["pick red cube into blue tray", "pick blue cube into red tray"]
        self.tokenizer = Tokenizer.fit(self.instructions, max_tokens=16)
        self.config = ModelConfig(image_height=24, image_width=32, hidden_dim=32,
                                  diffusion_steps=8, inference_steps=4, max_tokens=16)
        torch.manual_seed(31)
        self.model = ActionDiffusion(self.config, len(self.tokenizer.words))
        self.images = torch.randint(0, 256, (2, 2, 3, 24, 32), dtype=torch.uint8)
        self.proprio = torch.randn(2, 23)
        self.tokens = torch.from_numpy(np.stack([self.tokenizer.encode(text) for text in self.instructions]))

    def test_configuration_is_explicit_and_legacy_defaults_to_last(self):
        legacy = asdict(self.config)
        legacy.pop("text_pooling")
        self.assertEqual(ModelConfig(**legacy).text_pooling, "last")
        for choice in ("last", "masked_mean"):
            self.assertEqual(replace(self.config, text_pooling=choice).text_pooling, choice)
        for invalid in (None, "mean", "last ", "", 0, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replace(self.config, text_pooling=invalid)

    def test_last_pooling_retains_exact_original_condition_and_legacy_actions(self):
        with torch.inference_mode():
            visual = self.model.vision(self.images.reshape(-1, 3, 24, 32).float() / 255.).reshape(2, 256)
            encoded, _ = self.model.text(self.model.word_embedding(self.tokens))
            lengths = (self.tokens != 0).sum(-1).clamp_min(1)
            language = encoded[torch.arange(2), lengths - 1]
            expected = self.model.condition(torch.cat([visual, self.model.proprio(self.proprio), language], -1))
            actual = self.model.encode_condition(self.images, self.proprio, self.tokens)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.pt"
            config = asdict(self.config)
            config.pop("text_pooling")
            torch.save(dict(schema=CHECKPOINT_SCHEMA, model_config=config, vocabulary=self.tokenizer.words,
                            normalization=dict(mean=[0.] * 23, std=[1.] * 23), action_contract=CONTRACT,
                            metadata={}, step=1, model_state=self.model.state_dict()), path)
            policy = DiffusionPolicy.load(path, CONTRACT, device="cpu")
            observation = dict(rgb={name: self.images[0, index].numpy().transpose(1, 2, 0)
                                    for index, name in enumerate(CONTRACT["cameras"])},
                               proprio=self.proprio[0].numpy(), instruction=self.instructions[0])
            # Compare identical public preprocessing/layouts, not a raw tensor
            # path whose convolution memory layout may round differently.
            explicit = DiffusionPolicy(self.model, self.tokenizer, dict(mean=[0.] * 23, std=[1.] * 23),
                                       CONTRACT, {}, device="cpu")
            explicit.reset(seed=47)
            actions = explicit.predict(observation)
            policy.reset(seed=47)
            np.testing.assert_array_equal(policy.predict(observation), actions)
            self.assertEqual(policy.config.text_pooling, "last")

    def test_pooling_changes_no_parameters_or_initialization_rng(self):
        torch.manual_seed(91)
        last = ActionDiffusion(self.config, len(self.tokenizer.words))
        torch.manual_seed(91)
        mean = ActionDiffusion(replace(self.config, text_pooling="masked_mean"), len(self.tokenizer.words))
        self.assertEqual(list(last.state_dict()), list(mean.state_dict()))
        for key in last.state_dict():
            torch.testing.assert_close(last.state_dict()[key], mean.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in last.parameters()), sum(p.numel() for p in mean.parameters()))

    def test_both_modes_ignore_trailing_padding_and_retain_word_order(self):
        for mode in ("last", "masked_mean"):
            model = ActionDiffusion(replace(self.config, text_pooling=mode), len(self.tokenizer.words))
            model.load_state_dict(self.model.state_dict())
            # Same multiset of words; the recurrent sequence carries their order.
            self.assertEqual(sorted(self.tokenizer.tokens(self.instructions[0])), sorted(self.tokenizer.tokens(self.instructions[1])))
            with torch.inference_mode():
                padded = model.encode_condition(self.images[:1].expand(2, -1, -1, -1, -1),
                                               self.proprio[:1].expand(2, -1), self.tokens)
                short = model.encode_condition(self.images[:1].expand(2, -1, -1, -1, -1),
                                              self.proprio[:1].expand(2, -1), self.tokens[:, :6])
                torch.testing.assert_close(padded, short, rtol=1e-6, atol=1e-7)
                self.assertGreater(float((padded[0] - padded[1]).abs().max()), 1e-5)

    def test_both_modes_backpropagate_through_language_vision_and_denoiser(self):
        for mode in ("last", "masked_mean"):
            model = ActionDiffusion(replace(self.config, text_pooling=mode), len(self.tokenizer.words))
            loss = model.training_loss(self.images, self.proprio, self.tokens, torch.zeros(2, 8, 8),
                                       torch.ones(2, 8, dtype=torch.bool), generator=torch.Generator().manual_seed(12))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for parameter in (model.word_embedding.weight, model.text.weight_hh_l0, model.vision[0].weight,
                              model.condition[0].weight, model.output.weight):
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0.)
            self.assertTrue(torch.equal(model.word_embedding.weight.grad[0], torch.zeros(32)))

    def test_warm_start_rejects_pooling_semantic_mismatch_but_accepts_legacy_last(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "parent.pt"
            config = asdict(self.config)
            config.pop("text_pooling")
            parent = dict(schema=CHECKPOINT_SCHEMA, model_config=config, vocabulary=self.tokenizer.words,
                          normalization=dict(mean=[0.] * 23, std=[1.] * 23, split="train", count=4),
                          action_contract=CONTRACT, step=1, model_state=self.model.state_dict(),
                          metadata=dict(training_episodes=[fixture_record(str(i), instruction=text)
                                                           for i, text in enumerate(self.instructions)], validation_episodes=[]))
            torch.save(parent, path)
            loaded, _, _, _ = train_policy.load_initialization(path, self.config, [fixture_record("new")], [], device="cpu")
            self.assertEqual(loaded.config.text_pooling, "last")
            with self.assertRaisesRegex(ValueError, "ModelConfig"):
                train_policy.load_initialization(path, replace(self.config, text_pooling="masked_mean"),
                                                 [fixture_record("new")], [], device="cpu")
            parent["model_config"]["text_pooling"] = "masked_mean"
            torch.save(parent, path)
            loaded, _, _, _ = train_policy.load_initialization(path, replace(self.config, text_pooling="masked_mean"),
                                                              [fixture_record("new")], [], device="cpu")
            self.assertEqual(loaded.config.text_pooling, "masked_mean")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is optional")
    def test_both_pooling_modes_support_deterministic_cuda_backward(self):
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            for mode in ("last", "masked_mean"):
                model = ActionDiffusion(replace(self.config, text_pooling=mode), len(self.tokenizer.words)).to("cuda")
                loss = model.training_loss(self.images.to("cuda"), self.proprio.to("cuda"), self.tokens.to("cuda"),
                                           torch.zeros(2, 8, 8, device="cuda"), torch.ones(2, 8, dtype=torch.bool, device="cuda"),
                                           generator=torch.Generator(device="cuda").manual_seed(12))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(model.text.weight_hh_l0.grad).all())
                self.assertGreater(float(model.text.weight_hh_l0.grad.abs().sum()), 0.)
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_cli_one_update_saves_opt_in_pooling_and_fresh_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "fit"
            argv = ["train_policy.py", "--train", "fixture.h5", "--output", str(output), "--device", "cpu",
                    "--steps", "1", "--batch-size", "4", "--threads", "1", "--image-size", "24", "32",
                    "--hidden-dim", "32", "--diffusion-steps", "8", "--inference-steps", "4", "--text-pooling", "masked_mean"]
            previous = torch.are_deterministic_algorithms_enabled()
            try:
                with patch.object(sys, "argv", argv), patch.object(train_policy, "expand_paths", return_value=["fixture.h5"]), \
                        patch.object(train_policy, "inspect_episodes", return_value=([fixture_record()], [])), \
                        patch.object(train_policy, "CompactChunkDataset", side_effect=TinyDataset):
                    train_policy.main()
            finally:
                torch.use_deterministic_algorithms(previous)
            saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(saved["model_config"]["text_pooling"], "masked_mean")
            self.assertEqual(saved["metadata"]["training"]["text_pooling"], "masked_mean")
            self.assertEqual(saved["metadata"]["initialization"]["kind"], "from_scratch")
            self.assertTrue(all(int(state["step"]) == 1 for state in saved["optimizer_state"]["state"].values()))


if __name__ == "__main__":
    unittest.main()
