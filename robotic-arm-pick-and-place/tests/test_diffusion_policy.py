"""Real forward/backward, masked targets, split isolation and policy loading."""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
try:
    import torch
except ImportError:
    raise unittest.SkipTest("Install optional requirements-learning.txt to test neural policies")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.data import CompactChunkDataset, check_data_separation, inspect_episodes
from learning.diffusion_policy import (ActionDiffusion, CHECKPOINT_SCHEMA, DiffusionPolicy,
                                      ModelConfig, Tokenizer, masked_noise_loss, masked_target_loss,
                                      preprocess_observation, validate_gripper_targets)
from utils.research_protocol import load_manifest, manifest_digest


CONTRACT = dict(id="forte-joint-reference-v2", shape=[8], arm_joint_count=7, policy_hz=20,
                cameras=["front", "wrist"], image_size_hw=[24, 32], privileged_fields_are_policy_inputs=False)


def observation(instruction="pick the red cube"):
    return dict(rgb={camera: np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
                     for camera in ("front", "wrist")}, proprio=np.zeros(23, np.float32), instruction=instruction)


def write_fixture(path, case, *, success=True, offset=0., source="fixture-source"):
    """Synthetic reader fixture only; never treated as a physical demonstration."""
    manifest = load_manifest()
    with h5py.File(path, "w") as f:
        f.attrs.update(metadata=json.dumps(dict(case, manifest_sha256=manifest_digest(manifest), policy="expert", custom_horizon=False)),
                       action_contract=json.dumps(CONTRACT), implementation=json.dumps(dict(source_sha256=source)),
                       complete=True, success=success, instruction="pick the red cube")
        f.create_dataset("observations/proprio", data=np.full((7, 23), offset, dtype=np.float32))
        for camera in CONTRACT["cameras"]:
            f.create_dataset(f"observations/rgb/{camera}", data=np.zeros((7, 24, 32, 3), dtype=np.uint8))
        f.create_dataset("observations/time_s", data=np.arange(7) * .05)
        f.create_dataset("privileged/integration_state", data=np.zeros((7, 1)))
        f.create_dataset("transitions/action", data=np.zeros((6, 8), dtype=np.float32))
        f.create_dataset("transitions/expert_action", data=np.full((6, 8), np.nan, dtype=np.float32))
        f.create_dataset("transitions/expert_action_valid", data=np.zeros(6, bool))
        f.create_dataset("transitions/source", data=["expert", "expert", "perturbed", "expert", "expert", "expert"], dtype=h5py.string_dtype())


class DiffusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(4)
        self.config = ModelConfig(image_height=24, image_width=32, hidden_dim=32, diffusion_steps=8, inference_steps=4)
        self.tokenizer = Tokenizer.fit(["pick the red cube", "pick the blue cube"])
        self.model = ActionDiffusion(self.config, len(self.tokenizer.words))

    def batch(self):
        image, proprio, tokens = preprocess_observation(observation(), self.config, self.tokenizer, CONTRACT)
        return (torch.from_numpy(np.stack([image] * 2)), torch.from_numpy(np.stack([proprio] * 2)),
                torch.from_numpy(np.stack([tokens] * 2)), torch.zeros(2, 8, 8),
                torch.tensor([[True, True, False, False, False, False, False, False]] * 2))

    def test_masked_targets_do_not_affect_loss_or_gradients(self):
        predicted = torch.zeros(1, 3, 8, requires_grad=True)
        target = torch.ones_like(predicted)
        mask = torch.tensor([[True, False, False]])
        loss = masked_noise_loss(predicted, target, mask)
        self.assertEqual(float(loss.detach()), 1.)
        loss.backward()
        self.assertTrue(torch.equal(predicted.grad[:, 1:], torch.zeros_like(predicted.grad[:, 1:])))
        for prediction_type in ("epsilon", "sample"):
            with self.subTest(prediction_type=prediction_type):
                model = ActionDiffusion(replace(self.config, prediction_type=prediction_type), len(self.tokenizer.words))
                batch = self.batch()
                loss1 = model.training_loss(*batch, generator=torch.Generator().manual_seed(33))
                loss1.backward()
                gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
                model.zero_grad(set_to_none=True)
                changed = list(batch)
                changed[3] = torch.where(batch[-1][..., None], batch[3], torch.full_like(batch[3], 1000))
                loss2 = model.training_loss(*changed, generator=torch.Generator().manual_seed(33))
                loss2.backward()
                torch.testing.assert_close(loss1, loss2, rtol=0, atol=0)
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(gradients[name], parameter.grad, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            masked_noise_loss(predicted, target, torch.zeros_like(mask))

    def test_real_backward_reaches_vision_language_and_denoiser(self):
        for prediction_type in ("epsilon", "sample"):
            with self.subTest(prediction_type=prediction_type):
                model = ActionDiffusion(replace(self.config, prediction_type=prediction_type), len(self.tokenizer.words))
                loss = model.training_loss(*self.batch())
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for parameter in (model.vision[0].weight, model.word_embedding.weight, model.output.weight):
                    self.assertIsNotNone(parameter.grad)
                    self.assertGreater(float(parameter.grad.abs().sum()), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is optional")
    def test_cuda_forward_backward_is_supported_with_determinism_enabled(self):
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            for prediction_type in ("epsilon", "sample"):
                with self.subTest(prediction_type=prediction_type):
                    model = ActionDiffusion(replace(self.config, prediction_type=prediction_type), len(self.tokenizer.words)).to("cuda")
                    loss = model.training_loss(*(value.to("cuda") for value in self.batch()),
                                               generator=torch.Generator(device="cuda").manual_seed(4))
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(torch.isfinite(model.vision[0].weight.grad).all())
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_text_order_unknowns_and_privileged_input_rejection(self):
        self.assertFalse(np.array_equal(self.tokenizer.encode("red blue"), self.tokenizer.encode("blue red")))
        self.assertEqual(self.tokenizer.encode("unseen")[0], 1)
        with self.assertRaises(ValueError):
            preprocess_observation(dict(observation(), target_pose=np.zeros(3)), self.config, self.tokenizer, CONTRACT)
        bad = observation()
        bad["rgb"]["front"] = np.zeros((12, 16, 3), np.uint8)
        with self.assertRaises(ValueError):
            preprocess_observation(bad, self.config, self.tokenizer, CONTRACT)

    def test_checkpoint_loading_contract_and_reset_reproducibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            for prediction_type, gripper_mode in (("epsilon", "continuous"), ("epsilon", "binary"),
                                                   ("sample", "continuous"), ("sample", "binary")):
                with self.subTest(prediction_type=prediction_type, gripper_mode=gripper_mode):
                    path = Path(temporary) / f"{prediction_type}_{gripper_mode}.pt"
                    config = replace(self.config, prediction_type=prediction_type, gripper_mode=gripper_mode)
                    torch.save(dict(schema=CHECKPOINT_SCHEMA, model_config=asdict(config), vocabulary=self.tokenizer.words,
                                    normalization=dict(mean=[0.] * 23, std=[1.] * 23), action_contract=CONTRACT,
                                    metadata=dict(training_source_sha256="fixture"), step=1, model_state=self.model.state_dict()), path)
                    policy = DiffusionPolicy.load(path, CONTRACT, device="cpu")
                    self.assertEqual(policy.config.prediction_type, prediction_type)
                    self.assertEqual(policy.config.gripper_mode, gripper_mode)
                    self.assertEqual(policy.metadata["inference_decoding"]["gripper_mode"], gripper_mode)
                    policy.reset(seed=7)
                    first = policy.predict(observation())
                    policy.predict(observation("pick the blue cube"))
                    policy.reset(seed=7)
                    np.testing.assert_array_equal(first, policy.predict(observation()))
                    self.assertEqual(first.shape, (8, 8))
                    self.assertEqual(first.dtype, np.float32)
                    self.assertTrue(np.isfinite(first).all() and (np.abs(first) <= 1).all())
                    with self.assertRaises(ValueError):
                        DiffusionPolicy.load(path, dict(CONTRACT, joint_reference_speed_rad_s=.5), device="cpu")

    def test_prediction_type_validation_and_legacy_loss_names(self):
        self.assertEqual(self.config.prediction_type, "epsilon")
        for invalid in (None, "v_prediction", "noise", "sample ", 1):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replace(self.config, prediction_type=invalid)
        torch.testing.assert_close(self.model.noise_loss(*self.batch(), generator=torch.Generator().manual_seed(7)),
                                   self.model.training_loss(*self.batch(), generator=torch.Generator().manual_seed(7)),
                                   rtol=0, atol=0)
        model = ActionDiffusion(replace(self.config, prediction_type="sample"), len(self.tokenizer.words))
        with self.assertRaisesRegex(ValueError, "clean actions"):
            model.noise_loss(*self.batch())
        with self.assertRaisesRegex(ValueError, "clean actions"):
            model.predict_noise(None, None, None)

    def test_training_objective_matches_declared_target_with_identical_corruption(self):
        images, proprio, tokens, actions, mask = self.batch()
        actions = torch.linspace(-1, 1, actions.numel()).reshape_as(actions)
        clean = torch.where(mask[..., None], actions, torch.zeros_like(actions))
        generator = torch.Generator().manual_seed(57)
        noise = torch.randn(actions.shape, generator=generator)
        timesteps = torch.randint(self.config.diffusion_steps, (len(actions),), generator=generator)
        alpha = self.model.alpha_bar[timesteps, None, None]
        expected_noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        for prediction_type, target in (("epsilon", noise), ("sample", clean)):
            with self.subTest(prediction_type=prediction_type):
                model = ActionDiffusion(replace(self.config, prediction_type=prediction_type), len(self.tokenizer.words))
                prediction = torch.full_like(actions, .25, requires_grad=True)
                with patch.object(model, "predict_target", return_value=prediction) as predict:
                    actual = model.training_loss(images, proprio, tokens, actions, mask,
                                                 generator=torch.Generator().manual_seed(57))
                torch.testing.assert_close(predict.call_args.args[0], expected_noisy, rtol=0, atol=0)
                torch.testing.assert_close(predict.call_args.args[1], timesteps, rtol=0, atol=0)
                expected = (prediction[mask] - target[mask]).square().mean()
                torch.testing.assert_close(actual, expected)
                actual.backward()
                torch.testing.assert_close(prediction.grad[~mask], torch.zeros_like(prediction.grad[~mask]), rtol=0, atol=0)

    def test_clean_action_ddim_uses_clipped_sample_and_consistent_noise(self):
        config = replace(self.config, prediction_type="sample", inference_steps=2)
        model = ActionDiffusion(config, len(self.tokenizer.words))
        images, proprio, tokens, actions, _ = self.batch()
        prediction = torch.linspace(-2, 2, actions.numel()).reshape_as(actions)
        clipped = prediction.clamp(-1, 1)
        initial = torch.randn(actions.shape, generator=torch.Generator().manual_seed(28))
        with patch.object(model, "predict_target", return_value=prediction) as predict:
            actual = model.sample(images, proprio, tokens, generator=torch.Generator().manual_seed(28))
        self.assertEqual(predict.call_count, 2)
        torch.testing.assert_close(predict.call_args_list[0].args[0], initial, rtol=0, atol=0)
        # Analytic deterministic DDIM transition from the final noise level to t=0.
        alpha_start, alpha_next = model.alpha_bar[-1], model.alpha_bar[0]
        expected_next = (alpha_next.sqrt() * clipped +
                         ((1 - alpha_next) / (1 - alpha_start)).sqrt() *
                         (initial - alpha_start.sqrt() * clipped))
        torch.testing.assert_close(predict.call_args_list[1].args[0], expected_next, rtol=1e-6, atol=2e-7)
        torch.testing.assert_close(actual, clipped, rtol=0, atol=0)

    def test_legacy_checkpoint_retains_original_epsilon_sampler(self):
        images, proprio, tokens, actions, _ = self.batch()
        # Preserve the original epsilon sampler arithmetic as a compatibility oracle.
        with torch.inference_mode():
            condition = self.model.encode_condition(images[:1], proprio[:1], tokens[:1])
            expected = torch.randn((1, 8, 8), generator=torch.Generator().manual_seed(7))
            schedule = torch.linspace(self.config.diffusion_steps - 1, 0, self.config.inference_steps).round().long().tolist()
            for index, step in enumerate(schedule):
                timestep = torch.full((1,), step, dtype=torch.long)
                alpha = self.model.alpha_bar[step]
                noise = self.model.predict_noise(expected, timestep, condition)
                clean = ((expected - (1 - alpha).sqrt() * noise) / alpha.sqrt()).clamp(-1, 1)
                if index + 1 == len(schedule):
                    expected = clean
                else:
                    noise = (expected - alpha.sqrt() * clean) / (1 - alpha).sqrt()
                    previous = self.model.alpha_bar[schedule[index + 1]]
                    expected = previous.sqrt() * clean + (1 - previous).sqrt() * noise
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.pt"
            legacy_config = asdict(self.config)
            legacy_config.pop("prediction_type")
            legacy_config.pop("gripper_mode")
            torch.save(dict(schema=CHECKPOINT_SCHEMA, model_config=legacy_config, vocabulary=self.tokenizer.words,
                            normalization=dict(mean=[0.] * 23, std=[1.] * 23), action_contract=CONTRACT,
                            metadata={}, step=1, model_state=self.model.state_dict()), path)
            loaded = DiffusionPolicy.load(path, CONTRACT, device="cpu")
            self.assertEqual(loaded.config.prediction_type, "epsilon")
            self.assertEqual(loaded.config.gripper_mode, "continuous")
            loaded.reset(seed=7)
            np.testing.assert_array_equal(loaded.predict(observation()), expected[0].numpy())

    def test_binary_gripper_changes_only_decoding_and_rejects_nonfinite_predictions(self):
        sampled = np.linspace(-.9, .9, 64, dtype=np.float32).reshape(8, 8)
        sampled[:, -1] = [-1., -.25, -1e-7, -0., 0., 1e-7, .25, 1.]
        for mode in ("continuous", "binary"):
            with self.subTest(gripper_mode=mode):
                model = ActionDiffusion(replace(self.config, gripper_mode=mode), len(self.tokenizer.words))
                policy = DiffusionPolicy(model, self.tokenizer, dict(mean=[0.] * 23, std=[1.] * 23),
                                         CONTRACT, {}, device="cpu")
                original = torch.from_numpy(sampled[None].copy())
                with patch.object(model, "sample", return_value=original):
                    output = policy.predict(observation())
                np.testing.assert_array_equal(original[0].numpy(), sampled)
                np.testing.assert_array_equal(output[:, :7], sampled[:, :7])
                if mode == "continuous":
                    np.testing.assert_array_equal(output, sampled)
                else:
                    np.testing.assert_array_equal(output[:, -1], [-1, -1, -1, 1, 1, 1, 1, 1])
                invalid = original.clone()
                invalid[0, 0, -1] = float("nan")
                with patch.object(model, "sample", return_value=invalid), self.assertRaisesRegex(ValueError, "nonfinite"):
                    policy.predict(observation())

    def test_binary_gripper_requires_endpoint_labels_and_ignores_masked_values(self):
        self.assertEqual(self.config.gripper_mode, "continuous")
        for invalid in (None, "auto", "task", "binary ", 1):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replace(self.config, gripper_mode=invalid)
        actions = np.zeros((2, 8, 8), dtype=np.float32)
        actions[..., -1] = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
        mask = np.ones((2, 8), dtype=bool)
        mask[:, -2:] = False
        actions[:, -2:, -1] = np.nan
        validate_gripper_targets(actions, mask, "binary")
        for width_m in (.034, .038):
            with self.subTest(book_nominal_jaw_width_m=width_m):
                changed = actions.copy()
                changed[0, 1, -1] = 2 * width_m / .074 - 1
                with self.assertRaisesRegex(ValueError, "intermediate apertures"):
                    validate_gripper_targets(changed, mask, "binary")
                validate_gripper_targets(changed, mask, "continuous")

    def test_trainer_rejects_intermediate_labels_in_either_split_before_subsampling(self):
        from scripts import train_policy
        manifest = load_manifest()
        train_case = next(c for c in manifest["cases"] if c["split"] == "train")
        val_case = next(c for c in manifest["cases"] if c["split"] == "validation")
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                train_path, val_path = Path(temporary) / "train.h5", Path(temporary) / "val.h5"
                output = Path(temporary) / "must_not_train"
                for bad_split, width_m in (("train", .034), ("validation", .038)):
                    with self.subTest(split=bad_split):
                        for path, case in ((train_path, train_case), (val_path, val_case)):
                            write_fixture(path, case)
                            with h5py.File(path, "r+") as episode:
                                episode["transitions/action"][:, -1] = 1.
                        bad_path = train_path if bad_split == "train" else val_path
                        with h5py.File(bad_path, "r+") as episode:
                            # This label is beyond the first valid chunk's fault boundary.
                            # --anchor-stride 999 would skip its anchor; it must still fail.
                            episode["transitions/action"][3, -1] = 2 * width_m / .074 - 1
                        argv = ["train_policy.py", "--train", str(train_path), "--validation", str(val_path),
                                "--gripper-mode", "binary", "--anchor-stride", "999", "--device", "cpu",
                                "--steps", "1", "--output", str(output)]
                        with patch.object(sys, "argv", argv), patch.object(train_policy, "CompactChunkDataset") as cache:
                            with self.assertRaisesRegex(ValueError, "Binary gripper requires"):
                                train_policy.main()
                            cache.assert_not_called()
                        self.assertFalse(output.exists())
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_frozen_split_cache_masks_and_train_only_statistics(self):
        manifest = load_manifest()
        train_case = next(c for c in manifest["cases"] if c["split"] == "train")
        val_case = next(c for c in manifest["cases"] if c["split"] == "validation")
        with tempfile.TemporaryDirectory() as temporary:
            train_path, val_path = Path(temporary) / "train.h5", Path(temporary) / "val.h5"
            write_fixture(train_path, train_case, offset=1.)
            write_fixture(val_path, val_case, offset=100.)
            train_records, _ = inspect_episodes([train_path], "train")
            validation_records, _ = inspect_episodes([val_path], "validation")
            check_data_separation(train_records, validation_records)
            with self.assertRaises(ValueError):
                inspect_episodes([val_path], "train")
            with self.assertRaises(ValueError):
                inspect_episodes([train_path], "test")
            train = CompactChunkDataset(train_records, self.config, self.tokenizer, split="train")
            validation = CompactChunkDataset(validation_records, self.config, self.tokenizer, split="validation")
            normalization = train.normalization()
            np.testing.assert_array_equal(normalization["mean"], np.ones(23))
            self.assertEqual(len(train), 5)
            np.testing.assert_array_equal(train.masks[0], [True, True, False, False, False, False, False, False])
            np.testing.assert_array_equal(train.masks[1], [True, False, False, False, False, False, False, False])
            np.testing.assert_array_equal(train.masks[2], [True, True, True, False, False, False, False, False])
            train.set_normalization(normalization)
            validation.set_normalization(normalization)
            with self.assertRaises(ValueError):
                validation.normalization()
            with self.assertRaises(ValueError):
                check_data_separation(train_records, train_records)
            self.assertEqual(train[0][0].dtype, torch.uint8)

    def test_diagnostic_and_errored_recordings_are_excluded(self):
        cases = [c for c in load_manifest()["cases"] if c["split"] == "train"]
        with tempfile.TemporaryDirectory() as temporary:
            good, excluded = Path(temporary) / "good.h5", Path(temporary) / "excluded.h5"
            write_fixture(good, cases[0])
            for field, value, reason in (("custom_horizon", True, "custom_horizon_diagnostic"),
                                         ("instruction_override", "place elsewhere", "instruction_override_diagnostic"),
                                         ("policy", "checkpoint", "non_expert_collector"),
                                         ("collector_error", "failure after score", "collector_error")):
                with self.subTest(field=field):
                    write_fixture(excluded, cases[1])
                    with h5py.File(excluded, "r+") as f:
                        if field == "collector_error":
                            f.attrs[field] = value
                        else:
                            meta = json.loads(f.attrs["metadata"])
                            meta[field] = value
                            f.attrs["metadata"] = json.dumps(meta)
                    records, skipped = inspect_episodes([good, excluded], "train")
                    self.assertEqual(len(records), 1)
                    self.assertEqual(skipped[0]["reason"], reason)


if __name__ == "__main__":
    unittest.main()
