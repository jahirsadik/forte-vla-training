"""Checkpoint evaluation lifecycle, policy isolation and inference timing."""

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_research as runner
from utils.policy import PolicySession


class FakeEnv:
    instances = []

    def __init__(self, **constructor):
        self.constructor = constructor
        self.horizon = constructor.get("horizon_steps", 4)
        self.control_dt = .05
        self.action_space = gym.spaces.Box(-1, 1, (8,), dtype=np.float32)
        self.action_contract = {"id": "test-contract", "shape": [8]}
        self.fingerprint, self.implementation = "model", {"source_sha256": "source"}
        self.closed = False
        self.steps = 0
        self.actions = []
        self.instances.append(self)

    def _observe(self):
        observation = dict(proprio=np.zeros(23, dtype=np.float32), instruction="Put the red cube in the blue tray.")
        if self.constructor["observation_mode"] == "rgb":
            observation["rgb"] = {name: np.zeros((8, 8, 3), dtype=np.uint8) for name in ("front", "wrist")}
        return observation

    def info(self):
        return dict(is_success=False, episode=dict(steps=self.steps, done=self.steps >= self.horizon),
                    metrics=dict(timeout=self.steps >= self.horizon))

    def reset(self, seed, options):
        self.seed, self.options = seed, options
        self.steps = 0
        return self._observe(), self.info()

    def step(self, action):
        self.actions.append(action.copy())
        self.steps += 1
        return self._observe(), 0., False, self.steps >= self.horizon, self.info()

    def close(self):
        self.closed = True


class FakePolicy:
    def __init__(self):
        self.metadata = dict(checkpoint_sha256="checkpoint", model_config=dict(horizon=3))
        self.observations = []
        self.sync_calls = 0
        self.resets = []

    def reset(self, seed=None):
        self.resets.append(seed)

    def synchronize(self):
        self.sync_calls += 1

    def predict(self, observation):
        self.observations.append(observation)
        return np.zeros((3, 8), dtype=np.float32)


class FakeWriter:
    instances = []
    fail_finish = False

    def __init__(self, path, env, obs, info, *, metadata):
        self.path, self.metadata = path, metadata
        self.sources = []
        self.finished = self.closed = False
        self.instances.append(self)

    def append(self, *args, **kwargs):
        self.sources.append(kwargs["source"])

    def finish(self, *args, **kwargs):
        self.finished = True
        if self.fail_finish:
            raise OSError("recording flush failed")

    def close(self):
        self.closed = True


class PolicyEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.case = dict(case_id="case", environment="sorting", split="validation", axis="development",
                         task_id="red_cube_to_blue", seed=1000, layout="shuffled", asset_set="validation",
                         appearance="nominal", appearance_seed=91000, recovery="none", clutter_level=1)
        self.manifest = dict(version="fixture", cases=[self.case])
        self.policy = FakePolicy()
        FakeEnv.instances.clear()
        FakeWriter.instances.clear()
        FakeWriter.fail_finish = False
        self.patches = [patch.dict(runner.ENVIRONMENTS, sorting=FakeEnv),
                        patch.object(runner, "AppearancePerturbation"),
                        patch.object(runner, "RecoveryFault", return_value=types.SimpleNamespace(events=[], apply=lambda action: (action, False))),
                        patch.object(runner, "make_policy", return_value=self.policy)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_checkpoint_enables_rgb_resets_policy_and_times_only_predictions(self):
        result = runner.run_case(self.case, self.manifest, checkpoint="pilot.pt", execute_actions=2)
        self.assertNotIn("error", result)
        self.assertEqual(self.policy.resets, [1000])
        self.assertEqual(len(self.policy.observations), 2)
        self.assertEqual(self.policy.sync_calls, 4)
        self.assertEqual(set(self.policy.observations[0]), {"rgb", "proprio", "instruction"})
        self.assertEqual(result["constructor"]["observation_mode"], "rgb")
        self.assertEqual(result["prediction_latency"]["count"], 2)
        self.assertEqual(result["prediction_latency"]["control_step_budget_ms"], 50)
        self.assertEqual(result["policy_metadata"]["checkpoint_sha256"], "checkpoint")
        self.assertNotIn("expert", result)
        self.assertFalse(result["custom_horizon"])
        self.assertTrue(FakeEnv.instances[0].closed)

    def test_policy_seed_changes_only_policy_randomness_and_is_recorded(self):
        with patch.object(runner, "EpisodeWriter", FakeWriter), patch.object(runner, "validate_episode", return_value={}):
            result = runner.run_case(self.case, self.manifest, checkpoint="pilot.pt",
                                     dataset_dir=Path("unused"), policy_seed=7)
        self.assertNotIn("error", result)
        self.assertEqual(self.policy.resets, [7])
        self.assertEqual(FakeEnv.instances[0].seed, 1000)
        self.assertEqual(result["seed"], 1000)
        self.assertEqual(result["policy_seed"], 7)
        self.assertEqual(FakeWriter.instances[0].metadata["policy_seed"], 7)
        self.assertEqual(self.case["seed"], 1000)
        for invalid in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                runner.run_case(self.case, self.manifest, policy_seed=invalid)

    def test_issued_action_feedback_precedes_fault_and_never_exposes_modified_action(self):
        issued = []
        self.policy.observe_action = lambda action: issued.append(action.copy())

        def inject_fault(proposed):
            # The callback must already have fired for this control step.
            self.assertEqual(len(issued), FakeEnv.instances[0].steps + 1)
            np.testing.assert_array_equal(issued[-1], proposed)
            actual = proposed.copy()
            actual[-1] = -1
            return actual, True

        fault = types.SimpleNamespace(events=[], apply=inject_fault)
        with patch.object(runner, "RecoveryFault", return_value=fault):
            result = runner.run_case(self.case, self.manifest, checkpoint="pilot.pt", execute_actions=2)
        self.assertNotIn("error", result)
        self.assertEqual(len(issued), 4)
        self.assertEqual(len(self.policy.observations), 2)
        self.assertEqual(result["prediction_latency"]["count"], 2)
        np.testing.assert_array_equal(issued, np.zeros((4, 8), dtype=np.float32))
        actual = np.asarray(FakeEnv.instances[0].actions)
        np.testing.assert_array_equal(actual[:, -1], -np.ones(4))
        np.testing.assert_array_equal(actual[:, :-1], np.asarray(issued)[:, :-1])

    def test_neural_recordings_never_receive_expert_labels_and_cleanup_on_flush_failure(self):
        FakeWriter.fail_finish = True
        with tempfile.TemporaryDirectory() as folder, patch.object(runner, "EpisodeWriter", FakeWriter):
            result = runner.run_case(self.case, self.manifest, checkpoint="pilot.pt", dataset_dir=folder)
        writer = FakeWriter.instances[0]
        self.assertEqual(writer.sources, ["policy"] * 4)
        self.assertEqual(writer.metadata["policy"], "diffusion_checkpoint")
        self.assertEqual(writer.metadata["policy_metadata"], self.policy.metadata)
        self.assertTrue(writer.closed)
        self.assertTrue(FakeEnv.instances[0].closed)
        self.assertIn("recording flush failed", result["error"])

    def test_bad_checkpoint_contract_closes_environment_before_recording(self):
        with patch.object(runner, "make_policy", side_effect=ValueError("Action contract mismatch")), \
             patch.object(runner, "EpisodeWriter", FakeWriter):
            result = runner.run_case(self.case, self.manifest, checkpoint="bad.pt", dataset_dir="unused")
        self.assertIn("Action contract mismatch", result["error"])
        self.assertFalse(result["is_success"])
        self.assertTrue(FakeEnv.instances[0].closed)
        self.assertEqual(FakeWriter.instances, [])

    def test_prediction_failure_is_recorded_without_false_success(self):
        with patch.object(self.policy, "predict", side_effect=RuntimeError("Inference failed")):
            result = runner.run_case(self.case, self.manifest, checkpoint="pilot.pt")
        self.assertIn("Inference failed", result["error"])
        self.assertFalse(result["is_success"])
        self.assertEqual(result["prediction_latency"]["failed_predictions"], 1)
        self.assertEqual(result["prediction_latency"]["count"], 0)
        self.assertTrue(FakeEnv.instances[0].closed)

    def test_instruction_override_changes_only_policy_language_and_is_explicit(self):
        override = "Put the red cube in the green tray."
        result = runner.run_case(self.case, self.manifest, checkpoint="pilot.pt", instruction_override=override)
        self.assertTrue(result["counterfactual_diagnostic"])
        self.assertEqual(result["effective_instruction"], override)
        self.assertNotEqual(result["instruction"], override)
        self.assertEqual(FakeEnv.instances[0].options["task_id"], self.case["task_id"])
        self.assertTrue(all(obs["instruction"] == override for obs in self.policy.observations))
        with self.assertRaisesRegex(ValueError, "cannot record"):
            runner.run_case(self.case, self.manifest, checkpoint="pilot.pt", instruction_override=override, dataset_dir="unused")

    def test_latency_quantiles_and_deadline_fraction(self):
        timed = runner.TimedPolicy(self.policy)
        timed.latencies = [.01, .03, .06, .10]
        summary = timed.summary(.05, 1)
        self.assertAlmostEqual(summary["p50_ms"], 45)
        self.assertAlmostEqual(summary["p95_ms"], 94)
        self.assertEqual(summary["fraction_within_control_step_budget"], .5)
        self.assertEqual(summary["control_step_overruns"], 2)


class CheckpointBoundaryTests(unittest.TestCase):
    def test_loader_receives_contract_copy_without_environment(self):
        module = types.ModuleType("learning.diffusion_policy")
        received = {}

        def load_policy(path, action_contract, *, device):
            received.update(path=path, contract=action_contract, device=device)
            return "loaded"

        module.load_policy = load_policy
        env = types.SimpleNamespace(action_contract={"id": "contract", "shape": [8]})
        with patch.dict(sys.modules, {"learning.diffusion_policy": module}):
            result = runner.make_policy("expert", env, 4, checkpoint="model.pt", device="cpu")
        self.assertEqual(result, "loaded")
        self.assertEqual(received, dict(path="model.pt", contract=env.action_contract, device="cpu"))
        self.assertIsNot(received["contract"], env.action_contract)
        self.assertIsNot(received["contract"]["shape"], env.action_contract["shape"])


class CommandFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.action_space = gym.spaces.Box(-1, 1, (8,), dtype=np.float32)

    def test_cached_steps_report_only_issued_actions_and_callback_cannot_change_them(self):
        class Policy:
            def __init__(self):
                self.predictions = 0
                self.issued = []

            def predict(self, observation):
                self.predictions += 1
                return np.asarray([[.1] * 8, [.2] * 8, [.9] * 8], dtype=np.float32)

            def observe_action(self, action):
                self.issued.append(action.copy())
                action[:] = -1  # Must affect neither the returned nor cached commands.

        policy = Policy()
        session = PolicySession(policy, self.action_space, execute_actions=2)
        first = session({})
        self.assertEqual(policy.predictions, 1)
        self.assertEqual(len(policy.issued), 1)
        np.testing.assert_array_equal(first, np.full(8, .1, dtype=np.float32))
        second = session({})
        self.assertEqual(policy.predictions, 1)
        self.assertEqual(len(policy.issued), 2)
        np.testing.assert_array_equal(second, np.full(8, .2, dtype=np.float32))
        third = session({})
        self.assertEqual(policy.predictions, 2)
        self.assertEqual(len(policy.issued), 3)
        np.testing.assert_array_equal(third, first)
        np.testing.assert_array_equal(policy.issued, np.stack([first, second, third]))
        self.assertFalse(any(np.any(action == np.float32(.9)) for action in policy.issued))

    def test_reset_discards_pending_commands_and_policy_feedback_memory(self):
        class Policy:
            def __init__(self):
                self.resets = []
                self.issued = []

            def reset(self, seed=None):
                self.resets.append(seed)
                self.issued.clear()

            def predict(self, observation):
                value = .1 if len(self.resets) == 1 else .3
                return np.asarray([[value] * 8, [value + .1] * 8], dtype=np.float32)

            def observe_action(self, action):
                self.issued.append(action.copy())

        policy = Policy()
        session = PolicySession(policy, self.action_space, execute_actions=2)
        session.reset(seed=4)
        np.testing.assert_array_equal(session({}), np.full(8, .1, dtype=np.float32))
        self.assertEqual(len(session.pending), 1)
        session.reset(seed=9)
        self.assertEqual(policy.resets, [4, 9])
        self.assertEqual(policy.issued, [])
        self.assertEqual(len(session.pending), 0)
        np.testing.assert_array_equal(session({}), np.full(8, .3, dtype=np.float32))
        self.assertEqual(len(policy.issued), 1)

    def test_legacy_callable_without_feedback_keeps_existing_behavior(self):
        expected = np.arange(8, dtype=np.float32) / 8
        session = PolicySession(lambda observation: expected, self.action_space)
        session.reset(seed=3)
        np.testing.assert_array_equal(session({}), expected)


if __name__ == "__main__":
    unittest.main()
