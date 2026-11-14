"""Collector boundary tests; physical takeover is tested with the real expert."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_corrections import collect_one, descriptor, read_plan, make_correction_expert
from learning.data import intervention_rollout_id
from utils.research_protocol import load_manifest


class CorrectionCollectionTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest()
        self.case = next(c for c in self.manifest["cases"] if c["case_id"] == "0fc510ebb88105fe8404")
        self.plan = dict(version="forte-release-correction-pilot-v1", case_id=self.case["case_id"],
                         execution_horizon_steps=8, confirmation_steps=2,
                         rollouts=[dict(policy_seed=42, minimum_policy_steps=2)],
                         collection_plan_sha256="a" * 64)

    def test_plan_rejects_heldout_or_duplicate_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(self.plan))
            self.assertEqual(read_plan(path)[1]["case_id"], self.case["case_id"])
            changed = deepcopy(self.plan)
            changed["rollouts"] *= 2
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_plan(path)
            for split in ("validation", "test"):
                changed = deepcopy(self.plan)
                changed["case_id"] = next(c["case_id"] for c in self.manifest["cases"] if c["split"] == split)
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "train"):
                    read_plan(path)

    def test_v2_expert_selection_is_explicit_and_binds_rollout_identity(self):
        original = descriptor(self.plan, self.plan["rollouts"][0], "b" * 64)
        plan = deepcopy(self.plan)
        plan["version"] = "forte-correction-pilot-v2"
        plan["rollouts"] = [dict(plan["rollouts"][0], expert=name) for name in
                            ("supported_goal_release_v1", "grasped_carry_recovery_v1")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            validated, _, _ = read_plan(path)
            release, carry = [descriptor(validated, rollout, "b" * 64) for rollout in validated["rollouts"]]
            self.assertEqual(release, original)
            self.assertEqual(carry["version"], "forte-carry-intervention-v1")
            self.assertNotEqual(intervention_rollout_id(self.case["case_id"], release),
                                intervention_rollout_id(self.case["case_id"], carry))
            for malformed in (dict(plan["rollouts"][0], expert="unknown"),
                              {k: v for k, v in plan["rollouts"][0].items() if k != "expert"}):
                path.write_text(json.dumps(dict(plan, rollouts=[malformed])))
                with self.assertRaises(ValueError):
                    read_plan(path)
            path.write_text(json.dumps(dict(self.plan, rollouts=plan["rollouts"])))
            with self.assertRaises(ValueError):
                read_plan(path)

    def test_factory_uses_declared_expert_without_defaulting_unknown_values(self):
        env = object()
        with patch("controllers.placement_correction.PlacementReleaseExpert") as release:
            self.assertIs(make_correction_expert("supported_goal_release_v1", env), release.return_value)
            release.assert_called_once_with(env)
        carry = lambda value: ("carry", value)
        with patch.dict(sys.modules, {"controllers.carry_correction": SimpleNamespace(CarryRecoveryExpert=carry)}):
            self.assertEqual(make_correction_expert("grasped_carry_recovery_v1", env), ("carry", env))
        with self.assertRaises(ValueError):
            make_correction_expert("unknown", env)

    def run_fake(self, *, late=False, fail_close=False, teacher_error=False, carry=False, gate=False):
        traces = dict(steps=[], resets=[], predictions=0, labels=[], events=[], experts=[], observations=[])
        class Environment:
            def __init__(self, **kwargs):
                self.index = 0
                self.horizon = 8
                self.action_space = gym.spaces.Box(-1., 1., shape=(8,), dtype=np.float32)
                self.action_contract = {"public": True}
            def _observe(self):
                return dict(proprio=np.full(23, self.index, np.float32), instruction="move cube",
                            rgb={c: np.zeros((4, 4, 3), np.uint8) for c in ("front", "wrist")})
            def info(self):
                return dict(is_success=self.index == 6, episode=dict(done=self.index == 6, steps=self.index), metrics={})
            def reset(self, **kwargs): return self._observe(), self.info()
            def step(self, action):
                traces["steps"].append(action.copy()); self.index += 1
                return self._observe(), 0., self.index == 6, False, self.info()
            def close(self):
                if fail_close: raise RuntimeError("close fixture")
        class Learner:
            metadata = {"fixture": True}
            def reset(self, seed=None): traces["resets"].append(seed)
            def predict(self, observation):
                self.assert_public(observation)
                traces["predictions"] += 1
                return np.tile(np.r_[np.full(7, .1), -1].astype(np.float32), (8, 1))
            @staticmethod
            def assert_public(observation):
                if set(observation) != {"rgb", "proprio", "instruction"}: raise ValueError("private input")
        class Teacher:
            def __init__(self, env): self.env, self.error, self.stage = env, None, "waiting"
            def observe(self, info):
                traces["observations"].append(info["episode"]["steps"])
                info["episode"]["steps"] = -999  # Collector passes an isolated copy.
            def should_intervene(self): return not gate or self.env.index >= 5
            def eligible(self): return self.env.index >= 3
            def reset(self, seed=None):
                if not self.eligible(): raise ValueError("ineligible")
                self.stage = "withdraw"
            def predict(self, observation):
                if teacher_error: self.error = "teacher fixture failure"
                return np.r_[np.zeros(7), 1].astype(np.float32)
        class Writer:
            def __init__(self, *args, **kwargs): self.file = SimpleNamespace(attrs={})
            def append(self, action, *args, **kwargs): traces["labels"].append((action.copy(), kwargs))
            def finish(self, info, *, events, **kwargs): traces["events"] = events
            def close(self): pass
        plan = deepcopy(self.plan)
        if carry:
            plan["version"] = "forte-correction-pilot-v2"
            plan["rollouts"][0]["expert"] = "grasped_carry_recovery_v1"
        if late:
            plan["confirmation_steps"] = 3
            plan["rollouts"][0]["minimum_policy_steps"] = 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            checkpoint = path / "checkpoint.pt"; checkpoint.write_bytes(b"fixture")
            def factory(expert, env):
                traces["experts"].append(expert)
                return Teacher(env)
            with patch("environments.sorting.SortingEnv", Environment), \
                 patch("scripts.collect_corrections.make_correction_expert", side_effect=factory), \
                 patch("environments.perturbations.AppearancePerturbation") as appearance, \
                 patch("scripts.collect_corrections.load_policy", return_value=Learner()), \
                 patch("scripts.collect_corrections.EpisodeWriter", Writer), \
                 patch("scripts.collect_corrections.validate_episode", return_value={}), \
                 patch("scripts.collect_corrections.implementation_provenance", return_value={}):
                result = collect_one(plan, self.case, self.manifest, checkpoint, path, 0, "cpu")
        return result, traces

    def test_v2_carry_takeover_keeps_one_expert_suffix_and_declared_provenance(self):
        result, traces = self.run_fake(carry=True)
        self.assertNotIn("error", result)
        self.assertTrue(result["is_success"] and result["assisted"])
        self.assertEqual(traces["experts"], ["grasped_carry_recovery_v1"])
        self.assertEqual(result["metadata"]["intervention"]["version"], "forte-carry-intervention-v1")
        self.assertEqual(result["metadata"]["intervention"]["actual_takeover_step"], 4)
        self.assertEqual(len(traces["events"]), 1)
        self.assertEqual([labels["source"] for _, labels in traces["labels"]],
                         ["policy"] * 4 + ["expert"] * 2)
        self.assertEqual(traces["observations"], list(range(7)))
        self.assertEqual(result["info"]["episode"]["steps"], 6)

    def test_history_gate_is_observed_from_reset_before_minimum_prefix(self):
        result, traces = self.run_fake(carry=True, gate=True)
        self.assertTrue(result["is_success"])
        self.assertFalse(result["assisted"])
        self.assertEqual(traces["observations"], list(range(7)))
        self.assertEqual(traces["events"], [])

    def test_takeover_discards_pending_actions_and_labels_only_actual_expert_suffix(self):
        result, traces = self.run_fake()
        self.assertNotIn("error", result)
        self.assertTrue(result["is_success"] and result["assisted"])
        self.assertEqual(result["metadata"]["intervention"]["actual_takeover_step"], 4)
        self.assertEqual(traces["predictions"], 1)
        self.assertEqual(traces["resets"], [42])
        self.assertEqual(traces["events"][0]["discarded_learner_actions"], 4)
        for index, (action, labels) in enumerate(traces["labels"]):
            np.testing.assert_array_equal(action, labels["proposed_action"])
            self.assertEqual(labels["intervention"], index >= 4)
            self.assertEqual(labels["source"], "expert" if index >= 4 else "policy")
            if index < 4: self.assertIsNone(labels["expert_action"])
            else: np.testing.assert_array_equal(action, labels["expert_action"])

    def test_autonomous_success_without_takeover_remains_unlabelled(self):
        result, traces = self.run_fake(late=True)
        self.assertTrue(result["is_success"])
        self.assertFalse(result["assisted"])
        self.assertIsNone(result["metadata"]["intervention"]["actual_takeover_step"])
        self.assertFalse(traces["events"])
        self.assertTrue(all(labels["expert_action"] is None for _, labels in traces["labels"]))

    def test_close_error_does_not_discard_recorded_outcome(self):
        result, traces = self.run_fake(fail_close=True)
        self.assertTrue(result["is_success"])
        self.assertIn("close fixture", result["error"])
        self.assertEqual(len(traces["labels"]), 6)

    def test_failed_first_expert_command_does_not_claim_executed_takeover(self):
        result, traces = self.run_fake(teacher_error=True)
        self.assertIn("teacher fixture failure", result["error"])
        self.assertFalse(result["assisted"])
        self.assertIsNone(result["metadata"]["intervention"]["actual_takeover_step"])
        self.assertEqual(len(traces["steps"]), 4)
        self.assertEqual(len(traces["labels"]), 4)
        self.assertEqual(traces["events"], [])


if __name__ == "__main__":
    unittest.main()
