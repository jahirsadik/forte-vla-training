"""Regression checks for physical labels and policy/data boundaries."""

from pathlib import Path
import sys
import unittest
import tempfile

import gymnasium as gym
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environments.sorting import SortingEnv
from utils.policy import PolicySession
from utils.research_protocol import load_manifest, manifest_digest, validate_manifest, verify_metadata
from utils.demonstrations import EpisodeWriter, validate_episode, training_samples, replay_episode
from environments.perturbations import AppearancePerturbation, RecoveryFault
from environments.bookshelf_task import BookshelfEnv


class ScoringAndPolicyTests(unittest.TestCase):
    def test_wedge_inclined_face_is_valid_supported_placement(self):
        with SortingEnv(horizon_steps=100) as env:
            env.reset(seed=0, options={"task_id": "yellow_wedge_to_blue"})
            angle = np.pi - np.arctan(0.045 / 0.060)
            quat = [np.cos(angle / 2), 0, np.sin(angle / 2), 0]
            position = np.asarray(env.goal["position"]).copy()
            position[2] += 0.06
            env._set_object_pose(env.target, position, quat)
            mujoco.mj_forward(env.model, env.data)
            self.assertFalse(env._measure()["supported"])
            # Deliberate scoring fixture; this is not an expert demonstration.
            env._episode["ever_grasped"] = env._episode["ever_lifted"] = True
            for _ in range(80):
                _, _, terminated, _, info = env.step(np.r_[np.zeros(7), 1])
                if terminated:
                    break
            self.assertTrue(info["is_success"], info)
            self.assertTrue(info["metrics"]["supported"])
            self.assertLess(abs(env._bounds(env.target)[0][2] - env.goal["surface_z"]), .001)

    def test_policy_resets_memory_and_discards_old_chunks(self):
        class Stateful:
            def __init__(self):
                self.episodes = 0
                self.calls = 0

            def reset(self, seed=None):
                self.episodes += 1
                self.calls = 0
                self.seed = seed

            def __call__(self, obs):
                self.calls += 1
                return np.array([[self.calls / 10, 1], [.9, 1]])

        policy = Stateful()
        session = PolicySession(policy, gym.spaces.Box(-1, 1, (2,), dtype=np.float32))
        for seed in range(10):
            session.reset(seed=seed)
            np.testing.assert_allclose(session({}), [.1, 1])
            self.assertEqual(policy.seed, seed)
        self.assertEqual(policy.episodes, 10)
        np.testing.assert_allclose(session({}), [.9, 1])
        self.assertEqual(policy.calls, 1)


class DataProtocolTests(unittest.TestCase):
    def test_frozen_membership_and_heldouts(self):
        manifest = load_manifest()
        validate_manifest(manifest)
        for case in manifest["cases"]:
            verify_metadata(dict(case, manifest_sha256=manifest_digest(manifest)), manifest)
        bad = dict(manifest["cases"][0], manifest_sha256=manifest_digest(manifest))
        bad["split"] = "test"
        with self.assertRaises(ValueError):
            verify_metadata(bad, manifest)
        self.assertEqual({c["axis"] for c in manifest["cases"] if c["split"]=="test"},
                         {"iid","objects","layouts","combinations","camera","lighting","missed_grasp","delayed_release"})

    def test_appearance_changes_images_without_changing_physics(self):
        with SortingEnv(observation_mode="rgb", image_size=(60,80)) as env:
            obs, _ = env.reset(seed=0)
            physical = env._physics_state().copy()
            perturb = AppearancePerturbation(env)
            for variant in ("camera_shift", "lighting_dim"):
                perturb.apply(variant, 44)
                np.testing.assert_array_equal(physical, env._physics_state())
                self.assertFalse(np.array_equal(obs["rgb"]["front"], env._observe()["rgb"]["front"]))
            perturb.apply()
            np.testing.assert_array_equal(obs["rgb"]["front"],env._observe()["rgb"]["front"])

    def test_synchronized_failure_data_and_policy_only_loader(self):
        manifest = load_manifest()
        case = next(c for c in manifest["cases"] if c["environment"]=="sorting" and c["split"]=="train")
        with tempfile.TemporaryDirectory() as directory, SortingEnv(observation_mode="rgb", image_size=(60,80), horizon_steps=3) as env:
            obs, info = env.reset(seed=case["seed"], options={"task_id":case["task_id"],"layout":case["layout"]})
            path = Path(directory)/"episode.h5"
            meta = dict(case, manifest_sha256=manifest_digest(manifest),
                        constructor=dict(observation_mode="rgb", image_size=[60,80], horizon_steps=3))
            writer = EpisodeWriter(path,env,obs,info,metadata=meta)
            action = np.r_[np.zeros(7),1].astype(np.float32)
            for t in range(3):
                obs,r,end,trunc,info = env.step(action)
                writer.append(action,obs,r,end,trunc,info,source="policy",
                              expert_action=action if t==1 else None, intervention=t==1)
            writer.finish(info)
            self.assertEqual(validate_episode(path)["transitions"],3)
            replay_episode(path)
            self.assertEqual(list(training_samples([path])), [])
            samples = list(training_samples([path], include_failures=True))
            self.assertEqual(len(samples),1)
            self.assertEqual(set(samples[0]),{"observation","action"})
            self.assertEqual(set(samples[0]["observation"]),{"proprio","rgb","instruction"})
            self.assertEqual(list(training_samples([path], split="test", include_failures=True)),[])

    def test_action_fault_is_bounded_one_shot_and_does_not_mutate_state(self):
        with SortingEnv() as env:
            env.reset(seed=0)
            state = env._physics_state().copy()
            fault = RecoveryFault(env,"missed_grasp")
            action = np.r_[np.zeros(7),-1].astype(np.float32)
            executed = [fault.apply(action) for _ in range(35)]
            self.assertEqual(sum(active for _,active in executed),30)
            self.assertEqual(len(fault.events),1)
            np.testing.assert_array_equal(state,env._physics_state())
            np.testing.assert_array_equal(executed[-1][0],action)

    def test_same_book_supports_counterfactual_destination(self):
        with BookshelfEnv() as env:
            first,_=env.reset(seed=0,options={"task_id":"extract_blue","layout":"shuffled"})
            physics = env._physics_state().copy()
            other,_=env.reset(seed=0,options={"task_id":"relocate_blue","layout":"shuffled"})
            np.testing.assert_array_equal(physics,env._physics_state())
            self.assertNotEqual(first["instruction"],other["instruction"])

    def test_each_relocation_has_a_valid_upright_goal(self):
        with BookshelfEnv() as env:
            for task in env.task_ids:
                if not task.startswith("relocate_"):
                    continue
                with self.subTest(task=task):
                    env.reset(seed=0,options={"task_id":task})
                    # Explicit scorer fixture, never a demonstration.
                    env._set_object_pose(env.target,env.goal["position"])
                    mujoco.mj_forward(env.model,env.data)
                    env._episode["ever_grasped"]=env._episode["ever_lifted"]=True
                    for _ in range(30):
                        _,_,done,_,info=env.step(np.r_[np.zeros(7),1])
                        if done:break
                    self.assertTrue(info["is_success"],info)


if __name__ == "__main__":
    unittest.main()
