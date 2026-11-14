"""Synchronized HDF5 episodes; privileged state is explicitly separated.

obs[t] precedes action[t]; obs[t+1] is its result. All outcomes are retained.
The default training reader admits only successful expert actions from train.
"""

import json
from pathlib import Path

import h5py
import numpy as np

from utils.research_protocol import load_manifest, verify_metadata, manifest_path


class EpisodeWriter:
    def __init__(self, path, env, observation, info, *, metadata, overwrite=False):
        if "rgb" not in observation:
            raise ValueError("Demonstration collection requires RGB observations")
        self.env = env
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "w" if overwrite else "x")
        self.file.attrs["schema"] = "forte-demonstrations-v1"
        # Replay must never rely on mutable camera/resolution defaults.
        metadata = dict(metadata, constructor=dict(metadata.get("constructor", {}),
                        image_size=list(env.image_size), cameras=list(env.cameras)))
        self.file.attrs["metadata"] = json.dumps(metadata, sort_keys=True)
        self.file.attrs["action_contract"] = json.dumps(env.action_contract, sort_keys=True)
        self.file.attrs["implementation"] = json.dumps(env.implementation, sort_keys=True)
        self.file.attrs["model_fingerprint"] = env.fingerprint
        self.file.attrs["instruction"] = observation["instruction"]
        self.file.attrs["complete"] = False
        self.file.create_dataset("privileged/initial_checkpoint_json", data=json.dumps(env.get_state()), dtype=h5py.string_dtype())
        self.count = 0
        self._observation(observation, info)

    def _append(self, name, value):
        value = np.asarray(value)
        if name not in self.file:
            self.file.create_dataset(name, shape=(0, *value.shape), maxshape=(None, *value.shape),
                                     dtype=value.dtype, chunks=(1, *value.shape), compression="gzip", compression_opts=1)
        dataset = self.file[name]
        dataset.resize(len(dataset) + 1, axis=0)
        dataset[-1] = value

    def _text(self, name, value):
        if name not in self.file:
            self.file.create_dataset(name, shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(), chunks=True)
        dataset = self.file[name]
        dataset.resize(len(dataset) + 1, axis=0)
        dataset[-1] = value

    def _observation(self, obs, info):
        if obs["instruction"] != self.file.attrs["instruction"]:
            raise ValueError("Language changed inside an episode")
        self._append("observations/proprio", np.asarray(obs["proprio"], dtype=np.float32))
        for camera, rgb in obs["rgb"].items():
            self._append(f"observations/rgb/{camera}", np.asarray(rgb, dtype=np.uint8))
        self._append("observations/time_s", np.float64(info["simulated_seconds"]))
        self._append("privileged/integration_state", self.env._physics_state())
        self._append("privileged/joint_reference_rad", self.env.robot.target_q)
        self._append("privileged/jaw_reference_m", np.float64(self.env.robot.gripper_target))

    def append(self, action, next_observation, reward, terminated, truncated, info, *, source="policy", stage="", intervention=False, expert_action=None, proposed_action=None):
        action = np.asarray(action, dtype=np.float32)
        if not self.env.action_space.contains(action):
            raise ValueError("Executed action violates the declared contract")
        for candidate in (expert_action, proposed_action):
            if candidate is not None and not self.env.action_space.contains(np.asarray(candidate, dtype=np.float32)):
                raise ValueError("Action label/proposal violates the declared contract")
        self._append("transitions/action", action)
        self._append("transitions/proposed_action", action if proposed_action is None else np.asarray(proposed_action, dtype=np.float32))
        self._append("transitions/reward", np.float64(reward))
        self._append("transitions/terminated", np.bool_(terminated))
        self._append("transitions/truncated", np.bool_(truncated))
        self._append("transitions/intervention", np.bool_(intervention))
        self._append("transitions/expert_action_valid", np.bool_(expert_action is not None))
        self._append("transitions/expert_action", np.full_like(action, np.nan) if expert_action is None else np.asarray(expert_action, dtype=np.float32))
        self._text("transitions/source", source)
        self._text("transitions/stage", stage)
        self._text("privileged/metrics_json", json.dumps(info["metrics"]))
        self._observation(next_observation, info)
        self.count += 1

    def finish(self, info, *, error=None, events=()):
        self.file.attrs["summary"] = json.dumps(info)
        self.file.attrs["success"] = bool(info["is_success"])
        self.file.attrs["collector_error"] = "" if error is None else str(error)
        # Complete means the collector finalized the file, including failed or
        # externally aborted trials; environment terminal flags remain truthful.
        self.file.attrs["complete"] = True
        self.file.attrs["environment_done"] = bool(info["episode"]["done"])
        self.file.attrs["events"] = json.dumps(events)
        self.file.attrs["transition_count"] = self.count
        self.file.flush()
        self.file.close()

    def close(self):
        if self.file.id.valid:
            self.file.close()


def validate_episode(path):
    with h5py.File(path, "r") as f:
        n = len(f["transitions/action"]) if "transitions/action" in f else 0
        contract = json.loads(f.attrs["action_contract"])
        if set(f["observations/rgb"]) != set(contract["cameras"]):
            raise ValueError("Recorded RGB cameras do not match the declared contract")
        image_shape = (n + 1, *contract["image_size_hw"], 3)
        if len(f["observations/proprio"]) != n + 1:
            raise ValueError("Expected T+1 observations for T actions")
        for image in f["observations/rgb"].values():
            if image.shape != image_shape or image.dtype != np.uint8:
                raise ValueError("RGB observation shape/type does not match the declared contract")
        if len(f["privileged/integration_state"]) != n + 1:
            raise ValueError("Privileged-state alignment error")
        dt = 1 / json.loads(f.attrs["action_contract"])["policy_hz"]
        np.testing.assert_allclose(np.diff(f["observations/time_s"][:]), dt, rtol=0, atol=1e-8)
        for key, dataset in f.get("transitions", {}).items():
            if len(dataset) != n:
                raise ValueError(f"Transition length mismatch: {key}")
        action_shape = tuple(contract["shape"])
        actions = f["transitions/action"][:] if n else np.empty((0, *action_shape))
        if actions.shape != (n, *action_shape):
            raise ValueError("Recorded actions do not match the declared contract")
        if f["observations/proprio"].shape != (n + 1, 2 * contract["arm_joint_count"] + 9):
            raise ValueError("Recorded proprioception does not match the declared joint count")
        if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
            raise ValueError("Invalid recorded actions")
        return dict(transitions=n, complete=bool(f.attrs["complete"]), success=bool(f.attrs.get("success", False)), metadata=json.loads(f.attrs["metadata"]))


def training_samples(paths, *, split="train", include_failures=False, manifest=None):
    """Yield only policy inputs and labeled action targets; never simulator state.

    include_failures admits labeled expert corrections, not arbitrary failed
    executed policy actions. A caller must opt in to validation/test explicitly.
    """
    signature = None
    for path in paths:
        validate_episode(path)
        with h5py.File(path, "r") as f:
            meta = json.loads(f.attrs["metadata"])
            verify_metadata(meta, load_manifest(manifest_path(meta["environment"])) if manifest is None else manifest)
            if meta.get("split") != split or not f.attrs["complete"]:
                continue
            success = bool(f.attrs.get("success", False))
            if not success and not include_failures:
                continue
            current = (f.attrs["action_contract"], json.loads(f.attrs["implementation"])["source_sha256"])
            if signature is not None and current != signature:
                raise ValueError("Refusing to mix action contracts or implementation versions in one training stream")
            signature = current
            for t in range(len(f["transitions/action"]) if "transitions/action" in f else 0):
                source = f["transitions/source"].asstr()[t]
                labeled = bool(f["transitions/expert_action_valid"][t])
                if not labeled and not (success and source == "expert"):
                    continue
                yield {
                    "observation": {
                        "proprio": f["observations/proprio"][t],
                        "instruction": f.attrs["instruction"],
                        "rgb": {k: v[t] for k, v in f["observations/rgb"].items()},
                    },
                    "action": f["transitions/expert_action"][t] if labeled else f["transitions/action"][t],
                }


def normalization_statistics(paths, *, manifest=None):
    """Streaming statistics from successful, labeled TRAIN transitions only."""
    count, sums, squares = 0, {}, {}
    for sample in training_samples(paths, manifest=manifest):
        values = {"proprio": sample["observation"]["proprio"], "action": sample["action"]}
        count += 1
        for key, value in values.items():
            x = np.asarray(value, dtype=np.float64)
            sums[key] = sums.get(key, 0) + x
            squares[key] = squares.get(key, 0) + x*x
    if not count:
        raise ValueError("No successful labeled training transitions")
    return {"split": "train", "count": count, **{key: {
        "mean": (sums[key]/count).tolist(),
        "std": np.sqrt(np.maximum(squares[key]/count-(sums[key]/count)**2, 1e-12)).tolist(),
    } for key in sums}}


def training_chunks(paths, *, horizon=8, split="train", include_failures=False, manifest=None):
    """Policy observations and contiguous action targets with an explicit loss mask.

    Chunks stop at episode ends or the first unlabeled/faulty transition. Padding
    is zero and must be ignored via action_mask; samples never cross episodes.
    Failed episodes require explicit opt-in and individually labeled corrections.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    signature = None
    for path in paths:
        validate_episode(path)
        with h5py.File(path, "r") as f:
            meta = json.loads(f.attrs["metadata"])
            verify_metadata(meta, load_manifest(manifest_path(meta["environment"])) if manifest is None else manifest)
            success = bool(f.attrs.get("success", False))
            if meta.get("split") != split or not f.attrs["complete"] or (not success and not include_failures):
                continue
            current = (f.attrs["action_contract"], json.loads(f.attrs["implementation"])["source_sha256"])
            if signature is not None and signature != current:
                raise ValueError("Refusing to mix action contracts or implementation versions in one training stream")
            signature = current
            if "transitions/action" not in f:
                continue
            actions = f["transitions/action"][:]
            labeled = f["transitions/expert_action_valid"][:]
            sources = f["transitions/source"].asstr()[:]
            interventions = (f["transitions/intervention"][:] if "transitions/intervention" in f
                             else np.zeros(len(actions), dtype=bool))
            valid = labeled | ((sources == "expert") & success)
            targets = np.where(labeled[:, None], f["transitions/expert_action"][:], actions)
            for t in np.flatnonzero(valid):
                end = int(t)
                while (end < min(len(actions), t + horizon) and valid[end]
                       and sources[end] == sources[t] and interventions[end] == interventions[t]):
                    end += 1
                chunk = np.zeros((horizon, actions.shape[1]), dtype=np.float32)
                mask = np.zeros(horizon, dtype=bool)
                chunk[:end-t], mask[:end-t] = targets[t:end], True
                yield dict(observation=dict(proprio=f["observations/proprio"][t],
                                             instruction=f.attrs["instruction"],
                                             rgb={k: v[t] for k, v in f["observations/rgb"].items()}),
                           action=chunk, action_mask=mask, transition_index=int(t),
                           source=str(sources[t]), is_intervention=bool(interventions[t]))


def replay_episode(path, *, manifest=None):
    """Replay executed actions; verify state, scores and policy observations."""
    from environments.bookshelf_task import BookshelfEnv
    from environments.sorting import SortingEnv
    from environments.bookshelf_hard import BookshelfHardEnv
    from environments.perturbations import AppearancePerturbation

    validate_episode(path)
    with h5py.File(path, "r") as f:
        metadata = json.loads(f.attrs["metadata"])
        verify_metadata(metadata, load_manifest(manifest_path(metadata["environment"])) if manifest is None else manifest)
        cls = {"bookshelf": BookshelfEnv, "sorting": SortingEnv, "bookshelf-hard": BookshelfHardEnv}[metadata["environment"]]
        with cls(**metadata["constructor"]) as env:
            AppearancePerturbation(env).apply(metadata["appearance"], metadata["appearance_seed"])
            obs, _ = env.set_state(json.loads(f["privileged/initial_checkpoint_json"].asstr()[()]))
            def check(index):
                np.testing.assert_allclose(env._physics_state(), f["privileged/integration_state"][index], atol=1e-9, rtol=0)
                np.testing.assert_allclose(obs["proprio"], f["observations/proprio"][index], atol=1e-6, rtol=0)
                for camera in env.cameras:
                    # A separate GL context can round rasterized pixels by one
                    # intensity level even with matching simulator state.
                    np.testing.assert_allclose(obs["rgb"][camera], f[f"observations/rgb/{camera}"][index], atol=1, rtol=0)
            check(0)
            for t, action in enumerate(f.get("transitions/action", [])):
                obs, reward, terminated, truncated, _ = env.step(action)
                np.testing.assert_allclose(reward, f["transitions/reward"][t], atol=1e-10, rtol=0)
                assert terminated == f["transitions/terminated"][t]
                assert truncated == f["transitions/truncated"][t]
                check(t+1)
    return validate_episode(path)
