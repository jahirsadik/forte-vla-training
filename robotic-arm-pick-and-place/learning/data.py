"""Validated episode selection and compact in-memory image cache for the pilot."""

import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.demonstrations import training_chunks, validate_episode
from utils.research_protocol import load_manifest, manifest_path, verify_metadata
from learning.diffusion_policy import check_contract, file_sha256, preprocess_observation
from learning.reference_actions import reference_targets
from learning.public_history import HISTORY_DIM, normalize_history, reconstruct_history


INTERVENTION_VERSION = "forte-release-intervention-v1"
INTERVENTION_EXPERT = "supported_goal_release_v1"
INTERVENTION_VERSION_BY_EXPERT = {
    INTERVENTION_EXPERT: INTERVENTION_VERSION,
    "grasped_carry_recovery_v1": "forte-carry-intervention-v1",
    "sustained_grasp_to_goal_v1": "forte-grasp-intervention-v1",
}
_INTERVENTION_FIELDS = {"version", "parent_checkpoint_sha256", "policy_seed",
                        "minimum_policy_steps", "confirmation_steps", "actual_takeover_step", "expert"}


def _sha256_value(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def intervention_rollout_id(case_id, intervention):
    """Identify a declared trial before its outcome or actual takeover is known."""
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("Intervention requires a frozen case ID")
    if not isinstance(intervention, dict) or set(intervention) != _INTERVENTION_FIELDS:
        raise ValueError("Unexpected intervention descriptor fields")
    if (not isinstance(intervention["expert"], str)
            or not isinstance(intervention["version"], str)
            or INTERVENTION_VERSION_BY_EXPERT.get(intervention["expert"]) != intervention["version"]
            or not _sha256_value(intervention["parent_checkpoint_sha256"])):
        raise ValueError("Unsupported intervention descriptor or parent checkpoint hash")
    for key, minimum in (("policy_seed", 0), ("minimum_policy_steps", 1), ("confirmation_steps", 1)):
        value = intervention[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Invalid intervention {key}")
    immutable = {key: intervention[key] for key in sorted(_INTERVENTION_FIELDS - {"actual_takeover_step"})}
    encoded = json.dumps(dict(case_id=case_id, intervention=immutable), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def trajectory_digest(episode, case_id):
    """Detect identical physical repeats even if rollout metadata was relabeled.

    Uses public measured trajectories and executed commands. Camera rasterization
    roundoff or metadata edits must not make the same physical trajectory count
    as additional imitation data.
    """
    digest = hashlib.sha256(case_id.encode())
    for name in ("observations/proprio", "transitions/action"):
        dataset = episode[name]
        digest.update(name.encode())
        digest.update(json.dumps(dataset.shape).encode())
        for start in range(0, len(dataset), 1024):
            digest.update(np.asarray(dataset[start:start + 1024], dtype="<f4").tobytes())
    return digest.hexdigest()


def validate_intervention(episode, meta):
    """Accept one actual learner-prefix/expert-suffix takeover, never relabeling.

    Failed finalized trials can be structurally valid and are retained, but the
    training selector below admits successful corrections only.
    """
    if meta.get("split") != "train":
        raise ValueError("Intervention training episodes are restricted to the train split")
    descriptor = meta.get("intervention")
    expected = intervention_rollout_id(meta["case_id"], descriptor)
    if meta.get("rollout_id") != expected or not _sha256_value(meta.get("collection_plan_sha256")):
        raise ValueError("Intervention rollout identity or collection plan hash mismatch")
    if (meta.get("recovery", "none") != "none" or meta.get("appearance", "nominal") != "nominal"
            or meta.get("custom_horizon") or meta.get("counterfactual_diagnostic")
            or meta.get("instruction_override") is not None
            or meta.get("effective_instruction") not in (None, str(episode.attrs["instruction"]))):
        raise ValueError("Interventions reject faults, appearance changes, overrides and custom horizons")
    for name in ("source", "intervention", "expert_action_valid", "expert_action", "proposed_action"):
        if f"transitions/{name}" not in episode:
            raise ValueError(f"Intervention is missing transitions/{name}")
    actions = episode["transitions/action"][:]
    proposed = episode["transitions/proposed_action"][:]
    source = episode["transitions/source"].asstr()[:]
    intervention = episode["transitions/intervention"][:]
    valid = episode["transitions/expert_action_valid"][:]
    labels = episode["transitions/expert_action"][:]
    n = len(actions)
    if (source.shape != (n,) or intervention.shape != (n,) or valid.shape != (n,)
            or labels.shape != actions.shape or proposed.shape != actions.shape
            or intervention.dtype != np.bool_ or valid.dtype != np.bool_
            or not np.array_equal(proposed, actions)):
        raise ValueError("Malformed intervention fields or hidden command overrides")
    takeover = descriptor["actual_takeover_step"]
    events = json.loads(episode.attrs.get("events", "[]"))
    if (not isinstance(events, list) or len(events) > 1
            or any(not isinstance(event, dict) or event.get("kind") != "expert_takeover"
                   or takeover is None or event.get("step") != takeover for event in events)):
        raise ValueError("Interventions reject fault events or inconsistent takeover events")
    if takeover is None:
        if np.any(source != "policy") or intervention.any() or valid.any():
            raise ValueError("A trial without takeover cannot contain expert suffix labels")
        return 0
    if (isinstance(takeover, bool) or not isinstance(takeover, int)
            or not descriptor["minimum_policy_steps"] <= takeover < n):
        raise ValueError("Invalid actual takeover step")
    if (np.any(source[:takeover] != "policy") or intervention[:takeover].any() or valid[:takeover].any()
            or np.any(source[takeover:] != "expert") or not intervention[takeover:].all()
            or not valid[takeover:].all()
            or not np.array_equal(labels[takeover:], actions[takeover:])):
        raise ValueError("Intervention must contain an unlabeled policy prefix and exactly executed expert suffix")
    if bool(episode.attrs.get("success", False)) and not episode.attrs.get("environment_done", False):
        raise ValueError("A successful intervention must finish the physical episode")
    return n - takeover


def inspect_episodes(paths, split, allow_interventions=False):
    """Reject split errors instead of silently treating test data as validation."""
    if split not in ("train", "validation"):
        raise ValueError("The trainer accepts train/validation data only, never test")
    if not isinstance(allow_interventions, bool):
        raise ValueError("allow_interventions must be an explicit boolean")
    records, skipped, seen_cases, seen_paths = [], [], set(), set()
    seen_expert_cases, seen_rollouts, seen_hashes, seen_trajectories = set(), set(), set(), set()
    signature = None
    for value in paths:
        path = Path(value).resolve()
        if str(path) in seen_paths:
            raise ValueError("Duplicate episode path")
        seen_paths.add(str(path))
        result = validate_episode(path)
        with h5py.File(path, "r") as episode:
            meta = json.loads(episode.attrs["metadata"])
            verify_metadata(meta, load_manifest(manifest_path(meta["environment"])))
            if meta["split"] != split:
                raise ValueError(f"Requested {split} dataset contains {meta['split']} episode: {path}")
            is_intervention = meta.get("policy") == "expert_intervention"
            if meta["case_id"] in seen_cases and not (allow_interventions and split == "train"):
                raise ValueError("Duplicate frozen case in dataset")
            seen_cases.add(meta["case_id"])
            supervised_transitions = None
            if is_intervention and allow_interventions:
                supervised_transitions = validate_intervention(episode, meta)
                if meta["rollout_id"] in seen_rollouts:
                    raise ValueError("Duplicate intervention rollout ID")
                seen_rollouts.add(meta["rollout_id"])
            elif meta.get("policy") == "expert":
                if meta["case_id"] in seen_expert_cases:
                    raise ValueError("Duplicate frozen expert case in dataset")
                seen_expert_cases.add(meta["case_id"])
            exclusion = None
            if not result["complete"] or not result["success"]:
                exclusion = "incomplete_or_unsuccessful"
            elif episode.attrs.get("collector_error", ""):
                exclusion = "collector_error"
            elif is_intervention and allow_interventions and supervised_transitions == 0:
                exclusion = "no_expert_suffix"
            elif meta.get("custom_horizon"):
                exclusion = "custom_horizon_diagnostic"
            elif (meta.get("counterfactual_diagnostic") or meta.get("instruction_override") is not None
                  or meta.get("effective_instruction") not in (None, str(episode.attrs["instruction"]))):
                exclusion = "instruction_override_diagnostic"
            elif meta.get("policy") != "expert" and not (is_intervention and allow_interventions):
                exclusion = "non_expert_collector"
            if exclusion:
                skipped.append(dict(path=str(path), case_id=meta["case_id"], reason=exclusion))
                continue
            contract = json.loads(episode.attrs["action_contract"])
            check_contract(contract)
            implementation = json.loads(episode.attrs["implementation"])
            current = (json.dumps(contract, sort_keys=True), implementation["source_sha256"])
            if signature is not None and current != signature:
                raise ValueError("Refusing mixed episode contracts or collection implementations")
            signature = current
            file_hash = file_sha256(path)
            trajectory_hash = trajectory_digest(episode, meta["case_id"])
            if file_hash in seen_hashes or trajectory_hash in seen_trajectories:
                raise ValueError("Duplicate episode content or physical trajectory")
            seen_hashes.add(file_hash)
            seen_trajectories.add(trajectory_hash)
            if supervised_transitions is None:
                valid = episode["transitions/expert_action_valid"][:]
                sources = episode["transitions/source"].asstr()[:]
                supervised_transitions = int((valid | (sources == "expert")).sum())
            records.append(dict(path=str(path), sha256=file_hash, trajectory_sha256=trajectory_hash,
                                case_id=meta["case_id"], family_id=meta["family_id"],
                                rollout_id=meta.get("rollout_id"), is_intervention=is_intervention,
                                intervention=meta.get("intervention"), collection_plan_sha256=meta.get("collection_plan_sha256"),
                                supervised_transitions=supervised_transitions,
                                manifest_sha256=meta["manifest_sha256"], environment=meta["environment"],
                                split=split, instruction=str(episode.attrs["instruction"]),
                                transitions=result["transitions"], action_contract=contract,
                                collection_implementation=implementation))
    if not records:
        raise ValueError(f"No complete successful {split} episodes")
    return records, skipped


def check_data_separation(train_records, validation_records):
    for key in ("path", "sha256", "case_id", "family_id", "trajectory_sha256", "rollout_id"):
        overlap = {r.get(key) for r in train_records if r.get(key) is not None} & {r.get(key) for r in validation_records if r.get(key) is not None}
        if overlap:
            raise ValueError(f"Training/validation overlap in {key}")
    for record in validation_records:
        if (record["action_contract"] != train_records[0]["action_contract"]
                or record["collection_implementation"]["source_sha256"] != train_records[0]["collection_implementation"]["source_sha256"]):
            raise ValueError("Training and validation must use the same collection implementation and contract")


def load_reference_targets(path, contract):
    """Reconstruct targets only for successful nominal expert command streams.

    Correction labels and hidden command overrides require a separate causal
    target definition. Reject them here instead of inventing counterfactual
    references from actions the expert never issued.
    """
    with h5py.File(path, "r") as episode:
        meta = json.loads(episode.attrs["metadata"])
        if (not episode.attrs.get("complete", False) or not episode.attrs.get("success", False)
                or episode.attrs.get("collector_error", "") or meta.get("policy") != "expert"
                or meta.get("recovery", "none") != "none"
                or meta.get("appearance", "nominal") != "nominal"
                or meta.get("custom_horizon") or meta.get("counterfactual_diagnostic")
                or meta.get("instruction_override") is not None
                or json.loads(episode.attrs.get("events", "[]"))):
            raise ValueError("Joint-reference targets require complete successful nominal expert episodes without faults")
        if json.loads(episode.attrs["action_contract"]) != contract:
            raise ValueError("Joint-reference target contract mismatch")
        actions = episode["transitions/action"][:]
        if "transitions/proposed_action" not in episode:
            raise ValueError("Joint-reference targets require recorded policy-issued proposed actions")
        proposed = episode["transitions/proposed_action"][:]
        sources = episode["transitions/source"].asstr()[:]
        if (proposed.shape != actions.shape or not np.array_equal(proposed, actions)
                or np.any(sources != "expert")
                or ("transitions/intervention" in episode and episode["transitions/intervention"][:].any())):
            raise ValueError("Joint-reference targets reject faults, interventions and non-expert command streams")
        valid = episode["transitions/expert_action_valid"][:]
        if valid.any() and not np.array_equal(episode["transitions/expert_action"][:][valid], actions[valid]):
            raise ValueError("Joint-reference targets reject counterfactual expert corrections")
        return reference_targets(episode["observations/proprio"][0], proposed, contract)


def load_public_history(path, contract, steps):
    """Read public observations and issued commands; never read privileged data."""
    with h5py.File(path, "r") as episode:
        if json.loads(episode.attrs["action_contract"]) != contract:
            raise ValueError("History action contract mismatch")
        if "transitions/proposed_action" not in episode:
            raise ValueError("History requires recorded policy-issued proposed actions")
        return reconstruct_history(episode["observations/proprio"][:],
                                   episode["transitions/proposed_action"][:], contract, steps)


class CompactChunkDataset(Dataset):
    """Resize one native frame at a time; retain only compact uint8 pixels in RAM.

    Uses the audited chunk reader for labels, boundaries and split membership.
    The cache is one array per field, preallocated from an anchor-count upper bound;
    there is no native-image list or duplicate full-frame training copy.
    """

    def __init__(self, records, config, tokenizer, *, split, anchor_stride=1):
        if isinstance(anchor_stride, bool) or anchor_stride < 1:
            raise ValueError("anchor_stride must be positive")
        self.records = records
        self.config = config
        self.split = split
        self.anchor_stride = anchor_stride
        maximum = sum((r.get("supervised_transitions", r["transitions"]) + anchor_stride - 1) // anchor_stride for r in records)
        self.images = np.empty((maximum, len(config.cameras), 3, config.image_height, config.image_width), dtype=np.uint8)
        self.proprio = np.empty((maximum, config.proprio_dim), dtype=np.float32)
        self.tokens = np.empty((maximum, config.max_tokens), dtype=np.int64)
        self.actions = np.empty((maximum, config.horizon, config.action_dim), dtype=np.float32)
        self.masks = np.empty((maximum, config.horizon), dtype=bool)
        self.episode_index = np.empty(maximum, dtype=np.int32)
        self.is_intervention = np.empty(maximum, dtype=bool)
        self.history = (np.empty((maximum, config.state_history_steps, HISTORY_DIM), dtype=np.float32)
                        if config.state_history_steps else None)
        self.count = 0
        for episode_index, record in enumerate(records):
            history = (load_public_history(record["path"], record["action_contract"], config.state_history_steps)
                       if config.state_history_steps else None)
            reference = (load_reference_targets(record["path"], record["action_contract"])
                         if getattr(config, "action_representation", "velocity") == "joint_reference" else None)
            for anchor, sample in enumerate(training_chunks([record["path"]], horizon=config.horizon, split=split)):
                if anchor % anchor_stride:
                    continue
                images, proprio, tokens = preprocess_observation(sample["observation"], config, tokenizer, record["action_contract"])
                action, mask = sample["action"], sample["action_mask"]
                if reference is not None:
                    start = sample["transition_index"]
                    action = np.zeros_like(action)
                    length = int(mask.sum())
                    action[:length] = reference[start:start + length]
                if not mask.any() or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
                    raise ValueError("Invalid supervised action chunk")
                self.images[self.count], self.proprio[self.count], self.tokens[self.count] = images, proprio, tokens
                self.actions[self.count], self.masks[self.count], self.episode_index[self.count] = action, mask, episode_index
                self.is_intervention[self.count] = bool(record.get("is_intervention", False) and sample["is_intervention"])
                if history is not None:
                    self.history[self.count] = history[sample["transition_index"]]
                self.count += 1
        if not self.count:
            raise ValueError("Dataset contains no valid expert chunk anchors")

    def normalization(self):
        if self.split != "train":
            raise ValueError("Only training data can fit normalization")
        values = self.proprio[:self.count].astype(np.float64)
        # A floor avoids amplifying float32 noise in almost-constant coordinates.
        return dict(mean=values.mean(0).tolist(), std=np.maximum(values.std(0), 1e-3).tolist(),
                    split="train", count=self.count, std_floor=1e-3)

    def set_normalization(self, normalization):
        if normalization.get("split") != "train":
            raise ValueError("Normalization must be fit on training observations")
        mean, std = np.asarray(normalization["mean"]), np.asarray(normalization["std"])
        if mean.shape != (23,) or std.shape != (23,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("Invalid proprio normalization")
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        sample = (torch.from_numpy(self.images[index]),
                torch.from_numpy((self.proprio[index] - self.mean) / self.std),
                torch.from_numpy(self.tokens[index]), torch.from_numpy(self.actions[index]),
                torch.from_numpy(self.masks[index]))
        if self.history is not None:
            sample += (torch.from_numpy(normalize_history(self.history[index], self.mean, self.std)),)
        return sample

    @property
    def cache_bytes(self):
        return (sum(getattr(self, name).nbytes for name in ("images", "proprio", "tokens", "actions", "masks", "episode_index", "is_intervention"))
                + (0 if self.history is None else self.history.nbytes))
