"""Public learner-prefix observations for parent-policy behavior preservation.

This cache supplies conditions for distillation, never imitation action labels.
The parent model, not the recorded learner commands, supplies distillation targets.
"""

import copy
import hashlib
import json
import re

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from learning.data import inspect_episodes
from learning.diffusion_policy import ModelConfig, preprocess_observation


def _public_digest(observation, cameras):
    digest = hashlib.sha256()
    instruction = observation["instruction"].encode("utf-8")
    digest.update(len(instruction).to_bytes(8, "little"))
    digest.update(instruction)
    for name, value in [("proprio", np.asarray(observation["proprio"], dtype="<f4")),
                        *((camera, observation["rgb"][camera]) for camera in cameras)]:
        header = json.dumps([name, list(value.shape), value.dtype.str], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


class PreservationDataset(Dataset):
    """Deduplicated pre-takeover observations with fixed parent normalization.

    ``records`` must be intervention-only records from ``inspect_episodes``.
    Successful episodes and their provenance are revalidated before caching.
    Prefix cutoff is exclusive; anchors are 0, stride, ... whose full teacher
    action chunk fits before both that cutoff and the first expert action.
    No future observations or action targets are returned.
    """

    def __init__(self, records, config, tokenizer, normalization, *, prefix_steps,
                 anchor_stride=8, parent_checkpoint_sha256):
        for name, value in (("prefix_steps", prefix_steps), ("anchor_stride", anchor_stride)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(parent_checkpoint_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", parent_checkpoint_sha256) is None:
            raise ValueError("A parent checkpoint SHA256 is required")
        if not isinstance(config, ModelConfig) or config.action_representation != "velocity":
            raise ValueError("Preservation supports the validated velocity ModelConfig only")
        if prefix_steps < config.horizon:
            raise ValueError("prefix_steps must cover a full model horizon")
        if tokenizer.max_tokens != config.max_tokens:
            raise ValueError("Tokenizer length does not match the model")
        if not isinstance(normalization, dict) or normalization.get("split") != "train":
            raise ValueError("Preservation requires fixed parent train normalization")
        mean = np.asarray(normalization.get("mean"), dtype=np.float64)
        std = np.asarray(normalization.get("std"), dtype=np.float64)
        if (mean.shape != (config.proprio_dim,) or std.shape != (config.proprio_dim,)
                or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)):
            raise ValueError("Invalid parent normalization")
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("Parent normalization cannot be represented safely in float32")
        self.mean.setflags(write=False)
        self.std.setflags(write=False)
        self.normalization = copy.deepcopy(normalization)
        records = list(records)
        if not records or any(r.get("split") != "train" or r.get("is_intervention") is not True for r in records):
            raise ValueError("Preservation requires successful intervention-only train records")
        verified, skipped = inspect_episodes([r["path"] for r in records], "train", allow_interventions=True)
        if skipped or len(verified) != len(records):
            raise ValueError("Preservation requires successful expert-takeover episodes")
        for supplied, fresh in zip(records, verified):
            if any(supplied.get(key) != value for key, value in fresh.items()):
                raise ValueError("Preservation record no longer matches its validated episode")
            if fresh["intervention"]["parent_checkpoint_sha256"] != parent_checkpoint_sha256:
                raise ValueError("Preservation parent checkpoint mismatch")
        self.records = copy.deepcopy(verified)
        limits = [min(prefix_steps, r["intervention"]["actual_takeover_step"]) for r in verified]
        anchor_stops = [max(0, limit - config.horizon + 1) for limit in limits]
        maximum = sum((stop + anchor_stride - 1) // anchor_stride for stop in anchor_stops)
        if not maximum:
            raise ValueError("Preservation selection contains zero full-chunk anchors")
        self.images = np.empty((maximum, len(config.cameras), 3, config.image_height, config.image_width), dtype=np.uint8)
        self.proprio = np.empty((maximum, config.proprio_dim), dtype=np.float32)
        self.tokens = np.empty((maximum, config.max_tokens), dtype=np.int64)
        self.count = 0
        anchors, duplicates, seen = [], [], {}
        for record, stop in zip(verified, anchor_stops):
            with h5py.File(record["path"], "r") as episode:
                for index in range(0, stop, anchor_stride):
                    # These are the only datasets used to build model tensors.
                    observation = dict(proprio=episode["observations/proprio"][index],
                                       instruction=str(episode.attrs["instruction"]),
                                       rgb={camera: episode[f"observations/rgb/{camera}"][index]
                                            for camera in config.cameras})
                    images, proprio, tokens = preprocess_observation(observation, config, tokenizer, record["action_contract"])
                    digest = _public_digest(observation, config.cameras)
                    anchor = dict(path=record["path"], rollout_id=record["rollout_id"],
                                  case_id=record["case_id"], episode_sha256=record["sha256"], transition_index=index,
                                  public_observation_sha256=digest,
                                  source_sha256=record["collection_implementation"]["source_sha256"])
                    if digest in seen:
                        duplicates.append(dict(anchor, duplicate_of_index=seen[digest]))
                        continue
                    normalized = (proprio - self.mean) / self.std
                    if not np.isfinite(normalized).all():
                        raise ValueError("Nonfinite normalized preservation observation")
                    seen[digest] = self.count
                    anchors.append(anchor)
                    self.images[self.count], self.proprio[self.count], self.tokens[self.count] = images, normalized, tokens
                    self.count += 1
        if not self.count:
            raise ValueError("Preservation selection contains zero anchors")
        # Exact-size copies release allocation reserved for duplicate observations.
        self.images = self.images[:self.count].copy()
        self.proprio = self.proprio[:self.count].copy()
        self.tokens = self.tokens[:self.count].copy()
        selection = dict(parent_checkpoint_sha256=parent_checkpoint_sha256,
                         prefix_steps=prefix_steps, anchor_stride=anchor_stride, horizon=config.horizon,
                         anchors=[{key: anchor[key] for key in ("rollout_id", "episode_sha256", "transition_index", "public_observation_sha256")}
                                  for anchor in anchors])
        selection_sha256 = hashlib.sha256(json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.metadata = dict(version="forte-parent-preservation-v1", split="train",
                             selection_sha256=selection_sha256,
                             selection_randomness="none; episode order then increasing transition index",
                             role="parent_behavior_distillation_conditions_only",
                             parent_checkpoint_sha256=parent_checkpoint_sha256,
                             prefix_steps=prefix_steps, anchor_stride=anchor_stride, horizon=config.horizon,
                             full_chunk_boundary="t + horizon <= min(prefix_steps, actual_takeover_step)",
                             candidate_anchor_count=maximum, unique_anchor_count=self.count,
                             duplicate_anchor_count=len(duplicates), anchors=anchors,
                             duplicate_anchors=duplicates, normalization=copy.deepcopy(self.normalization),
                             normalization_refit=False,
                             source_sha256=verified[0]["collection_implementation"]["source_sha256"],
                             input_fields=["rgb", "proprio", "instruction"], action_labels=False,
                             deduplication="exact native RGB, float32 public proprioception and instruction")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return (torch.from_numpy(self.images[index]), torch.from_numpy(self.proprio[index]),
                torch.from_numpy(self.tokens[index]))

    @property
    def cache_bytes(self):
        return sum(value.nbytes for value in (self.images, self.proprio, self.tokens, self.mean, self.std))
