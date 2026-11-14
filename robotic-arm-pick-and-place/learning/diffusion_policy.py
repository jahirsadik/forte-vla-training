"""Small text-conditioned action diffusion baseline, trained from scratch.

This is a deliberately compact baseline inspired by Diffusion Policy, not an
implementation of its published vision backbone/temporal U-Net or a pretrained
VLA. Only the public RGB/proprioception/instruction observation enters the model.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from torch import nn

CHECKPOINT_SCHEMA = "forte-text-diffusion-v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def learner_source_sha256():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "learning").glob("*.py")) + [root / "scripts/train_policy.py"]
    for path in paths:
        if path.exists():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelConfig:
    horizon: int = 8
    action_dim: int = 8
    proprio_dim: int = 23
    image_height: int = 96
    image_width: int = 96
    cameras: tuple = ("front", "wrist")
    hidden_dim: int = 256
    diffusion_steps: int = 100
    inference_steps: int = 16
    max_tokens: int = 64
    prediction_type: str = "epsilon"
    gripper_mode: str = "continuous"
    action_representation: str = "velocity"
    text_pooling: str = "last"
    state_history_steps: int = 0

    def __post_init__(self):
        object.__setattr__(self, "cameras", tuple(self.cameras))
        if (self.horizon != 8 or self.action_dim != 8 or self.proprio_dim != 23
                or self.cameras != ("front", "wrist")):
            raise ValueError("This baseline implements the fixed 8-step, 8D, two-camera Forte interface")
        if min(self.image_height, self.image_width) < 24 or self.hidden_dim < 32:
            raise ValueError("Image/model dimensions are too small")
        if not 2 <= self.inference_steps <= self.diffusion_steps or self.diffusion_steps < 4:
            raise ValueError("Invalid diffusion schedule")
        if self.max_tokens < 2:
            raise ValueError("Invalid tokenizer length")
        if self.prediction_type not in ("epsilon", "sample"):
            raise ValueError("prediction_type must be epsilon or sample")
        if self.gripper_mode not in ("continuous", "binary"):
            raise ValueError("gripper_mode must be continuous or binary")
        if self.action_representation not in ("velocity", "joint_reference"):
            raise ValueError("action_representation must be velocity or joint_reference")
        if self.text_pooling not in ("last", "masked_mean"):
            raise ValueError("text_pooling must be last or masked_mean")
        if type(self.state_history_steps) is not int or not 0 <= self.state_history_steps <= 32:
            raise ValueError("state_history_steps must be an integer in 0..32")


def inference_decoding(config):
    """Record the explicit checkpoint decoding choice; never infer it from a task."""
    return dict(gripper_mode=config.gripper_mode,
                jaw_rule=("+1 when the sampled jaw action is >= 0, otherwise -1"
                          if config.gripper_mode == "binary" else "sampled continuous normalized jaw action unchanged"),
                arm_rule=("absolute joint references normalized by public joint limits, converted to rate-limited velocity actions using issued-command history"
                          if config.action_representation == "joint_reference"
                          else "all seven sampled arm coordinates unchanged"))


def validate_gripper_targets(actions, mask, gripper_mode):
    """Binary decoding requires exact endpoint labels; ignore unlabeled/padded data."""
    if gripper_mode not in ("continuous", "binary"):
        raise ValueError("gripper_mode must be continuous or binary")
    if gripper_mode == "continuous":
        return
    actions, mask = np.asarray(actions), np.asarray(mask)
    if actions.ndim < 2 or actions.shape[-1] != 8 or mask.shape != actions.shape[:-1] or mask.dtype != np.bool_:
        raise ValueError("Invalid supervised action/mask shape for binary gripper validation")
    jaw = actions[..., -1][mask]
    if np.any((jaw != -1) & (jaw != 1)):
        raise ValueError("Binary gripper requires every valid supervised jaw target to be exactly -1 or +1; intermediate apertures require continuous mode")


class Tokenizer:
    """Deterministic train-only word vocabulary; token order is retained."""

    def __init__(self, words, max_tokens=64):
        if len(set(words)) != len(words) or list(words[:2]) != ["<pad>", "<unk>"]:
            raise ValueError("Invalid checkpoint vocabulary")
        self.words = list(words)
        self.lookup = {word: i for i, word in enumerate(words)}
        self.max_tokens = max_tokens

    @staticmethod
    def tokens(text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Instruction must be nonempty text")
        return re.findall(r"[a-z0-9]+", text.lower())

    @classmethod
    def fit(cls, instructions, max_tokens=64):
        words = sorted({word for text in instructions for word in cls.tokens(text)})
        return cls(["<pad>", "<unk>", *words], max_tokens)

    def encode(self, text):
        words = self.tokens(text)
        if not words or len(words) > self.max_tokens:
            raise ValueError("Instruction token length is outside the declared interface")
        output = np.zeros(self.max_tokens, dtype=np.int64)
        output[:len(words)] = [self.lookup.get(word, 1) for word in words]
        return output


def preprocess_observation(observation, config, tokenizer, contract):
    """Identical PIL bilinear resize and uint8 representation for train/inference."""
    if set(observation) != {"rgb", "proprio", "instruction"}:
        raise ValueError("Policy inputs must contain only rgb, proprio and instruction")
    proprio = np.asarray(observation["proprio"], dtype=np.float32)
    if proprio.shape != (config.proprio_dim,) or not np.isfinite(proprio).all():
        raise ValueError("Invalid policy proprioception")
    if set(observation["rgb"]) != set(config.cameras):
        raise ValueError("Policy camera names do not match the checkpoint")
    images = []
    expected = (*contract["image_size_hw"], 3)
    for camera in config.cameras:
        frame = np.asarray(observation["rgb"][camera])
        if frame.shape != expected or frame.dtype != np.uint8:
            raise ValueError("Policy RGB must match native checkpoint capture shape and uint8 dtype")
        resized = Image.fromarray(frame).resize((config.image_width, config.image_height), Image.Resampling.BILINEAR)
        images.append(np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1))
    return np.stack(images), proprio.copy(), tokenizer.encode(observation["instruction"])


class ResidualDenoiserBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(width, 2 * width)
        self.net = nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))

    def forward(self, x, condition):
        scale, bias = self.film(condition).chunk(2, dim=-1)
        return x + self.net(self.norm(x) * (1 + scale) + bias)


class SpatialGridPool(nn.Module):
    """Nonoverlapping 3×3 mean bins with deterministic CUDA backward.

    AdaptiveAvgPool2d has no deterministic CUDA backward in the tested PyTorch
    runtime. Explicit disjoint bins retain spatial features without that kernel.
    Unequal dimensions distribute the remainder into the final bins.
    """

    def forward(self, images):
        height, width = images.shape[-2:]
        rows = []
        for row in range(3):
            values = []
            for column in range(3):
                region = images[..., row * height // 3:(row + 1) * height // 3,
                                column * width // 3:(column + 1) * width // 3]
                values.append(region.mean(dim=(-2, -1)))
            rows.append(torch.stack(values, dim=-1))
        return torch.stack(rows, dim=-2)


class ActionDiffusion(nn.Module):
    def __init__(self, config, vocabulary_size):
        super().__init__()
        self.config = config
        self.vision = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            SpatialGridPool(), nn.Flatten(), nn.Linear(64 * 9, 128), nn.SiLU(),
        )
        self.word_embedding = nn.Embedding(vocabulary_size, 32, padding_idx=0)
        self.text = nn.GRU(32, 64, batch_first=True)
        self.proprio = nn.Sequential(nn.Linear(config.proprio_dim, 64), nn.SiLU())
        self.condition = nn.Sequential(nn.Linear(len(config.cameras) * 128 + 128, config.hidden_dim), nn.SiLU())
        self.time = nn.Sequential(nn.Linear(64, config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, config.hidden_dim))
        self.action = nn.Linear(config.horizon * config.action_dim, config.hidden_dim)
        self.blocks = nn.ModuleList([ResidualDenoiserBlock(config.hidden_dim) for _ in range(3)])
        self.output = nn.Linear(config.hidden_dim, config.horizon * config.action_dim)
        # Construct after all legacy parameters to preserve their seeded draws.
        self.history_encoder = None
        if config.state_history_steps:
            from learning.public_history import HISTORY_DIM
            self.history_encoder = nn.Sequential(
                nn.Linear(config.state_history_steps * HISTORY_DIM, 128), nn.SiLU(),
                nn.Linear(128, 64), nn.SiLU())
        # Cosine cumulative schedule from improved DDPM, with finite endpoints.
        times = torch.linspace(0, config.diffusion_steps, config.diffusion_steps + 1, dtype=torch.float64)
        cumulative = torch.cos(((times / config.diffusion_steps + .008) / 1.008) * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(.0001, .999)
        self.register_buffer("alpha_bar", torch.cumprod(1 - beta, dim=0).float())

    def encode_condition(self, images, proprio, tokens, history=None):
        batch, cameras = images.shape[:2]
        images = images.reshape(-1, *images.shape[2:]).float() / 255.
        visual = self.vision(images).reshape(batch, cameras * 128)
        encoded, _ = self.text(self.word_embedding(tokens))
        lengths = (tokens != 0).sum(-1).clamp_min(1)
        if self.config.text_pooling == "last":
            # Keep historical checkpoint arithmetic exactly unchanged.
            language = encoded[torch.arange(batch, device=tokens.device), lengths - 1]
        else:
            # GRU outputs remain order-dependent; padding contributes no mass.
            mask = (tokens != 0).unsqueeze(-1)
            language = (encoded * mask).sum(dim=1) / lengths.unsqueeze(-1)
        proprio_features = self.proprio(proprio)
        if self.history_encoder is not None:
            from learning.public_history import HISTORY_DIM
            if history is None or history.shape != (batch, self.config.state_history_steps, HISTORY_DIM):
                raise ValueError("History-enabled model requires matching causal history features")
            proprio_features = proprio_features + self.history_encoder(history.flatten(1))
        elif history is not None:
            raise ValueError("History was supplied to a legacy current-frame model")
        return self.condition(torch.cat([visual, proprio_features, language], dim=-1))

    def predict_target(self, noisy_actions, timesteps, condition):
        """Predict epsilon or clean actions according to the saved configuration."""
        frequency = torch.exp(torch.arange(32, device=noisy_actions.device) * (-math.log(10000) / 31))
        phase = timesteps.float()[:, None] * frequency[None]
        condition = condition + self.time(torch.cat([phase.sin(), phase.cos()], dim=-1))
        x = self.action(noisy_actions.flatten(1))
        for block in self.blocks:
            x = block(x, condition)
        return self.output(x).reshape(-1, self.config.horizon, self.config.action_dim)

    def training_loss(self, images, proprio, tokens, actions, mask, history=None, *, generator=None):
        # Invalid targets never condition valid outputs through the joint MLP.
        clean = torch.where(mask[..., None], actions, torch.zeros_like(actions))
        noise = torch.randn(actions.shape, device=actions.device, generator=generator)
        timestep = torch.randint(self.config.diffusion_steps, (len(actions),), device=actions.device, generator=generator)
        alpha = self.alpha_bar[timestep, None, None]
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        predicted = self.predict_target(noisy, timestep, self.encode_condition(images, proprio, tokens, history))
        target = noise if self.config.prediction_type == "epsilon" else clean
        return masked_target_loss(predicted, target, mask)

    def predict_noise(self, noisy_actions, timesteps, condition):
        """Compatibility entry point for the original epsilon-only baseline."""
        if self.config.prediction_type != "epsilon":
            raise ValueError("This model predicts clean actions; use predict_target")
        return self.predict_target(noisy_actions, timesteps, condition)

    def noise_loss(self, images, proprio, tokens, actions, mask, *, generator=None):
        """Compatibility entry point; never label clean-action loss as noise loss."""
        if self.config.prediction_type != "epsilon":
            raise ValueError("This model predicts clean actions; use training_loss")
        return self.training_loss(images, proprio, tokens, actions, mask, generator=generator)

    @torch.inference_mode()
    def sample(self, images, proprio, tokens, *, generator, history=None):
        condition = self.encode_condition(images, proprio, tokens, history)
        x = torch.randn((len(images), self.config.horizon, self.config.action_dim), device=images.device, generator=generator)
        schedule = torch.linspace(self.config.diffusion_steps - 1, 0, self.config.inference_steps).round().long().tolist()
        for index, step in enumerate(schedule):
            timestep = torch.full((len(x),), step, device=x.device, dtype=torch.long)
            alpha = self.alpha_bar[step]
            prediction = self.predict_target(x, timestep, condition)
            if self.config.prediction_type == "epsilon":
                clean = ((x - (1 - alpha).sqrt() * prediction) / alpha.sqrt()).clamp(-1, 1)
            else:
                clean = prediction.clamp(-1, 1)
            if index + 1 == len(schedule):
                x = clean
            else:
                # Deterministic DDIM (eta=0), with noise consistent with clipped x0.
                noise = (x - alpha.sqrt() * clean) / (1 - alpha).sqrt()
                previous = self.alpha_bar[schedule[index + 1]]
                x = previous.sqrt() * clean + (1 - previous).sqrt() * noise
        return x


def masked_target_loss(predicted, target, mask):
    if predicted.shape != target.shape or mask.shape != predicted.shape[:-1] or mask.dtype != torch.bool:
        raise ValueError("Diffusion target/mask shape mismatch")
    count = mask.sum() * predicted.shape[-1]
    if not count:
        raise ValueError("Diffusion batch contains no labeled actions")
    return torch.where(mask[..., None], (predicted - target).square(), 0.).sum() / count


def masked_noise_loss(predicted, noise, mask):
    """Backward-compatible name for the original masked epsilon objective."""
    return masked_target_loss(predicted, noise, mask)


def resolve_device(device):
    if str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type not in ("cpu", "cuda") or (resolved.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested CPU/CUDA device is unavailable")
    return resolved


def check_contract(contract, expected=None):
    if (contract.get("id") != "forte-joint-reference-v2" or contract.get("shape") != [8]
            or contract.get("arm_joint_count") != 7 or contract.get("policy_hz") != 20
            or contract.get("cameras") != ["front", "wrist"]
            or contract.get("privileged_fields_are_policy_inputs") is not False):
        raise ValueError("Checkpoint requires the declared Forte public 8D/20Hz interface")
    if expected is not None and contract != expected:
        differing = sorted(k for k in set(contract) | set(expected) if contract.get(k) != expected.get(k))
        raise ValueError("Checkpoint action/observation contract mismatch: " + ", ".join(differing))


class DiffusionPolicy:
    def __init__(self, model, tokenizer, normalization, contract, metadata, device="auto"):
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.config = model.config
        self.tokenizer = tokenizer
        self.contract = contract
        self.metadata = dict(metadata, inference_decoding=inference_decoding(self.config))
        self.history = None
        if self.config.state_history_steps:
            from learning.public_history import PublicHistory, history_spec
            self.history = PublicHistory(contract, self.config.state_history_steps)
            self.metadata["public_history"] = history_spec(self.config.state_history_steps)
        self.history_mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.history_std = np.asarray(normalization["std"], dtype=np.float32)
        self.reference_codec = None
        if self.config.action_representation == "joint_reference":
            from learning.reference_actions import ReferenceActionCodec
            self.reference_codec = ReferenceActionCodec(contract)
        self.metadata["model_action_representation"] = self.config.action_representation
        self.metadata["reference_feedback"] = (
            "initial public proprioception plus policy-issued actions before hidden faults; no simulator references"
            if self.reference_codec is not None else "none")
        self.mean = torch.as_tensor(normalization["mean"], device=self.device, dtype=torch.float32)
        self.std = torch.as_tensor(normalization["std"], device=self.device, dtype=torch.float32)
        if (self.mean.shape != (23,) or self.std.shape != (23,) or not torch.isfinite(self.mean).all()
                or not torch.isfinite(self.std).all() or torch.any(self.std <= 0)):
            raise ValueError("Invalid train-only proprio normalization")
        self.reset(seed=0)

    @classmethod
    def load(cls, path, action_contract=None, *, device="auto"):
        # Plain metadata/tensors only; never load pickled model classes.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("Unsupported learned-policy checkpoint")
        contract = checkpoint["action_contract"]
        check_contract(contract, action_contract)
        config = ModelConfig(**checkpoint["model_config"])
        tokenizer = Tokenizer(checkpoint["vocabulary"], config.max_tokens)
        model = ActionDiffusion(config, len(tokenizer.words))
        model.load_state_dict(checkpoint["model_state"], strict=True)
        for tensor in model.state_dict().values():
            if not torch.isfinite(tensor).all():
                raise ValueError("Checkpoint contains nonfinite parameters")
        metadata = dict(checkpoint["metadata"], checkpoint_sha256=file_sha256(path),
                        checkpoint_path=str(Path(path).resolve()), model_config=asdict(config),
                        inference_learner_source_sha256=learner_source_sha256(),
                        weights="ema", optimizer_step=checkpoint["step"])
        return cls(model, tokenizer, checkpoint["normalization"], contract, metadata, device)

    def reset(self, seed=None):
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(0 if seed is None else int(seed))
        if self.reference_codec is not None:
            self.reference_codec.reset()
        if self.history is not None:
            self.history.reset()
        self._awaiting_action = False

    def observe_proprio(self, proprio):
        """The session calls this before every action, not just model calls."""
        if self.history is not None:
            self.history.observe_proprio(proprio)

    def observe_action(self, action):
        """Advance intended command state only for actions issued by the session."""
        if self.reference_codec is not None:
            self.reference_codec.observe_action(action)
        if self.history is not None:
            self.history.observe_action(action)
        self._awaiting_action = False

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict(self, observation):
        if (self.reference_codec is not None or self.history is not None) and self._awaiting_action:
            raise RuntimeError("Stateful policies require issued-action feedback; use PolicySession")
        images, proprio, tokens = preprocess_observation(observation, self.config, self.tokenizer, self.contract)
        image_tensor = torch.from_numpy(images[None]).to(self.device)
        proprio_tensor = (torch.from_numpy(proprio[None]).to(self.device) - self.mean) / self.std
        token_tensor = torch.from_numpy(tokens[None]).to(self.device)
        history_kwargs = {}
        if self.history is not None:
            from learning.public_history import normalize_history
            if (not self.history.awaiting_action
                    or not np.array_equal(self.history.last_proprio, proprio)):
                raise RuntimeError("History policies require per-step public observation feedback; use PolicySession")
            features = normalize_history(self.history.raw(), self.history_mean, self.history_std)
            history_kwargs["history"] = torch.from_numpy(features[None]).to(self.device)
        actions = self.model.sample(image_tensor, proprio_tensor, token_tensor, generator=self.generator, **history_kwargs)
        output = actions[0].cpu().numpy().astype(np.float32)
        if not np.isfinite(output).all():
            raise ValueError("Learned policy produced nonfinite actions")
        if self.config.gripper_mode == "binary":
            output[:, -1] = np.where(output[:, -1] >= 0, 1., -1.)
        if self.reference_codec is not None:
            output = self.reference_codec.decode_chunk(output, proprio)
        if self.reference_codec is not None or self.history is not None:
            self._awaiting_action = True
        return output


def load_policy(checkpoint_path, action_contract, *, device="auto"):
    return DiffusionPolicy.load(checkpoint_path, action_contract, device=device)


def make_policy(contract):
    """Compatibility factory for --policy learning.diffusion_policy:make_policy."""
    path = os.environ.get("FORTE_POLICY_CHECKPOINT")
    if not path:
        raise ValueError("Set FORTE_POLICY_CHECKPOINT to a trained checkpoint")
    return load_policy(path, contract, device=os.environ.get("FORTE_POLICY_DEVICE", "auto"))
