"""Frozen-parent DDIM behavior preservation on public train observations.

These targets are parent denoiser predictions, never expert action labels.
The trace follows ActionDiffusion.sample, including clipped-x0 noise correction.
"""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math

import torch


def preservation_options(weight, prefix_steps, stride, noise_seed_count, *, init_checkpoint,
                         allow_interventions, prediction_type, action_representation):
    """Keep existing training unchanged unless the complete opt-in is supplied."""
    if weight is None:
        if prefix_steps is not None or stride != 8 or noise_seed_count != 2:
            raise ValueError("Preservation options require --preservation-weight")
        return False
    if isinstance(weight, bool) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("--preservation-weight must be finite and positive")
    if any(type(value) is not int or value < 1 for value in (prefix_steps, stride, noise_seed_count)):
        raise ValueError("Preservation prefix steps, stride and noise-seed count must be positive integers")
    if init_checkpoint is None or not allow_interventions:
        raise ValueError("Preservation requires --init-checkpoint and --allow-interventions")
    if prediction_type != "epsilon" or action_representation != "velocity":
        raise ValueError("Preservation currently requires epsilon prediction and velocity actions")
    return True


@torch.no_grad()
def sampling_trace(model, images, proprio, tokens, *, generator=None, initial_noise=None):
    """Return actual x_t/epsilon/timestep traces and the unchanged DDIM sample."""
    if model.config.prediction_type != "epsilon" or model.config.action_representation != "velocity":
        raise ValueError("Parent sampling traces require epsilon/velocity configuration")
    if (generator is None) == (initial_noise is None):
        raise ValueError("Supply exactly one initial-noise tensor or generator")
    shape = (len(images), model.config.horizon, model.config.action_dim)
    x = (torch.randn(shape, device=images.device, generator=generator) if initial_noise is None
         else initial_noise.clone())
    if x.shape != shape or x.device != images.device or not torch.isfinite(x).all():
        raise ValueError("Invalid initial-noise shape, device or values")
    condition = model.encode_condition(images, proprio, tokens)
    schedule = torch.linspace(model.config.diffusion_steps - 1, 0, model.config.inference_steps).round().long().tolist()
    states, targets = [], []
    for index, step in enumerate(schedule):
        timestep = torch.full((len(x),), step, device=x.device, dtype=torch.long)
        prediction = model.predict_target(x, timestep, condition)
        states.append(x.clone())
        targets.append(prediction.clone())
        alpha = model.alpha_bar[step]
        clean = ((x - (1 - alpha).sqrt() * prediction) / alpha.sqrt()).clamp(-1, 1)
        if index + 1 == len(schedule):
            x = clean
        else:
            noise = (x - alpha.sqrt() * clean) / (1 - alpha).sqrt()
            previous = model.alpha_bar[schedule[index + 1]]
            x = previous.sqrt() * clean + (1 - previous).sqrt() * noise
    return dict(noisy=torch.stack(states, dim=1), targets=torch.stack(targets, dim=1),
                timesteps=torch.tensor(schedule, dtype=torch.long), sample=x)


def tensor_digest(tensors, metadata=None):
    digest = hashlib.sha256()
    if metadata is not None:
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class PreservationCache:
    """Batched frozen-teacher traces with an independent training sampler RNG."""

    def __init__(self, teacher, dataset, *, noise_seeds, batch_size=64, device):
        if not len(dataset) or type(batch_size) is not int or batch_size < 1:
            raise ValueError("Preservation requires nonempty observations and positive batch size")
        if (not noise_seeds or any(type(seed) is not int or seed < 0 for seed in noise_seeds)
                or len(set(noise_seeds)) != len(noise_seeds)):
            raise ValueError("Preservation noise seeds must be distinct nonnegative integers")
        self.dataset = dataset
        self.config = teacher.config
        teacher.to(device).eval().requires_grad_(False)
        if self.config.prediction_type != "epsilon" or self.config.action_representation != "velocity":
            raise ValueError("Preservation currently requires epsilon/velocity configuration")
        count, draws, steps = len(dataset), len(noise_seeds), self.config.inference_steps
        shape = (count, draws, steps, self.config.horizon, self.config.action_dim)
        self.noisy = torch.empty(shape, dtype=torch.float32)
        self.targets = torch.empty_like(self.noisy)
        self.samples = torch.empty((count, draws, self.config.horizon, self.config.action_dim))
        for draw, seed in enumerate(noise_seeds):
            generator = torch.Generator(device=device).manual_seed(seed)
            # Drawing once per seed makes noise independent of cache batch size.
            initial = torch.randn((count, self.config.horizon, self.config.action_dim),
                                  device=device, generator=generator)
            for start in range(0, count, batch_size):
                stop = min(start + batch_size, count)
                batch = self.observations(range(start, stop), device)
                trace = sampling_trace(teacher, *batch, initial_noise=initial[start:stop])
                self.noisy[start:stop, draw].copy_(trace["noisy"].cpu())
                self.targets[start:stop, draw].copy_(trace["targets"].cpu())
                self.samples[start:stop, draw].copy_(trace["sample"].cpu())
                self.timesteps = trace["timesteps"]
        if not torch.isfinite(self.noisy).all() or not torch.isfinite(self.targets).all():
            raise ValueError("Nonfinite parent preservation trace")
        self.metadata = dict(
            version="forte-parent-ddim-preservation-v1", targets="frozen parent epsilon predictions; not expert labels",
            selection=deepcopy(dataset.metadata), model_config=asdict(self.config), noise_seeds=list(noise_seeds),
            schedule=self.timesteps.tolist(), observations=count, trace_targets=count * draws * steps,
            sampling="uniform observation, noise seed and inference timestep with replacement",
            parent_model_state_sha256=tensor_digest(teacher.state_dict()),
            batch_size=batch_size, device=str(device),
        )
        self.metadata["cache_sha256"] = tensor_digest(
            dict(noisy=self.noisy, targets=self.targets, timesteps=self.timesteps, samples=self.samples), self.metadata)
        self.cache_bytes = sum(value.numel() * value.element_size()
                               for value in (self.noisy, self.targets, self.samples, self.timesteps))

    def observations(self, indices, device):
        examples = [self.dataset[int(index)] for index in indices]
        if any(len(example) != 3 for example in examples):
            raise ValueError("Preservation observations must contain only images, proprioception and tokens")
        return [torch.stack(values).to(device) for values in zip(*examples)]

    def loss(self, student, batch_size, *, generator):
        if student.config != self.config:
            raise ValueError("Preservation student/parent ModelConfig mismatch")
        device = next(student.parameters()).device
        count, draws, steps = self.noisy.shape[:3]
        indices = torch.randint(count * draws * steps, (batch_size,), generator=generator)
        observation = indices // (draws * steps)
        draw = (indices // steps) % draws
        step = indices % steps
        images, proprio, tokens = self.observations(observation.tolist(), device)
        condition = student.encode_condition(images, proprio, tokens)
        prediction = student.predict_target(self.noisy[observation, draw, step].to(device),
                                            self.timesteps[step].to(device), condition)
        return (prediction - self.targets[observation, draw, step].to(device)).square().mean()
