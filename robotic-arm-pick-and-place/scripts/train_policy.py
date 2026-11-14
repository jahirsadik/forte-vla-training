"""Train the compact text-conditioned diffusion pilot on frozen HDF5 episodes."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import glob
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from learning.data import CompactChunkDataset, check_data_separation, inspect_episodes
from learning.diffusion_policy import (ActionDiffusion, CHECKPOINT_SCHEMA, DiffusionPolicy, ModelConfig, Tokenizer,
                                      file_sha256, inference_decoding, learner_source_sha256, resolve_device,
                                      validate_gripper_targets)
from learning.preservation import PreservationCache, preservation_options
from learning.preservation_data import PreservationDataset


def expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise ValueError(f"No episodes match {pattern}")
        paths.extend(matches)
    return paths


def transfer(batch, device):
    return [value.to(device, non_blocking=device.type == "cuda") for value in batch]


def correction_fraction(allow_interventions, requested, action_representation):
    """Resolve the opt-in sampler without silently changing ordinary training."""
    if not allow_interventions:
        if requested is not None:
            raise ValueError("--correction-fraction requires --allow-interventions")
        return None
    if action_representation != "velocity":
        raise ValueError("Intervention training currently supports velocity targets only")
    fraction = .25 if requested is None else requested
    if isinstance(fraction, bool) or not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Correction fraction must be finite and strictly between zero and one")
    return float(fraction)


def training_sampler(dataset, fraction, *, seed, instruction_balanced=False):
    """Allocate probability mass by label group, not by episode/prefix length."""
    if not isinstance(instruction_balanced, bool):
        raise ValueError("instruction_balanced must be a boolean")
    if instruction_balanced:
        if fraction is not None:
            raise ValueError("Instruction balancing cannot be combined with correction-group weighting")
        if dataset.split != "train":
            raise ValueError("Instruction balancing requires training data")
        indices = np.asarray(dataset.episode_index[:len(dataset)])
        if (indices.shape != (len(dataset),) or indices.dtype.kind not in "iu"
                or not len(indices) or np.any(indices < 0) or np.any(indices >= len(dataset.records))):
            raise ValueError("Invalid episode membership for instruction balancing")
        instructions = [record["instruction"] for record in dataset.records]
        if any(not isinstance(value, str) or not value.strip() for value in instructions):
            raise ValueError("Instruction balancing requires explicit nonempty language")
        labels = np.asarray([instructions[index] for index in indices])
        groups, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        weights = 1. / (len(groups) * counts[inverse])
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.float64), len(dataset),
                                        replacement=True, generator=torch.Generator().manual_seed(seed))
        return sampler, dict(kind="instruction_balanced_anchors", replacement=True,
                             draws_per_epoch=len(dataset), seed=seed,
                             instructions=[dict(instruction=str(label), anchors=int(count),
                                                expected_probability=1. / len(groups))
                                           for label, count in zip(groups, counts, strict=True)])
    if fraction is None:
        return None, dict(kind="uniform_shuffled_anchors", anchors=len(dataset))
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Correction fraction must lie strictly between zero and one")
    flags = np.asarray(dataset.is_intervention[:len(dataset)])
    if flags.shape != (len(dataset),) or flags.dtype != np.bool_:
        raise ValueError("Dataset must identify each supervised intervention anchor")
    corrections = int(flags.sum())
    nominal = len(flags) - corrections
    if not corrections or not nominal:
        raise ValueError("Intervention sampling requires both nominal expert and intervention anchors")
    weights = np.where(flags, fraction / corrections, (1 - fraction) / nominal)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.float64), len(dataset),
                                    replacement=True, generator=torch.Generator().manual_seed(seed))
    return sampler, dict(kind="weighted_supervised_anchors", replacement=True,
                         draws_per_epoch=len(dataset), nominal_anchors=nominal,
                         intervention_anchors=corrections,
                         expected_intervention_probability=fraction,
                         expected_nominal_probability=1 - fraction, seed=seed)


def _parent_records(metadata, role):
    """Retain the data ancestry needed to detect leakage on later warm starts."""
    key = "training_episodes" if role == "train" else "validation_episodes"
    records = metadata.get(key, [])
    ancestors = metadata.get("initialization", {}).get("ancestral_" + key, [])
    if not isinstance(records, list) or not isinstance(ancestors, list):
        raise ValueError("Parent checkpoint lacks inspectable data provenance")
    result, seen = [], set()
    for record in records + ancestors:
        if (not isinstance(record, dict) or record.get("split") != role
                or any(not isinstance(record.get(k), str) or not record[k]
                       for k in ("path", "sha256", "case_id", "family_id", "instruction"))):
            raise ValueError(f"Parent checkpoint contains invalid or non-{role} data provenance")
        identity = (record["sha256"], record["case_id"])
        if identity not in seen:
            result.append(deepcopy(record))
            seen.add(identity)
    if role == "train" and not result:
        raise ValueError("Warm start requires declared parent training episodes")
    return result


def _reject_overlap(left, right, explanation):
    for key in ("path", "sha256", "case_id", "family_id"):
        if {r[key] for r in left} & {r[key] for r in right}:
            raise ValueError(f"{explanation}: overlap in {key}")


def load_initialization(path, config, train_records, validation_records, *, device):
    """Load parent EMA weights and its fitted preprocessing; never resume Adam."""
    path = Path(path).resolve()
    digest = file_sha256(path)
    parent = torch.load(path, map_location="cpu", weights_only=True)
    if parent.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Unsupported initialization checkpoint schema")
    if ModelConfig(**parent["model_config"]) != config:
        raise ValueError("Initialization checkpoint ModelConfig must exactly match requested training")
    if parent["action_contract"] != train_records[0]["action_contract"]:
        raise ValueError("Initialization checkpoint action/observation contract mismatch")
    metadata = parent.get("metadata", {})
    parent_train = _parent_records(metadata, "train")
    parent_validation = _parent_records(metadata, "validation")
    _reject_overlap(parent_train, parent_validation, "Parent train/validation leakage")
    _reject_overlap(parent_train, validation_records, "Validation overlaps parent training ancestry")
    _reject_overlap(parent_validation, train_records, "Training overlaps parent validation ancestry")
    normalization = deepcopy(parent["normalization"])
    if (normalization.get("split") != "train" or isinstance(normalization.get("count"), bool)
            or not isinstance(normalization.get("count"), int) or normalization["count"] < 1):
        raise ValueError("Warm start requires parent train-only normalization with a positive sample count")
    policy = DiffusionPolicy.load(path, train_records[0]["action_contract"], device=device)
    if policy.metadata["checkpoint_sha256"] != digest:
        raise ValueError("Initialization checkpoint changed while being loaded")
    expected_words = Tokenizer.fit([r["instruction"] for r in parent_train], config.max_tokens).words
    if policy.tokenizer.words != expected_words:
        raise ValueError("Parent vocabulary is not explained by its training ancestry")
    unknown = sorted({word for record in train_records
                      for word in Tokenizer.tokens(record["instruction"])
                      if word not in policy.tokenizer.lookup})
    if unknown:
        raise ValueError("Warm start cannot extend the frozen parent vocabulary: " + ", ".join(unknown))
    initialization = dict(
        kind="warm_start", checkpoint=str(path), checkpoint_sha256=digest,
        parent_step=parent["step"], parent_learner_source_sha256=metadata.get("learner_source_sha256"),
        parent_collection_source_sha256=metadata.get("training_source_sha256"),
        weights="parent EMA model_state", optimizer="fresh AdamW; parent optimizer state not loaded",
        vocabulary="frozen parent training vocabulary", normalization="frozen parent train-only statistics",
        ancestral_training_episodes=parent_train, ancestral_validation_episodes=parent_validation,
    )
    return policy.model, policy.tokenizer, normalization, initialization


def validate_gripper_records(records, gripper_mode):
    """Check every eligible label before caching/subsampling, in either split."""
    if gripper_mode != "binary":
        return
    for record in records:
        with h5py.File(record["path"], "r") as episode:
            labeled = episode["transitions/expert_action_valid"][:]
            valid = labeled | ((episode["transitions/source"].asstr()[:] == "expert") & bool(episode.attrs["success"]))
            targets = np.where(labeled[:, None], episode["transitions/expert_action"][:], episode["transitions/action"][:])
            try:
                validate_gripper_targets(targets, valid, gripper_mode)
            except ValueError as exc:
                raise ValueError(f"{record['path']}: {exc}") from exc


@torch.inference_mode()
def validation_loss(model, loader, device, seed):
    """Fixed diffusion noise for comparable validation measurements; no test data."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total, count = 0., 0
    for batch in loader:
        batch = transfer(batch, device)
        valid = int(batch[4].sum()) * model.config.action_dim
        loss = model.training_loss(*batch, generator=generator)
        total += float(loss) * valid
        count += valid
    return total / count


def save_checkpoint(path, model, ema, optimizer, config, tokenizer, normalization, contract, metadata, step):
    checkpoint = dict(schema=CHECKPOINT_SCHEMA, model_config=asdict(config), vocabulary=tokenizer.words,
                      normalization=normalization, action_contract=contract, metadata=metadata, step=step,
                      model_state={key: tensor.detach().cpu() for key, tensor in ema.state_dict().items()},
                      training_model_state={key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
                      optimizer_state=optimizer.state_dict())
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", required=True, help="Train-split HDF5 paths or quoted globs")
    parser.add_argument("--validation", nargs="+", default=[], help="Validation-split HDF5 paths or quoted globs; never test")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing outputs are never overwritten")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--instruction-balanced", action="store_true",
                        help="Equal sampling mass per training instruction, independent of trajectory length; incompatible with correction weighting")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--image-size", nargs=2, type=int, default=(96, 96), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--inference-steps", type=int, default=16)
    parser.add_argument("--text-pooling", choices=("last", "masked_mean"), default="last",
                        help="Pool the final nonpadding GRU state (legacy default) or the mean of all nonpadding GRU states")
    parser.add_argument("--state-history-steps", type=int, choices=range(33), default=0,
                        help="Opt-in causal proprioception/issued-command history at control rate; 0 preserves current-frame behavior")
    parser.add_argument("--prediction-type", choices=("epsilon", "sample"), default="epsilon",
                        help="Denoise Gaussian noise (epsilon) or directly predict clean action chunks (sample)")
    parser.add_argument("--gripper-mode", choices=("continuous", "binary"), default="continuous",
                        help="Inference jaw decoding; binary requires every supervised jaw label to be -1 or +1")
    parser.add_argument("--action-representation", choices=("velocity", "joint_reference"), default="velocity",
                        help="Model arm targets: public normalized reference velocities, or absolute references normalized by joint limits")
    parser.add_argument("--allow-interventions", action="store_true",
                        help="Admit validated successful expert-takeover suffixes in TRAIN only")
    parser.add_argument("--correction-fraction", type=float,
                        help="Expected sampler mass for intervention anchors; defaults to 0.25 with --allow-interventions")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Warm start parent EMA weights with frozen vocabulary/statistics and a fresh optimizer")
    parser.add_argument("--preservation-weight", type=float,
                        help="Opt-in positive weight for frozen-parent denoising behavior on early public observations")
    parser.add_argument("--preserve-prefix-steps", type=int,
                        help="Exclusive learner-prefix cutoff for parent behavior preservation; never expert labels")
    parser.add_argument("--preservation-stride", type=int, default=8)
    parser.add_argument("--preservation-noise-seeds", type=int, default=2,
                        help="Number of independent fixed-noise parent DDIM traces per preserved observation")
    parser.add_argument("--anchor-stride", type=int, default=1, help="Cache every Nth valid anchor, independently per episode")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--ema-decay", type=float, default=.995)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.threads, args.anchor_stride, args.log_every, args.eval_every) < 1:
        parser.error("Steps, batch size, threads, stride and logging intervals must be positive")
    if not 0 < args.learning_rate or not 0 <= args.ema_decay < 1:
        parser.error("Invalid learning rate or EMA decay")
    try:
        fraction = correction_fraction(args.allow_interventions, args.correction_fraction, args.action_representation)
        preserve = preservation_options(
            args.preservation_weight, args.preserve_prefix_steps, args.preservation_stride,
            args.preservation_noise_seeds, init_checkpoint=args.init_checkpoint,
            allow_interventions=args.allow_interventions, prediction_type=args.prediction_type,
            action_representation=args.action_representation)
    except ValueError as exc:
        parser.error(str(exc))
    if args.instruction_balanced and fraction is not None:
        parser.error("--instruction-balanced cannot be combined with correction-group weighting")
    if args.state_history_steps and preserve:
        parser.error("History-aware behavior preservation requires a separate causal prefix cache and is not supported")
    if args.output.exists():
        parser.error("Output exists; choose a new directory to preserve the experiment")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    # Stable ordering/algorithms on a fixed runtime; cross-device equality is not promised.
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    train_records, skipped_train = inspect_episodes(expand_paths(args.train), "train",
                                                   allow_interventions=args.allow_interventions)
    validation_records, skipped_validation = inspect_episodes(expand_paths(args.validation), "validation") if args.validation else ([], [])
    check_data_separation(train_records, validation_records)
    config = ModelConfig(image_height=args.image_size[0], image_width=args.image_size[1], hidden_dim=args.hidden_dim,
                         diffusion_steps=args.diffusion_steps, inference_steps=args.inference_steps,
                         prediction_type=args.prediction_type, gripper_mode=args.gripper_mode,
                         action_representation=args.action_representation, text_pooling=args.text_pooling,
                         state_history_steps=args.state_history_steps)
    validate_gripper_records(train_records, config.gripper_mode)
    validate_gripper_records(validation_records, config.gripper_mode)
    loss_metric = "noise_mse" if config.prediction_type == "epsilon" else "clean_action_mse"
    objective = "masked epsilon prediction" if config.prediction_type == "epsilon" else "masked clean-action prediction"
    model_targets = dict(
        action_representation=config.action_representation,
        arm=("absolute arm reference positions normalized by the public operating joint bounds: "
             "2 * (q_ref - lower) / (upper - lower) - 1"
             if config.action_representation == "joint_reference"
             else "existing normalized joint-reference velocities in [-1, 1]"),
        jaw="existing normalized absolute nominal jaw width unchanged",
        public_action_contract_id=train_records[0]["action_contract"]["id"],
    )
    if config.action_representation == "joint_reference":
        model_targets.update(
            inference_translation="absolute references are translated back to the same public rate-limited velocity actions using policy-issued-command history",
            decoder_feedback="policy-issued public commands before environment-hidden action faults",
            data_eligibility="fault and correction streams are unsupported and rejected by the reference-target dataset",
        )
    model = None
    normalization = None
    initialization = dict(kind="from_scratch", optimizer="fresh AdamW",
                          vocabulary="fit on admitted training instructions",
                          normalization="fit on admitted training anchors")
    if args.init_checkpoint is not None:
        model, tokenizer, normalization, initialization = load_initialization(
            args.init_checkpoint, config, train_records, validation_records, device=device)
    else:
        tokenizer = Tokenizer.fit([record["instruction"] for record in train_records], config.max_tokens)
    print(f"Caching {len(train_records)} train and {len(validation_records)} validation episodes at {args.image_size}", flush=True)
    train = CompactChunkDataset(train_records, config, tokenizer, split="train", anchor_stride=args.anchor_stride)
    if normalization is None:
        normalization = train.normalization()
    train.set_normalization(normalization)
    validation = None
    if validation_records:
        validation = CompactChunkDataset(validation_records, config, tokenizer, split="validation", anchor_stride=args.anchor_stride)
        validation.set_normalization(normalization)
    generator = torch.Generator().manual_seed(args.seed)
    sampler, sampling = training_sampler(train, fraction, seed=args.seed + 1,
                                        instruction_balanced=args.instruction_balanced)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=sampler is None,
                              sampler=sampler, generator=generator,
                              num_workers=0, pin_memory=device.type == "cuda", drop_last=False)
    validation_loader = None if validation is None else DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                                                                   num_workers=0, pin_memory=device.type == "cuda")
    if model is None:
        model = ActionDiffusion(config, len(tokenizer.words)).to(device)
    preservation = None
    preservation_generator = None
    if preserve:
        observations = PreservationDataset(
            [record for record in train_records if record["is_intervention"]], config, tokenizer, normalization,
            prefix_steps=args.preserve_prefix_steps, anchor_stride=args.preservation_stride,
            parent_checkpoint_sha256=initialization["checkpoint_sha256"])
        print(f"Caching frozen-parent DDIM traces for {len(observations)} public prefix observations", flush=True)
        teacher = deepcopy(model)
        preservation = PreservationCache(
            teacher, observations, noise_seeds=[args.seed + 200000 + index for index in range(args.preservation_noise_seeds)],
            batch_size=args.batch_size, device=device)
        del teacher
        preservation_generator = torch.Generator().manual_seed(args.seed + 300000)
    ema = deepcopy(model).eval()
    ema.text.flatten_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    metadata = dict(learner_source_sha256=learner_source_sha256(), training_source_sha256=train_records[0]["collection_implementation"]["source_sha256"],
                    training_episodes=train_records, validation_episodes=validation_records,
                    skipped_train=skipped_train, skipped_validation=skipped_validation,
                    initialization=initialization, sampling=sampling,
                    inference_decoding=inference_decoding(config),
                    model_targets=model_targets,
                    preprocessing=dict(resize_hw=args.image_size, resize="PIL bilinear", channel_order="camera,RGB,H,W", pixel_scale="uint8 / 255",
                                       observation_history_steps=max(1, config.state_history_steps), rgb_history_steps=1),
                    training=dict(seed=args.seed, steps=args.steps, batch_size=args.batch_size, learning_rate=args.learning_rate,
                                  ema_decay=args.ema_decay, anchor_stride=args.anchor_stride, optimizer="AdamW", gradient_clip_norm=1.,
                                  objective=objective, prediction_type=config.prediction_type, loss_metric=loss_metric,
                                  action_representation=config.action_representation,
                                  text_pooling=config.text_pooling,
                                  state_history_steps=config.state_history_steps,
                                  allow_interventions=args.allow_interventions,
                                  checkpoint_selection="minimum fixed-noise validation loss" if validation is not None else "final step; no validation data"),
                    runtime=dict(torch=str(torch.__version__), numpy=np.__version__, device=str(device), threads=args.threads),
                    samples=dict(train=len(train), validation=0 if validation is None else len(validation)),
                    model_parameters=sum(p.numel() for p in model.parameters()),
                    scope=(("public-state-history, " if config.state_history_steps else "current-frame, ") + ("warm-start" if args.init_checkpoint else "from-scratch")
                           + " engineering baseline; no generalization or hardware claim"))
    if config.state_history_steps:
        from learning.public_history import history_spec
        metadata["public_history"] = history_spec(config.state_history_steps)
    if preservation is not None:
        metadata["preservation"] = dict(preservation.metadata, weight=args.preservation_weight,
                                         sampler_seed=args.seed + 300000,
                                         normalization="frozen parent training statistics",
                                         vocabulary="frozen parent training vocabulary")
        metadata["training"]["objective"] += " + weighted frozen-parent epsilon preservation"
    args.output.mkdir(parents=True)
    (args.output / "experiment.json").write_text(json.dumps(dict(metadata, model_config=asdict(config), normalization=normalization, vocabulary=tokenizer.words), indent=2) + "\n")
    print(json.dumps(dict(device=str(device), samples=metadata["samples"], model_parameters=metadata["model_parameters"],
                          cache_mib=(train.cache_bytes + (0 if validation is None else validation.cache_bytes)) / 2**20)), flush=True)
    iterator, losses, best = iter(train_loader), [], float("inf")
    preservation_losses, total_losses = [], []
    start = time.perf_counter()
    with (args.output / "metrics.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            model.train()
            expert_loss = model.training_loss(*transfer(batch, device))
            loss = expert_loss
            if preservation is not None:
                preservation_loss = preservation.loss(model, len(batch[0]), generator=preservation_generator)
                loss = loss + args.preservation_weight * preservation_loss
                preservation_losses.append(float(preservation_loss.detach()))
                total_losses.append(float(loss.detach()))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for averaged, current in zip(ema.parameters(), model.parameters()):
                    averaged.lerp_(current, 1 - args.ema_decay)
            losses.append(float(expert_loss.detach()))
            record = None
            if step % args.log_every == 0 or step == 1 or step == args.steps:
                record = dict(step=step, **{"train_" + loss_metric: float(np.mean(losses))}, gradient_norm=float(gradient_norm),
                              elapsed_seconds=time.perf_counter() - start)
                if preservation is not None:
                    record.update(train_preservation_epsilon_mse=float(np.mean(preservation_losses)),
                                  train_total_loss=float(np.mean(total_losses)),
                                  preservation_weight=args.preservation_weight)
                    preservation_losses.clear()
                    total_losses.clear()
                losses.clear()
            if step % args.eval_every == 0 or step == args.steps:
                record = record or dict(step=step, elapsed_seconds=time.perf_counter() - start)
                if validation_loader is not None:
                    measured = validation_loss(ema, validation_loader, device, args.seed + 100000)
                    record["validation_" + loss_metric] = measured
                    if measured < best:
                        best = measured
                        save_checkpoint(args.output / "best.pt", model, ema, optimizer, config, tokenizer, normalization,
                                        train_records[0]["action_contract"], metadata, step)
                save_checkpoint(args.output / "last.pt", model, ema, optimizer, config, tokenizer, normalization,
                                train_records[0]["action_contract"], metadata, step)
            if record is not None:
                line = json.dumps(record)
                log.write(line + "\n")
                log.flush()
                print(line, flush=True)
    summary = dict(status="complete", steps=args.steps, elapsed_seconds=time.perf_counter() - start,
                   **{"best_validation_" + loss_metric: None if validation is None else best},
                   prediction_type=config.prediction_type,
                   selected_checkpoint="last.pt" if validation is None else "best.pt",
                   closed_loop_evaluation="pending; supervised loss is not task success")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
