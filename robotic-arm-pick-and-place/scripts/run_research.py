"""Run frozen VLA cases, retaining synchronized demonstrations and failures."""

import argparse
from collections import defaultdict
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from controllers.benchmark_expert import BenchmarkExpert
from controllers.contact_books import ContactBookExpert
from environments.benchmark import DEFAULT_IMAGE_SIZE
from environments.bookshelf_task import BookshelfEnv
from environments.sorting import SortingEnv
from environments.bookshelf_hard import BookshelfHardEnv
from environments.perturbations import AppearancePerturbation, RecoveryFault
from utils.demonstrations import EpisodeWriter, validate_episode
from utils.policy import PolicySession
from utils.actions import LEARNING_INTERFACE
from utils.research_protocol import MANIFEST, load_manifest, manifest_digest, manifest_path

ENVIRONMENTS = {"bookshelf": BookshelfEnv, "sorting": SortingEnv, "bookshelf-hard": BookshelfHardEnv}


def make_policy(name, env, seed, *, checkpoint=None, device="auto"):
    if checkpoint is not None:
        # The loader validates units, camera shape/order, and protocol against
        # the checkpoint. No environment handle enters the learned policy.
        from learning.diffusion_policy import load_policy
        return load_policy(checkpoint, deepcopy(env.action_contract), device=device)
    if name == "expert":
        return ContactBookExpert(env) if isinstance(env, BookshelfEnv) else BenchmarkExpert(env)
    if name == "hold":
        return lambda obs: np.r_[np.zeros(len(env.robot.joint_names)), 1].astype(np.float32)
    if name == "random":
        rng = np.random.default_rng(seed)
        return lambda obs: rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
    module, factory = name.split(":")
    # External factory receives the public contract, never an environment handle.
    return getattr(importlib.import_module(module), factory)(deepcopy(env.action_contract))


class TimedPolicy:
    """Time actual predictions, excluding actions taken from a cached chunk.

    Neural adapters synchronize their device before and after prediction so GPU
    launch time is not mistaken for completed inference time. Preprocessing and
    returning the action to CPU are included; rendering and physics are not.
    """

    def __init__(self, policy):
        self.policy = policy
        self.latencies = []
        self.failed_predictions = 0

    def predict(self, observation):
        synchronize = getattr(self.policy, "synchronize", lambda: None)
        synchronize()
        started = time.perf_counter()
        try:
            result = getattr(self.policy, "predict", self.policy)(observation)
            synchronize()
        except Exception:
            self.failed_predictions += 1
            raise
        self.latencies.append(time.perf_counter() - started)
        return result

    def observe_action(self, action):
        """Forward issued-command feedback without simulator or fault state."""
        observe_action = getattr(self.policy, "observe_action", None)
        if observe_action is not None:
            observe_action(action)

    def observe_proprio(self, proprio):
        """Forward public measured state on every step, including cached steps."""
        callback = getattr(self.policy, "observe_proprio", None)
        if callback is not None:
            callback(proprio)

    def summary(self, control_dt, execute_actions):
        samples = np.asarray(self.latencies, dtype=float)
        result = dict(
            count=len(samples), failed_predictions=self.failed_predictions,
            timing_scope="prediction including preprocessing and CPU action return; excludes rendering and physics",
            device_synchronized=callable(getattr(self.policy, "synchronize", None)),
            control_step_budget_ms=1000 * control_dt,
            execution_horizon_steps=execute_actions,
        )
        if len(samples):
            result.update(mean_ms=float(samples.mean() * 1000),
                          p50_ms=float(np.percentile(samples, 50) * 1000),
                          p95_ms=float(np.percentile(samples, 95) * 1000),
                          max_ms=float(samples.max() * 1000),
                          fraction_within_control_step_budget=float(np.mean(samples <= control_dt)),
                          control_step_overruns=int(np.sum(samples > control_dt)))
        return result


def run_case(case, manifest, *, policy_name="expert", dataset_dir=None, horizon=None, rgb=False,
             image_size=DEFAULT_IMAGE_SIZE, execute_actions=LEARNING_INTERFACE["execution_horizon_steps"],
             checkpoint=None, device="auto", instruction_override=None, policy_seed=None):
    if policy_seed is not None and (type(policy_seed) is not int or policy_seed < 0):
        raise ValueError("policy_seed must be a nonnegative integer")
    reset_seed = case["seed"] if policy_seed is None else policy_seed
    if instruction_override is not None and (not isinstance(instruction_override, str) or not instruction_override.strip()):
        raise ValueError("instruction_override must be a nonempty instruction")
    if instruction_override is not None and dataset_dir is not None:
        raise ValueError("Instruction-override diagnostics cannot record ordinary demonstration files")
    effective_policy = "diffusion_checkpoint" if checkpoint is not None else policy_name
    constructor = dict(asset_set=case["asset_set"], observation_mode="rgb" if rgb or dataset_dir or checkpoint is not None else "proprio",
                       image_size=list(image_size), cameras=["front", "wrist"])
    if horizon is not None:
        constructor["horizon_steps"] = horizon
    metadata = dict(case, manifest_sha256=manifest_digest(manifest), policy=effective_policy,
                    constructor=constructor, custom_horizon=horizon is not None,
                    execution_horizon_steps=execute_actions, policy_seed=reset_seed)
    result = dict(case_id=case["case_id"], environment=case["environment"], split=case["split"],
                  axis=case["axis"], task_id=case["task_id"], seed=case["seed"], is_success=False,
                  execution_horizon_steps=execute_actions, policy=effective_policy,
                  policy_seed=reset_seed,
                  custom_horizon=horizon is not None,
                  counterfactual_diagnostic=instruction_override is not None)
    env, writer, info, policy = None, None, None, None
    timed_policy = None
    fault = None
    try:
        env = ENVIRONMENTS[case["environment"]](**constructor)
        options = dict(task_id=case["task_id"], layout=case["layout"])
        if case["environment"] == "sorting":
            options["clutter_level"] = case["clutter_level"]
        obs, info = env.reset(seed=case["seed"], options=options)
        AppearancePerturbation(env).apply(case["appearance"], case["appearance_seed"])
        obs = env._observe()
        result.update(model_fingerprint=env.fingerprint, implementation=env.implementation,
                      action_contract=env.action_contract, constructor=constructor)
        result["instruction"] = obs["instruction"]
        result["effective_instruction"] = obs["instruction"] if instruction_override is None else instruction_override
        policy = make_policy(policy_name, env, reset_seed, checkpoint=checkpoint, device=device)
        if checkpoint is not None:
            result["checkpoint"] = str(Path(checkpoint).resolve())
            result["policy_metadata"] = policy.metadata
            metadata.update(checkpoint=result["checkpoint"], policy_metadata=policy.metadata)
        # Reset the actual policy through its public lifecycle. The timed wrapper
        # deliberately does not expose simulator state or task identifiers.
        lifecycle = PolicySession(policy, env.action_space, execute_actions=execute_actions)
        lifecycle.reset(seed=reset_seed)
        timed_policy = TimedPolicy(policy)
        session = PolicySession(timed_policy, env.action_space, execute_actions=execute_actions)
        if dataset_dir:
            path = Path(dataset_dir) / (case["case_id"] + ".h5")
            writer = EpisodeWriter(path, env, obs, info, metadata=metadata)
            result["dataset"] = str(path)
        fault = RecoveryFault(env, case["recovery"])
        for _ in range(env.horizon):
            policy_observation = obs if instruction_override is None else dict(obs, instruction=instruction_override)
            # PolicySession reports the issued command before any hidden fault.
            # Only the later modified action is executed and recorded as actual.
            proposed = session(policy_observation)
            if getattr(policy, "error", None):
                raise RuntimeError(policy.error)
            stage = getattr(policy, "stage", "policy")
            action, perturbed = fault.apply(proposed)
            obs, reward, terminated, truncated, info = env.step(action)
            if writer:
                writer.append(action, obs, reward, terminated, truncated, info,
                              proposed_action=proposed,
                              source="perturbed" if perturbed else ("expert" if effective_policy == "expert" else "policy"),
                              stage=stage)
            if terminated or truncated:
                break
        result["is_success"] = bool(info["is_success"])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if policy is not None and effective_policy == "expert":
            result["expert"] = dict(stage=policy.stage,
                                   failure_stage=getattr(policy, "failure_stage", None),
                                   close_retries=policy.close_retries,
                                   lift_retries=policy.lift_retries,
                                   release_retries=policy.release_retries,
                                   planning_attempts=policy.planning_attempts,
                                   contact_preparation=getattr(policy, "contact_summary", None))
        if info is not None:
            result["info"] = info
        result["fault_events"] = [] if fault is None else fault.events
        if timed_policy is not None:
            result["prediction_latency"] = timed_policy.summary(env.control_dt, execute_actions)
        try:
            if writer:
                try:
                    writer.finish(info, error=result.get("error"), events=result["fault_events"])
                    result["dataset_validation"] = validate_episode(writer.path)
                finally:
                    writer.close()
        except Exception as exc:
            result["recording_error"] = f"{type(exc).__name__}: {exc}"
            result.setdefault("error", result["recording_error"])
        finally:
            if env:
                try:
                    env.close()
                except Exception as exc:
                    result["close_error"] = f"{type(exc).__name__}: {exc}"
                    result.setdefault("error", result["close_error"])
    return result


def summarize(results):
    groups = defaultdict(list)
    for item in results:
        groups[f"{item['environment']}/{item['split']}/{item['axis']}"].append(item)
    return {key: dict(cases=len(items), successes=sum(x["is_success"] for x in items),
                      errors=sum("error" in x for x in items),
                      timeouts=sum(x.get("info", {}).get("metrics", {}).get("timeout", False) for x in items))
            for key, items in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Defaults to the selected environment's frozen manifest")
    parser.add_argument("--env", choices=ENVIRONMENTS)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--axis")
    parser.add_argument("--layout", help="Filter frozen layout profile")
    parser.add_argument("--task")
    parser.add_argument("--asset-set")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--policy-seed", type=int, help="Policy randomness only; frozen scene/reset seed stays unchanged")
    parser.add_argument("--case-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--policy", help="expert (default), hold, random, or module:factory(contract)")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate a trained diffusion checkpoint; enables RGB automatically")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Checkpoint inference device")
    parser.add_argument("--instruction-override", help="Counterfactual diagnostic only: change policy language while retaining the original task scoring; incompatible with recording")
    parser.add_argument("--execute-actions", type=int, default=LEARNING_INTERFACE["execution_horizon_steps"],
                        help="Execute this many predictions before observing/replanning (default: 1)")
    parser.add_argument("--dataset-dir", type=Path, help="Enables synchronized front/wrist RGB recording")
    parser.add_argument("--rgb", action="store_true", help="RGB inputs without recording")
    parser.add_argument("--image-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"),
                        default=DEFAULT_IMAGE_SIZE, help="Native camera capture size (default: 480 640); model resizing belongs in the policy adapter")
    parser.add_argument("--horizon", type=int, help="Nonstandard smoke horizon, explicitly flagged")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT/"outputs/research_run.json")
    args = parser.parse_args()
    if args.policy_seed is not None and args.policy_seed < 0:
        parser.error("--policy-seed must be nonnegative")
    if args.checkpoint is not None and args.policy is not None:
        parser.error("Use either --checkpoint or --policy")
    if args.instruction_override is not None and (not args.instruction_override.strip() or args.dataset_dir is not None):
        parser.error("--instruction-override requires nonempty text and cannot be combined with --dataset-dir")
    if args.instruction_override is not None and args.checkpoint is None and args.policy in (None, "expert", "hold", "random"):
        parser.error("--instruction-override requires a checkpoint or external language-conditioned policy")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error("Checkpoint does not exist")
    policy_name = args.policy or "expert"
    manifest = load_manifest(args.manifest or manifest_path(args.env))
    filters = dict(environment=args.env, split=args.split, axis=args.axis, task_id=args.task,
                   asset_set=args.asset_set, seed=args.seed, case_id=args.case_id, layout=args.layout)
    cases = [c for c in manifest["cases"] if all(v is None or c[k] == v for k,v in filters.items())]
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        parser.error("No frozen cases match the filters")
    if args.list:
        print(json.dumps(cases, indent=2))
        return
    if args.output.exists():
        parser.error("Output exists; use a new run path to preserve results")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(manifest_sha256=manifest_digest(manifest), manifest_version=manifest["version"],
                  policy="diffusion_checkpoint" if args.checkpoint is not None else policy_name,
                  checkpoint=str(args.checkpoint.resolve()) if args.checkpoint is not None else None,
                  counterfactual_diagnostic=args.instruction_override is not None,
                  instruction_override=args.instruction_override,
                  policy_seed=args.policy_seed,
                  custom_horizon=args.horizon, requested_case_ids=[c["case_id"] for c in cases], results=[])
    for case in cases:
        result = run_case(case, manifest, policy_name=policy_name, dataset_dir=args.dataset_dir,
                          horizon=args.horizon, rgb=args.rgb, image_size=args.image_size, execute_actions=args.execute_actions,
                          checkpoint=args.checkpoint, device=args.device, instruction_override=args.instruction_override,
                          policy_seed=args.policy_seed)
        report["results"].append(result)
        report["summary"] = summarize(report["results"])
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)
        print(f"{case['case_id']} {case['task_id']} success={result['is_success']} "
              f"steps={result.get('info',{}).get('episode',{}).get('steps',0)} {result.get('error','')[:350]}", flush=True)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
