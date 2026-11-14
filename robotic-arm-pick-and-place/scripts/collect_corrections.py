"""Collect train-only learner prefixes followed by physical expert interventions.

Interventions are explicitly assisted rollouts, never autonomous policy scores.
Only executed expert suffix actions receive expert labels. All outcomes persist.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from learning.diffusion_policy import file_sha256, load_policy
from learning.data import intervention_rollout_id, INTERVENTION_VERSION_BY_EXPERT
from utils.contracts import implementation_provenance
from utils.demonstrations import EpisodeWriter, validate_episode
from utils.policy import PolicySession
from utils.research_protocol import load_manifest, manifest_digest


def rollout_expert(plan, rollout):
    version = plan.get("version")
    fields = {"policy_seed", "minimum_policy_steps"}
    if version == "forte-correction-pilot-v2":
        fields.add("expert")
    elif version != "forte-release-correction-pilot-v1":
        raise ValueError("Unsupported correction collection plan")
    if not isinstance(rollout, dict) or set(rollout) != fields:
        raise ValueError("Unexpected rollout fields")
    expert = rollout.get("expert", "supported_goal_release_v1")
    if not isinstance(expert, str) or expert not in INTERVENTION_VERSION_BY_EXPERT:
        raise ValueError("Unsupported correction expert")
    return expert


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    if not isinstance(plan, dict) or plan.get("version") not in (
            "forte-release-correction-pilot-v1", "forte-correction-pilot-v2"):
        raise ValueError("Unsupported correction collection plan")
    manifest = load_manifest()
    cases = [c for c in manifest["cases"] if c["case_id"] == plan.get("case_id")]
    if len(cases) != 1:
        raise ValueError("Plan must name one existing frozen case")
    case = cases[0]
    if (case["split"] != "train" or case["environment"] != "sorting"
            or case["appearance"] != "nominal" or case["recovery"] != "none"):
        raise ValueError("Correction pilot requires a nominal sorting train case")
    for key in ("execution_horizon_steps", "confirmation_steps"):
        if type(plan.get(key)) is not int or plan[key] < 1:
            raise ValueError(f"Invalid {key}")
    if not isinstance(plan.get("rollouts"), list) or not 1 <= len(plan["rollouts"]) <= 32:
        raise ValueError("Declare between one and 32 correction rollouts")
    identities = set()
    for rollout in plan["rollouts"]:
        expert = rollout_expert(plan, rollout)
        if (type(rollout["policy_seed"]) is not int or rollout["policy_seed"] < 0
                or type(rollout["minimum_policy_steps"]) is not int
                or rollout["minimum_policy_steps"] < 1):
            raise ValueError("Invalid policy seed or minimum prefix length")
        key = (expert, rollout["policy_seed"], rollout["minimum_policy_steps"])
        if key in identities:
            raise ValueError("Duplicate declared rollout")
        identities.add(key)
    return plan, case, manifest


def descriptor(plan, rollout, checkpoint_sha256):
    expert = rollout_expert(plan, rollout)
    return dict(version=INTERVENTION_VERSION_BY_EXPERT[expert], expert=expert,
                parent_checkpoint_sha256=checkpoint_sha256, policy_seed=rollout["policy_seed"],
                minimum_policy_steps=rollout["minimum_policy_steps"],
                confirmation_steps=plan["confirmation_steps"], actual_takeover_step=None)


def make_correction_expert(expert, env):
    if expert == "supported_goal_release_v1":
        from controllers.placement_correction import PlacementReleaseExpert
        return PlacementReleaseExpert(env)
    if expert == "grasped_carry_recovery_v1":
        from controllers.carry_correction import CarryRecoveryExpert
        return CarryRecoveryExpert(env)
    if expert == "sustained_grasp_to_goal_v1":
        from controllers.grasp_correction import GraspToGoalExpert
        return GraspToGoalExpert(env)
    raise ValueError("Unsupported correction expert")


def collect_one(plan, case, manifest, checkpoint, output, index, device):
    from controllers.benchmark_expert import BenchmarkExpert
    from environments.sorting import SortingEnv
    from environments.perturbations import AppearancePerturbation
    checkpoint_hash = file_sha256(checkpoint)
    nominal = index == -1
    correction = None if nominal else descriptor(plan, plan["rollouts"][index], checkpoint_hash)
    rollout_id = ("expert_" + case["case_id"] if nominal
                  else intervention_rollout_id(case["case_id"], correction))
    constructor = dict(asset_set=case["asset_set"], observation_mode="rgb",
                       image_size=[480, 640], cameras=["front", "wrist"])
    metadata = dict(case, manifest_sha256=manifest_digest(manifest), constructor=constructor,
                    policy="expert" if nominal else "expert_intervention", custom_horizon=False,
                    execution_horizon_steps=1 if nominal else plan["execution_horizon_steps"],
                    collection_plan_sha256=plan["collection_plan_sha256"])
    if not nominal:
        metadata.update(rollout_id=rollout_id, intervention=correction,
                        parent_checkpoint=str(checkpoint.resolve()))
    result = dict(case_id=case["case_id"], rollout_id=rollout_id, nominal_expert=nominal,
                  assisted=not nominal, is_success=False, metadata=metadata,
                  implementation=implementation_provenance())
    path = output / "episodes" / (rollout_id + ".h5")
    env = writer = teacher = None
    events, eligible_steps = [], 0
    teacher_active, executed_expert_steps = False, 0
    pending_takeover = None
    info = None
    try:
        env = SortingEnv(**constructor)
        obs, info = env.reset(seed=case["seed"], options=dict(task_id=case["task_id"],
                              layout=case["layout"], clutter_level=case["clutter_level"]))
        AppearancePerturbation(env).apply(case["appearance"], case["appearance_seed"])
        obs = env._observe()
        if nominal:
            policy = BenchmarkExpert(env)
            session = PolicySession(policy, env.action_space, execute_actions=1)
            session.reset(seed=case["seed"])
        else:
            policy = load_policy(checkpoint, deepcopy(env.action_contract), device=device)
            session = PolicySession(policy, env.action_space, execute_actions=plan["execution_horizon_steps"])
            session.reset(seed=correction["policy_seed"])
            teacher = make_correction_expert(correction["expert"], env)
            metadata["policy_metadata"] = policy.metadata
            if correction["minimum_policy_steps"] >= env.horizon:
                raise ValueError("Minimum policy prefix exhausts the standard episode horizon")
            # A carry-stall detector needs history from reset, including earlier
            # goal entry. This state belongs only to the privileged collector.
            observe = getattr(teacher, "observe", None)
            if observe is not None:
                observe(deepcopy(info))
        writer = EpisodeWriter(path, env, obs, info, metadata=metadata)
        for step in range(env.horizon):
            if not nominal and not teacher_active:
                candidate = step >= correction["minimum_policy_steps"] and teacher.eligible()
                if candidate:
                    candidate = getattr(teacher, "should_intervene", lambda: True)()
                eligible_steps = eligible_steps + 1 if candidate else 0
                if eligible_steps >= correction["confirmation_steps"]:
                    discarded = len(session.pending)
                    session.pending.clear()
                    session = PolicySession(teacher, env.action_space, execute_actions=1)
                    session.reset(seed=correction["policy_seed"])
                    teacher_active = True
                    pending_takeover = dict(kind="expert_takeover", step=step,
                                            discarded_learner_actions=discarded,
                                            precondition_metrics=deepcopy(info["metrics"]))
            intervening = not nominal and teacher_active
            controller = teacher if intervening else policy
            action = session(obs)
            if getattr(controller, "error", None):
                raise RuntimeError(controller.error)
            obs, reward, terminated, truncated, info = env.step(action)
            executed_expert_steps += int(intervening)
            writer.append(action, obs, reward, terminated, truncated, info,
                          source="expert" if nominal or intervening else "policy",
                          intervention=intervening, expert_action=action if intervening else None,
                          proposed_action=action, stage=getattr(controller, "stage", "policy"))
            if pending_takeover is not None:
                # Commit a suffix boundary only after the first expert command
                # was applied and recorded; failed planning is not an intervention.
                correction["actual_takeover_step"] = step
                metadata["intervention"] = dict(correction)
                writer.file.attrs["metadata"] = json.dumps(metadata, sort_keys=True)
                events.append(pending_takeover)
                pending_takeover = None
            if teacher is not None:
                observe = getattr(teacher, "observe", None)
                if observe is not None:
                    observe(deepcopy(info))
            if terminated or truncated:
                break
        result["is_success"] = bool(info["is_success"])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result.update(info=info, events=events, metadata=metadata)
        result["assisted"] = executed_expert_steps > 0
        result["expert_actions_executed"] = executed_expert_steps
        if teacher is not None:
            result["correction_expert"] = dict(stage=teacher.stage, error=teacher.error)
            if hasattr(teacher, "diagnostics"):
                result["correction_expert"]["diagnostics"] = deepcopy(teacher.diagnostics)
        if writer is not None:
            try:
                writer.finish(info, error=result.get("error"), events=events)
                writer.close()
                result["dataset"] = str(path.resolve())
                result["dataset_validation"] = validate_episode(path)
            except Exception as exc:
                result["recording_error"] = f"{type(exc).__name__}: {exc}"
                result.setdefault("error", result["recording_error"])
                writer.close()
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                result["close_error"] = f"{type(exc).__name__}: {exc}"
                result.setdefault("error", result["close_error"])
    return result


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    plan, case, manifest = read_plan(args.plan)
    plan["collection_plan_sha256"] = file_sha256(args.plan)
    if args.worker_index is not None:
        if args.worker_index < -1 or args.worker_index >= len(plan["rollouts"]):
            parser.error("invalid worker index")
        result = collect_one(plan, case, manifest, args.checkpoint, args.output_dir, args.worker_index, args.device)
        write_json(args.output_dir / "cases" / f"{args.worker_index}.json", result)
        return 1 if result.get("error") else 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for folder in ("episodes", "cases", "logs"):
        (args.output_dir / folder).mkdir()
    archived_plan = args.output_dir / "input_plan.json"
    archived_plan.write_bytes(args.plan.read_bytes())
    if file_sha256(archived_plan) != plan["collection_plan_sha256"]:
        raise ValueError("Collection plan changed while being archived")
    write_json(args.output_dir / "plan.json", plan)
    report = dict(scope="Train-only assisted correction collection, not autonomous policy evaluation",
                  plan=plan, checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=file_sha256(args.checkpoint),
                  implementation=implementation_provenance(), results=[], complete=False,
                  input_plan_archive=str(archived_plan.resolve()))
    def worker(index):
        command = [sys.executable, str(Path(__file__).resolve()), "--plan", str(archived_plan.resolve()),
                   "--checkpoint", str(args.checkpoint.resolve()), "--output-dir", str(args.output_dir.resolve()),
                   "--device", args.device, "--worker-index", str(index)]
        started = time.monotonic()
        with (args.output_dir / "logs" / f"{index}.log").open("w") as stream:
            proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                  env=dict(os.environ, MUJOCO_GL="egl", OMP_NUM_THREADS="2",
                                           OPENBLAS_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8"))
        path = args.output_dir / "cases" / f"{index}.json"
        try:
            result = json.loads(path.read_text())
        except Exception as exc:
            result = dict(is_success=False, error=f"Missing/invalid worker report: {exc}")
        result.update(worker_index=index, process_exit_code=proc.returncode,
                      elapsed_seconds=time.monotonic() - started, command=command)
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker, i) for i in range(-1, len(plan["rollouts"]))]):
            try:
                result = future.result()
            except Exception as exc:
                result = dict(is_success=False, error=f"Worker orchestration error: {exc}", process_exit_code=-1)
            report["results"].append(result)
            write_json(args.output_dir / "collection.json", report)
            print(json.dumps({k: result.get(k) for k in ("worker_index", "rollout_id", "is_success", "error", "process_exit_code")}), flush=True)
    report["complete"] = len(report["results"]) == len(plan["rollouts"]) + 1
    report["source_unchanged"] = report["implementation"]["source_sha256"] == implementation_provenance()["source_sha256"]
    report["checkpoint_unchanged"] = report["checkpoint_sha256"] == file_sha256(args.checkpoint)
    write_json(args.output_dir / "collection.json", report)
    return 1 if (not report["source_unchanged"] or not report["checkpoint_unchanged"]
                 or any(r.get("error") or r["process_exit_code"] for r in report["results"])) else 0


if __name__ == "__main__":
    raise SystemExit(main())
