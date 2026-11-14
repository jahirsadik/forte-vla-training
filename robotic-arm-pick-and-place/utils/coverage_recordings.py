"""Audit a declared nominal collection without replacing failed instructions.

Readiness here means a complete balanced engineering dataset, not VLA accuracy.
Initial private state is inspected only to verify identical counterfactual scenes;
it is never exported as a learner input.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np

from utils.coverage_plan import plan_digest, scene_key, pair_key, select_collection_cases
from utils.demonstrations import validate_episode
from utils.research_protocol import verify_metadata
from utils.contracts import robot_definition_digest
from learning.diffusion_policy import check_contract

PROJECT = Path(__file__).resolve().parents[1]


def expected_instruction(case):
    """Use the same scene configuration as SortingEnv, without creating an env."""
    config = json.loads((PROJECT / "assets/sorting/manifest.json").read_text())["configuration"]
    obj, separator, tray = case["task_id"].rpartition("_to_")
    descriptions = {entry["name"]: entry["description"] for entry in config["objects"]}
    if not separator or obj not in descriptions or tray not in {entry["name"] for entry in config["trays"]}:
        raise ValueError("Unknown sorting instruction")
    return f"Pick up the {descriptions[obj]} and place it in the {tray} tray. Leave the other objects."


def _finite_dataset(dataset):
    """Scan small state arrays in chunks; do not load all native RGB frames."""
    if not np.issubdtype(dataset.dtype, np.number):
        return False
    return all(np.isfinite(dataset[start:start + 1024]).all() for start in range(0, len(dataset), 1024))


def _transition_fields(episode, n):
    if not n:
        return
    for name in ("source", "intervention", "expert_action_valid", "terminated", "truncated", "reward"):
        if episode[f"transitions/{name}"].shape != (n,):
            raise ValueError("Malformed scalar transition fields")
    for name in ("intervention", "expert_action_valid", "terminated", "truncated"):
        if episode[f"transitions/{name}"].dtype != np.bool_:
            raise ValueError("Transition flags must be boolean")
    for name in ("action", "proposed_action", "expert_action"):
        if episode[f"transitions/{name}"].shape != (n, 8) or episode[f"transitions/{name}"].dtype != np.float32:
            raise ValueError("Action fields must have float32 shape (T,8)")
    if not _finite_dataset(episode["transitions/reward"]):
        raise ValueError("Nonfinite rewards")


def _validate_outcome(episode, case, n, protocol):
    """Check recorded scoring consistency; physics correctness still needs replay."""
    for name in ("complete", "success", "environment_done"):
        if not isinstance(episode.attrs[name], (bool, np.bool_)):
            raise ValueError("Outcome attributes must be boolean")
    count = episode.attrs["transition_count"]
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count != n:
        raise ValueError("Finalized transition count mismatch")
    if not 0 <= n <= protocol["horizon_steps"]:
        raise ValueError("Recording exceeds the standard horizon")
    summary = json.loads(episode.attrs["summary"])
    metrics, history = summary["metrics"], summary["episode"]
    if (summary["task_id"] != case["task_id"] or summary["protocol_version"] != protocol["protocol_version"]
            or summary["model_fingerprint"] != episode.attrs["model_fingerprint"]
            or type(history["steps"]) is not int or history["steps"] != n
            or type(history["done"]) is not bool or history["done"] != bool(episode.attrs["environment_done"])
            or type(summary["is_success"]) is not bool or summary["is_success"] != bool(episode.attrs["success"])
            or type(metrics["success"]) is not bool or metrics["success"] != summary["is_success"]):
        raise ValueError("Episode outcome disagrees with the recorded summary")
    terminal, truncated = False, False
    if n:
        terminated, truncations = episode["transitions/terminated"][:], episode["transitions/truncated"][:]
        terminal, truncated = bool(terminated[-1]), bool(truncations[-1])
        if terminated[:-1].any() or truncations[:-1].any() or terminal and truncated:
            raise ValueError("Invalid terminal transition sequence")
        recorded_metrics = episode["privileged/metrics_json"]
        if recorded_metrics.shape != (n,) or json.loads(recorded_metrics.asstr()[-1]) != metrics:
            raise ValueError("Final metrics disagree with the summary")
    if history["done"] != (terminal or truncated):
        raise ValueError("Terminal flags disagree with environment completion")
    if summary["is_success"]:
        valid = all(metrics.get(name) is True for name in ("within_goal", "supported", "released", "stable", "placement_valid"))
        valid &= all(history.get(name) is True for name in ("ever_grasped", "ever_lifted"))
        valid &= all(not isinstance(metrics[name], bool) and np.isfinite(metrics[name]) and metrics[name] >= 0
                     for name in ("linear_speed_m_s", "angular_speed_rad_s", "distractor_translation_m", "distractor_rotation_rad", "reward_sparse"))
        valid &= bool(terminal and not truncated and not metrics.get("failure_reason") and not metrics.get("timeout")
                      and not metrics.get("dropped_objects") and type(history.get("wrong_grasp_steps")) is int
                      and history["wrong_grasp_steps"] == 0 and not isinstance(history["dwell"], bool)
                      and np.isfinite(history["dwell"]) and history["dwell"] + 1e-9 >= protocol["success_dwell_seconds"]
                      and metrics["linear_speed_m_s"] <= protocol["linear_speed_tolerance_m_s"]
                      and metrics["angular_speed_rad_s"] <= protocol["angular_speed_tolerance_rad_s"]
                      and metrics["distractor_translation_m"] <= protocol["distractor_translation_limit_m"]
                      and metrics["distractor_rotation_rad"] <= np.deg2rad(protocol["distractor_rotation_limit_deg"])
                      and metrics["reward_sparse"] == 1 and episode["transitions/reward"][-1] == 1)
        if not valid:
            raise ValueError("Claimed success violates the recorded physical placement checks")


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_recordings(paths, plan, manifest, *, source_sha256, subset="nominal_seed0"):
    """Return complete accounting; malformed, failed and missing files stay visible."""
    cases = select_collection_cases(plan, manifest, subset)
    if any(case["recovery"] != "none" for case in cases):
        raise ValueError("This balanced nominal audit does not admit recovery-fault labels")
    if not isinstance(source_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
        raise ValueError("A frozen collection source SHA-256 is required")
    protocol = json.loads((PROJECT / "configs/benchmarks.json").read_text())
    expected = {case["case_id"]: case for case in cases}
    rows, errors, seen_paths, seen_ids, initial, scene_checks = [], [], set(), set(), {}, {}
    contract_signature = None
    for value in paths:
        path = Path(value).resolve()
        row = dict(path=str(path), eligible=False)
        rows.append(row)
        try:
            if str(path) in seen_paths:
                raise ValueError("Duplicate episode path")
            seen_paths.add(str(path))
            validation = validate_episode(path)
            with h5py.File(path, "r") as episode:
                if episode.attrs["schema"] != "forte-demonstrations-v1":
                    raise ValueError("Unsupported demonstration schema")
                metadata = json.loads(episode.attrs["metadata"])
                case = verify_metadata(metadata, manifest)
                row.update(case_id=case["case_id"], task_id=case["task_id"],
                           scene_id=scene_key(case), transitions=validation["transitions"],
                           complete=validation["complete"], success=validation["success"])
                if case["case_id"] not in expected:
                    raise ValueError("Episode outside the predeclared collection subset")
                if case["case_id"] in seen_ids:
                    raise ValueError("Duplicate frozen case in collection")
                seen_ids.add(case["case_id"])
                if (metadata.get("policy") != "expert" or metadata.get("custom_horizon") is not False
                        or metadata.get("counterfactual_diagnostic", False) is not False
                        or metadata.get("instruction_override") is not None
                        or type(metadata.get("execution_horizon_steps")) is not int
                        or metadata["execution_horizon_steps"] != 1):
                    raise ValueError("Collection must use the ordinary one-step expert protocol")
                constructor = dict(asset_set=case["asset_set"], observation_mode="rgb",
                                   image_size=plan["collection"]["image_size_hw"], cameras=plan["collection"]["cameras"])
                if json.dumps(metadata["constructor"], sort_keys=True) != json.dumps(constructor, sort_keys=True):
                    raise ValueError("Noncanonical constructor or hidden horizon override")
                implementation = json.loads(episode.attrs["implementation"])
                if implementation["source_sha256"] != source_sha256:
                    raise ValueError("Collection source differs from frozen source")
                contract = json.loads(episode.attrs["action_contract"])
                check_contract(contract)
                if (contract["image_size_hw"] != plan["collection"]["image_size_hw"]
                        or contract["cameras"] != plan["collection"]["cameras"]
                        or contract["arm_joint_count"] != 7 or contract["shape"] != [8]
                        or contract["policy_hz"] != 20
                        or contract.get("privileged_fields_are_policy_inputs") is not False
                        or contract.get("robot_definition_sha256") != robot_definition_digest()
                        or contract.get("environment_protocol") != protocol["protocol_version"]
                        or contract.get("representation") != "normalized_joint_reference_velocity_and_absolute_jaw_width"
                        or contract.get("dtype") != "float32" or contract.get("normalized_range") != [-1, 1]
                        or contract.get("joint_reference_speed_rad_s") != protocol["joint_reference_speed_rad_s"]
                        or any(type(value) is not int for value in contract["image_size_hw"] + contract["shape"])
                        or type(contract["arm_joint_count"]) is not int
                        or isinstance(contract["policy_hz"], bool)):
                    raise ValueError("Unexpected image, action or policy-observation contract")
                signature = json.dumps(contract, sort_keys=True)
                if contract_signature is not None and signature != contract_signature:
                    raise ValueError("Mixed action contracts")
                contract_signature = signature
                n = row["transitions"]
                _transition_fields(episode, n)
                _validate_outcome(episode, case, n, protocol)
                if not all(_finite_dataset(episode[name]) for name in ("observations/proprio", "privileged/integration_state")):
                    raise ValueError("Nonfinite measured state or proprioception")
                for camera in contract["cameras"]:
                    pixels = episode[f"observations/rgb/{camera}"]
                    if pixels.dtype != np.uint8:
                        raise ValueError("Native RGB must remain uint8")
                if json.loads(episode.attrs.get("events", "[]")):
                    raise ValueError("Nominal collection contains fault/intervention events")
                if n:
                    actions = episode["transitions/action"][:]
                    if (np.any(episode["transitions/source"].asstr()[:] != "expert")
                            or episode["transitions/intervention"][:].any()
                            or not np.array_equal(actions, episode["transitions/proposed_action"][:])):
                        raise ValueError("Nominal recording contains non-expert or overridden actions")
                    valid = episode["transitions/expert_action_valid"][:]
                    if valid.any() and not np.array_equal(episode["transitions/expert_action"][:][valid], actions[valid]):
                        raise ValueError("Explicit expert labels differ from executed actions")
                state = episode["privileged/integration_state"][0]
                proprio = episode["observations/proprio"][0]
                rgb = {camera: episode[f"observations/rgb/{camera}"][0] for camera in contract["cameras"]}
                instruction = str(episode.attrs["instruction"])
                if (instruction != expected_instruction(case)
                        or metadata.get("effective_instruction") not in (None, instruction)):
                    raise ValueError("Recorded language does not match the frozen task instruction")
                row["instruction"] = instruction
                sid = row["scene_id"]
                check = scene_checks.setdefault(sid, dict(cases=[], physics_identical=True,
                    proprio_identical=True, rgb_max_absolute_difference=0, instructions=[]))
                if sid not in initial:
                    initial[sid] = state.copy(), proprio.copy(), rgb
                baseline_state, baseline_proprio, baseline_rgb = initial[sid]
                check["physics_identical"] &= np.array_equal(state, baseline_state)
                check["proprio_identical"] &= np.array_equal(proprio, baseline_proprio)
                check["rgb_max_absolute_difference"] = max(check["rgb_max_absolute_difference"],
                    max(int(np.abs(rgb[camera].astype(np.int16) - baseline_rgb[camera]).max()) for camera in rgb))
                check["cases"].append(case["case_id"])
                check["instructions"].append(instruction)
                row["eligible"] = bool(validation["complete"] and validation["success"] and n > 0
                    and episode.attrs.get("environment_done", False) and not episode.attrs.get("collector_error", ""))
                if not row["eligible"]:
                    row["exclusion"] = "unsuccessful_incomplete_or_collector_error"
            row["sha256"] = file_digest(path)
        except Exception as exc:
            row["eligible"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(dict(path=str(path), error=row["error"]))
    for sid, check in scene_checks.items():
        check["passed"] = bool(check["physics_identical"] and check["proprio_identical"]
            and check["rgb_max_absolute_difference"] <= 1
            and len(set(check["instructions"])) == len(check["instructions"]))
        if not check["passed"]:
            errors.append(dict(scene_id=sid, error="Counterfactual scenes or instruction identities differ"))
    eligible = {r["case_id"] for r in rows if r["eligible"]}
    pairs = defaultdict(list)
    for case in cases:
        pairs[pair_key(case)].append(case["case_id"])
    pair_rows = [dict(scene_id=key[0], object=key[2], case_ids=ids,
                      complete=len(ids) == 2 and set(ids) <= eligible
                          and scene_checks.get(key[0], {}).get("passed", False))
                 for key, ids in sorted(pairs.items())]
    missing = sorted(set(expected) - seen_ids)
    ready = not errors and not missing and len(rows) == len(cases) and len(eligible) == len(cases)
    ready &= all(pair["complete"] for pair in pair_rows)
    return dict(scope="Balanced nominal train-data readiness only; no generalization or policy-performance claim",
                plan_sha256=plan_digest(plan), source_sha256=source_sha256, subset=subset,
                requested_case_ids=list(expected), expected_episodes=len(cases), episodes=rows,
                missing_case_ids=missing, errors=errors, eligible_episodes=len(eligible),
                counterfactual_scenes=scene_checks, destination_pairs=pair_rows,
                complete_destination_pairs=sum(pair["complete"] for pair in pair_rows),
                required_destination_pairs=len(pair_rows), ready_for_balanced_nominal_fit=bool(ready),
                admitted_paths=[r["path"] for r in rows if r["eligible"]] if ready else [],
                eligible_paths=[r["path"] for r in rows if r["eligible"]],
                policy_inputs=["rgb", "proprio", "instruction"],
                physics_replay="Not performed by this structural audit")
