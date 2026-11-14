"""The learner and environment share one explicit joint-reference action definition."""

import json
from pathlib import Path
import numpy as np

LEARNING_INTERFACE = json.loads((Path(__file__).resolve().parents[1] / "configs/learning_interface.json").read_text())


def decode_action(action, arm_reference, jaw_reference, contract):
    """Return the next reference and slew-limited jaw reference in rad/metres.

    arm_reference is controller state, not measured joint position. This function
    has no simulator access and cannot move an object or add a force.
    """
    action = np.asarray(action, dtype=np.float64)
    reference = np.asarray(arm_reference, dtype=np.float64)
    if contract["id"] != "forte-joint-reference-v2":
        raise ValueError("Unsupported action semantics")
    if action.shape != tuple(contract["shape"]) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
        raise ValueError("Action must be a finite normalized vector in [-1, 1]")
    if reference.shape != (contract["arm_joint_count"],) or not np.isfinite(reference).all() or not np.isfinite(jaw_reference):
        raise ValueError("Invalid controller reference")
    dt = 1 / contract["policy_hz"]
    delta = action[:-1] * contract["joint_reference_speed_rad_s"] * dt
    arm = np.clip(reference + delta, contract["joint_lower_limits_rad"], contract["joint_upper_limits_rad"])
    desired = contract["gripper_closed_m"] + (action[-1] + 1) / 2 * (contract["gripper_open_m"] - contract["gripper_closed_m"])
    change = np.clip(desired - jaw_reference, -contract["gripper_speed_limit_m_s"] * dt,
                     contract["gripper_speed_limit_m_s"] * dt)
    return arm, float(jaw_reference + change)


def encode_reference_step(next_arm_reference, arm_reference, nominal_jaw_width, contract):
    """Convert a reachable reference step to learner units; reject an oversized step.

    This is not an IK or pretrained-model adapter. A motor/Cartesian action cannot
    be passed here without a separately validated coordinate/control conversion.
    """
    before, after = np.asarray(arm_reference, dtype=float), np.asarray(next_arm_reference, dtype=float)
    n = contract["arm_joint_count"]
    if before.shape != (n,) or after.shape != (n,):
        raise ValueError("Expected one reference per arm joint")
    arm = (after - before) / (contract["joint_reference_speed_rad_s"] / contract["policy_hz"])
    jaw = 2 * (nominal_jaw_width - contract["gripper_closed_m"]) / (contract["gripper_open_m"] - contract["gripper_closed_m"]) - 1
    action = np.r_[arm, jaw]
    if not np.isfinite(action).all() or np.any(np.abs(action) > 1 + 1e-9):
        raise ValueError("Reference step or jaw width exceeds the action range")
    if np.any(after < contract["joint_lower_limits_rad"]) or np.any(after > contract["joint_upper_limits_rad"]):
        raise ValueError("Joint reference exceeds operating bounds")
    return np.clip(action, -1, 1).astype(np.float32)
