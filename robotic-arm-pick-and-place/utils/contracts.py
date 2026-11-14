"""Versioned control/observation specification and implementation provenance."""

import hashlib
from importlib.metadata import version
import os
import platform
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[1]


def robot_definition_digest():
    digest = hashlib.sha256()
    for relative in ("configs/forte.json", "assets/forte_arm/forte.xml",
                     "assets/forte_arm/source_7dof/manifest.json"):
        digest.update(relative.encode())
        digest.update((PROJECT / relative).read_bytes())
    return digest.hexdigest()


def implementation_provenance():
    digest = hashlib.sha256()
    for folder in ("environments", "controllers", "robots", "utils", "scripts", "configs", "learning"):
        for path in sorted((PROJECT / folder).rglob("*")):
            if path.suffix in (".py", ".json"):
                digest.update(str(path.relative_to(PROJECT)).encode())
                digest.update(path.read_bytes())
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    return {"source_sha256": digest.hexdigest(), "git_revision": revision,
            "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                        "mujoco_gl": os.environ.get("MUJOCO_GL", "default"),
                        **{name: version(name) for name in ("mujoco", "numpy", "gymnasium", "h5py")}}}


def action_contract(env):
    return {
        "id": "forte-joint-reference-v2",
        "environment_protocol": env.protocol["protocol_version"],
        "robot_definition_sha256": robot_definition_digest(),
        "representation": "normalized_joint_reference_velocity_and_absolute_jaw_width",
        "joint_names": list(env.robot.joint_names),
        "joint_aliases": {"wrist_roll": "independent wrist yaw (source joint name retained)"},
        "joint_coordinate_semantics": "source simulation reference, unwrapped radians within explicit operating bounds; not hardware encoder calibration",
        "arm_joint_count": len(env.robot.joint_names),
        "shape": list(env.action_space.shape),
        "dtype": "float32",
        "joint_lower_limits_rad": env.robot.lower.tolist(),
        "joint_upper_limits_rad": env.robot.upper.tolist(),
        "joint_effort_limits_nm": env.robot.effort_limits.tolist(),
        "normalized_range": [-1, 1],
        "policy_hz": 1 / env.control_dt,
        "physics_dt_s": float(env.model.opt.timestep),
        "joint_reference_speed_rad_s": env.protocol["joint_reference_speed_rad_s"],
        "reference_update": "q_ref_next=clip(q_ref+action_arm*speed*policy_dt,joint_limits)",
        "controller": "torque-limited joint PD with robot gravity compensation; no privileged payload compensation",
        "gripper_closed_m": env.robot.closed_width,
        "gripper_open_m": env.robot.open_width,
        "gripper_speed_limit_m_s": env.protocol["gripper_reference_speed_m_s"],
        "gripper_sign": "-1 closed; +1 open",
        "jaw_width_definition": "twice the opening slider coordinate; physical pad gap includes the geometry offset",
        "physical_pad_gap_offset_m": env.robot.gripper.config["pad_gap_offset_m"],
        "tool_frame": "grasp_site: +X approach, +Y jaw axis; offset from source grasp recorded in robot configuration",
        "proprio_order": ["arm_q_rad", "arm_dq_rad_s", "jaw_width_m", "jaw_velocity_m_s", "world_tool_xyz_m", "world_tool_quat_wxyz"],
        "image_layout": "HWC uint8 RGB",
        "cameras": list(env.cameras),
        "image_size_hw": list(env.image_size),
        "privileged_fields_are_policy_inputs": False,
    }
