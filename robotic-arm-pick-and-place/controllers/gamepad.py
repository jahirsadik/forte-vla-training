"""Xbox input -> the existing bounded Forte action contract (simulation only).

Cartesian control uses differential IK on scratch reference kinematics. It never
edits live physics. Joint control exposes all seven joints one at a time.
"""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from utils.actions import encode_reference_step


@dataclass(frozen=True)
class PadState:
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    lt: float = 0.0
    rt: float = 0.0
    buttons: tuple = ()
    connected: bool = True

    def __post_init__(self):
        axes = np.array([self.lx, self.ly, self.rx, self.ry, self.lt, self.rt])
        if not np.isfinite(axes).all() or np.any(np.abs(axes[:4]) > 1) or np.any(axes[4:] < 0) or np.any(axes[4:] > 1):
            raise ValueError("Gamepad axes must be finite normalized values")

    def down(self, button):
        return button in self.buttons

    def neutral(self, deadzone=0.12):
        return (self.connected and not self.down("rb") and
                max(abs(self.lx), abs(self.ly), abs(self.rx), abs(self.ry), self.lt, self.rt) <= deadzone)


@dataclass(frozen=True)
class TeleopSettings:
    deadzone: float = 0.12
    precision_scale: float = 0.25
    translation_m_s: float = 0.06
    rotation_rad_s: float = 0.4
    joint_rad_s: float = 0.25
    jaw_m_s: float = 0.04
    damping: float = 0.04
    rotation_weight_m: float = 0.20
    max_reference_lag_rad: float = 0.12

    def __post_init__(self):
        if not all(np.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError("Teleop settings must be finite and positive")
        if self.deadzone >= 1 or self.precision_scale > 1:
            raise ValueError("Invalid deadzone/precision scale")


def deadzone(value, width):
    return float(np.sign(value) * max(0, abs(value) - width) / (1 - width))


class GamepadControl:
    modes = ("cartesian", "joint")

    def __init__(self, env, mode="cartesian", settings=None):
        if mode not in self.modes:
            raise ValueError("Choose cartesian or joint control")
        self.env, self.robot, self.mode = env, env.robot, mode
        self.settings = settings or TeleopSettings()
        self.selected_joint = 0
        self.previous_buttons = set()
        self.scratch = mujoco.MjData(env.model)
        self.jp, self.jr = np.zeros((3, env.model.nv)), np.zeros((3, env.model.nv))

    def action(self, pad):
        cfg, robot, dt = self.settings, self.robot, self.env.control_dt
        rising = set(pad.buttons) - self.previous_buttons
        self.previous_buttons = set(pad.buttons) if pad.connected else set()
        if self.mode == "joint" and pad.connected:
            self.selected_joint = (self.selected_joint + int("right" in rising) - int("left" in rising)) % 7
        q = robot.target_q.copy()
        width = robot.gripper_target
        enabled = pad.connected and pad.down("rb")
        diag = dict(mode=self.mode, selected_joint=self.selected_joint,
                    joint_name=robot.joint_names[self.selected_joint], enabled=bool(enabled),
                    precision=pad.down("lb"), rotation_mode=pad.down("y"),
                    twist_world=[0.] * 6, reference_limited=False)
        if not enabled:
            return encode_reference_step(q, q, width, self.env.action_contract), diag

        scale = cfg.precision_scale if pad.down("lb") else 1.0
        lx, ly, rx, ry, lt, rt = [deadzone(x, cfg.deadzone) for x in
                                  (pad.lx, pad.ly, pad.rx, pad.ry, pad.lt, pad.rt)]
        if self.mode == "joint":
            velocity = np.zeros(7)
            velocity[self.selected_joint] = -ly * cfg.joint_rad_s * scale
        else:
            # Robot base/world frame: +X forward, +Y left, +Z up.
            twist = np.zeros(6)
            if pad.down("y"):
                twist[3:] = np.array([-ly, -lx, -rx]) * cfg.rotation_rad_s * scale
            else:
                twist[:3] = np.array([-ly, -lx, -ry]) * cfg.translation_m_s * scale
                twist[5] = -rx * cfg.rotation_rad_s * scale
            self.scratch.qpos[:] = robot.data.qpos
            self.scratch.qpos[robot.qpos_indices] = q
            self.scratch.qvel[:] = 0
            mujoco.mj_forward(self.env.model, self.scratch)
            mujoco.mj_jacSite(self.env.model, self.scratch, self.jp, self.jr, robot.site_id)
            jac = np.vstack([self.jp[:, robot.dof_indices], self.jr[:, robot.dof_indices]])
            weights = np.array([1., 1., 1., *([cfg.rotation_weight_m] * 3)])
            weighted = weights[:, None] * jac
            inverse = weighted.T @ np.linalg.solve(weighted @ weighted.T + cfg.damping**2 * np.eye(6), np.eye(6))
            velocity = inverse @ (weights * twist)
            # A true numerical null-space direction, avoiding damped inverse
            # leakage into the requested gripper pose during elbow adjustment.
            bias = np.zeros(7)
            bias[2] = (int(pad.down("right")) - int(pad.down("left"))) * cfg.joint_rad_s * scale
            velocity += (np.eye(7) - np.linalg.pinv(weighted, rcond=1e-5) @ weighted) @ bias
            diag["twist_world"] = twist.tolist()
            diag["ik_residual"] = float(np.linalg.norm(weights * (jac @ velocity - twist)))

        speed = self.env.action_contract["joint_reference_speed_rad_s"]
        velocity /= max(1., np.max(np.abs(velocity)) / speed)
        candidate = np.clip(q + velocity * dt, robot.lower, robot.upper)
        # Block only commands that increase excessive tracking error; always
        # permit backing out. Do not jump references to measured positions.
        lag = q - robot.arm_position
        new_lag = candidate - robot.arm_position
        blocked = (np.abs(new_lag) > cfg.max_reference_lag_rad) & (np.abs(new_lag) > np.abs(lag))
        candidate[blocked] = q[blocked]
        diag["reference_limited"] = bool(np.any(blocked) or np.any(np.abs(candidate - q - velocity * dt) > 1e-10))
        width = float(np.clip(width + (rt - lt) * cfg.jaw_m_s * scale * dt, robot.closed_width, robot.open_width))
        return encode_reference_step(candidate, q, width, self.env.action_contract), diag
