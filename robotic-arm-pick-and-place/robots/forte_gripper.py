"""Forte parallel gripper: widths in metres, one force input in newtons.

Positive left-slider motion opens the jaws. The right slider is a passive
equality-coupled follower, not a second independently commanded joint.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "forte.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["gripper"]


class ForteGripper:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.config = load_config()
        np.testing.assert_allclose(self.config["approach_axis_local"], [1, 0, 0])
        np.testing.assert_allclose(self.config["closing_axis_local"], [0, 1, 0])
        driver = model.joint(self.config["driver_joint"])
        follower = model.joint(self.config["follower_joint"])
        if any(int(j.type[0]) != int(mujoco.mjtJoint.mjJNT_SLIDE) for j in (driver, follower)):
            raise ValueError("Forte gripper joints must be sliders")
        self.driver_qpos = int(driver.qposadr[0])
        self.follower_qpos = int(follower.qposadr[0])
        self.driver_dof = int(driver.dofadr[0])
        self.follower_dof = int(follower.dofadr[0])
        self.lower, self.upper = map(float, driver.range)
        np.testing.assert_allclose(follower.range, [self.lower, self.upper])
        matches = [i for i in range(model.neq)
                   if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
                   and model.eq_obj1id[i] == driver.id
                   and model.eq_obj2id[i] == follower.id]
        if len(matches) != 1:
            raise ValueError("Expected exactly one right-to-left finger coupling")
        np.testing.assert_allclose(model.eq_data[matches[0], :5], [0, 1, 0, 0, 0])
        self.site = model.site(self.config["tool_site"]).id
        self.contact_geoms = {side: {model.geom(name).id for name in names}
                              for side, names in self.config["finger_contact_geoms"].items()}
        self.left_face = next(iter(self.contact_geoms["l"]))
        self.right_face = next(iter(self.contact_geoms["r"]))
        self.min_width = self.width_from_position(self.lower)
        self.max_width = self.width_from_position(self.upper)
        for name in ("open_width_m", "empty_close_width_m", "grasp_reference_width_m"):
            self.position_from_width(self.config[name])
        if any(not np.isfinite(self.config[k]) or self.config[k] <= 0 for k in
               ("kp_n_per_m", "kd_n_s_per_m", "effort_limit_n", "speed_limit_m_per_s")):
            raise ValueError("Gripper gains and limits must be finite and positive")
        # Applied generalized forces bypass actuator limits: enforce the bound here.
        model_effort = float(model.jnt_actfrcrange[driver.id, 1])
        if self.config["effort_limit_n"] > model_effort:
            raise ValueError("Configured gripper force exceeds the derived model limit")

    def width_from_position(self, position: float) -> float:
        return float(2 * position)

    def position_from_width(self, width: float) -> float:
        if not np.isfinite(width):
            raise ValueError("Gripper width must be finite")
        position = width / 2
        if position < self.lower - 1e-12 or position > self.upper + 1e-12:
            raise ValueError(f"Width {width:g} m is outside the Forte opening range")
        return float(np.clip(position, self.lower, self.upper))

    def initialize(self, data: mujoco.MjData, width: float) -> None:
        """Assign both coordinates only at fixture/reset initialization."""
        position = self.position_from_width(width)
        data.qpos[self.driver_qpos] = position
        data.qpos[self.follower_qpos] = position
        data.qvel[[self.driver_dof, self.follower_dof]] = 0

    def apply(self, data: mujoco.MjData, width: float, driver_speed: float = 0.0) -> float:
        target = self.position_from_width(width)
        if not np.isfinite(driver_speed) or abs(driver_speed) > self.config["speed_limit_m_per_s"]:
            raise ValueError("Requested slider speed exceeds the configured limit")
        raw = (self.config["kp_n_per_m"] * (target - data.qpos[self.driver_qpos])
               + self.config["kd_n_s_per_m"] * (driver_speed - data.qvel[self.driver_dof]))
        effort = float(np.clip(raw, -self.config["effort_limit_n"], self.config["effort_limit_n"]))
        data.qfrc_applied[self.driver_dof] = effort
        data.qfrc_applied[self.follower_dof] = 0
        return effort

    def geometry(self, data: mujoco.MjData) -> tuple[float, float]:
        """Measure inner-face gap and transverse center error in the tool frame."""
        rotation = data.site_xmat[self.site].reshape(3, 3)
        center = data.site_xpos[self.site]
        intervals, midpoints = [], []
        for geom in (self.left_face, self.right_face):
            mesh = self.model.geom_dataid[geom]
            start, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
            vertices = self.model.mesh_vert[start:start + count]
            world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            local = (world - center) @ rotation
            intervals.append((float(local[:, 1].min()), float(local[:, 1].max())))
            midpoints.append((local.min(axis=0) + local.max(axis=0)) / 2)
        intervals.sort()
        gap = intervals[1][0] - intervals[0][1]
        midpoint = np.mean(midpoints, axis=0)
        return float(gap), float(np.linalg.norm(midpoint[1:]))
