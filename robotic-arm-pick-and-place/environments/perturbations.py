"""Independent sensor robustness and explicitly logged recoverable action faults."""

import mujoco
import numpy as np


class AppearancePerturbation:
    def __init__(self, env):
        self.env = env
        m = env.model
        self.original = {name: getattr(m, name).copy() for name in ("cam_pos", "cam_quat", "cam_fovy", "light_diffuse", "light_ambient", "light_specular")}
        self.headlight = {name: getattr(m.vis.headlight, name).copy() for name in ("ambient", "diffuse", "specular")}

    def apply(self, variant="nominal", seed=0):
        m = self.env.model
        for name, value in self.original.items():
            getattr(m, name)[:] = value
        for name, value in self.headlight.items():
            getattr(m.vis.headlight, name)[:] = value
        rng = np.random.default_rng(seed)
        if variant == "camera_shift":
            for name in self.env.cameras:
                i = m.camera(name).id
                m.cam_pos[i] += rng.uniform(-.012 if name == "wrist" else -.035, .012 if name == "wrist" else .035, 3)
                axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
                angle = rng.uniform(-np.deg2rad(6), np.deg2rad(6))
                delta = np.r_[np.cos(angle/2),axis*np.sin(angle/2)]
                q = np.zeros(4); mujoco.mju_mulQuat(q,m.cam_quat[i].copy(),delta)
                m.cam_quat[i] = q
                m.cam_fovy[i] *= rng.uniform(.94,1.06)
        elif variant == "lighting_dim":
            m.light_diffuse[:] *= rng.uniform(.4,.7,(m.nlight,1))
            m.light_specular[:] *= .5
            m.vis.headlight.ambient[:] *= .6
            m.vis.headlight.diffuse[:] *= .65
        elif variant != "nominal":
            raise ValueError("Unknown appearance perturbation")
        mujoco.mj_forward(m,self.env.data)


class RecoveryFault:
    """A one-shot gripper-command fault; never teleports objects or edits scores."""

    def __init__(self, env, kind="none"):
        if kind not in ("none", "missed_grasp", "delayed_release"):
            raise ValueError("Unknown recovery fault")
        self.env, self.kind = env, kind
        self.remaining = 0
        self.triggered = False
        self.events = []

    def apply(self, proposed):
        action = np.asarray(proposed,dtype=np.float32).copy()
        e = self.env._episode
        trigger = (self.kind == "missed_grasp" and action[-1] < 0) or (self.kind == "delayed_release" and e["ever_lifted"] and action[-1] > .5)
        if trigger and not self.triggered:
            self.triggered = True
            self.remaining = round(1.5/self.env.control_dt)
            self.events.append(dict(kind=self.kind, step=e["steps"], duration_steps=self.remaining))
        active = self.remaining > 0
        if active:
            action[-1] = 1 if self.kind == "missed_grasp" else -1
            self.remaining -= 1
        return action, active
