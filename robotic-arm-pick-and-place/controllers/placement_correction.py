"""Privileged, state-conditioned release correction for supported sorting goals.

The learned policy never receives this expert or its gate. All live movement is
through the existing normalized action interface. Geometry checks use scratch
data; neither object poses nor episode/scoring history are modified.
"""

import mujoco
import numpy as np

from controllers.ik import rotation_error


class PlacementReleaseExpert:
    expert_id = "supported_goal_release_v1"

    def __init__(self, env):
        self.env, self.robot, self.model = env, env.robot, env.model
        self.scratch = mujoco.MjData(self.model)
        self.jp, self.jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        self.moving_geoms = {int(g) for g in env.robot_collision_geoms
                             if self.model.geom_bodyid[g] in self.robot.moving_body_ids}
        self.environment_geoms = [g for g in range(self.model.ngeom)
                                  if self.model.geom_contype[g] and g not in env.robot_geoms]
        self.stage, self.error = "uninitialized", None
        self.steps = 0
        self.unsupported_steps = 0
        self.screen_rejections = []
        self.direction = None
        self.position = self.quaternion = None

    def _other_collisions(self, data):
        target = set(self.env.objects[self.env.target]["geoms"])
        collisions = {}
        for contact in data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if not ({a, b} & self.moving_geoms) or {a, b} & target:
                continue
            if contact.dist < -1e-5:
                collisions[tuple(sorted((a, b)))] = float(contact.dist)
        return collisions

    def eligible(self):
        episode, metrics = self.env._episode, getattr(self.env, "_last_metrics", {})
        candidate = bool(
            "trays" in getattr(self.env, "scene_config", {})
            and episode is not None and not episode["done"]
            and episode["ever_grasped"] and episode["ever_lifted"]
            and metrics["within_goal"] and metrics["supported"] and metrics["stable"]
            and not metrics["released"] and not metrics.get("failure_reason")
            and not metrics["dropped_objects"] and not self._other_collisions(self.env.data)
        )
        if not candidate:
            return False
        contacts, _, _ = self.env._contacts()
        return not any(value["l"] and value["r"] for name, value in contacts.items()
                       if name != self.env.target)

    def reset(self, seed=None):
        if not self.eligible():
            raise ValueError("Release correction requires an active, stable, supported target in its correct sorting goal")
        self.stage, self.error = "open", None
        self.steps, self.unsupported_steps = 0, 0
        self.screen_rejections.clear()
        self.position = self.robot.tool_position.copy()
        self.quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(self.quaternion, self.env.data.site_xmat[self.robot.site_id])
        self.away = -self.env.data.site_xmat[self.robot.site_id].reshape(3, 3)[:, 0].copy()
        self.direction = None

    def _distances(self, data):
        return {(a, b): mujoco.mj_geomDistance(self.model, data, a, b, .02, None)
                for a in self.moving_geoms for b in self.environment_geoms}

    def _screen(self, candidate, baseline):
        """Permit escape from existing target contact, never deepen penetration.

        The starting posture may already lie near a configured joint bound. New
        reference endpoints stay inside the actual bounds, without imposing a
        new inward margin that would make escape from that posture impossible.
        """
        target = set(self.env.objects[self.env.target]["geoms"])
        start = self.robot.arm_position
        for fraction in (.25, .5, 1.):
            self.scratch.qpos[:] = self.env.data.qpos
            self.scratch.qpos[self.robot.qpos_indices] = start + fraction * (candidate - start)
            self.scratch.qvel[:] = 0
            mujoco.mj_forward(self.model, self.scratch)
            if self._other_collisions(self.scratch):
                return False, "new robot/self or robot/environment collision"
            fraction_effort = np.max(np.abs(self.scratch.qfrc_bias[self.robot.dof_indices])
                                     / self.robot.effort_limits)
            if fraction_effort > .95:
                return False, "static gravity torque margin"
            for (a, b), before in baseline.items():
                after = mujoco.mj_geomDistance(self.model, self.scratch, a, b, .02, None)
                # Existing intended target penetration can decrease to zero;
                # an already separated pair must not acquire penetration.
                minimum = min(before, 0.) if b in target else 0.
                if after < minimum - 1e-5:
                    return False, f"clearance worsened: {self.model.geom(a).name}/{self.model.geom(b).name}"
        return True, "ok"

    def _withdrawal_action(self):
        robot, data = self.robot, self.env.data
        current = np.zeros(4)
        mujoco.mju_mat2Quat(current, data.site_xmat[robot.site_id])
        mujoco.mj_jacSite(self.model, data, self.jp, self.jr, robot.site_id)
        jacobian = np.vstack([self.jp[:, robot.dof_indices], .2 * self.jr[:, robot.dof_indices]])
        mobility = np.clip(((robot.upper - robot.lower) / (2 * np.pi)) ** 2, .02, 1.)
        weighted = mobility[:, None] * jacobian.T
        pseudo = weighted @ np.linalg.solve(jacobian @ weighted + .01**2 * np.eye(6), np.eye(6))
        baseline = self._distances(data)
        directions = [self.direction] if self.direction is not None else [np.array([0., 0., 1.]), self.away]
        speed = self.env.protocol["joint_reference_speed_rad_s"]
        for direction in directions:
            velocity = 2 * (self.position + .08 * direction - robot.tool_position)
            velocity *= min(1., .025 / max(float(np.linalg.norm(velocity)), 1e-12))
            angular = 2 * rotation_error(self.quaternion, current)
            joint_velocity = pseudo @ np.r_[velocity, .2 * angular]
            joint_velocity *= min(1., speed / max(float(np.max(np.abs(joint_velocity))), 1e-12))
            for scale in (1., .5, .25, .125):
                endpoint = np.clip(robot.arm_position + scale * joint_velocity * self.env.control_dt,
                                   robot.lower, robot.upper)
                # Clamp reference changes exactly like the public action path.
                action = np.r_[np.clip((endpoint - robot.target_q) / (speed * self.env.control_dt), -1., 1.), 1.].astype(np.float32)
                # Controller references can lag or preload a contact. They are
                # not actual geometry: screen the measured-pose DLS endpoint,
                # while slewing the reference toward it through bounded actions.
                accepted, reason = self._screen(endpoint, baseline)
                if accepted:
                    self.direction = direction.copy()
                    return action
                self.screen_rejections.append(reason)
        raise ValueError("No screened release withdrawal: " + "; ".join(self.screen_rejections[-4:]))

    def predict(self, observation):
        hold = np.r_[np.zeros(7), 1.].astype(np.float32)
        if self.stage == "uninitialized":
            raise RuntimeError("Reset the correction expert at an eligible takeover state")
        if self.error is not None:
            return hold
        try:
            metrics = self.env._last_metrics
            if metrics.get("failure_reason") or not metrics["within_goal"] or metrics["dropped_objects"]:
                raise ValueError("Target left the supported-goal correction domain or task already failed")
            self.unsupported_steps = 0 if metrics["supported"] else self.unsupported_steps + 1
            if self.unsupported_steps > 10:
                raise ValueError("Target lost support during release correction")
            if self.steps >= 250:
                raise ValueError("Release correction exhausted its bounded withdrawal")
            self.steps += 1
            if metrics["released"]:
                self.stage = "settle"
                return hold
            if self.steps <= 20:
                self.stage = "open"
                return hold
            self.stage = "withdraw"
            return self._withdrawal_action()
        except ValueError as exc:
            self.error, self.stage = str(exc), "failed"
            return hold

    __call__ = predict
