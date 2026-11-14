"""Privileged closed-jaw escape and replanning for stalled sorting carries.

All live movement uses the public eight-dimensional action interface. Held-object
transforms are used only in scratch collision checks. The optional intervention
gate observes history from reset; none of that history is a learned-policy input.
"""

from collections import deque

import mujoco
import numpy as np

from controllers.benchmark_expert import BenchmarkExpert, BenchmarkIK
from controllers.ik import downward_quaternion, rotation_error
from controllers.placement_correction import PlacementReleaseExpert


class CarryRecoveryExpert:
    expert_id = "grasped_carry_recovery_v1"

    def __init__(self, env):
        self.env, self.robot, self.model = env, env.robot, env.model
        self.ik = BenchmarkIK(env)
        self.ik.target_contacts = True
        self.moving_geoms = set(map(int, self.ik.moving_collision_geoms))
        self.stage, self.error = "uninitialized", None
        self.diagnostics = {}
        self.history = deque(maxlen=41)
        self.history_complete = False
        self.ever_entered_goal = False
        self.steps = self.escape_steps = self.grasp_loss_steps = 0
        self.commanded_release = False
        self.release_unsupported_steps = 0
        self.expert = self.release = None
        self.previous_geometry = None

    def observe(self, info):
        """Call once after reset and each executed action, including learner steps."""
        step = int(info["episode"]["steps"])
        if self.history and step == self.history[-1]["step"]:
            return  # Duplicate reads must not manufacture a progress window.
        if step == 0:
            self.history.clear()
            self.history_complete = True
            self.ever_entered_goal = False
        elif not self.history or step != self.history[-1]["step"] + 1:
            self.history_complete = False
        metrics = info["metrics"]
        self.ever_entered_goal |= bool(metrics["within_goal"])
        self.history.append(dict(step=step, distance=float(metrics["goal_distance_m"]),
                                 lag=float(np.max(np.abs(self.robot.target_q - self.robot.arm_position)))))

    def should_intervene(self):
        """Stall evidence; the collector applies minimum-prefix and confirmation rules."""
        if not self.history_complete or len(self.history) < 41 or self.ever_entered_goal:
            return False
        progress = self.history[0]["distance"] - self.history[-1]["distance"]
        lag = float(np.mean([item["lag"] for item in list(self.history)[-10:]]))
        self.diagnostics["gate"] = dict(step=self.history[-1]["step"], progress_40_steps_m=progress,
                                        mean_reference_lag_10_steps_rad=lag,
                                        ever_entered_goal=self.ever_entered_goal,
                                        complete_history=self.history_complete)
        return bool(progress <= .003 and lag > .05 and self.eligible())

    def _fixture(self, geom):
        name = self.model.geom(geom).name or ""
        return name == "table_top" or name.startswith("tray_")

    def _forbidden_contact(self):
        """Only existing fixture contact and intended target/finger contact may escape."""
        target = set(self.env.objects[self.env.target]["geoms"])
        actors = self.moving_geoms | target
        for contact in self.env.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if contact.dist >= -1e-5 or not ({a, b} & actors):
                continue
            if self.ik.allowed_payload_contact(a, b):
                continue
            if (a in actors and self._fixture(b)) or (b in actors and self._fixture(a)):
                continue
            return True
        return False

    def eligible(self):
        episode, metrics = self.env._episode, getattr(self.env, "_last_metrics", {})
        if not ("trays" in getattr(self.env, "scene_config", {}) and episode is not None
                and not episode["done"] and episode["ever_grasped"] and episode["ever_lifted"]
                and metrics["grasped"] and not metrics.get("failure_reason")
                and not metrics["dropped_objects"]
                and (not metrics["within_goal"] or not metrics["supported"])):
            return False
        contacts, _, _ = self.env._contacts()
        if any(contact["l"] and contact["r"] for name, contact in contacts.items() if name != self.env.target):
            return False
        return not self._forbidden_contact()

    def reset(self, seed=None):
        if not self.eligible():
            raise ValueError("Carry correction requires an active lifted target grasp outside a supported sorting goal")
        self.stage, self.error = "escape", None
        self.steps = self.escape_steps = self.grasp_loss_steps = 0
        self.commanded_release = False
        self.release_unsupported_steps = 0
        self.expert = self.release = self.previous_geometry = None
        self.quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(self.quaternion, self.env.data.site_xmat[self.robot.site_id])
        self.escape_goal = self.robot.tool_position.copy() + [0, 0, .07]
        target = set(self.env.objects[self.env.target]["geoms"])
        self.pairs = [(a, b) for a in self.moving_geoms for b in self.ik.environment_geoms
                      if not self.ik.allowed_payload_contact(a, b)]
        self.pairs += [(a, b) for a in target for b in self.ik.environment_geoms if b not in target]
        self.diagnostics.update(takeover_step=int(self.env._episode["steps"]), escape_actions=0,
                                initial_reference_lag_rad=float(np.max(np.abs(self.robot.target_q - self.robot.arm_position))),
                                escape_rejections=[], route_planning=[])
        self.diagnostics["initial_contacts"] = []
        for index, contact in enumerate(self.env.data.contact):
            a, b = int(contact.geom1), int(contact.geom2)
            if not ({a, b} & (self.moving_geoms | target)):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.env.data, index, force)
            self.diagnostics["initial_contacts"].append(dict(a=self.model.geom(a).name, b=self.model.geom(b).name,
                                                            distance_m=float(contact.dist), force_n=float(np.linalg.norm(force[:3]))))

    def _screen_escape(self, candidate):
        """Screen measured geometry and its virtual held object, never controller references."""
        baseline = {pair: mujoco.mj_geomDistance(self.model, self.env.data, *pair, .02, None)
                    for pair in self.pairs}
        start = self.robot.arm_position
        for fraction in (.25, .5, 1.):
            q = start + fraction * (candidate - start)
            self.ik.pose(q)
            for (a, b), before in baseline.items():
                after = mujoco.mj_geomDistance(self.model, self.ik.data, a, b, .02, None)
                if after < min(before, 0.) - 1e-5:
                    return False, f"Escape clearance worsens: {self.model.geom(a).name}/{self.model.geom(b).name}"
            for contact in self.ik.data.contact:
                a, b = int(contact.geom1), int(contact.geom2)
                if a in self.env.robot_geoms and b in self.env.robot_geoms and contact.dist < -1e-5:
                    return False, "Escape robot self collision"
            gravity = (self.ik.data.qfrc_bias[self.robot.dof_indices]
                       + self.robot.kp * self.ik.payload_reference_bias(q))
            if np.max(np.abs(gravity) / self.robot.effort_limits) > .95:
                return False, "Escape static gravity torque margin"
        return True, "ok"

    def _escape_action(self):
        self.escape_steps += 1
        self.diagnostics["escape_actions"] = self.escape_steps
        if self.escape_steps > 300:
            raise ValueError("Carry correction exhausted its bounded escape")
        self.ik.set_payload(self.env.target)
        if (self.robot.tool_position[2] >= self.escape_goal[2] - .006
                and self.ik.screen(self.robot.arm_position)[0]):
            self._start_carry()
            return self._carry_action()
        current = np.zeros(4)
        mujoco.mju_mat2Quat(current, self.env.data.site_xmat[self.robot.site_id])
        mujoco.mj_jacSite(self.model, self.env.data, self.ik.jp, self.ik.jr, self.robot.site_id)
        jacobian = np.vstack([self.ik.jp[:, self.robot.dof_indices], .2 * self.ik.jr[:, self.robot.dof_indices]])
        mobility = np.clip(((self.robot.upper - self.robot.lower) / (2 * np.pi)) ** 2, .02, 1.)
        weighted = mobility[:, None] * jacobian.T
        pseudo = weighted @ np.linalg.solve(jacobian @ weighted + .01**2 * np.eye(6), np.eye(6))
        velocity = 2 * (self.escape_goal - self.robot.tool_position)
        velocity *= min(1., .025 / max(float(np.linalg.norm(velocity)), 1e-12))
        dq = pseudo @ np.r_[velocity, .4 * rotation_error(self.quaternion, current)]
        speed = self.env.protocol["joint_reference_speed_rad_s"]
        dq *= min(1., speed / max(float(np.max(np.abs(dq))), 1e-12))
        reasons = []
        for scale in (1., .5, .25, .125):
            q = np.clip(self.robot.arm_position + scale * dq * self.env.control_dt, self.robot.lower, self.robot.upper)
            accepted, reason = self._screen_escape(q)
            if accepted:
                reference = np.clip(q + self.ik.payload_reference_bias(q), self.robot.lower, self.robot.upper)
                return np.r_[np.clip((reference - self.robot.target_q) / (speed * self.env.control_dt), -1., 1.), -1.].astype(np.float32)
            reasons.append(reason)
        self.diagnostics["escape_rejections"] = reasons
        raise ValueError("No screened carry escape: " + "; ".join(reasons))

    def _start_carry(self):
        # A fresh planner starts at the measured lifted grasp. Never run its
        # approach/reset or inherit a queue from a different physical state.
        expert = BenchmarkExpert(self.env)
        expert.ik.target_contacts = True
        expert.ik.set_payload(self.env.target)
        expert.width, expert.kind, expert.phase, expert.stage = self.robot.closed_width, "sorting", 2, "transfer"
        expert.quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(expert.quaternion, self.env.data.site_xmat[self.robot.site_id])
        expert.destination = np.array(self.env.goal["position"])
        if abs(expert.destination[1]) > .1:
            expert.destination[:2] -= [.035, .035 * np.sign(expert.destination[1])]
        expert.place_quaternion = downward_quaternion(np.arctan2(expert.destination[1], expert.destination[0]))
        expert.close_retries = expert.release_retries = expert.lift_retries = expert.grasp_loss_steps = 0
        expert.planning_attempts = []
        self.expert, self.stage = expert, "transfer"
        self.diagnostics["route_planning"].append(dict(step=int(self.env._episode["steps"]),
                                                       measured_grasp_offset_m=expert.ik.payload["offset"].tolist(),
                                                       reference_lag_rad=float(np.max(np.abs(self.robot.target_q - self.robot.arm_position)))))

    def _carry_action(self):
        if self.previous_geometry is not None and np.max(np.abs(self.robot.arm_position - self.previous_geometry)) > .05:
            raise ValueError("Carry tracking departed from screened route")
        if self._forbidden_contact():
            raise ValueError("Carry encountered an unexpected object or robot contact")
        target = set(self.env.objects[self.env.target]["geoms"])
        for contact in self.env.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if contact.dist >= -1e-5 or not ({a, b} & (target | self.moving_geoms)):
                continue
            if self.ik.allowed_payload_contact(a, b):
                continue
            other = b if a in target else a
            if ({a, b} & target and self.expert.phase >= 4
                    and self.model.geom(other).name == self.env.goal["support"]):
                continue
            raise ValueError("Carry route acquired fixture contact; refuse to advance its queue")
        if not self.expert.commands:
            if self.expert.phase == 4 and not self.env._last_metrics["within_goal"]:
                raise ValueError("Refuse planned opening outside the correct goal")
            self.expert._advance()
        geometry = self.expert.commands[0][2].copy()
        action = self.expert(None)
        self.stage = self.expert.stage
        if self.expert.error:
            raise ValueError(self.expert.error)
        # The executed planned geometry is separated from the command reference,
        # which includes load compensation. Check tracking on the next call.
        self.previous_geometry = geometry
        self.commanded_release |= bool(action[-1] == 1.)
        return action

    def predict(self, observation):
        hold = np.r_[np.zeros(7), 1. if self.commanded_release else -1.].astype(np.float32)
        if self.stage == "uninitialized":
            raise RuntimeError("Reset the carry expert at a physically eligible state")
        if self.error is not None:
            return hold
        try:
            if self.env._episode["done"]:
                self.stage = "complete"
                return hold
            metrics = self.env._last_metrics
            if metrics.get("failure_reason") or metrics["dropped_objects"]:
                raise ValueError("Carry task failed or object dropped")
            self.steps += 1
            if self.commanded_release:
                if not metrics["within_goal"]:
                    raise ValueError("Released target left the correct goal")
                self.release_unsupported_steps = 0 if metrics["supported"] else self.release_unsupported_steps + 1
                # BenchmarkExpert opens for1.5s; the jaws need that physical
                # interval before the object can fall the final few millimetres.
                # Add0.5s for settling, without extending the episode horizon.
                if self.release_unsupported_steps > round(2. / self.env.control_dt):
                    raise ValueError("Released target did not settle onto goal support")
            if self.release is not None:
                action = self.release.predict(observation)
                self.stage = "release_" + self.release.stage
                if self.release.error:
                    raise ValueError(self.release.error)
                return action
            release = PlacementReleaseExpert(self.env)
            if release.eligible():
                release.reset()
                self.release = release
                self.commanded_release = True
                self.stage = "release_open"
                return release.predict(observation)
            self.grasp_loss_steps = 0 if metrics["grasped"] or self.commanded_release else self.grasp_loss_steps + 1
            if self.grasp_loss_steps >= 3:
                raise ValueError("Target grasp lost before supported release")
            action = self._escape_action() if self.expert is None else self._carry_action()
            if action[-1] not in (-1., 1.):
                raise ValueError("Carry correction emitted a nonbinary jaw command")
            return action
        except ValueError as exc:
            self.error, self.stage = str(exc), "failed"
            return hold

    __call__ = predict
