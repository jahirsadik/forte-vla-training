"""Privileged, contact-driven extraction after a collision-screened grasp fails.

Live motion consists exclusively of the benchmark's public eight-value actions.
Book coordinates, contacts, constraints and scoring history are never assigned.
"""

import mujoco
import numpy as np
from collections import deque

from controllers.benchmark_expert import (
    BenchmarkExpert, BenchmarkIK, approach_quaternion, quaternion_product,
    rotation_matrix,
)


class ContactBookIK(BenchmarkIK):
    def __init__(self, env):
        super().__init__(env)
        self.sliding_geoms = set()
        self.supported_payload = False

    def payload_contact_threshold(self, name, support):
        if name in self.sliding_geoms:
            return -0.0005
        return super().payload_contact_threshold(name, support)

    def payload_reference_bias(self, q):
        if self.supported_payload:
            return np.zeros_like(q)
        return super().payload_reference_bias(q)


class TopSpinePull:
    """Closed-loop fingertip pressure and withdrawal, followed by clearance.

    The clearance-selected 3/8 N setpoints and 15 N abort limit are
    simulation development settings, not hardware-calibrated force limits.
    Contact is allowed only between the
    target and the existing finger geoms; all other robot clearances remain active.
    """

    force_target_n = 8.0
    force_limit_n = 15.0
    penetration_limit_m = .002
    max_press_steps = 300
    max_drag_m = .17

    def __init__(self, expert):
        self.expert, self.env = expert, expert.env
        dense = "hard_layout" in self.env.scene_config
        self.force_target_n = 3.0 if dense else 8.0
        self.press_preflight_m = .020 if dense else 0.
        self.mode = "approach"
        self.filtered_force = 0.0
        self.press_steps = 0
        self.drag = 0.0
        self.contacted = False
        self.initial_position = self.env.data.xpos[self.env.objects[self.env.target]["body"]].copy()
        self.report = dict(strategy="top_spine_pull", force_target_n=self.force_target_n,
                           force_limit_n=self.force_limit_n, peak_force_n=0.,
                           press_preflight_m=self.press_preflight_m,
                           press_steps=0, commanded_drag_m=0., completed=False,
                           planning_attempts=[])

    def contact_force(self):
        e = self.env
        target = e.objects[e.target]["geoms"]
        fingers = set().union(*e.finger_geoms.values())
        total = 0.
        for index, contact in enumerate(e.data.contact):
            a, b = int(contact.geom1), int(contact.geom2)
            if (a in target and b in fingers) or (b in target and a in fingers):
                force = np.zeros(6)
                mujoco.mj_contactForce(e.model, e.data, index, force)
                total += abs(force[0])
        return float(total)

    def _screen_press_approach(self, point, quaternion):
        """Screen room to establish pressure, not just the free-space endpoint.

        The target's finger contacts alone are allowed in this scratch corridor.
        Neighbours, fixtures and self-collisions retain the ordinary checks. The
        corridor is a planning reserve; live pressure still uses force feedback.
        """
        ik = self.expert.ik
        q = ik.solve_best(point, quaternion, self.env.robot.arm_position)
        if not self.press_preflight_m:
            return
        previous = ik.target_contacts
        ik.target_contacts = True
        try:
            for depth in np.arange(.005, self.press_preflight_m + .001, .005):
                result = ik.solve(point + [0., 0., -depth], quaternion, q)
                if not result.success:
                    raise ValueError("Contact approach preflight: " + result.reason)
                q = result.q
            # Retain the original pressure only when there is ample screened
            # reference travel. Decide before contact: changing pressure after
            # a blocked press can already have tipped the book into a bad path.
            self.force_target_n = 3.
            self.report['pressure_reserve_m'] = self.press_preflight_m
            self.report['pressure_reserve_limit'] = None
            for depth in np.arange(.025, .051, .005):
                result = ik.solve(point + [0., 0., -depth], quaternion, q)
                if not result.success:
                    self.report['pressure_reserve_limit'] = result.reason
                    break
                self.report['pressure_reserve_m'] = float(depth)
                q = result.q
            else:
                self.force_target_n = 8.
            self.report['force_target_n'] = self.force_target_n
        finally:
            ik.target_contacts = previous

    def plan(self):
        x, e = self.expert, self.env
        ik = x.ik
        ik.set_payload(None)
        ik.target_contacts = ik.releasing = False
        ik.planning_width = 0.
        x.width = 0.
        ik.pose(e.robot.arm_position)
        data, model = ik.data, e.model
        origin = data.site_xpos[e.robot.site_id]
        rotation = data.site_xmat[e.robot.site_id].reshape(3, 3)
        local = {}
        for side, ids in e.finger_geoms.items():
            gid = next(iter(ids))
            mesh = model.geom_dataid[gid]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start:start+count]
            world = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            local[side] = (world - origin) @ rotation
        low, high = e._bounds(e.target)
        center = e.data.xpos[e.objects[e.target]["body"]]
        for pitch in (30., 45., 15., 60.):
            for roll in (-35., 35., -60., 60., 0.):
                angle = np.deg2rad(roll) / 2
                quaternion = quaternion_product(
                    approach_quaternion(0., np.deg2rad(pitch)),
                    np.array([np.cos(angle), np.sin(angle), 0., 0.]),
                )
                rotation = rotation_matrix(quaternion)
                projected = {side: vertices @ rotation.T for side, vertices in local.items()}
                side = min(projected, key=lambda key: projected[key][:, 2].min())
                vertices = projected[side]
                tip = vertices[vertices[:, 2] < vertices[:, 2].min() + .0005].mean(axis=0)
                point = np.array([low[0] + .008, center[1], high[2] + .004]) - tip
                x.commands.clear()
                try:
                    self._screen_press_approach(point, quaternion)
                    hover = point + [-.05, 0., .015]
                    q = ik.solve_best(hover, quaternion, e.robot.arm_position)
                    x._hold(.8, 0.)
                    x._joint_route(q, 0.)
                    x._cartesian_segment(point, quaternion, 0.)
                    self.quaternion = quaternion
                    self.press_position = point.copy()
                    self.base_z = point[2]
                    self.report["approach"] = dict(pitch_deg=pitch, roll_deg=roll,
                                                   finger=side, tool_position_m=point.tolist())
                    return
                except ValueError as exc:
                    self.report["planning_attempts"].append(
                        dict(pitch_deg=pitch, roll_deg=roll, error=str(exc)))
        x.commands.clear()
        raise ValueError("No screened top-spine contact approach")

    def _action(self, reference, width):
        x = self.expert
        scale = self.env.protocol["joint_reference_speed_rad_s"] * self.env.control_dt
        return np.r_[np.clip((reference - x.robot.target_q) / scale, -1., 1.),
                     x._width_action(width)].astype(np.float32)

    def next_action(self):
        x, e = self.expert, self.env
        force = self.contact_force()
        self.report["peak_force_n"] = max(self.report["peak_force_n"], force)
        if force > self.force_limit_n:
            raise ValueError("Top-spine contact force exceeded its abort limit")
        if e._last_metrics["robot_penetration_m"] > self.penetration_limit_m:
            raise ValueError("Top-spine contact exceeded its penetration guard")
        x.stage = "contact_" + self.mode
        if x.commands:
            reference, width, _ = x.commands.popleft()
            return self._action(reference, width)
        if self.mode == "approach":
            self.mode = "press_pull"
            x.ik.target_contacts = True
        if self.mode == "press_pull":
            if self.press_steps >= self.max_press_steps or self.drag >= self.max_drag_m:
                if not self.contacted:
                    raise ValueError("Top-spine preparation could not establish contact")
                self.mode = "clear"
                x._cartesian_segment(self.press_position + [-.02, 0., .10], self.quaternion, 0.)
                return self.next_action()
            self.filtered_force = .7 * self.filtered_force + .3 * force
            self.contacted |= self.filtered_force > self.force_target_n * .75
            dz = np.clip((self.filtered_force - self.force_target_n) * .0002, -.0009, .0009)
            self.press_position[2] = np.clip(self.press_position[2] + dz,
                                             self.base_z - .12, self.base_z + .015)
            if self.contacted and self.filtered_force > self.force_target_n * .6:
                self.press_position[0] -= .0012
                self.drag += .0012
            result = x.ik.solve(self.press_position, self.quaternion, x.robot.arm_position)
            if not result.success:
                raise ValueError("Top-spine contact tracking: " + result.reason)
            self.press_steps += 1
            self.report.update(press_steps=self.press_steps, commanded_drag_m=self.drag)
            x.stage = "contact_press_pull"
            return self._action(result.q, 0.)
        if self.mode == "clear":
            self.mode = "settle"
            x._hold(1.5, 0.)
            return self.next_action()
        position = e.data.xpos[e.objects[e.target]["body"]]
        withdrawal = float(self.initial_position[0] - position[0])
        self.report.update(withdrawal_m=withdrawal, final_book_position_m=position.tolist())
        if withdrawal < .04:
            raise ValueError("Top-spine preparation did not expose enough of the book")
        self.report["completed"] = True
        return None


class ContactBookExpert(BenchmarkExpert):
    """Use the nominal expert first; prepare blocked extraction grasps physically."""

    def __init__(self, env):
        super().__init__(env)
        self.ik = ContactBookIK(env)
        self.preparation = None
        self.contact_summary = None
        self.prepared = self.fallen = False

    def reset(self, seed=None):
        self.preparation = None
        self.contact_summary = None
        self.prepared = self.fallen = False
        self.ik.sliding_geoms.clear()
        self.ik.supported_payload = False
        try:
            super().reset(seed)
        except ValueError as exc:
            if self.kind != "extract" or not str(exc).startswith("No book approach:"):
                raise
            self.commands.clear()
            self.stage = "contact_approach"
            self.preparation = TopSpinePull(self)
            self.contact_summary = self.preparation.report
            self.contact_summary["direct_approach_error"] = str(exc)
            self.preparation.plan()

    def __call__(self, observation):
        if self.preparation is not None and self.error is None:
            try:
                action = self.preparation.next_action()
                if action is not None:
                    return action
                self.preparation = None
                self.prepared = True
                e = self.env
                rotation = e.data.xmat[e.objects[e.target]["body"]].reshape(3, 3)
                self.fallen = rotation[2, 2] < np.cos(np.deg2rad(50.))
                self.contact_summary["regrasp_family"] = "top" if self.fallen else "side"
                self.ik.target_contacts = False
                self.ik.planning_width = None
                target = e.objects[e.target]["geoms"][0]
                # Allow only pre-existing target/fixture contacts during withdrawal.
                for name in ("shelf_upper", "shelf_left", "shelf_right"):
                    gid = e.model.geom(name).id
                    if mujoco.mj_geomDistance(e.model, e.data, target, gid, .003, None) < .002:
                        self.ik.sliding_geoms.add(name)
                self.phase = 0
                self.stage = "approach"
                self._build_approach()
            except ValueError as exc:
                self.error, self.stage = str(exc), "failed"
                if self.contact_summary is not None:
                    self.contact_summary["error"] = self.error
        return super().__call__(observation)

    def _book_grasp_candidates(self, object_position):
        if not self.fallen:
            yield from super()._book_grasp_candidates(object_position)
            return
        _, high = self.env._bounds(self.env.target)
        for pitch in (90., 75., 60., 45.):
            for offset in (.08, .05, 0., -.05):
                for below_top in (.04, .055, .025):
                    point = np.array([object_position[0]+offset, object_position[1], high[2]-below_top])
                    yield pitch, offset, below_top, point, approach_quaternion(0., np.deg2rad(pitch))

    def _screen_book_withdraw(self, q):
        if not self.prepared:
            return super()._screen_book_withdraw(q)
        e = self.env
        front = e.scene_config["shelf"]["center_xy_m"][0] - e.scene_config["shelf"]["depth_m"] / 2
        start = self.grasp_position.copy()
        withdrawn = start.copy()
        withdrawn[0] -= max(0., e._bounds(e.target)[1][0] - front + .035)
        for end in (withdrawn, withdrawn + [0., 0., .025]):
            for point in np.linspace(start, end, max(2, int(np.linalg.norm(end-start)/.008)+1))[1:]:
                result = self.ik.solve(point, self.quaternion, q)
                if not result.success:
                    raise ValueError("Prepared book withdrawal preflight: " + result.reason)
                q = result.q
            start = end

    def _advance(self):
        if not (self.prepared and self.phase == 1 and self.env._last_metrics["grasped"]):
            return super()._advance()
        e = self.env
        self.phase, self.stage = 2, "contact_withdraw"
        self.ik.set_payload(e.target)
        if not self.fallen:
            body = e.objects[e.target]["body"]
            angle = np.deg2rad(12.) / 2
            turn = np.array([np.cos(angle), 0., np.sin(angle), 0.])
            self._book_body_segment(e.data.xpos[body] + [-.04, 0., 0.],
                                    quaternion_product(turn, e.data.xquat[body]))
        point, self.quaternion = self.ik.pose(self._planned_geometry())
        gid = self.ik.payload["geom"]
        extent = np.abs(self.ik.data.geom_xmat[gid].reshape(3, 3)) @ e.model.geom_size[gid]
        front = e.scene_config["shelf"]["center_xy_m"][0] - e.scene_config["shelf"]["depth_m"] / 2
        point[0] -= max(0., self.ik.data.geom_xpos[gid][0] + extent[0] - front + .035)
        self.ik.supported_payload = True
        self._cartesian_segment(point, self.quaternion, self.width)
        self.ik.supported_payload = False
        self.ik.sliding_geoms.clear()
        self._cartesian_segment(point + [0., 0., .075 if self.fallen else .025],
                                self.quaternion, self.width)
        self._hold(.3)

    def _book_goal_candidates(self):
        if not self.fallen:
            yield from super()._book_goal_candidates()
            return
        e = self.env
        half = np.array(e.objects[e.target]["half_extents"])
        base = e.data.xquat[e.objects[e.target]["body"]].copy()
        initial_extent = np.abs(rotation_matrix(base)) @ half
        for yaw in (np.pi/2, -np.pi/2, 0.):
            yaw_q = np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])
            for angle in (55., 40., 70.):
                for sign in (1., -1.):
                    roll = sign * np.deg2rad(angle)
                    turn = np.array([np.cos(roll/2), np.sin(roll/2), 0., 0.])
                    quaternion = quaternion_product(yaw_q, quaternion_product(turn, base))
                    flat = quaternion_product(yaw_q, quaternion_product(
                        np.array([np.sqrt(.5), sign*np.sqrt(.5), 0., 0.]), base))
                    flat_extent = np.abs(rotation_matrix(flat)) @ half
                    room = np.array(e.goal["high"][:2]) - e.goal["low"][:2]
                    if np.any(2*flat_extent[:2] > room - .02):
                        continue
                    extent = np.abs(rotation_matrix(quaternion)) @ half
                    goal = np.array(e.goal["position"])
                    goal[0] = e.goal["high"][0] - flat_extent[0] - .01
                    goal[2] = e.goal["surface_z"] + extent[2] + .005
                    shift = sign * (initial_extent[2]*(1-np.sin(abs(roll)))
                                    + initial_extent[1]*np.cos(roll))
                    goal += rotation_matrix(yaw_q) @ np.array([0., shift, 0.])
                    yield goal, quaternion

    def _book_transfer(self):
        if not self.fallen:
            return super()._book_transfer()
        e = self.env
        body = e.objects[e.target]["body"]
        self.stage = "carry"
        point = e.data.xpos[body].copy()
        point[0], point[2] = max(point[0], .40), e.table_surface + .22
        self._book_body_segment(point, e.data.xquat[body].copy())
        self._book_body_segment(point, self.book_orientation)
        self._book_body_segment(self.book_destination + [0., 0., .065], self.book_orientation)
        _, self.quaternion = self._book_tool_goal(self.book_destination, self.book_orientation)

    def _fallback_book_transfer(self):
        """Try the complete grasped-object route without prescribing its sweep.

        A straight object-orientation interpolation can collide even when a
        nearby redundant arm posture provides a valid joint-space route. This
        fallback runs only after all existing complete routes failed, preserving
        the nominal expert's already validated behavior.
        """
        if self.kind != 'extract':
            return False
        self.stage = 'carry'
        prefix = deque(self.commands)
        start = self._planned_geometry()
        previous_orientation = self.quaternion.copy()
        attempts = 0
        errors = []
        for destination, orientation in self._book_goal_candidates():
            tool, quaternion = self._book_tool_goal(destination + [0., 0., .065], orientation)
            for result in self.ik.continuous_candidates(tool, quaternion, start):
                if not result.success:
                    continue
                attempts += 1
                self.commands = deque(prefix)
                try:
                    self._joint_route(result.q, self.width)
                    carry_end = len(self.commands)
                    final_tool, final_quat = self._book_tool_goal(destination, orientation)
                    self._cartesian_segment(final_tool, final_quat, self.width)
                    while len(self.commands) > carry_end:
                        self.commands.pop()
                    self.book_destination = destination
                    self.book_orientation = orientation
                    self.quaternion = quaternion
                    self.planning_attempts.append(dict(route='object_aware_joint_transfer',
                                                      attempts=attempts, success=True))
                    return True
                except ValueError as exc:
                    errors.append(str(exc))
        self.commands = prefix
        self.quaternion = previous_orientation
        self.planning_attempts.append(dict(route='object_aware_joint_transfer', attempts=attempts,
                                          success=False, errors=errors[-3:]))
        return False
