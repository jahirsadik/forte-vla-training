"""Privileged physical expert for the benchmark action interface.

IK uses scratch state. Live motion is exclusively env.step(normalized_action):
no object attachments, pose assignment, extra payload forces, or stage-history edits.
"""

from collections import deque
import mujoco
import numpy as np

from controllers.ik import PoseIK, downward_quaternion


def approach_quaternion(yaw, pitch):
    q = np.zeros(4)
    mujoco.mju_mulQuat(
        q,
        np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]),
        np.array([np.cos(pitch / 2), 0, np.sin(pitch / 2), 0]),
    )
    return q


def quaternion_product(a, b):
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, a, b)
    return q


def rotation_matrix(q):
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, q)
    return matrix.reshape(3, 3)


def interpolate_quaternion(a, b, t):
    if np.dot(a, b) < 0:
        b = -b
    angle = np.arccos(np.clip(np.dot(a, b), -1, 1))
    if angle < 1e-7:
        return a.copy()
    return (np.sin((1 - t) * angle) * a + np.sin(t * angle) * b) / np.sin(angle)


class BenchmarkIK(PoseIK):
    def __init__(self, env):
        self.env = env
        self.robot, self.model = env.robot, env.model
        self.data = mujoco.MjData(self.model)
        self.position_tolerance = 0.0007
        self.rotation_tolerance = np.deg2rad(0.8)
        self.joint_margin = 0.015
        self.solve_joint_margin = 0.030  # Reserve tracking error beyond path screening margin.
        self.torque_fraction = 0.95
        self.environment_clearance = 0.001
        self.robot_geoms = np.array(
            [g in env.robot_geoms for g in range(self.model.ngeom)]
        )
        self.moving_collision_geoms = np.asarray([g for g in env.robot_collision_geoms
            if self.model.geom_bodyid[g] in self.robot.moving_body_ids])
        self.environment_geoms = [
            g
            for g in range(self.model.ngeom)
            if self.model.geom_contype[g] and g not in env.robot_geoms
        ]
        self.jp, self.jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        self.payload = None
        self.target_contacts = False
        self.releasing = False
        self.last_clearance = float("inf")
        self.planning_width = None
        self.clearance_pair = None

    def pose(self, q):
        result = super().pose(q)
        if self.planning_width is not None:
            self.robot.gripper.initialize(self.data, self.planning_width)
            mujoco.mj_forward(self.model, self.data)
        return result

    def allowed_payload_contact(self, a, b):
        target = self.env.objects[self.env.target]["geoms"]
        if a not in target and b not in target:
            return False
        other = b if a in target else a
        return self.target_contacts and any(
            other in geoms for geoms in self.env.finger_geoms.values()
        )

    def environment_distance(self):
        distance = 0.02
        self.clearance_pair = None
        for a in self.moving_collision_geoms:
            for b in self.environment_geoms:
                if self.allowed_payload_contact(int(a), b):
                    continue
                if self.releasing and b in self.env.objects[self.env.target]["geoms"]:
                    # Withdrawal starts beside the just-released book. Permit
                    # positive gaps below the ordinary 1 mm planning margin;
                    # penetration remains forbidden and scoring is unchanged.
                    separation = mujoco.mj_geomDistance(
                        self.model, self.data, int(a), b, 0.02, None
                    )
                    if separation < 0:
                        self.clearance_pair = (
                            self.model.geom(int(a)).name,
                            self.model.geom(b).name,
                            float(separation),
                        )
                        return -1.0
                    continue
                distance = min(
                    distance,
                    mujoco.mj_geomDistance(
                        self.model, self.data, int(a), b, distance, None
                    ),
                )
                if distance < self.environment_clearance:
                    self.clearance_pair = (
                        self.model.geom(int(a)).name,
                        self.model.geom(b).name,
                        float(distance),
                    )
                    return float(distance)
        if self.payload is not None:
            held = self.payload["geom"]
            for b in self.environment_geoms:
                if b == held:
                    continue
                name = self.model.geom(b).name
                support = (
                    name in ("table_top", "shelf_bottom")
                    or name.startswith("tray_")
                    and name.endswith("_floor")
                )
                threshold = self.payload_contact_threshold(name, support)
                separation = mujoco.mj_geomDistance(
                    self.model, self.data, held, b, 0.02, None
                )
                if separation < threshold:
                    self.clearance_pair = (
                        self.model.geom(held).name,
                        name,
                        float(separation),
                    )
                    return -1.0
        return float(distance)

    def payload_contact_threshold(self, name, support):
        # Intentional carried-book contacts retain real collision response.
        dense_book_contact = "hard_layout" in self.env.scene_config and name.startswith("book_")
        return -0.0005 if support or dense_book_contact else self.environment_clearance

    def screen(self, q):
        accepted, reason, margin, fraction = super().screen(q)
        if not accepted and reason == "environment clearance":
            reason += f": {self.clearance_pair}"
        return accepted, reason, margin, fraction

    def set_payload(self, body_name):
        self.payload = None
        if body_name is None:
            return
        live = self.robot.data
        tool_quat = np.zeros(4)
        mujoco.mju_mat2Quat(tool_quat, live.site_xmat[self.robot.site_id])
        self.set_payload_pose(body_name, self.robot.tool_position, tool_quat)

    def set_payload_pose(self, body_name, tool_position, tool_quat):
        """A virtual grasp transform for scratch planning only."""
        spec = self.env.objects[body_name]
        live = self.robot.data
        rotation = rotation_matrix(tool_quat)
        inverse, relative = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(inverse, tool_quat)
        mujoco.mju_mulQuat(relative, inverse, live.xquat[spec["body"]])
        self.payload = dict(
            body=spec["body"],
            geom=spec["geoms"][0],
            qpos=spec["qpos"],
            offset=rotation.T @ (live.xpos[spec["body"]] - tool_position),
            quaternion=relative,
        )

    def payload_reference_bias(self, q):
        """Steady PD reference offset, expressed solely through public actions."""
        if self.payload is None:
            return np.zeros_like(q)
        self.pose(q)
        body = self.payload["body"]
        mujoco.mj_jac(
            self.model,
            self.data,
            self.jp,
            None,
            self.data.xipos[body],
            int(self.model.site_bodyid[self.robot.site_id]),
        )
        torque = self.jp[:, self.robot.dof_indices].T @ (
            -self.model.body_mass[body] * self.model.opt.gravity
        )
        return torque / self.robot.kp

    def solve_best(self, position, quaternion, seed):
        reasons = []
        base = self.model.body("forte_mount").pos
        yaw = np.arctan2(position[1] - base[1], position[0] - base[0])
        seeds = [seed, self.robot.home]
        for roll, wrist in ((0, 1.2), (2.8, -1.3), (-2.8, -1.3), (0, -0.5)):
            seeds.append(np.array([-yaw, -1.2, 0, -1.6, roll, -wrist, 0]))
        for q in seeds:
            result = self.solve(position, quaternion, q)
            if result.success:
                return result.q
            reasons.append(result.reason)
        raise ValueError(
            f"Unreachable/screened pose {np.round(position, 3).tolist()}: {reasons}"
        )

    def continuous_candidates(self, position, quaternion, seed):
        """Nearby redundant postures; callers still screen every connecting path."""
        result = self.solve(position, quaternion, seed)
        yield result
        self.pose(seed)
        mujoco.mj_jacSite(self.model, self.data, self.jp, self.jr, self.robot.site_id)
        jacobian = np.vstack([self.jp[:, self.robot.dof_indices],
                              .2 * self.jr[:, self.robot.dof_indices]])
        _, _, vectors = np.linalg.svd(jacobian, full_matrices=True)
        null = vectors[-1]
        for offset in (.2, -.2, .5, -.5, 1., -1.):
            yield self.solve(position, quaternion, seed + offset * null)


class BenchmarkExpert:
    """Finite-state expert with observable failure reasons and explicit reset.

    This object is privileged and must never be passed to a VLA as observations.
    It emits the exact same eight-value action contract used by other policies.
    """

    sorting_cartesian_minimum_steps = 5

    def __init__(self, env):
        self.env, self.robot = env, env.robot
        self.ik = BenchmarkIK(env)
        self.stage = "uninitialized"
        self.error = None
        self.failure_stage = None
        self.commands = deque()
        self.phase = 0
        # Keep the original cadence for books and manually initialized carry
        # continuations. Ordinary sorting selects its validated faster cadence
        # when building the approach; joint-reference speed limits still apply.
        self.cartesian_minimum_steps = 10

    def reset(self, seed=None):
        self.commands.clear()
        self.ik.set_payload(None)
        self.ik.target_contacts = False
        self.ik.releasing = False
        self.ik.planning_width = None
        self.stage = "approach"
        self.error = None
        self.failure_stage = None
        self.phase = 0
        self.cartesian_minimum_steps = 10
        self.width = self.robot.open_width
        self.quaternion = None
        self.close_retries = self.release_retries = 0
        self.lift_retries = 0
        self.grasp_loss_steps = 0
        self.planning_attempts = []
        self._build_approach()

    def _width_action(self, width):
        return (
            2
            * (width - self.robot.closed_width)
            / (self.robot.open_width - self.robot.closed_width)
            - 1
        )

    def _hold(self, seconds, width=None):
        width = self.width if width is None else width
        for _ in range(round(seconds / self.env.control_dt)):
            reference = (
                self.robot.target_q.copy()
                if not self.commands
                else self.commands[-1][0].copy()
            )
            geometry = self._planned_geometry()
            self.commands.append((reference, width, geometry))

    def _planned_geometry(self):
        # Joint references differ from measured joints under contact/load.
        # Treating references as actual poses can create fictitious collisions.
        return (
            self.commands[-1][2].copy()
            if self.commands
            else self.robot.arm_position.copy()
        )

    def _joint_segment(self, end, width, *, screen=True, minimum_steps=10):
        start = self._planned_geometry()
        start_reference = (
            self.robot.target_q.copy()
            if not self.commands
            else self.commands[-1][0].copy()
        )
        end_reference = end + self.ik.payload_reference_bias(end)
        if np.any(end_reference < self.robot.lower) or np.any(
            end_reference > self.robot.upper
        ):
            raise ValueError(f"{self.stage}: payload reference exceeds joint limits")
        speed = self.env.protocol["joint_reference_speed_rad_s"]
        steps = max(
            minimum_steps,
            int(
                np.ceil(
                    1.875
                    * np.max(np.abs(end_reference - start_reference))
                    / (speed * self.env.control_dt)
                )
            ),
        )
        if screen:
            for q in np.linspace(
                start, end, max(2, int(np.max(np.abs(end - start)) / 0.025) + 1)
            ):
                accepted, reason, _, _ = self.ik.screen(q)
                if not accepted:
                    raise ValueError(f"{self.stage}: path screening failed: {reason}")
        for t in np.linspace(1 / steps, 1, steps):
            blend = 10 * t**3 - 15 * t**4 + 6 * t**5
            self.commands.append(
                (
                    start_reference + blend * (end_reference - start_reference),
                    width,
                    start + blend * (end - start),
                )
            )

    def _cartesian_segment(self, position, quaternion, width):
        start_q = self._planned_geometry()
        start_pos, _ = self.ik.pose(start_q)
        q = start_q
        for p in np.linspace(
            start_pos,
            position,
            max(2, int(np.linalg.norm(position - start_pos) / 0.008) + 1),
        )[1:]:
            checkpoint = len(self.commands)
            reasons = []
            for result in self.ik.continuous_candidates(p, quaternion, q):
                if result.success:
                    try:
                        self._joint_segment(
                            result.q, width,
                            minimum_steps=self.cartesian_minimum_steps,
                        )
                        q = result.q
                        break
                    except ValueError as exc:
                        reasons.append(str(exc))
                        while len(self.commands) > checkpoint:
                            self.commands.pop()
                else:
                    reasons.append(result.reason)
            else:
                raise ValueError(
                    f"{self.stage}: Cartesian IK: {reasons}; point={np.round(p,4).tolist()}"
                )

    def _joint_route(self, end, width):
        """Try a direct move, then turn the shoulder before unfolding the arm."""
        count = len(self.commands)
        start = self._planned_geometry()
        try:
            self._joint_segment(end, width)
            return
        except ValueError as direct_error:
            reason = str(direct_error)
        for yaw in (end[0], (start[0] + end[0]) / 2):
            while len(self.commands) > count:
                self.commands.pop()
            intermediate = start.copy()
            intermediate[0] = yaw
            try:
                self._joint_segment(intermediate, width)
                self._joint_segment(end, width)
                return
            except ValueError:
                continue
        while len(self.commands) > count:
            self.commands.pop()
        raise ValueError(reason)

    def _build_approach(self):
        e = self.env
        object_pos = e.data.xpos[e.objects[e.target]["body"]].copy()
        if hasattr(e, "scene_config") and "trays" in e.scene_config:
            self.kind = "sorting"
            self.cartesian_minimum_steps = self.sorting_cartesian_minimum_steps
            self.destination = np.array(e.goal["position"])
            if abs(self.destination[1]) > 0.1:
                self.destination[:2] -= [0.035, 0.035 * np.sign(self.destination[1])]
            self.place_quaternion = downward_quaternion(
                np.arctan2(self.destination[1], self.destination[0])
            )
            self.grasp_position = object_pos.copy()
            low, high = e._bounds(e.target)
            # Keep the tips clear of support when grasping a low-profile object.
            self.grasp_position[2] = max(self.grasp_position[2], low[2] + 0.019)
            self.hover = object_pos + [0, 0, 0.14]
            yaw = float(
                np.arctan2(
                    e.data.xmat[e.objects[e.target]["body"]].reshape(3, 3)[1, 0],
                    e.data.xmat[e.objects[e.target]["body"]].reshape(3, 3)[0, 0],
                )
            )
            wedge = (
                next(
                    obj
                    for obj in e.scene_config["objects"]
                    if "object_" + obj["name"] == e.target
                )["shape"]
                == "wedge"
            )
            if wedge:
                # Pinch parallel side faces near the thick end of the wedge.
                self.grasp_position += e.data.xmat[e.objects[e.target]["body"]].reshape(
                    3, 3
                ) @ np.array([-0.012, 0, 0])
                self.grasp_position[2] = low[2] + 0.019
                self.hover = self.grasp_position + [0, 0, 0.14]
            reasons = []
            candidates = [
                (angle, pitch)
                for pitch in (np.pi / 2, np.deg2rad(115), np.deg2rad(130))
                for angle in (
                    (yaw, yaw + np.pi, 0.6, 0.3, -0.3)
                    if wedge
                    else (
                        0.0,
                        yaw,
                        0.3,
                        -0.3,
                        yaw + np.pi,
                        yaw + np.pi / 2,
                        yaw - np.pi / 2,
                    )
                )
            ]
            # This feasibility query is independent of the grasp candidate.
            # Do it once; unsuccessful approaches must not repeat the same
            # expensive destination search before trying a different grasp.
            destination = self.destination.copy()
            destination[2] = e.goal["surface_z"] + 0.10
            self.ik.solve_best(
                destination, self.place_quaternion, self.robot.arm_position
            )
            for candidate, pitch in candidates:
                self.quaternion = approach_quaternion(candidate, pitch)
                self.commands.clear()
                try:
                    q = self.ik.solve_best(
                        self.hover, self.quaternion, self.robot.arm_position
                    )
                    self._joint_segment(q, self.width)
                    self._cartesian_segment(
                        self.grasp_position, self.quaternion, self.width
                    )
                    break
                except ValueError as exc:
                    reasons.append(str(exc))
            else:
                raise ValueError("No sorting approach: " + "; ".join(reasons))
        else:
            self.kind = (
                "insert" if e.tasks[e.task_id]["operation"] == "insert" else "extract"
            )
            self._build_book_approach(object_pos)
        self._hold(0.35)

    def _book_goal_candidates(self):
        e = self.env
        half = np.array(e.objects[e.target]["half_extents"])
        if self.kind == "insert":
            goal = np.array(e.goal["position"])
            goal[0] += 0.020
            goal[2] += 0.004
            yield goal, np.array([1.0, 0, 0, 0])
            return
        # Rest an edge near the mat before releasing; gravity settles the book
        # flat. The final scorer still requires the original flat orientation.
        for angle in (55.0, 70.0, 40.0, 80.0):
            for sign, yaw in (
                (1.0, 0.0),
                (-1.0, 0.0),
                (1.0, -np.pi / 2),
                (-1.0, np.pi / 2),
                (1.0, np.pi),
            ):
                roll = sign * np.deg2rad(angle)
                turn = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
                quat = quaternion_product(
                    turn, np.array([np.cos(roll / 2), np.sin(roll / 2), 0, 0])
                )
                extent = np.abs(rotation_matrix(quat)) @ half
                goal = np.array(e.goal["position"])
                if yaw == 0:
                    # Use the far side of the existing mat to leave clearance
                    # between the folded forearm and the larger new base.
                    goal[0] = e.goal["high"][0] - half[0] - 0.005
                goal[2] = e.goal["surface_z"] + extent[2] + 0.005
                shift = sign * (
                    half[2] * (1 - np.sin(abs(roll))) + half[1] * np.cos(roll)
                )
                goal += rotation_matrix(turn) @ np.array([0, shift, 0])
                yield goal, quat

    def _book_tool_goal(self, body_position, body_quaternion):
        inverse = self.ik.payload["quaternion"] * np.array([1, -1, -1, -1])
        quat = quaternion_product(body_quaternion, inverse)
        return body_position - rotation_matrix(quat) @ self.ik.payload["offset"], quat

    def _book_grasp_candidates(self, object_position):
        half = np.array(self.env.objects[self.env.target]["half_extents"])
        for pitch in (15.0, 0.0, 30.0, -15.0, 45.0):
            for inset in (0.030, 0.020, 0.040, 0.050):
                heights = ((half[2] - 0.045, half[2] - 0.065, 0.0, half[2] - 0.025)
                           if self.kind == "extract" else (half[2] - 0.025, half[2] - 0.045, 0.0))
                for dz in heights:
                    yield (pitch, inset, dz, object_position + [-half[0] + inset, 0, dz],
                           approach_quaternion(0, np.deg2rad(pitch)))

    def _build_book_approach(self, object_position):
        e = self.env
        half = np.array(e.objects[e.target]["half_extents"])
        self.width = 2 * half[1] + 0.006
        reasons = []
        for pitch, inset, dz, position, quaternion in self._book_grasp_candidates(object_position):
            self.grasp_position, self.quaternion = position, quaternion
            self.commands.clear()
            self.ik.set_payload(None)
            self.ik.target_contacts = False
            self.ik.planning_width = self.width
            try:
                grasp_q = self.ik.solve_best(
                    self.grasp_position,
                    self.quaternion,
                    self.robot.arm_position,
                )
                self.ik.set_payload_pose(
                    e.target, self.grasp_position, self.quaternion
                )
                self.ik.target_contacts = True
                if e.task_id != "insert_orange":
                    self._screen_book_withdraw(grasp_q)
                for goal, quat in self._book_goal_candidates():
                    tool, tool_quat = self._book_tool_goal(goal, quat)
                    try:
                        self.ik.solve_best(tool, tool_quat, grasp_q)
                        self.book_destination, self.book_orientation = (
                            goal,
                            quat,
                        )
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(
                        "No reachable book placement for this grasp"
                    )
                self.ik.set_payload(None)
                self.ik.target_contacts = False
                self._hold(0.8, self.width)
                self.hover = (
                    self.grasp_position
                    - rotation_matrix(self.quaternion)[:, 0] * 0.045
                )
                high = self.hover + (
                    [0, 0, 0.15] if e.task_id == "insert_orange" else [0, 0, 0]
                )
                q = self.ik.solve_best(
                    high, self.quaternion, self.robot.arm_position
                )
                self._joint_route(q, self.width)
                if e.task_id == "insert_orange":
                    self._cartesian_segment(
                        self.hover, self.quaternion, self.width
                    )
                self._cartesian_segment(
                    self.grasp_position, self.quaternion, self.width
                )
                return
            except ValueError as exc:
                reasons.append(str(exc))
                self.planning_attempts.append(
                    dict(
                        pitch_deg=pitch,
                        inset_m=inset,
                        height_offset_m=dz,
                        error=str(exc),
                    )
                )
        self.ik.set_payload(None)
        raise ValueError("No book approach: " + "; ".join(reasons[-4:]))

    def _screen_book_withdraw(self, q):
        """Reject grasps that cannot lift and clear the shelf within joint limits."""
        shelf = self.env.scene_config["shelf"]
        front = shelf["center_xy_m"][0] - shelf["depth_m"] / 2
        start = self.grasp_position.copy()
        lifted = start + [0, 0, 0.025]
        withdrawn = lifted.copy()
        withdrawn[0] -= self.env._bounds(self.env.target)[1][0] - front + 0.012
        for endpoint in (lifted, withdrawn):
            for point in np.linspace(
                start,
                endpoint,
                max(2, int(np.linalg.norm(endpoint - start) / 0.008) + 1),
            )[1:]:
                result = self.ik.solve(point, self.quaternion, q)
                if not result.success:
                    raise ValueError("Book withdrawal preflight: " + result.reason)
                q = result.q
            start = endpoint

    def _advance(self):
        e = self.env
        if self.phase == 0:
            self.phase = 1
            self.stage = "close"
            self.ik.target_contacts = True
            self.ik.planning_width = None
            self.width = self.robot.closed_width
            self._hold(1.5, self.width)
        elif self.phase == 1:
            if not e._last_metrics["grasped"]:
                if self.close_retries < 2:
                    self.close_retries += 1
                    self.stage = "retry_close"
                    self._hold(1.5, self.width)
                    return
                raise ValueError("Grasp failed: no bilateral contact")
            self.phase = 2
            self.stage = "lift"
            self.ik.set_payload(e.target)
            lift = self.robot.tool_position + [
                0,
                0,
                0.075 if self.kind == "sorting" else 0.025,
            ]
            self._cartesian_segment(lift, self.quaternion, self.width)
            self._hold(0.3)
        elif self.phase == 2:
            if (
                self.kind != "sorting"
                and e._last_metrics["grasped"]
                and not e._episode["ever_lifted"]
                and self.lift_retries < 2
            ):
                self.lift_retries += 1
                self.ik.set_payload(e.target)
                self._cartesian_segment(
                    self.robot.tool_position + [0, 0, 0.01], self.quaternion, self.width
                )
                self._hold(0.3)
                return
            if not e._episode["ever_lifted"] or not e._last_metrics["grasped"]:
                raise ValueError("Lift failed or grasp slipped")
            self.phase = 3
            self.stage = "transfer"
            self.ik.set_payload(e.target)
            if self.kind != "sorting":
                checkpoint = len(self.commands)
                destinations = [(self.book_destination, self.book_orientation)]
                if self.kind == "extract":
                    destinations += list(self._book_goal_candidates())
                errors = []
                for destination, orientation in destinations:
                    self.book_destination, self.book_orientation = destination, orientation
                    approach_quat = self.quaternion.copy()
                    try:
                        self._book_transfer()
                        # A reachable hover is insufficient: check the descent
                        # before executing the carry, then replan it from the
                        # measured payload pose at the next phase.
                        carry_end = len(self.commands)
                        tool, quat = self._book_tool_goal(destination, orientation)
                        self._cartesian_segment(tool, quat, self.width)
                        while len(self.commands) > carry_end:
                            self.commands.pop()
                        break
                    except ValueError as exc:
                        errors.append(str(exc))
                        self.quaternion = approach_quat
                        while len(self.commands) > checkpoint:
                            self.commands.pop()
                else:
                    if not self._fallback_book_transfer():
                        raise ValueError("No complete book transfer: " + "; ".join(errors[-3:]))
                self._hold(0.25)
                return
            rotate = abs(np.dot(self.quaternion, self.place_quaternion)) < 0.99999
            self.quaternion = self.place_quaternion
            goal = self.destination.copy()
            goal[2] = e.goal["surface_z"] + 0.10
            tool = goal - rotation_matrix(self.quaternion) @ self.ik.payload["offset"]
            if rotate:
                q = self.ik.solve_best(tool, self.quaternion, self._planned_geometry())
                try:
                    self._joint_segment(q, self.width)
                except ValueError:
                    # Clear tall neighbouring objects before rotating a low grasp.
                    start_position, start_quaternion = self.ik.pose(
                        self._planned_geometry()
                    )
                    self._cartesian_segment(
                        start_position + [0, 0, 0.06], start_quaternion, self.width
                    )
                    raised_tool = tool + [0, 0, 0.04]
                    raised_q = self.ik.solve_best(
                        raised_tool, self.quaternion, self._planned_geometry()
                    )
                    self._joint_route(raised_q, self.width)
                    self._cartesian_segment(tool, self.quaternion, self.width)
            else:
                self._cartesian_segment(tool, self.quaternion, self.width)
            self._hold(0.25)
        elif self.phase == 3:
            self.phase = 4
            self.stage = "place"
            self.ik.set_payload(e.target)
            if self.kind != "sorting":
                tool, self.quaternion = self._book_tool_goal(
                    self.book_destination, self.book_orientation
                )
                self._cartesian_segment(tool, self.quaternion, self.width)
                self._hold(0.3)
                return
            rotation = e.data.site_xmat[self.robot.site_id].reshape(3, 3)
            low, _ = e._bounds(e.target)
            body_pos = e.data.xpos[e.objects[e.target]["body"]]
            goal = self.destination.copy()
            goal[2] = e.goal["surface_z"] + (body_pos[2] - low[2]) + 0.003
            tool = goal - rotation @ self.ik.payload["offset"]
            self._cartesian_segment(tool, self.quaternion, self.width)
            self._hold(0.3)
        elif self.phase == 4:
            self.phase = 5
            self.stage = "release"
            self.width = (
                self.robot.open_width
                if self.kind in ("sorting", "extract")
                else 2 * e.objects[e.target]["half_extents"][1] + 0.010
            )
            if self.kind != "sorting":
                # Remove the loaded reference offset as the shelf/mat takes
                # the weight, preventing an upward kick when the jaws open.
                self.ik.set_payload(None)
                widths = (
                    [self.width]
                    if self.kind == "extract"
                    else [
                        2 * e.objects[e.target]["half_extents"][1] + gap
                        for gap in (0.010, 0.008, 0.006)
                    ]
                )
                for width in widths:
                    self.ik.planning_width = width
                    try:
                        self._joint_segment(self.robot.arm_position.copy(), width)
                        self.width = width
                        break
                    except ValueError as exc:
                        last_error = exc
                else:
                    raise last_error
                self.ik.planning_width = None
            self._hold(1.5 if self.kind == "sorting" else 0.6, self.width)
        elif self.phase == 5:
            if e._last_metrics["grasped"] and self.release_retries < 2:
                self.release_retries += 1
                self.stage = "retry_release"
                self._hold(1.5, self.width)
                return
            if e._last_metrics["grasped"]:
                raise ValueError("Release failed: object remains grasped")
            self.phase = 6
            self.stage = "retreat"
            self.ik.set_payload(None)
            self.ik.releasing = self.kind != "sorting"
            offset = (
                [0, 0, 0.12]
                if self.kind == "sorting"
                else (
                    [-0.12 if e.task_id == "insert_orange" else -0.06, 0, 0]
                    if self.kind == "insert"
                    else [0, 0, 0.12]
                )
            )
            checkpoint = len(self.commands)
            offsets = [offset]
            if self.kind != "sorting":
                # Clear the falling book while avoiding a folded forearm/base
                # collision. Each alternative retains ordinary path screening.
                offsets = (offset, [.07, 0, .10], [.05, .06, .10],
                           [.05, -.06, .10]) if self.kind == "extract" else (
                               [-.04, 0, .01], [-.06, 0, 0], offset)
            errors = []
            for displacement in offsets:
                try:
                    retreat = self.robot.tool_position + displacement
                    self._cartesian_segment(retreat, self.quaternion, self.width)
                    break
                except ValueError as exc:
                    errors.append(str(exc))
                    while len(self.commands) > checkpoint:
                        self.commands.pop()
            else:
                raise ValueError("No screened release retreat: " + "; ".join(errors))
            self._hold(0.75)
        else:
            self.stage = "settle"
            self._hold(0.5)

    def _fallback_book_transfer(self):
        """Optional complete, screened route after nominal candidates fail."""
        return False

    def _book_transfer(self):
        e = self.env
        if self.kind == "extract" or e.task_id.startswith("relocate_"):
            self.stage = "withdraw"
            front = (
                e.scene_config["shelf"]["center_xy_m"][0]
                - e.scene_config["shelf"]["depth_m"] / 2
            )
            retreat = self.robot.tool_position.copy()
            retreat[0] -= e._bounds(e.target)[1][0] - front + 0.035
            self._cartesian_segment(retreat, self.quaternion, self.width)
        self.stage = "carry"
        body_hover = self.book_destination.copy()
        if self.kind == "insert":
            front = (e.scene_config["shelf"]["center_xy_m"][0]
                     - e.scene_config["shelf"]["depth_m"] / 2)
            body_hover[0] = front - e.objects[e.target]["half_extents"][0] - .025
            body_hover[2] += .02
        else:
            body_hover[2] += 0.065
        tool, quaternion = self._book_tool_goal(body_hover, self.book_orientation)
        if self.kind == "extract":
            # The seven-joint model has a narrow configured base-yaw range.
            # First try rolling in front of the cabinet instead of routing the
            # book around its side. All intermediate poses retain the ordinary
            # payload, joint-limit, torque, and collision checks.
            checkpoint = len(self.commands)
            try:
                self.ik.pose(self._planned_geometry())
                forward = self.ik.data.xpos[e.objects[e.target]["body"]].copy()
                forward[0] = e.goal["position"][0] + 0.07
                forward[2] = e.table_surface + 0.26
                self._book_body_segment(forward, np.array([1.0, 0, 0, 0]))
                self._book_body_segment(forward, self.book_orientation)
                self._book_body_segment(body_hover, self.book_orientation)
                self.quaternion = quaternion
                return
            except ValueError as exc:
                self.planning_attempts.append({"route": "front_of_shelf", "error": str(exc)})
                while len(self.commands) > checkpoint:
                    self.commands.pop()
            # Roll after clearing the shelf, before sweeping the long book in yaw.
            self.ik.pose(self._planned_geometry())
            position = self.ik.data.xpos[e.objects[e.target]["body"]].copy()
            # A transverse move close to the base can cross an elbow-limit
            # region. Raise outside the cabinet before traversing its centre.
            if position[1] < e.scene_config["shelf"]["center_xy_m"][1] + 0.12:
                position[2] = e.table_surface + 0.34
                self._book_body_segment(position, np.array([1.0, 0, 0, 0]))
                position[1] = (
                    e.scene_config["shelf"]["center_xy_m"][1]
                    + e.scene_config["shelf"]["width_m"] / 2
                    + 0.03
                )
                upright_tool, upright_quat = self._book_tool_goal(
                    position, np.array([1.0, 0, 0, 0])
                )
                self._joint_route(
                    self.ik.solve_best(
                        upright_tool, upright_quat, self._planned_geometry()
                    ),
                    self.width,
                )
            position[1] = (
                e.scene_config["shelf"]["center_xy_m"][1]
                + e.scene_config["shelf"]["width_m"] / 2
                + 0.03
            )
            position[2] = e.table_surface + 0.26
            self._book_body_segment(position, np.array([1.0, 0, 0, 0]))
            rotation = rotation_matrix(self.book_orientation)
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
            inverse_turn = np.array([np.cos(yaw / 2), 0, 0, -np.sin(yaw / 2)])
            roll = quaternion_product(inverse_turn, self.book_orientation)
            self._book_body_segment(position, roll)
            # Raise the rolled book in free space before the joint-space turn;
            # a low turn can sweep the long book through the table or cabinet.
            position[0] = e.goal["position"][0] + 0.07
            position[2] = e.goal["surface_z"] + 0.34
            self._book_body_segment(position, roll)
            q = self.ik.solve_best(tool, quaternion, self._planned_geometry())
            self._joint_route(q, self.width)
        else:
            checkpoint = len(self.commands)
            try:
                self._cartesian_segment(tool, quaternion, self.width)
            except ValueError as direct_error:
                while len(self.commands) > checkpoint:
                    self.commands.pop()
                self.planning_attempts.append({"route": "direct_insertion", "error": str(direct_error)})
                # Pass the table book beside its spine before entering the
                # empty shelf bay. Screen every segment and every payload pose.
                errors = []
                for side in (.065, -.065, .09, -.09):
                    try:
                        self.ik.pose(self._planned_geometry())
                        entry = self.ik.data.xpos[e.objects[e.target]["body"]].copy()
                        entry[1] = self.book_destination[1] + side
                        self._book_body_segment(entry, self.book_orientation)
                        entry[0] = self.book_destination[0]
                        entry[2] = body_hover[2]
                        self._book_body_segment(entry, self.book_orientation)
                        entry[1] = self.book_destination[1]
                        self._book_body_segment(entry, self.book_orientation)
                        break
                    except ValueError as exc:
                        errors.append(str(exc))
                        while len(self.commands) > checkpoint:
                            self.commands.pop()
                else:
                    raise ValueError("No screened shelf entry: " + "; ".join(errors))
        self.quaternion = quaternion

    def _book_body_segment(self, position, quaternion):
        """Interpolate the carried book pose, avoiding wide joint-space sweeps."""
        q = self._planned_geometry()
        self.ik.pose(q)
        body = self.env.objects[self.env.target]["body"]
        start = self.ik.data.xpos[body].copy()
        start_quat = self.ik.data.xquat[body].copy()
        angle = 2 * np.arccos(np.clip(abs(np.dot(start_quat, quaternion)), 0, 1))
        steps = max(
            2,
            int(np.ceil(np.linalg.norm(position - start) / 0.008)),
            int(np.ceil(angle / 0.07)),
        )
        for t in np.linspace(1 / steps, 1, steps):
            center = start + t * (position - start)
            orientation = interpolate_quaternion(start_quat, quaternion, t)
            tool, tool_quat = self._book_tool_goal(center, orientation)
            checkpoint = len(self.commands)
            reasons = []
            for result in self.ik.continuous_candidates(tool, tool_quat, q):
                if result.success:
                    try:
                        self._joint_segment(result.q, self.width, minimum_steps=5)
                        q = result.q
                        break
                    except ValueError as exc:
                        reasons.append(str(exc))
                        while len(self.commands) > checkpoint:
                            self.commands.pop()
                else:
                    reasons.append(result.reason)
            else:
                raise ValueError(
                    f"{self.stage}: book-pose path: {reasons}; body={np.round(center,4).tolist()}, quat={np.round(orientation,3).tolist()}"
                )

    def __call__(self, observation):
        if self.error is not None:
            return np.r_[
                np.zeros(len(self.robot.home)), self._width_action(self.width)
            ].astype(np.float32)
        try:
            if self.phase in (2, 3, 4):
                self.grasp_loss_steps = (
                    0
                    if self.env._last_metrics["grasped"]
                    else self.grasp_loss_steps + 1
                )
                if self.grasp_loss_steps >= 3:
                    raise ValueError("Object slipped before commanded release")
            if not self.commands:
                self._advance()
            target, width, _ = self.commands.popleft()
            delta = target - self.robot.target_q
            scale = (
                self.env.protocol["joint_reference_speed_rad_s"] * self.env.control_dt
            )
            return np.r_[
                np.clip(delta / scale, -1, 1), self._width_action(width)
            ].astype(np.float32)
        except ValueError as exc:
            self.error = str(exc)
            self.failure_stage = self.stage
            self.stage = "failed"
            return np.r_[
                np.zeros(len(self.robot.home)), self._width_action(self.width)
            ].astype(np.float32)
