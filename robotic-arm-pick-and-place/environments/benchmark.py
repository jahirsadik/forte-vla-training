"""Gymnasium benchmark contract shared by the Forte manipulation tasks.

Actions move bounded joint references, never objects. Privileged state is
available explicitly to experts and scoring, not in policy observations.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import string

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from environments.bookshelf import load_bookshelf, render_options, robot_geom_ids
from environments.geometry import object_bounds
from utils.contracts import action_contract, implementation_provenance
from utils.actions import decode_action

PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL = PROJECT / "configs/benchmarks.json"
DEFAULT_IMAGE_SIZE = (480, 640)  # Height, width; native capture, before model preprocessing.


class ManipulationBenchmark(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}
    protocol_file = PROTOCOL

    def __init__(
        self,
        scene,
        objects,
        tasks,
        *,
        observation_mode="proprio",
        reward_mode="sparse",
        render_mode=None,
        horizon_steps=None,
        cameras=("front", "wrist"),
        image_size=DEFAULT_IMAGE_SIZE,
    ):
        super().__init__()
        if observation_mode not in ("proprio", "rgb"):
            raise ValueError("observation_mode must be proprio or rgb")
        if reward_mode not in ("sparse", "dense"):
            raise ValueError("reward_mode must be sparse or dense")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError("Unsupported render mode")
        self.protocol = json.loads(self.protocol_file.read_text())
        self.robot = load_bookshelf(scene)
        self.model, self.data = self.robot.model, self.robot.data
        self.objects, self.tasks = deepcopy(objects), deepcopy(tasks)
        self.object_names = tuple(self.objects)
        self.task_ids = tuple(self.tasks)
        self.observation_mode, self.reward_mode = observation_mode, reward_mode
        self.render_mode, self.cameras = render_mode, tuple(cameras)
        if len(image_size) != 2 or any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
            for v in image_size
        ):
            raise ValueError("image_size must contain integer height and width")
        self.image_size = tuple(int(v) for v in image_size)
        h, w = self.image_size
        if (
            not 0 < h <= self.model.vis.global_.offheight
            or not 0 < w <= self.model.vis.global_.offwidth
        ):
            raise ValueError("Image size exceeds the scene framebuffer")
        for name in (*self.cameras, "overview"):
            self.model.camera(name)
        self.horizon = (
            self.protocol["horizon_steps"] if horizon_steps is None else horizon_steps
        )
        if not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError("horizon_steps must be a positive integer")
        self.control_dt = 1 / self.protocol["control_hz"]
        self.substeps = round(self.control_dt / self.model.opt.timestep)
        if not np.isclose(self.substeps * self.model.opt.timestep, self.control_dt):
            raise ValueError(
                "Control interval must be an integer number of physics steps"
            )
        self.robot_geoms = robot_geom_ids(self.model)
        self.robot_collision_geoms = [
            g for g in self.robot_geoms if self.model.geom_contype[g]
        ]
        self.finger_geoms = self.robot.gripper.contact_geoms
        for name, spec in self.objects.items():
            body = self.model.body(name).id
            joint = self.model.body_jntadr[body]
            spec.update(
                body=body,
                qpos=int(self.model.jnt_qposadr[joint]),
                dof=int(self.model.jnt_dofadr[joint]),
            )
            spec["geoms"] = [
                g
                for g in range(self.model.ngeom)
                if self.model.geom_bodyid[g] == body and self.model.geom_contype[g]
            ]
        n = len(self.robot.joint_names)
        self.action_space = spaces.Box(-1, 1, (n + 1,), dtype=np.float32)
        observation = {
            "proprio": spaces.Box(-np.inf, np.inf, (2 * n + 9,), dtype=np.float32),
            "instruction": spaces.Text(
                220, min_length=1, charset=string.ascii_letters + string.digits + " ,.-"
            ),
        }
        if observation_mode == "rgb":
            observation["rgb"] = spaces.Dict(
                {name: spaces.Box(0, 255, (h, w, 3), np.uint8) for name in self.cameras}
            )
        self.observation_space = spaces.Dict(observation)
        self._renderer = self._viewer = None
        self._episode = None
        self.action_contract = action_contract(self)
        self.implementation = implementation_provenance()
        # Compiled dynamics, geometry, textures and settings identify replay compatibility.
        binary = np.empty(mujoco.mj_sizeModel(self.model), dtype=np.uint8)
        mujoco.mj_saveModel(self.model, None, binary)
        signature = dict(
            protocol=self.protocol,
            tasks=self.tasks,
            reward=reward_mode,
            horizon=self.horizon,
            observations=observation_mode,
            cameras=self.cameras,
            image_size=self.image_size,
            controller=json.loads((PROJECT / "configs/forte.json").read_text()),
            implementation=self.implementation["source_sha256"],
            action_contract=self.action_contract,
        )
        self.fingerprint = hashlib.sha256(
            binary.tobytes() + json.dumps(signature, sort_keys=True).encode()
        ).hexdigest()

    def _sample_layout(self, options):
        raise NotImplementedError

    def _make_goal(self, task):
        raise NotImplementedError

    def _set_object_pose(self, name, position, quaternion=(1, 0, 0, 0)):
        """Reset-time placement only; step() never writes an object's coordinates."""
        spec = self.objects[name]
        self.data.qpos[spec["qpos"] : spec["qpos"] + 7] = [*position, *quaternion]
        self.data.qvel[spec["dof"] : spec["dof"] + 6] = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = {} if options is None else dict(options)
        unknown = set(options) - {"task_id", "randomize", "clutter_level", "layout"}
        if unknown:
            raise ValueError(f"Unknown reset options: {sorted(unknown)}")
        # Consume the same draw even for a selected task: paired instructions
        # with the same seed produce exactly the same physical initial state.
        sampled = self.task_ids[int(self.np_random.integers(len(self.task_ids)))]
        task_id = options.get("task_id", sampled)
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task {task_id!r}; choose from {self.task_ids}")
        # A failed reset must not leave an earlier episode's scoring history
        # active on top of newly reset or partially sampled physics.
        self._episode = None
        for attempt in range(self.protocol["reset_attempts"]):
            self.robot.reset()
            self._sample_layout(options)
            mujoco.mj_forward(self.model, self.data)
            for _ in range(
                round(self.protocol["settle_seconds"] / self.model.opt.timestep)
            ):
                self.robot.step()
            penetration = max([0.0, *[-float(c.dist) for c in self.data.contact]])
            stable = all(
                np.linalg.norm(self.data.qvel[s["dof"] : s["dof"] + 6]) < 0.05
                for s in self.objects.values()
            )
            if penetration <= self.protocol["reset_penetration_tolerance_m"] and stable:
                break
        else:
            raise RuntimeError("Could not sample a stable nonpenetrating reset")
        self.data.time = 0.0
        self.task_id = task_id
        self.goal = self._make_goal(self.tasks[task_id])
        self.instruction = self.tasks[task_id]["instruction"]
        self.target = self.tasks[task_id]["object"]
        self._episode = dict(
            steps=0,
            done=False,
            ever_grasped=False,
            ever_lifted=False,
            dwell=0.0,
            episode_return=0.0,
            wrong_grasp_steps=0,
            peak_contact_force_n=0.0,
            peak_penetration_m=0.0,
            peak_torque_fraction=0.0,
            peak_distractor_translation_m=0.0,
            peak_distractor_rotation_rad=0.0,
            initial_positions={
                n: self.data.xpos[s["body"]].tolist() for n, s in self.objects.items()
            },
            initial_quaternions={
                n: self.data.xquat[s["body"]].tolist() for n, s in self.objects.items()
            },
            initial_bottoms={n: float(self._bounds(n)[0][2]) for n in self.objects},
            reset_attempts=attempt + 1,
            options=options,
        )
        self._previous_potential = self._potential()
        self._last_metrics = self._measure()
        self._last_metrics.update(success=False, failure_reason=None, timeout=False)
        return self._observe(), self._info()

    def _bounds(self, name):
        return object_bounds(self.model, self.data, self.objects[name]["geoms"])

    def _contacts(self):
        by_object = {
            n: {"l": False, "r": False, "robot": False, "supports": set()}
            for n in self.objects
        }
        peak_force = peak_penetration = 0.0
        force = np.zeros(6)
        geom_object = {g: n for n, s in self.objects.items() for g in s["geoms"]}
        for i, contact in enumerate(self.data.contact):
            a, b = int(contact.geom1), int(contact.geom2)
            if a in self.robot_geoms or b in self.robot_geoms:
                peak_penetration = max(peak_penetration, -float(contact.dist))
            mujoco.mj_contactForce(self.model, self.data, i, force)
            normal = abs(float(force[0]))
            if a in self.robot_geoms or b in self.robot_geoms:
                peak_force = max(peak_force, float(np.linalg.norm(force[:3])))
            if normal < self.protocol["contact_force_threshold_n"]:
                continue
            for obj, other in ((geom_object.get(a), b), (geom_object.get(b), a)):
                if obj is None:
                    continue
                c = by_object[obj]
                c["robot"] |= other in self.robot_geoms
                for side in ("l", "r"):
                    c[side] |= other in self.finger_geoms[side]
                c["supports"].add(other)
        return by_object, peak_force, peak_penetration

    def _orientation_error(self):
        rotation = self.data.xmat[self.objects[self.target]["body"]].reshape(3, 3)
        if self.goal["orientation"] == "flat":
            return float(np.arccos(np.clip(abs(rotation[2, 1]), 0, 1)))
        if self.goal["orientation"] == "upright":
            return float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
        return 0.0

    def _measure(self):
        p, e = self.protocol, self._episode
        spec = self.objects[self.target]
        low, high = self._bounds(self.target)
        contacts, force, penetration = self._contacts()
        target_contact = contacts[self.target]
        within = bool(
            np.all(low >= self.goal["low"]) and np.all(high <= self.goal["high"])
        )
        supported = (
            self.model.geom(self.goal["support"]).id in target_contact["supports"]
        )
        supported &= (
            abs(low[2] - self.goal["surface_z"]) <= p["support_height_tolerance_m"]
        )
        linear = float(np.linalg.norm(self.data.qvel[spec["dof"] : spec["dof"] + 3]))
        angular = float(
            np.linalg.norm(self.data.qvel[spec["dof"] + 3 : spec["dof"] + 6])
        )
        orientation = self._orientation_error()
        stable = (
            linear <= p["linear_speed_tolerance_m_s"]
            and angular <= p["angular_speed_tolerance_rad_s"]
        )
        clearance = 0.015
        if within:
            for a in spec["geoms"]:
                for b in self.robot_collision_geoms:
                    clearance = min(
                        clearance,
                        mujoco.mj_geomDistance(
                            self.model, self.data, a, b, clearance, None
                        ),
                    )
        released = not target_contact["robot"] and clearance >= p["release_clearance_m"]
        displacement = rotation_error = 0.0
        dropped = []
        for name, other in self.objects.items():
            if self._bounds(name)[0][2] < self.table_surface - p["drop_below_table_m"]:
                dropped.append(name)
            if name == self.target:
                continue
            displacement = max(
                displacement,
                float(
                    np.linalg.norm(
                        self.data.xpos[other["body"]] - e["initial_positions"][name]
                    )
                ),
            )
            dot = abs(
                np.dot(self.data.xquat[other["body"]], e["initial_quaternions"][name])
            )
            rotation_error = max(
                rotation_error, float(2 * np.arccos(np.clip(dot, 0, 1)))
            )
        return dict(
            goal_distance_m=float(
                np.linalg.norm(self.data.xpos[spec["body"]] - self.goal["position"])
            ),
            orientation_error_rad=orientation,
            within_goal=within,
            supported=bool(supported),
            released=bool(released),
            stable=bool(stable),
            linear_speed_m_s=linear,
            angular_speed_rad_s=angular,
            grasped=target_contact["l"] and target_contact["r"],
            distractor_translation_m=displacement,
            distractor_rotation_rad=rotation_error,
            dropped_objects=dropped,
            contact_force_n=force,
            robot_penetration_m=penetration,
            placement_valid=bool(
                within
                and supported
                and released
                and stable
                and orientation <= np.deg2rad(p["orientation_tolerance_deg"])
            ),
        )

    def _potential(self):
        position = self.data.xpos[self.objects[self.target]["body"]]
        reach = np.exp(-np.linalg.norm(self.robot.tool_position - position) / 0.12)
        place = np.exp(-np.linalg.norm(position - self.goal["position"]) / 0.15)
        align = np.exp(-self._orientation_error())
        return float(0.2 * reach + 0.4 * place + 0.1 * align * place)

    def step(self, action):
        if self._episode is None or self._episode["done"]:
            raise gym.error.ResetNeeded(
                "Call reset before stepping or after an episode ends"
            )
        action = np.asarray(action, dtype=np.float64)
        if (
            action.shape != self.action_space.shape
            or not np.isfinite(action).all()
            or np.any(np.abs(action) > 1)
        ):
            raise ValueError("Action must be a finite normalized vector in [-1, 1]")
        e, p = self._episode, self.protocol
        start_q = self.robot.target_q.copy()
        start_width = self.robot.gripper_target
        target, next_width = decode_action(action, start_q, start_width, self.action_contract)
        width_delta = next_width - start_width
        physics_before = self._physics_state()
        touched_target = wrong_grasp = False
        numerical_failure = False
        for i in range(self.substeps):
            # A clipped endpoint can interpolate one ULP beyond a limit (most
            # visibly the shoulder-pitch upper bound of zero). Preserve the
            # existing interior arithmetic and clamp only this roundoff.
            reference = np.clip(
                start_q + (target - start_q) * (i + 1) / self.substeps,
                self.robot.lower, self.robot.upper,
            )
            self.robot.set_arm_target(
                reference,
                (target - start_q) / self.control_dt,
            )
            self.robot.set_gripper_width(
                start_width + width_delta * (i + 1) / self.substeps,
                width_delta / (2 * self.control_dt),
            )
            try:
                self.robot.step()
            except RuntimeError:
                mujoco.mj_resetData(self.model, self.data)
                mujoco.mj_setState(
                    self.model,
                    self.data,
                    physics_before,
                    mujoco.mjtState.mjSTATE_INTEGRATION,
                )
                mujoco.mj_forward(self.model, self.data)
                numerical_failure = True
                break
            contacts, force, penetration = self._contacts()
            c = contacts[self.target]
            grasped = c["l"] and c["r"]
            e["ever_grasped"] |= grasped
            e["ever_lifted"] |= bool(
                grasped
                and self._bounds(self.target)[0][2]
                > e["initial_bottoms"][self.target] + p["lift_height_m"]
            )
            touched_target |= c["robot"]
            wrong_grasp |= any(
                c["l"] and c["r"] for n, c in contacts.items() if n != self.target
            )
            e["peak_contact_force_n"] = max(e["peak_contact_force_n"], force)
            e["peak_penetration_m"] = max(e["peak_penetration_m"], penetration)
            e["peak_torque_fraction"] = max(
                e["peak_torque_fraction"],
                float(
                    np.max(np.abs(self.robot.last_effort) / self.robot.effort_limits)
                ),
            )
        e["steps"] += 1
        e["wrong_grasp_steps"] += int(wrong_grasp)
        metrics = self._measure()
        e["peak_distractor_translation_m"] = max(
            e["peak_distractor_translation_m"], metrics["distractor_translation_m"]
        )
        e["peak_distractor_rotation_rad"] = max(
            e["peak_distractor_rotation_rad"], metrics["distractor_rotation_rad"]
        )
        reason = None
        if numerical_failure:
            reason = "numerical_failure"
        elif metrics["dropped_objects"]:
            reason = "object_dropped"
        elif e["peak_distractor_translation_m"] > p[
            "distractor_translation_limit_m"
        ] or e["peak_distractor_rotation_rad"] > np.deg2rad(
            p["distractor_rotation_limit_deg"]
        ):
            reason = "distractor_disturbed"
        elif e["peak_penetration_m"] > p["penetration_failure_m"]:
            reason = "excessive_robot_penetration"
        eligible = (
            metrics["placement_valid"]
            and e["ever_grasped"]
            and e["ever_lifted"]
            and not touched_target
            and reason is None
        )
        e["dwell"] = e["dwell"] + self.control_dt if eligible else 0.0
        success = e["dwell"] + 1e-9 >= p["success_dwell_seconds"]
        terminated = bool(success or reason)
        truncated = bool(e["steps"] >= self.horizon and not terminated)
        sparse = float(success) - float(reason is not None)
        potential = 0.0 if terminated else self._potential()
        shaping = (
            potential - self._previous_potential if self.reward_mode == "dense" else 0.0
        )
        reward = sparse + shaping
        self._previous_potential = potential
        e["done"] = terminated or truncated
        e["episode_return"] += reward
        metrics.update(
            success=success,
            failure_reason=reason,
            timeout=truncated,
            reward_sparse=sparse,
            reward_shaping=shaping,
        )
        self._last_metrics = metrics
        if self.render_mode == "human":
            self.render()
        return self._observe(), reward, terminated, truncated, self._info()

    def _observe(self):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.robot.site_id])
        gripper = self.robot.gripper
        proprio = np.r_[
            self.robot.arm_position,
            self.robot.arm_velocity,
            gripper.width_from_position(self.data.qpos[gripper.driver_qpos]),
            2 * self.data.qvel[gripper.driver_dof],
            self.robot.tool_position,
            quat,
        ].astype(np.float32)
        obs = {"proprio": proprio, "instruction": self.instruction}
        if self.observation_mode == "rgb":
            obs["rgb"] = {name: self._image(name) for name in self.cameras}
        return obs

    def _info(self):
        e = self._episode
        return dict(
            task_id=self.task_id,
            protocol_version=self.protocol["protocol_version"],
            model_fingerprint=self.fingerprint,
            is_success=self._last_metrics["success"],
            metrics=deepcopy(self._last_metrics),
            episode={
                k: deepcopy(v) for k, v in e.items() if not k.startswith("initial_")
            },
            simulated_seconds=float(self.data.time),
        )

    def expert_state(self):
        """Privileged state for experts/diagnostics; intentionally absent from obs."""
        if self._episode is None:
            raise gym.error.ResetNeeded("Reset before requesting expert state")
        return dict(
            task_id=self.task_id,
            target=self.target,
            goal=deepcopy(self.goal),
            objects={
                n: dict(
                    position=self.data.xpos[s["body"]].copy(),
                    quaternion=self.data.xquat[s["body"]].copy(),
                )
                for n, s in self.objects.items()
            },
        )

    def _physics_state(self):
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.model, kind))
        mujoco.mj_getState(self.model, self.data, state, kind)
        return state

    def get_state(self):
        """JSON-serializable replay checkpoint including controller, RNG and scoring history."""
        if self._episode is None:
            raise gym.error.ResetNeeded("Reset before saving a checkpoint")
        return dict(
            fingerprint=self.fingerprint,
            physics=self._physics_state().tolist(),
            task_id=self.task_id,
            episode=deepcopy(self._episode),
            metrics=deepcopy(self._last_metrics),
            potential=self._previous_potential,
            rng=deepcopy(self.np_random.bit_generator.state),
            target_q=self.robot.target_q.tolist(),
            target_dq=self.robot.target_dq.tolist(),
            gripper_target=self.robot.gripper_target,
            gripper_speed=self.robot.gripper_speed,
        )

    def set_state(self, state):
        if state["fingerprint"] != self.fingerprint:
            raise ValueError(
                "Checkpoint model/protocol does not match this environment"
            )
        physics = np.asarray(state["physics"], dtype=float)
        if (
            physics.shape != self._physics_state().shape
            or not np.isfinite(physics).all()
        ):
            raise ValueError("Invalid checkpoint physics state")
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(
            self.model, self.data, physics, mujoco.mjtState.mjSTATE_INTEGRATION
        )
        self.robot.set_payload(None)
        self.robot.set_arm_target(state["target_q"], state["target_dq"])
        self.robot.set_gripper_width(state["gripper_target"], state["gripper_speed"])
        mujoco.mj_forward(self.model, self.data)
        self.task_id = state["task_id"]
        self.target = self.tasks[self.task_id]["object"]
        self.instruction = self.tasks[self.task_id]["instruction"]
        self.goal = self._make_goal(self.tasks[self.task_id])
        self._episode, self._last_metrics = (
            deepcopy(state["episode"]),
            deepcopy(state["metrics"]),
        )
        self._previous_potential = state["potential"]
        self.np_random.bit_generator.state = deepcopy(state["rng"])
        return self._observe(), self._info()

    def _image(self, camera):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.image_size[0], width=self.image_size[1]
            )
        self._renderer.update_scene(
            self.data, camera=camera, scene_option=render_options()
        )
        return self._renderer.render().copy()

    def render(self):
        if self.render_mode == "human":
            from mujoco import viewer

            if self._viewer is None:
                self._viewer = viewer.launch_passive(self.model, self.data)
                self._viewer.opt.geomgroup[:] = render_options().geomgroup
                self._viewer.opt.sitegroup[:] = 0
                self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self._viewer.cam.fixedcamid = self.model.camera("overview").id
            self._viewer.sync()
            return None
        return self._image("overview")

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
