"""Language-selected sorting of six rigid props into three open receptacles."""

import numpy as np
import mujoco

from environments.benchmark import ManipulationBenchmark, PROJECT
from environments.variants import scene_variant

SCENE = PROJECT / "assets/sorting/scene.xml"


class SortingEnv(ManipulationBenchmark):
    def __init__(self, *, asset_set="nominal", **kwargs):
        scene, self.scene_config = scene_variant("sorting", asset_set)
        self.asset_set = asset_set
        self.table_surface = self.scene_config["table"]["surface_z_m"]
        objects, tasks = {}, {}
        for obj in self.scene_config["objects"]:
            name = "object_" + obj["name"]
            objects[name] = dict(half_extents=(np.array(obj["size_m"]) / 2).tolist())
            for tray in self.scene_config["trays"]:
                tasks[obj["name"] + "_to_" + tray["name"]] = dict(
                    object=name,
                    tray=tray["name"],
                    instruction=f"Pick up the {obj['description']} and place it in the {tray['name']} tray. Leave the other objects.",
                )
        super().__init__(scene, objects, tasks, **kwargs)

    def _sample_layout(self, options):
        level = options.get("clutter_level", 1)
        if level not in (0, 1, 2):
            raise ValueError("clutter_level must be 0, 1, or 2")
        randomize = options.get("randomize", True)
        cfg = self.scene_config["layout"]
        weight = level / 2
        xs = (1 - weight) * np.array(cfg["x_m"]) + weight * np.array(cfg["clutter_x_m"])
        ys = (1 - weight) * np.array(cfg["y_m"]) + weight * np.array(cfg["clutter_y_m"])
        slots = np.array([(x, y) for y in ys for x in xs])
        layout = options.get("layout", "nominal")
        if layout == "arc":
            angles = np.arange(6) * np.pi / 3
            slots = np.column_stack((.305 + .115 * np.cos(angles), -.04 + .115 * np.sin(angles)))
        elif layout not in ("nominal", "shuffled"):
            raise ValueError("Unknown sorting layout")
        if randomize:
            slots = slots[self.np_random.permutation(len(slots))]
        occupied = []
        for obj, slot in zip(self.scene_config["objects"], slots, strict=True):
            half = np.array(obj["size_m"]) / 2
            radius = np.linalg.norm(half)
            for _ in range(100):
                xy = slot + (
                    self.np_random.uniform(-cfg["jitter_m"], cfg["jitter_m"], 2)
                    if randomize
                    else 0
                )
                if all(
                    np.linalg.norm(xy - prev) > radius + r + cfg["minimum_gap_m"]
                    for prev, r in occupied
                ):
                    break
            else:
                raise RuntimeError("Could not sample separated sorting objects")
            occupied.append((xy, radius))
            yaw = self.np_random.uniform(-np.pi, np.pi) if randomize else 0
            quaternion = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            if obj["shape"] == "capsule":
                q = quaternion.copy()
                mujoco.mju_mulQuat(
                    quaternion, q, np.array([np.sqrt(0.5), 0, np.sqrt(0.5), 0])
                )
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, quaternion)
            extent_z = np.abs(rotation.reshape(3, 3)[2]) @ half
            self._set_object_pose(
                "object_" + obj["name"],
                [*xy, self.table_surface + extent_z + 0.0005],
                quaternion,
            )

    def _make_goal(self, task):
        tray = self.scene_config["tray"]
        entry = next(t for t in self.scene_config["trays"] if t["name"] == task["tray"])
        xy = np.array(entry["center_xy_m"])
        extent = np.array(tray["size_xy_m"]) / 2 - tray["wall_thickness_m"] - 0.001
        surface = self.table_surface + tray["floor_thickness_m"]
        return dict(
            position=np.r_[
                xy, surface + self.objects[task["object"]]["half_extents"][2]
            ],
            low=np.r_[
                xy - extent, surface - self.protocol["support_height_tolerance_m"]
            ],
            high=np.r_[xy + extent, surface + 0.20],
            surface_z=surface,
            support="tray_" + task["tray"] + "_floor",
            orientation="any",
        )
