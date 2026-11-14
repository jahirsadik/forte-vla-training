"""Dense books with genuine rigid-body contacts and a reserved narrow insertion slot."""

from pathlib import Path
import mujoco
import numpy as np

from environments.bookshelf_task import BookshelfEnv


def packed_positions(config, order, layout):
    """Pack from the slot outward using each book's actual thickness, without overlap."""
    packing, shelf = config["hard_layout"], config["shelf"]
    if layout not in packing["gap_m"]:
        raise ValueError(f"Unknown hard layout {layout!r}; choose from {tuple(packing['gap_m'])}")
    specs = {b["name"]: b for b in config["books"]}
    insert = specs[config["insertion_target"]["book"]]
    slot_width = insert["thickness_m"] + 2 * packing["slot_clearance_each_side_m"]
    center = shelf["center_xy_m"][1] + config["insertion_target"]["shelf_offset_y_m"]
    front = shelf["center_xy_m"][0] - shelf["depth_m"] / 2
    positions = {}
    half = (len(order) + 1) // 2
    for sign, bank in ((1, order[:half]), (-1, order[half:])):
        edge = center + sign * slot_width / 2
        for i, name in enumerate(bank):
            spec = specs[name]
            y = edge + sign * spec["thickness_m"] / 2
            if abs(y - shelf["center_xy_m"][1]) + spec["thickness_m"] / 2 > shelf["width_m"] / 2 - shelf["board_m"]:
                raise ValueError("Packed row exceeds the shelf's interior width")
            protrusion = packing["stagger_protrusion_m"] if layout == "staggered" and i % 2 == 0 else 0.
            positions[name] = [front + spec["depth_m"] / 2 + .008 - protrusion, y,
                               config["table"]["surface_z_m"] + shelf["bottom_surface_m"] + spec["height_m"] / 2 + .0005]
            gap = packing["gap_m"][layout][i % len(packing["gap_m"][layout])]
            edge += sign * (spec["thickness_m"] + gap)
    return positions


class BookshelfHardEnv(BookshelfEnv):
    scene_kind = "bookshelf_hard"
    protocol_file = Path(__file__).resolve().parents[1] / "configs/benchmarks_hard.json"

    def _sample_layout(self, options):
        if "clutter_level" in options:
            raise ValueError("bookshelf-hard uses named packing layouts, not clutter_level")
        cfg, packing = self.scene_config, self.scene_config["hard_layout"]
        layout = options.get("layout", packing["default"])
        randomize = options.get("randomize", True)
        names = [b["name"] for b in cfg["books"] if "table_xy_m" not in b]
        order = list(self.np_random.permutation(names)) if randomize else names
        positions = packed_positions(cfg, order, layout)
        for book in cfg["books"]:
            if "table_xy_m" in book:
                xy = np.array(self.protocol["bookshelf"]["orange_start_xy_m"], dtype=float)
                if randomize:
                    jitter = self.protocol["bookshelf"]["orange_jitter_xy_m"]
                    xy += self.np_random.uniform(-np.array(jitter), jitter)
                position = [*xy, self.table_surface + book["height_m"] / 2 + .0005]
            else:
                position = positions[book["name"]]
                if randomize:
                    position[0] += self.np_random.uniform(-packing["front_jitter_m"], packing["front_jitter_m"])
            self._set_object_pose("book_" + book["name"], position)

    def _make_goal(self, task):
        goal = super()._make_goal(task)
        if task["operation"] == "insert":
            packing = self.scene_config["hard_layout"]
            orange = next(b for b in self.scene_config["books"] if b["name"] == "orange")
            slot_half = orange["thickness_m"] / 2 + packing["slot_clearance_each_side_m"]
            # Containment uses the actual reserved slot, not the wider easy-task box.
            goal["low"][1] = goal["position"][1] - slot_half
            goal["high"][1] = goal["position"][1] + slot_half
        return goal

    def _measure(self):
        metrics = super()._measure()
        packing = self.scene_config["hard_layout"]
        restored = (metrics["distractor_translation_m"] <= packing["final_neighbor_translation_m"]
                    and metrics["distractor_rotation_rad"] <= np.deg2rad(packing["final_neighbor_rotation_deg"]))
        metrics["neighbors_within_final_tolerance"] = bool(restored)
        metrics["placement_valid"] &= bool(restored)
        ordered = sorted((n for n in self.objects if n != "book_orange"),
                         key=lambda n: self.data.xpos[self.objects[n]["body"], 1])
        gaps, pairs = [], []
        for left, right in zip(ordered, ordered[1:]):
            a, b = self.objects[left]["geoms"][0], self.objects[right]["geoms"][0]
            distance = float(mujoco.mj_geomDistance(self.model, self.data, a, b, 1., None))
            gaps.append(distance)
            if distance <= .0001:
                pairs.append([left, right])
        metrics["minimum_book_gap_m"] = min(gaps, default=0.)
        metrics["touching_book_pairs"] = pairs
        metrics["target_withdrawal_m"] = float(self._episode["initial_positions"][self.target][0]
                                              - self.data.xpos[self.objects[self.target]["body"], 0])
        return metrics
