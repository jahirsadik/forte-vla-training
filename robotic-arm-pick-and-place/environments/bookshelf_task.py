"""Book extraction to a mat and upright insertion into the empty shelf slot."""

import numpy as np

from environments.benchmark import ManipulationBenchmark
from environments.variants import scene_variant


class BookshelfEnv(ManipulationBenchmark):
    scene_kind = "bookshelf"

    def __init__(self, *, asset_set="nominal", **kwargs):
        scene, self.scene_config = scene_variant(self.scene_kind, asset_set)
        self.asset_set = asset_set
        self.table_surface = self.scene_config["table"]["surface_z_m"]
        objects, tasks = {}, {}
        for book in self.scene_config["books"]:
            name = "book_" + book["name"]
            objects[name] = dict(
                half_extents=[
                    book[k] / 2 for k in ("depth_m", "thickness_m", "height_m")
                ]
            )
            if "table_xy_m" not in book and not book.get("distractor_only", False):
                tasks["extract_" + book["name"]] = dict(
                    object=name,
                    operation="extract",
                    instruction=f"Take the {book['name']} book from the shelf and place it flat on the mat.",
                )
                tasks["relocate_" + book["name"]] = dict(
                    object=name,
                    operation="insert",
                    instruction=f"Move the {book['name']} book to the empty shelf slot and leave it upright.",
                )
        tasks["insert_orange"] = dict(
            object="book_orange",
            operation="insert",
            instruction="Pick up the orange book and place it upright in the empty shelf slot.",
        )
        super().__init__(scene, objects, tasks, **kwargs)

    def _sample_layout(self, options):
        if "clutter_level" in options:
            raise ValueError(
                "clutter_level is a sorting option; bookshelf uses spaced slots"
            )
        randomize = options.get("randomize", True)
        p, cfg = self.protocol["bookshelf"], self.scene_config
        shelf = cfg["shelf"]
        front = shelf["center_xy_m"][0] - shelf["depth_m"] / 2
        row_shift = (
            self.np_random.uniform(-p["row_jitter_m"], p["row_jitter_m"])
            if randomize
            else 0
        )
        offsets = np.array(p["shelf_offsets_y_m"])
        layout = options.get("layout", "nominal")
        if layout not in ("nominal", "shuffled", "staggered"):
            raise ValueError("Unknown bookshelf layout")
        if layout != "nominal":
            offsets = offsets[self.np_random.permutation(len(offsets))]
        slots = iter(offsets)
        row_index = 0
        for book in cfg["books"]:
            name = "book_" + book["name"]
            if "table_xy_m" in book:
                xy = np.array(p["orange_start_xy_m"], dtype=float)
                if randomize:
                    xy += self.np_random.uniform(
                        -np.array(p["orange_jitter_xy_m"]), p["orange_jitter_xy_m"]
                    )
                pos = [*xy, self.table_surface + book["height_m"] / 2 + 0.0005]
            else:
                jitter = (
                    self.np_random.uniform(-p["depth_jitter_m"], p["depth_jitter_m"])
                    if randomize
                    else 0
                )
                pos = [
                    front + book["depth_m"] / 2 + 0.008 + jitter + (.018 * (row_index % 2) if layout == "staggered" else 0),
                    shelf["center_xy_m"][1] + next(slots) + row_shift,
                    self.table_surface
                    + shelf["bottom_surface_m"]
                    + book["height_m"] / 2
                    + 0.0005,
                ]
                row_index += 1
            self._set_object_pose(name, pos)

    def _make_goal(self, task):
        p, shelf = self.protocol["bookshelf"], self.scene_config["shelf"]
        half = np.array(self.objects[task["object"]]["half_extents"])
        tol = self.protocol["support_height_tolerance_m"]
        if task["operation"] == "extract":
            xy = np.array(p["placement_center_xy_m"])
            extent = np.array(p["placement_size_xy_m"]) / 2
            return dict(
                position=np.r_[xy, self.table_surface + half[1]],
                low=np.r_[xy - extent, self.table_surface - tol],
                high=np.r_[xy + extent, self.table_surface + 0.06],
                surface_z=self.table_surface,
                support="table_top",
                orientation="flat",
            )
        center = self.data.site("insertion_target").xpos.copy()
        sx = shelf["center_xy_m"][0]
        surface = self.table_surface + shelf["bottom_surface_m"]
        center[2] = surface + half[2]
        return dict(
            position=center,
            low=np.array(
                [sx - shelf["depth_m"] / 2 + 0.002, center[1] - 0.028, surface - tol]
            ),
            high=np.array(
                [
                    sx + shelf["depth_m"] / 2 - shelf["board_m"] - 0.002,
                    center[1] + 0.028,
                    self.table_surface
                    + shelf["upper_surface_m"]
                    - shelf["board_m"]
                    - 0.002,
                ]
            ),
            surface_z=surface,
            support="shelf_bottom",
            orientation="upright",
        )
