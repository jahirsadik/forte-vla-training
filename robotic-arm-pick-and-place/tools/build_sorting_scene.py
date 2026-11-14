"""Build a portable sorting workbench with original objects and open tray colliders."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

try:
    from tools.build_bookshelf_scene import (
        PROJECT,
        SOURCE_ROBOT,
        apply_lab_style,
        box,
        camera,
        compose_robot_scene,
        element,
        textures,
        write_xml,
    )
except ModuleNotFoundError:
    from build_bookshelf_scene import (
        PROJECT,
        SOURCE_ROBOT,
        apply_lab_style,
        box,
        camera,
        compose_robot_scene,
        element,
        textures,
        write_xml,
    )

CONFIG = PROJECT / "configs/sorting.json"
DESTINATION = PROJECT / "assets/sorting"


def build(config_path=CONFIG, destination=DESTINATION):
    cfg = json.loads(Path(config_path).read_text())
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    textures(destination / "textures")
    t, tray = cfg["table"], cfg["tray"]
    surface = t["surface_z_m"]
    root = ET.Element("mujoco", model="sorting_environment")
    element(
        root,
        "compiler",
        angle="radian",
        meshdir=os.path.relpath(SOURCE_ROBOT.parent / "source_7dof", destination),
        texturedir="textures",
        fusestatic="false",
    )
    element(
        root,
        "option",
        timestep=0.001,
        integrator="implicitfast",
        cone="elliptic",
        impratio=10,
        iterations=100,
        gravity=[0, 0, -9.81],
    )
    element(root, "statistic", center=[0.3, 0, 0.9], extent=1.6)
    visual = element(root, "visual")
    element(visual, "global", offwidth=1280, offheight=960)
    element(
        visual, "headlight", ambient=[0.25] * 3, diffuse=[0.45] * 3, specular=[0.1] * 3
    )
    element(visual, "quality", shadowsize=4096)
    asset = element(root, "asset")
    element(asset, "texture", name="worktop", type="2d", file="worktop.png")
    element(
        asset,
        "material",
        name="worktop",
        texture="worktop",
        texuniform="true",
        texrepeat=[2, 2],
    )
    for name, rgba in [
        ("metal", [0.2, 0.22, 0.24, 1]),
        ("floor", [0.65, 0.67, 0.66, 1]),
        ("wall", [0.71, 0.73, 0.73, 1]),
    ]:
        element(asset, "material", name=name, rgba=rgba)
    world = element(root, "worldbody")
    element(
        world,
        "light",
        pos=[-0.6, -0.8, 2.8],
        dir=[0.3, 0.2, -1],
        diffuse=[0.8, 0.78, 0.73],
    )
    element(
        world,
        "light",
        pos=[0.8, 0.8, 2.2],
        dir=[0, 0, -1],
        diffuse=[0.35, 0.4, 0.45],
        castshadow="false",
    )
    element(
        world,
        "geom",
        name="floor",
        type="plane",
        size=[3, 3, 0.05],
        material="floor",
        group=2,
    )
    friction = [0.7, 0.005, 0.0001]
    box(world, "room_wall", [1.7, 0, 1.5], [0.08, 6, 3], "wall", friction)
    box(world, "room_side_wall", [0, 1.7, 1.5], [6, 0.08, 3], "wall", friction)
    table = element(world, "body", name="workbench", pos=[*t["center_xy_m"], 0])
    box(
        table,
        "table_top",
        [0, 0, surface - t["top_thickness_m"] / 2],
        [*t["size_xy_m"], t["top_thickness_m"]],
        "worktop",
        friction,
    )
    length, width = t["size_xy_m"]
    for i, x in enumerate((-0.5 * length + 0.06, 0.5 * length - 0.06)):
        for j, y in enumerate((-0.5 * width + 0.06, 0.5 * width - 0.06)):
            box(
                table,
                f"leg_{i}_{j}",
                [x, y, (surface - t["top_thickness_m"]) / 2],
                [t["leg_width_m"]] * 2 + [surface - t["top_thickness_m"]],
                "metal",
                friction,
            )
    dx, dy = tray["size_xy_m"]
    wall, floor, height = (
        tray[k] for k in ("wall_thickness_m", "floor_thickness_m", "wall_height_m")
    )
    for entry in cfg["trays"]:
        name = "tray_" + entry["name"]
        element(asset, "material", name=name, rgba=[*entry["color"], 1], specular=0.18)
        body = element(world, "body", name=name, pos=[*entry["center_xy_m"], surface])
        box(body, name + "_floor", [0, 0, floor / 2], [dx, dy, floor], name, friction)
        for sign in (-1, 1):
            box(
                body,
                name + f"_wall_x_{sign}",
                [sign * (dx - wall) / 2, 0, floor + height / 2],
                [wall, dy, height],
                name,
                friction,
            )
            box(
                body,
                name + f"_wall_y_{sign}",
                [0, sign * (dy - wall) / 2, floor + height / 2],
                [dx - 2 * wall, wall, height],
                name,
                friction,
            )
        element(
            body, "site", name=name + "_goal", pos=[0, 0, floor], size=0.005, group=5
        )
    slots = [(x, y) for y in cfg["layout"]["y_m"] for x in cfg["layout"]["x_m"]]
    for obj, xy in zip(cfg["objects"], slots, strict=True):
        name = "object_" + obj["name"]
        hx, hy, hz = np.array(obj["size_m"]) / 2
        body = element(
            world,
            "body",
            name=name,
            pos=[*xy, surface + (hx if obj["shape"] == "capsule" else hz) + 0.0005],
            quat=[np.sqrt(0.5), 0, np.sqrt(0.5), 0]
            if obj["shape"] == "capsule"
            else [1, 0, 0, 0],
        )
        element(body, "freejoint", name=name + "_joint")
        shape = obj["shape"]
        if shape == "wedge":
            vertices = np.array(
                [
                    [-hx, -hy, -hz],
                    [hx, -hy, -hz],
                    [-hx, -hy, hz],
                    [-hx, hy, -hz],
                    [hx, hy, -hz],
                    [-hx, hy, hz],
                ]
            )
            element(asset, "mesh", name="wedge", vertex=vertices.ravel())
            geometry = dict(type="mesh", mesh="wedge")
        elif shape == "box":
            geometry = dict(type="box", size=[hx, hy, hz])
        else:
            geometry = dict(
                type=shape, size=[hx, hz - hx if shape == "capsule" else hz]
            )
        element(
            body,
            "geom",
            name=name + "_collision",
            **geometry,
            mass=obj["mass_kg"],
            rgba=[*obj["color"], 1],
            group=3,
            friction=[0.65, 0.005, 0.0001],
            condim=4,
            solref=[0.006, 1],
            solimp=[0.95, 0.99, 0.001],
        )
        element(
            body,
            "geom",
            name=name + "_visual",
            **geometry,
            density=0,
            rgba=[*obj["color"], 1],
            group=1,
            contype=0,
            conaffinity=0,
        )
    for name, entry in cfg["cameras"].items():
        camera(world, name, **entry)
    apply_lab_style(root, table, cfg, element)
    write_xml(root, destination / "environment.xml")
    scene = compose_robot_scene(destination, cfg, "forte_sorting_scene")
    manifest = dict(
        configuration=cfg,
        provenance="Original parametric sorting assets; retained repository Forte robot",
        robot_source_sha256=hashlib.sha256(SOURCE_ROBOT.read_bytes()).hexdigest(),
        limitations=[
            "Estimated geometry and contact properties",
            "Seven-arm-joint Forte from pinned Kim source; hardware limits provisional",
        ],
    )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    args = parser.parse_args()
    print(build(args.config, args.output_dir))
