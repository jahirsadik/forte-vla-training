"""Build the photo-inspired bookshelf from editable SI dimensions and existing Forte assets."""

from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

try:
    from tools.scene_appearance import apply_lab_style, spine_texture, write_textures
except ModuleNotFoundError:
    from scene_appearance import apply_lab_style, spine_texture, write_textures

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs/bookshelf.json"
DESTINATION = PROJECT / "assets/bookshelf"
SOURCE_ROBOT = PROJECT / "assets/forte_arm/forte.xml"


def numbers(values):
    return " ".join(f"{float(v):.10g}" for v in values)


def element(parent, tag, **attributes):
    return ET.SubElement(
        parent,
        tag,
        {
            k: numbers(v) if isinstance(v, (list, tuple, np.ndarray)) else str(v)
            for k, v in attributes.items()
        },
    )


def write_xml(root, path):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def camera(parent, name, pos, lookat, fovy=45):
    z = np.asarray(pos, dtype=float) - lookat
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    element(
        parent, "camera", name=name, pos=pos, xyaxes=np.r_[x, np.cross(z, x)], fovy=fovy
    )


def textures(destination):
    """Deterministic original lab finishes; no external assets."""
    write_textures(destination)


def box(parent, name, pos, size, material, friction, **kwargs):
    return element(
        parent,
        "geom",
        name=name,
        type="box",
        pos=pos,
        size=np.asarray(size) / 2,
        material=material,
        group=2,
        friction=friction,
        **kwargs,
    )


def compose_robot_scene(destination, cfg, model_name="forte_bookshelf_scene"):
    """Compose the retained Forte asset with any independently authored environment."""
    surface = cfg["table"]["surface_z_m"]
    robot_root = ET.parse(SOURCE_ROBOT).getroot()
    for tag in ("compiler", "option"):
        for node in robot_root.findall(tag):
            robot_root.remove(node)
    robot_world = robot_root.find("worldbody")
    children = list(robot_world)
    robot_world.clear()
    mount = element(
        robot_world,
        "body",
        name="forte_mount",
        pos=[*cfg["robot"]["base_xy_m"], surface + cfg["robot"]["mount_clearance_m"]],
    )
    for child in children:
        mount.append(child)
    # Bare aluminium and dark motor/end-effector finishes from the lab reference.
    asset = robot_root.find("asset")
    element(
        asset,
        "material",
        name="forte_aluminium",
        rgba=[0.68, 0.70, 0.71, 1],
        specular=0.65,
        shininess=0.55,
    )
    element(
        asset,
        "material",
        name="forte_anodized",
        rgba=[0.12, 0.13, 0.14, 1],
        specular=0.30,
        shininess=0.30,
    )
    for geom in mount.iter("geom"):
        if geom.get("type") == "mesh":
            geom.attrib.pop("rgba", None)
            dark = geom.get("mesh") in {"l_end_effector_link", "r_end_effector_link"}
            geom.set("material", "forte_anodized" if dark else "forte_aluminium")
        else:
            geom.set("group", "3")
    # Cosmetic dark finish follows the model's existing motor housing envelopes.
    for parent in list(mount.iter()):
        for geom in list(parent.findall("geom")):
            name = geom.get("name", "")
            if name.endswith(("motor_housing", "lower_motor", "upper_motor")):
                finish = deepcopy(geom)
                finish.attrib.update(
                    name=name + "_finish",
                    contype="0",
                    conaffinity="0",
                    density="0",
                    group="1",
                    material="forte_anodized",
                )
                finish.attrib.pop("rgba", None)
                finish.set(
                    "size", numbers(np.fromstring(geom.get("size"), sep=" ") * 1.003)
                )
                parent.append(finish)
    wrist_camera = cfg["wrist_camera"]
    wrist = mount.find(f".//body[@name='{wrist_camera['frame']}']")
    if wrist is None:
        raise ValueError("Wrist camera frame is absent from the robot")
    camera(
        wrist,
        "wrist",
        wrist_camera["pos"],
        wrist_camera["lookat"],
        wrist_camera["fovy"],
    )
    write_xml(robot_root, destination / "robot.xml")
    scene = ET.Element("mujoco", model=model_name)
    element(scene, "include", file="environment.xml")
    element(scene, "include", file="robot.xml")
    scene_path = destination / "scene.xml"
    write_xml(scene, scene_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    qpos = model.qpos0.copy()
    arm = json.loads((PROJECT / "configs/forte.json").read_text())
    joint_names = list(arm["arm"]["joints"])
    for name, q in zip(joint_names, cfg["robot"]["home_rad"], strict=True):
        joint = model.joint(name)
        low, high = arm["arm"]["joints"][name]["position_limits_rad"]
        if not low <= q <= high:
            raise ValueError("Home exceeds joint limits")
        qpos[joint.qposadr[0]] = q
    opening = arm["gripper"]["open_width_m"] / 2
    for key in ("driver_joint", "follower_joint"):
        qpos[model.joint(arm["gripper"][key]).qposadr[0]] = opening
    element(element(scene, "keyframe"), "key", name="home", qpos=qpos)
    write_xml(scene, scene_path)
    return scene_path


def build(config_path=CONFIG, destination=DESTINATION):
    cfg = json.loads(Path(config_path).read_text())
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    t, s, p = cfg["table"], cfg["shelf"], cfg["physics"]
    if (
        min(
            t["size_xy_m"]
            + [t["surface_z_m"], s["width_m"], s["depth_m"], s["board_m"]]
        )
        <= 0
    ):
        raise ValueError("Furniture dimensions must be positive")
    if not s["board_m"] < s["bottom_surface_m"] < s["upper_surface_m"] < s["height_m"]:
        raise ValueError("Shelf board surfaces must be ordered within the side panels")
    names = [b["name"] for b in cfg["books"]]
    if len(set(names)) != len(names):
        raise ValueError("Book names must be unique")
    for b in cfg["books"]:
        if min(b[k] for k in ("depth_m", "thickness_m", "height_m", "mass_kg")) <= 0:
            raise ValueError("Book dimensions and mass must be positive")
        if b["height_m"] >= s["upper_surface_m"] - s["board_m"] - s["bottom_surface_m"]:
            raise ValueError("Book is taller than the lower shelf cavity")
        if b["depth_m"] + 0.008 > s["depth_m"] - s["board_m"]:
            raise ValueError("Book is deeper than the shelf cavity")
        if (
            "shelf_offset_y_m" in b
            and abs(b["shelf_offset_y_m"]) + b["thickness_m"] / 2
            >= s["width_m"] / 2 - s["board_m"]
        ):
            raise ValueError("Book extends through a shelf side panel")
    textures(destination / "textures")
    root = ET.Element("mujoco", model="bookshelf_environment")
    # Absolute paths are used only during generation; emitted paths are relative to this directory.
    import os

    meshdir = os.path.relpath(SOURCE_ROBOT.parent / "source_7dof", destination)
    element(
        root,
        "compiler",
        angle="radian",
        meshdir=meshdir,
        texturedir="textures",
        fusestatic="false",
    )
    element(
        root,
        "option",
        timestep=p["timestep_s"],
        integrator="implicitfast",
        cone="elliptic",
        impratio=10,
        iterations=100,
        gravity=[0, 0, -9.81],
    )
    element(root, "statistic", center=[0.3, 0, 0.9], extent=1.6)
    visual = element(root, "visual")
    element(visual, "global", offwidth=1280, offheight=960, azimuth=135, elevation=-25)
    element(
        visual, "headlight", ambient=[0.25] * 3, diffuse=[0.45] * 3, specular=[0.1] * 3
    )
    element(visual, "quality", shadowsize=4096)
    asset = element(root, "asset")
    for name in ("wood", "worktop", "paper"):
        element(asset, "texture", name=name, type="2d", file=name + ".png")
        element(
            asset,
            "material",
            name=name,
            texture=name,
            texuniform="true",
            texrepeat=[2, 2] if name == "worktop" else [1, 1],
            specular=0.12,
            shininess=0.15,
        )
    for name, rgba in [
        ("metal", [0.22, 0.24, 0.25, 1]),
        ("rubber", [0.05, 0.055, 0.06, 1]),
        ("floor_mat", [0.65, 0.67, 0.66, 1]),
        ("wall_mat", [0.71, 0.73, 0.73, 1]),
    ]:
        element(asset, "material", name=name, rgba=rgba, specular=0.15)
    custom = element(root, "custom")
    element(custom, "text", name="geometry_provenance", data=cfg["scale_status"])
    world = element(root, "worldbody")
    element(
        world,
        "light",
        name="key_light",
        pos=[-0.6, -0.8, 2.8],
        dir=[0.3, 0.2, -1],
        diffuse=[0.8, 0.78, 0.73],
        castshadow="true",
    )
    element(
        world,
        "light",
        name="fill_light",
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
        material="floor_mat",
        group=2,
    )
    box(
        world,
        "room_back_wall",
        [1.7, 0, 1.5],
        [0.08, 6, 3],
        "wall_mat",
        p["table_friction"],
    )
    box(
        world,
        "room_side_wall",
        [0, 1.7, 1.5],
        [6, 0.08, 3],
        "wall_mat",
        p["table_friction"],
    )
    table = element(world, "body", name="workbench", pos=[*t["center_xy_m"], 0])
    length, width = t["size_xy_m"]
    surface, thick, leg = t["surface_z_m"], t["top_thickness_m"], t["leg_width_m"]
    box(
        table,
        "table_top",
        [0, 0, surface - thick / 2],
        [length, width, thick],
        "worktop",
        p["table_friction"],
    )
    for i, x in enumerate((-(length / 2 - 0.06), length / 2 - 0.06)):
        for j, y in enumerate((-(width / 2 - 0.06), width / 2 - 0.06)):
            box(
                table,
                f"table_leg_{i}_{j}",
                [x, y, (surface - thick) / 2],
                [leg, leg, surface - thick],
                "metal",
                p["table_friction"],
            )
            box(
                table,
                f"table_foot_{i}_{j}",
                [x, y, 0.012],
                [leg + 0.012, leg + 0.012, 0.024],
                "rubber",
                p["table_friction"],
            )
    for side in (-1, 1):
        box(
            table,
            f"table_apron_{side}",
            [0, side * (width / 2 - 0.06), surface - 0.12],
            [length - 0.1, 0.025, 0.14],
            "metal",
            p["table_friction"],
        )
    box(
        table,
        "table_rear_brace",
        [length / 2 - 0.06, 0, 0.22],
        [0.025, width - 0.12, 0.055],
        "metal",
        p["table_friction"],
    )
    shelf = element(world, "body", name="bookshelf", pos=[*s["center_xy_m"], surface])
    benchmark = json.loads((PROJECT / "configs/benchmarks.json").read_text())[
        "bookshelf"
    ]
    element(asset, "material", name="placement_mat", rgba=[0.12, 0.34, 0.36, 1])
    box(
        world,
        "placement_mat",
        [*benchmark["placement_center_xy_m"], surface + 0.0001],
        [*benchmark["placement_size_xy_m"], 0.0002],
        "placement_mat",
        p["table_friction"],
        contype=0,
        conaffinity=0,
        density=0,
    )
    d, w, h, b = (s[k] for k in ("depth_m", "width_m", "height_m", "board_m"))
    for suffix, sign in [("left", 1), ("right", -1)]:
        box(
            shelf,
            "shelf_" + suffix,
            [0, sign * (w - b) / 2, h / 2],
            [d, b, h],
            "wood",
            p["shelf_friction"],
        )
    box(
        shelf,
        "shelf_back",
        [(d - b) / 2, 0, h / 2],
        [b, w - 2 * b, h],
        "wood",
        p["shelf_friction"],
    )
    for name, z in [("bottom", s["bottom_surface_m"]), ("upper", s["upper_surface_m"])]:
        board = box(
            shelf,
            "shelf_" + name,
            [-b / 2, 0, z - b / 2],
            [d - b, w - 2 * b, b],
            "wood",
            p["shelf_friction"],
        )
        # The original box remains the collision envelope. Its visible front
        # lip is rounded like the reference shelf, inside that same envelope.
        board.set("group", "3")
        visual_args = dict(contype=0, conaffinity=0, density=0)
        visible = box(
            shelf,
            "shelf_" + name + "_finish",
            [-b / 4, 0, z - b / 2],
            [d - 1.5 * b, w - 2 * b, b],
            "wood",
            p["shelf_friction"],
            **visual_args,
        )
        visible.set("group", "1")
        element(
            shelf,
            "geom",
            name="shelf_" + name + "_rounded_lip",
            type="capsule",
            fromto=[
                -d / 2 + b / 2,
                -(w / 2 - 1.5 * b),
                z - b / 2,
                -d / 2 + b / 2,
                w / 2 - 1.5 * b,
                z - b / 2,
            ],
            size=[b / 2],
            material="wood",
            group=1,
            **visual_args,
        )
    element(
        shelf,
        "site",
        name="shelf_opening",
        pos=[-d / 2, 0, (s["bottom_surface_m"] + s["upper_surface_m"] - b) / 2],
        size=0.005,
        group=5,
    )
    shelf_front = s["center_xy_m"][0] - d / 2
    for index, book in enumerate(cfg["books"]):
        name = "book_" + book["name"]
        bd, bt, bh = (book[k] for k in ("depth_m", "thickness_m", "height_m"))
        flat = "table_xy_m" in book
        pos = (
            [*book["table_xy_m"], surface + bt / 2 + 0.0005]
            if flat
            else [
                shelf_front + bd / 2 + 0.008,
                s["center_xy_m"][1] + book["shelf_offset_y_m"],
                surface + s["bottom_surface_m"] + bh / 2 + 0.0005,
            ]
        )
        quat = [np.sqrt(0.5), np.sqrt(0.5), 0, 0] if flat else [1, 0, 0, 0]
        body = element(world, "body", name=name, pos=pos, quat=quat)
        element(body, "freejoint", name=name + "_joint")
        mass = book["mass_kg"]
        inertia = (
            mass
            / 12
            * np.array([bt * bt + bh * bh, bd * bd + bh * bh, bd * bd + bt * bt])
        )
        element(body, "inertial", pos=[0, 0, 0], mass=mass, diaginertia=inertia)
        element(
            body,
            "geom",
            name=name + "_collision",
            type="box",
            size=[bd / 2, bt / 2, bh / 2],
            group=3,
            rgba=[*book["color"], 1],
            friction=p["book_friction"],
            condim=4,
            solref=[0.006, 1],
            solimp=[0.95, 0.99, 0.001],
        )
        # One physical rigid envelope; decorative covers/pages neither collide nor add mass.
        material = name + "_cover"
        element(
            asset,
            "material",
            name=material,
            rgba=[*book["color"], 1],
            specular=0.22,
            shininess=0.2,
        )
        visual_args = dict(contype=0, conaffinity=0, density=0)
        cover = 0.0016
        for sign in (-1, 1):
            g = box(
                body,
                name + f"_cover_{sign}",
                [0, sign * (bt - cover) / 2, 0],
                [bd, cover, bh],
                material,
                p["book_friction"],
                **visual_args,
            )
            g.set("group", "1")
        g = box(
            body,
            name + "_pages",
            [0.001, 0, 0],
            [bd - 0.004, bt - 2 * cover, bh - 0.004],
            "paper",
            p["book_friction"],
            **visual_args,
        )
        g.set("group", "1")
        g = box(
            body,
            name + "_spine",
            [-(bd - cover) / 2, 0, 0],
            [cover, bt, bh],
            material,
            p["book_friction"],
            **visual_args,
        )
        g.set("group", "1")
        for z in (-bh * 0.34, bh * 0.34):
            element(
                body,
                "geom",
                name=name + ("_band_low" if z < 0 else "_band_high"),
                type="box",
                pos=[-bd / 2 - 0.00005, 0, z],
                size=[0.00005, bt * 0.38, 0.0013],
                rgba=[0.8, 0.72, 0.44, 1],
                group=1,
                **visual_args,
            )
        printed = name + "_printed_spine"
        element(
            asset,
            "texture",
            name=printed,
            type="2d",
            file=spine_texture(destination / "textures", book, index),
        )
        element(
            asset,
            "material",
            name=printed,
            texture=printed,
            texuniform="false",
            texrepeat=[1, 1],
            specular=0.16,
            shininess=0.12,
        )
        element(
            body,
            "geom",
            name=printed,
            type="box",
            pos=[-bd / 2 - 0.00012, 0, 0],
            xyaxes=[0, -1, 0, 0, 0, 1],
            size=[bt * 0.48, bh * 0.49, 0.00003],
            material=printed,
            group=1,
            **visual_args,
        )
        element(
            body, "site", name=name + "_grasp", pos=[-bd / 2, 0, 0], size=0.003, group=5
        )
    target_book = next(
        v for v in cfg["books"] if v["name"] == cfg["insertion_target"]["book"]
    )
    element(
        world,
        "site",
        name="insertion_target",
        pos=[
            shelf_front + target_book["depth_m"] / 2 + 0.008,
            s["center_xy_m"][1] + cfg["insertion_target"]["shelf_offset_y_m"],
            surface + s["bottom_surface_m"] + target_book["height_m"] / 2,
        ],
        size=[
            target_book["depth_m"] / 2,
            target_book["thickness_m"] / 2,
            target_book["height_m"] / 2,
        ],
        type="box",
        rgba=[0.1, 0.9, 0.6, 0.2],
        group=5,
    )
    element(
        world,
        "site",
        name="robot_mount",
        pos=[*cfg["robot"]["base_xy_m"], surface + cfg["robot"]["mount_clearance_m"]],
        size=0.005,
        group=5,
    )
    for name, values in cfg["cameras"].items():
        camera(world, name, values["pos"], values["lookat"], values["fovy"])
    apply_lab_style(root, table, cfg, element)
    write_xml(root, destination / "environment.xml")

    scene_path = compose_robot_scene(destination, cfg)
    manifest = {
        "reference": "forte-bookshelf.jpeg supplied by user; visual layout only",
        "scale_status": cfg["scale_status"],
        "frames": "metres, radians, z up; shelf opens toward -X; book local axes: depth X, thickness Y, height Z",
        "robot_source": os.path.relpath(SOURCE_ROBOT, destination),
        "robot_source_sha256": hashlib.sha256(SOURCE_ROBOT.read_bytes()).hexdigest(),
        "configuration": cfg,
        "limitations": [
            "Rigid closed books; no page bending or cover articulation",
            "Seven-arm-joint Forte from pinned Kim source; hardware limits provisional",
            "Estimated geometry, masses, friction and camera poses; uncalibrated",
            "Expert coverage remains incomplete; no trained vision policy or hardware validation",
        ],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return scene_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    args = parser.parse_args()
    print(build(args.config, args.output_dir))
