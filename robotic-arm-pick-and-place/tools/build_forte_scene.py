"""Build, check, and preview the pinned seven-joint Forte model and cube scene.

All paths are resolved relative to this repository. The preview holds a fixed
configuration with physics paused; it is a model inspection, not a controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT / "assets" / "forte_arm"
SOURCE = ASSETS / "source_7dof"
SOURCE_MODEL = SOURCE / "robot.xml"
MANIFEST = SOURCE / "manifest.json"
ROBOT = ASSETS / "forte.xml"
SCENE = ASSETS / "scene.xml"
PANDA_SCENE = PROJECT / "assets" / "franka_emika_panda" / "scene.xml"
ARM_JOINTS = (
    "shoulder_yaw", "shoulder_pitch", "shoulder_roll", "elbow_pitch",
    "lower_arm_roll", "wrist_pitch", "wrist_roll",
)
LEFT = "gripper_left_joint"
RIGHT = "gripper_right_joint"
CONFIG = json.loads((PROJECT / "configs" / "forte.json").read_text(encoding="utf-8"))
ARM_CONFIG = CONFIG["arm"]
HOME = tuple(ARM_CONFIG["joints"][name]["home_rad"] for name in ARM_JOINTS)
GRIPPER_CONFIG = CONFIG["gripper"]
WORKSPACE = CONFIG["workspace"]
COLORS = ("red", "green", "blue")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_source() -> ET.Element:
    """Verify the pinned upstream snapshot, independent of Kim's working checkout."""
    manifest = json.loads(MANIFEST.read_text())
    for entry in manifest["files"]:
        path = SOURCE / entry["file"]
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Source integrity check failed: {path}")
    return ET.parse(SOURCE_MODEL).getroot()


def write_xml(root: ET.Element, destination: Path) -> None:
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def build() -> None:
    robot_root = check_source()
    robot_root.set("model", "forte_kim_7dof")
    for keyframe in robot_root.findall("keyframe"):
        robot_root.remove(keyframe)  # Each composed scene owns its full qpos key.
    robot_root.find("compiler").set("meshdir", "source_7dof")
    # One bounded qfrc_applied backend owns all control; upstream actuator
    # definitions remain byte-identical in source_7dof/robot.xml for provenance.
    robot_root.remove(robot_root.find("actuator"))
    option = robot_root.find("option")
    if option is None:
        option = ET.SubElement(robot_root, "option")
    option.attrib.update(timestep="0.001", gravity="0 0 -9.81", integrator="implicitfast",
                         cone="elliptic", impratio="10")
    for name, entry in ARM_CONFIG["joints"].items():
        joint = robot_root.find(f".//joint[@name='{name}']")
        limit = entry["effort_limit_nm"]
        joint.set("actuatorfrcrange", f"{-limit} {limit}")
    for name in (LEFT, RIGHT):
        robot_root.find(f".//joint[@name='{name}']").set("actuatorfrcrange", "-20 20")
    tool = robot_root.find(".//site[@name='grasp']")
    upstream_quat = np.fromstring(tool.get("quat"), sep=" ")
    upstream_quat /= np.linalg.norm(upstream_quat)
    # Canonical +X approach, +Y closing, +Z up. Source grasp uses +Z approach.
    rotation = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    correction, quaternion = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(correction, rotation.ravel())
    mujoco.mju_mulQuat(quaternion, upstream_quat, correction)
    canonical_rotation = np.zeros(9)
    mujoco.mju_quat2Mat(canonical_rotation, quaternion)
    position = (np.fromstring(tool.get("pos"), sep=" ") + canonical_rotation.reshape(3, 3)
                @ np.asarray(GRIPPER_CONFIG["source_grasp_offset_canonical_m"]))
    body = robot_root.find(f".//body[@name='{GRIPPER_CONFIG['tool_body']}']")
    frame = ET.SubElement(body, "body", name="forte_tool_frame", pos=" ".join(f"{x:.17g}" for x in position),
                          quat=" ".join(f"{x:.17g}" for x in quaternion))
    ET.SubElement(frame, "site", name="grasp_site", size="0.004", rgba="0 1 0 1")
    write_xml(robot_root, ROBOT)

    # Reuse the Panda environment, leaving its original files intact.
    scene_root = ET.parse(PANDA_SCENE).getroot()
    scene_root.set("model", "forte_pick_and_place_scene")
    scene_root.find("include").set("file", "forte.xml")
    table = scene_root.find("worldbody/body[@name='table']")
    table.set("pos", " ".join(map(str, WORKSPACE["table_center_m"])))
    table.find("geom").set("size", " ".join(map(str, WORKSPACE["table_half_size_m"])))
    table_z = WORKSPACE["table_center_m"][2] + WORKSPACE["table_half_size_m"][2]
    for color in COLORS:
        scene_root.find(f"worldbody/body[@name='cube_{color}']").set(
            "pos", " ".join(map(str, WORKSPACE["cube_spawn_m"][color])))
        scene_root.find(f"worldbody/site[@name='target_{color}']").set(
            "pos", " ".join(map(str, [*WORKSPACE["target_xy_m"][color], table_z + 0.001])))
    # Keep the environment visible when hiding the robot collision primitives
    # (group 0) or its CAD meshes (group 1) in the inspection viewer.
    for geom in scene_root.findall("worldbody//geom"):
        geom.set("group", "2")
    write_xml(scene_root, SCENE)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    for name, value in zip(ARM_JOINTS, HOME):
        data.qpos[model.joint(name).qposadr[0]] = value
    for name in (LEFT, RIGHT):
        data.qpos[model.joint(name).qposadr[0]] = GRIPPER_CONFIG["open_width_m"] / 2
    # Preview the cubes resting on the table; this does not change qpos0.
    for color in COLORS:
        joint_id = int(model.body(f"cube_{color}").jntadr[0])
        data.qpos[model.jnt_qposadr[joint_id] + 2] = table_z + model.geom(f"cube_{color}").size[2]
    keyframes = ET.SubElement(scene_root, "keyframe")
    ET.SubElement(keyframes, "key", {
        "name": "home", "qpos": " ".join(f"{value:.17g}" for value in data.qpos),
    })
    write_xml(scene_root, SCENE)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check() -> tuple[mujoco.MjModel, mujoco.MjData]:
    check_source()
    original = mujoco.MjModel.from_xml_path(str(SOURCE_MODEL))
    robot = mujoco.MjModel.from_xml_path(str(ROBOT))
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    require((robot.nq, robot.nv, robot.njnt, robot.nu, robot.neq) == (9, 9, 9, 0, 1),
            "Expected seven arm hinges, two coupled sliders, one force backend")
    require((model.nq, model.nv, model.njnt, model.nu, model.neq) == (30, 27, 12, 0, 1),
            "Unexpected full-scene coordinates or constraints")
    # A fixed canonical tool frame adds a massless body. Compare named source
    # bodies rather than array positions, which now differ after the insertion.
    for body_id in range(1, original.nbody):
        name = original.body(body_id).name
        new_id = robot.body(name).id
        for attr in ("body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat"):
            np.testing.assert_allclose(getattr(robot, attr)[new_id], getattr(original, attr)[body_id], atol=1e-10)
    for attr in ("jnt_axis", "jnt_pos", "jnt_range", "jnt_limited", "jnt_type"):
        np.testing.assert_allclose(getattr(robot, attr), getattr(original, attr), atol=1e-10)
    for attr in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity"):
        np.testing.assert_allclose(getattr(robot, attr), getattr(original, attr), atol=1e-10)
    require(robot.ngeom == original.ngeom and robot.nmesh == original.nmesh, "Source geometry count changed")
    np.testing.assert_allclose(robot.eq_data, original.eq_data, atol=1e-10)
    require(model.opt.timestep == 0.001, "Expected 1 ms physics")
    require(model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST, "Expected implicitfast")
    np.testing.assert_allclose(model.opt.gravity, [0, 0, -9.81])
    model.site("grasp_site")
    model.camera("demo_cam")
    for path in (ROBOT, SCENE):
        for node in ET.parse(path).getroot().iter():
            for attr in ("file", "meshdir", "texturedir"):
                value = node.get(attr)
                require(value is None or not Path(value).is_absolute(), f"Nonportable asset path: {value}")

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    require(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(), "Nonfinite home state")
    penetrations = [
        f"{model.geom(int(c.geom1)).name} / {model.geom(int(c.geom2)).name}: {c.dist:.6g} m"
        for c in data.contact if c.dist < -1e-5
    ]
    require(not penetrations, "Home pose has penetrating contacts: " + "; ".join(penetrations))
    require(not any(w.number for w in data.warning), "MuJoCo warnings at home")
    print("PASS: SHA-256 matches for all pinned seven-joint source files")
    print("PASS: joint types/limits, fixed base, source inertials and collision geometry")
    print("PASS: 7 arm DoF + 1 coupled gripper DoF; source left = right constraint")
    print(f"PASS: scene nq={model.nq}, nv={model.nv}; 3 cubes, 3 targets, table, floor, camera")
    print("PASS: relative asset paths; 1 ms timestep; gravity -9.81; implicitfast")
    print("PASS: finite preview home pose, no penetrating contacts or MuJoCo warnings")
    print("NOTE: preview pauses physics; use tools/check.py hold for controlled gravity holding.")
    return model, data


def snapshot(model: mujoco.MjModel, data: mujoco.MjData, destination: Path) -> None:
    import imageio.v2 as imageio

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    option = mujoco.MjvOption()
    option.geomgroup[0] = 0  # URDF collision primitives
    option.geomgroup[1] = 1  # CAD visuals
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera="demo_cam", scene_option=option)
        imageio.imwrite(destination, renderer.render())
    print(f"Saved paused model preview: {destination}")


def preview(model: mujoco.MjModel, data: mujoco.MjData, collisions: bool) -> None:
    import mujoco.viewer

    print("PAUSED MODEL PREVIEW: physics does not advance; the robot is not being held by PD.")
    print("Inspect the arm, fingers, table and cubes. Close the viewer to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = (0.4, 0, 0.3)
            viewer.cam.distance = 1.6
            viewer.cam.azimuth = 140
            viewer.cam.elevation = -25
            viewer.opt.geomgroup[3] = int(collisions)
            viewer.opt.geomgroup[1] = int(not collisions)
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-from", type=Path, metavar="FOLDER",
                        help="Retired: the pinned seven-joint source is already vendored")
    parser.add_argument("--check", action="store_true", help="Validate existing generated files without rebuilding")
    parser.add_argument("--preview", action="store_true", help="Inspect the existing scene with physics paused")
    parser.add_argument("--collision-geoms", action="store_true", help="Show primitives instead of CAD in preview")
    parser.add_argument("--snapshot", type=Path, metavar="PNG", help="Save an offscreen paused-scene image")
    args = parser.parse_args()
    if args.collision_geoms and not args.preview:
        parser.error("--collision-geoms requires --preview")
    if args.import_from and (args.check or args.preview):
        parser.error("--import-from cannot be combined with --check or --preview")
    try:
        if args.import_from:
            raise ValueError("Legacy URDF import is retired; use the pinned source_7dof snapshot")
        if not args.check and not args.preview:
            build()
            print(f"Built {ROBOT}\nBuilt {SCENE}")
        model, data = check()
        if args.snapshot:
            snapshot(model, data, args.snapshot)
        if args.preview:
            preview(model, data, args.collision_geoms)
    except (OSError, ValueError, KeyError, RuntimeError, AssertionError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
