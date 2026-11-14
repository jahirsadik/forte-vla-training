"""Load the bookshelf scene with the existing bounded Forte hold controller."""

from pathlib import Path
import mujoco
import numpy as np
from robots.forte import ForteRobot

SCENE = Path(__file__).resolve().parents[1] / "assets/bookshelf/scene.xml"


def load_bookshelf(scene=SCENE):
    model = mujoco.MjModel.from_xml_path(str(scene))
    robot = ForteRobot(model, mujoco.MjData(model))
    # The scene has its own parking pose; ForteRobot.reset uses robot.home.
    robot.home = model.key("home").qpos[robot.qpos_indices].copy()
    robot.reset()
    return robot


def render_options(collisions=False):
    options = mujoco.MjvOption()
    options.geomgroup[:] = [0, int(not collisions), 1, int(collisions), 0, 0]
    options.sitegroup[:] = 0
    return options


def robot_geom_ids(model):
    root = model.body("forte_mount").id
    descendants = {root}
    for body in range(root + 1, model.nbody):
        if model.body_parentid[body] in descendants:
            descendants.add(body)
    return {i for i in range(model.ngeom) if model.geom_bodyid[i] in descendants}


def validate_hold(robot, seconds=3.0):
    """Check physical settling and bounded home hold, not task or reachability success."""
    model, data = robot.model, robot.data
    book_ids = [i for i in range(model.nbody) if model.body(i).name.startswith("book_")]
    initial = data.xpos[book_ids].copy()
    robot_geoms = robot_geom_ids(model)
    peak_robot_penetration = 0.0
    peak_book_penetration = 0.0
    peak_effort = 0.0
    for _ in range(round(seconds / model.opt.timestep)):
        robot.step()
        peak_effort = max(peak_effort, float(np.max(np.abs(robot.last_effort))))
        for c in data.contact:
            if int(c.geom1) in robot_geoms or int(c.geom2) in robot_geoms:
                peak_robot_penetration = max(peak_robot_penetration, -float(c.dist))
            else:
                peak_book_penetration = max(peak_book_penetration, -float(c.dist))
    book_velocity = []
    for body in book_ids:
        joint = model.body_jntadr[body]
        start = model.jnt_dofadr[joint]
        book_velocity.append(float(np.linalg.norm(data.qvel[start : start + 6])))
    return {
        "simulated_seconds": float(data.time),
        "book_count": len(book_ids),
        "book_max_translation_m": float(
            np.max(np.linalg.norm(data.xpos[book_ids] - initial, axis=1))
        ),
        "book_max_velocity_norm": max(book_velocity),
        "peak_robot_penetration_m": peak_robot_penetration,
        "peak_book_penetration_m": peak_book_penetration,
        "peak_arm_effort_nm": peak_effort,
        "arm_max_hold_error_rad": float(
            np.max(np.abs(robot.arm_position - robot.home))
        ),
        "warning_count": sum(int(w.number) for w in data.warning),
    }
