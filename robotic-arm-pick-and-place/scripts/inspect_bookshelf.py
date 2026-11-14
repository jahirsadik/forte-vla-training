"""Inspect the bookshelf with active bounded robot hold and free physical books."""

from pathlib import Path
import argparse
import json
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import mujoco
from environments.bookshelf import SCENE, load_bookshelf, render_options, validate_hold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument(
        "--render", action="store_true", help="Save overview/front/overhead/wrist PNGs"
    )
    parser.add_argument("--collisions", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT / "outputs/bookshelf"
    )
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    robot = load_bookshelf(args.scene)
    report = validate_hold(robot, args.seconds)
    report["passed"] = (
        report["warning_count"] == 0
        and report["peak_robot_penetration_m"] < 1e-5
        and report["peak_book_penetration_m"] < 0.001
        and report["book_max_translation_m"] < 0.003
        and report["book_max_velocity_norm"] < 0.01
        and report["arm_max_hold_error_rad"] < 0.01
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    options = render_options(args.collisions)
    if args.render:
        from PIL import Image

        with mujoco.Renderer(robot.model, height=900, width=1200) as renderer:
            for camera in ("overview", "front", "overhead", "wrist"):
                renderer.update_scene(robot.data, camera=camera, scene_option=options)
                suffix = "_collisions" if args.collisions else ""
                Image.fromarray(renderer.render()).save(
                    args.output_dir / f"{camera}{suffix}.png"
                )
        print(f"Renders: {args.output_dir}", flush=True)
    if not args.headless:
        from mujoco import viewer as viewer_module

        with viewer_module.launch_passive(robot.model, robot.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = robot.model.camera("overview").id
            viewer.opt.geomgroup[:] = options.geomgroup
            viewer.opt.sitegroup[:] = options.sitegroup
            start = time.perf_counter() - robot.data.time
            while viewer.is_running():
                for _ in range(16):
                    robot.step()
                viewer.sync()
                time.sleep(max(0, start + robot.data.time - time.perf_counter()))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
