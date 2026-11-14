"""Xbox teleoperation with two camera views and separate, unreviewed recordings."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import textwrap
import time

# Offscreen MuJoCo rendering is separate from the SDL control window.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controllers.gamepad import PadState, TeleopSettings
from utils.teleoperation import ENVIRONMENTS, PROJECT, TeleopSession, archive_source, available_cases


BUTTONS = dict(a="A", b="B", x="X", y="Y", lb="LEFTSHOULDER", rb="RIGHTSHOULDER",
               left="DPAD_LEFT", right="DPAD_RIGHT", up="DPAD_UP", down="DPAD_DOWN",
               menu="START", view="BACK", ls="LEFTSTICK", rs="RIGHTSTICK")
AXES = ("LEFTX", "LEFTY", "RIGHTX", "RIGHTY", "TRIGGERLEFT", "TRIGGERRIGHT")


class XboxDevice:
    """SDL's named controller mapping handles USB/Bluetooth axis differences."""
    def __init__(self, pygame, index=0):
        from pygame._sdl2 import controller
        self.pg, self.api, self.index = pygame, controller, index
        controller.init()
        self.pad = None
        self.identity = {}
        self.error = "No supported controller. Connect Xbox via USB or Bluetooth."

    def inventory(self):
        return [dict(index=i, name=self.api.name_forindex(i), supported=self.api.is_controller(i))
                for i in range(self.api.get_count())]

    def read(self):
        if self.pad is not None and not self.pad.attached():
            self.pad.quit()
            self.pad = None
        if self.pad is None:
            if self.index >= self.api.get_count() or not self.api.is_controller(self.index):
                return PadState(connected=False)
            self.pad = self.api.Controller(self.index)
            joystick = self.pad.as_joystick()
            self.identity = dict(name=self.pad.name, guid=joystick.get_guid(),
                                 instance_id=joystick.get_instance_id(), mapping=self.pad.get_mapping())
            self.error = ""
        try:
            axes = [self.pad.get_axis(getattr(self.pg, "CONTROLLER_AXIS_" + x)) for x in AXES]
            sticks = [max(-1., min(1., x / (32768. if x < 0 else 32767.))) for x in axes[:4]]
            triggers = [max(0., min(1., x / 32767.)) for x in axes[4:]]
            buttons = tuple(k for k, v in BUTTONS.items() if self.pad.get_button(getattr(self.pg, "CONTROLLER_BUTTON_" + v)))
            return PadState(*sticks, *triggers, buttons=buttons)
        except self.pg.error as exc:
            self.error = str(exc)
            self.pad.quit()
            self.pad = None
            return PadState(connected=False)

    def close(self):
        if self.pad is not None:
            self.pad.quit()


class ControlWindow:
    size = (1280, 920)

    def __init__(self, pg):
        self.pg = pg
        pg.font.init()
        sizes = pg.display.get_desktop_sizes()
        w, h = sizes[0] if sizes else (1440, 1080)
        self.window = pg.display.set_mode((min(1280, w - 60), min(920, h - 100)), pg.RESIZABLE)
        pg.display.set_caption("Forte | Xbox teleoperation | Simulation only")
        self.canvas = pg.Surface(self.size)
        self.font = pg.font.SysFont("DejaVu Sans", 18)
        self.small = pg.font.SysFont("DejaVu Sans", 16)
        self.title = pg.font.SysFont("DejaVu Sans", 27, bold=True)

    def text(self, value, x, y, color="#dbe4ef", font=None):
        self.canvas.blit((font or self.font).render(str(value), True, color), (x, y))

    def draw(self, session, pad, device, rate):
        pg, s = self.pg, session
        self.canvas.fill("#111924")
        active = s.writer is not None and not s.paused
        status = "RECORDING" if active else ("PAUSED" if s.writer else "READY / SAVED")
        self.text(f"FORTE  /  {s.mode.upper()}  /  {status}", 20, 12,
                  "#6ce3ab" if active else "#ffc06b", self.title)
        self.text(f"{s.case['environment']} | task {s.case_index+1}/{len(s.cases)} | seed {s.case['seed']} | sim {s.info['simulated_seconds']:.1f}s / {s.env.horizon*s.env.control_dt:.0f}s | {rate:.1f} steps/s", 20, 49)
        for i, line in enumerate(textwrap.wrap(s.obs["instruction"], 115)):
            self.text(line, 20, 77 + i*23)
        for camera, x in (("front", 0), ("wrist", 640)):
            pixels = s.obs["rgb"][camera]
            image = pg.image.frombuffer(pixels.tobytes(), (pixels.shape[1], pixels.shape[0]), "RGB")
            self.canvas.blit(pg.transform.smoothscale(image, (640, 480)), (x, 144))
            self.text(camera.upper() + " CAMERA", x+18, 118, "#8ba8c6", self.small)
        self.text(s.message, 18, 633, "#ffc06b", self.small)
        enabled = pad.connected and pad.down("rb")
        self.text(f"{'CONNECTED' if pad.connected else 'DISCONNECTED'} | {'RB ENABLED' if enabled else 'RB released: hold references'} | {'PRECISION' if pad.down('lb') else 'NORMAL SPEED'}", 18, 660)
        joint = s.control.selected_joint
        if s.mode == "cartesian":
            lines = ["RB held: left stick = forward/back (+X/-X), right/left (-Y/+Y); right stick up/down = height (+Z/-Z).",
                     "Right stick left/right = yaw (+/-Z). Hold Y: left stick up/down = roll (+/-X), left/right = pitch (+/-Y).",
                     "D-pad left/right = elbow posture bias while holding tool pose approximately. LB = 1/4 speed."]
        else:
            lines = [f"Selected joint {joint+1}/7: {s.env.robot.joint_names[joint]}  |  reference {s.env.robot.target_q[joint]:+.3f} rad",
                     "D-pad left/right selects joint (one press per joint). Hold RB + left stick up/down to increase/decrease angle.",
                     "All seven joints are available, including independent wrist yaw. LB = 1/4 speed."]
        lines += ["Both modes: hold RB + LT to close / RT to open gripper. Release both triggers to keep current width.",
                  "A / Space: start or pause/resume    B / Enter: save/end    X / Tab: change mode (between recordings)",
                  "Menu / N: next instruction (between recordings)    View / Esc: save and quit"]
        for i, line in enumerate(lines):
            self.text(line, 18, 692 + i*25, font=self.small)
        limited = "REFERENCE LIMIT / TRACKING LAG" if s.diagnostics.get("reference_limited") else ""
        self.text(f"LS {pad.lx:+.2f},{pad.ly:+.2f}   RS {pad.rx:+.2f},{pad.ry:+.2f}   LT {pad.lt:.2f} RT {pad.rt:.2f}   {limited}", 18, 849, "#8ba8c6", self.small)
        self.text(f"Session: {s.path.name} | all outcomes saved; human labels await review", 18, 878, font=self.small)
        self.window.blit(pg.transform.smoothscale(self.canvas, self.window.get_size()), (0, 0))
        pg.display.flip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cartesian", "joint"), default="cartesian")
    parser.add_argument("--env", choices=ENVIRONMENTS, default="sorting")
    parser.add_argument("--task", help="Training instruction ID; use --list-tasks")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--layout", help="Frozen training layout; default shuffled (mixed for bookshelf-hard)")
    parser.add_argument("--seconds", type=int, help="Optional practice horizon; logged separately from the benchmark default")
    parser.add_argument("--operator", default="local")
    parser.add_argument("--output-dir", type=Path, help="Separate root; default outputs/teleoperation")
    parser.add_argument("--controller-index", type=int, default=0)
    parser.add_argument("--deadzone", type=float, default=.12)
    parser.add_argument("--check-controller", action="store_true", help="Print mapped devices and exit without simulation")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--source-archive", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.seconds is not None and args.seconds < 1 or args.controller_index < 0:
        parser.error("seconds must be positive and controller index nonnegative")
    settings = TeleopSettings(deadzone=args.deadzone)
    _, cases = available_cases(args.env, args.seed, args.layout)
    if args.list_tasks:
        print("\n".join(c["task_id"] for c in cases))
        return 0
    if not args.check_controller and args.source_archive is None:
        root = (args.output_dir or PROJECT / "outputs/teleoperation").resolve()
        root.mkdir(parents=True, exist_ok=True)
        archive = archive_source(root)
        print(f"Using frozen teleoperation source: {archive}", flush=True)
        arguments = list(sys.argv[1:] if argv is None else argv)
        os.execv(sys.executable, [sys.executable, str(archive / "scripts/run_teleop.py"),
                                 *arguments, "--output-dir", str(root), "--source-archive", str(archive)])
    try:
        import pygame as pg
    except ImportError:
        parser.error("Install requirements-teleop.txt in the repository Conda environment")
    pg.display.init()
    device = XboxDevice(pg, args.controller_index)
    session = None
    error = None
    try:
        if args.check_controller:
            pg.display.set_mode((1, 1), pg.HIDDEN)
            pg.event.pump()
            print(json.dumps(dict(devices=device.inventory(), state=asdict(device.read()), identity=device.identity), indent=2))
            return 0
        window = ControlWindow(pg)
        pg.event.pump()
        device.read()
        session = TeleopSession(environment=args.env, mode=args.mode, task=args.task, seed=args.seed,
                               layout=args.layout, operator=args.operator, output_root=args.output_dir,
                               settings=settings, horizon_steps=None if args.seconds is None else args.seconds * 20,
                               device=device.identity, source_archive=args.source_archive)
        print(f"Recordings: {session.path}\nA starts; hold RB to move; B saves; X changes mode.", flush=True)
        previous = set()
        running = True
        focus = True
        clock = pg.time.Clock()
        last_tick = time.monotonic()
        disk_checked = last_tick
        rate = 0.
        while running:
            commands = set()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.WINDOWFOCUSLOST:
                    focus = False
                    session.pause("window_focus_lost")
                elif event.type == pg.WINDOWFOCUSGAINED:
                    focus = True
                elif event.type == pg.KEYDOWN:
                    key = {pg.K_SPACE: "a", pg.K_RETURN: "b", pg.K_TAB: "x", pg.K_n: "menu", pg.K_ESCAPE: "view"}.get(event.key)
                    if key:
                        commands.add(key)
            old_instance = device.identity.get("instance_id")
            pad = device.read()
            if old_instance != device.identity.get("instance_id") and session.writer is not None:
                session.event("controller_changed", device=device.identity)
                session.pause("controller_changed")
            commands.update(set(pad.buttons) - previous)
            previous = set(pad.buttons)
            now = time.monotonic()
            if not pad.connected and session.writer is not None and not session.paused:
                session.pause("controller_disconnected")
            # Event processing stalls must not replay a backlog of stick commands.
            if now - last_tick > .5 and session.writer is not None and not session.paused:
                session.pause("control_loop_stall")
            last_tick = now
            if "view" in commands:
                running = False
            if not running:
                break
            if focus:
                # End wins over resume if two lifecycle buttons arrive together.
                if "b" in commands:
                    session.finish()
                elif "a" in commands:
                    session.report["device"] = device.identity
                    session.toggle_record(pad)
                elif "x" in commands:
                    session.toggle_mode()
                elif "menu" in commands:
                    session.next_task()
            if now - disk_checked > 2:
                disk_checked = now
                if shutil.disk_usage(session.path).free < 1024**3:
                    session.finish("low_disk", error="Less than 1 GiB free; recording stopped")
                    raise OSError("Less than 1 GiB free; recording saved and stopped")
            before = time.monotonic()
            if session.step(pad):
                elapsed = time.monotonic() - before
                rate = .9 * rate + .1 * min(20., 1 / max(elapsed, 1e-6))
            window.draw(session, pad, device, rate)
            clock.tick(20)  # No catch-up steps; simulation always advances exactly 0.05 s.
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if session is not None:
                session.close(error=error)
        finally:
            device.close()
            pg.quit()


if __name__ == "__main__":
    raise SystemExit(main())
