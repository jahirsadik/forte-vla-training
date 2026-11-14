"""Isolated, review-first human recordings using the normal benchmark physics."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

from controllers.gamepad import GamepadControl, TeleopSettings
from environments.bookshelf_task import BookshelfEnv
from environments.bookshelf_hard import BookshelfHardEnv
from environments.sorting import SortingEnv
from utils.contracts import PROJECT, implementation_provenance
from utils.demonstrations import EpisodeWriter, validate_episode
from utils.research_protocol import load_manifest, manifest_digest, manifest_path

ENVIRONMENTS = {"sorting": SortingEnv, "bookshelf": BookshelfEnv, "bookshelf-hard": BookshelfHardEnv}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def source_inventory(project=PROJECT):
    result = {}
    for folder in ("environments", "controllers", "robots", "utils", "scripts", "configs", "learning", "assets", "tools"):
        for path in sorted((project / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                result[str(path.relative_to(project))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def archive_source(root):
    inventory = source_inventory()
    identity = manifest_digest(inventory)
    archive = Path(root) / "sources" / identity
    if archive.exists():
        if source_inventory(archive) != inventory:
            raise ValueError("Existing teleoperation source archive was modified")
    else:
        temporary = archive.with_name(identity + ".tmp-" + uuid.uuid4().hex[:8])
        for name in inventory:
            destination = temporary / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PROJECT / name, destination)
        if source_inventory(temporary) != inventory:
            raise ValueError("Source changed while archiving; restart teleoperation")
        write_json(temporary / "archive.json", dict(created_utc=utc_now(), files=inventory))
        temporary.rename(archive)
    return archive


def available_cases(environment, seed=0, layout=None):
    manifest = load_manifest(manifest_path(environment))
    layout = layout or ("mixed" if environment == "bookshelf-hard" else "shuffled")
    cases = [c for c in manifest["cases"] if c["environment"] == environment and c["split"] == "train"
             and c["axis"] == "development" and c["asset_set"] == "nominal"
             and c["seed"] == seed and c["layout"] == layout and c["recovery"] == "none"]
    if not cases:
        raise ValueError("Choose a frozen training seed/layout; held-out cases are not teleoperation practice scenes")
    return manifest, cases


class TeleopSession:
    def __init__(self, *, environment="sorting", mode="cartesian", task=None, seed=0, layout=None,
                 output_root=None, operator="local", settings=None, horizon_steps=None,
                 image_size=(480, 640), device=None, source_archive=None):
        self.manifest, self.cases = available_cases(environment, seed, layout)
        self.case_index = 0
        if task is not None:
            matches = [i for i, c in enumerate(self.cases) if c["task_id"] == task]
            if len(matches) != 1:
                raise ValueError(f"Unknown training task. Choose: {[c['task_id'] for c in self.cases]}")
            self.case_index = matches[0]
        if mode not in GamepadControl.modes:
            raise ValueError("Unknown control mode")
        self.settings = settings or TeleopSettings()
        root = Path(output_root or PROJECT / "outputs/teleoperation").resolve()
        root.mkdir(parents=True, exist_ok=True)
        source = Path(source_archive).resolve() if source_archive is not None else None
        if source is not None:
            # MuJoCo's compiled fingerprint includes asset paths. Collection and
            # replay must both execute from this exact frozen project directory.
            if source != PROJECT or source_inventory(source) != json.loads((source / "archive.json").read_text())["files"]:
                raise ValueError("Launch recording from the verified source archive")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = root / (stamp + "_" + environment + "_" + uuid.uuid4().hex[:8])
        (self.path / "episodes").mkdir(parents=True)
        self.constructor = dict(asset_set="nominal", observation_mode="rgb", cameras=["front", "wrist"], image_size=list(image_size))
        if horizon_steps is not None:
            self.constructor["horizon_steps"] = horizon_steps
        self.report = dict(schema="forte-human-teleop-v1", started_utc=utc_now(), operator=operator,
                           environment=environment, source_archive=str(source) if source else None,
                           device=device or {}, settings=asdict(self.settings), episodes=[], status="open",
                           review_status="unreviewed", training_admitted=False)
        self.env = ENVIRONMENTS[environment](**self.constructor)
        self.report["implementation"] = self.env.implementation
        self.mode = mode
        self.writer = None
        self.paused = True
        self.events = []
        self.message = "Ready. Release sticks/triggers and RB, then A / Space to record."
        self.wall_start = None
        self.last_step_wall = None
        self.diagnostics = {}
        self.reset()
        self.save_report()

    @property
    def case(self):
        return self.cases[self.case_index]

    def save_report(self):
        write_json(self.path / "session.json", self.report)

    def event(self, kind, **extra):
        self.events.append(dict(kind=kind, utc=utc_now(), step=self.info["episode"]["steps"], **extra))

    def reset(self):
        if self.writer is not None:
            raise ValueError("Save/end this episode before resetting")
        options = dict(task_id=self.case["task_id"], layout=self.case["layout"])
        if self.case["environment"] == "sorting":
            options["clutter_level"] = self.case["clutter_level"]
        self.obs, self.info = self.env.reset(seed=self.case["seed"], options=options)
        self.control = GamepadControl(self.env, self.mode, self.settings)
        self.events = []
        self.paused = True
        self.diagnostics = {}

    def toggle_mode(self):
        if self.writer is not None:
            self.message = "Save/end with B first; each recording uses one control mode."
            return
        self.mode = "joint" if self.mode == "cartesian" else "cartesian"
        self.reset()
        self.message = f"{self.mode.upper()} selected. A / Space starts a new recording."

    def next_task(self):
        if self.writer is not None:
            self.message = "Save/end with B before changing task."
            return
        self.case_index = (self.case_index + 1) % len(self.cases)
        self.reset()
        self.message = "Next training instruction. A / Space to record."

    def toggle_record(self, pad):
        if self.writer is not None and not self.paused:
            self.pause("operator_pause")
            return False
        if not pad.neutral(self.settings.deadzone):
            self.message = "To start/resume: connect pad, release RB, center sticks and release triggers."
            return False
        if self.writer is None:
            if implementation_provenance()["source_sha256"] != self.env.implementation["source_sha256"]:
                raise RuntimeError("Source changed during session; restart to keep recordings replayable")
            self.reset()
            episode_id = f"{len(self.report['episodes'])+1:04d}_{self.mode}_{self.case['task_id']}_{uuid.uuid4().hex[:8]}"
            metadata = dict(self.case, manifest_sha256=manifest_digest(self.manifest),
                            constructor=self.constructor, policy="human_gamepad", custom_horizon="horizon_steps" in self.constructor,
                            teleoperation=dict(protocol="forte-human-teleop-v1", mode=self.mode, operator=self.report["operator"],
                                               settings=asdict(self.settings), review_status="unreviewed", training_admitted=False,
                                               source_archive=self.report["source_archive"], device=self.report["device"]))
            self.writer = EpisodeWriter(self.path / "episodes" / (episode_id + ".h5"), self.env, self.obs, self.info, metadata=metadata)
            self.wall_start = time.monotonic()
            self.last_step_wall = None
            self.event("recording_started", mode=self.mode)
        else:
            self.event("resumed")
        self.paused = False
        self.message = "RECORDING. Hold RB to move; LB slows down. B saves/ends."
        return True

    def pause(self, reason):
        if self.writer is not None and not self.paused:
            self.event("paused", reason=reason)
            self.writer.file.flush()
        self.paused = True
        self.message = f"PAUSED ({reason}). Release controls, then A / Space to resume."

    def step(self, pad):
        if self.writer is None or self.paused:
            return False
        if not pad.connected:
            self.pause("controller_disconnected")
            return False
        now = time.monotonic()
        action, diagnostics = self.control.action(pad)
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        # Human actions remain distinct from verified scripted expert labels.
        # Review/admission is a later explicit operation, never automatic.
        self.writer.append(action, self.obs, reward, terminated, truncated, self.info,
                           source="human", stage=self.mode + ("_enabled" if diagnostics["enabled"] else "_hold"))
        self.writer._text("teleoperation/input_json", json.dumps(asdict(pad)))
        self.writer._text("teleoperation/controller_json", json.dumps(diagnostics))
        self.writer._append("teleoperation/wall_elapsed_s", now - self.wall_start)
        self.writer._append("teleoperation/wall_interval_s", 0. if self.last_step_wall is None else now - self.last_step_wall)
        self.last_step_wall = now
        self.diagnostics = diagnostics
        if self.writer.count % 20 == 0:
            self.writer.file.flush()
        if terminated or truncated:
            self.finish("environment_done")
        return True

    def finish(self, reason="operator_end", error=None):
        if self.writer is None:
            return None
        writer, self.writer = self.writer, None
        self.paused = True
        self.event("recording_ended", reason=reason)
        row = dict(path=str(writer.path), mode=self.mode, case_id=self.case["case_id"], task=self.case["task_id"],
                   reason=reason, error=error, review_status="unreviewed", training_admitted=False,
                   pause_count=sum(e["kind"] == "paused" for e in self.events), completed_utc=utc_now())
        try:
            writer.finish(self.info, error=error, events=self.events)
            row.update(validate_episode(writer.path))
        finally:
            writer.close()
            self.report["episodes"].append(row)
            self.save_report()
        outcome = "SUCCESS" if self.info["is_success"] else ("FAILED" if self.info["episode"]["done"] else "STOPPED")
        self.message = f"Saved {writer.count} steps: {outcome}. A retries; X changes mode; Menu changes task."
        return row

    def close(self, reason="window_closed", error=None):
        try:
            self.finish(reason, error=error)
        finally:
            self.report.update(status="closed", closed_utc=utc_now())
            self.save_report()
            self.env.close()
