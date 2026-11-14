"""Inspect retained human trials, optionally replaying with their archived source."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np

from utils.demonstrations import replay_episode, validate_episode
from utils.teleoperation import source_inventory, write_json


def inspect_episode(path):
    result = dict(path=str(path), passed=False)
    try:
        validation = validate_episode(path)
        n = validation["transitions"]
        with h5py.File(path, "r") as f:
            meta = validation["metadata"]
            assert meta["policy"] == "human_gamepad"
            assert meta["teleoperation"]["protocol"] == "forte-human-teleop-v1"
            assert meta["teleoperation"]["review_status"] == "unreviewed"
            mode = meta["teleoperation"]["mode"]
            assert mode in ("cartesian", "joint")
            if n:
                assert all(s == "human" for s in f["transitions/source"].asstr()[:])
                assert not f["transitions/expert_action_valid"][:].any()
                for name in ("input_json", "controller_json", "wall_elapsed_s", "wall_interval_s"):
                    assert len(f["teleoperation/" + name]) == n
                elapsed = f["teleoperation/wall_elapsed_s"][:]
                assert np.isfinite(elapsed).all() and np.all(np.diff(elapsed) >= 0)
                for raw, diag in zip(f["teleoperation/input_json"].asstr()[:], f["teleoperation/controller_json"].asstr()[:]):
                    raw, diag = json.loads(raw), json.loads(diag)
                    assert raw["connected"] and diag["mode"] == mode
                    assert diag["enabled"] == ("rb" in raw["buttons"])
            events = json.loads(f.attrs["events"])
            summary = json.loads(f.attrs["summary"])
            result.update(passed=bool(validation["complete"]), mode=mode, task=meta["task_id"],
                          transitions=n, success=validation["success"], environment_done=bool(f.attrs["environment_done"]),
                          end_reason=events[-1].get("reason") if events else None,
                          collector_error=f.attrs["collector_error"], pauses=sum(e["kind"] == "paused" for e in events),
                          failure_reason=summary["metrics"].get("failure_reason"),
                          simulated_seconds=summary["simulated_seconds"], training_admitted=False,
                          source_sha256=json.loads(f.attrs["implementation"])["source_sha256"])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--replay-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.replay_worker:
        result = replay_episode(args.session)
        print(json.dumps(dict(passed=True, transitions=result["transitions"])))
        return 0
    session = args.session.resolve()
    report = json.loads((session / "session.json").read_text())
    archive = Path(report["source_archive"]) if report["source_archive"] else None
    if args.replay:
        if not archive or source_inventory(archive) != json.loads((archive / "archive.json").read_text())["files"]:
            raise ValueError("Missing or modified source archive; refusing ambiguous replay")
    results = []
    for path in sorted((session / "episodes").glob("*.h5")):
        item = inspect_episode(path)
        if args.replay and item["passed"]:
            process = subprocess.run([sys.executable, str(archive / "scripts/inspect_teleop.py"), str(path), "--replay-worker"],
                                     capture_output=True, text=True, env=dict(os.environ, MUJOCO_GL="egl"))
            item["replay"] = dict(passed=process.returncode == 0, stdout=process.stdout, stderr=process.stderr)
            item["passed"] &= process.returncode == 0
        results.append(item)
    saved = {str(Path(r["path"]).resolve()) for r in report["episodes"]}
    observed = {r["path"] for r in results}
    result = dict(passed=bool(results) and all(r["passed"] for r in results) and saved == observed,
                  session=str(session), episodes=results, episode_count=len(results),
                  successful_episodes=sum(r.get("success", False) for r in results),
                  unindexed_episodes=sorted(observed-saved), missing_episodes=sorted(saved-observed),
                  training_admitted=False, source_archive=str(archive) if archive else None)
    write_json(session / "inspection.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
