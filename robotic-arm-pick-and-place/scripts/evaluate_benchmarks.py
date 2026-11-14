"""Run frozen task/seed suites, save metrics and action replay, or inspect a benchmark."""

import argparse
from collections import Counter, defaultdict
import importlib
import json
from pathlib import Path
import sys
import time

import gymnasium
import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from environments.bookshelf_task import BookshelfEnv
from environments.benchmark import DEFAULT_IMAGE_SIZE
from environments.sorting import SortingEnv
from environments.bookshelf_hard import BookshelfHardEnv
from utils.policy import PolicySession
from utils.actions import LEARNING_INTERFACE

ENVIRONMENTS = {"bookshelf": BookshelfEnv, "sorting": SortingEnv, "bookshelf-hard": BookshelfHardEnv}


def confidence_interval(successes, count):
    """Wilson binomial interval at 95%, with all requested episodes in the denominator."""
    if not count:
        return None
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0, center - half), min(1, center + half)]


def summarize(episodes):
    n = len(episodes)
    successes = sum(e.get("is_success", False) for e in episodes)
    valid = [e for e in episodes if "reset_error" not in e]
    return dict(
        episodes=n,
        successes=successes,
        success_rate=successes / n if n else None,
        success_rate_ci95=confidence_interval(successes, n),
        invalid_resets=n - len(valid),
        timeouts=sum(e.get("metrics", {}).get("timeout", False) for e in episodes),
        failures=dict(
            Counter(
                e.get("metrics", {}).get("failure_reason") or "invalid_reset"
                for e in episodes
                if "reset_error" in e or e.get("metrics", {}).get("failure_reason")
            )
        ),
        mean_return=float(np.mean([e["episode"]["episode_return"] for e in valid]))
        if valid
        else None,
        mean_steps=float(np.mean([e["episode"]["steps"] for e in valid]))
        if valid
        else None,
        peak_contact_force_n=max(
            [0, *[e["episode"]["peak_contact_force_n"] for e in valid]]
        ),
        peak_distractor_translation_m=max(
            [0, *[e["episode"]["peak_distractor_translation_m"] for e in valid]]
        ),
    )


def save_views(env, destination):
    from PIL import Image

    for camera in ("overview", "front", "overhead", "wrist"):
        Image.fromarray(env._image(camera)).save(destination / f"{camera}.png")


def replay(path, render=False):
    record = json.loads(path.read_text())
    with ENVIRONMENTS[record["environment"]](**record["constructor"]) as env:
        env.set_state(record["initial_state"])
        trajectory = np.load(path.with_suffix(".npz"), allow_pickle=False)
        for i, action in enumerate(trajectory["actions"]):
            _, reward, terminated, truncated, _ = env.step(action)
            np.testing.assert_allclose(
                reward, trajectory["rewards"][i], atol=1e-12, rtol=0
            )
            assert terminated == trajectory["terminated"][i]
            assert truncated == trajectory["truncated"][i]
        np.testing.assert_allclose(
            env._physics_state(), record["final_physics"], atol=1e-10, rtol=0
        )
        if render:
            save_views(env, path.parent)
    print(
        f"PASS: replayed {len(trajectory['actions'])} actions and matched rewards, endings and final physics"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVIRONMENTS, default="bookshelf")
    parser.add_argument("--suite", choices=("smoke", "train", "test"), default="smoke")
    parser.add_argument("--task", help="Limit to one task ID; omitted runs all tasks")
    parser.add_argument("--layout", help="Reset layout; hard profiles: narrow, mixed, touching, staggered")
    parser.add_argument(
        "--policy",
        default="hold",
        help="hold, random, or importable module:function taking a policy observation",
    )
    parser.add_argument("--reward", choices=("sparse", "dense"), default="sparse")
    parser.add_argument("--execute-actions", type=int, default=LEARNING_INTERFACE["execution_horizon_steps"])
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Include RGB camera observations for a vision policy",
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--image-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"),
                        default=DEFAULT_IMAGE_SIZE, help="Native camera capture size (default: 480 640)")
    parser.add_argument(
        "--render", action="store_true", help="Save initial camera views"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Save replay checkpoints and action/reward arrays",
    )
    parser.add_argument("--clutter-level", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override horizon; labels report as a custom protocol",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--replay",
        type=Path,
        help="Replay one recorded episode JSON and verify its physics",
    )
    args = parser.parse_args()
    if args.replay:
        replay(args.replay, args.render)
        return 0
    suites = json.loads((PROJECT / "configs/benchmark_suites.json").read_text())
    suite = suites[args.suite]
    constructor = dict(
        reward_mode=args.reward,
        observation_mode="rgb" if args.rgb else "proprio",
        render_mode="human" if args.viewer else None,
        image_size=list(args.image_size),
        horizon_steps=args.max_steps
        if args.max_steps is not None
        else suite["horizon_steps"],
    )
    destination = (
        args.output_dir or PROJECT / "outputs/benchmarks" / args.env / args.suite
    )
    destination.mkdir(parents=True, exist_ok=True)
    custom_policy = None
    if args.policy not in ("hold", "random"):
        module, name = args.policy.rsplit(":", 1)
        custom_policy = getattr(importlib.import_module(module), name)
    episodes = []
    with ENVIRONMENTS[args.env](**constructor) as env:
        session = PolicySession(custom_policy, env.action_space, execute_actions=args.execute_actions) if custom_policy is not None else None
        tasks = [args.task] if args.task else env.task_ids
        for task in tasks:
            if task not in env.tasks:
                parser.error(f"Unknown task {task}; choose from {env.task_ids}")
            for seed in suite["seeds"]:
                options = {"task_id": task}
                if args.layout is not None:
                    options["layout"] = args.layout
                if args.env == "sorting":
                    options["clutter_level"] = args.clutter_level
                try:
                    observation, info = env.reset(seed=seed, options=options)
                except RuntimeError as error:
                    episodes.append(
                        dict(
                            task_id=task,
                            seed=seed,
                            is_success=False,
                            reset_error=str(error),
                        )
                    )
                    continue
                initial = env.get_state() if args.record else None
                if session is not None:
                    session.reset(seed=seed)
                if args.render and not episodes:
                    save_views(env, destination)
                actions, rewards, ends, truncations = [], [], [], []
                rng = np.random.default_rng(seed)
                for _ in range(env.horizon):
                    started = time.perf_counter()
                    if session is not None:
                        action = session(observation)
                    elif args.policy == "random":
                        action = rng.uniform(-1, 1, env.action_space.shape).astype(
                            np.float32
                        )
                    else:
                        action = np.r_[np.zeros(len(env.robot.home)), 1].astype(
                            np.float32
                        )
                    observation, reward, terminated, truncated, info = env.step(action)
                    actions.append(np.asarray(action).copy())
                    rewards.append(reward)
                    ends.append(terminated)
                    truncations.append(truncated)
                    if args.viewer:
                        time.sleep(
                            max(0, env.control_dt - (time.perf_counter() - started))
                        )
                    if terminated or truncated:
                        break
                info["seed"] = seed
                episodes.append(info)
                if args.record:
                    stem = destination / f"episode_{len(episodes) - 1:03d}"
                    # A headless replay uses the same physics/protocol fingerprint.
                    replay_constructor = dict(constructor, render_mode=None)
                    record = dict(
                        environment=args.env,
                        constructor=replay_constructor,
                        initial_state=initial,
                        final_physics=env._physics_state().tolist(),
                        summary=info,
                    )
                    stem.with_suffix(".json").write_text(
                        json.dumps(record, indent=2) + "\n"
                    )
                    np.savez_compressed(
                        stem.with_suffix(".npz"),
                        actions=actions,
                        rewards=rewards,
                        terminated=ends,
                        truncated=truncations,
                    )
        groups = defaultdict(list)
        for episode in episodes:
            groups[episode["task_id"]].append(episode)
        report = dict(
            environment=args.env,
            suite=args.suite,
            custom_horizon=args.max_steps is not None,
            task_subset=args.task,
            policy=args.policy,
            execution_horizon_steps=args.execute_actions,
            constructor=constructor,
            protocol_version=env.protocol["protocol_version"],
            model_fingerprint=env.fingerprint,
            versions=dict(
                mujoco=mujoco.__version__,
                gymnasium=gymnasium.__version__,
                numpy=np.__version__,
            ),
            summary=summarize(episodes),
            per_task={k: summarize(v) for k, v in groups.items()},
            episodes=episodes,
        )
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {destination / 'report.json'}")
    return int(report["summary"]["invalid_resets"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
