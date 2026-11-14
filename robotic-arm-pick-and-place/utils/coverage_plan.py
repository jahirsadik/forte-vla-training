"""Frozen, train-only sorting coverage and counterfactual collection selections.

The plan references existing research cases. It does not mint cases, move split
membership, or choose collection episodes after inspecting expert outcomes.
"""

from copy import deepcopy
import json
from pathlib import Path

from utils.research_protocol import load_manifest, manifest_digest, validate_manifest

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = PROJECT / "configs/sorting_coverage.json"
VERSION = "forte-sorting-coverage-v1"
OBJECTS = ("red_cube", "blue_cylinder", "green_capsule", "yellow_wedge", "purple_bar", "orange_puck")
TRAYS = ("red", "blue", "green")
ASSET_SETS = ("nominal", "train_small")
SEEDS = (0, 1, 2)
RECOVERIES = ("missed_grasp", "delayed_release")
RESET_FIELDS = ("environment", "asset_set", "seed", "layout", "appearance", "appearance_seed", "clutter_level")


def plan_digest(plan):
    """Canonical JSON digest, independent of file whitespace and key order."""
    return manifest_digest(plan)


def scene_key(case):
    """Identity of a complete initial scene, independent of instruction/fault."""
    return manifest_digest({key: case[key] for key in RESET_FIELDS})[:20]


def pair_key(case):
    """Group the two allowed destinations for an object in the same scene."""
    obj, separator, tray = case["task_id"].rpartition("_to_")
    if not separator or obj not in OBJECTS or tray not in TRAYS:
        raise ValueError("Coverage case must use a known sorting object and tray")
    return scene_key(case), case["recovery"], obj


def build_plan(manifest):
    """Build the fixed v1 recipe. Call before any coverage or collection runs."""
    validate_manifest(manifest)
    expected_heldout = [f"{obj}_to_{TRAYS[(index + 2) % len(TRAYS)]}"
                        for index, obj in enumerate(OBJECTS)]
    if manifest.get("heldout_combinations", {}).get("sorting") != expected_heldout:
        raise ValueError("Coverage v1 requires the unchanged sorting combination holdouts")
    tasks = [f"{obj}_to_{tray}" for obj in OBJECTS for tray in TRAYS
             if f"{obj}_to_{tray}" not in expected_heldout]
    by_spec = {}
    for case in manifest["cases"]:
        if case["environment"] != "sorting" or case["split"] != "train":
            continue
        key = (case["task_id"], case["asset_set"], case["seed"], case["axis"], case["recovery"])
        by_spec.setdefault(key, []).append(case)

    def one(task, asset, seed, recovery):
        axis = "development" if recovery == "none" else recovery
        matches = by_spec.get((task, asset, seed, axis, recovery), [])
        if len(matches) != 1:
            raise ValueError(f"Coverage needs exactly one frozen training case for {task}/{asset}/{seed}/{recovery}")
        case = matches[0]
        expected = dict(environment="sorting", split="train", asset_set=asset, seed=seed,
                        axis=axis, recovery=recovery, layout="shuffled", appearance="nominal",
                        appearance_seed=seed + 90000, clutter_level=1)
        if any(type(case.get(key)) is not type(value) or case.get(key) != value
               for key, value in expected.items()):
            raise ValueError("Coverage case changed its frozen reset or perturbation fields")
        return case

    nominal = [one(task, asset, seed, "none") for asset in ASSET_SETS for seed in SEEDS for task in tasks]
    recovery = [one(task, "nominal", 0, fault) for fault in RECOVERIES for task in tasks]
    scene_groups = []
    for asset in ASSET_SETS:
        for seed in SEEDS:
            members = [case for case in nominal if case["asset_set"] == asset and case["seed"] == seed]
            first = members[0]
            scene_groups.append(dict(
                scene_id=scene_key(first), family_id=first["family_id"],
                reset={key: first[key] for key in RESET_FIELDS},
                nominal_case_ids=[case["case_id"] for case in members],
                recovery_case_ids=[case["case_id"] for case in recovery if scene_key(case) == scene_key(first)],
                task_ids=tasks,
                destination_pairs={obj: [task for task in tasks if task.rpartition("_to_")[0] == obj]
                                   for obj in OBJECTS},
            ))
    return dict(
        version=VERSION,
        scope="Train-only expert coverage and counterfactual collection; no validation/test execution or generalization claim",
        manifest_sha256=manifest_digest(manifest),
        environment="sorting", split="train",
        groups=dict(nominal=[case["case_id"] for case in nominal],
                    recovery=[case["case_id"] for case in recovery]),
        scene_groups=scene_groups,
        collection_subsets=dict(
            nominal_seed0=[case["case_id"] for case in nominal if case["seed"] == 0],
            recovery_seed0=[case["case_id"] for case in recovery]),
        collection=dict(split="train", image_size_hw=[480, 640], cameras=["front", "wrist"],
                        execution_horizon_steps=1,
                        selection="Declared case IDs only; retain failures; admit only separately validated complete expert demonstrations",
                        counterfactual_unit="Complete paired destinations per object and scene; report missing/failed members without replacing them"),
    )


def validate_plan(plan, manifest):
    """Reject split leakage, omissions, altered recipe, and stale provenance.

    V1 is intentionally a closed schema. A changed experiment requires a new
    version/recipe instead of silently weakening coverage to successful cases.
    """
    if not isinstance(plan, dict) or plan.get("version") != VERSION:
        raise ValueError("Unsupported sorting coverage plan version")
    if plan.get("manifest_sha256") != manifest_digest(manifest):
        raise ValueError("Coverage plan manifest hash mismatch")
    expected = build_plan(manifest)
    # JSON typing matters: Python dict equality would accept True for1 or1.0.
    if plan_digest(plan) != plan_digest(expected):
        raise ValueError("Coverage plan differs from the frozen train-only recipe (case IDs, groups, counterfactual pairs, reset fields or collection settings)")
    return plan


def load_plan(path=DEFAULT_PLAN, manifest=None):
    manifest = load_manifest() if manifest is None else manifest
    plan = json.loads(Path(path).read_text())
    return validate_plan(plan, manifest)


def select_cases(plan, manifest, group="all"):
    """Return detached case dictionaries without changing manifest or plan."""
    validate_plan(plan, manifest)
    if group not in ("nominal", "recovery", "all"):
        raise ValueError("Coverage group must be nominal, recovery or all")
    ids = plan["groups"][group] if group != "all" else plan["groups"]["nominal"] + plan["groups"]["recovery"]
    by_id = {case["case_id"]: case for case in manifest["cases"]}
    return [deepcopy(by_id[case_id]) for case_id in ids]


def select_collection_cases(plan, manifest, subset):
    """Select a predeclared collection subset, never a success-derived filter."""
    validate_plan(plan, manifest)
    if subset not in plan["collection_subsets"]:
        raise ValueError("Unknown declared collection subset")
    by_id = {case["case_id"]: case for case in manifest["cases"]}
    return [deepcopy(by_id[case_id]) for case_id in plan["collection_subsets"][subset]]
