"""Frozen split membership and independently varied evaluation axes."""

import hashlib
import json
from pathlib import Path
from utils.contracts import robot_definition_digest

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT / "configs/research_manifest.json"
HARD_MANIFEST = PROJECT / "configs/research_manifest_hard.json"


def manifest_path(environment):
    return HARD_MANIFEST if environment == "bookshelf-hard" else MANIFEST


def hard_definition_digest():
    digest = hashlib.sha256()
    for name in ("bookshelf_hard.json", "benchmarks_hard.json"):
        digest.update(name.encode())
        digest.update((PROJECT / "configs" / name).read_bytes())
    return digest.hexdigest()


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_manifest():
    books = ["red", "blue", "teal", "green", "yellow", "olive"]
    book_tasks = [f"{op}_{book}" for book in books for op in ("extract", "relocate")] + ["insert_orange"]
    objects = ["red_cube", "blue_cylinder", "green_capsule", "yellow_wedge", "purple_bar", "orange_puck"]
    trays = ["red", "blue", "green"]
    sort_tasks = [f"{obj}_to_{tray}" for obj in objects for tray in trays]
    heldout = {
        "bookshelf": ["relocate_blue", "relocate_olive"],
        "sorting": [f"{obj}_to_{trays[(i + 2) % 3]}" for i, obj in enumerate(objects)],
    }
    cases = []
    for environment, tasks in (("bookshelf", book_tasks), ("sorting", sort_tasks)):
        seen = [task for task in tasks if task not in heldout[environment]]
        for split, seeds, assets in (("train", [0, 1, 2], ["nominal", "train_small"]), ("validation", [1000, 1001], ["validation"])):
            for asset in assets:
                for task in seen:
                    for seed in seeds:
                        cases.append(dict(environment=environment, split=split, axis="development", asset_set=asset, task_id=task, seed=seed, layout="shuffled", appearance="nominal", recovery="none"))
        for axis, asset, layout, appearance, recovery, task_group in (
            ("iid", "nominal", "shuffled", "nominal", "none", seen),
            ("objects", "heldout", "shuffled", "nominal", "none", seen),
            ("layouts", "nominal", "staggered" if environment == "bookshelf" else "arc", "nominal", "none", seen),
            ("combinations", "nominal", "shuffled", "nominal", "none", heldout[environment]),
            ("camera", "nominal", "shuffled", "camera_shift", "none", seen),
            ("lighting", "nominal", "shuffled", "lighting_dim", "none", seen),
            ("missed_grasp", "nominal", "shuffled", "nominal", "missed_grasp", seen),
            ("delayed_release", "nominal", "shuffled", "nominal", "delayed_release", seen),
        ):
            for task in task_group:
                for seed in (10000, 10001, 10002):
                    cases.append(dict(environment=environment, split="test", axis=axis, asset_set=asset, task_id=task, seed=seed, layout=layout, appearance=appearance, recovery=recovery))
        for recovery in ("missed_grasp", "delayed_release"):
            for task in seen:
                for seed in (0, 1, 2):
                    cases.append(dict(environment=environment, split="train", axis=recovery, asset_set="nominal", task_id=task, seed=seed, layout="shuffled", appearance="nominal", recovery=recovery))
    for case in cases:
        case["appearance_seed"] = case["seed"] + 90000
        case["clutter_level"] = 1
        family = {k: case[k] for k in ("environment", "asset_set", "seed", "layout")}
        case["family_id"] = manifest_digest(family)[:20]
        case["case_id"] = manifest_digest(case)[:20]
    return dict(version="forte-research-v2", protocol="forte-manipulation-v3", robot_definition_sha256=robot_definition_digest(), scope="pilot single-object tasks; no multi-step compositional generalization claim", object_holdout_scope="new dimension/mass instances of known shape categories; not unseen categories", heldout_combinations=heldout, cases=cases)


def load_manifest(path=MANIFEST):
    manifest = json.loads(Path(path).read_text())
    validate_manifest(manifest)
    return manifest


def build_hard_manifest():
    """Independent dense-book curriculum; the original frozen suite is untouched."""
    base = build_manifest()
    cases = []
    for old in base["cases"]:
        if old["environment"] != "bookshelf":
            continue
        layouts = ["touching" if old["axis"] == "layouts" else "mixed"]
        if old["split"] == "train" and old["axis"] == "development":
            layouts += ["narrow", "staggered"]
        for layout in layouts:
            case = {k: v for k, v in old.items() if k not in ("case_id", "family_id")}
            case.update(environment="bookshelf-hard", layout=layout)
            family = {k: case[k] for k in ("environment", "asset_set", "seed", "layout")}
            case["family_id"] = manifest_digest(family)[:20]
            case["case_id"] = manifest_digest(case)[:20]
            cases.append(case)
    return dict(version="forte-bookshelf-hard-research-v1", protocol="forte-bookshelf-hard-v1",
                robot_definition_sha256=robot_definition_digest(), environment_definition_sha256=hard_definition_digest(),
                scope="dense rigid books; contact-rich expert feasibility remains task dependent",
                heldout_combinations={"bookshelf-hard": base["heldout_combinations"]["bookshelf"]}, cases=cases)


def validate_manifest(manifest):
    versions = {("forte-research-v2", "forte-manipulation-v3"),
                ("forte-bookshelf-hard-research-v1", "forte-bookshelf-hard-v1")}
    if (manifest.get("version"), manifest.get("protocol")) not in versions or manifest.get("robot_definition_sha256") != robot_definition_digest():
        raise ValueError("Manifest does not match the current seven-joint robot/protocol")
    if manifest["protocol"] == "forte-bookshelf-hard-v1" and manifest.get("environment_definition_sha256") != hard_definition_digest():
        raise ValueError("Hard manifest does not match its frozen scene/protocol configuration")
    cases = manifest["cases"]
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate manifest case IDs")
    memberships = {}
    for case in cases:
        fields = {k: v for k, v in case.items() if k != "case_id"}
        if manifest_digest(fields)[:20] != case["case_id"]:
            raise ValueError("Case ID does not match its parameters")
        family_fields = {k: case[k] for k in ("environment", "asset_set", "seed", "layout")}
        if manifest_digest(family_fields)[:20] != case["family_id"]:
            raise ValueError("Family ID does not match its source scene")
        family = case["family_id"]
        previous = memberships.setdefault(family, case["split"])
        if previous != case["split"]:
            raise ValueError("A source family crosses dataset splits")
        heldout = manifest["heldout_combinations"][case["environment"]]
        if case["split"] in ("train", "validation") and case["task_id"] in heldout:
            raise ValueError("Held-out task combination leaked into development")
        if case["split"] == "train" and case["asset_set"] not in ("nominal", "train_small"):
            raise ValueError("Held-out object instance leaked into training")


def verify_metadata(metadata, manifest):
    if metadata.get("manifest_sha256") != manifest_digest(manifest):
        raise ValueError("Dataset manifest hash mismatch")
    case = next((c for c in manifest["cases"] if c["case_id"] == metadata.get("case_id")), None)
    if case is None or any(metadata.get(k) != v for k, v in case.items()):
        raise ValueError("Dataset case does not match the frozen manifest")
    return case
