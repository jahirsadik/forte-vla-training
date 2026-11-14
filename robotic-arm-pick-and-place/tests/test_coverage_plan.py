"""Frozen development coverage cannot be narrowed by outcomes or leak splits."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.coverage_plan import (build_plan, load_plan, pair_key, plan_digest, scene_key,
                                 select_cases, select_collection_cases, validate_plan)
from utils.research_protocol import load_manifest, manifest_digest


class CoveragePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest()
        cls.plan = load_plan(manifest=cls.manifest)

    def test_complete_coverage_uses_only_existing_training_cases(self):
        cases = select_cases(self.plan, self.manifest)
        self.assertEqual(len(cases), 96)
        self.assertEqual(len({case["case_id"] for case in cases}), 96)
        self.assertEqual({case["split"] for case in cases}, {"train"})
        self.assertEqual({case["environment"] for case in cases}, {"sorting"})
        self.assertFalse({case["task_id"] for case in cases} & set(self.manifest["heldout_combinations"]["sorting"]))
        self.assertEqual(Counter(case["recovery"] for case in cases), {"none": 72, "missed_grasp": 12, "delayed_release": 12})
        self.assertTrue(all(case in self.manifest["cases"] for case in cases))
        self.assertEqual(self.plan, build_plan(self.manifest))

    def test_scene_groups_and_all_destination_pairs_are_complete(self):
        cases = select_cases(self.plan, self.manifest, "nominal")
        counts = Counter(pair_key(case) for case in cases)
        self.assertEqual(len(counts), 36)
        self.assertEqual(set(counts.values()), {2})
        self.assertEqual(len(self.plan["scene_groups"]), 6)
        for group in self.plan["scene_groups"]:
            members = [case for case in cases if scene_key(case) == group["scene_id"]]
            self.assertEqual(len(members), 12)
            self.assertEqual(group["nominal_case_ids"], [case["case_id"] for case in members])
            self.assertEqual(set(group["task_ids"]), {case["task_id"] for case in members})
            self.assertEqual(set(map(len, group["destination_pairs"].values())), {2})

    def test_collection_subsets_are_fixed_balanced_and_detached(self):
        for name in ("nominal_seed0", "recovery_seed0"):
            cases = select_collection_cases(self.plan, self.manifest, name)
            self.assertEqual(len(cases), 24)
            self.assertEqual({case["seed"] for case in cases}, {0})
            self.assertEqual(set(Counter(pair_key(case) for case in cases).values()), {2})
            before = deepcopy(self.manifest)
            cases[0]["split"] = "test"
            self.assertEqual(self.manifest, before)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            select_collection_cases(self.plan, self.manifest, "successful_only")

    def test_missing_duplicate_reordered_and_cross_split_cases_are_rejected(self):
        for kind in ("missing", "duplicate", "reorder", "validation", "test", "heldout"):
            plan = deepcopy(self.plan)
            ids = plan["groups"]["nominal"]
            if kind == "missing":
                ids.pop()
            elif kind == "duplicate":
                ids[-1] = ids[0]
            elif kind == "reorder":
                ids[0], ids[1] = ids[1], ids[0]
            else:
                candidate = next(case for case in self.manifest["cases"] if
                                 case["environment"] == "sorting" and
                                 (case["split"] == kind if kind != "heldout" else case["axis"] == "combinations"))
                ids[-1] = candidate["case_id"]
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "frozen train-only"):
                validate_plan(plan, self.manifest)

    def test_scene_pair_or_collection_mutations_cannot_silently_reduce_scope(self):
        for change in (lambda p: p["scene_groups"][0]["nominal_case_ids"].pop(),
                       lambda p: p["scene_groups"][0]["destination_pairs"]["red_cube"].pop(),
                       lambda p: p["scene_groups"][0]["reset"].update(clutter_level=2),
                       lambda p: p["scene_groups"][0]["reset"].update(clutter_level=True),
                       lambda p: p["collection"].update(execution_horizon_steps=1.0),
                       lambda p: p["collection_subsets"]["nominal_seed0"].pop(),
                       lambda p: p["collection"].update(image_size_hw=[240, 320]),
                       lambda p: p.update(unexpected=True)):
            plan = deepcopy(self.plan)
            change(plan)
            with self.assertRaisesRegex(ValueError, "frozen train-only"):
                validate_plan(plan, self.manifest)

    def test_versions_stale_hashes_and_mutated_manifest_are_rejected(self):
        plan = deepcopy(self.plan)
        plan["version"] = "v0"
        with self.assertRaisesRegex(ValueError, "version"):
            validate_plan(plan, self.manifest)
        plan = deepcopy(self.plan)
        plan["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_plan(plan, self.manifest)
        manifest = deepcopy(self.manifest)
        manifest["cases"][0]["seed"] = 7
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_plan(self.plan, manifest)
        plan["manifest_sha256"] = manifest_digest(manifest)
        with self.assertRaisesRegex(ValueError, "Case ID"):
            validate_plan(plan, manifest)

    def test_full_scene_identity_changes_with_every_reset_field(self):
        case = select_cases(self.plan, self.manifest, "nominal")[0]
        self.assertEqual(scene_key(case), scene_key(dict(case, task_id="red_cube_to_blue", recovery="missed_grasp")))
        for field, value in dict(environment="bookshelf", asset_set="train_small", seed=9,
                                 layout="arc", appearance="lighting_dim", appearance_seed=7,
                                 clutter_level=2).items():
            with self.subTest(field=field):
                self.assertNotEqual(scene_key(case), scene_key(dict(case, **{field: value})))

    def test_selection_and_digest_do_not_mutate_inputs(self):
        before = deepcopy(self.plan), deepcopy(self.manifest)
        cases = select_cases(self.plan, self.manifest, "recovery")
        self.assertEqual(len(cases), 24)
        cases[0]["task_id"] = "changed"
        self.assertEqual((self.plan, self.manifest), before)
        reversed_keys = dict(reversed(list(self.plan.items())))
        self.assertEqual(plan_digest(self.plan), plan_digest(reversed_keys))
        with self.assertRaisesRegex(ValueError, "group"):
            select_cases(self.plan, self.manifest, "test")

    def test_serialized_plan_roundtrip_and_missing_case_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plan.json"
            path.write_text(json.dumps(self.plan))
            self.assertEqual(load_plan(path, self.manifest), self.plan)
        manifest = deepcopy(self.manifest)
        missing = self.plan["groups"]["nominal"][0]
        manifest["cases"] = [case for case in manifest["cases"] if case["case_id"] != missing]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_plan(manifest)


if __name__ == "__main__":
    unittest.main()
