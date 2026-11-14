"""Collection failures remain attributable and cannot count as eligible data."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import collect_policy_data as collector
from utils.research_protocol import load_manifest


class PolicyCollectionTests(unittest.TestCase):
    def setUp(self):
        self.case = dict(case_id="first", environment="sorting", split="train", task_id="red_cube_to_blue",
                         seed=0, axis="development", asset_set="nominal", layout="shuffled")

    def worker(self, *, returncode=0, source="expected", error=None, malformed=None, mismatch=False, dataset=True):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            for directory in ("cases", "logs", "episodes/train"):
                (output / directory).mkdir(parents=True)
            episode = output / "episodes/train/first.h5"
            if dataset:
                episode.touch()
            item = dict(self.case, is_success=error is None, implementation={"source_sha256": source},
                        error=error, dataset=str(episode),
                        dataset_validation=dict(complete=True, success=error is None, transitions=12, metadata=self.case))
            if mismatch:
                item["case_id"] = "wrong"
            report = dict(requested_case_ids=[self.case["case_id"]], results=[item])
            text = json.dumps(report) if malformed is None else malformed
            (output / "cases/first.json").write_text(text)
            with patch.object(collector.subprocess, "run", return_value=types.SimpleNamespace(returncode=returncode)):
                return collector.collect_one(self.case, output, [480, 640], "expected")

    def test_clean_success_requires_matching_validated_episode(self):
        result = self.worker()
        self.assertTrue(result["is_success"])
        self.assertTrue(result["eligible_success"])
        self.assertEqual(result["transitions"], 12)
        self.assertEqual(collector.collection_summary([result], ["first"]),
                         dict(complete=True, successful_episodes=1, eligible_successes=1, worker_errors=0, case_errors=0))
        missing = self.worker(dataset=False)
        self.assertTrue(missing["is_success"])
        self.assertFalse(missing["eligible_success"])
        self.assertIn("lacks a matching", missing["worker_error"])

    def test_success_from_crashed_or_changed_worker_is_preserved_but_ineligible(self):
        for arguments in (dict(returncode=143), dict(source="changed")):
            with self.subTest(arguments=arguments):
                result = self.worker(**arguments)
                self.assertTrue(result["is_success"])
                self.assertFalse(result["eligible_success"])
                summary = collector.collection_summary([result], ["first"])
                self.assertEqual(summary["successful_episodes"], 1)
                self.assertEqual(summary["eligible_successes"], 0)
                self.assertEqual(summary["worker_errors"], 1)

    def test_caught_case_error_is_separate_from_process_failure(self):
        result = self.worker(error="RuntimeError: planning failed")
        self.assertFalse(result["eligible_success"])
        summary = collector.collection_summary([result], ["first"])
        self.assertEqual(summary["worker_errors"], 0)
        self.assertEqual(summary["case_errors"], 1)

    def test_malformed_empty_and_wrong_case_reports_become_structured_failures(self):
        for arguments in (dict(malformed="{"), dict(malformed='{"results": []}'),
                          dict(malformed="[]"), dict(mismatch=True)):
            with self.subTest(arguments=arguments):
                result = self.worker(**arguments)
                self.assertEqual(result["case_id"], "first")
                self.assertFalse(result["eligible_success"])
                self.assertIn("Invalid worker report", result["worker_error"])

    def test_launch_and_missing_report_failures_retain_case_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "logs").mkdir()
            with patch.object(collector.subprocess, "run", side_effect=OSError("launch unavailable")):
                result = collector.collect_one(self.case, output, [480, 640], "expected")
            self.assertIsNone(result["exit_code"])
            self.assertEqual(result["case_id"], "first")
            self.assertIn("launch unavailable", result["worker_error"])
            with patch.object(collector.subprocess, "run", return_value=types.SimpleNamespace(returncode=1)):
                result = collector.collect_one(self.case, output, [480, 640], "expected")
            self.assertEqual(result["exit_code"], 1)
            self.assertIn("no case report", result["worker_error"])

    def test_unexpected_worker_exception_does_not_lose_other_results(self):
        second = dict(self.case, case_id="second")

        def run(case, output, *args):
            if case["case_id"] == "first":
                raise RuntimeError("unexpected parser bug")
            result = collector.empty_result(case, output)
            result["exit_code"] = 0
            return result

        with patch.object(collector, "collect_one", side_effect=run):
            results = list(collector.collect_results([self.case, second], Path("unused"), [480, 640], "expected", 2))
        self.assertEqual({r["case_id"] for r in results}, {"first", "second"})
        summary = collector.collection_summary(results, ["first", "second"])
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["worker_errors"], 1)
        self.assertEqual(summary["eligible_successes"], 0)
        self.assertFalse(collector.collection_summary(results[:1], ["first", "second"])["complete"])

    def test_plan_uses_existing_splits_and_rejects_duplicate_or_test_cases(self):
        manifest = load_manifest()
        original = deepcopy(manifest)
        plan = dict(tasks={"sorting": ["red_cube_to_blue"]},
                    splits={"train": dict(seeds=[0], asset_set="nominal"),
                            "validation": dict(seeds=[1000], asset_set="validation")},
                    axis="development", layout="shuffled")
        cases = collector.select_cases(plan, manifest)
        self.assertEqual({c["split"] for c in cases}, {"train", "validation"})
        self.assertEqual(manifest, original)
        duplicate = deepcopy(plan)
        duplicate["tasks"]["sorting"].append("red_cube_to_blue")
        with self.assertRaisesRegex(ValueError, "unique"):
            collector.select_cases(duplicate, manifest)
        invalid = deepcopy(plan)
        invalid["splits"]["test"] = invalid["splits"].pop("validation")
        with self.assertRaisesRegex(ValueError, "train/validation"):
            collector.select_cases(invalid, manifest)


if __name__ == "__main__":
    unittest.main()
