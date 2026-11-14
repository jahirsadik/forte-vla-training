"""Coverage attribution, missing-case accounting and counterfactual integrity."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import sweep_expert_coverage as runner
from utils.coverage_plan import load_plan, select_cases
from utils.research_protocol import load_manifest, manifest_digest


class ExpertCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest()
        cls.plan = load_plan(manifest=cls.manifest)
        cls.manifest_hash = manifest_digest(cls.manifest)
        cls.cases = select_cases(cls.plan, cls.manifest, "nominal")

    def make_report(self, case=None, *, success=True, error=None, record=False):
        case = self.cases[0] if case is None else case
        episode = dict(steps=12, done=success, ever_grasped=True, ever_lifted=True, dwell=.5 if success else 0.,
                       options=dict(task_id=case["task_id"], layout=case["layout"], clutter_level=case["clutter_level"]),
                       **{key: .1 for key in runner.PEAK_FIELDS})
        metrics = dict(success=success, within_goal=success, supported=success, released=success,
                       stable=success, placement_valid=success, failure_reason=None, timeout=False, dropped_objects=[])
        item = {key: case[key] for key in ("case_id", "environment", "split", "axis", "task_id", "seed")}
        item.update(implementation=dict(source_sha256="expected"), policy="expert", custom_horizon=False,
                    counterfactual_diagnostic=False, execution_horizon_steps=1, policy_seed=case["seed"],
                    constructor=dict(asset_set=case["asset_set"], cameras=["front", "wrist"], image_size=[480, 640],
                                     observation_mode="rgb" if record else "proprio"),
                    instruction="Place this object in that tray.", effective_instruction="Place this object in that tray.",
                    is_success=success, info=dict(task_id=case["task_id"], protocol_version="forte-manipulation-v3",
                                                 is_success=success, metrics=metrics, episode=episode),
                    fault_events=[], expert=dict(stage="complete" if success else "grasp"))
        if error:
            item["error"] = error
        return dict(requested_case_ids=[case["case_id"]], manifest_sha256=self.manifest_hash,
                    policy="expert", checkpoint=None, custom_horizon=None, counterfactual_diagnostic=False,
                    instruction_override=None, results=[item])

    def worker(self, report=None, *, case=None, returncode=0, record=False, dataset=True):
        case = self.cases[0] if case is None else case
        report = self.make_report(case, record=record) if report is None else deepcopy(report)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for folder in ("cases", "logs", "episodes"):
                (output / folder).mkdir()
            if record:
                path = output / "episodes" / (case["case_id"] + ".h5")
                if dataset:
                    path.touch()
                report["results"][0].update(dataset=str(path), dataset_validation=dict(
                    complete=True, success=report["results"][0]["is_success"], transitions=12, metadata=case))
            (output / "cases" / (case["case_id"] + ".json")).write_text(report if isinstance(report, str) else json.dumps(report))
            with patch.object(runner.subprocess, "run", return_value=types.SimpleNamespace(returncode=returncode)):
                return runner.run_one(case, output, "expected", self.manifest_hash, record=record)

    def test_clean_worker_is_physically_valid_and_recording_is_explicit(self):
        result = self.worker()
        self.assertTrue(result["valid_result"])
        self.assertTrue(result["eligible_success"])
        self.assertIsNone(result["dataset"])
        self.assertNotIn("--horizon", result["command"])
        recorded = self.worker(record=True)
        self.assertTrue(recorded["eligible_success"])
        self.assertIn("--dataset-dir", recorded["command"])
        self.assertFalse(self.worker(record=True, dataset=False)["eligible_success"])

    def test_wrong_identity_source_manifest_and_control_are_not_coverage(self):
        for mutate in (
            lambda r: r["results"][0].update(case_id="wrong"),
            lambda r: r["results"][0]["implementation"].update(source_sha256="changed"),
            lambda r: r.update(manifest_sha256="wrong"),
            lambda r: r["results"][0].update(execution_horizon_steps=8),
            lambda r: r["results"][0]["constructor"].update(horizon_steps=9000),
            lambda r: r["results"][0]["info"]["episode"]["options"].update(layout="arc"),
            lambda r: r["results"][0].update(effective_instruction="changed instruction"),
        ):
            report = self.make_report()
            mutate(report)
            result = self.worker(report)
            self.assertFalse(result["eligible_success"])
            self.assertFalse(result["valid_result"])
            self.assertIn("Invalid worker report", result["errors"][-1])

    def test_physical_success_flags_and_nonfinite_metrics_cannot_inflate_counts(self):
        for mutate in (
            lambda i: i["info"]["metrics"].update(released=False),
            lambda i: i["info"]["episode"].update(ever_lifted=False),
            lambda i: i["info"]["episode"].update(dwell=.1),
            lambda i: i["info"]["episode"].update(peak_contact_force_n=float("nan")),
            lambda i: i["info"].update(is_success=False),
        ):
            report = self.make_report()
            mutate(report["results"][0])
            self.assertFalse(self.worker(report)["valid_result"])

    def test_crashed_success_is_retained_but_not_eligible(self):
        result = self.worker(returncode=143)
        self.assertTrue(result["is_success"])
        self.assertFalse(result["valid_result"])
        self.assertFalse(result["eligible_success"])
        self.assertIn("143", result["errors"][0])

    def test_controller_failure_is_a_valid_failed_trial_not_a_missing_worker(self):
        result = self.worker(self.make_report(success=False, error="RuntimeError: grasp route infeasible"))
        summary = runner.aggregate(self.cases[:1], [result])
        self.assertTrue(result["valid_result"])
        self.assertFalse(result["eligible_success"])
        self.assertEqual(summary["worker_errors"], 0)
        self.assertEqual(summary["case_errors"], 1)
        self.assertEqual(summary["failure_stages"], {"expert_error/grasp": 1})
        result["expert"].update(stage="failed", failure_stage="transfer")
        self.assertEqual(runner.aggregate(self.cases[:1], [result])["failure_stages"], {"expert_error/transfer": 1})

    def test_malformed_reports_and_launch_exceptions_retain_requested_identity(self):
        for malformed in ("{", "[]", '{"results":[]}'):
            result = self.worker(malformed)
            self.assertEqual(result["case_id"], self.cases[0]["case_id"])
            self.assertFalse(result["valid_result"])
            self.assertTrue(result["errors"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "logs").mkdir()
            with patch.object(runner.subprocess, "run", side_effect=OSError("cannot launch")):
                result = runner.run_one(self.cases[0], output, "expected", self.manifest_hash)
            self.assertEqual(result["case_id"], self.cases[0]["case_id"])
            self.assertIsNone(result["process_exit_code"])
            self.assertIn("cannot launch", result["errors"][0])

    def test_recovery_success_requires_actual_declared_fault_exposure(self):
        case = select_cases(self.plan, self.manifest, "recovery")[0]
        report = self.make_report(case)
        no_fault = self.worker(report, case=case)
        self.assertTrue(no_fault["is_success"])
        self.assertFalse(no_fault["eligible_success"])
        report["results"][0]["fault_events"] = [dict(kind=case["recovery"], step=4, duration_steps=30)]
        self.assertTrue(self.worker(report, case=case)["eligible_success"])
        report["results"][0]["fault_events"][0]["kind"] = "unexpected"
        self.assertFalse(self.worker(report, case=case)["valid_result"])

    def test_incomplete_counterfactual_pairs_keep_requested_denominator(self):
        cases = self.cases[:2]
        first = self.worker(case=cases[0])
        summary = runner.aggregate(cases, [first])
        self.assertFalse(summary["complete"])
        self.assertEqual((summary["requested"], summary["completed"], summary["missing"]), (2, 1, 1))
        self.assertEqual(summary["counterfactual_pairs"]["requested"], 1)
        self.assertEqual(summary["counterfactual_pairs"]["successful"], 0)
        second = self.worker(case=cases[1])
        finished = runner.aggregate(cases, [first, second])
        self.assertTrue(finished["complete"])
        self.assertEqual(finished["counterfactual_pairs"]["successful"], 1)
        one = runner.aggregate(cases[:1], [first])
        self.assertEqual(one["counterfactual_pairs"]["requested"], 0)
        self.assertEqual(one["counterfactual_pairs"]["unpaired_groups"], 1)
        for invalid in ([first, first], [dict(first, case_id="undeclared")]):
            with self.assertRaises(ValueError):
                runner.aggregate(cases, invalid)

    def test_explicit_subsets_cannot_expand_or_relabel_the_frozen_group(self):
        original = deepcopy(self.plan)
        chosen = runner.choose_cases(self.plan, self.manifest, "nominal", [self.cases[1]["case_id"], self.cases[0]["case_id"]])
        self.assertEqual(chosen, self.cases[:2])
        self.assertEqual(self.plan, original)
        self.assertEqual(len(runner.choose_cases(self.plan, self.manifest, "all", collection_subset="nominal_seed0")), 24)
        for ids in ([], [self.cases[0]["case_id"]]*2, ["not_a_case"]):
            with self.assertRaises(ValueError):
                runner.choose_cases(self.plan, self.manifest, "nominal", ids)
        with self.assertRaises(ValueError):
            runner.choose_cases(self.plan, self.manifest, "nominal", collection_subset="recovery_seed0")

    def test_unexpected_worker_failure_does_not_drop_other_requested_results(self):
        def work(case, output, *args, **kwargs):
            if case["case_id"] == self.cases[0]["case_id"]:
                raise RuntimeError("worker bug")
            return runner.empty_result(case, output)
        with patch.object(runner, "run_one", side_effect=work):
            results = list(runner.sweep_results(self.cases[:2], Path("unused"), "source", "manifest", workers=2))
        self.assertEqual({row["case_id"] for row in results}, {case["case_id"] for case in self.cases[:2]})
        self.assertTrue(runner.aggregate(self.cases[:2], results)["complete"])
        self.assertEqual(runner.aggregate(self.cases[:2], results)["worker_errors"], 2)


if __name__ == "__main__":
    unittest.main()
