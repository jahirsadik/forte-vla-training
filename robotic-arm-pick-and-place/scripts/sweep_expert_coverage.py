"""Sweep declared training cases without hiding expert or worker failures.

Each worker starts a fresh MuJoCo process. Coverage does not require RGB;
``--record`` additionally retains native-camera demonstrations, including failures.
Use a new output directory for each immutable implementation and selection.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from utils.contracts import implementation_provenance
from utils.coverage_plan import DEFAULT_PLAN, load_plan, plan_digest, select_cases, scene_key
from utils.research_protocol import MANIFEST, load_manifest, manifest_digest

IMAGE_SIZE = [480, 640]
PEAK_FIELDS = ("peak_contact_force_n", "peak_penetration_m", "peak_torque_fraction",
               "peak_distractor_translation_m", "peak_distractor_rotation_rad")


def choose_cases(plan, manifest, group="all", case_ids=None, collection_subset=None):
    """A rerun may select declared cases but cannot create or relabel one."""
    group_cases = select_cases(plan, manifest, group)
    if case_ids is not None and collection_subset is not None:
        raise ValueError("Choose case IDs or a declared collection subset, not both")
    if collection_subset is not None:
        if collection_subset not in plan["collection_subsets"]:
            raise ValueError("Unknown declared collection subset")
        case_ids = plan["collection_subsets"][collection_subset]
    if case_ids is None:
        return group_cases
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("Case selection must be nonempty and unique")
    allowed = {case["case_id"]: case for case in group_cases}
    if set(case_ids) - set(allowed):
        raise ValueError("Requested case is outside the declared group")
    # Stable plan order makes invocation order immaterial to the experiment.
    return [case for case in group_cases if case["case_id"] in case_ids]


def empty_result(case, output):
    return dict(case_id=case["case_id"], case=deepcopy(case), process_exit_code=None,
                report=str(output / "cases" / (case["case_id"] + ".json")),
                log=str(output / "logs" / (case["case_id"] + ".log")),
                valid_result=False, is_success=False, eligible_success=False,
                fault_exposed=False, errors=[])


def validate_worker_report(report, case, expected_source, expected_manifest, *, record=False,
                           expected_dataset=None):
    """Validate attribution before a worker's success can enter coverage counts."""
    if not isinstance(report, dict) or report.get("requested_case_ids") != [case["case_id"]]:
        raise ValueError("Worker report does not match exactly the requested case")
    if report.get("manifest_sha256") != expected_manifest:
        raise ValueError("Worker manifest hash mismatch")
    if (report.get("policy") != "expert" or report.get("checkpoint") is not None
            or report.get("custom_horizon") is not None or report.get("counterfactual_diagnostic") is not False
            or report.get("instruction_override") is not None):
        raise ValueError("Worker policy, horizon or instruction protocol differs")
    results = report.get("results")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise ValueError("Worker report must contain exactly one result")
    item = results[0]
    identity = ("case_id", "environment", "split", "axis", "task_id", "seed")
    if any(item.get(key) != case[key] for key in identity):
        raise ValueError("Worker result identity differs from the frozen case")
    if item.get("implementation", {}).get("source_sha256") != expected_source:
        raise ValueError("Worker source hash mismatch")
    if (item.get("policy") != "expert" or item.get("custom_horizon") is not False
            or item.get("counterfactual_diagnostic") is not False
            or item.get("execution_horizon_steps") != 1 or item.get("policy_seed") != case["seed"]):
        raise ValueError("Worker control protocol differs")
    constructor = item.get("constructor", {})
    if (constructor.get("asset_set") != case["asset_set"] or "horizon_steps" in constructor
            or constructor.get("cameras") != ["front", "wrist"] or constructor.get("image_size") != IMAGE_SIZE
            or constructor.get("observation_mode") != ("rgb" if record else "proprio")):
        raise ValueError("Worker constructor differs from declared assets/capture")
    if item.get("instruction") != item.get("effective_instruction") or not item.get("instruction"):
        raise ValueError("Worker instruction changed")
    info = item.get("info", {})
    metrics, episode = info.get("metrics", {}), info.get("episode", {})
    if info.get("task_id") != case["task_id"]:
        raise ValueError("Scoring task differs from requested instruction")
    protocol = json.loads((PROJECT / "configs/benchmarks.json").read_text())
    if info.get("protocol_version") != protocol["protocol_version"]:
        raise ValueError("Worker scoring protocol differs")
    steps = episode.get("steps")
    if type(steps) is not int or not 0 <= steps <= protocol["horizon_steps"]:
        raise ValueError("Invalid standard-horizon step count")
    expected_options = dict(task_id=case["task_id"], layout=case["layout"], clutter_level=case["clutter_level"])
    if episode.get("options") != expected_options:
        raise ValueError("Worker reset options differ")
    if type(item.get("is_success")) is not bool or item["is_success"] != info.get("is_success") or item["is_success"] != metrics.get("success"):
        raise ValueError("Inconsistent physical success flags")
    for field in PEAK_FIELDS:
        if not isinstance(episode.get(field), (int, float)) or not math.isfinite(episode[field]):
            raise ValueError("Missing or nonfinite episode metric: " + field)
    if item["is_success"] and not (
            episode.get("done") and episode.get("ever_grasped") and episode.get("ever_lifted")
            and float(episode.get("dwell", 0)) + 1e-9 >= protocol["success_dwell_seconds"]
            and all(metrics.get(key) is True for key in ("within_goal", "supported", "released", "stable", "placement_valid"))
            and not metrics.get("failure_reason") and not metrics.get("timeout") and not metrics.get("dropped_objects")):
        raise ValueError("Success lacks required physical placement checks")
    events = item.get("fault_events")
    if not isinstance(events, list):
        raise ValueError("Missing fault-event log")
    if case["recovery"] == "none":
        if events:
            raise ValueError("Unexpected recovery fault in nominal case")
    elif any(event.get("kind") != case["recovery"] for event in events) or len(events) > 1:
        raise ValueError("Recovery events differ from declared one-shot fault")
    if record:
        validation = item.get("dataset_validation", {})
        dataset = item.get("dataset")
        if (not dataset or Path(dataset).resolve() != expected_dataset.resolve() or not expected_dataset.is_file()
                or validation.get("transitions") != steps
                or validation.get("success") != item["is_success"]
                or any(validation.get("metadata", {}).get(key) != value for key, value in case.items())):
            raise ValueError("Recording lacks matching validated native episode")
        if item["is_success"] and validation.get("complete") is not True:
            raise ValueError("Successful recording is incomplete")
    elif item.get("dataset") is not None:
        raise ValueError("Unrequested recording")
    return item


def run_one(case, output, source_hash, manifest_hash, *, manifest_path=MANIFEST, record=False):
    result = empty_result(case, output)
    expected_dataset = output / "episodes" / (case["case_id"] + ".h5")
    command = [sys.executable, str(PROJECT / "scripts/run_research.py"), "--manifest", str(manifest_path),
               "--env", "sorting", "--split", "train", "--case-id", case["case_id"],
               "--policy", "expert", "--execute-actions", "1", "--output", result["report"]]
    if record:
        command += ["--dataset-dir", str(expected_dataset.parent), "--image-size", *map(str, IMAGE_SIZE)]
    result["command"] = command
    started = time.monotonic()
    try:
        with Path(result["log"]).open("x") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, MUJOCO_GL="egl", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
        result["process_exit_code"] = process.returncode
        if process.returncode != 0:
            result["errors"].append(f"Worker exited with status {process.returncode}")
    except Exception as exc:
        result["errors"].append(f"Worker launch failed: {type(exc).__name__}: {exc}")
    result["elapsed_seconds"] = time.monotonic() - started
    try:
        report = json.loads(Path(result["report"]).read_text())
        item = validate_worker_report(report, case, source_hash, manifest_hash,
                                      record=record, expected_dataset=expected_dataset)
        result.update(is_success=item["is_success"], case_error=item.get("error"),
                      valid_result=result["process_exit_code"] == 0, expert=item.get("expert", {}),
                      metrics=item["info"]["metrics"], episode=item["info"]["episode"],
                      instruction=item["instruction"], fault_events=item["fault_events"],
                      source_sha256=item["implementation"]["source_sha256"],
                      action_contract=item.get("action_contract"), model_fingerprint=item.get("model_fingerprint"),
                      dataset=item.get("dataset"), dataset_validation=item.get("dataset_validation"),
                      prediction_latency=item.get("prediction_latency"))
        result["fault_exposed"] = bool(item["fault_events"])
        result["eligible_success"] = bool(result["valid_result"] and item["is_success"] and not item.get("error")
                                           and (case["recovery"] == "none" or result["fault_exposed"]))
    except Exception as exc:
        result["errors"].append(f"Invalid worker report: {type(exc).__name__}: {exc}")
    return result


def sweep_results(cases, output, source_hash, manifest_hash, *, workers=2, manifest_path=MANIFEST, record=False):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, case, output, source_hash, manifest_hash,
                                   manifest_path=manifest_path, record=record): case for case in cases}
        for future in as_completed(futures):
            try:
                yield future.result()
            except Exception as exc:
                result = empty_result(futures[future], output)
                result["errors"].append(f"Unexpected worker exception: {type(exc).__name__}: {exc}")
                yield result


def aggregate(cases, results):
    by_id = {row["case_id"]: row for row in results}
    if len(by_id) != len(results) or set(by_id) - {case["case_id"] for case in cases}:
        raise ValueError("Duplicate or undeclared results cannot enter a coverage summary")

    def summarize(subset):
        rows = [by_id[case["case_id"]] for case in subset if case["case_id"] in by_id]
        valid = [row for row in rows if row["valid_result"]]
        failures = defaultdict(int)
        for row in rows:
            if row["eligible_success"]:
                continue
            if not row["valid_result"]:
                reason = "worker_or_provenance_error"
            elif row.get("case_error"):
                expert = row.get("expert", {})
                reason = "expert_error/" + str(expert.get("failure_stage") or expert.get("stage", "unknown"))
            elif row.get("metrics", {}).get("failure_reason"):
                reason = "physical_failure/" + str(row["metrics"]["failure_reason"])
            elif row.get("metrics", {}).get("timeout"):
                expert = row.get("expert", {})
                reason = "timeout/" + str(expert.get("failure_stage") or expert.get("stage", "unknown"))
            else:
                reason = "recovery_fault_not_exercised" if row["is_success"] else "unfinished"
            failures[reason] += 1
        peaks = {}
        for field in PEAK_FIELDS:
            values = [row["episode"][field] for row in valid]
            peaks[field] = dict(count=len(values), mean=sum(values) / len(values) if values else None,
                                maximum=max(values) if values else None)
        return dict(requested=len(subset), completed=len(rows), valid_results=len(valid),
                    missing=len(subset) - len(rows), complete=len(rows) == len(subset),
                    reported_successes=sum(row["is_success"] for row in rows),
                    eligible_successes=sum(row["eligible_success"] for row in rows),
                    fault_exposed=sum(row["fault_exposed"] for row in rows),
                    case_errors=sum(bool(row.get("case_error")) for row in rows),
                    worker_errors=sum(not row["valid_result"] for row in rows),
                    ever_grasped=sum(bool(row.get("episode", {}).get("ever_grasped")) for row in valid),
                    ever_lifted=sum(bool(row.get("episode", {}).get("ever_lifted")) for row in valid),
                    final_released=sum(bool(row.get("metrics", {}).get("released")) for row in valid),
                    failure_stages=dict(sorted(failures.items())), peaks=peaks)

    axes = {"object": lambda c: c["task_id"].rsplit("_to_", 1)[0],
            "task": lambda c: c["task_id"], "scene": scene_key,
            "asset_set": lambda c: c["asset_set"], "recovery": lambda c: c["recovery"],
            "layout_seed": lambda c: f"{c['layout']}/seed{c['seed']}"}
    summary = summarize(cases)
    summary["groups"] = {}
    for name, key in axes.items():
        groups = defaultdict(list)
        for case in cases:
            groups[key(case)].append(case)
        summary["groups"][name] = {key: summarize(subset) for key, subset in sorted(groups.items())}
    pairs = defaultdict(list)
    for case in cases:
        obj = case["task_id"].rsplit("_to_", 1)[0]
        pairs[(scene_key(case), case["recovery"], obj)].append(case)
    paired = []
    for (scene, recovery, obj), subset in sorted(pairs.items()):
        destinations = {case["task_id"].rsplit("_to_", 1)[1] for case in subset}
        full_pair = len(subset) == len(destinations) == 2
        paired.append(dict(scene_id=scene, recovery=recovery, object=obj,
                           case_ids=[case["case_id"] for case in subset], full_pair_requested=full_pair,
                           both_successful=full_pair and all(by_id.get(case["case_id"], {}).get("eligible_success", False) for case in subset)))
    summary["counterfactual_pairs"] = dict(requested=sum(pair["full_pair_requested"] for pair in paired),
                                            successful=sum(pair["both_successful"] for pair in paired),
                                            unpaired_groups=sum(not pair["full_pair_requested"] for pair in paired), details=paired)
    summary["pair_scope"] = "Same initial scene and object, two different instructed destinations; reset equality requires the separate scene audit."
    return summary


def save_report(output, report):
    temporary = output / "coverage.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "coverage.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--group", choices=("nominal", "recovery", "all"), default="all")
    parser.add_argument("--case-id", action="append", help="Explicit declared subset; repeat for multiple cases")
    parser.add_argument("--collection-subset", help="Named subset already frozen in the plan")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--record", action="store_true", help="Record native 480x640 front/wrist RGB, including failures")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    manifest = load_manifest(args.manifest)
    plan = load_plan(args.plan, manifest)
    try:
        cases = choose_cases(plan, manifest, args.group, args.case_id, args.collection_subset)
    except ValueError as exc:
        parser.error(str(exc))
    if args.list:
        print(json.dumps(cases, indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --list is used")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("Output directory exists; preserve old evidence and choose a new path")
    output.mkdir(parents=True)
    for folder in ("cases", "logs") + (("episodes",) if args.record else ()):
        (output / folder).mkdir()
    (output / "input_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    source = implementation_provenance()
    group_ids = [case["case_id"] for case in select_cases(plan, manifest, args.group)]
    requested = [case["case_id"] for case in cases]
    report = dict(version="forte-expert-coverage-v1", scope="Training-only expert feasibility, not learned-policy or held-out performance",
                  plan_sha256=plan_digest(plan), plan_file_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                  manifest_sha256=manifest_digest(manifest), implementation=source,
                  collector_project=str(PROJECT), group=args.group, declared_group_case_ids=group_ids,
                  requested_case_ids=requested, subset=requested != group_ids, collection_subset=args.collection_subset,
                  record=args.record, image_size_hw=IMAGE_SIZE, standard_horizon=True,
                  results=[], summary=aggregate(cases, []), complete=False, full_group_complete=False)
    save_report(output, report)
    for result in sweep_results(cases, output, source["source_sha256"], manifest_digest(manifest),
                                workers=args.workers, manifest_path=output / "input_manifest.json", record=args.record):
        report["results"].append(result)
        report["results"].sort(key=lambda row: requested.index(row["case_id"]))
        report["summary"] = aggregate(cases, report["results"])
        report["complete"] = report["summary"]["complete"]
        report["full_group_complete"] = report["complete"] and not report["subset"]
        save_report(output, report)
        print(json.dumps({key: result[key] for key in ("case_id", "is_success", "eligible_success", "valid_result", "errors")}), flush=True)
    report["source_unchanged"] = implementation_provenance()["source_sha256"] == source["source_sha256"]
    report["valid"] = report["complete"] and report["source_unchanged"] and report["summary"]["worker_errors"] == 0
    save_report(output, report)
    print(json.dumps({key: report[key] for key in ("complete", "full_group_complete", "source_unchanged", "valid")}), flush=True)
    # Expert failures are experimental outcomes; invalid execution/provenance is a failed sweep.
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
