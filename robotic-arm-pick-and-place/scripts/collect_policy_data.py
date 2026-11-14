"""Collect a declared train/validation pilot using isolated native-RGB workers.

The plan selects existing frozen cases; it never creates or relabels a split.
Each process owns its MuJoCo/EGL context. Failures and exit codes stay in reports.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from utils.contracts import implementation_provenance
from utils.research_protocol import load_manifest, manifest_digest


def select_cases(plan, manifest):
    if set(plan["splits"]) - {"train", "validation"}:
        raise ValueError("Pilot collection is restricted to declared train/validation splits")
    cases = []
    for environment, tasks in plan["tasks"].items():
        for split, settings in plan["splits"].items():
            for task in tasks:
                for seed in settings["seeds"]:
                    matches = [c for c in manifest["cases"] if
                               c["environment"] == environment and c["task_id"] == task and
                               c["split"] == split and c["seed"] == seed and
                               c["asset_set"] == settings["asset_set"] and
                               c["axis"] == plan["axis"] and c["layout"] == plan["layout"]]
                    if len(matches) != 1:
                        raise ValueError(f"Expected exactly one frozen case: {environment}/{task}/{split}/{seed}")
                    cases.append(matches[0])
    if len({c["case_id"] for c in cases}) != len(cases) or not cases:
        raise ValueError("Pilot cases must be nonempty and unique")
    return cases


def empty_result(case, output):
    return {"case_id": case["case_id"], "split": case["split"], "task_id": case["task_id"],
            "exit_code": None, "report": str(output / "cases" / f"{case['case_id']}.json"),
            "log": str(output / "logs" / f"{case['case_id']}.log"),
            "is_success": False, "eligible_success": False}


def collect_one(case, output, image_size, source_hash):
    result = empty_result(case, output)
    result_path = output / "cases" / f"{case['case_id']}.json"
    log_path = output / "logs" / f"{case['case_id']}.log"
    expected_dataset = output / "episodes" / case["split"] / f"{case['case_id']}.h5"
    command = [sys.executable, str(PROJECT / "scripts/run_research.py"),
               "--env", case["environment"], "--split", case["split"],
               "--case-id", case["case_id"], "--dataset-dir", str(output / "episodes" / case["split"]),
               "--image-size", *map(str, image_size), "--output", str(result_path)]
    try:
        with log_path.open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, MUJOCO_GL="egl", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
        result["exit_code"] = process.returncode
    except Exception as exc:
        result["worker_error"] = f"Worker launch failed: {type(exc).__name__}: {exc}"
        result["error"] = result["worker_error"]
        return result
    try:
        if not result_path.exists():
            raise ValueError("Worker produced no case report")
        report = json.loads(result_path.read_text())
        if (report.get("requested_case_ids") != [case["case_id"]]
                or len(report.get("results", [])) != 1):
            raise ValueError("Worker report does not contain exactly the requested case")
        item = report["results"][0]
        if any(item.get(key) != case[key] for key in ("case_id", "environment", "split", "task_id", "seed")):
            raise ValueError("Worker result identity differs from the requested frozen case")
        validation = item.get("dataset_validation", {})
        result.update(is_success=bool(item["is_success"]), error=item.get("error"), case_error=item.get("error"),
                      dataset=item.get("dataset"), transitions=validation.get("transitions"),
                      source_sha256=item.get("implementation", {}).get("source_sha256"))
        if result["source_sha256"] != source_hash:
            result["source_mismatch"] = True
        valid_dataset = (result["dataset"] is not None
                         and Path(result["dataset"]).resolve() == expected_dataset.resolve()
                         and expected_dataset.is_file()
                         and validation.get("complete") is True
                         and validation.get("success") is True
                         and isinstance(validation.get("transitions"), int)
                         and validation["transitions"] > 0
                         and all(validation.get("metadata", {}).get(key) == value for key, value in case.items()))
        # A saved successful episode from a crashed process is retained as
        # evidence, but it does not count as a cleanly collected training case.
        result["eligible_success"] = bool(result["is_success"] and valid_dataset
                                           and result["exit_code"] == 0
                                           and not result.get("source_mismatch") and not result.get("case_error"))
        if result["is_success"] and not valid_dataset:
            result["worker_error"] = "Successful result lacks a matching complete validated episode"
            result["error"] = result.get("error") or result["worker_error"]
    except Exception as exc:
        result["worker_error"] = f"Invalid worker report: {type(exc).__name__}: {exc}"
        result["error"] = result["worker_error"]
    return result


def collect_results(cases, output, image_size, source_hash, workers):
    """Retain a result for every requested case, including unexpected failures."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(collect_one, c, output, image_size, source_hash): c for c in cases}
        for future in as_completed(futures):
            try:
                item = future.result()
            except Exception as exc:
                item = empty_result(futures[future], output)
                item["worker_error"] = f"Collector worker failed: {type(exc).__name__}: {exc}"
                item["error"] = item["worker_error"]
            yield item


def collection_summary(results, requested_case_ids):
    return dict(
        complete=len(results) == len(requested_case_ids) and {r["case_id"] for r in results} == set(requested_case_ids),
        successful_episodes=sum(r["is_success"] for r in results),
        eligible_successes=sum(r.get("eligible_success", False) for r in results),
        worker_errors=sum(r["exit_code"] != 0 or bool(r.get("source_mismatch")) or bool(r.get("worker_error")) for r in results),
        case_errors=sum(bool(r.get("case_error")) for r in results),
    )


def save_report(output, report):
    temporary = output / "collection.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output / "collection.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PROJECT / "configs/first_policy_pilot.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    plan = json.loads(args.plan.read_text())
    manifest = load_manifest()
    cases = select_cases(plan, manifest)
    if args.list:
        print(json.dumps(cases, indent=2))
        return 0
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("Output directory exists; use a new path to preserve previous data")
    for folder in ("cases", "logs", "episodes/train", "episodes/validation"):
        (output / folder).mkdir(parents=True)
    source = implementation_provenance()
    report = {"plan": plan, "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
              "manifest_sha256": manifest_digest(manifest), "implementation": source,
              "collector_project": str(PROJECT), "requested_case_ids": [c["case_id"] for c in cases], "results": []}
    report.update(collection_summary(report["results"], report["requested_case_ids"]))
    save_report(output, report)
    for item in collect_results(cases, output, plan["image_size_hw"], source["source_sha256"], args.workers):
        report["results"].append(item)
        report["results"].sort(key=lambda x: report["requested_case_ids"].index(x["case_id"]))
        report.update(collection_summary(report["results"], report["requested_case_ids"]))
        save_report(output, report)
        print(json.dumps(item), flush=True)
    summary = collection_summary(report["results"], report["requested_case_ids"])
    print(json.dumps(summary), flush=True)
    return 1 if not summary["complete"] or summary["worker_errors"] or summary["case_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
