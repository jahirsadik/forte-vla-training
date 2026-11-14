"""Check the complete declared nominal sorting collection, including failures."""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from utils.coverage_plan import DEFAULT_PLAN, load_plan
from utils.coverage_recordings import audit_recordings
from utils.research_protocol import load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; preserve the earlier audit")
    manifest = load_manifest()
    plan = load_plan(args.plan, manifest)
    report = audit_recordings(sorted(args.episodes.rglob("*.h5")), plan, manifest,
                              source_sha256=args.source_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("ready_for_balanced_nominal_fit", "expected_episodes",
         "eligible_episodes", "complete_destination_pairs", "required_destination_pairs", "missing_case_ids", "errors")}))
    return 0 if report["ready_for_balanced_nominal_fit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
