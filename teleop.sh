#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$repo_dir/.conda-env/bin/python" "$repo_dir/robotic-arm-pick-and-place/scripts/run_teleop.py" "$@"
