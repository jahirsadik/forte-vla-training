"""Check episode alignment, optionally replay dynamics or compute train statistics."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.demonstrations import validate_episode, replay_episode, normalization_statistics
from utils.research_protocol import load_manifest, verify_metadata, manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--statistics", type=Path)
    args = parser.parse_args()
    for path in args.episodes:
        result = (replay_episode if args.replay else validate_episode)(path)
        verify_metadata(result["metadata"], load_manifest(manifest_path(result["metadata"]["environment"])))
        print(json.dumps(dict(path=str(path), **result)))
    if args.statistics:
        result = normalization_statistics(args.episodes)
        args.statistics.write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
