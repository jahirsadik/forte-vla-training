"""Build the independently configured dense bookshelf benchmark."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def build(config_path=PROJECT / "configs/bookshelf_hard.json", destination=PROJECT / "assets/bookshelf_hard"):
    from environments.bookshelf_hard import packed_positions
    from tools.build_bookshelf_scene import build as build_bookshelf

    config = json.loads(Path(config_path).read_text())
    names = [b["name"] for b in config["books"] if "table_xy_m" not in b]
    positions = packed_positions(config, names, config["hard_layout"]["default"])
    for book in config["books"]:
        if book["name"] in positions:
            book["shelf_offset_y_m"] = positions[book["name"]][1] - config["shelf"]["center_xy_m"][1]
    # The common generator owns visual/collision geometry. Runtime reset supplies
    # shuffle/protrusion variations using the same thickness-based packing rule.
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "configuration.json"
        path.write_text(json.dumps(config))
        return build_bookshelf(path, Path(destination))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/bookshelf_hard.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "assets/bookshelf_hard")
    args = parser.parse_args()
    print(build(args.config, args.output_dir))
