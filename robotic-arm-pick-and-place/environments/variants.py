"""Reproducible, disjoint object-instance catalogs compiled from original assets."""

from copy import deepcopy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ASSET_SETS = {
    "nominal": (1., 1., 1.),
    "train_small": (.94, .92, .94),
    "validation": (1.03, .95, 1.04),
    "heldout": (.90, 1.12, 1.08),
}


@contextmanager
def _cache_lock(path):
    """OS-owned lock, released even when a worker exits unexpectedly."""
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0, 2)
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def scene_variant(kind, asset_set):
    if asset_set not in ASSET_SETS:
        raise ValueError(f"Unknown asset set {asset_set!r}")
    original = PROJECT / "assets" / kind
    cfg = deepcopy(json.loads((original / "manifest.json").read_text())["configuration"])
    if asset_set == "nominal":
        return original / "scene.xml", cfg
    scale = ASSET_SETS[asset_set]
    if kind in ("bookshelf", "bookshelf_hard"):
        for book in cfg["books"]:
            for key, factor in zip(("depth_m", "thickness_m", "height_m"), scale):
                book[key] *= factor
            book["mass_kg"] *= scale[0] * scale[1] * scale[2]
            book["title"] = f"{book.get('title', book['name'])} {asset_set[:1].upper()}"
    else:
        for obj in cfg["objects"]:
            factors = list(scale)
            if obj["shape"] in ("cylinder", "capsule"):
                factors[1] = factors[0]
            obj["size_m"] = [v * s for v, s in zip(obj["size_m"], factors)]
            obj["mass_kg"] *= factors[0] * factors[1] * factors[2]
    # Include generator content so edited construction code never reuses stale XML.
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode())
    for file in ("build_bookshelf_scene.py", "build_sorting_scene.py", "scene_appearance.py", "build_bookshelf_hard_scene.py"):
        digest.update((PROJECT / "tools" / file).read_bytes())
    digest.update((PROJECT / "configs/benchmarks.json").read_bytes())
    if kind == "bookshelf_hard":
        digest.update((PROJECT / "environments/bookshelf_hard.py").read_bytes())
        digest.update((PROJECT / "configs/benchmarks_hard.json").read_bytes())
    digest.update((PROJECT / "configs/forte.json").read_bytes())
    digest.update((PROJECT / "assets/forte_arm/source_7dof/manifest.json").read_bytes())
    digest.update((PROJECT / "assets/forte_arm/forte.xml").read_bytes())
    directory = PROJECT / "outputs/scene_cache" / f"{kind}_{asset_set}_{digest.hexdigest()[:16]}"
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Multiple evaluation workers can request the same cold cache. The XML may
    # exist while another worker is still writing PNGs, so it is not a commit
    # marker. Lock the build and publish readiness only after every asset exists.
    with _cache_lock(directory.with_suffix(".lock")):
        if not (directory / ".ready").exists():
            directory.mkdir(parents=True, exist_ok=True)
            config_path = directory / "configuration.json"
            config_path.write_text(json.dumps(cfg, indent=2) + "\n")
            if kind == "bookshelf_hard":
                from tools.build_bookshelf_hard_scene import build
            elif kind == "bookshelf":
                from tools.build_bookshelf_scene import build
            else:
                from tools.build_sorting_scene import build
            build(config_path=config_path, destination=directory)
            (directory / ".ready").write_text(digest.hexdigest() + "\n")
    return directory / "scene.xml", cfg
