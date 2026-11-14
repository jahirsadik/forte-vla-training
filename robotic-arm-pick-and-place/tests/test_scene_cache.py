"""Independent workers must not read a partially generated scene or texture."""

from concurrent.futures import ProcessPoolExecutor
import sys
import tempfile
from pathlib import Path
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def compile_cold_variant(root):
    import mujoco
    import environments.variants as variants
    variants.PROJECT = Path(root)
    path, _ = variants.scene_variant('bookshelf_hard', 'train_small')
    model = mujoco.MjModel.from_xml_path(str(path))
    return str(path), model.nq, (path.parent / '.ready').exists()


class SceneCacheTests(unittest.TestCase):
    def test_concurrent_cold_cache_readers_compile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('assets', 'tools', 'configs', 'environments'):
                (root / name).symlink_to(PROJECT / name, target_is_directory=True)
            with ProcessPoolExecutor(max_workers=3) as workers:
                results = list(workers.map(compile_cold_variant, [temporary] * 3))
            self.assertEqual(len(set(results)), 1)
            self.assertTrue(results[0][2])
            self.assertGreater(results[0][1], 80)


if __name__ == '__main__':
    unittest.main()
