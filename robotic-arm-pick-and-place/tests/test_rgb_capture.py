"""Native capture, replay metadata, and physics independence of image resolution."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environments.bookshelf_task import BookshelfEnv
from environments.sorting import SortingEnv
from scripts.run_research import run_case
from utils.demonstrations import replay_episode, training_samples, validate_episode
from utils.research_protocol import load_manifest


class NativeCaptureTests(unittest.TestCase):
    def test_resolution_changes_pixels_but_not_dynamics(self):
        for cls in (BookshelfEnv, SortingEnv):
            with self.subTest(environment=cls.__name__), \
                    cls(observation_mode="rgb") as native, \
                    cls(observation_mode="rgb", image_size=(240, 320)) as small:
                a, _ = native.reset(seed=0)
                b, _ = small.reset(seed=0)
                self.assertTrue(native.observation_space.contains(a))
                self.assertNotEqual(native.fingerprint, small.fingerprint)
                for camera in ("front", "wrist"):
                    self.assertEqual(a["rgb"][camera].shape, (480, 640, 3))
                    self.assertEqual(b["rgb"][camera].shape, (240, 320, 3))
                    self.assertGreater(np.std(a["rgb"][camera]), 1)
                np.testing.assert_array_equal(native._physics_state(), small._physics_state())
                action = np.array([.1, 0, 0, 0, 0, 0, -.1, 1], dtype=np.float32)
                for _ in range(4):
                    ao, ar, ae, at, _ = native.step(action)
                    bo, br, be, bt, _ = small.step(action)
                    np.testing.assert_array_equal(native._physics_state(), small._physics_state())
                    np.testing.assert_array_equal(ao["proprio"], bo["proprio"])
                    self.assertEqual((ar, ae, at), (br, be, bt))

    def test_native_collection_replays_and_rejects_mislabeled_pixels(self):
        manifest = load_manifest()
        case = next(c for c in manifest["cases"] if c["environment"] == "sorting"
                    and c["split"] == "train" and c["axis"] == "development"
                    and c["asset_set"] == "nominal" and c["seed"] == 0)
        with tempfile.TemporaryDirectory() as directory:
            result = run_case(case, manifest, policy_name="hold", dataset_dir=directory, horizon=3)
            self.assertNotIn("error", result)
            path = result["dataset"]
            self.assertEqual(replay_episode(path)["transitions"], 3)
            with h5py.File(path, "r+") as episode:
                constructor = json.loads(episode.attrs["metadata"])["constructor"]
                self.assertEqual(constructor["image_size"], [480, 640])
                self.assertEqual(constructor["cameras"], ["front", "wrist"])
                for camera in constructor["cameras"]:
                    image = episode[f"observations/rgb/{camera}"]
                    self.assertEqual(image.shape, (4, 480, 640, 3))
                    self.assertEqual(image.dtype, np.uint8)
                    self.assertEqual(image.compression, "gzip")
                contract = json.loads(episode.attrs["action_contract"])
                contract["image_size_hw"] = [240, 320]
                episode.attrs["action_contract"] = json.dumps(contract)
            with self.assertRaisesRegex(ValueError, "RGB observation shape"):
                validate_episode(path)
            with self.assertRaisesRegex(ValueError, "RGB observation shape"):
                list(training_samples([path], include_failures=True))
            with h5py.File(path, "r+") as episode:
                contract["image_size_hw"] = [480, 640]
                episode.attrs["action_contract"] = json.dumps(contract)
                del episode["observations/rgb/wrist"]
            with self.assertRaisesRegex(ValueError, "RGB cameras"):
                validate_episode(path)


if __name__ == "__main__":
    unittest.main()
