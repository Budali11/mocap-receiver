from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from mocap_receiver.smplx_visualize import (
    build_parser,
    load_smplx_motion,
    normalize_smplx_gender,
)


def _stageii_fields(frame_count: int = 2) -> dict[str, np.ndarray]:
    root = np.zeros((frame_count, 3), dtype=np.float64)
    body = np.zeros((frame_count, 63), dtype=np.float64)
    jaw = np.zeros((frame_count, 3), dtype=np.float64)
    eye = np.zeros((frame_count, 6), dtype=np.float64)
    hand = np.zeros((frame_count, 90), dtype=np.float64)
    return {
        "gender": np.asarray("neutral", dtype="<U7"),
        "surface_model_type": np.asarray("smplx", dtype="<U5"),
        "mocap_frame_rate": np.asarray(60.0, dtype=np.float64),
        "trans": np.zeros((frame_count, 3), dtype=np.float64),
        "poses": np.concatenate((root, body, jaw, eye, hand), axis=1),
        "betas": np.zeros(16, dtype=np.float64),
        "root_orient": root,
        "pose_body": body,
        "pose_jaw": jaw,
        "pose_eye": eye,
        "pose_hand": hand,
    }


class SmplxVisualizationTests(unittest.TestCase):
    def test_parser_defaults_to_gpu_and_local_model_directory(self) -> None:
        arguments = build_parser().parse_args(["merged.npz"])
        self.assertEqual(arguments.backend, "gpu")
        self.assertEqual(arguments.device, "auto")
        self.assertEqual(arguments.model_dir.name, "smplx_model")

    def test_loads_merged_stageii_motion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "merged.npz"
            np.savez_compressed(path, **_stageii_fields(3))

            motion = load_smplx_motion(path)

            self.assertEqual(motion.poses.shape, (3, 165))
            self.assertEqual(motion.trans.shape, (3, 3))
            self.assertEqual(motion.betas.shape, (16,))
            self.assertEqual(motion.frame_count, 3)
            self.assertEqual(motion.mocap_framerate, 60.0)
            self.assertEqual(motion.gender, "neutral")

    def test_rejects_inconsistent_component_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            fields = _stageii_fields()
            fields["poses"] = fields["poses"].copy()
            fields["poses"][0, 80] = 1.0
            np.savez_compressed(path, **fields)

            with self.assertRaisesRegex(ValueError, "poses must equal"):
                load_smplx_motion(path)

    def test_rejects_classic_smpl_pose_width(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classic.npz"
            fields = _stageii_fields()
            fields["poses"] = np.zeros((2, 72), dtype=np.float64)
            np.savez_compressed(path, **fields)

            with self.assertRaisesRegex(ValueError, r"\(N, 165\)"):
                load_smplx_motion(path)

    def test_gender_aliases(self) -> None:
        self.assertEqual(normalize_smplx_gender("N"), "neutral")
        self.assertEqual(normalize_smplx_gender("male"), "male")
        self.assertEqual(normalize_smplx_gender("F"), "female")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            normalize_smplx_gender("unknown")


if __name__ == "__main__":
    unittest.main()
