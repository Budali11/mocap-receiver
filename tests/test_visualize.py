from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mocap_receiver.visualize import (
    SmplMotion,
    axis_angle_to_matrix,
    build_parser,
    canonical_smpl_rest_offsets,
    canonical_smpl_rest_positions,
    forward_kinematics,
    load_smpl_motion,
    play_motion,
)


class VisualizationMathTests(unittest.TestCase):
    def test_interactive_player_defaults_to_gpu_backend(self) -> None:
        arguments = build_parser().parse_args(["motion.npz"])
        self.assertEqual(arguments.backend, "gpu")

    def test_identity_pose_matches_rest_skeleton_at_translation(self) -> None:
        rest = canonical_smpl_rest_positions()
        translation = np.array([1.0, 2.0, 3.0])

        joints = forward_kinematics(np.zeros(72), translation)

        np.testing.assert_allclose(joints, rest + translation, atol=1e-10)
        np.testing.assert_allclose(joints[0], translation, atol=1e-10)

    def test_root_rotation_rotates_complete_skeleton(self) -> None:
        rest = canonical_smpl_rest_positions()
        pose = np.zeros(72)
        pose[1] = math.pi / 2.0
        root_rotation = axis_angle_to_matrix(pose[:3])

        joints = forward_kinematics(pose, np.zeros(3))

        np.testing.assert_allclose(joints, (root_rotation @ rest.T).T, atol=1e-10)

    def test_rest_offsets_have_smpl_shape_and_nonzero_bones(self) -> None:
        offsets = canonical_smpl_rest_offsets()
        self.assertEqual(offsets.shape, (24, 3))
        np.testing.assert_allclose(offsets[0], 0.0)
        self.assertTrue(np.all(np.linalg.norm(offsets[1:], axis=1) > 0.0))

    def test_loads_flat_and_joint_shaped_pose_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            flat_path = Path(directory) / "flat.npz"
            joint_path = Path(directory) / "joint.npz"
            fields = {
                "trans": np.zeros((2, 3), dtype=np.float32),
                "betas": np.zeros(10, dtype=np.float32),
                "gender": np.asarray("neutral"),
                "mocap_framerate": np.asarray(60.0),
            }
            np.savez(flat_path, poses=np.zeros((2, 72)), **fields)
            np.savez(joint_path, poses=np.zeros((2, 24, 3)), **fields)

            flat = load_smpl_motion(flat_path)
            joint = load_smpl_motion(joint_path)

            self.assertEqual(flat.poses.shape, (2, 72))
            self.assertEqual(joint.poses.shape, (2, 72))
            self.assertEqual(flat.frame_count, 2)
            self.assertEqual(flat.mocap_framerate, 60.0)

    def test_rejects_non_smpl_pose_width(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.npz"
            np.savez(path, poses=np.zeros((2, 156)), trans=np.zeros((2, 3)))
            with self.assertRaisesRegex(ValueError, "poses must have shape"):
                load_smpl_motion(path)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib not installed")
    def test_saves_gif_preview(self) -> None:
        motion = SmplMotion(
            poses=np.zeros((2, 72)),
            trans=np.array([[0.0, 1.11, 0.0], [0.05, 1.11, 0.0]]),
            betas=np.zeros(10),
            gender="neutral",
            mocap_framerate=60.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "preview.gif"
            play_motion(
                motion=motion,
                frame_indices=np.array([0, 1], dtype=np.int64),
                speed=1.0,
                view_radius=1.2,
                fixed_camera=False,
                repeat=False,
                save_path=output_path,
                dpi=50,
            )
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
