"""Tests for the coordinate-system converter."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mocap_receiver.coordinate_converter import (
    COORDINATE_TRANSFORMS,
    convert_frame,
    convert_motion,
    main,
    save_motion_npz,
)
from mocap_receiver.converter import matrix_to_rotvec
from mocap_receiver.visualize import SmplMotion, axis_angle_to_matrix, load_smpl_motion


class SingleFrameConversionTests(unittest.TestCase):
    """Unit tests for convert_frame()."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.identity_pose = np.zeros(72, dtype=np.float64)  # T-pose
        cls.identity_trans = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # A known rotation: 90° about SMPL Y (up) — spins the character left.
        cls.rot_y_90_pose = np.zeros(72, dtype=np.float64)
        cls.rot_y_90_pose[:3] = [0.0, np.pi / 2, 0.0]  # axis-angle: Y axis, 90°
        cls.rot_y_90_trans = np.array([0.2, 1.05, 0.1], dtype=np.float64)

    # ------------------------------------------------------------------
    # Y-up (identity) leaves data unchanged
    # ------------------------------------------------------------------

    def test_y_up_identity_pose_unchanged(self):
        new_pose, new_trans = convert_frame(
            self.identity_pose, self.identity_trans, COORDINATE_TRANSFORMS["y_up"]
        )
        np.testing.assert_allclose(new_pose, self.identity_pose, atol=1e-12)
        np.testing.assert_allclose(new_trans, self.identity_trans, atol=1e-12)

    def test_y_up_rotated_pose_unchanged(self):
        new_pose, new_trans = convert_frame(
            self.rot_y_90_pose, self.rot_y_90_trans, COORDINATE_TRANSFORMS["y_up"]
        )
        np.testing.assert_allclose(new_pose, self.rot_y_90_pose, atol=1e-12)
        np.testing.assert_allclose(new_trans, self.rot_y_90_trans, atol=1e-12)

    # ------------------------------------------------------------------
    # Body pose is never touched by any transform
    # ------------------------------------------------------------------

    def test_body_pose_unchanged_by_z_up(self):
        body_pose = np.arange(69, dtype=np.float64) * 0.01
        pose = np.concatenate((np.zeros(3), body_pose))
        new_pose, _ = convert_frame(pose, self.identity_trans, COORDINATE_TRANSFORMS["z_up"])
        np.testing.assert_allclose(new_pose[3:], body_pose, atol=1e-12)

    def test_body_pose_unchanged_by_x_up(self):
        body_pose = np.arange(69, dtype=np.float64) * 0.01
        pose = np.concatenate((np.zeros(3), body_pose))
        new_pose, _ = convert_frame(pose, self.identity_trans, COORDINATE_TRANSFORMS["x_up"])
        np.testing.assert_allclose(new_pose[3:], body_pose, atol=1e-12)

    # ------------------------------------------------------------------
    # Round-trip: apply a transform and its inverse and get back the original
    # ------------------------------------------------------------------

    def test_z_up_round_trip(self):
        M = COORDINATE_TRANSFORMS["z_up"]
        M_inv = M.T  # orthogonal
        p1, t1 = convert_frame(self.rot_y_90_pose, self.rot_y_90_trans, M)
        p2, t2 = convert_frame(p1, t1, M_inv)
        np.testing.assert_allclose(p2, self.rot_y_90_pose, atol=1e-9)
        np.testing.assert_allclose(t2, self.rot_y_90_trans, atol=1e-9)

    def test_x_up_round_trip(self):
        M = COORDINATE_TRANSFORMS["x_up"]
        M_inv = M.T
        p1, t1 = convert_frame(self.rot_y_90_pose, self.rot_y_90_trans, M)
        p2, t2 = convert_frame(p1, t1, M_inv)
        np.testing.assert_allclose(p2, self.rot_y_90_pose, atol=1e-9)
        np.testing.assert_allclose(t2, self.rot_y_90_trans, atol=1e-9)

    # ------------------------------------------------------------------
    # Sanity: the transformed rotations are still valid (axis-angle norm ≤ π)
    # ------------------------------------------------------------------

    def test_z_up_rotation_norm_within_pi(self):
        M = COORDINATE_TRANSFORMS["z_up"]
        p, _ = convert_frame(self.rot_y_90_pose, self.rot_y_90_trans, M)
        self.assertLessEqual(float(np.linalg.norm(p[:3])), np.pi + 1e-9)

    def test_x_up_rotation_norm_within_pi(self):
        M = COORDINATE_TRANSFORMS["x_up"]
        p, _ = convert_frame(self.rot_y_90_pose, self.rot_y_90_trans, M)
        self.assertLessEqual(float(np.linalg.norm(p[:3])), np.pi + 1e-9)


class BatchConversionTests(unittest.TestCase):
    """Tests that exercise convert_motion with multi-frame data."""

    @classmethod
    def setUpClass(cls) -> None:
        # Three-frame mini motion
        cls.poses = np.zeros((3, 72), dtype=np.float64)
        cls.poses[0, :3] = [0.1, 0.0, 0.0]  # small X rotation
        cls.poses[1, :3] = [0.0, 0.2, 0.0]  # small Y rotation
        cls.poses[2, :3] = [0.0, 0.0, 0.3]  # small Z rotation
        # non-trivial body pose on last frame
        cls.poses[2, 3:6] = [0.05, -0.03, 0.02]
        cls.trans = np.array(
            [[0.0, 1.0, 0.0], [0.1, 1.05, 0.2], [-0.1, 0.98, -0.15]],
            dtype=np.float64,
        )
        cls.motion = SmplMotion(
            poses=cls.poses,
            trans=cls.trans,
            betas=np.zeros(10, dtype=np.float64),
            gender="neutral",
            mocap_framerate=60.0,
        )

    def test_three_systems_present_by_default(self):
        results = convert_motion(self.motion)
        self.assertSetEqual(set(results.keys()), {"y_up", "z_up", "x_up"})

    def test_every_output_has_same_frame_count(self):
        results = convert_motion(self.motion)
        for name, m in results.items():
            self.assertEqual(m.frame_count, self.motion.frame_count, f"{name} frame_count mismatch")

    def test_betas_are_preserved(self):
        results = convert_motion(self.motion)
        for name, m in results.items():
            np.testing.assert_allclose(
                m.betas, self.motion.betas, atol=1e-12,
                err_msg=f"{name} betas changed",
            )

    def test_gender_preserved(self):
        results = convert_motion(self.motion)
        for name, m in results.items():
            self.assertEqual(m.gender, "neutral", f"{name} gender changed")

    def test_framerate_preserved(self):
        results = convert_motion(self.motion)
        for name, m in results.items():
            self.assertAlmostEqual(
                m.mocap_framerate, 60.0, places=6,
                msg=f"{name} framerate changed",
            )

    def test_y_up_is_identity(self):
        results = convert_motion(self.motion, target_systems=["y_up"])
        m = results["y_up"]
        np.testing.assert_allclose(m.poses, self.motion.poses, atol=1e-12)
        np.testing.assert_allclose(m.trans, self.motion.trans, atol=1e-12)

    def test_subset_conversion(self):
        results = convert_motion(self.motion, target_systems=["z_up"])
        self.assertSetEqual(set(results.keys()), {"z_up"})


class FileRoundTripTests(unittest.TestCase):
    """Save a converted motion as NPZ, load it back, and verify correctness."""

    def setUp(self) -> None:
        poses = np.zeros((2, 72), dtype=np.float64)
        poses[0, :3] = [0.0, 0.4, 0.0]
        poses[1, :3] = [0.3, 0.0, 0.0]
        trans = np.array([[0.0, 1.0, 0.0], [0.1, 0.9, 0.2]], dtype=np.float64)
        self.motion = SmplMotion(
            poses=poses,
            trans=trans,
            betas=np.zeros(10, dtype=np.float64),
            gender="neutral",
            mocap_framerate=30.0,
        )

    def test_y_up_file_round_trip(self):
        converted = convert_motion(self.motion, target_systems=["y_up"])["y_up"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.npz"
            save_motion_npz(converted, path)
            self.assertTrue(path.is_file())

            loaded = load_smpl_motion(path)
            np.testing.assert_allclose(loaded.poses, converted.poses, atol=1e-6)
            np.testing.assert_allclose(loaded.trans, converted.trans, atol=1e-6)
            np.testing.assert_allclose(loaded.betas, converted.betas, atol=1e-6)

    def test_z_up_file_round_trip(self):
        converted = convert_motion(self.motion, target_systems=["z_up"])["z_up"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.npz"
            save_motion_npz(converted, path)
            loaded = load_smpl_motion(path)
            np.testing.assert_allclose(loaded.poses, converted.poses, atol=1e-6)
            np.testing.assert_allclose(loaded.trans, converted.trans, atol=1e-6)

    def test_x_up_file_round_trip(self):
        converted = convert_motion(self.motion, target_systems=["x_up"])["x_up"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.npz"
            save_motion_npz(converted, path)
            loaded = load_smpl_motion(path)
            np.testing.assert_allclose(loaded.poses, converted.poses, atol=1e-6)
            np.testing.assert_allclose(loaded.trans, converted.trans, atol=1e-6)


class CliIntegrationTests(unittest.TestCase):
    """End-to-end tests of the ``main()`` entry point."""

    @classmethod
    def setUpClass(cls) -> None:
        # build a minimal valid NPZ that load_smpl_motion can parse
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls.source_path = Path(cls._tmp_dir.name) / "dummy_motion.npz"
        poses = np.zeros((4, 72), dtype=np.float32)
        poses[:, :3] = np.random.default_rng(42).uniform(-0.5, 0.5, (4, 3))
        trans = np.zeros((4, 3), dtype=np.float32)
        trans[:, 1] = 1.0
        np.savez(
            cls.source_path,
            poses=poses,
            trans=trans,
            betas=np.zeros(10, dtype=np.float32),
            gender=np.asarray("neutral"),
            mocap_framerate=np.float32(30.0),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp_dir.cleanup()

    def test_cli_generates_three_outputs_by_default(self):
        with tempfile.TemporaryDirectory() as out_tmp:
            rc = main([str(self.source_path), "--output-dir", out_tmp])
            self.assertEqual(rc, 0)

            # find the timestamp subdirectory
            subdirs = list(Path(out_tmp).iterdir())
            self.assertEqual(len(subdirs), 1)
            ts_dir = subdirs[0]
            self.assertTrue(ts_dir.is_dir())

            for sys_name in ("y_up", "z_up", "x_up"):
                sys_dir = ts_dir / sys_name
                self.assertTrue(sys_dir.is_dir(), f"missing directory for {sys_name}")
                npz = sys_dir / "dummy_motion.npz"
                self.assertTrue(npz.is_file(), f"missing NPZ for {sys_name}")
                motion = load_smpl_motion(npz)
                self.assertEqual(motion.frame_count, 4)

    def test_cli_subset_systems(self):
        with tempfile.TemporaryDirectory() as out_tmp:
            rc = main([
                str(self.source_path), "--output-dir", out_tmp,
                "--systems", "z_up",
            ])
            self.assertEqual(rc, 0)
            subdirs = list(Path(out_tmp).iterdir())
            ts_dir = subdirs[0]
            self.assertTrue((ts_dir / "z_up").is_dir())
            self.assertFalse((ts_dir / "y_up").exists())
            self.assertFalse((ts_dir / "x_up").exists())


if __name__ == "__main__":
    unittest.main()
