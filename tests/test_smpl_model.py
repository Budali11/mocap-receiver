from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mocap_receiver.smpl_model import (
    DEFAULT_SMPL_MODEL_DIR,
    MODEL_FILENAMES,
    SmplModel,
    normalize_gender,
    resolve_model_path,
)
from mocap_receiver.visualize import SmplMotion, axis_angle_to_matrix, play_motion


@unittest.skipUnless(
    (DEFAULT_SMPL_MODEL_DIR / MODEL_FILENAMES["neutral"]).is_file(),
    "local SMPL v1.1 models are not available",
)
class SmplModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = SmplModel.from_directory(DEFAULT_SMPL_MODEL_DIR, "neutral")
        cls.body = cls.model.with_betas(np.zeros(10))

    def test_all_three_configured_model_files_exist(self) -> None:
        for gender in ("neutral", "male", "female"):
            path = resolve_model_path(DEFAULT_SMPL_MODEL_DIR, gender)
            self.assertTrue(path.is_file())

    def test_female_and_male_models_load(self) -> None:
        for gender in ("female", "male"):
            model = SmplModel.from_directory(DEFAULT_SMPL_MODEL_DIR, gender)
            self.assertEqual(model.gender, gender)
            self.assertEqual(model.v_template.shape, (6890, 3))
            self.assertEqual(model.faces.shape, (13776, 3))

    def test_model_arrays_have_classic_smpl_shapes(self) -> None:
        self.assertEqual(self.model.v_template.shape, (6890, 3))
        self.assertEqual(self.model.shapedirs.shape, (6890, 3, 10))
        self.assertEqual(self.model.posedirs.shape, (6890 * 3, 207))
        self.assertEqual(self.model.weights.shape, (6890, 24))
        self.assertEqual(self.model.faces.shape, (13776, 3))
        self.assertGreaterEqual(int(self.model.faces.min()), 0)
        self.assertLess(int(self.model.faces.max()), 6890)

    def test_identity_lbs_matches_shaped_template_and_places_pelvis(self) -> None:
        pelvis_position = np.array([0.0, 1.11, 0.0])
        frame = self.body.pose(np.zeros(72), pelvis_position)
        expected_shift = pelvis_position - self.body.rest_joints[0]

        np.testing.assert_allclose(
            frame.vertices,
            self.body.shaped_vertices + expected_shift,
            atol=2e-6,
        )
        np.testing.assert_allclose(frame.joints[0], pelvis_position, atol=1e-7)
        np.testing.assert_allclose(
            frame.joints,
            self.body.rest_joints + expected_shift,
            atol=2e-6,
        )

    def test_root_rotation_rotates_mesh_about_requested_pelvis(self) -> None:
        pose = np.zeros(72)
        pose[1] = math.pi / 2.0
        pelvis = np.array([0.2, 1.0, -0.3])
        frame = self.body.pose(pose, pelvis)
        rotation = axis_angle_to_matrix(pose[:3])
        centered = self.body.shaped_vertices - self.body.rest_joints[0]
        expected = (rotation @ centered.T).T + pelvis

        np.testing.assert_allclose(frame.vertices, expected, atol=3e-6)
        np.testing.assert_allclose(frame.joints[0], pelvis, atol=1e-7)

    def test_shape_coefficients_change_body_shape(self) -> None:
        betas = np.zeros(10)
        betas[0] = 1.0
        shaped = self.model.with_betas(betas)
        self.assertGreater(
            float(np.max(np.abs(shaped.shaped_vertices - self.body.shaped_vertices))),
            1e-4,
        )

    def test_gender_aliases(self) -> None:
        self.assertEqual(normalize_gender("F"), "female")
        self.assertEqual(normalize_gender("male"), "male")
        self.assertEqual(normalize_gender("Neutral"), "neutral")

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib not installed")
    def test_offline_viewer_renders_model_mesh_to_gif(self) -> None:
        motion = SmplMotion(
            poses=np.zeros((2, 72)),
            trans=np.array([[0.0, 1.11, 0.0], [0.02, 1.11, 0.0]]),
            betas=np.zeros(10),
            gender="neutral",
            mocap_framerate=60.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "mesh.gif"
            play_motion(
                motion=motion,
                frame_indices=np.array([0, 1], dtype=np.int64),
                speed=1.0,
                view_radius=1.2,
                fixed_camera=False,
                repeat=False,
                save_path=output_path,
                dpi=40,
                smpl_body=self.body,
                mesh_face_step=50,
            )
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

    @unittest.skipUnless(
        os.environ.get("SMPL_GPU_TEST") == "1" and importlib.util.find_spec("pyglet"),
        "set SMPL_GPU_TEST=1 to create an OpenGL smoke-test window",
    )
    def test_gpu_player_uses_opengl_renderer(self) -> None:
        from mocap_receiver.gpu_visualize import play_motion_gpu

        motion = SmplMotion(
            poses=np.zeros((4, 72)),
            trans=np.repeat(np.array([[0.0, 1.11, 0.0]]), repeats=4, axis=0),
            betas=np.zeros(10),
            gender="neutral",
            mocap_framerate=60.0,
        )
        info = play_motion_gpu(
            motion=motion,
            frame_indices=np.arange(4, dtype=np.int64),
            smpl_body=self.body,
            speed=1.0,
            view_radius=1.2,
            fixed_camera=False,
            repeat=False,
            width=320,
            height=240,
            vsync=False,
            visible=False,
            auto_close_after_frames=4,
        )
        self.assertGreaterEqual(info["draw_count"], 4)
        self.assertIn("NVIDIA", info["renderer"].upper())
        self.assertIsNotNone(info["pixel_standard_deviation"])
        self.assertGreater(info["pixel_standard_deviation"], 1.0)


if __name__ == "__main__":
    unittest.main()
