from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from mocap_receiver.coordinate_converter import COORDINATE_TRANSFORMS, convert_frame
from mocap_receiver.merge_body_hand import (
    MergeError,
    build_parser,
    load_body_recording,
    load_hand_recording,
    main,
    merge_files,
)


STAGEII_KEYS = {
    "gender",
    "surface_model_type",
    "mocap_frame_rate",
    "mocap_time_length",
    "markers_latent",
    "latent_labels",
    "markers_latent_vids",
    "trans",
    "poses",
    "betas",
    "num_betas",
    "root_orient",
    "pose_body",
    "pose_hand",
    "pose_jaw",
    "pose_eye",
}


def write_body(
    path: Path,
    frames: list[int],
    frame_rate: float = 60.0,
    gender: str = "neutral",
    include_frame_index: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(frames)
    poses = np.arange(count * 72, dtype=np.float64).reshape(count, 72) / 100.0
    trans = np.arange(count * 3, dtype=np.float64).reshape(count, 3) / 10.0
    fields: dict[str, object] = {
        "poses": poses,
        "trans": trans,
        "betas": np.zeros(10, dtype=np.float64),
        "gender": np.asarray(gender),
        "mocap_framerate": np.asarray(frame_rate, dtype=np.float64),
    }
    if include_frame_index:
        fields["frame_index"] = np.asarray(frames, dtype=np.int64)
    np.savez(path, **fields)
    return poses, trans


def write_hand(
    path: Path,
    frames: list[int],
    frame_rate: float = 60.0,
    gender: str = "neutral",
    include_components: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(frames)
    pose_hand = 10.0 + np.arange(count * 90, dtype=np.float64).reshape(count, 90) / 100.0
    pose_jaw = 20.0 + np.arange(count * 3, dtype=np.float64).reshape(count, 3) / 100.0
    pose_eye = 30.0 + np.arange(count * 6, dtype=np.float64).reshape(count, 6) / 100.0
    root = np.zeros((count, 3), dtype=np.float64)
    body = np.zeros((count, 63), dtype=np.float64)
    poses = np.concatenate((root, body, pose_jaw, pose_eye, pose_hand), axis=1)
    fields: dict[str, object] = {
        "frame_index": np.asarray(frames, dtype=np.int64),
        "poses": poses,
        "betas": np.arange(16, dtype=np.float64) / 10.0,
        "num_betas": np.asarray(16, dtype=np.int64),
        "gender": np.asarray(gender),
        "surface_model_type": np.asarray("smplx"),
        "mocap_frame_rate": np.asarray(frame_rate, dtype=np.float64),
    }
    if include_components:
        fields.update(pose_hand=pose_hand, pose_jaw=pose_jaw, pose_eye=pose_eye)
    np.savez(path, **fields)
    return pose_hand, pose_jaw, pose_eye


class MergeBodyHandTests(unittest.TestCase):
    def test_axis_up_parser_defaults_to_y(self) -> None:
        self.assertEqual(build_parser().get_default("axis_up"), "y")

    def test_exact_frame_intersection_and_stageii_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body_path = root / "body.npz"
            hand_path = root / "hand.npz"
            output_path = root / "merged.npz"
            body_poses, body_trans = write_body(body_path, [10, 12, 13, 15])
            hand_pose, hand_jaw, hand_eye = write_hand(
                hand_path, [9, 10, 11, 13, 15, 16]
            )

            stats = merge_files(body_path, hand_path, output_path)

            self.assertEqual(stats.body_frames, 4)
            self.assertEqual(stats.hand_frames, 6)
            self.assertEqual(stats.common_frames, 3)
            self.assertEqual(stats.body_only_frames, 1)
            self.assertEqual(stats.hand_only_frames, 3)
            self.assertEqual((stats.first_frame, stats.last_frame), (10, 15))
            self.assertEqual(stats.gap_count, 2)
            self.assertEqual(stats.missing_frame_count, 3)

            with np.load(output_path, allow_pickle=True) as merged:
                self.assertEqual(set(merged.files), STAGEII_KEYS)
                self.assertNotIn("frame_index", merged.files)
                self.assertNotIn("hand_positions", merged.files)
                self.assertNotIn("hand_global_orient", merged.files)
                self.assertEqual(merged["poses"].shape, (3, 165))
                self.assertEqual(merged["root_orient"].shape, (3, 3))
                self.assertEqual(merged["pose_body"].shape, (3, 63))
                self.assertEqual(merged["pose_hand"].shape, (3, 90))
                self.assertEqual(merged["pose_jaw"].shape, (3, 3))
                self.assertEqual(merged["pose_eye"].shape, (3, 6))
                self.assertEqual(merged["trans"].shape, (3, 3))

                body_rows = [0, 2, 3]
                hand_rows = [1, 3, 4]
                np.testing.assert_allclose(merged["root_orient"], body_poses[body_rows, 0:3])
                np.testing.assert_allclose(merged["pose_body"], body_poses[body_rows, 3:66])
                np.testing.assert_allclose(merged["trans"], body_trans[body_rows])
                np.testing.assert_allclose(merged["pose_hand"], hand_pose[hand_rows])
                np.testing.assert_allclose(merged["pose_jaw"], hand_jaw[hand_rows])
                np.testing.assert_allclose(merged["pose_eye"], hand_eye[hand_rows])
                expected_poses = np.concatenate(
                    (
                        merged["root_orient"],
                        merged["pose_body"],
                        merged["pose_jaw"],
                        merged["pose_eye"],
                        merged["pose_hand"],
                    ),
                    axis=1,
                )
                np.testing.assert_array_equal(merged["poses"], expected_poses)
                np.testing.assert_allclose(merged["betas"], np.arange(16) / 10.0)
                self.assertEqual(int(merged["num_betas"]), 16)
                self.assertEqual(str(merged["gender"]), "neutral")
                self.assertEqual(str(merged["surface_model_type"]), "smplx")
                self.assertEqual(float(merged["mocap_frame_rate"]), 60.0)
                self.assertAlmostEqual(float(merged["mocap_time_length"]), 3.0 / 60.0)
                self.assertEqual(merged["markers_latent"].shape, (0, 3))
                self.assertEqual(merged["latent_labels"].shape, (0,))
                self.assertEqual(merged["markers_latent_vids"].item(), {})
                self.assertEqual(merged["gender"].dtype, np.dtype("<U7"))
                self.assertEqual(merged["surface_model_type"].dtype, np.dtype("<U5"))
                self.assertEqual(merged["latent_labels"].dtype, np.dtype("<U9"))
                self.assertEqual(merged["markers_latent_vids"].dtype, np.dtype(object))
                self.assertEqual(merged["num_betas"].dtype, np.int64)
                self.assertEqual(merged["mocap_frame_rate"].dtype, np.float64)
                self.assertEqual(merged["mocap_time_length"].dtype, np.float64)
                for key in (
                    "trans",
                    "poses",
                    "betas",
                    "root_orient",
                    "pose_body",
                    "pose_hand",
                    "pose_jaw",
                    "pose_eye",
                ):
                    self.assertEqual(merged[key].dtype, np.float64)

    def test_hand_components_can_be_read_from_full_poses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body_path = root / "body.npz"
            hand_path = root / "hand.npz"
            output_path = root / "merged.npz"
            write_body(body_path, [1, 2])
            expected_hand, expected_jaw, expected_eye = write_hand(
                hand_path, [1, 2], include_components=False
            )

            merge_files(body_path, hand_path, output_path)

            with np.load(output_path, allow_pickle=True) as merged:
                np.testing.assert_allclose(merged["pose_hand"], expected_hand)
                np.testing.assert_allclose(merged["pose_jaw"], expected_jaw)
                np.testing.assert_allclose(merged["pose_eye"], expected_eye)

    def test_x_and_z_up_transform_only_root_and_translation(self) -> None:
        for axis_up in ("x", "z"):
            with self.subTest(axis_up=axis_up), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                body_path = root / "body.npz"
                hand_path = root / "hand.npz"
                output_path = root / "merged.npz"
                body_poses, body_trans = write_body(body_path, [10, 11])
                hand_pose, hand_jaw, hand_eye = write_hand(hand_path, [10, 11])

                merge_files(
                    body_path,
                    hand_path,
                    output_path,
                    axis_up=axis_up,
                )

                expected_root = []
                expected_trans = []
                matrix = COORDINATE_TRANSFORMS[f"{axis_up}_up"]
                for pose, translation in zip(body_poses, body_trans):
                    converted_pose, converted_translation = convert_frame(
                        pose,
                        translation,
                        matrix,
                    )
                    expected_root.append(converted_pose[:3])
                    expected_trans.append(converted_translation)
                with np.load(output_path, allow_pickle=True) as merged:
                    np.testing.assert_allclose(merged["root_orient"], expected_root)
                    np.testing.assert_allclose(merged["trans"], expected_trans)
                    np.testing.assert_allclose(merged["pose_body"], body_poses[:, 3:66])
                    np.testing.assert_allclose(merged["pose_hand"], hand_pose)
                    np.testing.assert_allclose(merged["pose_jaw"], hand_jaw)
                    np.testing.assert_allclose(merged["pose_eye"], hand_eye)
                    expected_poses = np.concatenate(
                        (
                            merged["root_orient"],
                            merged["pose_body"],
                            merged["pose_jaw"],
                            merged["pose_eye"],
                            merged["pose_hand"],
                        ),
                        axis=1,
                    )
                    np.testing.assert_array_equal(merged["poses"], expected_poses)

    def test_invalid_axis_up_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body_path = root / "body.npz"
            hand_path = root / "hand.npz"
            write_body(body_path, [1])
            write_hand(hand_path, [1])
            with self.assertRaisesRegex(MergeError, "axis_up"):
                merge_files(
                    body_path,
                    hand_path,
                    root / "merged.npz",
                    axis_up="invalid",
                )

    def test_missing_body_frame_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "body.npz"
            write_body(path, [1, 2], include_frame_index=False)
            with self.assertRaisesRegex(MergeError, "no 'frame_index'"):
                load_body_recording(path)

    def test_duplicate_frame_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hand.npz"
            write_hand(path, [4, 4])
            with self.assertRaisesRegex(MergeError, "duplicate frame 4"):
                load_hand_recording(path)

    def test_no_common_frames_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body_path = root / "body.npz"
            hand_path = root / "hand.npz"
            write_body(body_path, [1, 2])
            write_hand(hand_path, [3, 4])
            with self.assertRaisesRegex(MergeError, "no common"):
                merge_files(body_path, hand_path, root / "merged.npz")

    def test_frame_rate_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body_path = root / "body.npz"
            hand_path = root / "hand.npz"
            write_body(body_path, [1], frame_rate=60.0)
            write_hand(hand_path, [1], frame_rate=120.0)
            with self.assertRaisesRegex(MergeError, "frame-rate mismatch"):
                merge_files(body_path, hand_path, root / "merged.npz")

    def test_existing_output_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body_path = root / "body.npz"
            hand_path = root / "hand.npz"
            output_path = root / "merged.npz"
            write_body(body_path, [1])
            write_hand(hand_path, [1])
            output_path.write_bytes(b"existing")

            with self.assertRaises(FileExistsError):
                merge_files(body_path, hand_path, output_path)
            self.assertEqual(output_path.read_bytes(), b"existing")
            stats = merge_files(body_path, hand_path, output_path, overwrite=True)
            self.assertEqual(stats.common_frames, 1)
            with np.load(output_path, allow_pickle=True) as merged:
                self.assertEqual(merged["poses"].shape, (1, 165))

    def test_cli_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body_path = root / "body.npz"
            hand_path = root / "hand.npz"
            output_path = root / "merged.npz"
            write_body(body_path, [100, 101, 102])
            write_hand(hand_path, [101, 102, 103])

            result = main(
                [str(body_path), str(hand_path), "--output", str(output_path)]
            )

            self.assertEqual(result, 0)
            with np.load(output_path, allow_pickle=True) as merged:
                self.assertEqual(merged["poses"].shape, (2, 165))


if __name__ == "__main__":
    unittest.main()
