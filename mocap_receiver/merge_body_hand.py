"""Merge frame-aligned body SMPL and hand SMPL-X NPZ recordings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray

from .coordinate_converter import COORDINATE_TRANSFORMS, convert_frame


LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class MergeError(ValueError):
    """Raised when two recordings cannot be aligned or combined safely."""


AXIS_UP_SYSTEMS = {
    "x": "x_up",
    "y": "y_up",
    "z": "z_up",
}


@dataclass(frozen=True)
class BodyRecording:
    frame_index: IntArray
    poses: FloatArray
    trans: FloatArray
    gender: str
    mocap_frame_rate: float


@dataclass(frozen=True)
class HandRecording:
    frame_index: IntArray
    pose_hand: FloatArray
    pose_jaw: FloatArray
    pose_eye: FloatArray
    betas: FloatArray
    gender: str
    mocap_frame_rate: float


@dataclass(frozen=True)
class MergeStats:
    body_frames: int
    hand_frames: int
    common_frames: int
    body_only_frames: int
    hand_only_frames: int
    first_frame: int
    last_frame: int
    gap_count: int
    missing_frame_count: int


@dataclass(frozen=True)
class MergedStageiiMotion:
    """The exact field set written to a Stage-II-style output archive."""

    gender: str
    mocap_frame_rate: float
    trans: FloatArray
    poses: FloatArray
    betas: FloatArray
    root_orient: FloatArray
    pose_body: FloatArray
    pose_hand: FloatArray
    pose_jaw: FloatArray
    pose_eye: FloatArray

    @property
    def frame_count(self) -> int:
        return int(self.poses.shape[0])

    def archive_fields(self) -> dict[str, Any]:
        """Return fields matching the key set of OK_B_stageii.npz."""

        empty_marker_vertices = np.empty((), dtype=object)
        empty_marker_vertices[()] = {}
        return {
            "gender": np.asarray(self.gender, dtype="<U7"),
            "surface_model_type": np.asarray("smplx", dtype="<U5"),
            "mocap_frame_rate": np.asarray(self.mocap_frame_rate, dtype=np.float64),
            "mocap_time_length": np.asarray(
                self.frame_count / self.mocap_frame_rate, dtype=np.float64
            ),
            "markers_latent": np.empty((0, 3), dtype=np.float64),
            "latent_labels": np.empty(0, dtype="<U9"),
            "markers_latent_vids": empty_marker_vertices,
            "trans": self.trans,
            "poses": self.poses,
            "betas": self.betas,
            "num_betas": np.asarray(16, dtype=np.int64),
            "root_orient": self.root_orient,
            "pose_body": self.pose_body,
            "pose_hand": self.pose_hand,
            "pose_jaw": self.pose_jaw,
            "pose_eye": self.pose_eye,
        }


def _numeric_array(
    archive: NpzFile,
    key: str,
    expected_tail: tuple[int, ...],
    frame_count: int,
    source_name: str,
) -> FloatArray:
    if key not in archive.files:
        raise MergeError(f"{source_name} file is missing required field {key!r}")
    try:
        array = np.asarray(archive[key], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MergeError(f"{source_name}.{key} must contain numeric values") from exc
    expected_shape = (frame_count, *expected_tail)
    if array.shape != expected_shape:
        raise MergeError(
            f"{source_name}.{key} must have shape {expected_shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise MergeError(f"{source_name}.{key} contains NaN or infinity")
    return array


def _static_numeric_array(
    archive: NpzFile,
    key: str,
    shape: tuple[int, ...],
    source_name: str,
) -> FloatArray:
    if key not in archive.files:
        raise MergeError(f"{source_name} file is missing required field {key!r}")
    try:
        array = np.asarray(archive[key], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MergeError(f"{source_name}.{key} must contain numeric values") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise MergeError(
            f"{source_name}.{key} must have shape {shape} and contain finite values"
        )
    return array


def _frame_indices(archive: NpzFile, source_name: str) -> IntArray:
    if "frame_index" not in archive.files:
        raise MergeError(
            f"{source_name} file has no 'frame_index'; regenerate it with the updated "
            "recorder because exact alignment cannot be inferred from array positions"
        )
    raw = np.asarray(archive["frame_index"])
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise MergeError(f"{source_name}.frame_index must be a one-dimensional integer array")
    indices = raw.astype(np.int64, copy=False)
    if np.unique(indices).size != indices.size:
        values, counts = np.unique(indices, return_counts=True)
        duplicate = int(values[np.flatnonzero(counts > 1)[0]])
        raise MergeError(f"{source_name}.frame_index contains duplicate frame {duplicate}")
    return indices


def _scalar_string(archive: NpzFile, key: str, source_name: str) -> str:
    if key not in archive.files:
        raise MergeError(f"{source_name} file is missing required field {key!r}")
    value = np.asarray(archive[key])
    if value.shape != ():
        raise MergeError(f"{source_name}.{key} must be a scalar string")
    result = str(value.item())
    if not result:
        raise MergeError(f"{source_name}.{key} must not be empty")
    return result


def _frame_rate(archive: NpzFile, source_name: str) -> float:
    keys = [key for key in ("mocap_frame_rate", "mocap_framerate") if key in archive.files]
    if not keys:
        raise MergeError(
            f"{source_name} file must contain 'mocap_frame_rate' or 'mocap_framerate'"
        )
    values: list[float] = []
    for key in keys:
        raw = np.asarray(archive[key])
        if raw.shape != ():
            raise MergeError(f"{source_name}.{key} must be a scalar")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise MergeError(f"{source_name}.{key} must be numeric") from exc
        if not np.isfinite(value) or value <= 0.0:
            raise MergeError(f"{source_name}.{key} must be a positive finite number")
        values.append(value)
    if len(values) == 2 and not np.isclose(values[0], values[1], atol=1e-6, rtol=1e-6):
        raise MergeError(f"{source_name} file contains conflicting frame-rate fields")
    return values[0]


def load_body_recording(path: Path) -> BodyRecording:
    """Load and validate a locally recorded classic SMPL body motion."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            frame_index = _frame_indices(archive, "body")
            frame_count = int(frame_index.size)
            poses = _numeric_array(archive, "poses", (72,), frame_count, "body")
            trans = _numeric_array(archive, "trans", (3,), frame_count, "body")
            gender = _scalar_string(archive, "gender", "body")
            mocap_frame_rate = _frame_rate(archive, "body")
    except (OSError, ValueError) as exc:
        if isinstance(exc, MergeError):
            raise
        raise MergeError(f"cannot load body NPZ {path}: {exc}") from exc
    return BodyRecording(frame_index, poses, trans, gender, mocap_frame_rate)


def _optional_pose_component(
    archive: NpzFile,
    key: str,
    poses: FloatArray | None,
    pose_slice: slice,
    width: int,
    frame_count: int,
) -> FloatArray:
    if key in archive.files:
        return _numeric_array(archive, key, (width,), frame_count, "hand")
    if poses is not None:
        return poses[:, pose_slice].copy()
    return np.zeros((frame_count, width), dtype=np.float64)


def load_hand_recording(path: Path) -> HandRecording:
    """Load and validate a locally recorded SMPL-X hand motion."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            frame_index = _frame_indices(archive, "hand")
            frame_count = int(frame_index.size)
            poses: FloatArray | None = None
            if "poses" in archive.files:
                poses = _numeric_array(archive, "poses", (165,), frame_count, "hand")
            if poses is None and "pose_hand" not in archive.files:
                raise MergeError("hand file must contain 'pose_hand' or 165-D 'poses'")
            pose_hand = _optional_pose_component(
                archive, "pose_hand", poses, slice(75, 165), 90, frame_count
            )
            pose_jaw = _optional_pose_component(
                archive, "pose_jaw", poses, slice(66, 69), 3, frame_count
            )
            pose_eye = _optional_pose_component(
                archive, "pose_eye", poses, slice(69, 75), 6, frame_count
            )
            if poses is not None and "pose_hand" in archive.files:
                if not np.allclose(poses[:, 75:165], pose_hand, atol=1e-6, rtol=1e-6):
                    raise MergeError("hand.poses and hand.pose_hand are inconsistent")
            betas = _static_numeric_array(archive, "betas", (16,), "hand")
            if "num_betas" in archive.files:
                num_betas = np.asarray(archive["num_betas"])
                if num_betas.shape != () or int(num_betas) != 16:
                    raise MergeError("hand.num_betas must be the scalar value 16")
            gender = _scalar_string(archive, "gender", "hand")
            if "surface_model_type" in archive.files:
                model_type = _scalar_string(archive, "surface_model_type", "hand")
                if model_type.lower() != "smplx":
                    raise MergeError("hand.surface_model_type must be 'smplx'")
            mocap_frame_rate = _frame_rate(archive, "hand")
    except (OSError, ValueError) as exc:
        if isinstance(exc, MergeError):
            raise
        raise MergeError(f"cannot load hand NPZ {path}: {exc}") from exc
    return HandRecording(
        frame_index,
        pose_hand,
        pose_jaw,
        pose_eye,
        betas,
        gender,
        mocap_frame_rate,
    )


def _convert_world_axes(
    root_orient: FloatArray,
    trans: FloatArray,
    axis_up: str,
) -> tuple[FloatArray, FloatArray]:
    """Convert root rotations and translations from canonical SMPL Y-up."""

    try:
        system_name = AXIS_UP_SYSTEMS[axis_up.lower()]
    except (AttributeError, KeyError) as exc:
        raise MergeError("axis_up must be one of: x, y, z") from exc
    if system_name == "y_up":
        return root_orient, trans

    coordinate_matrix = COORDINATE_TRANSFORMS[system_name]
    converted_root = np.empty_like(root_orient, dtype=np.float64)
    converted_trans = np.empty_like(trans, dtype=np.float64)
    temporary_pose = np.zeros(72, dtype=np.float64)
    for frame in range(root_orient.shape[0]):
        temporary_pose[:3] = root_orient[frame]
        converted_pose, converted_trans[frame] = convert_frame(
            temporary_pose,
            trans[frame],
            coordinate_matrix,
        )
        converted_root[frame] = converted_pose[:3]
    return converted_root, converted_trans


def merge_recordings(
    body: BodyRecording,
    hand: HandRecording,
    axis_up: str = "y",
) -> tuple[MergedStageiiMotion, MergeStats]:
    """Align exact shared frames and construct a full SMPL-X motion."""

    if body.gender.lower() != hand.gender.lower():
        raise MergeError(
            f"gender mismatch: body is {body.gender!r}, hand is {hand.gender!r}"
        )
    if not np.isclose(
        body.mocap_frame_rate,
        hand.mocap_frame_rate,
        atol=1e-6,
        rtol=1e-6,
    ):
        raise MergeError(
            "mocap frame-rate mismatch: "
            f"body={body.mocap_frame_rate:g}, hand={hand.mocap_frame_rate:g}"
        )

    common, body_rows, hand_rows = np.intersect1d(
        body.frame_index,
        hand.frame_index,
        assume_unique=True,
        return_indices=True,
    )
    if common.size == 0:
        raise MergeError("body and hand recordings have no common frame_index values")

    root_orient = body.poses[body_rows, 0:3].copy()
    # Classic SMPL has 23 non-root joints.  SMPL-X body_pose contains the
    # first 21; the final classic left_hand/right_hand terminal rotations are
    # replaced by the fully articulated hand pose below.
    pose_body = body.poses[body_rows, 3:66].copy()
    trans = body.trans[body_rows].copy()
    root_orient, trans = _convert_world_axes(root_orient, trans, axis_up)
    pose_jaw = hand.pose_jaw[hand_rows].copy()
    pose_eye = hand.pose_eye[hand_rows].copy()
    pose_hand = hand.pose_hand[hand_rows].copy()
    poses = np.concatenate(
        (root_orient, pose_body, pose_jaw, pose_eye, pose_hand), axis=1
    )
    if poses.shape != (common.size, 165):
        raise RuntimeError(f"internal error: merged poses has shape {poses.shape}")

    differences = np.diff(common)
    gaps = differences[differences > 1]
    stats = MergeStats(
        body_frames=int(body.frame_index.size),
        hand_frames=int(hand.frame_index.size),
        common_frames=int(common.size),
        body_only_frames=int(body.frame_index.size - common.size),
        hand_only_frames=int(hand.frame_index.size - common.size),
        first_frame=int(common[0]),
        last_frame=int(common[-1]),
        gap_count=int(gaps.size),
        missing_frame_count=int(np.sum(gaps - 1, dtype=np.int64)),
    )
    motion = MergedStageiiMotion(
        gender=body.gender.lower(),
        mocap_frame_rate=body.mocap_frame_rate,
        trans=trans.astype(np.float64, copy=False),
        poses=poses.astype(np.float64, copy=False),
        betas=hand.betas.astype(np.float64, copy=True),
        root_orient=root_orient.astype(np.float64, copy=False),
        pose_body=pose_body.astype(np.float64, copy=False),
        pose_hand=pose_hand.astype(np.float64, copy=False),
        pose_jaw=pose_jaw.astype(np.float64, copy=False),
        pose_eye=pose_eye.astype(np.float64, copy=False),
    )
    return motion, stats


def save_merged_motion(
    motion: MergedStageiiMotion, output_path: Path, overwrite: bool = False
) -> None:
    """Atomically write a merged motion with the exact reference key set."""

    if output_path.suffix.lower() != ".npz":
        raise MergeError("output file must use the .npz extension")
    if not output_path.parent.is_dir():
        raise MergeError(f"output directory does not exist: {output_path.parent}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)

    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary_path.open("wb") as stream:
            np.savez_compressed(stream, **motion.archive_fields())
        if output_path.exists() and not overwrite:
            raise FileExistsError(output_path)
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def merge_files(
    body_path: Path,
    hand_path: Path,
    output_path: Path,
    overwrite: bool = False,
    axis_up: str = "y",
) -> MergeStats:
    body = load_body_recording(body_path)
    hand = load_hand_recording(hand_path)
    motion, stats = merge_recordings(body, hand, axis_up=axis_up)
    save_merged_motion(motion, output_path, overwrite=overwrite)
    return stats


def _input_npz(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file not found: {path}")
    if path.suffix.lower() != ".npz":
        raise argparse.ArgumentTypeError("input must be a .npz file")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge body SMPL and hand SMPL-X NPZ recordings by exact shared "
            "frame_index and write an OK_B_stageii-compatible NPZ."
        )
    )
    parser.add_argument("body_file", type=_input_npz, help="body SMPL recording")
    parser.add_argument("hand_file", type=_input_npz, help="hand SMPL-X recording")
    parser.add_argument(
        "--output", required=True, type=Path, help="destination merged .npz file"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace the output file if it exists"
    )
    parser.add_argument(
        "--axis-up",
        choices=tuple(AXIS_UP_SYSTEMS),
        default="y",
        help=(
            "output up axis: y=SMPL X-left/Y-up/Z-forward (default), "
            "z=X-right/Y-forward/Z-up, x=X-up/Y-forward/Z-left"
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="logging verbosity (default: INFO)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        stats = merge_files(
            args.body_file,
            args.hand_file,
            args.output,
            overwrite=args.overwrite,
            axis_up=args.axis_up,
        )
    except (MergeError, FileExistsError, OSError) as exc:
        parser.error(str(exc))

    LOGGER.info(
        "Merged body=%d hand=%d common=%d body_only=%d hand_only=%d",
        stats.body_frames,
        stats.hand_frames,
        stats.common_frames,
        stats.body_only_frames,
        stats.hand_only_frames,
    )
    LOGGER.info(
        "Common frame range=%d..%d gaps=%d missing=%d",
        stats.first_frame,
        stats.last_frame,
        stats.gap_count,
        stats.missing_frame_count,
    )
    LOGGER.info("Wrote %s (axis-up=%s)", args.output, args.axis_up)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
