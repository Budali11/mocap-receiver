"""Convert SMPL NPZ motion files between coordinate systems.

This module transforms the global pose orientation and root translation of every
frame so the same motion can be consumed by downstream systems that expect
different world-axis conventions (Y-up / Z-up / X-up).

The body pose (local joint rotations in the SMPL kinematic tree) is left
unchanged — only the global frame is affected.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .converter import matrix_to_rotvec
from .visualize import SmplMotion, axis_angle_to_matrix, load_smpl_motion


LOGGER = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Coordinate-system rotation matrices.
#
# Each matrix rotates vectors FROM the canonical SMPL frame
#      SMPL:  X-left   Y-up    Z-forward
# INTO the target frame.  All three are proper rotations (determinant +1).
# ---------------------------------------------------------------------------

# Z-up (common robotics / MuJoCo convention): X-right  Y-forward  Z-up
_Z_UP: FloatArray = np.array(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)

# X-up:  X-up  Y-forward  Z-left
_X_UP: FloatArray = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)

COORDINATE_TRANSFORMS: dict[str, FloatArray] = {
    "y_up": np.eye(3, dtype=np.float64),
    "z_up": _Z_UP,
    "x_up": _X_UP,
}


def _batch_axis_angle_to_matrix(aa: FloatArray) -> FloatArray:
    """Convert (N, 3) axis-angle vectors to (N, 3, 3) rotation matrices."""
    n = aa.shape[0]
    matrices = np.empty((n, 3, 3), dtype=np.float64)
    for i in range(n):
        matrices[i] = axis_angle_to_matrix(aa[i])
    return matrices


def _batch_matrix_to_axis_angle(matrices: FloatArray) -> FloatArray:
    """Convert (N, 3, 3) rotation matrices to (N, 3) axis-angle vectors."""
    n = matrices.shape[0]
    vectors = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        vectors[i] = matrix_to_rotvec(matrices[i])
    return vectors


def convert_frame(
    pose: FloatArray,
    trans: FloatArray,
    coord_matrix: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Rotate the *global* pose and translation of one frame into a new frame.

    Parameters
    ----------
    pose : shape (72,)
        Flat SMPL pose: ``[global_orient (3), body_pose (69)]``, axis-angle.
    trans : shape (3,)
        Root translation in the source coordinate system.
    coord_matrix : shape (3, 3)
        Orthonormal rotation from the source system to the target system.

    Returns
    -------
    new_pose : shape (72,)
    new_trans : shape (3,)
    """
    global_orient = pose[:3]
    body_pose = pose[3:]

    # Global orientation: R_new = M @ R_old @ M^T
    r_old = axis_angle_to_matrix(global_orient)
    r_new = coord_matrix @ r_old @ coord_matrix.T
    new_global_orient = matrix_to_rotvec(r_new)

    # Translation: t_new = M @ t_old
    new_trans = coord_matrix @ trans

    new_pose = np.concatenate((new_global_orient, body_pose))
    return new_pose, new_trans


def convert_motion(
    motion: SmplMotion,
    target_systems: Sequence[str] | None = None,
) -> dict[str, SmplMotion]:
    """Produce a copy of *motion* in each requested coordinate system.

    Parameters
    ----------
    motion : SmplMotion
    target_systems : sequence of str, optional
        Subset of ``{"y_up", "z_up", "x_up"}``.  Defaults to all three.

    Returns
    -------
    dict mapping system name → new SmplMotion.
    """
    if target_systems is None:
        target_systems = list(COORDINATE_TRANSFORMS)

    results: dict[str, SmplMotion] = {}
    n_frames = motion.poses.shape[0]

    for name in target_systems:
        coord = COORDINATE_TRANSFORMS[name]
        new_poses = np.empty_like(motion.poses)  # (N, 72)
        new_trans = np.empty_like(motion.trans)  # (N, 3)

        for i in range(n_frames):
            new_poses[i], new_trans[i] = convert_frame(
                motion.poses[i], motion.trans[i], coord
            )

        results[name] = SmplMotion(
            poses=new_poses,
            trans=new_trans,
            betas=motion.betas.copy(),
            gender=motion.gender,
            mocap_framerate=motion.mocap_framerate,
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _existing_npz(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file not found: {path}")
    if path.suffix.lower() != ".npz":
        raise argparse.ArgumentTypeError("input must be a .npz file")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert SMPL NPZ motion to Y-up / Z-up / X-up coordinate systems."
    )
    parser.add_argument(
        "motion_file",
        type=_existing_npz,
        help="SMPL .npz motion file to convert",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="parent directory for timestamped output (default: output/)",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=list(COORDINATE_TRANSFORMS),
        default=list(COORDINATE_TRANSFORMS),
        help="coordinate systems to generate (default: y_up z_up x_up)",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="logging verbosity (default: INFO)",
    )
    return parser


def save_motion_npz(motion: SmplMotion, path: Path) -> None:
    """Write a SmplMotion to an NPZ file in the standard layout."""
    np.savez(
        path,
        poses=motion.poses.astype(np.float32),
        trans=motion.trans.astype(np.float32),
        betas=motion.betas.astype(np.float32),
        gender=np.asarray(motion.gender),
        mocap_framerate=np.float32(motion.mocap_framerate),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    motion = load_smpl_motion(args.motion_file)
    LOGGER.info(
        "Loaded %s: %d frames, %.1f Hz, %s gender",
        args.motion_file.name,
        motion.frame_count,
        motion.mocap_framerate,
        motion.gender,
    )

    converted = convert_motion(motion, target_systems=args.systems)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = args.motion_file.stem

    for name, conv_motion in converted.items():
        out_dir = args.output_dir / timestamp / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}.npz"
        save_motion_npz(conv_motion, out_path)
        LOGGER.info("Wrote %s → %s", name, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
