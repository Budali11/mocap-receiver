"""Visualize recorded SMPL ``.npz`` motion as a 24-joint 3D animation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .converter import (
    SMPL_JOINT_NAMES,
    SMPL_PARENTS,
    SMPL_SOURCE_INDICES,
    SOURCE_INITIAL_POSITIONS,
    WS_GEO_TO_SMPL,
)
from .smpl_model import (
    DEFAULT_SMPL_MODEL_DIR,
    SmplBody,
    SmplModel,
    SmplModelError,
    normalize_gender,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SmplMotion:
    poses: FloatArray
    trans: FloatArray
    betas: FloatArray
    gender: str
    mocap_framerate: float

    @property
    def frame_count(self) -> int:
        return int(self.poses.shape[0])


def load_smpl_motion(path: Path) -> SmplMotion:
    """Load and strictly validate a classic 24-joint SMPL motion file."""

    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load SMPL motion file {path}: {exc}") from exc

    with archive:
        missing = {"poses", "trans"} - set(archive.files)
        if missing:
            raise ValueError(f"SMPL motion is missing fields: {', '.join(sorted(missing))}")
        poses = np.asarray(archive["poses"], dtype=np.float64)
        if poses.ndim == 3 and poses.shape[1:] == (24, 3):
            poses = poses.reshape(poses.shape[0], 72)
        trans = np.asarray(archive["trans"], dtype=np.float64)
        betas = np.asarray(archive["betas"], dtype=np.float64) if "betas" in archive else np.zeros(10)
        raw_gender = archive["gender"] if "gender" in archive else np.asarray("neutral")
        raw_framerate = archive["mocap_framerate"] if "mocap_framerate" in archive else np.asarray(60.0)

    if poses.ndim != 2 or poses.shape[1] != 72:
        raise ValueError(f"poses must have shape (N, 72) or (N, 24, 3), got {poses.shape}")
    if trans.shape != (poses.shape[0], 3):
        raise ValueError(f"trans must have shape ({poses.shape[0]}, 3), got {trans.shape}")
    if betas.ndim != 1 or betas.size < 10:
        raise ValueError(f"betas must contain at least 10 values, got shape {betas.shape}")
    if poses.shape[0] == 0:
        raise ValueError("SMPL motion contains no frames")
    if not np.all(np.isfinite(poses)) or not np.all(np.isfinite(trans)):
        raise ValueError("poses and trans must contain only finite values")
    if not np.all(np.isfinite(betas)):
        raise ValueError("betas must contain only finite values")

    try:
        framerate = float(np.asarray(raw_framerate).item())
    except (TypeError, ValueError) as exc:
        raise ValueError("mocap_framerate must be a scalar number") from exc
    if not math.isfinite(framerate) or framerate <= 0.0:
        raise ValueError("mocap_framerate must be a positive finite number")

    gender_value = np.asarray(raw_gender).item()
    if isinstance(gender_value, bytes):
        gender = gender_value.decode("utf-8", errors="replace")
    else:
        gender = str(gender_value)
    return SmplMotion(
        poses=poses,
        trans=trans,
        betas=betas[:10],
        gender=gender,
        mocap_framerate=framerate,
    )


def axis_angle_to_matrix(rotation_vector: FloatArray) -> FloatArray:
    """Convert one 3D axis-angle vector to a rotation matrix."""

    vector = np.asarray(rotation_vector, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("rotation_vector must contain three finite values")
    angle = float(np.linalg.norm(vector))
    if angle < 1e-10:
        x, y, z = vector
        skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        return np.eye(3) + skew + 0.5 * (skew @ skew)
    x, y, z = vector / angle
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _sample_polyline(points: FloatArray, distances: FloatArray, target: float) -> FloatArray:
    if target >= distances[-1]:
        return points[-1].copy()
    upper = int(np.searchsorted(distances, target, side="right"))
    lower = max(0, upper - 1)
    segment = float(distances[upper] - distances[lower])
    amount = 0.0 if segment <= 1e-12 else (target - distances[lower]) / segment
    return (1.0 - amount) * points[lower] + amount * points[upper]


def canonical_smpl_rest_positions() -> FloatArray:
    """Return an approximate 24-joint SMPL rest skeleton in SMPL coordinates.

    The skeleton is derived from the VD Suit rest skeleton because a licensed
    SMPL model is intentionally not bundled with this project. Betas therefore
    do not alter this lightweight preview skeleton.
    """

    source_positions = np.asarray(SOURCE_INITIAL_POSITIONS, dtype=np.float64)
    target_positions = np.empty((24, 3), dtype=np.float64)
    for target_index, source_index in enumerate(SMPL_SOURCE_INDICES):
        if source_index is not None:
            target_positions[target_index] = WS_GEO_TO_SMPL @ source_positions[source_index]

    spine_indices = np.asarray((0, 9, 10, 11, 12))
    source_spine = source_positions[spine_indices]
    distances = np.concatenate(
        (np.array([0.0]), np.cumsum(np.linalg.norm(np.diff(source_spine, axis=0), axis=1)))
    )
    total = float(distances[-1])
    for target_index, fraction in zip((3, 6, 9), (1.0 / 3.0, 2.0 / 3.0, 1.0)):
        sampled = _sample_polyline(source_spine, distances, total * fraction)
        target_positions[target_index] = WS_GEO_TO_SMPL @ sampled

    # VD Suit ends at the wrist. Add short terminal hand bones for the two
    # terminal joints present in classic SMPL.
    target_positions[22] = target_positions[20] + np.array([0.10, 0.0, 0.0])
    target_positions[23] = target_positions[21] + np.array([-0.10, 0.0, 0.0])
    target_positions -= target_positions[0]
    return target_positions


def canonical_smpl_rest_offsets() -> FloatArray:
    positions = canonical_smpl_rest_positions()
    offsets = np.zeros_like(positions)
    for index in range(1, 24):
        offsets[index] = positions[index] - positions[SMPL_PARENTS[index]]
    return offsets


def forward_kinematics(
    pose: FloatArray,
    translation: FloatArray,
    rest_offsets: FloatArray | None = None,
) -> FloatArray:
    """Compute 24 world-space joint positions for one SMPL pose frame."""

    pose_array = np.asarray(pose, dtype=np.float64)
    translation_array = np.asarray(translation, dtype=np.float64)
    if pose_array.shape == (24, 3):
        pose_array = pose_array.reshape(72)
    if pose_array.shape != (72,) or not np.all(np.isfinite(pose_array)):
        raise ValueError("pose must have shape (72,) or (24, 3) with finite values")
    if translation_array.shape != (3,) or not np.all(np.isfinite(translation_array)):
        raise ValueError("translation must have shape (3,) with finite values")
    offsets = canonical_smpl_rest_offsets() if rest_offsets is None else np.asarray(rest_offsets)
    if offsets.shape != (24, 3) or not np.all(np.isfinite(offsets)):
        raise ValueError("rest_offsets must have shape (24, 3) with finite values")

    local_rotations = np.stack(
        [axis_angle_to_matrix(vector) for vector in pose_array.reshape(24, 3)]
    )
    global_rotations = np.empty((24, 3, 3), dtype=np.float64)
    joint_positions = np.empty((24, 3), dtype=np.float64)
    global_rotations[0] = local_rotations[0]
    joint_positions[0] = translation_array
    for index in range(1, 24):
        parent = SMPL_PARENTS[index]
        joint_positions[index] = joint_positions[parent] + global_rotations[parent] @ offsets[index]
        global_rotations[index] = global_rotations[parent] @ local_rotations[index]
    return joint_positions


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return result


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play a classic SMPL .npz motion file.")
    parser.add_argument("motion_file", type=Path, help="SMPL .npz motion file")
    parser.add_argument("--start-frame", type=int, default=0, help="first frame (default: 0)")
    parser.add_argument("--end-frame", type=int, help="exclusive last frame (default: end)")
    parser.add_argument("--stride", type=_positive_int, default=1, help="display every Nth frame")
    parser.add_argument("--speed", type=_positive_float, default=1.0, help="playback speed multiplier")
    parser.add_argument(
        "--view-radius",
        type=_positive_float,
        default=1.2,
        help="camera half-width in meters (default: 1.2)",
    )
    parser.add_argument("--fixed-camera", action="store_true", help="show the entire root trajectory")
    parser.add_argument("--no-loop", action="store_true", help="stop instead of looping at the end")
    parser.add_argument("--save", type=Path, help="save animation as .gif or .mp4 instead of opening a window")
    parser.add_argument("--dpi", type=_positive_int, default=120, help="saved animation DPI")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_SMPL_MODEL_DIR,
        help=f"directory containing SMPL v1.1 model files (default: {DEFAULT_SMPL_MODEL_DIR})",
    )
    parser.add_argument(
        "--gender",
        choices=("auto", "neutral", "male", "female"),
        default="auto",
        help="SMPL model gender; auto uses the NPZ metadata (default: auto)",
    )
    parser.add_argument(
        "--skeleton",
        action="store_true",
        help="render the lightweight skeleton instead of the SMPL mesh",
    )
    parser.add_argument(
        "--mesh-face-step",
        type=_positive_int,
        default=1,
        help="draw every Nth mesh triangle; larger values render faster (default: 1)",
    )
    parser.add_argument(
        "--backend",
        choices=("gpu", "matplotlib"),
        default="gpu",
        help="interactive rendering backend (default: gpu)",
    )
    parser.add_argument(
        "--window-width", type=_positive_int, default=1000, help="GPU window width"
    )
    parser.add_argument(
        "--window-height", type=_positive_int, default=800, help="GPU window height"
    )
    parser.add_argument("--no-vsync", action="store_true", help="disable GPU vertical sync")
    return parser


def _set_follow_camera(ax: object, root: FloatArray, radius: float) -> None:
    # Plot axes are [SMPL X, SMPL Z, SMPL Y] so world Y appears vertical.
    ax.set_xlim(root[0] - radius, root[0] + radius)
    ax.set_ylim(root[2] - radius, root[2] + radius)
    ax.set_zlim(root[1] - radius, root[1] + radius)


def _set_fixed_camera(ax: object, translations: FloatArray, radius: float) -> None:
    minimum = translations.min(axis=0)
    maximum = translations.max(axis=0)
    center = (minimum + maximum) * 0.5
    half_extent = np.maximum((maximum - minimum) * 0.5 + radius, radius)
    ax.set_xlim(center[0] - half_extent[0], center[0] + half_extent[0])
    ax.set_ylim(center[2] - half_extent[2], center[2] + half_extent[2])
    ax.set_zlim(center[1] - half_extent[1], center[1] + half_extent[1])


def play_motion(
    motion: SmplMotion,
    frame_indices: NDArray[np.int64],
    speed: float,
    view_radius: float,
    fixed_camera: bool,
    repeat: bool,
    save_path: Path | None,
    dpi: int,
    smpl_body: SmplBody | None = None,
    mesh_face_step: int = 1,
) -> None:
    """Create, display, or save the Matplotlib animation."""

    try:
        import matplotlib
        if save_path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required; install it with: "
            "python -m pip install -e .[visualization]"
        ) from exc

    offsets = canonical_smpl_rest_offsets()
    bones = [(parent, index) for index, parent in enumerate(SMPL_PARENTS) if parent >= 0]
    bone_colors = []
    joint_colors = []
    for name in SMPL_JOINT_NAMES:
        if name.startswith("left_"):
            color = "#2878d0"
        elif name.startswith("right_"):
            color = "#dc4c4c"
        else:
            color = "#333333"
        joint_colors.append(color)
    for _parent, child in bones:
        bone_colors.append(joint_colors[child])

    figure = plt.figure(figsize=(8, 8))
    ax = figure.add_subplot(111, projection="3d")
    ax.set_xlabel("SMPL X (left)")
    ax.set_ylabel("SMPL Z (forward)")
    ax.set_zlabel("SMPL Y (up)")
    ax.set_box_aspect((1.0, 1.0, 1.3))
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    first_frame = int(frame_indices[0])
    initial_mesh = (
        smpl_body.pose(motion.poses[first_frame], motion.trans[first_frame])
        if smpl_body is not None
        else None
    )
    initial_joints = (
        initial_mesh.joints
        if initial_mesh is not None
        else forward_kinematics(motion.poses[first_frame], motion.trans[first_frame], offsets)
    )
    plotted = initial_joints[:, [0, 2, 1]]
    bone_collection = Line3DCollection(
        [[plotted[parent], plotted[child]] for parent, child in bones],
        colors=bone_colors,
        linewidths=1.5 if initial_mesh is not None else 3.0,
        alpha=0.65 if initial_mesh is not None else 1.0,
    )
    ax.add_collection3d(bone_collection)
    scatter = ax.scatter(
        plotted[:, 0],
        plotted[:, 1],
        plotted[:, 2],
        c=joint_colors,
        s=9 if initial_mesh is not None else 22,
        depthshade=True,
    )
    mesh_collection = None
    mesh_faces = None
    if initial_mesh is not None and smpl_body is not None:
        mesh_faces = smpl_body.model.faces[::mesh_face_step]
        mesh_points = initial_mesh.vertices[:, [0, 2, 1]]
        mesh_collection = Poly3DCollection(
            mesh_points[mesh_faces],
            facecolor="#d7a07d",
            edgecolor="none",
            alpha=0.92,
        )
        ax.add_collection3d(mesh_collection)
    title = ax.set_title("")
    if fixed_camera:
        _set_fixed_camera(ax, motion.trans[frame_indices], view_radius)
    else:
        _set_follow_camera(ax, initial_joints[0], view_radius)

    def update(animation_index: int) -> tuple[object, ...]:
        frame = int(frame_indices[animation_index])
        mesh_frame = (
            smpl_body.pose(motion.poses[frame], motion.trans[frame])
            if smpl_body is not None
            else None
        )
        joints = (
            mesh_frame.joints
            if mesh_frame is not None
            else forward_kinematics(motion.poses[frame], motion.trans[frame], offsets)
        )
        plot_points = joints[:, [0, 2, 1]]
        bone_collection.set_segments(
            [[plot_points[parent], plot_points[child]] for parent, child in bones]
        )
        scatter._offsets3d = (plot_points[:, 0], plot_points[:, 1], plot_points[:, 2])
        if mesh_collection is not None and mesh_faces is not None and mesh_frame is not None:
            mesh_points = mesh_frame.vertices[:, [0, 2, 1]]
            mesh_collection.set_verts(mesh_points[mesh_faces])
        title.set_text(
            f"SMPL {'mesh' if smpl_body is not None else 'skeleton'} | "
            f"frame {frame}/{motion.frame_count - 1} | "
            f"{frame / motion.mocap_framerate:.2f} s"
        )
        if not fixed_camera:
            _set_follow_camera(ax, joints[0], view_radius)
        artists: list[object] = [bone_collection, scatter, title]
        if mesh_collection is not None:
            artists.append(mesh_collection)
        return tuple(artists)

    frame_step = 1 if frame_indices.size < 2 else int(frame_indices[1] - frame_indices[0])
    interval_ms = 1000.0 * frame_step / (motion.mocap_framerate * speed)
    animation = FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=interval_ms,
        repeat=repeat,
        blit=False,
    )

    if save_path is not None:
        output_fps = motion.mocap_framerate * speed / frame_step
        writer = "pillow" if save_path.suffix.lower() == ".gif" else "ffmpeg"
        animation.save(str(save_path), writer=writer, fps=output_fps, dpi=dpi)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        motion = load_smpl_motion(args.motion_file)
    except ValueError as exc:
        parser.error(str(exc))

    end_frame = motion.frame_count if args.end_frame is None else args.end_frame
    if args.start_frame < 0 or args.start_frame >= motion.frame_count:
        parser.error(f"--start-frame must be between 0 and {motion.frame_count - 1}")
    if end_frame <= args.start_frame or end_frame > motion.frame_count:
        parser.error(
            f"--end-frame must be greater than {args.start_frame} and at most {motion.frame_count}"
        )
    if args.save is not None and args.save.suffix.lower() not in {".gif", ".mp4"}:
        parser.error("--save path must end in .gif or .mp4")

    frame_indices = np.arange(args.start_frame, end_frame, args.stride, dtype=np.int64)
    duration = (end_frame - args.start_frame) / motion.mocap_framerate
    print(
        f"Loaded {motion.frame_count} frames, {motion.mocap_framerate:g} Hz, "
        f"{duration:.2f} s, gender={motion.gender}"
    )
    smpl_body = None
    if not args.skeleton:
        try:
            gender = normalize_gender(motion.gender) if args.gender == "auto" else args.gender
            model = SmplModel.from_directory(args.model_dir, gender)
            smpl_body = model.with_betas(motion.betas)
            print(
                f"Loaded {gender} SMPL mesh: {model.v_template.shape[0]} vertices, "
                f"{model.faces.shape[0]} faces"
            )
        except SmplModelError as exc:
            parser.error(f"{exc}; use --skeleton for model-free preview")
    backend = args.backend
    if args.save is not None and backend == "gpu":
        print("--save uses the Matplotlib export backend")
        backend = "matplotlib"
    if args.skeleton and backend == "gpu":
        print("--skeleton uses the Matplotlib backend")
        backend = "matplotlib"
    try:
        if backend == "gpu":
            if smpl_body is None:
                raise RuntimeError("GPU mesh playback requires an SMPL model")
            from .gpu_visualize import play_motion_gpu

            print(
                "GPU controls: Space=pause, Left/Right=step, +/-=speed, "
                "mouse drag=orbit, wheel=zoom, W=wireframe, R=reset, Esc=exit"
            )
            gpu_info = play_motion_gpu(
                motion=motion,
                frame_indices=frame_indices,
                smpl_body=smpl_body,
                speed=args.speed,
                view_radius=args.view_radius,
                fixed_camera=args.fixed_camera,
                repeat=not args.no_loop,
                width=args.window_width,
                height=args.window_height,
                vsync=not args.no_vsync,
            )
            print(
                f"OpenGL renderer: {gpu_info['renderer']} | "
                f"vendor={gpu_info['vendor']} | version={gpu_info['opengl_version']}"
            )
        else:
            play_motion(
                motion=motion,
                frame_indices=frame_indices,
                speed=args.speed,
                view_radius=args.view_radius,
                fixed_camera=args.fixed_camera,
                repeat=not args.no_loop,
                save_path=args.save,
                dpi=args.dpi,
                smpl_body=smpl_body,
                mesh_face_step=args.mesh_face_step,
            )
    except KeyboardInterrupt:
        print("Stopped by user")
    except (RuntimeError, OSError, ValueError, SmplModelError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
