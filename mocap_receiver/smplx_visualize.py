"""Play merged Stage-II SMPL-X motions with the official LBS implementation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .smpl_model import SmplMeshFrame


FloatArray = NDArray[np.float64]
DEFAULT_SMPLX_MODEL_DIR = Path(__file__).resolve().parent.parent / "smplx_model"
SMPLX_MODEL_FILENAMES = {
    "female": "SMPLX_FEMALE.npz",
    "male": "SMPLX_MALE.npz",
    "neutral": "SMPLX_NEUTRAL.npz",
}


@dataclass(frozen=True)
class SmplxMotion:
    poses: FloatArray
    trans: FloatArray
    betas: FloatArray
    gender: str
    mocap_framerate: float

    @property
    def frame_count(self) -> int:
        return int(self.poses.shape[0])


def _scalar_string(value: Any, field: str) -> str:
    try:
        item = np.asarray(value).item()
    except ValueError as exc:
        raise ValueError(f"{field} must be a scalar string") from exc
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return str(item)


def load_smplx_motion(path: Path) -> SmplxMotion:
    """Load and validate the merged ``OK_B_stageii``-style motion layout."""

    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load SMPL-X motion file {path}: {exc}") from exc

    with archive:
        required = {
            "poses",
            "trans",
            "betas",
            "root_orient",
            "pose_body",
            "pose_hand",
            "pose_jaw",
            "pose_eye",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                f"SMPL-X motion is missing fields: {', '.join(sorted(missing))}"
            )
        poses = np.asarray(archive["poses"], dtype=np.float64)
        trans = np.asarray(archive["trans"], dtype=np.float64)
        betas = np.asarray(archive["betas"], dtype=np.float64)
        root_orient = np.asarray(archive["root_orient"], dtype=np.float64)
        pose_body = np.asarray(archive["pose_body"], dtype=np.float64)
        pose_jaw = np.asarray(archive["pose_jaw"], dtype=np.float64)
        pose_eye = np.asarray(archive["pose_eye"], dtype=np.float64)
        pose_hand = np.asarray(archive["pose_hand"], dtype=np.float64)
        raw_gender = archive["gender"] if "gender" in archive else np.asarray("neutral")
        if "mocap_frame_rate" in archive:
            raw_framerate = archive["mocap_frame_rate"]
        elif "mocap_framerate" in archive:
            raw_framerate = archive["mocap_framerate"]
        else:
            raw_framerate = np.asarray(60.0)
        surface_model_type = (
            _scalar_string(archive["surface_model_type"], "surface_model_type")
            if "surface_model_type" in archive
            else "smplx"
        )

    if surface_model_type.strip().lower() != "smplx":
        raise ValueError(
            f"surface_model_type must be 'smplx', got {surface_model_type!r}"
        )
    if poses.ndim != 2 or poses.shape[1] != 165:
        raise ValueError(f"poses must have shape (N, 165), got {poses.shape}")
    frame_count = poses.shape[0]
    expected_shapes = {
        "trans": (frame_count, 3),
        "root_orient": (frame_count, 3),
        "pose_body": (frame_count, 63),
        "pose_jaw": (frame_count, 3),
        "pose_eye": (frame_count, 6),
        "pose_hand": (frame_count, 90),
    }
    arrays = {
        "trans": trans,
        "root_orient": root_orient,
        "pose_body": pose_body,
        "pose_jaw": pose_jaw,
        "pose_eye": pose_eye,
        "pose_hand": pose_hand,
    }
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {arrays[name].shape}")
    if frame_count == 0:
        raise ValueError("SMPL-X motion contains no frames")
    if betas.ndim != 1 or betas.size == 0:
        raise ValueError(f"betas must be a non-empty one-dimensional array, got {betas.shape}")
    if not all(np.all(np.isfinite(array)) for array in (*arrays.values(), poses, betas)):
        raise ValueError("SMPL-X numeric fields must contain only finite values")

    rebuilt_poses = np.concatenate(
        (root_orient, pose_body, pose_jaw, pose_eye, pose_hand), axis=1
    )
    if not np.allclose(poses, rebuilt_poses, atol=1e-9, rtol=1e-9):
        raise ValueError(
            "poses must equal root_orient + pose_body + pose_jaw + pose_eye + pose_hand"
        )
    try:
        framerate = float(np.asarray(raw_framerate).item())
    except (TypeError, ValueError) as exc:
        raise ValueError("mocap_frame_rate must be a scalar number") from exc
    if not math.isfinite(framerate) or framerate <= 0.0:
        raise ValueError("mocap_frame_rate must be a positive finite number")

    return SmplxMotion(
        poses=poses,
        trans=trans,
        betas=betas,
        gender=_scalar_string(raw_gender, "gender"),
        mocap_framerate=framerate,
    )


def normalize_smplx_gender(value: str) -> str:
    aliases = {
        "f": "female",
        "female": "female",
        "m": "male",
        "male": "male",
        "n": "neutral",
        "neutral": "neutral",
    }
    try:
        return aliases[value.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported SMPL-X gender {value!r}") from exc


def resolve_torch_device(requested: str) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "SMPL-X playback requires PyTorch and smplx in this environment"
        ) from exc
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this PyTorch build has no CUDA support; "
            "install a CUDA-enabled PyTorch build or use --device cpu"
        )
    return requested


class SmplxBody:
    """A fixed-shape SMPL-X body backed by the official ``smplx.lbs`` function."""

    def __init__(
        self,
        model_path: Path,
        betas: FloatArray,
        device: str = "auto",
    ) -> None:
        try:
            import torch
            from smplx.lbs import lbs
        except ImportError as exc:
            raise RuntimeError(
                "SMPL-X playback requires the official smplx package and PyTorch"
            ) from exc

        self.device = resolve_torch_device(device)
        self._torch = torch
        self._lbs = lbs
        try:
            archive = np.load(model_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot load SMPL-X model {model_path}: {exc}") from exc
        with archive:
            required = {
                "v_template",
                "shapedirs",
                "posedirs",
                "J_regressor",
                "weights",
                "kintree_table",
                "f",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(
                    f"SMPL-X model is missing fields: {', '.join(sorted(missing))}"
                )
            v_template = np.asarray(archive["v_template"], dtype=np.float32)
            shapedirs = np.asarray(archive["shapedirs"], dtype=np.float32)
            raw_posedirs = np.asarray(archive["posedirs"], dtype=np.float32)
            joint_regressor = np.asarray(archive["J_regressor"], dtype=np.float32)
            weights = np.asarray(archive["weights"], dtype=np.float32)
            kintree_table = np.asarray(archive["kintree_table"], dtype=np.int64)
            faces = np.asarray(archive["f"], dtype=np.int32)

        vertex_count = v_template.shape[0] if v_template.ndim == 2 else -1
        if v_template.shape != (vertex_count, 3) or vertex_count <= 0:
            raise ValueError(f"v_template must have shape (V, 3), got {v_template.shape}")
        joint_count = joint_regressor.shape[0] if joint_regressor.ndim == 2 else -1
        if joint_count != 55 or joint_regressor.shape != (55, vertex_count):
            raise ValueError(
                f"J_regressor must have shape (55, {vertex_count}), got {joint_regressor.shape}"
            )
        if shapedirs.ndim != 3 or shapedirs.shape[:2] != (vertex_count, 3):
            raise ValueError(
                f"shapedirs must have shape ({vertex_count}, 3, S), got {shapedirs.shape}"
            )
        beta_array = np.asarray(betas, dtype=np.float32)
        if beta_array.ndim != 1 or beta_array.size > shapedirs.shape[2]:
            raise ValueError(
                f"model provides {shapedirs.shape[2]} shape directions but got {beta_array.size} betas"
            )
        expected_pose_basis = (joint_count - 1) * 9
        if raw_posedirs.shape != (vertex_count, 3, expected_pose_basis):
            raise ValueError(
                "posedirs must have shape "
                f"({vertex_count}, 3, {expected_pose_basis}), got {raw_posedirs.shape}"
            )
        if weights.shape != (vertex_count, joint_count):
            raise ValueError(
                f"weights must have shape ({vertex_count}, {joint_count}), got {weights.shape}"
            )
        if kintree_table.shape != (2, joint_count):
            raise ValueError(
                f"kintree_table must have shape (2, {joint_count}), got {kintree_table.shape}"
            )
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"f must have shape (F, 3), got {faces.shape}")

        parents = kintree_table[0].copy()
        parents[0] = -1
        if np.any(parents[1:] < 0) or np.any(parents[1:] >= joint_count):
            raise ValueError("SMPL-X kinematic tree contains invalid parent indices")

        tensor = lambda value, dtype=torch.float32: torch.as_tensor(
            np.ascontiguousarray(value), dtype=dtype, device=self.device
        )
        self.faces = faces
        # Existing OpenGL renderer accesses ``body.model.faces``.
        self.model = self
        self.parents = parents
        self.model_path = model_path
        self._betas = tensor(beta_array.reshape(1, -1))
        self._v_template = tensor(v_template)
        self._shapedirs = tensor(shapedirs[:, :, : beta_array.size])
        posedirs = raw_posedirs.reshape(-1, expected_pose_basis).T
        self._posedirs = tensor(posedirs)
        self._joint_regressor = tensor(joint_regressor)
        self._parents = tensor(parents, dtype=torch.long)
        self._weights = tensor(weights)

    @property
    def vertex_count(self) -> int:
        return int(self._v_template.shape[0])

    def pose(self, pose: Any, translation: Any) -> SmplMeshFrame:
        """Evaluate one frame and place the model pelvis at ``translation``."""

        pose_array = np.asarray(pose, dtype=np.float32)
        translation_array = np.asarray(translation, dtype=np.float32)
        if pose_array.shape != (165,) or not np.all(np.isfinite(pose_array)):
            raise ValueError("pose must have shape (165,) and contain finite values")
        if translation_array.shape != (3,) or not np.all(np.isfinite(translation_array)):
            raise ValueError("translation must have shape (3,) and contain finite values")

        torch = self._torch
        with torch.inference_mode():
            pose_tensor = torch.as_tensor(
                np.ascontiguousarray(pose_array.reshape(1, 165)),
                dtype=torch.float32,
                device=self.device,
            )
            vertices, joints = self._lbs(
                self._betas,
                pose_tensor,
                self._v_template,
                self._shapedirs,
                self._posedirs,
                self._joint_regressor,
                self._parents,
                self._weights,
                pose2rot=True,
            )
            target_pelvis = torch.as_tensor(
                translation_array, dtype=torch.float32, device=self.device
            ).reshape(1, 1, 3)
            shift = target_pelvis - joints[:, 0:1]
            vertices = vertices + shift
            joints = joints + shift
            vertex_array = vertices[0].detach().cpu().numpy().copy()
            joint_array = joints[0].detach().cpu().numpy().copy()
        return SmplMeshFrame(vertices=vertex_array, joints=joint_array)


def load_smplx_body(
    model_dir: Path,
    gender: str,
    betas: FloatArray,
    device: str,
) -> SmplxBody:
    normalized_gender = normalize_smplx_gender(gender)
    model_path = model_dir / SMPLX_MODEL_FILENAMES[normalized_gender]
    if not model_path.is_file():
        raise ValueError(f"SMPL-X {normalized_gender} model not found: {model_path}")
    return SmplxBody(model_path=model_path, betas=betas, device=device)


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
    parser = argparse.ArgumentParser(
        description="Play a merged OK_B_stageii-compatible SMPL-X .npz motion."
    )
    parser.add_argument("motion_file", type=Path, help="merged SMPL-X .npz motion")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_SMPLX_MODEL_DIR,
        help=f"directory containing SMPLX_*.npz models (default: {DEFAULT_SMPLX_MODEL_DIR})",
    )
    parser.add_argument(
        "--gender",
        choices=("auto", "neutral", "male", "female"),
        default="auto",
        help="model gender; auto uses NPZ metadata (default: auto)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="PyTorch LBS device (default: auto)",
    )
    parser.add_argument(
        "--backend",
        choices=("gpu", "matplotlib"),
        default="gpu",
        help="interactive rendering backend (default: gpu)",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="first frame")
    parser.add_argument("--end-frame", type=int, help="exclusive last frame")
    parser.add_argument("--stride", type=_positive_int, default=1, help="display every Nth frame")
    parser.add_argument("--speed", type=_positive_float, default=1.0, help="playback speed")
    parser.add_argument(
        "--view-radius", type=_positive_float, default=1.2, help="camera half-width in meters"
    )
    parser.add_argument("--fixed-camera", action="store_true", help="show the root trajectory")
    parser.add_argument("--no-loop", action="store_true", help="stop at the last frame")
    parser.add_argument(
        "--save", type=Path, help="save a .gif or .mp4 with the Matplotlib backend"
    )
    parser.add_argument("--dpi", type=_positive_int, default=120, help="saved animation DPI")
    parser.add_argument(
        "--mesh-face-step",
        type=_positive_int,
        default=4,
        help="Matplotlib only: draw every Nth triangle (default: 4)",
    )
    parser.add_argument("--window-width", type=_positive_int, default=1000)
    parser.add_argument("--window-height", type=_positive_int, default=800)
    parser.add_argument("--no-vsync", action="store_true", help="disable GPU vertical sync")
    return parser


def _set_follow_camera(ax: Any, pelvis: FloatArray, radius: float) -> None:
    ax.set_xlim(pelvis[0] - radius, pelvis[0] + radius)
    ax.set_ylim(pelvis[2] - radius, pelvis[2] + radius)
    ax.set_zlim(pelvis[1] - radius, pelvis[1] + radius)


def _set_fixed_camera(ax: Any, translations: FloatArray, radius: float) -> None:
    minimum = translations.min(axis=0)
    maximum = translations.max(axis=0)
    center = (minimum + maximum) * 0.5
    half_extent = np.maximum((maximum - minimum) * 0.5 + radius, radius)
    ax.set_xlim(center[0] - half_extent[0], center[0] + half_extent[0])
    ax.set_ylim(center[2] - half_extent[2], center[2] + half_extent[2])
    ax.set_zlim(center[1] - half_extent[1], center[1] + half_extent[1])


def play_motion_matplotlib(
    motion: SmplxMotion,
    frame_indices: NDArray[np.int64],
    body: SmplxBody,
    speed: float,
    view_radius: float,
    fixed_camera: bool,
    repeat: bool,
    save_path: Path | None,
    dpi: int,
    mesh_face_step: int,
) -> None:
    try:
        import matplotlib

        if save_path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required; install with: python -m pip install -e .[visualization]"
        ) from exc

    figure = plt.figure(figsize=(8, 8))
    ax = figure.add_subplot(111, projection="3d")
    ax.set_xlabel("SMPL-X X (left)")
    ax.set_ylabel("SMPL-X Z (forward)")
    ax.set_zlabel("SMPL-X Y (up)")
    ax.set_box_aspect((1.0, 1.0, 1.3))
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    first = int(frame_indices[0])
    first_mesh = body.pose(motion.poses[first], motion.trans[first])
    faces = body.faces[::mesh_face_step]
    plotted_vertices = first_mesh.vertices[:, [0, 2, 1]]
    mesh_collection = Poly3DCollection(
        plotted_vertices[faces], facecolor="#d7a07d", edgecolor="none", alpha=0.96
    )
    ax.add_collection3d(mesh_collection)
    title = ax.set_title("")
    if fixed_camera:
        _set_fixed_camera(ax, motion.trans[frame_indices], view_radius)
    else:
        _set_follow_camera(ax, first_mesh.joints[0], view_radius)

    def update(animation_index: int) -> tuple[Any, ...]:
        frame = int(frame_indices[animation_index])
        mesh = body.pose(motion.poses[frame], motion.trans[frame])
        mesh_collection.set_verts(mesh.vertices[:, [0, 2, 1]][faces])
        title.set_text(
            f"SMPL-X mesh | frame {frame}/{motion.frame_count - 1} | "
            f"{frame / motion.mocap_framerate:.2f} s"
        )
        if not fixed_camera:
            _set_follow_camera(ax, mesh.joints[0], view_radius)
        return mesh_collection, title

    frame_step = 1 if frame_indices.size < 2 else int(frame_indices[1] - frame_indices[0])
    animation = FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=1000.0 * frame_step / (motion.mocap_framerate * speed),
        repeat=repeat,
        blit=False,
    )
    if save_path is None:
        plt.show()
    else:
        writer = "pillow" if save_path.suffix.lower() == ".gif" else "ffmpeg"
        animation.save(
            str(save_path),
            writer=writer,
            fps=motion.mocap_framerate * speed / frame_step,
            dpi=dpi,
        )
        print(f"Saved visualization to {save_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        motion = load_smplx_motion(args.motion_file)
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

    gender = motion.gender if args.gender == "auto" else args.gender
    try:
        gender = normalize_smplx_gender(gender)
        body = load_smplx_body(
            model_dir=args.model_dir,
            gender=gender,
            betas=motion.betas,
            device=args.device,
        )
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    frame_indices = np.arange(args.start_frame, end_frame, args.stride, dtype=np.int64)
    print(
        f"Loaded {motion.frame_count} frames, {motion.mocap_framerate:g} Hz, "
        f"gender={gender}; SMPL-X {body.vertex_count} vertices on {body.device}"
    )
    if body.device == "cpu" and args.backend == "gpu":
        print(
            "OpenGL rendering uses the GPU, but SMPL-X skinning uses CPU because "
            "the installed PyTorch build has no CUDA support."
        )

    backend = "matplotlib" if args.save is not None else args.backend
    try:
        if backend == "gpu":
            from .gpu_visualize import play_motion_gpu

            print(
                "GPU controls: Space=pause, Left/Right=step, +/-=speed, "
                "mouse drag=orbit, wheel=zoom, W=wireframe, R=reset, Esc=exit"
            )
            play_motion_gpu(
                motion=motion,
                frame_indices=frame_indices,
                smpl_body=body,
                speed=args.speed,
                view_radius=args.view_radius,
                fixed_camera=args.fixed_camera,
                repeat=not args.no_loop,
                width=args.window_width,
                height=args.window_height,
                vsync=not args.no_vsync,
                model_label="SMPL-X",
            )
        else:
            play_motion_matplotlib(
                motion=motion,
                frame_indices=frame_indices,
                body=body,
                speed=args.speed,
                view_radius=args.view_radius,
                fixed_camera=args.fixed_camera,
                repeat=not args.no_loop,
                save_path=args.save,
                dpi=args.dpi,
                mesh_face_step=args.mesh_face_step,
            )
    except RuntimeError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
