"""Writers for standard SMPL motion sequence files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


class SmplNpzRecorder:
    """Collect frames and save an AMASS-style SMPL motion ``.npz`` file.

    SMPL uses 24 axis-angle joint rotations, so ``poses`` has shape (N, 72).
    The recorder stores frames in reasonably sized NumPy chunks to avoid the
    per-frame overhead of keeping thousands of small Python objects alive.
    """

    def __init__(
        self,
        path: Path,
        mocap_framerate: float = 60.0,
        overwrite: bool = False,
        chunk_size: int = 4096,
    ) -> None:
        if not np.isfinite(mocap_framerate) or mocap_framerate <= 0.0:
            raise ValueError("mocap_framerate must be a positive finite number")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if not path.parent.is_dir():
            raise FileNotFoundError(f"output directory does not exist: {path.parent}")
        if path.exists() and not overwrite:
            raise FileExistsError(path)

        self.path = path
        self.mocap_framerate = float(mocap_framerate)
        self.overwrite = overwrite
        self._chunk_size = chunk_size
        self._pose_chunks: list[NDArray[np.float64]] = []
        self._trans_chunks: list[NDArray[np.float64]] = []
        self._index_chunks: list[NDArray[np.int64]] = []
        self._pose_chunk = np.empty((chunk_size, 72), dtype=np.float64)
        self._trans_chunk = np.empty((chunk_size, 3), dtype=np.float64)
        self._index_chunk = np.empty(chunk_size, dtype=np.int64)
        self._chunk_length = 0
        self._frame_count = 0
        self._betas: NDArray[np.float64] | None = None
        self._closed = False

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def record_encoded(self, payload: bytes) -> None:
        """Decode and record one converter-produced SMPL JSON frame."""

        message = json.loads(payload)
        if not isinstance(message, Mapping):
            raise ValueError("encoded SMPL frame must contain a JSON object")
        self.record(message)

    def record(self, message: Mapping[str, Any]) -> None:
        """Record one converted SMPL frame."""

        if self._closed:
            raise RuntimeError("cannot record after the SMPL motion file has been closed")
        if message.get("type") != "smpl_frame":
            raise ValueError("SMPL motion recorder only accepts smpl_frame messages")

        global_orient = self._array(message.get("global_orient"), (3,), "global_orient")
        body_pose = self._array(message.get("body_pose"), (69,), "body_pose")
        translation = self._array(message.get("transl"), (3,), "transl")
        betas = self._array(message.get("betas"), (10,), "betas")
        frame_index = message.get("frame_index")
        if not isinstance(frame_index, int) or isinstance(frame_index, bool):
            raise ValueError("frame_index must be an integer")
        if self._betas is None:
            self._betas = betas.copy()
        elif not np.allclose(self._betas, betas, atol=1e-6, rtol=1e-6):
            raise ValueError("betas must remain constant throughout one SMPL motion")

        if self._chunk_length == self._chunk_size:
            self._seal_current_chunk()
        self._pose_chunk[self._chunk_length, :3] = global_orient
        self._pose_chunk[self._chunk_length, 3:] = body_pose
        self._trans_chunk[self._chunk_length] = translation
        self._index_chunk[self._chunk_length] = frame_index
        self._chunk_length += 1
        self._frame_count += 1

    def close(self) -> None:
        """Finalize and atomically publish the compressed SMPL motion file."""

        if self._closed:
            return

        poses, translations, frame_indices = self._final_arrays()
        betas = self._betas if self._betas is not None else np.zeros(10, dtype=np.float64)
        temporary_path = self.path.with_name(self.path.name + ".tmp")
        try:
            with temporary_path.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    poses=poses,
                    trans=translations,
                    frame_index=frame_indices,
                    betas=betas,
                    gender=np.asarray("neutral"),
                    mocap_framerate=np.asarray(self.mocap_framerate, dtype=np.float64),
                )
            if self.path.exists() and not self.overwrite:
                raise FileExistsError(self.path)
            temporary_path.replace(self.path)
        except BaseException:
            # Keep no partially written file under the requested final name.
            temporary_path.unlink(missing_ok=True)
            raise
        self._closed = True

    def __enter__(self) -> "SmplNpzRecorder":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @staticmethod
    def _array(value: Any, shape: tuple[int, ...], field: str) -> NDArray[np.float64]:
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must contain numeric values") from exc
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{field} must have shape {shape} and contain finite values")
        return array

    def _seal_current_chunk(self) -> None:
        self._pose_chunks.append(self._pose_chunk)
        self._trans_chunks.append(self._trans_chunk)
        self._index_chunks.append(self._index_chunk)
        self._pose_chunk = np.empty((self._chunk_size, 72), dtype=np.float64)
        self._trans_chunk = np.empty((self._chunk_size, 3), dtype=np.float64)
        self._index_chunk = np.empty(self._chunk_size, dtype=np.int64)
        self._chunk_length = 0

    def _final_arrays(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        pose_parts = [*self._pose_chunks]
        trans_parts = [*self._trans_chunks]
        index_parts = [*self._index_chunks]
        if self._chunk_length:
            pose_parts.append(self._pose_chunk[: self._chunk_length].copy())
            trans_parts.append(self._trans_chunk[: self._chunk_length].copy())
            index_parts.append(self._index_chunk[: self._chunk_length].copy())
        if not pose_parts:
            return (
                np.empty((0, 72), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty(0, dtype=np.int64),
            )
        if len(pose_parts) == 1:
            return pose_parts[0], trans_parts[0], index_parts[0]
        return (
            np.concatenate(pose_parts),
            np.concatenate(trans_parts),
            np.concatenate(index_parts),
        )


class SmplxHandNpzRecorder:
    """Collect ``smplx_hand_frame`` messages into a Stage-II-style NPZ.

    Each UDP message represents one frame and carries flat arrays.  On close,
    the recorder stacks them into the sequence shapes used by SMPL-X motion
    files, notably ``poses`` (N, 165) and ``pose_hand`` (N, 90).
    """

    def __init__(
        self,
        path: Path,
        mocap_framerate: float = 60.0,
        overwrite: bool = False,
        chunk_size: int = 4096,
    ) -> None:
        if not np.isfinite(mocap_framerate) or mocap_framerate <= 0.0:
            raise ValueError("mocap_framerate must be a positive finite number")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if not path.parent.is_dir():
            raise FileNotFoundError(f"output directory does not exist: {path.parent}")
        if path.exists() and not overwrite:
            raise FileExistsError(path)

        self.path = path
        self.mocap_framerate = float(mocap_framerate)
        self.overwrite = overwrite
        self._chunk_size = chunk_size
        self._pose_chunks: list[NDArray[np.float64]] = []
        self._trans_chunks: list[NDArray[np.float64]] = []
        self._position_chunks: list[NDArray[np.float64]] = []
        self._orient_chunks: list[NDArray[np.float64]] = []
        self._index_chunks: list[NDArray[np.int64]] = []
        self._pose_chunk = np.empty((chunk_size, 165), dtype=np.float64)
        self._trans_chunk = np.empty((chunk_size, 3), dtype=np.float64)
        self._position_chunk = np.empty((chunk_size, 2, 3), dtype=np.float64)
        self._orient_chunk = np.empty((chunk_size, 2, 3), dtype=np.float64)
        self._index_chunk = np.empty(chunk_size, dtype=np.int64)
        self._chunk_length = 0
        self._frame_count = 0
        self._betas: NDArray[np.float64] | None = None
        self._closed = False

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def record_encoded(self, payload: bytes) -> None:
        """Decode and record one converter-produced SMPL-X JSON frame."""

        message = json.loads(payload)
        if not isinstance(message, Mapping):
            raise ValueError("encoded SMPL-X frame must contain a JSON object")
        self.record(message)

    def record(self, message: Mapping[str, Any]) -> None:
        """Validate and append one converted hand frame."""

        if self._closed:
            raise RuntimeError("cannot record after the SMPL-X motion file has been closed")
        if message.get("type") != "smplx_hand_frame":
            raise ValueError("SMPL-X hand recorder only accepts smplx_hand_frame messages")

        poses = self._array(message.get("poses"), (165,), "poses")
        translation = self._array(message.get("trans"), (3,), "trans")
        root_orient = self._array(message.get("root_orient"), (3,), "root_orient")
        pose_body = self._array(message.get("pose_body"), (63,), "pose_body")
        pose_jaw = self._array(message.get("pose_jaw"), (3,), "pose_jaw")
        pose_eye = self._array(message.get("pose_eye"), (6,), "pose_eye")
        pose_hand = self._array(message.get("pose_hand"), (90,), "pose_hand")
        betas = self._array(message.get("betas"), (16,), "betas")
        hand_positions = self._array(
            message.get("hand_positions"), (2, 3), "hand_positions"
        )
        hand_global_orient = self._array(
            message.get("hand_global_orient"), (2, 3), "hand_global_orient"
        )
        rebuilt_poses = np.concatenate(
            (root_orient, pose_body, pose_jaw, pose_eye, pose_hand)
        )
        if not np.allclose(poses, rebuilt_poses, atol=1e-6, rtol=1e-6):
            raise ValueError(
                "poses must equal root_orient + pose_body + pose_jaw + pose_eye + pose_hand"
            )

        frame_index = message.get("frame_index")
        if not isinstance(frame_index, int) or isinstance(frame_index, bool):
            raise ValueError("frame_index must be an integer")
        if self._betas is None:
            self._betas = betas.copy()
        elif not np.allclose(self._betas, betas, atol=1e-6, rtol=1e-6):
            raise ValueError("betas must remain constant throughout one SMPL-X motion")

        if self._chunk_length == self._chunk_size:
            self._seal_current_chunk()
        index = self._chunk_length
        self._pose_chunk[index] = poses
        self._trans_chunk[index] = translation
        self._position_chunk[index] = hand_positions
        self._orient_chunk[index] = hand_global_orient
        self._index_chunk[index] = frame_index
        self._chunk_length += 1
        self._frame_count += 1

    def close(self) -> None:
        """Finalize and atomically publish the compressed SMPL-X NPZ file."""

        if self._closed:
            return

        poses, translations, positions, orientations, frame_indices = self._final_arrays()
        betas = self._betas if self._betas is not None else np.zeros(16, dtype=np.float64)
        temporary_path = self.path.with_name(self.path.name + ".tmp")
        try:
            with temporary_path.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    gender=np.asarray("neutral"),
                    surface_model_type=np.asarray("smplx"),
                    mocap_frame_rate=np.asarray(self.mocap_framerate, dtype=np.float64),
                    mocap_time_length=np.asarray(
                        self._frame_count / self.mocap_framerate, dtype=np.float64
                    ),
                    trans=translations,
                    poses=poses,
                    betas=betas,
                    num_betas=np.asarray(16, dtype=np.int64),
                    root_orient=poses[:, 0:3],
                    pose_body=poses[:, 3:66],
                    pose_jaw=poses[:, 66:69],
                    pose_eye=poses[:, 69:75],
                    pose_hand=poses[:, 75:165],
                    frame_index=frame_indices,
                    hand_side_order=np.asarray(("left", "right")),
                    hand_positions=positions,
                    hand_global_orient=orientations,
                )
            if self.path.exists() and not self.overwrite:
                raise FileExistsError(self.path)
            temporary_path.replace(self.path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        self._closed = True

    def __enter__(self) -> "SmplxHandNpzRecorder":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @staticmethod
    def _array(value: Any, shape: tuple[int, ...], field: str) -> NDArray[np.float64]:
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must contain numeric values") from exc
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{field} must have shape {shape} and contain finite values")
        return array

    def _seal_current_chunk(self) -> None:
        self._pose_chunks.append(self._pose_chunk)
        self._trans_chunks.append(self._trans_chunk)
        self._position_chunks.append(self._position_chunk)
        self._orient_chunks.append(self._orient_chunk)
        self._index_chunks.append(self._index_chunk)
        self._pose_chunk = np.empty((self._chunk_size, 165), dtype=np.float64)
        self._trans_chunk = np.empty((self._chunk_size, 3), dtype=np.float64)
        self._position_chunk = np.empty((self._chunk_size, 2, 3), dtype=np.float64)
        self._orient_chunk = np.empty((self._chunk_size, 2, 3), dtype=np.float64)
        self._index_chunk = np.empty(self._chunk_size, dtype=np.int64)
        self._chunk_length = 0

    def _final_arrays(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.int64],
    ]:
        pose_parts = [*self._pose_chunks]
        trans_parts = [*self._trans_chunks]
        position_parts = [*self._position_chunks]
        orient_parts = [*self._orient_chunks]
        index_parts = [*self._index_chunks]
        if self._chunk_length:
            end = self._chunk_length
            pose_parts.append(self._pose_chunk[:end].copy())
            trans_parts.append(self._trans_chunk[:end].copy())
            position_parts.append(self._position_chunk[:end].copy())
            orient_parts.append(self._orient_chunk[:end].copy())
            index_parts.append(self._index_chunk[:end].copy())
        if not pose_parts:
            return (
                np.empty((0, 165), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 2, 3), dtype=np.float64),
                np.empty((0, 2, 3), dtype=np.float64),
                np.empty(0, dtype=np.int64),
            )
        if len(pose_parts) == 1:
            return (
                pose_parts[0],
                trans_parts[0],
                position_parts[0],
                orient_parts[0],
                index_parts[0],
            )
        return (
            np.concatenate(pose_parts),
            np.concatenate(trans_parts),
            np.concatenate(position_parts),
            np.concatenate(orient_parts),
            np.concatenate(index_parts),
        )
