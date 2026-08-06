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
        self._pose_chunks: list[NDArray[np.float32]] = []
        self._trans_chunks: list[NDArray[np.float32]] = []
        self._pose_chunk = np.empty((chunk_size, 72), dtype=np.float32)
        self._trans_chunk = np.empty((chunk_size, 3), dtype=np.float32)
        self._chunk_length = 0
        self._frame_count = 0
        self._betas: NDArray[np.float32] | None = None
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
        if self._betas is None:
            self._betas = betas.copy()
        elif not np.allclose(self._betas, betas, atol=1e-6, rtol=1e-6):
            raise ValueError("betas must remain constant throughout one SMPL motion")

        if self._chunk_length == self._chunk_size:
            self._seal_current_chunk()
        self._pose_chunk[self._chunk_length, :3] = global_orient
        self._pose_chunk[self._chunk_length, 3:] = body_pose
        self._trans_chunk[self._chunk_length] = translation
        self._chunk_length += 1
        self._frame_count += 1

    def close(self) -> None:
        """Finalize and atomically publish the compressed SMPL motion file."""

        if self._closed:
            return

        poses, translations = self._final_arrays()
        betas = self._betas if self._betas is not None else np.zeros(10, dtype=np.float32)
        temporary_path = self.path.with_name(self.path.name + ".tmp")
        try:
            with temporary_path.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    poses=poses,
                    trans=translations,
                    betas=betas,
                    gender=np.asarray("neutral"),
                    mocap_framerate=np.asarray(self.mocap_framerate, dtype=np.float32),
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
    def _array(value: Any, shape: tuple[int, ...], field: str) -> NDArray[np.float32]:
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must contain numeric values") from exc
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{field} must have shape {shape} and contain finite values")
        return array

    def _seal_current_chunk(self) -> None:
        self._pose_chunks.append(self._pose_chunk)
        self._trans_chunks.append(self._trans_chunk)
        self._pose_chunk = np.empty((self._chunk_size, 72), dtype=np.float32)
        self._trans_chunk = np.empty((self._chunk_size, 3), dtype=np.float32)
        self._chunk_length = 0

    def _final_arrays(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        pose_parts = [*self._pose_chunks]
        trans_parts = [*self._trans_chunks]
        if self._chunk_length:
            pose_parts.append(self._pose_chunk[: self._chunk_length].copy())
            trans_parts.append(self._trans_chunk[: self._chunk_length].copy())
        if not pose_parts:
            return (
                np.empty((0, 72), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
            )
        if len(pose_parts) == 1:
            return pose_parts[0], trans_parts[0]
        return np.concatenate(pose_parts), np.concatenate(trans_parts)
