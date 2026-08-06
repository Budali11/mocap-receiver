"""Pure NumPy loader and linear blend skinning for legacy SMPL v1.1 models."""

from __future__ import annotations

import copyreg
from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from numpy.typing import NDArray

from .converter import SMPL_PARENTS


DEFAULT_SMPL_MODEL_DIR = Path(
    r"D:\Users\budali11\Documents\phibotics\SMPL_python_v.1.1.0\smpl\models"
)

MODEL_FILENAMES = {
    "female": "basicmodel_f_lbs_10_207_0_v1.1.0.pkl",
    "male": "basicmodel_m_lbs_10_207_0_v1.1.0.pkl",
    "neutral": "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
}

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int32]


class SmplModelError(ValueError):
    """Raised when a legacy SMPL model is absent or incompatible."""


class _LegacyState:
    """Inert target for trusted legacy SciPy/Chumpy pickle state."""

    def __new__(cls, *args: object, **kwargs: object) -> "_LegacyState":
        return object.__new__(cls)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.state = state


class _RestrictedLegacyUnpickler(pickle.Unpickler):
    """Load the known SMPL pickle globals without importing old dependencies."""

    _SAFE_GLOBALS = {
        ("copy_reg", "_reconstructor"): copyreg._reconstructor,
        ("__builtin__", "object"): object,
        ("__builtin__", "set"): set,
        ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
    }

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("scipy.sparse") and name in {"csc_matrix", "csr_matrix"}:
            return _LegacyState
        if module == "chumpy.ch" and name == "Ch":
            return _LegacyState
        try:
            return self._SAFE_GLOBALS[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(
                f"unsupported global in legacy SMPL pickle: {module}.{name}"
            ) from exc


def _load_legacy_pickle(stream: BinaryIO) -> dict[str, Any]:
    loaded = _RestrictedLegacyUnpickler(stream, encoding="latin1").load()
    if not isinstance(loaded, dict):
        raise SmplModelError("SMPL model pickle must contain a dictionary")
    return loaded


def _chumpy_array(value: Any, field: str) -> NDArray[Any]:
    if isinstance(value, np.ndarray):
        return value
    state = getattr(value, "state", None)
    if isinstance(state, dict) and isinstance(state.get("x"), np.ndarray):
        return state["x"]
    raise SmplModelError(f"SMPL model field {field} is not a supported array")


def _sparse_to_dense(value: Any, field: str) -> FloatArray:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float32)
    state = getattr(value, "state", None)
    if not isinstance(state, dict):
        raise SmplModelError(f"SMPL model field {field} is not a supported sparse matrix")
    try:
        shape = tuple(int(item) for item in state["_shape"])
        indptr = np.asarray(state["indptr"], dtype=np.int64)
        indices = np.asarray(state["indices"], dtype=np.int64)
        data = np.asarray(state["data"], dtype=np.float32)
        sparse_format = state["format"]
    except (KeyError, TypeError, ValueError) as exc:
        raise SmplModelError(f"SMPL sparse field {field} has invalid state") from exc
    if len(shape) != 2:
        raise SmplModelError(f"SMPL sparse field {field} must be two-dimensional")

    dense = np.zeros(shape, dtype=np.float32)
    if sparse_format == "csc":
        if indptr.shape != (shape[1] + 1,):
            raise SmplModelError(f"SMPL sparse field {field} has invalid CSC pointers")
        for column in range(shape[1]):
            start, end = int(indptr[column]), int(indptr[column + 1])
            dense[indices[start:end], column] = data[start:end]
    elif sparse_format == "csr":
        if indptr.shape != (shape[0] + 1,):
            raise SmplModelError(f"SMPL sparse field {field} has invalid CSR pointers")
        for row in range(shape[0]):
            start, end = int(indptr[row]), int(indptr[row + 1])
            dense[row, indices[start:end]] = data[start:end]
    else:
        raise SmplModelError(f"SMPL sparse field {field} uses unsupported format {sparse_format!r}")
    return dense


def resolve_model_path(model_dir: Path, gender: str) -> Path:
    try:
        filename = MODEL_FILENAMES[gender]
    except KeyError as exc:
        raise SmplModelError(f"unsupported SMPL gender {gender!r}") from exc
    path = model_dir / filename
    if not path.is_file():
        raise SmplModelError(f"SMPL {gender} model not found: {path}")
    return path


def normalize_gender(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {"f": "female", "female": "female", "m": "male", "male": "male", "n": "neutral", "neutral": "neutral"}
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise SmplModelError(f"unsupported SMPL gender {value!r}") from exc


def _rotation_matrices(pose: NDArray[np.float64]) -> FloatArray:
    vectors = pose.reshape(24, 3)
    matrices = np.empty((24, 3, 3), dtype=np.float32)
    identity = np.eye(3, dtype=np.float64)
    for index, vector in enumerate(vectors):
        angle = float(np.linalg.norm(vector))
        if angle < 1e-10:
            x, y, z = vector
            skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
            matrix = identity + skew + 0.5 * (skew @ skew)
        else:
            x, y, z = vector / angle
            skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
            matrix = identity + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
        matrices[index] = matrix
    return matrices


@dataclass(frozen=True)
class SmplMeshFrame:
    vertices: FloatArray
    joints: FloatArray


@dataclass(frozen=True)
class SmplModel:
    """Immutable SMPL model arrays shared by one or more shaped bodies."""

    gender: str
    v_template: FloatArray
    shapedirs: FloatArray
    posedirs: FloatArray
    joint_regressor: FloatArray
    weights: FloatArray
    faces: IntArray
    parents: tuple[int, ...]

    @classmethod
    def load(cls, path: Path, gender: str) -> "SmplModel":
        try:
            with path.open("rb") as stream:
                data = _load_legacy_pickle(stream)
        except (OSError, pickle.UnpicklingError, EOFError) as exc:
            raise SmplModelError(f"cannot load SMPL model {path}: {exc}") from exc

        required = {
            "v_template",
            "shapedirs",
            "posedirs",
            "J_regressor",
            "weights",
            "f",
            "kintree_table",
        }
        missing = required - set(data)
        if missing:
            raise SmplModelError(f"SMPL model is missing fields: {', '.join(sorted(missing))}")

        v_template = np.asarray(data["v_template"], dtype=np.float32)
        shapedirs_full = _chumpy_array(data["shapedirs"], "shapedirs")
        shapedirs = np.asarray(shapedirs_full[:, :, :10], dtype=np.float32).copy()
        posedirs = np.asarray(data["posedirs"], dtype=np.float32).reshape(-1, 207).copy()
        joint_regressor = _sparse_to_dense(data["J_regressor"], "J_regressor")
        weights = np.asarray(data["weights"], dtype=np.float32)
        faces = np.asarray(data["f"], dtype=np.int32)
        kintree = np.asarray(data["kintree_table"])

        if v_template.shape != (6890, 3):
            raise SmplModelError(f"v_template must have shape (6890, 3), got {v_template.shape}")
        if shapedirs.shape != (6890, 3, 10):
            raise SmplModelError(f"shapedirs must have shape (6890, 3, 10), got {shapedirs.shape}")
        if posedirs.shape != (6890 * 3, 207):
            raise SmplModelError(f"posedirs has incompatible shape {posedirs.shape}")
        if joint_regressor.shape != (24, 6890):
            raise SmplModelError(f"J_regressor must have shape (24, 6890), got {joint_regressor.shape}")
        if weights.shape != (6890, 24):
            raise SmplModelError(f"weights must have shape (6890, 24), got {weights.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise SmplModelError(f"faces must have shape (F, 3), got {faces.shape}")
        if kintree.shape != (2, 24):
            raise SmplModelError(f"kintree_table must have shape (2, 24), got {kintree.shape}")

        joint_ids = [int(value) for value in kintree[1]]
        id_to_index = {joint_id: index for index, joint_id in enumerate(joint_ids)}
        parents = [-1]
        for index in range(1, 24):
            try:
                parents.append(id_to_index[int(kintree[0, index])])
            except KeyError as exc:
                raise SmplModelError(f"invalid parent ID for SMPL joint {index}") from exc
        if tuple(parents) != SMPL_PARENTS:
            raise SmplModelError(f"model kinematic tree does not match classic SMPL: {parents}")

        return cls(
            gender=normalize_gender(gender),
            v_template=v_template.copy(),
            shapedirs=shapedirs,
            posedirs=posedirs,
            joint_regressor=joint_regressor,
            weights=weights.copy(),
            faces=faces.copy(),
            parents=tuple(parents),
        )

    @classmethod
    def from_directory(cls, model_dir: Path, gender: str) -> "SmplModel":
        normalized_gender = normalize_gender(gender)
        return cls.load(resolve_model_path(model_dir, normalized_gender), normalized_gender)

    def with_betas(self, betas: NDArray[Any]) -> "SmplBody":
        beta_array = np.asarray(betas, dtype=np.float32)
        if beta_array.shape != (10,) or not np.all(np.isfinite(beta_array)):
            raise SmplModelError("betas must have shape (10,) and contain finite values")
        shaped_vertices = self.v_template + np.tensordot(
            self.shapedirs, beta_array, axes=([2], [0])
        )
        rest_joints = self.joint_regressor @ shaped_vertices
        return SmplBody(
            model=self,
            betas=beta_array.copy(),
            shaped_vertices=np.asarray(shaped_vertices, dtype=np.float32),
            rest_joints=np.asarray(rest_joints, dtype=np.float32),
        )


@dataclass(frozen=True)
class SmplBody:
    """A fixed-shape SMPL body that can be posed repeatedly in real time."""

    model: SmplModel
    betas: FloatArray
    shaped_vertices: FloatArray
    rest_joints: FloatArray

    def pose(
        self,
        pose: NDArray[Any],
        translation: NDArray[Any],
        translation_is_pelvis: bool = True,
    ) -> SmplMeshFrame:
        pose_array = np.asarray(pose, dtype=np.float64)
        translation_array = np.asarray(translation, dtype=np.float32)
        if pose_array.shape == (24, 3):
            pose_array = pose_array.reshape(72)
        if pose_array.shape != (72,) or not np.all(np.isfinite(pose_array)):
            raise SmplModelError("pose must have shape (72,) and contain finite values")
        if translation_array.shape != (3,) or not np.all(np.isfinite(translation_array)):
            raise SmplModelError("translation must have shape (3,) and contain finite values")

        rotations = _rotation_matrices(pose_array)
        pose_feature = (rotations[1:] - np.eye(3, dtype=np.float32)).reshape(207)
        pose_offsets = (self.model.posedirs @ pose_feature).reshape(6890, 3)
        posed_vertices = self.shaped_vertices + pose_offsets

        global_transforms = np.zeros((24, 4, 4), dtype=np.float32)
        global_transforms[:, 3, 3] = 1.0
        global_transforms[0, :3, :3] = rotations[0]
        global_transforms[0, :3, 3] = self.rest_joints[0]
        for index in range(1, 24):
            parent = self.model.parents[index]
            local = np.eye(4, dtype=np.float32)
            local[:3, :3] = rotations[index]
            local[:3, 3] = self.rest_joints[index] - self.rest_joints[parent]
            global_transforms[index] = global_transforms[parent] @ local

        skinning_transforms = global_transforms.copy()
        for index in range(24):
            skinning_transforms[index, :3, 3] -= (
                global_transforms[index, :3, :3] @ self.rest_joints[index]
            )
        per_vertex_transforms = np.einsum(
            "vj,jab->vab", self.model.weights, skinning_transforms, optimize=True
        )
        homogeneous = np.concatenate(
            (posed_vertices, np.ones((6890, 1), dtype=np.float32)), axis=1
        )
        vertices = np.einsum(
            "vab,vb->va", per_vertex_transforms, homogeneous, optimize=True
        )[:, :3]
        joints = global_transforms[:, :3, 3].copy()

        shift = (
            translation_array - joints[0]
            if translation_is_pelvis
            else translation_array
        )
        vertices += shift
        joints += shift
        return SmplMeshFrame(
            vertices=np.asarray(vertices, dtype=np.float32),
            joints=np.asarray(joints, dtype=np.float32),
        )
