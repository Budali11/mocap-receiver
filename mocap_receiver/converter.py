"""Skeleton validation and VD Suit-to-SMPL pose retargeting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class ConversionError(ValueError):
    """Raised when a dynamic frame cannot be converted safely."""


class SkeletonError(ConversionError):
    """Raised when a skeleton message is incompatible with the converter."""


SOURCE_JOINT_SPECS: tuple[tuple[str, int], ...] = (
    ("Hips", -1),
    ("RightUpperLeg", 0),
    ("RightLowerLeg", 1),
    ("RightFoot", 2),
    ("RightToe", 3),
    ("LeftUpperLeg", 0),
    ("LeftLowerLeg", 5),
    ("LeftFoot", 6),
    ("LeftToe", 7),
    ("Spine", 0),
    ("Spine1", 9),
    ("Spine2", 10),
    ("Spine3", 11),
    ("Neck", 12),
    ("Head", 13),
    ("RightShoulder", 12),
    ("RightUpperArm", 15),
    ("RightLowerArm", 16),
    ("RightHand", 17),
    ("LeftShoulder", 12),
    ("LeftUpperArm", 19),
    ("LeftLowerArm", 20),
    ("LeftHand", 21),
)


SOURCE_INITIAL_POSITIONS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 1.11000001),
    (0.104999997, 0.0, 1.02199996),
    (0.104999997, 0.0, 0.565999985),
    (0.104999997, 0.0, 0.0989999995),
    (0.104999997, 0.120999999, 0.0189999994),
    (-0.104999997, 0.0, 1.02199996),
    (-0.104999997, 0.0, 0.565999985),
    (-0.104999997, 0.0, 0.0989999995),
    (-0.104999997, 0.120999999, 0.0189999994),
    (0.0, 0.0, 1.20700002),
    (0.0, 0.0, 1.31900001),
    (0.0, 0.0, 1.44000006),
    (0.0, 0.0120000001, 1.55700004),
    (0.0, 0.0, 1.67700005),
    (0.0, 0.0, 1.77699995),
    (0.0489999987, 0.0, 1.597),
    (0.204999998, 0.0, 1.597),
    (0.463999987, 0.0, 1.597),
    (0.748000026, 0.0, 1.597),
    (-0.0489999987, 0.0, 1.597),
    (-0.204999998, 0.0, 1.597),
    (-0.463999987, 0.0, 1.597),
    (-0.748000026, 0.0, 1.597),
)


SMPL_JOINT_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)


SMPL_PARENTS: tuple[int, ...] = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
)


# Source index for each directly mapped SMPL joint. None marks a resampled spine.
SMPL_SOURCE_INDICES: tuple[int | None, ...] = (
    0,
    5,
    1,
    None,
    6,
    2,
    None,
    7,
    3,
    None,
    8,
    4,
    13,
    19,
    15,
    14,
    20,
    16,
    21,
    17,
    22,
    18,
    22,
    18,
)


# WS_Geo [right, forward, up] -> SMPL [left, up, forward].
WS_GEO_TO_SMPL: FloatArray = np.array(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class SkeletonDefinition:
    """Validated source skeleton data in source joint index order."""

    positions: FloatArray
    offsets: FloatArray

    @property
    def spine_distances(self) -> FloatArray:
        indices = (0, 9, 10, 11, 12)
        points = self.positions[np.asarray(indices)]
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        return np.concatenate((np.array([0.0]), np.cumsum(lengths)))


def _vector3(value: Any, field: str, error_type: type[ConversionError]) -> FloatArray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise error_type(f"{field} must contain three numeric values") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise error_type(f"{field} must contain three finite numeric values")
    return vector


def _quaternion(value: Any, field: str) -> FloatArray:
    try:
        quaternion = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ConversionError(f"{field} must contain four numeric values") from exc
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ConversionError(f"{field} must contain four finite numeric values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ConversionError(f"{field} must not be a zero quaternion")
    return quaternion / norm


def builtin_skeleton_message() -> dict[str, Any]:
    """Return the example skeleton as a protocol-compatible dictionary."""

    joints: list[dict[str, Any]] = []
    for index, ((name, parent), position_tuple) in enumerate(
        zip(SOURCE_JOINT_SPECS, SOURCE_INITIAL_POSITIONS)
    ):
        position = np.asarray(position_tuple, dtype=np.float64)
        if parent == -1:
            offset = position
        else:
            offset = position - np.asarray(SOURCE_INITIAL_POSITIONS[parent])
        joints.append(
            {
                "index": index,
                "name": name,
                "parent_index": parent,
                "initial_position": position.tolist(),
                "offset": offset.tolist(),
            }
        )
    return {
        "type": "skeleton",
        "version": 1,
        "joint_count": len(SOURCE_JOINT_SPECS),
        "coordinate_system": "WS_Geo",
        "position_unit": "m",
        "quaternion_order": "wxyz",
        "joints": joints,
    }


def validate_skeleton(message: Mapping[str, Any]) -> SkeletonDefinition:
    """Validate protocol metadata and the fixed 23-joint topology."""

    expected_metadata = {
        "type": "skeleton",
        "version": 1,
        "joint_count": len(SOURCE_JOINT_SPECS),
        "coordinate_system": "WS_Geo",
        "position_unit": "m",
        "quaternion_order": "wxyz",
    }
    for field, expected in expected_metadata.items():
        if message.get(field) != expected:
            raise SkeletonError(
                f"skeleton.{field} must be {expected!r}, got {message.get(field)!r}"
            )

    joints_value = message.get("joints")
    if not isinstance(joints_value, list) or len(joints_value) != len(SOURCE_JOINT_SPECS):
        raise SkeletonError(f"skeleton.joints must contain {len(SOURCE_JOINT_SPECS)} joints")

    indexed: dict[int, Mapping[str, Any]] = {}
    for raw_joint in joints_value:
        if not isinstance(raw_joint, Mapping):
            raise SkeletonError("each skeleton joint must be a JSON object")
        index = raw_joint.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise SkeletonError("each skeleton joint index must be an integer")
        if index in indexed:
            raise SkeletonError(f"duplicate skeleton joint index {index}")
        indexed[index] = raw_joint

    if set(indexed) != set(range(len(SOURCE_JOINT_SPECS))):
        raise SkeletonError("skeleton joint indices must be contiguous from 0 to 22")

    positions = np.empty((len(SOURCE_JOINT_SPECS), 3), dtype=np.float64)
    offsets = np.empty_like(positions)
    for index, (expected_name, expected_parent) in enumerate(SOURCE_JOINT_SPECS):
        joint = indexed[index]
        if joint.get("name") != expected_name:
            raise SkeletonError(
                f"joint {index} name must be {expected_name!r}, got {joint.get('name')!r}"
            )
        if joint.get("parent_index") != expected_parent:
            raise SkeletonError(
                f"joint {index} parent must be {expected_parent}, "
                f"got {joint.get('parent_index')!r}"
            )
        positions[index] = _vector3(
            joint.get("initial_position"),
            f"joint {index} initial_position",
            SkeletonError,
        )
        offsets[index] = _vector3(
            joint.get("offset"), f"joint {index} offset", SkeletonError
        )

    for index, (_, parent) in enumerate(SOURCE_JOINT_SPECS):
        expected_offset = positions[index] if parent == -1 else positions[index] - positions[parent]
        if not np.allclose(offsets[index], expected_offset, atol=1e-4, rtol=1e-4):
            raise SkeletonError(
                f"joint {index} offset is inconsistent with its initial positions"
            )
        if parent != -1 and np.linalg.norm(expected_offset) < 1e-6:
            raise SkeletonError(f"joint {index} has a zero-length rest bone")

    skeleton = SkeletonDefinition(positions=positions, offsets=offsets)
    if skeleton.spine_distances[-1] < 1e-6:
        raise SkeletonError("skeleton spine has zero total length")
    return skeleton


def quaternion_to_matrix(quaternion: FloatArray) -> FloatArray:
    """Convert a normalized wxyz quaternion to a 3x3 rotation matrix."""

    w, x, y, z = quaternion
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(matrix: FloatArray) -> FloatArray:
    """Convert a valid 3x3 rotation matrix to a normalized wxyz quaternion."""

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        diagonal_index = int(np.argmax(np.diag(matrix)))
        if diagonal_index == 0:
            scale = np.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif diagonal_index == 1:
            scale = np.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return quaternion / np.linalg.norm(quaternion)


def matrix_to_rotvec(matrix: FloatArray) -> FloatArray:
    """Convert a rotation matrix to a principal axis-angle rotation vector."""

    quaternion = matrix_to_quaternion(matrix)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    vector_norm = float(np.linalg.norm(quaternion[1:]))
    if vector_norm < 1e-10:
        return 2.0 * quaternion[1:]
    angle = 2.0 * np.arctan2(vector_norm, float(quaternion[0]))
    return quaternion[1:] * (angle / vector_norm)


def quaternion_slerp(first: FloatArray, second: FloatArray, amount: float) -> FloatArray:
    """Shortest-path spherical interpolation between normalized quaternions."""

    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = first + amount * (second - first)
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    result = (
        np.sin((1.0 - amount) * angle) / sin_angle * first
        + np.sin(amount * angle) / sin_angle * second
    )
    return result / np.linalg.norm(result)


def _sample_spine_quaternion(
    source_quaternions: FloatArray, distances: FloatArray, target_distance: float
) -> FloatArray:
    spine_indices = (0, 9, 10, 11, 12)
    if target_distance >= distances[-1]:
        return source_quaternions[spine_indices[-1]].copy()
    upper = int(np.searchsorted(distances, target_distance, side="right"))
    lower = max(0, upper - 1)
    segment_length = float(distances[upper] - distances[lower])
    amount = 0.0 if segment_length <= 1e-12 else (target_distance - distances[lower]) / segment_length
    return quaternion_slerp(
        source_quaternions[spine_indices[lower]],
        source_quaternions[spine_indices[upper]],
        float(amount),
    )


class SmplConverter:
    """Stateful converter that accepts skeleton updates and converts frames."""

    def __init__(self) -> None:
        self._skeleton = validate_skeleton(builtin_skeleton_message())
        self.has_received_skeleton = False

    @property
    def skeleton(self) -> SkeletonDefinition:
        return self._skeleton

    def update_skeleton(self, message: Mapping[str, Any]) -> None:
        """Validate and atomically install a new rest skeleton."""

        validated = validate_skeleton(message)
        self._skeleton = validated
        self.has_received_skeleton = True

    def convert_frame(self, message: Mapping[str, Any]) -> dict[str, Any]:
        """Convert a validated version-1 frame into standard SMPL parameters."""

        if message.get("type") != "frame":
            raise ConversionError(f"message type must be 'frame', got {message.get('type')!r}")
        if message.get("version") != 1:
            raise ConversionError(f"frame.version must be 1, got {message.get('version')!r}")
        frame_index = message.get("frame_index")
        if not isinstance(frame_index, int) or isinstance(frame_index, bool):
            raise ConversionError("frame.frame_index must be an integer")
        joints = message.get("joints")
        if not isinstance(joints, list) or len(joints) != len(SOURCE_JOINT_SPECS):
            raise ConversionError(f"frame.joints must contain {len(SOURCE_JOINT_SPECS)} joints")

        positions = np.empty((len(SOURCE_JOINT_SPECS), 3), dtype=np.float64)
        quaternions = np.empty((len(SOURCE_JOINT_SPECS), 4), dtype=np.float64)
        for index, joint in enumerate(joints):
            if not isinstance(joint, Mapping):
                raise ConversionError(f"frame joint {index} must be a JSON object")
            positions[index] = _vector3(
                joint.get("position"), f"frame joint {index} position", ConversionError
            )
            quaternions[index] = _quaternion(
                joint.get("quaternion"), f"frame joint {index} quaternion"
            )

        source_for_smpl: list[FloatArray | None] = [
            None if source_index is None else quaternions[source_index]
            for source_index in SMPL_SOURCE_INDICES
        ]
        spine_distances = self._skeleton.spine_distances
        total_spine_length = float(spine_distances[-1])
        for target_index, fraction in zip((3, 6, 9), (1.0 / 3.0, 2.0 / 3.0, 1.0)):
            source_for_smpl[target_index] = _sample_spine_quaternion(
                quaternions, spine_distances, total_spine_length * fraction
            )

        global_matrices = np.empty((len(SMPL_JOINT_NAMES), 3, 3), dtype=np.float64)
        for index, quaternion in enumerate(source_for_smpl):
            if quaternion is None:  # Defensive: all virtual joints were filled above.
                raise RuntimeError(f"SMPL joint {index} has no source rotation")
            source_matrix = quaternion_to_matrix(quaternion)
            global_matrices[index] = WS_GEO_TO_SMPL @ source_matrix @ WS_GEO_TO_SMPL.T

        local_matrices = np.empty_like(global_matrices)
        local_matrices[0] = global_matrices[0]
        for index in range(1, len(SMPL_JOINT_NAMES)):
            parent = SMPL_PARENTS[index]
            local_matrices[index] = global_matrices[parent].T @ global_matrices[index]

        rotvecs = np.stack([matrix_to_rotvec(matrix) for matrix in local_matrices])
        translation = WS_GEO_TO_SMPL @ positions[0]
        return {
            "type": "smpl_frame",
            "version": 1,
            "frame_index": frame_index,
            "coordinate_system": "SMPL_Xleft_Yup_Zforward",
            "rotation_representation": "axis_angle",
            "transl": translation.tolist(),
            "global_orient": rotvecs[0].tolist(),
            "body_pose": rotvecs[1:].reshape(-1).tolist(),
            "betas": [0.0] * 10,
        }
