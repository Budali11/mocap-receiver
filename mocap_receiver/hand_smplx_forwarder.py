"""Convert VD Suit hand UDP JSON to SMPL-X hand poses and forward it.

The Stage-II layout used here follows ``OK_B_stageii.npz``::

    poses = root_orient + pose_body + pose_jaw + pose_eye + pose_hand

Only the two articulated hands are observable in the source stream.  The
unobserved body, face and root fields are therefore zero.  Source wrist world
positions and orientations are retained in explicit extension fields.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import socket
from typing import Any, BinaryIO, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .converter import (
    ConversionError,
    SkeletonError,
    WS_GEO_TO_SMPL,
    matrix_to_rotvec,
    quaternion_to_matrix,
)
from .protocol import decode_json_messages, encode_smpl_frame
from .recording import SmplxHandNpzRecorder
from .server import ReceiverStats, run_udp_loop


LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]


# SMPL-X/MANO hand order.  ``pose_hand`` contains the complete left tuple
# followed by the complete right tuple, with three axis-angle values per joint.
SMPLX_FINGER_JOINT_NAMES: tuple[str, ...] = (
    "index1",
    "index2",
    "index3",
    "middle1",
    "middle2",
    "middle3",
    "pinky1",
    "pinky2",
    "pinky3",
    "ring1",
    "ring2",
    "ring3",
    "thumb1",
    "thumb2",
    "thumb3",
)
SMPLX_HAND_JOINT_NAMES: tuple[str, ...] = tuple(
    f"{side}_{name}" for side in ("left", "right") for name in SMPLX_FINGER_JOINT_NAMES
)


def _source_hand_specs(side: str, start: int) -> tuple[tuple[str, int], ...]:
    """Build one 20-joint VD Suit hand topology."""

    local_specs = (
        ("Hand", -1),
        ("ThumbFinger", 0),
        ("ThumbFinger1", 1),
        ("ThumbFinger2", 2),
        ("IndexFinger", 0),
        ("IndexFinger1", 4),
        ("IndexFinger2", 5),
        ("IndexFinger3", 6),
        ("MiddleFinger", 0),
        ("MiddleFinger1", 8),
        ("MiddleFinger2", 9),
        ("MiddleFinger3", 10),
        ("RingFinger", 0),
        ("RingFinger1", 12),
        ("RingFinger2", 13),
        ("RingFinger3", 14),
        ("PinkyFinger", 0),
        ("PinkyFinger1", 16),
        ("PinkyFinger2", 17),
        ("PinkyFinger3", 18),
    )
    return tuple(
        (f"{side}{name}", -1 if parent == -1 else start + parent)
        for name, parent in local_specs
    )


SOURCE_HAND_JOINT_SPECS: tuple[tuple[str, int], ...] = (
    _source_hand_specs("Right", 0) + _source_hand_specs("Left", 20)
)

# Each tuple is one SMPL-X finger chain in the standard order above.  VD Suit
# has an extra palm/metacarpal joint for the four non-thumb fingers.  Selecting
# *Finger1 as the first SMPL-X joint and taking it relative to the wrist folds
# the skipped palm joint's global effect into the first SMPL-X rotation.
SOURCE_CHAINS: dict[str, tuple[tuple[int, int, int], ...]] = {
    "left": (
        (25, 26, 27),  # index
        (29, 30, 31),  # middle
        (37, 38, 39),  # pinky
        (33, 34, 35),  # ring
        (21, 22, 23),  # thumb
    ),
    "right": (
        (5, 6, 7),
        (9, 10, 11),
        (17, 18, 19),
        (13, 14, 15),
        (1, 2, 3),
    ),
}
SOURCE_WRIST_INDICES = {"left": 20, "right": 0}


@dataclass(frozen=True)
class HandSkeletonDefinition:
    """Validated 40-joint rest skeleton in source index order."""

    positions: FloatArray
    offsets: FloatArray


def _finite_vector(value: Any, size: int, field: str, error: type[ConversionError]) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise error(f"{field} must contain {size} numeric values") from exc
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise error(f"{field} must contain {size} finite numeric values")
    return result


def _normalized_quaternion(value: Any, field: str) -> FloatArray:
    result = _finite_vector(value, 4, field, ConversionError)
    norm = float(np.linalg.norm(result))
    if norm < 1e-12:
        raise ConversionError(f"{field} must not be a zero quaternion")
    return result / norm


def validate_hand_skeleton(message: Mapping[str, Any]) -> HandSkeletonDefinition:
    """Validate the metadata, names and topology of a VD Suit hand skeleton."""

    expected_metadata = {
        "type": "skeleton",
        "version": 1,
        "joint_count": len(SOURCE_HAND_JOINT_SPECS),
        "coordinate_system": "WS_Geo",
        "position_unit": "m",
        "quaternion_order": "wxyz",
    }
    for field, expected in expected_metadata.items():
        if message.get(field) != expected:
            raise SkeletonError(
                f"skeleton.{field} must be {expected!r}, got {message.get(field)!r}"
            )

    raw_joints = message.get("joints")
    if not isinstance(raw_joints, list) or len(raw_joints) != len(SOURCE_HAND_JOINT_SPECS):
        raise SkeletonError(
            f"skeleton.joints must contain {len(SOURCE_HAND_JOINT_SPECS)} joints"
        )

    indexed: dict[int, Mapping[str, Any]] = {}
    for raw_joint in raw_joints:
        if not isinstance(raw_joint, Mapping):
            raise SkeletonError("each skeleton joint must be a JSON object")
        index = raw_joint.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise SkeletonError("each skeleton joint index must be an integer")
        if index in indexed:
            raise SkeletonError(f"duplicate skeleton joint index {index}")
        indexed[index] = raw_joint

    expected_indices = set(range(len(SOURCE_HAND_JOINT_SPECS)))
    if set(indexed) != expected_indices:
        raise SkeletonError("skeleton joint indices must be contiguous from 0 to 39")

    positions = np.empty((len(SOURCE_HAND_JOINT_SPECS), 3), dtype=np.float64)
    offsets = np.empty_like(positions)
    for index, (expected_name, expected_parent) in enumerate(SOURCE_HAND_JOINT_SPECS):
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
        positions[index] = _finite_vector(
            joint.get("initial_position"),
            3,
            f"joint {index} initial_position",
            SkeletonError,
        )
        offsets[index] = _finite_vector(
            joint.get("offset"), 3, f"joint {index} offset", SkeletonError
        )

    for index, (_, parent) in enumerate(SOURCE_HAND_JOINT_SPECS):
        expected_offset = positions[index] if parent == -1 else positions[index] - positions[parent]
        if not np.allclose(offsets[index], expected_offset, atol=1e-4, rtol=1e-4):
            raise SkeletonError(
                f"joint {index} offset is inconsistent with its initial positions"
            )
        if parent != -1 and np.linalg.norm(expected_offset) < 1e-6:
            raise SkeletonError(f"joint {index} has a zero-length rest bone")
    return HandSkeletonDefinition(positions=positions, offsets=offsets)


def _clean_list(values: FloatArray) -> list[Any]:
    """Create compact, deterministic JSON numbers without measurable pose loss."""

    rounded = np.round(np.asarray(values, dtype=np.float64), decimals=9)
    rounded[np.abs(rounded) < 5e-10] = 0.0
    return rounded.tolist()


class SmplxHandConverter:
    """Convert fixed-topology VD Suit global hand rotations to SMPL-X."""

    def __init__(self, mocap_framerate: float = 60.0) -> None:
        if not math.isfinite(mocap_framerate) or mocap_framerate <= 0.0:
            raise ValueError("mocap_framerate must be a positive finite number")
        self.mocap_framerate = float(mocap_framerate)
        self.skeleton: HandSkeletonDefinition | None = None
        self.has_received_skeleton = False

    def update_skeleton(self, message: Mapping[str, Any]) -> None:
        """Validate and atomically install an input hand skeleton."""

        skeleton = validate_hand_skeleton(message)
        self.skeleton = skeleton
        self.has_received_skeleton = True

    @staticmethod
    def _hand_pose(global_matrices: FloatArray, side: str) -> FloatArray:
        wrist_index = SOURCE_WRIST_INDICES[side]
        local_rotvecs: list[FloatArray] = []
        for chain in SOURCE_CHAINS[side]:
            parent_index = wrist_index
            for source_index in chain:
                source_local = global_matrices[parent_index].T @ global_matrices[source_index]
                smplx_local = WS_GEO_TO_SMPL @ source_local @ WS_GEO_TO_SMPL.T
                local_rotvecs.append(matrix_to_rotvec(smplx_local))
                parent_index = source_index
        return np.stack(local_rotvecs).reshape(45)

    def convert_frame(self, message: Mapping[str, Any]) -> dict[str, Any]:
        """Convert one version-1 source frame to a Stage-II-compatible JSON frame."""

        if message.get("type") != "frame":
            raise ConversionError(f"message type must be 'frame', got {message.get('type')!r}")
        if message.get("version") != 1:
            raise ConversionError(f"frame.version must be 1, got {message.get('version')!r}")
        frame_index = message.get("frame_index")
        if not isinstance(frame_index, int) or isinstance(frame_index, bool):
            raise ConversionError("frame.frame_index must be an integer")
        raw_joints = message.get("joints")
        if not isinstance(raw_joints, list) or len(raw_joints) != len(SOURCE_HAND_JOINT_SPECS):
            raise ConversionError(
                f"frame.joints must contain {len(SOURCE_HAND_JOINT_SPECS)} joints"
            )

        positions = np.empty((len(SOURCE_HAND_JOINT_SPECS), 3), dtype=np.float64)
        global_matrices = np.empty((len(SOURCE_HAND_JOINT_SPECS), 3, 3), dtype=np.float64)
        for index, joint in enumerate(raw_joints):
            if not isinstance(joint, Mapping):
                raise ConversionError(f"frame joint {index} must be a JSON object")
            positions[index] = _finite_vector(
                joint.get("position"), 3, f"frame joint {index} position", ConversionError
            )
            quaternion = _normalized_quaternion(
                joint.get("quaternion"), f"frame joint {index} quaternion"
            )
            global_matrices[index] = quaternion_to_matrix(quaternion)

        left_pose = self._hand_pose(global_matrices, "left")
        right_pose = self._hand_pose(global_matrices, "right")
        pose_hand = np.concatenate((left_pose, right_pose))

        # OK_B_stageii.npz stores these components in this exact 165-D order.
        root_orient = np.zeros(3, dtype=np.float64)
        pose_body = np.zeros(63, dtype=np.float64)
        pose_jaw = np.zeros(3, dtype=np.float64)
        pose_eye = np.zeros(6, dtype=np.float64)
        poses = np.concatenate((root_orient, pose_body, pose_jaw, pose_eye, pose_hand))

        hand_positions = np.stack(
            (
                WS_GEO_TO_SMPL @ positions[SOURCE_WRIST_INDICES["left"]],
                WS_GEO_TO_SMPL @ positions[SOURCE_WRIST_INDICES["right"]],
            )
        )
        hand_global_orient = np.stack(
            tuple(
                matrix_to_rotvec(
                    WS_GEO_TO_SMPL
                    @ global_matrices[SOURCE_WRIST_INDICES[side]]
                    @ WS_GEO_TO_SMPL.T
                )
                for side in ("left", "right")
            )
        )

        return {
            "type": "smplx_hand_frame",
            "version": 1,
            "frame_index": frame_index,
            "surface_model_type": "smplx",
            "gender": "neutral",
            "mocap_frame_rate": self.mocap_framerate,
            "coordinate_system": "SMPL_Xleft_Yup_Zforward",
            "rotation_representation": "axis_angle",
            "trans": [0.0, 0.0, 0.0],
            "poses": _clean_list(poses),
            "betas": [0.0] * 16,
            "num_betas": 16,
            "root_orient": _clean_list(root_orient),
            "pose_body": _clean_list(pose_body),
            "pose_hand": _clean_list(pose_hand),
            "pose_jaw": _clean_list(pose_jaw),
            "pose_eye": _clean_list(pose_eye),
            # Extensions preserve source information that has no standalone
            # place in pose_hand.  Both arrays are explicitly left then right.
            "hand_side_order": ["left", "right"],
            "hand_positions": _clean_list(hand_positions),
            "hand_global_orient": _clean_list(hand_global_orient),
        }


class HandDatagramProcessor:
    """Decode input JSON/JSONL and emit compact SMPL-X JSON datagrams."""

    def __init__(self, converter: SmplxHandConverter | None = None) -> None:
        self.converter = converter or SmplxHandConverter()
        self.stats = ReceiverStats()

    def process(self, payload: bytes) -> list[bytes]:
        self.stats.datagrams += 1
        messages, decode_errors = decode_json_messages(payload)
        for error in decode_errors:
            self.stats.dropped += 1
            LOGGER.warning("Dropped input: %s", error)

        outputs: list[bytes] = []
        for message in messages:
            self.stats.messages += 1
            try:
                converted = self._process_message(message)
            except ConversionError as exc:
                self.stats.dropped += 1
                LOGGER.warning("Dropped input: %s", exc)
                continue
            if converted is not None:
                outputs.append(encode_smpl_frame(converted))
                self.stats.outputs += 1
        return outputs

    def _process_message(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        message_type = message.get("type")
        if message_type == "skeleton":
            self.converter.update_skeleton(message)
            self.stats.skeletons += 1
            LOGGER.info("Installed and validated source hand skeleton")
            return None
        if message_type == "frame":
            converted = self.converter.convert_frame(message)
            self.stats.frames += 1
            return converted
        raise ConversionError(f"unsupported message type {message_type!r}")


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


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


def resolve_output_format(path: Path, requested_format: str) -> str:
    """Resolve auto output format from the destination suffix."""

    if requested_format != "auto":
        return requested_format
    return "smplx-npz" if path.suffix.lower() == ".npz" else "jsonl"


def open_output_file(path: Path, append: bool, overwrite: bool) -> BinaryIO:
    """Open a JSONL output with an explicit existing-file policy."""

    if append:
        mode = "ab"
    elif overwrite:
        mode = "wb"
    else:
        mode = "xb"
    return path.open(mode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Receive 40-joint VD Suit hand JSON over UDP, convert it to SMPL-X "
            "Stage-II fields, then forward and/or save the converted frames."
        )
    )
    parser.add_argument(
        "--listen-host", default="0.0.0.0", help="local IPv4 bind address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--listen-port", required=True, type=_port, help="UDP port receiving VD Suit hand data"
    )
    parser.add_argument(
        "--target-host", default="127.0.0.1", help="destination IP/hostname (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--target-port", type=_port, help="optional destination UDP port for SMPL-X frames"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="optional local .npz SMPL-X motion or .jsonl stream file",
    )
    parser.add_argument(
        "--output-format",
        choices=("auto", "smplx-npz", "jsonl"),
        default="auto",
        help="file format; auto selects smplx-npz for .npz, otherwise jsonl",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--append-output",
        action="store_true",
        help="append to an existing JSONL output file",
    )
    output_mode.add_argument(
        "--overwrite-output",
        action="store_true",
        help="replace an existing output file",
    )
    parser.add_argument(
        "--mocap-framerate",
        type=_positive_float,
        default=60.0,
        help="frame rate written into output metadata (default: 60)",
    )
    parser.add_argument(
        "--receive-size",
        type=_positive_int,
        default=65535,
        help="maximum input UDP payload size (default: 65535)",
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
    if args.target_port is None and args.output_file is None:
        parser.error("at least one of --target-port or --output-file is required")
    if args.output_file is None and (args.append_output or args.overwrite_output):
        parser.error("--append-output/--overwrite-output requires --output-file")
    if args.output_file is None and args.output_format != "auto":
        parser.error("--output-format requires --output-file")
    output_format = (
        resolve_output_format(args.output_file, args.output_format)
        if args.output_file is not None
        else None
    )
    if output_format == "smplx-npz" and args.append_output:
        parser.error("--append-output is only supported for JSONL; use a new .npz file")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    input_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    output_socket = (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if args.target_port is not None
        else None
    )
    output_stream: BinaryIO | None = None
    motion_recorder: SmplxHandNpzRecorder | None = None
    processor = HandDatagramProcessor(SmplxHandConverter(args.mocap_framerate))
    try:
        input_socket.bind((args.listen_host, args.listen_port))
        if args.output_file is not None:
            try:
                if output_format == "smplx-npz":
                    motion_recorder = SmplxHandNpzRecorder(
                        args.output_file,
                        mocap_framerate=args.mocap_framerate,
                        overwrite=args.overwrite_output,
                    )
                else:
                    output_stream = open_output_file(
                        args.output_file, args.append_output, args.overwrite_output
                    )
            except FileExistsError:
                parser.error(
                    f"output file already exists: {args.output_file}; use "
                    "--append-output or --overwrite-output"
                )
            except OSError as exc:
                parser.error(f"cannot open output file {args.output_file}: {exc}")
            LOGGER.info("Recording %s to %s", output_format, args.output_file)
        if args.target_port is not None:
            LOGGER.info(
                "Listening on udp://%s:%d and forwarding SMPL-X hands to udp://%s:%d",
                args.listen_host,
                args.listen_port,
                args.target_host,
                args.target_port,
            )
        else:
            LOGGER.info("Listening on udp://%s:%d", args.listen_host, args.listen_port)
        run_udp_loop(
            input_socket,
            output_socket,
            (args.target_host, args.target_port) if args.target_port is not None else None,
            processor,  # type: ignore[arg-type]
            receive_size=args.receive_size,
            output_stream=output_stream,
            motion_recorder=motion_recorder,  # type: ignore[arg-type]
        )
    except KeyboardInterrupt:
        LOGGER.info("Stopped by user")
    finally:
        input_socket.close()
        if output_socket is not None:
            output_socket.close()
        if output_stream is not None:
            output_stream.close()
        if motion_recorder is not None:
            motion_recorder.close()

    stats = processor.stats
    LOGGER.info(
        "Stats: datagrams=%d messages=%d skeletons=%d frames=%d outputs=%d dropped=%d",
        stats.datagrams,
        stats.messages,
        stats.skeletons,
        stats.frames,
        stats.outputs,
        stats.dropped,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
