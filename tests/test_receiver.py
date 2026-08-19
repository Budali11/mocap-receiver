from __future__ import annotations

import copy
import io
import json
import math
from pathlib import Path
import socket
import tempfile
import threading
import unittest

import numpy as np

from mocap_receiver.converter import (
    SMPL_PARENTS,
    WS_GEO_TO_SMPL,
    SkeletonError,
    SmplConverter,
    builtin_skeleton_message,
    quaternion_to_matrix,
)
from mocap_receiver.cli import open_output_file, resolve_output_format
from mocap_receiver.protocol import decode_json_messages
from mocap_receiver.recording import SmplNpzRecorder
from mocap_receiver.server import DatagramProcessor, run_udp_loop


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    axis = rotvec / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def reconstruct_global_matrices(output: dict[str, object]) -> np.ndarray:
    rotvecs = np.vstack(
        (
            np.asarray(output["global_orient"], dtype=np.float64),
            np.asarray(output["body_pose"], dtype=np.float64).reshape(23, 3),
        )
    )
    local = np.stack([rotvec_to_matrix(rotvec) for rotvec in rotvecs])
    global_matrices = np.empty_like(local)
    global_matrices[0] = local[0]
    for index in range(1, 24):
        global_matrices[index] = global_matrices[SMPL_PARENTS[index]] @ local[index]
    return global_matrices


class ConverterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        payload = (REPOSITORY_ROOT / "vdsuit_udp_stream_example.json").read_bytes()
        cls.messages, errors = decode_json_messages(payload)
        if errors:
            raise AssertionError(errors)
        cls.skeleton_message = cls.messages[0]
        cls.frame_message = cls.messages[1]

    def test_example_identity_frame(self) -> None:
        converter = SmplConverter()
        output = converter.convert_frame(self.frame_message)

        self.assertEqual(output["type"], "smpl_frame")
        self.assertEqual(output["frame_index"], 123)
        np.testing.assert_allclose(output["transl"], [0.0, 1.11000001, 0.0])
        np.testing.assert_allclose(output["global_orient"], np.zeros(3), atol=1e-12)
        self.assertEqual(len(output["body_pose"]), 69)
        np.testing.assert_allclose(output["body_pose"], np.zeros(69), atol=1e-12)
        self.assertEqual(output["betas"], [0.0] * 10)

    def test_world_rotation_is_changed_to_smpl_basis(self) -> None:
        frame = copy.deepcopy(self.frame_message)
        half_angle = math.pi / 4.0
        frame["joints"][0]["quaternion"] = [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]

        output = SmplConverter().convert_frame(frame)

        np.testing.assert_allclose(output["global_orient"], [0.0, math.pi / 2.0, 0.0], atol=1e-9)

    def test_global_child_rotation_becomes_local_smpl_rotation(self) -> None:
        frame = copy.deepcopy(self.frame_message)
        half_angle = math.pi / 4.0
        # Source LeftLowerLeg is SMPL left_knee (index 4).
        frame["joints"][6]["quaternion"] = [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]

        output = SmplConverter().convert_frame(frame)
        body_pose = np.asarray(output["body_pose"]).reshape(23, 3)

        np.testing.assert_allclose(body_pose[3], [0.0, math.pi / 2.0, 0.0], atol=1e-9)

    def test_spine_resampling_preserves_upper_torso_global_orientation(self) -> None:
        frame = copy.deepcopy(self.frame_message)
        for source_index, angle in zip((9, 10, 11, 12), (0.1, 0.2, 0.3, 0.4)):
            frame["joints"][source_index]["quaternion"] = [
                math.cos(angle / 2.0),
                0.0,
                0.0,
                math.sin(angle / 2.0),
            ]

        output = SmplConverter().convert_frame(frame)
        reconstructed = reconstruct_global_matrices(output)
        source_upper = np.asarray(frame["joints"][12]["quaternion"])
        expected = WS_GEO_TO_SMPL @ quaternion_to_matrix(source_upper) @ WS_GEO_TO_SMPL.T

        np.testing.assert_allclose(reconstructed[9], expected, atol=1e-9)

    def test_skeleton_update_and_validation(self) -> None:
        converter = SmplConverter()
        self.assertFalse(converter.has_received_skeleton)
        converter.update_skeleton(self.skeleton_message)
        self.assertTrue(converter.has_received_skeleton)

        invalid = copy.deepcopy(self.skeleton_message)
        invalid["joints"][5]["name"] = "NotLeftUpperLeg"
        with self.assertRaises(SkeletonError):
            converter.update_skeleton(invalid)

    def test_builtin_skeleton_matches_expected_protocol(self) -> None:
        message = builtin_skeleton_message()
        self.assertEqual(message["joint_count"], 23)
        SmplConverter().update_skeleton(message)


class ProtocolAndServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = (REPOSITORY_ROOT / "vdsuit_udp_stream_example.json").read_bytes()
        cls.messages, errors = decode_json_messages(cls.payload)
        if errors:
            raise AssertionError(errors)

    def test_decodes_jsonl_example(self) -> None:
        messages, errors = decode_json_messages(self.payload)
        self.assertEqual(errors, [])
        self.assertEqual([message["type"] for message in messages], ["skeleton", "frame"])

    def test_valid_jsonl_record_survives_malformed_neighbor(self) -> None:
        frame_bytes = json.dumps(self.messages[1]).encode("utf-8")
        messages, errors = decode_json_messages(b"not-json\n" + frame_bytes)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["type"], "frame")
        self.assertEqual(len(errors), 1)

    def test_frame_before_skeleton_uses_builtin_definition(self) -> None:
        processor = DatagramProcessor()
        outputs = processor.process(json.dumps(self.messages[1]).encode("utf-8"))

        self.assertEqual(len(outputs), 1)
        output = json.loads(outputs[0])
        self.assertEqual(output["type"], "smpl_frame")
        self.assertFalse(processor.converter.has_received_skeleton)

    def test_idle_cli_loop_always_uses_interruptible_socket_timeout(self) -> None:
        class InterruptingSocket:
            def __init__(self) -> None:
                self.timeout: float | None = None

            def settimeout(self, value: float) -> None:
                self.timeout = value

            def recvfrom(self, _size: int) -> tuple[bytes, object]:
                raise KeyboardInterrupt

        input_socket = InterruptingSocket()
        with self.assertRaises(KeyboardInterrupt):
            run_udp_loop(
                input_socket,  # type: ignore[arg-type]
                None,
                None,
                DatagramProcessor(),
                output_stream=io.BytesIO(),
            )
        self.assertEqual(input_socket.timeout, 0.2)

    def test_udp_jsonl_loop(self) -> None:
        input_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        input_socket.bind(("127.0.0.1", 0))
        input_address = input_socket.getsockname()
        capture_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        capture_socket.bind(("127.0.0.1", 0))
        capture_socket.settimeout(2.0)
        output_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        stop_event = threading.Event()
        processor = DatagramProcessor()
        recording = io.BytesIO()

        thread = threading.Thread(
            target=run_udp_loop,
            args=(
                input_socket,
                output_socket,
                capture_socket.getsockname(),
                processor,
                stop_event,
            ),
            kwargs={"output_stream": recording},
            daemon=True,
        )
        thread.start()
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sender.sendto(self.payload, input_address)
            converted_bytes, _ = capture_socket.recvfrom(65535)
            converted = json.loads(converted_bytes)
            self.assertEqual(converted["type"], "smpl_frame")
            self.assertEqual(converted["frame_index"], 123)
        finally:
            stop_event.set()
            thread.join(timeout=1.0)
            sender.close()
            input_socket.close()
            output_socket.close()
            capture_socket.close()

        self.assertFalse(thread.is_alive())
        self.assertEqual(processor.stats.skeletons, 1)
        self.assertEqual(processor.stats.frames, 1)
        self.assertEqual(processor.stats.outputs, 1)
        recorded_lines = recording.getvalue().splitlines()
        self.assertEqual(len(recorded_lines), 1)
        self.assertEqual(json.loads(recorded_lines[0])["frame_index"], 123)

    def test_output_file_requires_explicit_existing_file_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "recording.jsonl"
            with open_output_file(output_path, append=False, overwrite=False) as stream:
                stream.write(b"first\n")

            with self.assertRaises(FileExistsError):
                open_output_file(output_path, append=False, overwrite=False)

            with open_output_file(output_path, append=True, overwrite=False) as stream:
                stream.write(b"second\n")
            self.assertEqual(output_path.read_bytes(), b"first\nsecond\n")

            with open_output_file(output_path, append=False, overwrite=True) as stream:
                stream.write(b"replacement\n")
            self.assertEqual(output_path.read_bytes(), b"replacement\n")

    def test_smpl_npz_motion_file(self) -> None:
        converted = SmplConverter().convert_frame(self.messages[1])
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "motion.npz"
            recorder = SmplNpzRecorder(output_path, mocap_framerate=60.0, chunk_size=2)
            recorder.record(converted)
            recorder.record(converted)
            recorder.record(converted)
            recorder.close()
            recorder.close()  # Closing is deliberately idempotent.

            with np.load(output_path, allow_pickle=False) as motion:
                self.assertEqual(
                    set(motion.files),
                    {
                        "poses",
                        "trans",
                        "frame_index",
                        "betas",
                        "gender",
                        "mocap_framerate",
                    },
                )
                self.assertEqual(motion["poses"].shape, (3, 72))
                self.assertEqual(motion["trans"].shape, (3, 3))
                np.testing.assert_array_equal(motion["frame_index"], [123, 123, 123])
                self.assertEqual(motion["betas"].shape, (10,))
                self.assertEqual(motion["poses"].dtype, np.float64)
                self.assertEqual(motion["trans"].dtype, np.float64)
                self.assertEqual(motion["betas"].dtype, np.float64)
                self.assertEqual(motion["mocap_framerate"].dtype, np.float64)
                np.testing.assert_allclose(motion["poses"], 0.0, atol=1e-7)
                np.testing.assert_allclose(motion["trans"][:, 1], 1.11000001, atol=1e-7)
                self.assertEqual(str(motion["gender"]), "neutral")
                self.assertEqual(float(motion["mocap_framerate"]), 60.0)

            with self.assertRaises(FileExistsError):
                SmplNpzRecorder(output_path)

    def test_output_format_is_inferred_from_extension(self) -> None:
        self.assertEqual(resolve_output_format(Path("motion.npz"), "auto"), "smpl-npz")
        self.assertEqual(resolve_output_format(Path("frames.jsonl"), "auto"), "jsonl")
        self.assertEqual(resolve_output_format(Path("anything.bin"), "smpl-npz"), "smpl-npz")

    def test_file_only_udp_loop(self) -> None:
        flushed = threading.Event()

        class SignalingBytesIO(io.BytesIO):
            def flush(self) -> None:
                super().flush()
                flushed.set()

        input_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        input_socket.bind(("127.0.0.1", 0))
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        stop_event = threading.Event()
        recording = SignalingBytesIO()
        processor = DatagramProcessor()
        thread = threading.Thread(
            target=run_udp_loop,
            args=(input_socket, None, None, processor, stop_event),
            kwargs={"output_stream": recording},
            daemon=True,
        )
        thread.start()
        try:
            sender.sendto(json.dumps(self.messages[1]).encode("utf-8"), input_socket.getsockname())
            self.assertTrue(flushed.wait(timeout=2.0))
        finally:
            stop_event.set()
            thread.join(timeout=1.0)
            sender.close()
            input_socket.close()

        self.assertEqual(json.loads(recording.getvalue())["frame_index"], 123)
        self.assertEqual(processor.stats.outputs, 1)

    def test_npz_file_only_udp_loop(self) -> None:
        recorded = threading.Event()

        class SignalingRecorder(SmplNpzRecorder):
            def record_encoded(self, payload: bytes) -> None:
                super().record_encoded(payload)
                recorded.set()

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "udp_motion.npz"
            recorder = SignalingRecorder(output_path)
            input_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            input_socket.bind(("127.0.0.1", 0))
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            stop_event = threading.Event()
            processor = DatagramProcessor()
            thread = threading.Thread(
                target=run_udp_loop,
                args=(input_socket, None, None, processor, stop_event),
                kwargs={"motion_recorder": recorder},
                daemon=True,
            )
            thread.start()
            try:
                sender.sendto(
                    json.dumps(self.messages[1]).encode("utf-8"),
                    input_socket.getsockname(),
                )
                self.assertTrue(recorded.wait(timeout=2.0))
            finally:
                stop_event.set()
                thread.join(timeout=1.0)
                sender.close()
                input_socket.close()
                recorder.close()

            self.assertFalse(thread.is_alive())
            with np.load(output_path, allow_pickle=False) as motion:
                self.assertEqual(motion["poses"].shape, (1, 72))
                self.assertEqual(motion["trans"].shape, (1, 3))
                np.testing.assert_array_equal(motion["frame_index"], [123])


if __name__ == "__main__":
    unittest.main()
