from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import socket
import tempfile
import threading
import unittest

import numpy as np

from mocap_receiver.converter import WS_GEO_TO_SMPL, quaternion_to_matrix
from mocap_receiver.hand_smplx_forwarder import (
    HandDatagramProcessor,
    SMPLX_HAND_JOINT_NAMES,
    SmplxHandConverter,
    resolve_output_format,
    validate_hand_skeleton,
)
from mocap_receiver.protocol import decode_json_messages
from mocap_receiver.recording import SmplxHandNpzRecorder
from mocap_receiver.server import run_udp_loop


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def quaternion_z(angle: float) -> list[float]:
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


class SmplxHandConverterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        payload = (REPOSITORY_ROOT / "vdsuit_hand_udp_stream_example.json").read_bytes()
        cls.messages, errors = decode_json_messages(payload)
        if errors:
            raise AssertionError(errors)
        cls.skeleton = cls.messages[0]
        cls.frame = cls.messages[1]

    def test_reference_example_layout_and_identity_pose(self) -> None:
        output = SmplxHandConverter().convert_frame(self.frame)

        self.assertEqual(output["type"], "smplx_hand_frame")
        self.assertEqual(output["surface_model_type"], "smplx")
        self.assertEqual(output["frame_index"], 123)
        self.assertEqual(len(output["poses"]), 165)
        self.assertEqual(len(output["pose_body"]), 63)
        self.assertEqual(len(output["pose_hand"]), 90)
        self.assertEqual(len(output["pose_jaw"]), 3)
        self.assertEqual(len(output["pose_eye"]), 6)
        self.assertEqual(len(SMPLX_HAND_JOINT_NAMES), 30)
        np.testing.assert_allclose(output["poses"], 0.0, atol=1e-12)
        np.testing.assert_allclose(output["pose_hand"], 0.0, atol=1e-12)
        np.testing.assert_allclose(
            output["hand_positions"],
            [[0.748, 1.597, 0.0], [-0.748, 1.597, 0.0]],
            atol=1e-12,
        )

    def test_poses_uses_stageii_component_order(self) -> None:
        frame = copy.deepcopy(self.frame)
        frame["joints"][25]["quaternion"] = quaternion_z(math.pi / 2.0)
        output = SmplxHandConverter().convert_frame(frame)
        expected = np.concatenate(
            (
                output["root_orient"],
                output["pose_body"],
                output["pose_jaw"],
                output["pose_eye"],
                output["pose_hand"],
            )
        )
        np.testing.assert_allclose(output["poses"], expected, atol=1e-9)

    def test_left_index_first_joint_and_basis_conversion(self) -> None:
        frame = copy.deepcopy(self.frame)
        # A coherent global-rotation chain: the first joint rotates and its
        # descendants keep the same global orientation.
        for index in (25, 26, 27):
            frame["joints"][index]["quaternion"] = quaternion_z(math.pi / 2.0)
        output = SmplxHandConverter().convert_frame(frame)
        hand_pose = np.asarray(output["pose_hand"]).reshape(30, 3)

        np.testing.assert_allclose(hand_pose[0], [0.0, math.pi / 2.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(hand_pose[1:], 0.0, atol=1e-8)

    def test_global_wrist_motion_cancels_from_local_fingers(self) -> None:
        frame = copy.deepcopy(self.frame)
        rotation = quaternion_z(0.7)
        # Set the left wrist and all mapped finger nodes to the same global
        # rotation.  Every SMPL-X finger local rotation should be identity.
        for index in (20, 25, 26, 27, 29, 30, 31, 37, 38, 39, 33, 34, 35, 21, 22, 23):
            frame["joints"][index]["quaternion"] = rotation

        output = SmplxHandConverter().convert_frame(frame)
        hand_pose = np.asarray(output["pose_hand"]).reshape(30, 3)
        np.testing.assert_allclose(hand_pose[:15], 0.0, atol=1e-9)

        source = quaternion_to_matrix(np.asarray(rotation, dtype=np.float64))
        expected_wrist = WS_GEO_TO_SMPL @ source @ WS_GEO_TO_SMPL.T
        angle = float(np.linalg.norm(output["hand_global_orient"][0]))
        self.assertAlmostEqual(angle, math.acos((np.trace(expected_wrist) - 1.0) / 2.0), places=8)

    def test_skeleton_is_validated_but_frame_does_not_wait_for_it(self) -> None:
        converter = SmplxHandConverter()
        self.assertFalse(converter.has_received_skeleton)
        converter.convert_frame(self.frame)
        converter.update_skeleton(self.skeleton)
        self.assertTrue(converter.has_received_skeleton)
        validate_hand_skeleton(self.skeleton)

        invalid = copy.deepcopy(self.skeleton)
        invalid["joints"][25]["name"] = "WrongName"
        with self.assertRaises(ValueError):
            converter.update_skeleton(invalid)


class HandUdpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = (REPOSITORY_ROOT / "vdsuit_hand_udp_stream_example.json").read_bytes()

    def test_jsonl_processor_emits_one_frame(self) -> None:
        processor = HandDatagramProcessor()
        outputs = processor.process(self.payload)
        self.assertEqual(len(outputs), 1)
        output = json.loads(outputs[0])
        self.assertEqual(output["type"], "smplx_hand_frame")
        self.assertEqual(len(output["pose_hand"]), 90)
        self.assertEqual(processor.stats.skeletons, 1)
        self.assertEqual(processor.stats.frames, 1)

    def test_real_udp_forwarding(self) -> None:
        input_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        input_socket.bind(("127.0.0.1", 0))
        capture_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        capture_socket.bind(("127.0.0.1", 0))
        capture_socket.settimeout(2.0)
        output_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        stop_event = threading.Event()
        processor = HandDatagramProcessor()
        thread = threading.Thread(
            target=run_udp_loop,
            args=(
                input_socket,
                output_socket,
                capture_socket.getsockname(),
                processor,
                stop_event,
            ),
            daemon=True,
        )
        thread.start()
        try:
            sender.sendto(self.payload, input_socket.getsockname())
            output_payload, _ = capture_socket.recvfrom(65535)
            output = json.loads(output_payload)
            self.assertEqual(output["frame_index"], 123)
            self.assertEqual(len(output["poses"]), 165)
        finally:
            stop_event.set()
            thread.join(timeout=1.0)
            sender.close()
            input_socket.close()
            output_socket.close()
            capture_socket.close()
        self.assertFalse(thread.is_alive())

    def test_smplx_npz_sequence_file(self) -> None:
        messages, errors = decode_json_messages(self.payload)
        self.assertEqual(errors, [])
        converted = SmplxHandConverter().convert_frame(messages[1])
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "hands.npz"
            recorder = SmplxHandNpzRecorder(
                output_path, mocap_framerate=60.0, chunk_size=2
            )
            recorder.record(converted)
            recorder.record(converted)
            recorder.record(converted)
            recorder.close()
            recorder.close()

            with np.load(output_path, allow_pickle=False) as motion:
                self.assertEqual(
                    set(motion.files),
                    {
                        "gender",
                        "surface_model_type",
                        "mocap_frame_rate",
                        "mocap_time_length",
                        "trans",
                        "poses",
                        "betas",
                        "num_betas",
                        "root_orient",
                        "pose_body",
                        "pose_hand",
                        "pose_jaw",
                        "pose_eye",
                        "frame_index",
                        "hand_side_order",
                        "hand_positions",
                        "hand_global_orient",
                    },
                )
                self.assertEqual(motion["poses"].shape, (3, 165))
                self.assertEqual(motion["pose_hand"].shape, (3, 90))
                self.assertEqual(motion["pose_body"].shape, (3, 63))
                self.assertEqual(motion["root_orient"].shape, (3, 3))
                self.assertEqual(motion["pose_jaw"].shape, (3, 3))
                self.assertEqual(motion["pose_eye"].shape, (3, 6))
                self.assertEqual(motion["hand_positions"].shape, (3, 2, 3))
                self.assertEqual(motion["hand_global_orient"].shape, (3, 2, 3))
                expected_poses = np.concatenate(
                    (
                        motion["root_orient"],
                        motion["pose_body"],
                        motion["pose_jaw"],
                        motion["pose_eye"],
                        motion["pose_hand"],
                    ),
                    axis=1,
                )
                np.testing.assert_allclose(motion["poses"], expected_poses, atol=1e-7)
                np.testing.assert_array_equal(motion["frame_index"], [123, 123, 123])
                np.testing.assert_array_equal(motion["hand_side_order"], ["left", "right"])
                self.assertEqual(str(motion["gender"]), "neutral")
                self.assertEqual(str(motion["surface_model_type"]), "smplx")
                self.assertEqual(int(motion["num_betas"]), 16)
                self.assertAlmostEqual(float(motion["mocap_time_length"]), 3.0 / 60.0)
                for key in (
                    "trans",
                    "poses",
                    "betas",
                    "root_orient",
                    "pose_body",
                    "pose_hand",
                    "pose_jaw",
                    "pose_eye",
                    "hand_positions",
                    "hand_global_orient",
                ):
                    self.assertEqual(motion[key].dtype, np.float64, key)
                self.assertEqual(motion["mocap_frame_rate"].dtype, np.float64)
                self.assertEqual(motion["mocap_time_length"].dtype, np.float64)
                self.assertEqual(motion["frame_index"].dtype, np.int64)
                self.assertEqual(motion["num_betas"].dtype, np.int64)

            with self.assertRaises(FileExistsError):
                SmplxHandNpzRecorder(output_path)

    def test_file_only_udp_loop_records_npz(self) -> None:
        recorded = threading.Event()

        class SignalingRecorder(SmplxHandNpzRecorder):
            def record_encoded(self, payload: bytes) -> None:
                super().record_encoded(payload)
                recorded.set()

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "udp_hands.npz"
            recorder = SignalingRecorder(output_path)
            input_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            input_socket.bind(("127.0.0.1", 0))
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            stop_event = threading.Event()
            processor = HandDatagramProcessor()
            thread = threading.Thread(
                target=run_udp_loop,
                args=(input_socket, None, None, processor, stop_event),
                kwargs={"motion_recorder": recorder},
                daemon=True,
            )
            thread.start()
            try:
                sender.sendto(self.payload, input_socket.getsockname())
                self.assertTrue(recorded.wait(timeout=2.0))
            finally:
                stop_event.set()
                thread.join(timeout=1.0)
                sender.close()
                input_socket.close()
                recorder.close()

            self.assertFalse(thread.is_alive())
            with np.load(output_path, allow_pickle=False) as motion:
                self.assertEqual(motion["poses"].shape, (1, 165))
                self.assertEqual(motion["pose_hand"].shape, (1, 90))
                self.assertEqual(motion["poses"].dtype, np.float64)
                self.assertEqual(motion["pose_hand"].dtype, np.float64)

    def test_output_format_is_inferred_from_extension(self) -> None:
        self.assertEqual(resolve_output_format(Path("hands.npz"), "auto"), "smplx-npz")
        self.assertEqual(resolve_output_format(Path("hands.jsonl"), "auto"), "jsonl")
        self.assertEqual(resolve_output_format(Path("hands.bin"), "smplx-npz"), "smplx-npz")


if __name__ == "__main__":
    unittest.main()
