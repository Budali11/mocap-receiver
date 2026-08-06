from __future__ import annotations

import copy
import json
from pathlib import Path
import socket
import threading
import time
import unittest

import numpy as np

from mocap_receiver.converter import SmplConverter
from mocap_receiver.live_visualize import (
    LatestFrameBuffer,
    LiveFrame,
    LiveFrameError,
    LiveUdpReceiver,
    parse_smpl_frame,
)
from mocap_receiver.protocol import decode_json_messages, encode_smpl_frame
from mocap_receiver.server import DatagramProcessor, run_udp_loop


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class LiveVisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_payload = (REPOSITORY_ROOT / "vdsuit_udp_stream_example.json").read_bytes()
        source_messages, errors = decode_json_messages(cls.source_payload)
        if errors:
            raise AssertionError(errors)
        cls.smpl_message = SmplConverter().convert_frame(source_messages[1])

    def test_parses_converter_output(self) -> None:
        frame = parse_smpl_frame(self.smpl_message, received_at=123.0)

        self.assertEqual(frame.frame_index, 123)
        self.assertEqual(frame.pose.shape, (72,))
        self.assertEqual(frame.translation.shape, (3,))
        self.assertIsNotNone(frame.betas)
        np.testing.assert_allclose(frame.betas, np.zeros(10))
        np.testing.assert_allclose(frame.pose, 0.0)
        np.testing.assert_allclose(frame.translation, [0.0, 1.11000001, 0.0])
        self.assertEqual(frame.received_at, 123.0)

    def test_rejects_incompatible_live_frame(self) -> None:
        invalid = copy.deepcopy(self.smpl_message)
        invalid["coordinate_system"] = "some_other_axes"
        with self.assertRaisesRegex(LiveFrameError, "coordinate_system"):
            parse_smpl_frame(invalid)

        invalid = copy.deepcopy(self.smpl_message)
        invalid["body_pose"] = [0.0] * 68
        with self.assertRaisesRegex(LiveFrameError, "body_pose"):
            parse_smpl_frame(invalid)

    def test_latest_buffer_discards_render_backlog_and_counts_gaps(self) -> None:
        buffer = LatestFrameBuffer()
        first = LiveFrame(10, np.zeros(72), np.zeros(3), time.monotonic())
        second = LiveFrame(12, np.ones(72), np.ones(3), time.monotonic())

        buffer.publish(first)
        buffer.publish(second)
        latest, is_new, stats = buffer.consume_latest()

        self.assertTrue(is_new)
        self.assertIs(latest, second)
        self.assertEqual(stats.valid_frames, 2)
        self.assertEqual(stats.sequence_gaps, 1)
        self.assertEqual(stats.overwritten_frames, 1)
        _latest, is_new_again, _stats = buffer.consume_latest()
        self.assertFalse(is_new_again)

    def test_udp_receiver_publishes_valid_frame_and_survives_invalid_packet(self) -> None:
        buffer = LatestFrameBuffer()
        receiver = LiveUdpReceiver("127.0.0.1", 0, buffer)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.start()
        try:
            sender.sendto(b"not-json", receiver.address)
            sender.sendto(encode_smpl_frame(self.smpl_message), receiver.address)
            self.assertTrue(buffer.wait_for_first_frame(timeout=2.0))
        finally:
            receiver.stop()
            sender.close()

        latest, is_new, stats = buffer.consume_latest()
        self.assertTrue(is_new)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.frame_index, 123)
        self.assertEqual(stats.datagrams, 2)
        self.assertEqual(stats.valid_frames, 1)
        self.assertEqual(stats.invalid_messages, 1)

    def test_idle_udp_receiver_stops_without_a_packet(self) -> None:
        receiver = LiveUdpReceiver("127.0.0.1", 0, LatestFrameBuffer())
        receiver.start()
        started = time.monotonic()
        receiver.stop(timeout=1.0)
        self.assertLess(time.monotonic() - started, 1.0)

    def test_complete_source_to_converter_to_live_viewer_udp_chain(self) -> None:
        live_buffer = LatestFrameBuffer()
        live_receiver = LiveUdpReceiver("127.0.0.1", 0, live_buffer)
        converter_input = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        converter_input.bind(("127.0.0.1", 0))
        converter_output = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        source_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        converter_stop = threading.Event()
        converter_processor = DatagramProcessor()
        converter_thread = threading.Thread(
            target=run_udp_loop,
            args=(
                converter_input,
                converter_output,
                live_receiver.address,
                converter_processor,
                converter_stop,
            ),
            daemon=True,
        )
        live_receiver.start()
        converter_thread.start()
        try:
            source_sender.sendto(self.source_payload, converter_input.getsockname())
            self.assertTrue(live_buffer.wait_for_first_frame(timeout=2.0))
        finally:
            converter_stop.set()
            converter_thread.join(timeout=1.0)
            live_receiver.stop()
            source_sender.close()
            converter_input.close()
            converter_output.close()

        frame, is_new, live_stats = live_buffer.consume_latest()
        self.assertTrue(is_new)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.frame_index, 123)
        self.assertEqual(converter_processor.stats.skeletons, 1)
        self.assertEqual(converter_processor.stats.outputs, 1)
        self.assertEqual(live_stats.valid_frames, 1)


if __name__ == "__main__":
    unittest.main()
