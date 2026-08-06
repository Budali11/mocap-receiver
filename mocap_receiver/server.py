"""Real-time UDP receiver and forwarding loop."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import socket
import threading
from typing import Any, BinaryIO, Mapping

from .converter import ConversionError, SmplConverter
from .protocol import decode_json_messages, encode_smpl_frame
from .recording import SmplNpzRecorder


LOGGER = logging.getLogger(__name__)
IDLE_POLL_SECONDS = 0.2


@dataclass
class ReceiverStats:
    datagrams: int = 0
    messages: int = 0
    skeletons: int = 0
    frames: int = 0
    outputs: int = 0
    dropped: int = 0


class DatagramProcessor:
    """Protocol-level state independent of socket ownership."""

    def __init__(self, converter: SmplConverter | None = None) -> None:
        self.converter = converter or SmplConverter()
        self.stats = ReceiverStats()

    def process(self, payload: bytes) -> list[bytes]:
        """Process a datagram and return zero or more encoded output frames."""

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
            LOGGER.info("Installed and validated source skeleton")
            return None
        if message_type == "frame":
            converted = self.converter.convert_frame(message)
            self.stats.frames += 1
            return converted
        raise ConversionError(f"unsupported message type {message_type!r}")


def run_udp_loop(
    input_socket: socket.socket,
    output_socket: socket.socket | None,
    target_address: tuple[str, int] | None,
    processor: DatagramProcessor,
    stop_event: threading.Event | None = None,
    receive_size: int = 65535,
    output_stream: BinaryIO | None = None,
    motion_recorder: SmplNpzRecorder | None = None,
) -> ReceiverStats:
    """Run the receive/convert loop and write to configured caller-owned sinks."""

    if target_address is not None and output_socket is None:
        raise ValueError("output_socket is required when target_address is configured")
    if target_address is None and output_stream is None and motion_recorder is None:
        raise ValueError("at least one UDP or file output must be configured")

    # A finite timeout is required even in the CLI's single-threaded mode.
    # In particular, a blocking Winsock recvfrom() may otherwise delay
    # KeyboardInterrupt indefinitely while no skeleton/frame packets arrive.
    input_socket.settimeout(IDLE_POLL_SECONDS)
    while stop_event is None or not stop_event.is_set():
        try:
            payload, _source_address = input_socket.recvfrom(receive_size)
        except socket.timeout:
            continue
        except OSError:
            if stop_event is not None and stop_event.is_set():
                break
            raise
        for output in processor.process(payload):
            if motion_recorder is not None:
                motion_recorder.record_encoded(output)
            if output_stream is not None:
                # Flush each JSONL record so a recording remains usable if the
                # process is interrupted or the machine loses power.
                output_stream.write(output + b"\n")
                output_stream.flush()
            if target_address is not None and output_socket is not None:
                try:
                    output_socket.sendto(output, target_address)
                except OSError as exc:
                    processor.stats.dropped += 1
                    LOGGER.error("Failed to forward converted frame: %s", exc)
    return processor.stats
