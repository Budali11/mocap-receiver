"""Command-line entry point for the real-time UDP service."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import socket
from typing import BinaryIO, Sequence

from .recording import SmplNpzRecorder
from .server import DatagramProcessor, run_udp_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receive VD Suit UDP frames and forward standard SMPL parameters."
    )
    parser.add_argument(
        "--listen-host",
        default="0.0.0.0",
        help="local IPv4 address to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--listen-port",
        required=True,
        type=_port,
        help="local UDP port that receives VD Suit messages",
    )
    parser.add_argument(
        "--target-host",
        default="127.0.0.1",
        help="destination IPv4 address or hostname (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--target-port",
        type=_port,
        help="optional destination UDP port for converted SMPL frames",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="optional local .npz SMPL motion or .jsonl stream file",
    )
    parser.add_argument(
        "--output-format",
        choices=("auto", "smpl-npz", "jsonl"),
        default="auto",
        help="file format; auto selects smpl-npz for .npz, otherwise jsonl",
    )
    parser.add_argument(
        "--mocap-framerate",
        type=_positive_float,
        default=60.0,
        help="frame rate stored in a SMPL .npz file (default: 60)",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--append-output",
        action="store_true",
        help="append to --output-file if it already exists",
    )
    output_mode.add_argument(
        "--overwrite-output",
        action="store_true",
        help="replace --output-file if it already exists",
    )
    parser.add_argument(
        "--receive-size",
        default=65535,
        type=_positive_int,
        help="maximum UDP payload bytes to read (default: 65535)",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="logging verbosity (default: INFO)",
    )
    return parser


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return result


def resolve_output_format(path: Path, requested_format: str) -> str:
    if requested_format != "auto":
        return requested_format
    return "smpl-npz" if path.suffix.lower() == ".npz" else "jsonl"


def open_output_file(path: Path, append: bool, overwrite: bool) -> BinaryIO:
    """Open a binary JSONL sink with an explicit existing-file policy."""

    if append:
        mode = "ab"
    elif overwrite:
        mode = "wb"
    else:
        mode = "xb"
    return path.open(mode)


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
    if output_format == "smpl-npz" and args.append_output:
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
    motion_recorder: SmplNpzRecorder | None = None
    processor = DatagramProcessor()
    try:
        input_socket.bind((args.listen_host, args.listen_port))
        if args.output_file is not None:
            try:
                if output_format == "smpl-npz":
                    motion_recorder = SmplNpzRecorder(
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
            logging.getLogger(__name__).info(
                "Recording %s to %s", output_format, args.output_file
            )
        if args.target_port is not None:
            logging.getLogger(__name__).info(
                "Listening on udp://%s:%d and forwarding to udp://%s:%d",
                args.listen_host,
                args.listen_port,
                args.target_host,
                args.target_port,
            )
        else:
            logging.getLogger(__name__).info(
                "Listening on udp://%s:%d", args.listen_host, args.listen_port
            )
        run_udp_loop(
            input_socket,
            output_socket,
            (args.target_host, args.target_port) if args.target_port is not None else None,
            processor,
            receive_size=args.receive_size,
            output_stream=output_stream,
            motion_recorder=motion_recorder,
        )
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Stopped by user")
    finally:
        input_socket.close()
        if output_socket is not None:
            output_socket.close()
        if output_stream is not None:
            output_stream.close()
        if motion_recorder is not None:
            motion_recorder.close()

    stats = processor.stats
    logging.getLogger(__name__).info(
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
