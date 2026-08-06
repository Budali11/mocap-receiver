"""UDP JSON/JSONL decoding and SMPL frame serialization."""

from __future__ import annotations

import json
from typing import Any


def decode_json_messages(payload: bytes) -> tuple[list[dict[str, Any]], list[str]]:
    """Decode one JSON object or newline-delimited JSON objects from a datagram.

    A malformed JSONL record does not prevent other valid records in the same
    datagram from being processed. Errors are returned as human-readable text.
    """

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [], [f"datagram is not valid UTF-8: {exc}"]

    if not text.strip():
        return [], ["datagram is empty"]

    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        messages: list[dict[str, Any]] = []
        errors: list[str] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                decoded_line = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"JSONL line {line_number} is invalid: {exc.msg}")
                continue
            if not isinstance(decoded_line, dict):
                errors.append(f"JSONL line {line_number} must contain a JSON object")
                continue
            messages.append(decoded_line)
        if not messages and not errors:
            errors.append("datagram contains no JSON objects")
        return messages, errors

    if not isinstance(decoded, dict):
        return [], ["datagram JSON value must be an object"]
    return [decoded], []


def encode_smpl_frame(message: dict[str, Any]) -> bytes:
    """Serialize one SMPL frame as compact UTF-8 JSON."""

    return json.dumps(
        message,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
