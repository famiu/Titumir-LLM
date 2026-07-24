"""Shared validation and filesystem helpers for dataset scripts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from string import Formatter
from typing import Any


def validate_conversation(example: object, context: str = "example") -> dict[str, Any]:
    """Validate and return a two-turn user/assistant conversation."""
    if not isinstance(example, dict):
        raise ValueError(f"{context} must be a JSON object")

    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"{context}.messages must contain exactly two messages")

    validated_messages = []
    for index, expected_role in enumerate(("user", "assistant")):
        message = messages[index]
        if not isinstance(message, dict):
            raise ValueError(f"{context}.messages[{index}] must be a JSON object")
        if message.get("role") != expected_role:
            raise ValueError(f"{context}.messages[{index}].role must be '{expected_role}'")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{context}.messages[{index}].content must be a non-empty string")
        validated_messages.append({"role": expected_role, "content": content})

    validated = dict(example)
    validated["messages"] = validated_messages
    return validated


def normalize_text(text: str) -> str:
    """Normalize text for comparison without changing emitted source text."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def conversation_key(example: dict[str, Any]) -> str:
    """Return a canonical key for normalized conversation content."""
    messages = example["messages"]
    normalized = [{"role": message["role"], "content": normalize_text(message["content"])} for message in messages]
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_outer_json_fence(value: str) -> str:
    """Strip one optional outer Markdown JSON fence."""
    stripped = value.strip()
    if not stripped.startswith("```"):
        return stripped

    first_newline = stripped.find("\n")
    if first_newline == -1 or not stripped.endswith("```"):
        return stripped
    opening = stripped[:first_newline].strip().lower()
    if opening not in {"```", "```json"}:
        return stripped
    return stripped[first_newline + 1 : -3].strip()


def validate_prompt_fields(template: str, allowed_fields: set[str]) -> None:
    """Reject unknown or missing prompt-template placeholders."""
    fields = {field for _, field, _, _ in Formatter().parse(template) if field is not None}
    unknown = fields - allowed_fields
    missing = allowed_fields - fields
    if unknown:
        raise ValueError(f"Unknown prompt placeholder(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"Missing prompt placeholder(s): {', '.join(sorted(missing))}")


@contextmanager
def atomic_text_writer(path: str | Path) -> Iterator[Any]:
    """Write a UTF-8 text file and atomically replace the destination on success."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            yield file
        os.replace(temporary_name, destination)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
