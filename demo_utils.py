"""Small dependency-free helpers shared by the release demo and stage entry points."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence, TypeVar


T = TypeVar("T")


def parse_subject_ids(value: str | Sequence[int | str] | None) -> list[str]:
    """Normalize subject IDs to zero-padded strings while preserving order."""
    if value is None:
        return []
    if isinstance(value, str):
        tokens: Iterable[int | str] = value.replace(";", ",").split(",")
    else:
        tokens = value

    result: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        text = str(token).strip()
        if not text:
            continue
        try:
            number = int(text)
        except ValueError as exc:
            raise ValueError(f"Invalid subject ID: {text!r}") from exc
        if number < 0:
            raise ValueError(f"Subject IDs must be non-negative: {text!r}")
        subject_id = f"{number:04d}"
        if subject_id not in seen:
            seen.add(subject_id)
            result.append(subject_id)
    return result


def filter_subject_items(
    items: Sequence[tuple[str, T]], requested: Sequence[int | str] | str | None
) -> list[tuple[str, T]]:
    """Select subject items in requested order and reject unavailable IDs."""
    requested_ids = parse_subject_ids(requested)
    if not requested_ids:
        return list(items)

    by_id = {parse_subject_ids([subject_id])[0]: payload for subject_id, payload in items}
    missing = [subject_id for subject_id in requested_ids if subject_id not in by_id]
    if missing:
        raise ValueError(
            "Requested subjects are missing from the input dataset: " + ", ".join(missing)
        )
    return [(subject_id, by_id[subject_id]) for subject_id in requested_ids]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, payload: object) -> None:
    """Write a JSON document atomically so interrupted demos do not leave valid-looking manifests."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
