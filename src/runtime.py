from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("shorts")


def safe_error(exc: BaseException) -> str:
    # SDK errors may include credentials or response bodies.
    from openai import APIStatusError
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        message = json.dumps(exc.errors(include_input=False, include_context=False, include_url=False))
    elif isinstance(exc, APIStatusError):
        # Do not dump the SDK's response body (which may echo a full request).
        message = f"HTTP {exc.status_code}: {exc.response.reason_phrase}"
    else:
        message = str(exc)
    for key, value in os.environ.items():
        if value and len(value) >= 4 and any(word in key.upper() for word in ("KEY", "TOKEN", "PASSWORD", "SECRET")):
            message = message.replace(value, "[REDACTED]")
    message = re.sub(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+", r"\1[REDACTED]", message)
    message = re.sub(r"(?i)(['\"]?authorization['\"]?\s*[:=]\s*['\"])[^'\"]+", r"\1[REDACTED]", message)
    message = re.sub(r"(?i)(['\"]?(?:api[_-]?key|token|password|secret)['\"]?\s*[:=]\s*['\"])[^'\"]+", r"\1[REDACTED]", message)
    message = re.sub(r"(?i)((?:api[_-]?key|token|password|secret)\s*[:=]\s*)[^\s,;&]+", r"\1[REDACTED]", message)
    return f"{type(exc).__name__}: {message[:2000]}"


def log_failure(stage: str, exc: Exception, **identity) -> dict:
    # Keep frame locations and exception chains, without locals, source lines or SDK payloads.
    frames = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        frames.append(safe_error(current))
        frames.extend(f"  {f.filename}:{f.lineno} in {f.name}" for f in traceback.extract_tb(current.__traceback__))
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    details = {"stage": stage, **identity, "error": safe_error(exc), "traceback": "\n".join(frames)}
    logger.error("Item failure at %s (%s): %s\n%s", stage, identity, details["error"],
                 details["traceback"], extra={"details": details})
    return details


def batch_status(successful: int, total: int) -> str:
    return "FAILED" if successful == 0 else "SUCCESS" if successful == total else "PARTIAL_SUCCESS"


class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = {"time": datetime.now(timezone.utc).isoformat(), "level": record.levelname,
                "event": record.getMessage()}
        data.update(getattr(record, "details", {}))
        return json.dumps(data, ensure_ascii=False, default=str)


@contextmanager
def log_to(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    previous = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous)


def event(name: str, **details):
    logger.info(name, extra={"details": details})


def atomic_text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_json(path: Path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def file_lock(path: Path):
    """OS locks are released on process death; the persistent file is not a stale lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another process is using this job: {path.parent}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
