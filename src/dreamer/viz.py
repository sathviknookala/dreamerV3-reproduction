from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def _chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def write_png(path: Path | str, rgb: np.ndarray) -> None:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"expected uint8 HWC RGB, got {rgb.dtype} {rgb.shape}")

    height, width = rgb.shape[:2]
    rows = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(rows, 9))
        + _chunk(b"IEND", b"")
    )


def paired_filmstrip(
    observed: np.ndarray,
    predicted: np.ndarray,
    scale: int = 2,
    gap: int = 2,
) -> np.ndarray:
    """Observed frames above their open-loop predictions, one column per prediction distance."""
    if observed.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {observed.shape} vs {predicted.shape}")

    def row(frames: np.ndarray) -> np.ndarray:
        frames = np.repeat(np.repeat(frames, scale, axis=1), scale, axis=2)
        separator = np.zeros((frames.shape[0], frames.shape[1], gap, 3), dtype=np.uint8)
        padded = np.concatenate([frames, separator], axis=2)
        n, h, w = padded.shape[:3]
        return padded.transpose(1, 0, 2, 3).reshape(h, n * w, 3)

    top, bottom = row(observed), row(predicted)
    divider = np.zeros((gap, top.shape[1], 3), dtype=np.uint8)
    return np.concatenate([top, divider, bottom], axis=0)
