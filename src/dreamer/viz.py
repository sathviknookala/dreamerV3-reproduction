from __future__ import annotations

import shutil
import struct
import subprocess
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


def write_video(path: Path | str, frames: np.ndarray, fps: int = 25) -> str:
    """Pipe raw rgb24 to ffmpeg; fall back to a .npz of the frames when ffmpeg is missing."""
    frames = np.ascontiguousarray(frames)

    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected uint8 (T, H, W, 3) frames, got {frames.dtype} {frames.shape}")

    path = Path(path)
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        fallback = path.with_suffix(".npz")
        np.savez_compressed(fallback, frames=frames)
        return str(fallback)

    height, width = frames.shape[1:3]
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(int(fps)),
        "-i", "-", "-an", "-pix_fmt", "yuv420p", str(path),
    ]
    subprocess.run(command, input=frames.tobytes(), check=True)
    return str(path)
