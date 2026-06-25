"""Minimal WAV I/O helpers using Python's standard library."""

from __future__ import annotations

import wave
from pathlib import Path
import struct

import numpy as np


def _wav_format_tag(path: Path) -> int | None:
    """Return the format tag from the WAV fmt chunk when it can be read."""
    with path.open("rb") as f:
        magic = f.read(4)
        if magic.startswith(b"ID3") or magic[:2] == b"\xff\xfb":
            raise ValueError("该文件实际是 MP3 编码，不是 WAV。请先转换为 PCM WAV 后再打开。")
        if magic == b"OggS":
            raise ValueError("该文件实际是 OGG/Opus 编码，不是 WAV。请先转换为 PCM WAV 后再打开。")
        if magic == b"fLaC":
            raise ValueError("该文件实际是 FLAC 编码，不是 WAV。请先转换为 PCM WAV 后再打开。")
        if magic != b"RIFF":
            raise ValueError("文件头不是 RIFF，说明它不是标准 WAV 文件或文件已损坏。")
        f.seek(8)
        if f.read(4) != b"WAVE":
            return None
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                return None
            size_bytes = f.read(4)
            if len(size_bytes) < 4:
                return None
            chunk_size = struct.unpack("<I", size_bytes)[0]
            if chunk_id == b"fmt ":
                data = f.read(min(chunk_size, 2))
                return struct.unpack("<H", data)[0] if len(data) == 2 else None
            f.seek(chunk_size + (chunk_size % 2), 1)


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """Read a PCM WAV file and return mono float samples in [-1, 1]."""
    path = Path(path)
    format_tag = _wav_format_tag(path)
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float64)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    elif sample_width == 4:
        if format_tag == 3:
            data = np.frombuffer(frames, dtype="<f4").astype(np.float64)
            data = np.clip(data, -1.0, 1.0)
        else:
            data = np.frombuffer(frames, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return sample_rate, data


def write_wav(path: str | Path, sample_rate: int, samples: np.ndarray) -> None:
    """Write mono 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
