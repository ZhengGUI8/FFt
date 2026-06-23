"""Framing, windowing, and spectrum analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fft_core import dominant_frequencies, rfft_magnitude


@dataclass(frozen=True)
class SpectrumResult:
    frequencies: np.ndarray
    magnitudes: np.ndarray
    peaks: list[tuple[float, float]]
    frame_start: int
    window_name: str


def window_values(name: str, length: int) -> np.ndarray:
    normalized = name.strip().lower()
    if normalized in {"rect", "rectangular", "rectangle", "矩形窗"}:
        return np.ones(length, dtype=np.float64)
    if normalized in {"hamming", "汉明窗"}:
        n = np.arange(length, dtype=np.float64)
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / (length - 1))
    raise ValueError(f"unsupported window: {name}")


def iter_frames(
    samples: np.ndarray,
    frame_size: int = 1024,
    hop_size: int = 512,
) -> tuple[int, np.ndarray]:
    """Yield overlapping frames. The last short frame is zero padded."""
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive")
    if samples.size == 0:
        return
    for start in range(0, samples.size, hop_size):
        frame = samples[start : start + frame_size]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size), mode="constant")
        yield start, frame
        if start + frame_size >= samples.size:
            break


def analyze_frame(
    frame: np.ndarray,
    sample_rate: int,
    frame_start: int = 0,
    n_fft: int = 1024,
    window_name: str = "hamming",
    peak_count: int = 5,
) -> SpectrumResult:
    windowed = frame[:n_fft] * window_values(window_name, min(frame.size, n_fft))
    if windowed.size < n_fft:
        windowed = np.pad(windowed, (0, n_fft - windowed.size), mode="constant")
    frequencies, magnitudes = rfft_magnitude(windowed, sample_rate, n_fft=n_fft)
    peaks = dominant_frequencies(frequencies, magnitudes, count=peak_count)
    return SpectrumResult(frequencies, magnitudes, peaks, frame_start, window_name)


def analyze_audio(
    samples: np.ndarray,
    sample_rate: int,
    frame_size: int = 1024,
    hop_size: int = 512,
    n_fft: int = 1024,
    window_name: str = "hamming",
) -> list[SpectrumResult]:
    results: list[SpectrumResult] = []
    for start, frame in iter_frames(samples, frame_size, hop_size):
        results.append(analyze_frame(frame, sample_rate, start, n_fft, window_name))
    return results
