"""Signal generators used by validation, analysis, and the GUI demo."""

from __future__ import annotations

import math

import numpy as np


def time_axis(duration: float, sample_rate: int) -> np.ndarray:
    samples = int(round(duration * sample_rate))
    return np.arange(samples, dtype=np.float64) / sample_rate


def sine_wave(
    frequency: float,
    duration: float,
    sample_rate: int,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    t = time_axis(duration, sample_rate)
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


def square_wave(
    frequency: float,
    duration: float,
    sample_rate: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    t = time_axis(duration, sample_rate)
    return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))


def demo_audio(duration: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Create a small synthetic audio clip with time-varying content."""
    t = time_axis(duration, sample_rate)
    envelope = np.minimum(1.0, t / 0.2) * np.minimum(1.0, (duration - t) / 0.25)
    chord = (
        0.55 * np.sin(2 * np.pi * 440.0 * t)
        + 0.35 * np.sin(2 * np.pi * 660.0 * t)
        + 0.22 * np.sin(2 * np.pi * 880.0 * t)
    )
    vibrato = np.sin(2 * np.pi * (330.0 + 4.0 * np.sin(2 * np.pi * 5.0 * t)) * t)
    click_free = envelope * (0.75 * chord + 0.25 * vibrato)
    return normalize_audio(click_free)


def normalize_audio(samples: np.ndarray, peak: float = 0.95) -> np.ndarray:
    maximum = float(np.max(np.abs(samples))) if samples.size else 0.0
    if math.isclose(maximum, 0.0):
        return samples.astype(np.float64)
    return (samples / maximum * peak).astype(np.float64)
