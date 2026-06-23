"""Self-contained FFT implementation for the audio spectrum analyzer.

The core algorithm deliberately does not call numpy.fft.  NumPy is used only
for convenient array storage and vectorized pre/post-processing.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def next_power_of_two(n: int) -> int:
    """Return the smallest power of two that is greater than or equal to n."""
    if n < 1:
        raise ValueError("n must be positive")
    return 1 << (n - 1).bit_length()


def _bit_reverse_indices(n: int) -> np.ndarray:
    bits = int(math.log2(n))
    indices = np.arange(n, dtype=np.uint32)
    reversed_indices = np.zeros(n, dtype=np.uint32)
    for _ in range(bits):
        reversed_indices = (reversed_indices << 1) | (indices & 1)
        indices >>= 1
    return reversed_indices.astype(np.int64)


def fft(signal: Iterable[complex], n: int | None = None) -> np.ndarray:
    """Compute the FFT with iterative radix-2 Cooley-Tukey butterflies.

    Args:
        signal: Real or complex input samples.
        n: Optional transform length. If omitted, the input is padded to the
            next power of two. If provided and not a power of two, it is also
            rounded up to the next power of two.

    Returns:
        Complex frequency-domain samples with length equal to the padded FFT
        size.
    """
    x = np.asarray(list(signal), dtype=np.complex128)
    if x.size == 0:
        raise ValueError("FFT input cannot be empty")

    target = next_power_of_two(x.size if n is None else n)
    if target < x.size:
        x = x[:target]
    elif target > x.size:
        x = np.pad(x, (0, target - x.size), mode="constant")

    n_fft = x.size
    x = x[_bit_reverse_indices(n_fft)].copy()

    size = 2
    while size <= n_fft:
        half = size // 2
        twiddles = np.exp(-2j * np.pi * np.arange(half) / size)
        for start in range(0, n_fft, size):
            even = x[start : start + half].copy()
            odd = x[start + half : start + size] * twiddles
            x[start : start + half] = even + odd
            x[start + half : start + size] = even - odd
        size *= 2

    return x


def ifft(spectrum: Iterable[complex]) -> np.ndarray:
    """Compute the inverse FFT using conjugation."""
    spectrum_array = np.asarray(list(spectrum), dtype=np.complex128)
    result = np.conjugate(fft(np.conjugate(spectrum_array)))
    return result / spectrum_array.size


def rfft_magnitude(
    signal: Iterable[float],
    sample_rate: int,
    n_fft: int = 1024,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-sided frequencies and magnitudes for a real signal."""
    spectrum = fft(signal, n_fft)
    half = n_fft // 2 + 1
    magnitudes = np.abs(spectrum[:half])
    if normalize:
        magnitudes = magnitudes * 2.0 / n_fft
        magnitudes[0] /= 2.0
        if n_fft % 2 == 0:
            magnitudes[-1] /= 2.0
    frequencies = np.arange(half) * sample_rate / n_fft
    return frequencies, magnitudes


def dominant_frequencies(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    count: int = 5,
    min_frequency: float = 20.0,
) -> list[tuple[float, float]]:
    """Pick the strongest spectral peaks above min_frequency."""
    if frequencies.size != magnitudes.size:
        raise ValueError("frequencies and magnitudes must have the same length")
    mask = frequencies >= min_frequency
    freq = frequencies[mask]
    mag = magnitudes[mask]
    if mag.size == 0:
        return []

    # Local maxima avoid returning several neighboring bins for the same tone.
    local = np.zeros_like(mag, dtype=bool)
    if mag.size >= 3:
        local[1:-1] = (mag[1:-1] >= mag[:-2]) & (mag[1:-1] >= mag[2:])
    else:
        local[:] = True
    candidates = np.where(local)[0]
    if candidates.size == 0:
        candidates = np.arange(mag.size)
    order = candidates[np.argsort(mag[candidates])][::-1]
    return [(float(freq[i]), float(mag[i])) for i in order[:count]]
