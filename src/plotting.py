"""Matplotlib plotting helpers for PNG output."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
        return plt
    except Exception as exc:
        raise RuntimeError("生成图片需要安装 matplotlib，请先运行：pip install -r requirements.txt") from exc


def save_waveform_plot(
    path: str | Path,
    times: np.ndarray,
    samples: np.ndarray,
    title: str = "音频信号时域波形 x(t)",
) -> Path:
    """Save a waveform figure as PNG."""
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("图片输出必须使用 .png 后缀")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _load_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    ax.plot(times, samples, color="#2563eb", linewidth=1.8)
    ax.axhline(0, color="#94a3b8", linewidth=0.9, linestyle="--")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("时间 t / s")
    ax.set_ylabel("幅值 A")
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_spectrum_plot(
    path: str | Path,
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    peaks: list[tuple[float, float]],
    title: str,
    max_frequency: float | None = None,
) -> Path:
    """Save a spectrum figure as PNG."""
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("图片输出必须使用 .png 后缀")
    path.parent.mkdir(parents=True, exist_ok=True)
    if max_frequency is not None:
        mask = frequencies <= max_frequency
        frequencies = frequencies[mask]
        magnitudes = magnitudes[mask]

    plt = _load_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    ax.plot(frequencies, magnitudes, color="#dc2626", linewidth=1.8)
    ax.fill_between(frequencies, magnitudes, color="#fecaca", alpha=0.38)
    y_max = max(float(np.max(magnitudes)), 1e-9)
    for frequency, magnitude in peaks[:5]:
        if max_frequency is not None and frequency > max_frequency:
            continue
        ax.scatter([frequency], [magnitude], color="#111827", s=28, zorder=3)
        ax.annotate(
            f"{frequency:.1f} Hz",
            xy=(frequency, magnitude),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
            color="#111827",
        )
        ax.vlines(frequency, 0, magnitude, color="#f87171", linewidth=0.9, linestyles="dashed")
    ax.set_ylim(0, y_max * 1.16)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("频率 f / Hz")
    ax.set_ylabel("幅度 |X(f)|")
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
