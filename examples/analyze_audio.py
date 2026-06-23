"""生成/读取音频，执行分帧频谱分析，并输出中文图表与结果说明。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio_io import read_wav, write_wav
from src.signal_gen import demo_audio
from src.plotting import save_spectrum_plot, save_waveform_plot
from src.spectrum import analyze_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="音频信号频谱分析演示")
    parser.add_argument("--input", type=Path, help="可选：输入 WAV 音频路径")
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--frame-size", type=int, default=2048)
    parser.add_argument("--hop-size", type=int, default=1024)
    parser.add_argument("--window", choices=["hamming", "rectangular"], default="hamming")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        sample_rate, samples = read_wav(args.input)
        wav_path = args.input
    else:
        sample_rate = args.sample_rate
        samples = demo_audio(sample_rate=sample_rate)
        wav_path = args.output_dir / "demo_audio.wav"
        write_wav(wav_path, sample_rate, samples)

    results = analyze_audio(
        samples,
        sample_rate,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        n_fft=args.n_fft,
        window_name=args.window,
    )
    middle = results[len(results) // 2]

    frame_start = middle.frame_start
    frame_end = min(frame_start + args.frame_size, samples.size)
    wave_segment = samples[frame_start:frame_end]
    if wave_segment.size < args.frame_size:
        wave_segment = np.pad(wave_segment, (0, args.frame_size - wave_segment.size), mode="constant")
    t = (np.arange(args.frame_size, dtype=np.float64) + frame_start) / sample_rate
    waveform_path = save_waveform_plot(
        args.output_dir / "时域波形.png",
        t,
        wave_segment,
        "中间分析帧的时域波形 x(t)",
    )
    hamming_path = save_spectrum_plot(
        args.output_dir / "汉明窗幅度谱.png",
        middle.frequencies,
        middle.magnitudes,
        middle.peaks,
        "汉明窗处理后的幅度谱 |X(f)|",
        max_frequency=3000,
    )

    rectangular = analyze_audio(
        samples,
        sample_rate,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        n_fft=args.n_fft,
        window_name="rectangular",
    )[len(results) // 2]
    rectangular_path = save_spectrum_plot(
        args.output_dir / "矩形窗幅度谱.png",
        rectangular.frequencies,
        rectangular.magnitudes,
        rectangular.peaks,
        "矩形窗处理后的幅度谱 |X(f)|",
        max_frequency=3000,
    )

    peak_csv = args.output_dir / "主要频率成分.csv"
    with peak_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["序号", "频率/Hz", "幅度", "窗函数", "结果说明"])
        for rank, (frequency, magnitude) in enumerate(middle.peaks, start=1):
            writer.writerow(
                [
                    rank,
                    f"{frequency:.3f}",
                    f"{magnitude:.8f}",
                    "汉明窗",
                    "该频率为当前分析帧中的局部谱峰，幅度越大表示该成分越显著。",
                ]
            )

    explanation_path = args.output_dir / "频谱分析结果说明.md"
    explanation_lines = [
        "# 频谱分析结果说明",
        "",
        f"- 音频文件：`{wav_path}`",
        f"- 采样率：`{sample_rate} Hz`",
        f"- FFT 点数：`{args.n_fft}`",
        f"- 频率分辨率：`Δf = fs / N = {sample_rate / args.n_fft:.4f} Hz`",
        f"- 分析帧数：`{len(results)}`",
        f"- 中间分析帧范围：`{frame_start / sample_rate:.4f} s ~ {(frame_start + args.frame_size) / sample_rate:.4f} s`",
        f"- 默认输出主频来自音频中间帧，窗函数为：`汉明窗`。",
        "",
        "## 主要频率成分",
        "",
        "| 序号 | 频率/Hz | 幅度 | 说明 |",
        "|---:|---:|---:|---|",
    ]
    for rank, (frequency, magnitude) in enumerate(middle.peaks, start=1):
        explanation_lines.append(
            f"| {rank} | {frequency:.3f} | {magnitude:.8f} | "
            "该频点为局部谱峰；演示音频中前三个峰分别对应约 440 Hz、660 Hz、880 Hz 的合成分量。 |"
        )
    explanation_lines.extend(
        [
            "",
            "## 输出图片说明",
            "",
            f"- `{waveform_path.name}`：中间分析帧的时域波形，横轴为绝对时间 `t/s`，纵轴为幅值 `A`。",
            f"- `{hamming_path.name}`：同一中间分析帧加汉明窗后的幅度谱，横轴为频率 `f/Hz`，纵轴为幅度 `|X(f)|`。",
            f"- `{rectangular_path.name}`：同一中间分析帧加矩形窗后的幅度谱，用于和汉明窗结果比较。",
            "",
            "汉明窗通常能降低旁瓣泄漏，使主频附近谱线更平滑；矩形窗主瓣较窄，但旁瓣较高，频谱泄漏更明显。",
        ]
    )
    explanation_path.write_text("\n".join(explanation_lines), encoding="utf-8")

    print(f"音频文件：{wav_path}")
    print(f"采样率：{sample_rate} Hz，FFT 点数：{args.n_fft}，频率分辨率：{sample_rate / args.n_fft:.4f} Hz")
    print(f"已分析帧数：{len(results)}")
    print("主要频率成分（汉明窗，中间帧）：")
    for rank, (frequency, magnitude) in enumerate(middle.peaks, start=1):
        print(f"{rank}. 频率 = {frequency:.2f} Hz，幅度 = {magnitude:.5f}")
    print("结果说明：前三个主峰对应演示音频中约 440 Hz、660 Hz、880 Hz 的合成分量；频点偏差由频率分辨率决定。")
    print(f"已保存图片、CSV 与说明文件到：{args.output_dir}")


if __name__ == "__main__":
    main()
