"""Tkinter GUI for dynamic audio waveform and spectrum display."""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .audio_io import read_wav, write_wav
from .signal_gen import demo_audio
from .spectrum import analyze_frame


class SpectrumAnalyzerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("FFT 音频信号频谱分析仪")
        self.root.configure(bg="#edf2f7")
        self.sample_rate = 16000
        self.samples = demo_audio(sample_rate=self.sample_rate)
        self.source_name = "演示音频"
        self.frame_size = 1024
        self.hop_size = 512
        self.position = 0
        self.running = False
        self.scale_mode = tk.StringVar(value="linear")
        self.window_name = tk.StringVar(value="hamming")
        self.sample_rate_var = tk.StringVar(value=str(self.sample_rate))
        self.fft_size_var = tk.StringVar(value="2048")
        self.status = tk.StringVar(value="已生成演示音频")
        self._drag_start_x: int | None = None
        self._drag_start_position = 0
        self._resize_job: str | None = None
        self._draw_job: str | None = None
        self._configure_style()

        self._build_ui()
        self.root.after(80, self._draw_current_frame)

    def _configure_style(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#edf2f7")
        style.configure("Toolbar.TFrame", background="#f8fafc")
        style.configure("TLabel", background="#edf2f7", foreground="#1f2937", font=("Microsoft YaHei", 10))
        style.configure("Toolbar.TLabel", background="#f8fafc")
        style.configure("Status.TLabel", background="#f8fafc", foreground="#475569")
        style.configure("TButton", font=("Microsoft YaHei", 10), padding=(10, 5))
        style.configure("TCombobox", font=("Microsoft YaHei", 10), padding=(4, 2))

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, padding=(16, 12, 16, 8))
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text="基于自主 FFT 的音频频谱分析仪",
            font=("Microsoft YaHei", 15, "bold"),
            foreground="#0f172a",
        ).pack(side=tk.LEFT)

        toolbar = ttk.Frame(self.root, padding=(10, 8), style="Toolbar.TFrame")
        toolbar.pack(fill=tk.X, padx=12, pady=(0, 8))

        ttk.Button(toolbar, text="打开 WAV", command=self.open_wav).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="生成演示", command=self.load_demo).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="播放/暂停", command=self.toggle_play).pack(side=tk.LEFT, padx=4)

        ttk.Label(toolbar, text="采样率", style="Toolbar.TLabel").pack(side=tk.LEFT, padx=(20, 4))
        sample_rate_box = ttk.Combobox(
            toolbar,
            textvariable=self.sample_rate_var,
            values=("8000", "16000", "22050", "44100"),
            state="readonly",
            width=8,
        )
        sample_rate_box.pack(side=tk.LEFT)
        sample_rate_box.bind("<<ComboboxSelected>>", self._on_sample_rate_change)
        ttk.Label(toolbar, text="Hz", style="Toolbar.TLabel").pack(side=tk.LEFT, padx=(3, 8))

        ttk.Label(toolbar, text="FFT点数", style="Toolbar.TLabel").pack(side=tk.LEFT, padx=(10, 4))
        fft_size_box = ttk.Combobox(
            toolbar,
            textvariable=self.fft_size_var,
            values=("1024", "2048", "4096"),
            state="readonly",
            width=7,
        )
        fft_size_box.pack(side=tk.LEFT)
        fft_size_box.bind("<<ComboboxSelected>>", self._on_fft_size_change)

        ttk.Label(toolbar, text="窗函数", style="Toolbar.TLabel").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Combobox(
            toolbar,
            textvariable=self.window_name,
            values=("hamming", "rectangular"),
            state="readonly",
            width=12,
        ).pack(side=tk.LEFT)

        ttk.Label(toolbar, text="频率刻度", style="Toolbar.TLabel").pack(side=tk.LEFT, padx=(20, 4))
        ttk.Combobox(
            toolbar,
            textvariable=self.scale_mode,
            values=("linear", "log"),
            state="readonly",
            width=8,
        ).pack(side=tk.LEFT)

        self.wave_canvas = tk.Canvas(
            self.root,
            width=980,
            height=210,
            bg="#ffffff",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
        )
        self.wave_canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
        self.wave_canvas.bind("<Configure>", self._on_canvas_resize)
        self.wave_canvas.bind("<ButtonPress-1>", self._start_wave_drag)
        self.wave_canvas.bind("<B1-Motion>", self._drag_waveform)
        self.wave_canvas.bind("<ButtonRelease-1>", self._end_wave_drag)
        self.spectrum_canvas = tk.Canvas(
            self.root,
            width=980,
            height=285,
            bg="#ffffff",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
        )
        self.spectrum_canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
        self.spectrum_canvas.bind("<Configure>", self._on_canvas_resize)

        status_bar = ttk.Frame(self.root, padding=(12, 5), style="Toolbar.TFrame")
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(status_bar, textvariable=self.status, style="Status.TLabel").pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(80, self._handle_resize_draw)

    def _handle_resize_draw(self) -> None:
        self._resize_job = None
        self._request_draw(0)

    def _request_draw(self, delay_ms: int = 16) -> None:
        if self._draw_job is not None:
            self.root.after_cancel(self._draw_job)
        self._draw_job = self.root.after(delay_ms, self._draw_current_frame)

    def _max_position(self) -> int:
        return max(int(self.samples.size - self.frame_size), 0)

    def _set_position(self, position: int) -> None:
        self.position = int(np.clip(position, 0, self._max_position()))
        self._request_draw(12)

    def _plot_bounds(self, canvas: tk.Canvas) -> tuple[int, int, int, int]:
        width = max(canvas.winfo_width(), 300)
        height = max(canvas.winfo_height(), 200)
        return 76, 46, width - 28, height - 48

    def _start_wave_drag(self, event: tk.Event) -> None:
        self.running = False
        self._drag_start_x = int(event.x)
        self._drag_start_position = self.position
        self.wave_canvas.configure(cursor="fleur")

    def _drag_waveform(self, event: tk.Event) -> None:
        if self._drag_start_x is None:
            return
        left, _top, right, _bottom = self._plot_bounds(self.wave_canvas)
        plot_width = max(right - left, 1)
        samples_per_pixel = self.frame_size / plot_width
        dx = int(event.x) - self._drag_start_x
        new_position = self._drag_start_position - int(round(dx * samples_per_pixel))
        self._set_position(new_position)

    def _end_wave_drag(self, _event: tk.Event) -> None:
        self._drag_start_x = None
        self.wave_canvas.configure(cursor="")

    def open_wav(self) -> None:
        initial_dir = Path("assets") if Path("assets").exists() else Path.cwd()
        path = filedialog.askopenfilename(
            parent=self.root,
            title="选择 WAV 音频文件",
            initialdir=str(initial_dir),
            filetypes=[("WAV 音频文件", "*.wav"), ("所有文件", "*.*")],
        )
        if not path:
            self.status.set("已取消打开 WAV 文件")
            return
        try:
            self.sample_rate, self.samples = read_wav(path)
        except Exception as exc:
            messagebox.showerror("无法打开 WAV", f"读取文件失败：\n{path}\n\n{exc}", parent=self.root)
            self.status.set("WAV 文件读取失败")
            return
        if self.samples.size == 0:
            messagebox.showwarning("空音频", f"文件没有可分析的采样数据：\n{path}", parent=self.root)
            self.status.set("WAV 文件为空")
            return

        self.source_name = Path(path).name
        self.sample_rate_var.set(str(self.sample_rate))
        self.position = 0
        self.running = False
        self._draw_current_frame()

    def load_demo(self) -> None:
        self.sample_rate = int(self.sample_rate_var.get())
        self.samples = demo_audio(sample_rate=self.sample_rate)
        self.source_name = "演示音频"
        self.position = 0
        output = Path("assets/demo_audio.wav")
        write_wav(output, self.sample_rate, self.samples)
        self.status.set(f"演示音频已保存：{output}")
        self._draw_current_frame()

    def _on_sample_rate_change(self, _event: tk.Event) -> None:
        self.running = False
        self.sample_rate = int(self.sample_rate_var.get())
        self.samples = demo_audio(sample_rate=self.sample_rate)
        self.source_name = "演示音频"
        self.position = 0
        output = Path("assets/demo_audio.wav")
        write_wav(output, self.sample_rate, self.samples)
        self.status.set(f"采样率已切换为 fs = {self.sample_rate} Hz，并重新生成演示音频")
        self._draw_current_frame()

    def _on_fft_size_change(self, _event: tk.Event) -> None:
        self.running = False
        self._draw_current_frame()

    def toggle_play(self) -> None:
        self.running = not self.running
        if self.running:
            self._try_play_audio()
            self._tick()

    def _try_play_audio(self) -> None:
        if sys.platform != "win32":
            return

        def worker() -> None:
            try:
                import winsound

                path = Path("assets/gui_preview.wav")
                write_wav(path, self.sample_rate, self.samples)
                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _tick(self) -> None:
        if not self.running:
            return
        self.position += self.hop_size
        if self.position + self.frame_size >= self.samples.size:
            self.position = 0
        self._draw_current_frame()
        interval_ms = int(self.hop_size / self.sample_rate * 1000)
        self.root.after(max(interval_ms, 20), self._tick)

    def _draw_current_frame(self) -> None:
        self._draw_job = None
        frame = self.samples[self.position : self.position + self.frame_size]
        if frame.size < self.frame_size:
            frame = np.pad(frame, (0, self.frame_size - frame.size))
        result = analyze_frame(
            frame,
            self.sample_rate,
            self.position,
            int(self.fft_size_var.get()),
            self.window_name.get(),
        )
        self._draw_waveform(frame)
        self._draw_spectrum(result.frequencies, result.magnitudes, result.peaks)
        peak_text = "、".join(f"{freq:.1f} Hz" for freq, _ in result.peaks[:3])
        total_time = self.samples.size / self.sample_rate
        frame_time = self.frame_size / self.sample_rate
        hop_time = self.hop_size / self.sample_rate
        fft_size = int(self.fft_size_var.get())
        frequency_resolution = self.sample_rate / fft_size
        self.status.set(
            f"{self.source_name} | 时长 {total_time:.2f}s | 帧 {self.position / self.sample_rate:.3f}-"
            f"{(self.position + self.frame_size) / self.sample_rate:.3f}s | "
            f"帧长 {frame_time:.3f}s / 帧移 {hop_time:.3f}s | "
            f"N={fft_size} / Δf={frequency_resolution:.2f}Hz | 主频：{peak_text}"
        )

    def _sample_for_canvas(self, values: np.ndarray, max_points: int) -> np.ndarray:
        if values.size <= max_points:
            return values
        indices = np.linspace(0, values.size - 1, max_points).astype(np.int64)
        return values[indices]

    def _plot_axes(
        self,
        canvas: tk.Canvas,
        title: str,
        x_label: str,
        y_label: str,
    ) -> tuple[int, int, int, int]:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 300)
        height = max(canvas.winfo_height(), 200)
        left, top, right, bottom = self._plot_bounds(canvas)
        canvas.create_rectangle(0, 0, width, height, fill="#ffffff", outline="")
        canvas.create_text(
            left,
            22,
            anchor=tk.W,
            text=title,
            font=("Microsoft YaHei", 13, "bold"),
            fill="#0f172a",
        )
        for i in range(5):
            y = top + i * (bottom - top) / 4
            canvas.create_line(left, y, right, y, fill="#e2e8f0")
        canvas.create_line(left, bottom, right + 8, bottom, fill="#334155", width=1.4, arrow=tk.LAST)
        canvas.create_line(left, bottom, left, top - 8, fill="#334155", width=1.4, arrow=tk.LAST)
        canvas.create_text((left + right) / 2, height - 18, text=x_label, font=("Microsoft YaHei", 10), fill="#475569")
        canvas.create_text(18, (top + bottom) / 2, text=y_label, angle=90, font=("Microsoft YaHei", 10), fill="#475569")
        return left, top, right, bottom

    def _draw_y_ticks(
        self,
        canvas: tk.Canvas,
        left: int,
        top: int,
        bottom: int,
        values: np.ndarray,
        formatter: str = "{:.2f}",
    ) -> None:
        for value in values:
            ratio = (value - values[0]) / (values[-1] - values[0]) if values[-1] != values[0] else 0.0
            y = bottom - ratio * (bottom - top)
            canvas.create_line(left - 5, y, left, y, fill="#334155")
            canvas.create_text(
                left - 9,
                y,
                anchor=tk.E,
                text=formatter.format(float(value)),
                fill="#64748b",
                font=("Microsoft YaHei", 9),
            )

    def _draw_waveform(self, frame: np.ndarray) -> None:
        left, top, right, bottom = self._plot_axes(
            self.wave_canvas,
            "当前分析帧的时域波形 x(t)",
            "当前帧绝对时间 t / s",
            "幅值 A",
        )
        mid = (top + bottom) / 2
        amp = max(float(np.max(np.abs(frame))), 1e-6)
        max_points = max((right - left) * 2, 200)
        plot_frame = self._sample_for_canvas(frame, max_points)
        xs = np.linspace(left, right, plot_frame.size)
        ys = mid - plot_frame / amp * (bottom - top) * 0.45
        points = [coord for xy in zip(xs, ys) for coord in xy]
        self.wave_canvas.create_line(left, mid, right, mid, fill="#cbd5e1", dash=(4, 4))
        self.wave_canvas.create_line(points, fill="#2563eb", width=1.8)

        y_ticks = np.linspace(-amp, amp, 5)
        self._draw_y_ticks(self.wave_canvas, left, top, bottom, y_ticks, "{:.2f}")

        start_time = self.position / self.sample_rate
        end_time = (self.position + self.frame_size) / self.sample_rate
        for tick in np.linspace(start_time, end_time, 5):
            ratio = (tick - start_time) / (end_time - start_time)
            x = left + ratio * (right - left)
            self.wave_canvas.create_line(x, bottom, x, bottom + 5, fill="#334155")
            self.wave_canvas.create_text(
                x,
                bottom + 18,
                text=f"{tick:.3f}",
                fill="#64748b",
                font=("Microsoft YaHei", 9),
            )

    def _draw_spectrum(
        self,
        frequencies: np.ndarray,
        magnitudes: np.ndarray,
        peaks: list[tuple[float, float]],
    ) -> None:
        scale_text = "对数频率轴" if self.scale_mode.get() == "log" else "线性频率轴"
        left, top, right, bottom = self._plot_axes(
            self.spectrum_canvas,
            f"频域幅度谱 |X(f)|（{scale_text}）",
            "频率 f / Hz",
            "幅度 |X(f)|",
        )
        max_freq = min(self.sample_rate / 2, 5000)
        mask = (frequencies >= 1.0) & (frequencies <= max_freq)
        freq = frequencies[mask]
        mag = magnitudes[mask]
        if self.scale_mode.get() == "log":
            x_values = np.log10(freq)
            x_min, x_max = np.log10(20), np.log10(max_freq)
        else:
            x_values = freq
            x_min, x_max = 0.0, max_freq
        y_max = max(float(np.max(mag)), 1e-6)
        max_points = max((right - left) * 2, 240)
        if x_values.size > max_points:
            indices = np.linspace(0, x_values.size - 1, max_points).astype(np.int64)
            x_values = x_values[indices]
            mag = mag[indices]
        xs = left + (x_values - x_min) / (x_max - x_min) * (right - left)
        ys = bottom - mag / y_max * (bottom - top)
        points = [coord for xy in zip(xs, ys) for coord in xy]
        self.spectrum_canvas.create_line(points, fill="#dc2626", width=1.8)
        self._draw_y_ticks(self.spectrum_canvas, left, top, bottom, np.linspace(0, y_max, 5), "{:.3f}")

        for freq_hz, mag_value in peaks[:4]:
            if freq_hz > max_freq:
                continue
            x_base = np.log10(freq_hz) if self.scale_mode.get() == "log" else freq_hz
            x = left + (x_base - x_min) / (x_max - x_min) * (right - left)
            y = bottom - mag_value / y_max * (bottom - top)
            self.spectrum_canvas.create_line(x, y, x, bottom, fill="#fca5a5", dash=(3, 3))
            self.spectrum_canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#111827", outline="#ffffff")
            self.spectrum_canvas.create_text(x + 7, y - 12, anchor=tk.W, text=f"{freq_hz:.1f} Hz", fill="#111827")
        for tick in np.linspace(0 if self.scale_mode.get() == "linear" else 20, max_freq, 6):
            if tick <= 0:
                x = left
            elif self.scale_mode.get() == "log":
                x = left + (np.log10(tick) - x_min) / (x_max - x_min) * (right - left)
            else:
                x = left + (tick - x_min) / (x_max - x_min) * (right - left)
            self.spectrum_canvas.create_line(x, bottom, x, bottom + 5, fill="#334155")
            self.spectrum_canvas.create_text(x, bottom + 18, text=f"{tick:.0f}", fill="#64748b", font=("Microsoft YaHei", 9))


def main() -> None:
    root = tk.Tk()
    app = SpectrumAnalyzerApp(root)
    root.minsize(900, 640)
    root.mainloop()


if __name__ == "__main__":
    main()
