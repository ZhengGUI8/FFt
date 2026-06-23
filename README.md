# 基于 FFT 的音频信号频谱分析仪

这是一个纯 Python 课程设计项目，完成了自主 FFT、合成信号验证、音频分帧加窗频谱分析，以及实时动态频谱 GUI。

## 环境

- Python 3.10+
- numpy
- matplotlib（用于输出 PNG 实验图）
- tkinter（Python 标准库，Windows/macOS 常规安装自带）

核心 FFT 位于 `src/fft_core.py`，没有调用 `numpy.fft.fft`。`numpy.fft.fft` 只在 `examples/validate_fft.py` 中作为验证参考。

## 运行

验证 FFT：

```powershell
python examples\validate_fft.py
```

生成演示音频、频谱图和主频 CSV：

```powershell
python examples\analyze_audio.py
```

启动实时频谱分析 GUI：

```powershell
python app.py
```

GUI 支持：

- 打开 WAV 文件或生成演示音频
- 播放/暂停动态帧刷新
- 汉明窗/矩形窗切换
- 线性/对数频率刻度切换
- 同时显示时域波形和频域幅度谱，并标注主要频率

## 项目结构

```text
src/
  fft_core.py       自主 FFT、IFFT、单边幅度谱、主频检测
  spectrum.py       分帧、加窗、逐帧频谱分析
  audio_io.py       WAV 读取和写入
  signal_gen.py     正弦波、方波、演示音频生成
  gui_tk.py         Tkinter 动态可视化界面
  plotting.py       matplotlib 中文 PNG 图表输出
examples/
  validate_fft.py   与 numpy.fft 的误差对比
  analyze_audio.py  音频频谱分析和图表生成
assets/
  demo_audio.wav
  FFT验证结果.json
  FFT验证结果说明.md
  时域波形.png
  汉明窗幅度谱.png
  矩形窗幅度谱.png
  主要频率成分.csv
  频谱分析结果说明.md
docs/
  report.md         课程设计报告
```
