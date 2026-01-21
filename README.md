# Speech-to-Text Transcription System

A comprehensive speech-to-text transcription system using Hugging Face Transformers with **CPU-only inference**, benchmarking capabilities, and pluggable model architecture.

## 🎯 Features

- **CPU-Only Inference** - No GPU required
- **Pluggable Models** - Easy to add/swap models
- **Comprehensive Benchmarking** - WER, CER, RTF, speed & accuracy metrics
- **Automatic Output Saving** - Transcriptions and metrics
- **Model Comparison** - Compare multiple models on same audio

## 🚀 Quick Start

### 1. Installation Python Dependency

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 2. Install FFmpeg (Required for Audio Processing)

FFmpeg is required for processing various audio formats. Install it based on your operating system:

**Ubuntu/Debian Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Fedora/RHEL/CentOS:**
```bash
sudo dnf install ffmpeg
# Or for older versions:
sudo yum install ffmpeg
```

**macOS:**

Using Homebrew (recommended):
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install ffmpeg
brew install ffmpeg
```

Using MacPorts:
```bash
sudo port install ffmpeg
```

**Windows:**

# Install ffmpeg

1. Download FFmpeg from: https://ffmpeg.org/download.html#build-windows
2. Extract the zip file to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your system PATH:
   - Right-click 'This PC' → Properties → Advanced System Settings
   - Click 'Environment Variables'
   - Under 'System Variables', find 'Path' and click 'Edit'
   - Click 'New' and add `C:\ffmpeg\bin`
   - Click OK on all windows
4. Restart your terminal/command prompt

**Verify FFmpeg Installation:**
```bash
ffmpeg -version
```

### 3. Run Transcription

```bash
python -m scripts.stt_pipeline
```

## 📊 Available Models

### Model Details

#### OpenAI whisper variants:

| Size   | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
| ------ | ---------: | ------------------ | ------------------ | ------------- | -------------- |
| tiny   |        39M | `tiny.en`          | `tiny`             | ~1 GB         | ~10×           |
| base   |        74M | `base.en`          | `base`             | ~1 GB         | ~7×            |
| small  |       244M | `small.en`         | `small`            | ~2 GB         | ~4×            |
| medium |       769M | `medium.en`        | `medium`           | ~5 GB         | ~2×            |
| large  |      1550M | N/A                | `large`            | ~10 GB        | 1×             |
| turbo  |       809M | N/A                | `turbo`            | ~6 GB         | ~8×            |

#### Other ASR Models:

| Model Key                 | HF Model ID                      | Parameters | Language Support | Speed / Relative              | Typical Use Case                           | Approx. Memory | Source              |
| ------------------------- | -------------------------------- | ---------- | ---------------- | ----------------------------- | ------------------------------------------ | -------------- | ------------------- |
| `distil-whisper`          | `distil-whisper/distil-large-v3` | ~756M      | **English only** | ~6× faster than Whisper large | High-accuracy English transcription on CPU | ~3 GB          | ([Hugging Face][1]) |
| `faster-whisper-tiny`     | `tiny.en` (Systran)              | 39M        | English          | Very fast (tiny model)        | Ultra-fast English on CPU/GPU              | ~1 GB          | ([Hugging Face][2]) |
| `faster-whisper-base`     | `base.en` (Systran)              | 74M        | English          | Fast                          | Balanced speed/accuracy English            | ~1 GB          | ([Hugging Face][3]) |
| `faster-whisper-small`    | `small.en` (Systran)             | 244M       | English          | Moderate speed                | Mid-range English accuracy                 | ~2 GB          | ([Hugging Face][4]) |
| `faster-whisper-medium`   | `medium.en` (Systran)            | 769M       | English          | Slower than small             | High-accuracy English                      | ~5 GB          | ([Hugging Face][5]) |
| `faster-whisper-large-v3` | `large-v3` (Systran)             | ~1550M     | Multilingual     | Slower                        | Top accuracy multilingual model            | ~10 GB         | ([Hugging Face][6]) |

[1]: https://huggingface.co/distil-whisper "Huggubg Face - huggingface/distil-whisper: Distilled variant of Whisper for speech recognition. 6x faster, 50% smaller, within 1% word error rate."
[2]: https://huggingface.co/Systran/faster-whisper-tiny.en "Systran/faster-whisper-tiny.en · Hugging Face"
[3]: https://huggingface.co/Systran/faster-whisper-base.en "Systran/faster-whisper-base.en · Hugging Face"
[4]: https://huggingface.co/Systran/faster-whisper-small.en "Systran/faster-whisper-small.en · Hugging Face"
[5]: https://huggingface.co/Systran/faster-whisper-medium "Systran/faster-whisper-medium · Hugging Face"
[6]: https://huggingface.co/Systran/faster-whisper-large-v3 "Systran/faster-whisper-large-v3 · Hugging Face"


## 📈 Benchmark Metrics

The system automatically calculates and saves the following metrics:

### Performance Metrics
- **Processing Time**: Total time to transcribe
- **Real-Time Factor (RTF)**: Processing time / Audio duration
  - RTF < 1.0 = Faster than real-time
  - RTF = 1.0 = Real-time
  - RTF > 1.0 = Slower than real-time
- **Model Load Time**: Time to load model into memory

### Accuracy Metrics (requires reference text)
- **WER (Word Error Rate)**: Percentage of word errors
- **CER (Character Error Rate)**: Percentage of character errors
- **Accuracy**: 100 - WER

### Output Metrics
- **Audio Duration**: Length of audio file
- **Word Count**: Number of words transcribed
- **Transcription Length**: Character count

## 🔍 Understanding WER (Word Error Rate)

WER measures transcription accuracy:
- **0-5%**: Excellent (professional quality)
- **5-10%**: Good (usable for most purposes)
- **10-20%**: Fair (may need correction)
- **>20%**: Poor (significant errors)

Formula: `WER = (Substitutions + Deletions + Insertions) / Total Words × 100`
