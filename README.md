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
python scripts\stt_pipeline.py
```

## 📊 Available Models

### Model Details

#### 1. Whisper Tiny (`whisper-tiny`)
- **Model**: `openai/whisper-tiny`
- **Size**: 39M parameters
- **Speed**: ~3-5x faster than base
- **Use Case**: Quick drafts, fast processing
- **Memory**: ~150MB RAM

#### 2. Whisper Base (`whisper-base`)
- **Model**: `openai/whisper-base`
- **Size**: 74M parameters
- **Speed**: Balanced
- **Use Case**: General purpose transcription
- **Memory**: ~300MB RAM

#### 3. Distil-Whisper (`distil-whisper`)
- **Model**: `distil-whisper/distil-large-v3`
- **Size**: 756M parameters (distilled for efficiency)
- **Speed**: 6x faster than Whisper Large
- **Use Case**: Best accuracy on CPU
- **Memory**: ~3GB RAM

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
