# Speech-to-Text Transcription System

A comprehensive speech-to-text transcription system using Hugging Face Transformers with **CPU-only inference**, benchmarking capabilities, and pluggable model architecture.

## 🎯 Features

- **CPU-Only Inference** - No GPU required
- **Pluggable Models** - Easy to add/swap models
- **Comprehensive Benchmarking** - WER, CER, RTF, speed & accuracy metrics
- **Automatic Output Saving** - Transcriptions and metrics
- **Model Comparison** - Compare multiple models on same audio

## 🚀 Quick Start

### 1. Installation

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 2. Run Transcription

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
