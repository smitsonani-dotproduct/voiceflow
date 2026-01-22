import librosa
import jiwer
from typing import Optional
from pathlib import Path

def audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    audio, sr = librosa.load(audio_path, sr=None)
    return len(audio) / sr

def real_time_factor(duration: float, processing_time: float) -> float:
    """ Real-Time Factor (RTF) is defined as processing-time / length-of-audio. """
    return processing_time / duration if duration > 0 else 0.0

def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate in percentage."""
    return jiwer.wer(reference, hypothesis) * 100

def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate in percentage."""
    return jiwer.cer(reference, hypothesis) * 100


def collect_metrics(
    audio_path: str,
    output: dict,
    reference_text: Optional[str] = None,
) -> dict:
    """Collect all benchmarking metrics for a single audio transcription."""
    
    model_info = output["model_info"]
    transcription = output["transcription"]
    processing_time = output["processing_time"]
    
    duration = audio_duration(audio_path)
    metrics = {
        "model_info": model_info,
        "audio_file": Path(audio_path).name,
        "audio_duration": round(duration, 2),
        "processing_time": round(processing_time, 2),
        "rtf": round(real_time_factor(duration, processing_time), 3),
        "char_count": len(transcription),
        "word_count": len(transcription.split()),
    }

    if reference_text:
        w = wer(reference_text, transcription)
        metrics["WER"] = round(w, 2)
        metrics["CER"] = round(cer(reference_text, transcription), 2)
        metrics["accuracy"] = round(100 - w, 2)

    return metrics
