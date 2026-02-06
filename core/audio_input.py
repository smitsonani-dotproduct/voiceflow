from pathlib import Path
from typing import Optional

SUPPORTED_FORMATS = (".wav", ".mp3", ".flac", ".mp4")

def get_audio_file(
    filename: str,
    samples_dir: str = "samples/audio"
) -> str:
    """
    Validate and return a single audio file path.
    """

    samples_path = Path(samples_dir)
    file_path = samples_path / filename

    if not samples_path.exists():
        raise FileNotFoundError(f"Directory not found: {samples_dir}")

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Not a file: {file_path}")

    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format: {file_path.suffix}. "
            f"Supported formats: {SUPPORTED_FORMATS}"
        )

    return str(file_path)

def load_reference_text(
    file_path: str,
    transcript_dir: str = "samples/transcript",
) -> Optional[str]:
    """
    Load reference transcript matching the audio filename.
    """
    audio_name = Path(file_path).stem
    transcript_path = Path(transcript_dir) / f"{audio_name}.txt"

    if transcript_path.exists():
        return transcript_path.read_text(encoding="utf-8").strip()

    return None

def get_transcription_file(filename: str, stt_model_name: str, outputs_dir: str = "outputs") -> str:
    """
    Get transcription file path from outputs directory.
    """
    outputs_path = Path(outputs_dir)
    model_dir = outputs_path / stt_model_name
    file_path = model_dir / filename
    
    if not outputs_path.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Transcription file not found: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"Not a file: {file_path}")
    
    return str(file_path)