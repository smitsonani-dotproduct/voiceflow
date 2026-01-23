from pathlib import Path
from typing import List, Optional

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