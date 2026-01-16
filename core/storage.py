import json
from pathlib import Path
from datetime import datetime
from typing import Dict

def save_transcription(
    audio_path: str,
    transcription: str,
    output_dir: str = "outputs",
    model_name: str = ""
) -> Path:
    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)  

    audio_name = Path(audio_path).stem
    file_path = model_dir / f"{audio_name}.txt"

    file_path.write_text(transcription, encoding="utf-8")

    return file_path

def save_metrics(
    audio_path:str,
    metrics: Dict,
    output_dir: str = "outputs",
    model_name: str = ""
) -> Path:
    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True,exist_ok=True)

    audio_name = Path(audio_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = model_dir / f"{audio_name}_{timestamp}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return file_path
