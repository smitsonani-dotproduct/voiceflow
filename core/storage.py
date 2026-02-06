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


def save_sentiment_analysis_result(
    transcription_file: str,
    sentiment_data: dict,
    token_usage: dict,
    model_name: str = "gpt-4o"
) -> Path:
    """
    Save sentiment analysis results and token usage to JSON file.
    Saves in a 'sentiment_analysis' subfolder within the model directory.
    Example structure: outputs/faster-whisper-small/sentiment_analysis/CallLog_1101101216_sentiment_20260206_1757.json
    """
    # Get the directory of the transcription file (e.g., outputs/faster-whisper-small)
    transcription_path = Path(transcription_file)
    model_dir = transcription_path.parent
    
    # Create sentiment_analysis subfolder
    sentiment_dir = model_dir / "sentiment_analysis"
    sentiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the result dictionary
    result = {
        "transcription_file": str(transcription_file),
        "model": model_name,
        "sentiment_analysis": sentiment_data,
        "token_usage": token_usage,
    }
    
    # Create filename with timestamp
    file_stem = transcription_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result_file = sentiment_dir / f"{file_stem}_{timestamp}.json"
    
    # Save to JSON file
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return result_file
