import time
from models.whisper import TransformerBasedSTTModel
from models.faster_whisper import FasterWhisperSTT
from models.mistral_ai import MistralAISTT
from models.base import MODELS

def transcribe_audio(file_path: str, model_name: str, device: str = "cpu") -> dict:
    if model_name not in MODELS:
        raise ValueError(
            f"Model '{model_name}' not supported. "
            f"Choose from {list(MODELS.keys())}"
        )

    model_type = MODELS[model_name]["type"]

    if model_type == "faster-whisper":
        model = FasterWhisperSTT(
            model_key=model_name,
            device=device,
        )
    elif model_type == "mistral-ai":
        model = MistralAISTT(
            model_key=model_name,
        )
    else:
        model = TransformerBasedSTTModel(
            model_key=model_name,
            device=device,
        )

    model.load_model()

    start = time.time()
    transcription = model.transcribe(file_path)
    processing_time = time.time() - start

    return {
        "transcription": transcription,
        "processing_time": processing_time,
        "model_info": model.info(),
    }
