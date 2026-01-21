import time
from models.whisper import TransformerBasedSTTModel
from models.faster_whisper import FasterWhisperSTT

MODELS = {
    # Openai Whisper
    "whisper-tiny": {
        "model_id": "openai/whisper-tiny",
        "type": "openai-whisper",
    },
    "whisper-tiny-en": {
        "model_id": "openai/whisper-tiny.en",
        "type": "openai-whisper",
    },
    "whisper-base": {
        "model_id": "openai/whisper-base",
        "type": "openai-whisper",
    },

    # Distil Whisper
    "distil-whisper": {
        "model_id": "distil-whisper/distil-large-v3",
        "type": "distil-whisper",
    },

    # Faster Whisper
    "faster-whisper-tiny": {
        "model_id": "tiny.en",
        "type": "faster-whisper",
    },
    "faster-whisper-base": {
        "model_id": "base.en",
        "type": "faster-whisper",
    },
    "faster-whisper-small": {
        "model_id": "small.en",
        "type": "faster-whisper",
    },
    "faster-whisper-medium": {
        "model_id": "medium.en",
        "type": "faster-whisper",
    },
    "faster-whisper-large-v3": {
        "model_id": "large-v3",
        "type": "faster-whisper",
    },
}


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
