import time
from typing import Optional
from faster_whisper import WhisperModel
from models.base import AudioTranscriptionModel


MODELS = {
    "faster-whisper-small": {
        "model_id": "small",
        "type": "faster-whisper",
    },
    "faster-whisper-medium": {
        "model_id": "medium",
        "type": "faster-whisper",
    },
    "faster-whisper-large-v3": {
        "model_id": "large-v3",
        "type": "faster-whisper",
    },
}


class FasterWhisperSTT(AudioTranscriptionModel):
    """
    Faster-Whisper Speech-to-Text Engine
    CPU-optimized (CTranslate2 based)
    """

    def __init__(
        self,
        model_key: str,
        device: str = "cpu",
        compute_type: Optional[str] = None,
    ):
        if model_key not in MODELS:
            raise ValueError(f"Unsupported model: {model_key}")

        self.model_key = model_key
        self.model_config = MODELS[model_key]

        # Auto compute type selection
        if compute_type is None:
            compute_type = "int8" if device == "cpu" else "float16"

        self.compute_type = compute_type

        super().__init__(
            model_name=self.model_config["model_id"],
            device=device,
        )

    def load_model(self) -> None:
        start = time.time()

        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

        self.load_time = time.time() - start

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            self.load_model()

        segments, _ = self.model.transcribe(
            audio_path,
            language="en",
            beam_size=5,
        )

        return " ".join(segment.text for segment in segments).strip()
