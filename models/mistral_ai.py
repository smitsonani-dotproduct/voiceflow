import time
from mistralai import Mistral, File
from models.base import MODELS


class MistralAISTT:
    """
    Mistral AI Speech-to-Text Engine
    Cloud-based API with diarization support
    """

    def __init__(self, model_key: str):
        """
        Initialize Mistral AI STT
        """
        if model_key not in MODELS:
            raise ValueError(f"Unsupported model: {model_key}")

        self.model_key = model_key
        self.model_config = MODELS[model_key]
        self.model_name = self.model_config["model_id"]
        
        # Get API key from model config
        self.api_key = self.model_config.get("api_key")
        if not self.api_key:
            raise ValueError(
                "MISTRAL_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
        
        # Get transcription parameters
        self.diarize = self.model_config.get("diarize", False)
        self.timestamp_granularities = self.model_config.get(
            "timestamp_granularities", ["segment"]
        )
        
        self.model = None
        self.load_time = 0.0

    def load_model(self) -> None:
        """Initialize Mistral AI client"""
        start = time.time()
        self.model = Mistral(api_key=self.api_key)
        self.load_time = time.time() - start

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file using Mistral AI API
        """
        if self.model is None:
            self.load_model()

        # Open and send audio file to Mistral AI
        with open(audio_path, "rb") as f:
            response = self.model.audio.transcriptions.complete(
                model=self.model_name,
                file=File(content=f, file_name=f.name),
                diarize=self.diarize,
                timestamp_granularities=self.timestamp_granularities,
            )

            for segment in response.segments:
                speaker = segment.speaker_id or "unknown"
                print(
                    f"[{segment.start:.1f}s → {segment.end:.1f}s] {speaker}: {segment.text.strip()}"
                )
        
            return response.text.strip()

    def info(self):
        """Return model information"""
        return {
            "model_name": self.model_name,
            "model_key": self.model_key,
            "type": "mistral-ai",
            "load_time": self.load_time,
            "diarize": self.diarize,
            "timestamp_granularities": self.timestamp_granularities,
        }
