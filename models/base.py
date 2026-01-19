from abc import ABC, abstractmethod
from typing import Dict, Any

class AudioTranscriptionModel(ABC):
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.pipeline = None
        self.load_time = 0.0

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        pass

    def info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "load_time": self.load_time,
        }
