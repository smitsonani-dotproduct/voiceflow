import os
from abc import ABC, abstractmethod
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()

MODELS: Dict[str, Dict[str, Any]] = {
    # STT models
    # OpenAI Whisper
    "whisper-tiny": {
        "model_id": "openai/whisper-tiny",
        "type": "openai-whisper",
        "multilingual": True,
        "size_mb": 151
    },
    "whisper-base": { 
        "model_id": "openai/whisper-base",
        "type": "openai-whisper",
        "multilingual": True,
        "size_mb": 290
    },
    "whisper-small": {
        "model_id": "openai/whisper-small",
        "type": "openai-whisper",
        "multilingual": True,
        "size_mb": 967
    },
    "whisper-medium": { 
        "model_id": "openai/whisper-medium",
        "type": "openai-whisper",
        "multilingual": True,
        "size_mb": 3060
    },
    "whisper-large-v3-turbo": {
        "model_id": "openai/whisper-large-v3-turbo",
        "type": "openai-whisper",
        "multilingual": True,
        "size_mb": 1620
    },
    "whisper-tiny-en": {
        "model_id": "openai/whisper-tiny.en",
        "type": "openai-whisper",
        "multilingual": False,
        "size_mb": 151
    },
    "whisper-base-en": {
        "model_id": "openai/whisper-base.en",
        "type": "openai-whisper",
        "multilingual": False,
        "size_mb": 290
    },
    "whisper-small-en": {
        "model_id": "openai/whisper-small.en",
        "type": "openai-whisper",
        "multilingual": False,
        "size_mb": 967
    },
    "whisper-medium-en": {
        "model_id": "openai/whisper-medium.en",
        "type": "openai-whisper",
        "multilingual": False,
        "size_mb": 3060
    },
    
    # Distil Whisper
    "distil-whisper-small-en": {
        "model_id": "distil-whisper/distil-small.en",
        "type": "distil-whisper",
        "multilingual": False,
        "size_mb": 332
    }, 
    "distil-whisper-medium-en": {
        "model_id": "distil-whisper/distil-medium.en",
        "type": "distil-whisper",
        "multilingual": False,
        "size_mb": 789
    }, 
    "distil-whisper-large-v3": {
        "model_id": "distil-whisper/distil-large-v3",
        "type": "distil-whisper",
        "multilingual": True,
        "size_mb": 1510
    },

    # Faster Whisper
    "faster-whisper-tiny": {
        "model_id": "tiny.en",
        "type": "faster-whisper",
        "default_language": "en",
        "size_mb": 78
    },
    "faster-whisper-base": {
        "model_id": "base.en",
        "type": "faster-whisper",
        "default_language": "en",
        "size_mb": 148
    },
    "faster-whisper-small": {
        "model_id": "small.en",
        "type": "faster-whisper",
        "default_language": "en",
        "size_mb": 486
    },
    "faster-whisper-medium": {
        "model_id": "medium.en",
        "type": "faster-whisper",
        "default_language": "en",
        "size_mb": 1530
    },
    "faster-whisper-large-v3": {
        "model_id": "large-v3",
        "type": "faster-whisper",
        "default_language": None,
        "size_mb": 3090
    },
    
    
    # Cloud API LLMs 
    "o4-mini":{
        "provider": "OpenAI",
        "openAIApiKey": os.getenv("OPENAI_API_KEY"),
        "modelName": "o4-mini",
        "temperature": 1,
        "timeout":3000000,
        "maxTokens": 100000,
        "modelKwargs": {
            "response_format": { "type": "json_object" },
        },
    },
    "gpt-5-mini": {
        "provider": "OpenAI",
        "openAIApiKey": os.getenv("OPENAI_API_KEY"),
        "modelName": "gpt-5-mini",
        "temperature": 1,
        "timeout":3000000,
        "maxTokens": 128000,
        "modelKwargs": {
            "response_format": { "type": "json_object" },
        },
    },
    "gpt-5.2": { # BEST reasoning model
        "provider": "OpenAI",
        "openAIApiKey": os.getenv("OPENAI_API_KEY"),
        "modelName": "gpt-5.2",
        "temperature": 1,
        "timeout":3000000,
        "maxTokens": 128000,
        "modelKwargs": {
            "response_format": { "type": "json_object" },
        },
    },
}


def getModelConfig(model_key: str) -> Dict[str, Any]:
    """
    Get model configuration by model key.
    """
    if model_key not in MODELS:
        available_models = ", ".join(MODELS.keys())
        raise ValueError(
            f"Model '{model_key}' not found. "
            f"Available models: {available_models}"
        )
    
    return MODELS[model_key]


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