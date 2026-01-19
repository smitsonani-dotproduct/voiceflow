import torch
import time
from typing import Any
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from models.base import AudioTranscriptionModel

MODELS = {
    "whisper-tiny": {
        "model_id": "openai/whisper-tiny",
        "type": "whisper",
    },
    "whisper-base": {
        "model_id": "openai/whisper-base",
        "type": "whisper",
    },
    "distil-whisper": {
        "model_id": "distil-whisper/distil-large-v3",
        "type": "distil",
    }
}


class STTModel(AudioTranscriptionModel):
    """
    Unified Models Speech-to-Text Engine
    Supports:
      - whisper-tiny
      - whisper-base
      - distil-whisper
    """

    def __init__(self, model_key: str, device: str = "cpu"):
        if model_key not in MODELS:
            raise ValueError(f"Unsupported model: {model_key}")

        self.model_key = model_key
        self.model_config = MODELS[model_key]

        super().__init__(
            model_name=self.model_config["model_id"],
            device=device,
        )

    def load_model(self) -> None:
        start = time.time()
        device_id = -1 if self.device == "cpu" else 0

        if self.model_config["type"] == "whisper":
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
            )
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            self.model.to(self.device)

        self.pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self.model,
            device=device_id,                                
            dtype=torch.float32, 
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30
            # stride_length_s=5
        )

        self.load_time = time.time() - start

    def transcribe(self, audio_path: str) -> str:
        if self.pipeline is None:
            self.load_model()

        result = self.pipeline(
            audio_path,
            generate_kwargs={"language": "english"},
            return_timestamps=False,
        )
        
        print('Result =>', result)

        if not isinstance(result, dict) or "text" not in result:
            raise RuntimeError(f"Unexpected pipeline output: {result}")

        return result["text"].strip()
