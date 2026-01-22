import torch
import time
# from typing import Any
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from models.base import AudioTranscriptionModel, MODELS

class TransformerBasedSTTModel(AudioTranscriptionModel):
    """
    Unified Models Speech-to-Text Engine
    CPU + GPU compatible Transformer-based STT
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

        self.is_gpu = torch.cuda.is_available() and device.startswith("cuda")
        self.dtype = torch.float16 if self.is_gpu else torch.float32

    def _get_pipeline_device(self) -> int:
        """
        HuggingFace pipeline expects:
        -1 for CPU
         N for cuda:N
        """
        if not self.is_gpu:
            return -1
        if self.device == "cuda":
            return 0
        return int(self.device.split(":")[1])

    def load_model(self) -> None:
        start = time.time()
        device_id = self._get_pipeline_device()

        if self.model_config["type"] == "openai-whisper":
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype=self.dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                dtype=self.dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )

        self.model.to(self.device)

        self.pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            dtype=self.dtype,
            device=device_id,
            chunk_length_s=30,
            # stride_length_s=5
        )

        self.load_time = time.time() - start

    def transcribe(self, audio_path: str) -> str:
        if self.pipeline is None:
            self.load_model()

        generate_kwargs = {}

        if self.model_config.get("multilingual"):
            generate_kwargs["language"] = "english"

        result = self.pipeline(
            audio_path,
            generate_kwargs=generate_kwargs,
            return_timestamps=False,
        )
        
        print('Result =>', result)

        if not isinstance(result, dict) or "text" not in result:
            raise RuntimeError(f"Unexpected pipeline output: {result}")

        return result["text"].strip()
