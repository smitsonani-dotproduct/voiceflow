from core.audio_input import get_audio_file
from core.transcription import transcribe_audio
from core.storage import save_transcription, save_metrics
from core.metrics import collect_metrics

def main():
    filename = 'call_recording_01.wav'
    file_path = get_audio_file(filename)
    print('file_path =>',file_path)
    
    model_name = "whisper-tiny"
    result = transcribe_audio(
        file_path=file_path,
        model_name=model_name
    )
    print('\n~~~  Result ~~~\n', result)
    
    save_transcription(
        audio_path=file_path,
        transcription=result["transcription"],
        model_name=model_name
    )
    print('\n~~~  Transcription saved ~~~\n')

    reference_text = """
    Hello, I'm Sarah Miller. I'm calling to inquire about the AC 7892 air conditioner unit. I saw it on your website and I had a few questions. First, what's the BTU rating? And second, does it come with a remote control or is that sold separately?
    """
    metrics = collect_metrics(
        audio_path=file_path,
        transcription=result["transcription"],
        processing_time=result["processing_time"],
        model_name=model_name,
        reference_text=reference_text
    )
    print('\n~~~ metrics ~~~\n',metrics)

    save_metrics(audio_path=file_path, metrics=metrics, model_name=model_name)

    print("\n~~~   Transcription complete   ~~~")


if __name__ == "__main__":
    main()
