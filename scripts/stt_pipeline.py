from core.audio_input import get_audio_file, load_reference_text
from core.transcription import transcribe_audio
from core.storage import save_transcription, save_metrics
from core.metrics import collect_metrics

def main():
    filename = 'call_recording_04.wav'
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

    reference_text = load_reference_text(file_path=filename)
    
    metrics = collect_metrics(
        audio_path=file_path,
        output=result,
        reference_text=reference_text
    )
    print('\n~~~ metrics ~~~\n',metrics)

    save_metrics(audio_path=file_path, metrics=metrics, model_name=model_name)

    print("\n~~~   Transcription complete   ~~~")


if __name__ == "__main__":
    main()
