from core.audio_input import get_audio_file, load_reference_text
from core.transcription import transcribe_audio
from core.storage import save_transcription, save_metrics
from core.metrics import collect_metrics
from core.openai import getLLMModelResponse
from core.prompt import get_prompt, formatSmartTemplate
import json

def main():
    filename = 'CallLog_1101101216.wav'
    file_path = get_audio_file(filename)
    print('file_path =>',file_path)
    
    model_name = "faster-whisper-small"

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

    reference_text = load_reference_text(file_path="CallLog_1101101216.txt")
    # print('\n~~~  Reference text ~~~\n', reference_text)
    
    metrics = collect_metrics(
        audio_path=file_path,
        output=result,
        reference_text=reference_text
    )
    print('\n~~~ metrics ~~~\n',metrics)

    save_metrics(audio_path=file_path, metrics=metrics, model_name=model_name)

    print("\n~~~   Transcription complete   ~~~")

    # Sentiment analysis with OpenAI
    print("\n~~~   Starting sentiment analysis   ~~~")
    
    # Get prompts from centralized configuration
    prompt = get_prompt("sentiment_analysis")

    # Format user message with transcription text
    formatted_user_message = formatSmartTemplate(
        prompt["user_message"],
        {"text": result["transcription"]}
    )
    
    result_sentiment = getLLMModelResponse(
        system_prompt=prompt["system_message"],
        user_prompt=formatted_user_message,
        model="gpt-4o",
    )
    
    # Parse the JSON response
    sentiment_data = json.loads(result_sentiment["content"])
    
    print('\n~~~ Sentiment Analysis Result ~~~')
    print(f"Sentiment data:\n",sentiment_data)
        
    # Save results to JSON file
    saved_file = save_sentiment_analysis_result(
        transcription_file=file_path,
        sentiment_data=sentiment_data,
        token_usage=result_sentiment['usage'],
        model_name=result_sentiment['model']
    )
    
    print(f"\n✅ Results saved to: {saved_file}")
    

if __name__ == "__main__":
    main()
