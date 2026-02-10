"""
Test sentiment analysis on already transcribed files from outputs folder.
This script reads existing transcription files and runs sentiment analysis on them.
"""

from core.openai import getLLMModelResponse
from core.prompt import get_prompt, formatSmartTemplate
from core.storage import save_sentiment_analysis_result
from core.audio_processing import get_transcription_file
import json
from pathlib import Path


def test_sentiment_on_transcription(transcription_file: str):
    """
    Run sentiment analysis on a single transcription file.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {Path(transcription_file).name}")
    print(f"{'='*60}")
    
    # Read the transcription
    with open(transcription_file, 'r', encoding='utf-8') as f:
        transcription_text = f.read()
    
    # Get prompts from centralized configuration
    prompt = get_prompt("sentiment_analysis")
    
    # Format user message with transcription text
    formatted_user_message = formatSmartTemplate(
        prompt["user_message"],
        {"text": transcription_text}
    )
    # print(f"Final user message:\n{formatted_user_message}")
    
    
    # Call OpenAI API
    result_sentiment = getLLMModelResponse(
        system_prompt=prompt["system_message"],
        user_prompt=formatted_user_message,
        model="gpt-5-mini",
    )
    
    # Parse the JSON response
    sentiment_data = json.loads(result_sentiment["content"])

    print('\n~~~ Sentiment Analysis Result ~~~')
    print(sentiment_data)
    
    # Save results to JSON file
    saved_file = save_sentiment_analysis_result(
        transcription_file=transcription_file,
        sentiment_data=sentiment_data,
        token_usage=result_sentiment['usage'],
        model_name=result_sentiment['model']
    )
    
    print(f"\n✅ Results saved to: {saved_file}")
    
    return sentiment_data
        

def main():
    """
    Main function to test sentiment analysis on a specific transcription file.
    Change the variables below to test different files.
    """
    # ===== CHANGE THESE VARIABLES TO TEST DIFFERENT FILES =====
    stt_model_name = "whisper-small-en"
    # faster-whisper-tiny
    # faster-whisper-base
    # faster-whisper-small
    # faster-whisper-medium (keep it last)
    # distil-whisper-small-en
    # distil-whisper-medium-en
    # whisper-tiny-en
    # whisper-base-en
    # whisper-small-en  

    file_name = "CallLog_9997307961.txt"
    # CallLog_1101101216
    # CallLog_1101103039
    # CallLog_1101103086
    # CallLog_1101105569
    # CallLog_1101106941
    # CallLog_2815624367
    # CallLog_2878500995 
    # CallLog_4078702335 
    # CallLog_4393101944
    # CallLog_5593943197
    # CallLog_5880420498
    # CallLog_6321240367
    # CallLog_6422145041
    # CallLog_6456770480
    # CallLog_6824048799
    # CallLog_7265120246
    # CallLog_7642861972
    # CallLog_9295409840
    # CallLog_9614990365
    # CallLog_9997307961
    
    # Get transcription file path using the helper function
    file_path = get_transcription_file(
        filename=file_name,
        stt_model_name=stt_model_name
    )
    
    # Run sentiment analysis on the specified file
    test_sentiment_on_transcription(file_path)
        

if __name__ == "__main__":
    main()
