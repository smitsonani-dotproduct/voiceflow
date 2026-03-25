from core.audio_processing import load_reference_text, get_audio_files
from core.transcription import transcribe_audio
from core.storage import save_combined_results
from core.metrics import collect_metrics
from core.openai import getLLMModelResponse
from core.prompt import get_prompt, formatSmartTemplate
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


def process_single_file(file_path: str, file_uuid: str, model_name: str) -> dict:
    """
    Process a single audio file: transcribe and analyze sentiment.
    Returns a dictionary with the combined results.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {file_uuid}")
    print(f"File path: {file_path}")
    print(f"{'='*60}")
    
    try:
        # Transcribe audio
        result = transcribe_audio(
            file_path=file_path,
            model_name=model_name
        )
        print(f"✅ Transcription complete for {file_uuid}")
        
        # Sentiment analysis with OpenAI
        print(f"🔄 Starting sentiment analysis for {file_uuid}")
        
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
            model="gpt-5-mini",
        )
        
        # Parse the JSON response
        sentiment_data = json.loads(result_sentiment["content"])
        
        print(f"✅ Sentiment analysis complete for {file_uuid}")
        
        # Return combined result
        return {
            "overview": {
                "uuid": file_uuid
            },
            "ai_overview": {
                "transcription": result["transcription"],
                "segments": result.get("segments", []),
                "summary": sentiment_data.get("summary", ""),
                "category": sentiment_data.get("category", ""),
                "satisfaction_score": sentiment_data.get("satisfaction_score", 0),
                "resolution": sentiment_data.get("resolution", "")
            }
        }
    
    except Exception as e:
        print(f"❌ Error processing {file_uuid}: {str(e)}")
        return {
            "overview": {
                "uuid": file_uuid
            },
            "ai_overview": {
                "error": str(e)
            }
        }


def main():
    # Configuration
    samples_dir = "samples/audio/2026-01"
    model_name = "voxtral-mini-latest"
    output_dir = "outputs/2026-01"
    max_workers = 3  # Number of parallel workers
    
    print(f"🎯 Starting STT Pipeline with {max_workers} parallel workers")
    print(f"Model: {model_name}")
    print(f"Audio directory: {samples_dir}\n")
    
    # Load all audio files from folder
    audio_files = get_audio_files(samples_dir)
    print(f"📁 Found {len(audio_files)} audio file(s) to process\n")
    
    # Process files in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_uuid = {
            executor.submit(process_single_file, file_path, file_uuid, model_name): file_uuid
            for file_path, file_uuid in audio_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_uuid):
            file_uuid = future_to_uuid[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"❌ Task failed for {file_uuid}: {str(e)}")
                results.append({
                    "overview": {"ext_uuid": file_uuid},
                    "ai_overview": {"error": str(e)}
                })
    
    # Save all results to single JSON file
    saved_file = save_combined_results(
        results=results,
        model_name=model_name,
        output_dir=output_dir
    )
     
    print(f"\n{'='*60}")
    print(f"✅ All processing complete!")
    print(f"📁 Results saved to: {saved_file}")
    print(f"📊 Processed {len(results)} file(s)")
    print(f"{'='*60}")
    

if __name__ == "__main__":
    main()
