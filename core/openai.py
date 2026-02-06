"""
Simple OpenAI API integration for sentiment analysis.
"""

import os
from openai import OpenAI
from typing import Dict, Any, Optional
from models.base import getModelConfig

def getLLMModelResponse(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    Simple function to call OpenAI API for text analysis.
    """
    model_config = getModelConfig(model)
    api_key = model_config.get("openAIApiKey")
    model_name = model_config.get("modelName")
    temperature = model_config.get("temperature")
    max_tokens = model_config.get("maxTokens")
    response_format = model_config.get("modelKwargs").get("response_format")

    if not api_key:
        raise ValueError(
            "OpenAI API key not provided. "
            "Set OPENAI_API_KEY environment variable."
        )
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Default prompts if not provided
    if system_prompt is None or user_prompt is None:
        raise ValueError(
            "System prompt or user prompt not provided. "
            "Set system_prompt and user_prompt parameters."
        )
    

    # Prepare API call parameters
    api_params = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        # "max_tokens": max_tokens
        "max_completion_tokens": max_tokens
    }
    
    # Add response format if specified
    if response_format:
        api_params["response_format"] = response_format
    
    # Call OpenAI API
    try:
        response = client.chat.completions.create(**api_params)
        
        # Extract and return results
        result = {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason
        }
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {str(e)}")
