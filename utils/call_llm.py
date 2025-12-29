import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

def call_llm(prompt: str) -> str:
    """Call LLM using OpenRouter API"""
    api_key = "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"
    model = "google/gemini-3-flash-preview"
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/PocketFlow", # Optional, for OpenRouter rankings
        "X-Title": "PocketFlow Video Processor", # Optional
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "reasoning": {"enabled": True}
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0]["message"]
            # Log reasoning details if present (useful for debugging)
            if "reasoning_details" in message and message["reasoning_details"]:
                logger.info("Model reasoning detected.")
            return message["content"]
        else:
            logger.error(f"Unexpected response from OpenRouter: {result}")
            return f"Error: Unexpected response format"
            
    except Exception as e:
        logger.error(f"Error calling OpenRouter: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Test script
    test_prompt = "Hello, how are you? Answer in one sentence."
    response = call_llm(test_prompt)
    print(f"Test response: {response}")
