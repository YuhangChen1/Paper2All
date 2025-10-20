# eval/core/llm_interface.py

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Union
from openai import AsyncOpenAI
from eval.core.utils import read_and_preprocess_image_as_base64

def _extract_json_from_string(text: str) -> str:
    """
    Extracts a JSON string from a Markdown code block (e.g., ```json ... ```).
    Returns the original string (stripped) if no code block is found.
    """
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text.strip()


async def call_llm_api(
    client: AsyncOpenAI,
    full_content_message: str,
    image_paths: List[str],
    model: str = "gemini-1.5-flash-latest",
    temperature: float = 0.01,
    tolerance: int = 5,
    response_schema: Optional[Dict[str, Any]] = None,
    n: int = 1,
    force_json_format_in_prompt: bool = False # This parameter is controlled by the new config
) -> List[Union[str, Dict[str, Any]]]:
    """
    Calls the LLM API requesting 'n' completions.
    Supports two modes for structured JSON output:
    1. Native Tool Use: Uses API's `tools` when `force_json_format_in_prompt` is False.
    2. Simulated Tool Use (via Prompt): Injects JSON structure into the prompt when `force_json_format_in_prompt` is True.
    """

    def _normalize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively converts string values in the schema to lowercase."""
        if isinstance(schema, dict):
            return {k: _normalize_schema(v) for k, v in schema.items()}
        if isinstance(schema, list):
            return [_normalize_schema(i) for i in schema]
        if isinstance(schema, str):
            return schema.lower()
        return schema

    messages = [{"role": "user", "content": []}]

    num_images = len(image_paths)
    if num_images <= 3:
        image_quality_setting = 'high'
    elif num_images <= 6:
        image_quality_setting = 'medium'
    elif num_images <= 9:
        image_quality_setting = 'low'
    else:
        image_quality_setting = 'very_low'

    processed_images_b64 = []
    for path in image_paths:
        b64_data = await read_and_preprocess_image_as_base64(path, quality=image_quality_setting)
        if b64_data:
            processed_images_b64.append(b64_data)
    
    api_kwargs = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "n": n
    }

    prompt_text = full_content_message

    if response_schema:
        if force_json_format_in_prompt:
            try:
                schema_str = json.dumps(response_schema, indent=2)
                prompt_suffix = (
                    "\n\nPlease provide your response in a JSON format enclosed within a ```json ... ``` block. "
                    "The JSON object must conform to the following schema:\n"
                    f"{schema_str}"
                )
                prompt_text += prompt_suffix
            except TypeError as e:
                return [{"status": "failed", "error": f"Failed to serialize response_schema to JSON: {e}"}]
        
        else: # Native Tool Use
            prompt_text += (
                "\n\nPlease output your response by using the 'structured_output' tool."
            )
            normalized_params = _normalize_schema(response_schema)
            api_kwargs["tools"] = [{"type": "function", "function": {"name": "structured_output", "description": "Generate a structured response.", "parameters": normalized_params}}]
            api_kwargs["tool_choice"] = {"type": "function", "function": {"name": "structured_output"}}

    messages[0]["content"].append({"type": "text", "text": prompt_text})
    for img_data in processed_images_b64:
        messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}})


    async def _call_llm_once() -> List[Union[str, Dict[str, Any]]]:
        try:
            response = await client.chat.completions.create(**api_kwargs)
            
            parsed_results = []
            for choice in response.choices:
                if choice.message.tool_calls:
                    tool_arguments_str = choice.message.tool_calls[0].function.arguments
                    try:
                        parsed_results.append(json.loads(tool_arguments_str))
                    except json.JSONDecodeError as e:
                        parsed_results.append({"status": "failed", "error": f"JSON parse error in tool call: {e}", "raw_response": tool_arguments_str})
                
                elif choice.message.content:
                    text_content = choice.message.content
                    if response_schema:
                        cleaned_text = _extract_json_from_string(text_content)
                        if not cleaned_text:
                            parsed_results.append({"status": "failed", "error": "Extracted JSON is empty", "raw_response": text_content})
                            continue
                        try:
                            parsed_results.append(json.loads(cleaned_text))
                        except json.JSONDecodeError as e:
                            parsed_results.append({"status": "failed", "error": f"JSON parse error after cleaning: {e}", "raw_response": text_content})
                    else:
                        parsed_results.append(text_content)
                
                else:
                    parsed_results.append({"status": "failed", "error": "Empty response from choice", "raw_response": None})
            
            return parsed_results
            
        except Exception as e:
            return [{"status": "failed", "error": f"API call failed: {e}"}]

    retry = 0
    final_results = []
    while retry < tolerance:
        final_results = await _call_llm_once()
        is_api_error = (
            len(final_results) == 1 and 
            isinstance(final_results[0], dict) and 
            "API call failed" in final_results[0].get("error", "")
        )
        if not is_api_error:
            break
        retry += 1
        print(f"WARNING: LLM API call retry {retry}/{tolerance}. Error: {final_results[0].get('error')}")
        await asyncio.sleep(0.1)

    return final_results