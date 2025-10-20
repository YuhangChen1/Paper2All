import re
import os
import cv2
import pdb
import json
import torch
import string
import subprocess
from os import path
import multiprocessing as mp
from transformers import pipeline
from ui_tars.action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code
import whisperx
from whisperx import load_audio
from whisperx.alignment import align
from openai import OpenAI
import base64
import mimetypes

try:
    print("INFO: Initializing OpenRouter API client...")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    MODEL_ID = "bytedance/ui-tars-1.5-7b" 
    print(f"INFO: OpenRouter client initialized. Using model: {MODEL_ID}")
except Exception as e:
    print(f"ERROR: Failed to initialize OpenRouter client. Check API key. Error: {e}")
    client = None
whisperx_model = "large-v3"

def encode_image_to_base64(image_path):
    """Encodes a local image file into a Base64 string for API calls."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_string}"

def draw_red_dots_on_image(image_path, point, radius: int = 5) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    red = (0, 0, 255)
    x, y = int(point[0]), int(point[1])
    cv2.circle(image, (x, y), radius, red, thickness=-1)
    cv2.imwrite("output.jpg", image)

def parse_script(script_text):
    pages = script_text.strip().split("###\n")
    result = []
    for page in pages:
        if not page.strip(): continue
        lines = page.strip().split("\n")
        page_data = []
        for line in lines:
            if "|" not in line: 
                continue
            text, cursor = line.split("|", 1)
            page_data.append([text.strip(), cursor.strip()])
        result.append(page_data)
    return result

def _clamp_point(x, y, w, h):
    return max(0, min(w - 1, x)), max(0, min(h - 1, y))

def _extract_xy_from_text(text):
    patterns = [
        r"point\s*=\s*'\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)'",
        r"start_box\s*=\s*'\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)'",
        r"point\s*=\s*'<point>\s*([0-9.]+)\s+([0-9.]+)\s*</point>'",
        r"pyautogui\.click\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1)), float(m.group(2))
            except Exception:
                continue
    return None

def _fallback_ui_tars_xy(api_text, w, h):
    try:
        parsed = parse_action_to_structure_output(
            api_text,
            factor=1000,
            origin_resized_height=h,
            origin_resized_width=w,
            model_type="qwen25vl",
        )
        if not isinstance(parsed, dict) or not parsed:
            return w / 2, h / 2

        code = parsing_response_to_pyautogui_code(
            responses=parsed,
            image_height=h,
            image_width=w,
        )
        m = re.search(r"pyautogui\.click\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", code)
        if not m:
            return w / 2, h / 2
        x, y = float(m.group(1)), float(m.group(2))
        return _clamp_point(x, y, w, h)
    except Exception as e:
        print(f"WARNING: ui_tars fallback failed: {e}")
        return w / 2, h / 2

def infer_cursor(instruction, image_path, device):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = img.shape[:2]

    prompt = (
        "You are a GUI agent. You are given a task and your action history, with screenshots. "
        "You must to perform the next action to complete the task.\n\n"
        "## Output Format\n\nAction: ...\n\n"
        "## Action Space\nclick(point='<point>x1 y1</point>')\n\n"
        f"## User Instruction {instruction}"
    )

    api_text = None
    token_usage = 0
    try:
        base64_image = encode_image_to_base64(image_path)
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": base64_image}},
                ],
            }],
            max_tokens=300,
        )
        api_text = resp.choices[0].message.content or ""
        print(f"DEBUG: API Raw Response -> {api_text}")
        token_usage = len(prompt) + len(api_text)
    except Exception as e:
        print(f"ERROR: API call failed for '{instruction}': {e}")
        api_text = "Action: click(point='<point>500 500</point>')"
        token_usage = 0

    xy = _extract_xy_from_text(api_text)
    if xy is not None:
        x, y = _clamp_point(xy[0], xy[1], w, h)
        return (x, y), token_usage

    print("INFO: Direct regex failed, falling back to ui_tars parser.")
    x_fb, y_fb = _fallback_ui_tars_xy(api_text, w, h)
    return (x_fb, y_fb), token_usage
    
def infer(args):
    slide_idx, sentence_idx, prompt, cursor_prompt, image_path, _ = args
    point, token = infer_cursor(cursor_prompt, image_path, device="api")
    result = {'slide': slide_idx, 'sentence': sentence_idx, 'speech_text': prompt, 'cursor_prompt': cursor_prompt, 'cursor': point, 'token': token}
    return result

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def get_audio_length(audio_path):
    command = ["ffmpeg", "-i", audio_path]
    result = subprocess.run(command, stderr=subprocess.PIPE, text=True)
    for line in result.stderr.splitlines():
        if "Duration" in line:
            duration_str = line.split("Duration:")[1].split(",")[0].strip()
            hours, minutes, seconds = map(float, duration_str.split(":"))
            return hours * 3600 + minutes * 60 + seconds
    return 0 

def timesteps(subtitles, aligned_result, audio_path):
    aligned_words_in_order = []
    for idx, segment in enumerate(aligned_result["segments"]):
        aligned_words_in_order.extend(segment["words"])
    aligned_words_num = len(aligned_words_in_order) - 1
    
    result = []
    current_idx = 0
    for idx, sentence in enumerate(subtitles):
        words_num = len(re.findall(r'\b\w+\b', sentence.lower()))
        start = aligned_words_in_order[min(aligned_words_num, current_idx)]["end"]
        
        current_idx += words_num
        end = aligned_words_in_order[min(aligned_words_num, current_idx)]["end"]

        duration = {"start": start, "end": end, "text": sentence}
        result.append(duration)
    
    result[0]["start"] = 0
    result[-1]["end"] = get_audio_length(audio_path)
    return result

def cursor_gen_per_sentence(script_path, slide_img_dir, slide_audio_dir, cursor_save_path, gpu_list):
    with open(script_path, 'r') as f:script_with_cursor = ''.join(f.readlines())
    parsed_speech = parse_script(script_with_cursor)
    cursor_token = 0 
    
    slide_imgs = [name for name in os.listdir(slide_img_dir)]
    slide_imgs = sorted(slide_imgs, key=lambda x: int(re.search(r'\d+', x).group()))
    print(slide_imgs)
    slide_imgs = [path.join(slide_img_dir, name) for name in slide_imgs]
    
    ## location
    num_gpus = len(gpu_list)
    process_idx = 0
    task_list = []
    for slide_idx in range(len(parsed_speech)):
        speech_with_cursor = parsed_speech[slide_idx]
        print(slide_idx)
        image_path = slide_imgs[slide_idx]
        for sentence_idx, (prompt, cursor_prompt) in enumerate(speech_with_cursor):
            gpu_id = gpu_list[process_idx % num_gpus]
            task_list.append((slide_idx, sentence_idx, prompt, cursor_prompt, image_path, gpu_id))
            process_idx += 1  
    
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1) as pool: cursor_result = pool.map(infer, task_list)
    
    slide_w, slide_h = cv2.imread(slide_imgs[0]).shape[:2]
    for index in range(len(cursor_result)):
        if cursor_result[index]["cursor_prompt"] == "no":
            cursor_result[index]["cursor"] == (slide_w//2, slide_h//2)
        cursor_token += cursor_result[index]["token"]
          
    ## timesteps
    slide_sentence_timesteps = []
    slide_audio = os.listdir(slide_audio_dir)
    slide_audio = sorted(slide_audio, key=lambda x: int(re.search(r'\d+', x).group()))
    slide_audio = [path.join(slide_audio_dir, name) for name in slide_audio]
    model = whisperx.load_model(whisperx_model, device="cuda")
    align_model, metadata = whisperx.load_align_model(language_code="en", device="cuda")
    
    for idx, slide_audio_path in enumerate(slide_audio):
        ## get slide subtitle
        subtitle = []
        cursor = []
        for info in cursor_result: 
            if info["slide"] == idx: 
                subtitle.append(clean_text(info["speech_text"]))
                cursor.append(info["cursor"])
        ## word timesteps  
        audio = load_audio(slide_audio_path)
        result = model.transcribe(slide_audio_path, language="en")
        aligned = align(transcript=result["segments"], align_model_metadata=metadata, model=align_model, audio=audio, device="cuda")
        sentence_timesteps = timesteps(subtitle, aligned, slide_audio_path) # get_sentence_timesteps(subtitle, aligned, slide_audio_path)
        for idx in range(len(sentence_timesteps)): sentence_timesteps[idx]["cursor"] = cursor[idx]
        slide_sentence_timesteps.append(sentence_timesteps)
    # merage
    start_time_now = 0
    new_slide_sentence_timesteps = []
    for sentence_timesteps in slide_sentence_timesteps:
        duration = 0
        for idx in range(len(sentence_timesteps)):
            if sentence_timesteps[idx]["start"] is None: sentence_timesteps[idx]["start"] = sentence_timesteps[idx-1]["end"]
            if sentence_timesteps[idx]["end"] is None: sentence_timesteps[idx]["end"] = sentence_timesteps[idx+1]["start"]

        for idx in range(len(sentence_timesteps)):
            sentence_timesteps[idx]["start"] += start_time_now
            sentence_timesteps[idx]["end"] += start_time_now
            duration += sentence_timesteps[idx]["end"] - sentence_timesteps[idx]["start"]
        start_time_now += duration
        new_slide_sentence_timesteps.extend(sentence_timesteps)
    
    with open(cursor_save_path.replace(".json", "_mid.json"), 'w') as f: json.dump(cursor_result, f, indent=2)
    with open(cursor_save_path, 'w') as f: json.dump(new_slide_sentence_timesteps, f, indent=2)
    return cursor_token/4