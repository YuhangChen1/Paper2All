import os, re, pdb, json
from PIL import Image
import pytesseract

import whisperx
import argparse
import torch
import numpy as np
from os import path
from pathlib import Path
from typing import List
from camel.models import ModelFactory
from camel.types import ModelType, ModelPlatformType
from camel.configs import GeminiConfig


os.environ["GEMINI_API_KEY"] = ""
prompt_path = "./prompt/content_sim_score.txt"

agent_config = {
    "model_type": ModelType.GEMINI_2_5_FLASH,
    "model_config": GeminiConfig().as_dict(),
    "model_platform": ModelPlatformType.GEMINI,}
actor_model = ModelFactory.create(
    model_platform=agent_config['model_platform'],
    model_type=agent_config['model_type'],
    model_config_dict=agent_config['model_config'],)

def extract_slide_texts(slide_dir):
    slide_texts = []
    for fname in sorted(os.listdir(slide_dir)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(slide_dir, fname)
            text = pytesseract.image_to_string(Image.open(path))
            slide_texts.append(text.strip())
    return slide_texts

def load_subtitles(sub_path):
    with open(sub_path, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]

def build_prompt(slides_1, subs_1, slides_2, subs_2):
    prompt = (
        "Human Presentation:\n"
        "Slides:\n" + "\n".join(slides_1) + "\n"
        "Subtitles:\n" + "\n".join(subs_1) + "\n\n"
        "Generated Presentation:\n"
        "Slides:\n" + "\n".join(slides_2) + "\n"
        "Subtitles:\n" + "\n".join(subs_2) + "\n\n")
    return prompt

def run_similarity_eval(slide_dir_1, slide_dir_2, sub_path_1, sub_path_2):
    slides_1 = extract_slide_texts(slide_dir_1)
    slides_2 = extract_slide_texts(slide_dir_2)
    subs_1 = load_subtitles(sub_path_1)
    subs_2 = load_subtitles(sub_path_2)

    with open(prompt_path, 'r') as f: prompt = f.readlines() 
    prompt = "\n".join(prompt)
    prompt_q = build_prompt(slides_1, subs_1, slides_2, subs_2)
    prompt = prompt + '/n' + prompt_q
    
    output = actor_model.run([{"role": "user", "content": prompt}])
    print("=== Similarity Evaluation ===\n")
    print(output.choices[0].message.content)
    return output.choices[0].message.content

def extract_plain_subtitle_with_whisperx(video_path: str, output_path: str, model_name: str = "large-v3", language: str = "en"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model(model_name, device=device, language=language)

    audio = whisperx.load_audio(video_path)
    result = model.transcribe(audio, batch_size=16)

    with open(output_path, "w") as f:
        for seg in result["segments"]:
            f.write(seg["text"].strip() + "\n")

def extract_similarity_scores(text):
    content_match = re.search(r"Content Similarity:\s*(\d+)/5", text)
    if content_match:
        content_score = int(content_match.group(1))
        return content_score

_num_at_start = re.compile(r'^\s*["\']?(\d+)')
def sort_by_leading_number(paths: List[str]) -> List[str]:
    def key(p: str):
        name = Path(p).name  
        m = _num_at_start.match(name)
        return (int(m.group(1)) if m else float('inf'), name)
    return sorted(paths, key=key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_dir", default="/path/to/result_dir")
    parser.add_argument("-g", "--gt_dir", default="/path/to/gt_dir")
    parser.add_argument("-s", "--save_dir", default="/path/to/save_dir")
    args = parser.parse_args()
    
    ## load exist result if have
    save_dir = args.save_dir
    save_dir = path.join(save_dir, path.basename(args.result_dir))
    save_path = path.join(save_dir, "content_sim.json")
    os.makedirs(save_dir, exist_ok=True)
    if path.exists(save_path):
        with open(save_path, 'r') as f: content_sim_list = json.load(f)
    else: content_sim_list = []
    
    ## path
    gt_dir, result_dir = args.gt_dir, args.result_dir
    groundtruth_list = sort_by_leading_number([path.join(gt_dir, name) for name in os.listdir(gt_dir)])
    result_list = sort_by_leading_number([path.join(result_dir, name) for name in os.listdir(result_dir)])
    
    ## eval
    for index in range(25, 100):
        # video -> subtitle
        if path.basename(args.result_dir) == "paper2video":
            p2v_video_path = path.join(result_list[index], "3_merage.mp4")
            if path.exists(p2v_video_path) is False: continue
        else:
            p2v_video_path = path.join(result_list[index], "result.mp4")
        if path.exists(p2v_video_path) is False: continue
        gt_video_path = path.join(groundtruth_list[index], "gt_presentation_video.mp4")
        extract_plain_subtitle_with_whisperx(gt_video_path, gt_video_path.replace(".mp4", "_sub.txt"))
        extract_plain_subtitle_with_whisperx(p2v_video_path, p2v_video_path.replace(".mp4", "_sub.txt"))
        
        # slide dir 
        gt_slide_dir = path.join(groundtruth_list[index], "slide_imgs")
        p2v_slide_dir = path.join(result_list[index], "slide_imgs")
        
        # eval
        result = run_similarity_eval(
            slide_dir_1=gt_slide_dir,
            slide_dir_2=p2v_slide_dir,
            sub_path_1=gt_video_path.replace(".mp4", "_sub.txt"),
            sub_path_2=p2v_video_path.replace(".mp4", "_sub.txt"))
        content_score = extract_similarity_scores(result)
        content_sim_list.append({
            "data_idx": index,
            "score": content_score
        })
    
        with open(save_path, 'w') as f: json.dump(content_sim_list, f)