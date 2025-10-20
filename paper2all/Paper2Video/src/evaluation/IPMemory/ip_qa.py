import os
import re
import json
import time
import random
import argparse, pdb
from os import path
import google.generativeai as genai
from moviepy.editor import VideoFileClip
from camel.models import ModelFactory
from camel.types import ModelType, ModelPlatformType
from camel.configs import GeminiConfig
from typing import List
from pathlib import Path


genai.configure(api_key="")  
    
_num_at_start = re.compile(r'^\s*["\']?(\d+)')
def sort_by_leading_number(paths: List[str]) -> List[str]:
    def key(p: str):
        name = Path(p).name  
        m = _num_at_start.match(name)
        return (int(m.group(1)) if m else float('inf'), name)
    return sorted(paths, key=key)
dataset_path = "/path/to/data"
dataset_list = sort_by_leading_number(os.listdir(dataset_path))


def eval_ip(root_path, clip_duration, model_list, prompt_path, question_path, test_type='image'):
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    gemini_model = genai.GenerativeModel("models/gemini-2.5-pro-flash")
    
    with open(prompt_path, 'r') as f: prompt = f.readlines()
    prompt = "/n".join(prompt)
    with open(question_path, 'r') as f: questions = json.load(f)
    
    result_each_question = []
    for question in questions:
        video_ids = question["videos"]
        querys = question["querys"]
        qs = question["questions"]
        
        ## get video clips
        video_clips_path = {}
        for model in model_list: video_clips_path[model] = []
        
        start_p2v = None
        for vid_id in video_ids:
            tmp_dir_id = path.join(tmp_dir, str(vid_id))
            os.makedirs(tmp_dir_id, exist_ok=True)
            for model in model_list:
                if model == 'p2v': video_path = path.join(root_path, "paper2video", str(vid_id), '3_merage.mp4')
                elif model == 'p2v-o': video_path = path.join(root_path, "paper2video_wo_presenter", str(vid_id), 'result.mp4')
                elif model == 'veo3': video_path = path.join(root_path, "veo3", str(vid_id)+".mp4")
                elif model == 'wan2.2': video_path = path.join(root_path, "wan2.2", str(int(vid_id)-1), "result.mp4")
                elif model == 'presentagent': video_path = path.join(root_path, "presentagent", str(vid_id), "result.mp4")
                elif model == 'human-made': video_path = path.join(dataset_path, dataset_list[int(vid_id)-1], "gt_presentation_video.mp4")
                
                video = VideoFileClip(video_path)
                start = random.uniform(0, video.duration-clip_duration-1)
                end = start + clip_duration
                if model == 'p2v' or model == "p2v-o":
                    if start_p2v is None:
                        start_p2v = random.uniform(0, video.duration-clip_duration-1)
                        start = start_p2v
                        end = start_p2v + clip_duration
                    else:
                        start = start_p2v
                        end = start_p2v + clip_duration
                else:
                    start = random.uniform(0, video.duration-clip_duration-1)
                    end = start + clip_duration
                
                clip_save_path = path.join(tmp_dir_id, model+".mp4")
                subclip = video.subclip(start, end)
                subclip.write_videofile(clip_save_path, codec="libx264", audio_codec="aac")
                video_clips_path[model].append(clip_save_path)
        ## test for each model, 4 qas
        result_each_model = {}
        for model in model_list:
            video_input = video_clips_path[model]
            videos = upload_videos(video_input)
            result_each_model[model] = []
            for idx, query in enumerate(querys):
                if test_type == 'image': 
                    query = query[0]
                    query_state = genai.upload_file(path=query, mime_type="image/png")
                elif test_type == 'aduio': 
                    query = query[1]

                answer = idx
                ori_idxs = [0, 1, 2, 3]
                shuffled_idx = ori_idxs.copy()
                random.shuffle(shuffled_idx)
                mapping = {orig: shuffled for orig, shuffled in zip(ori_idxs, shuffled_idx)}
                new_answer = mapping[idx]
                new_qs = [qs[mapping[idx]] for idx in ori_idxs]
                
                contents = [prompt, "Here are the quary", genai.get_file(query_state.name), "Here are the video clips"]
                contents.extend(videos)
                contents.extend(["Here are the questions"])
                contents.extend(new_qs)
                
                response = gemini_model.generate_content(contents)
                #pdb.set_trace()
                match = re.search(r"My choice:\s*(\d+)", response.text)
                if match: choice_num = int(match.group(1)) - 1
                if choice_num == new_answer:
                    result_each_model[model].append([query, new_qs, choice_num, new_answer, True])
                else:
                    result_each_model[model].append([query, new_qs, choice_num, new_answer, False])
        result_each_question.append(result_each_model)
        print(result_each_question)
    with open("ip_qa_result.json", 'w') as f: json.dump(result_each_question, f, indent=4)
                
def upload_videos(video_list):
    videos = video_list.copy()
    for idx, value in enumerate(videos): 
        videos[idx] = genai.upload_file(path=value, mime_type="video/mp4")
    while True:
        flag = True
        for idx, value in enumerate(videos): 
            file_state = genai.get_file(videos[idx].name)
            if file_state.state.name != "ACTIVE": 
                flag = False
                time.sleep(5)
                print(f"waiting 5 seconds...")
                break
        if flag: break
    for idx, value in enumerate(videos): 
        videos[idx] = genai.get_file(videos[idx].name)
    return videos

if __name__ == "__main__":
    clip_duration = 4
    prompt_path = "./prompt/ip_qa.txt"
    model_list = ["p2v", "p2v-o", "veo3", "wan2.2", "presentagent", "human-made"]
    root_path = "/path/to/result"
    question_path = "ip_qa.json"
    eval_ip(root_path, clip_duration, model_list, prompt_path, question_path)