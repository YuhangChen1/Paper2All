"""
    construct question about Academic IP
    input query: 4 video clips from 4 different paper presentation + query (image/audio)
    input question: 4 understanding qa from corresponding paper
    output task: choose the right question to ask
"""
import os, re
import json
import random
import itertools
from os import path
from typing import List
from pathlib import Path
from tqdm import tqdm

def generate_combinations(total_num, comb_size):
    return list(itertools.combinations(range(total_num), comb_size))

def generate_ip_task(vaild_data_name, num_qa_pair):
    combs = list(itertools.combinations(range(len(vaild_data_name)), 4))
    combs = random.sample(combs, num_qa_pair)
    
    qa_list = []
    for comb in combs:
        ## questions
        question_list = []
        question_index = random.randint(1, 50)
        for index in comb:
            question_path = path.join(vaild_data_name[index][1], "4o-mini_qa.json")
            with open(question_path, 'r') as f: question = json.load(f)["understanding"]["questions"]
            question_list.append(question["Question {}".format(str(question_index))]["question"])
        ## query        
        query_list = []
        for index in comb:
            ref_img_path = path.join(vaild_data_name[index][1], "ref_img.png")
            ref_audio_path = path.join(vaild_data_name[index][1], "ref_audio.wav")
            query_list.append((ref_img_path, ref_audio_path))
        ## qa
        qa = {}
        qa["videos"] = []
        for idx in range(len(comb)):
            qa["videos"].append(vaild_data_name[comb[idx]][0])
            
        qa["querys"] = query_list
        qa["questions"] = question_list
        qa_list.append(qa)
    with open("ip_qa.json", 'w') as f: json.dump(qa_list, f, indent=4)
    
_num_at_start = re.compile(r'^\s*["\']?(\d+)')
def sort_by_leading_number(paths: List[str]) -> List[str]:
    def key(p: str):
        name = Path(p).name  
        m = _num_at_start.match(name)
        return (int(m.group(1)) if m else float('inf'), name)
    return sorted(paths, key=key)

if __name__ == "__main__":
    num_qa_pair = 10 # C (num_data) (4)
    root_dir = "/path/to/result"
    gt_dir = "/path/to/data"
    
    all_data_name = sort_by_leading_number(os.listdir(root_dir))
    all_groundtruth = sort_by_leading_number(os.listdir(gt_dir))
    vaild_data_name = []
    for data_idx in range(len(all_data_name)):
        if path.basename(root_dir) == "paper2video":
            video_result_1 = path.join(root_dir, all_data_name[data_idx], "3_merage.mp4")
            video_result_2 = path.join(root_dir.replace("paper2video", "presentagent"), all_data_name[data_idx], "result.mp4")
    generate_ip_task(vaild_data_name, num_qa_pair)
