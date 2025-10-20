'''
    Using VideoLLM (Gemini) as judger 
'''
import os, re, json
import time
import argparse
import google.generativeai as genai
from os import path
from typing import List
from pathlib import Path
from tqdm import tqdm


genai.configure(api_key="")  
def eval_gemini(gt_vid_path, gen_vid_path):
    model = genai.GenerativeModel("models/gemini-2.5-pro")
    gt_vid = genai.upload_file(path=gt_vid_path, mime_type="video/mp4")
    gen_vid = genai.upload_file(path=gen_vid_path, mime_type="video/mp4")
    while True:
        refreshed_1 = genai.get_file(gt_vid.name)
        refreshed_2 = genai.get_file(gen_vid.name)
        if refreshed_1.state.name == "ACTIVE" and refreshed_2.state.name == "ACTIVE": break
        elif refreshed_1.state.name == "FAILED" or refreshed_2.state.name == "FAILED": 
            #raise RuntimeError("❌ File processing failed.")
            return None
        else:
            print(f"waiting 5 seconds...")
            time.sleep(5)

    prompt_path = "./prompt/which_is_better.txt"
    with open(prompt_path, 'r') as f: prompt = f.readlines()
    prompt = "/n".join(prompt)
    print("Sending prompt to Gemini...")
    response = model.generate_content([prompt, refreshed_1, refreshed_2])
    print("\n===== Evaluation Result =====")
    print(response.text)
    print("=============================\n")

    return response.text

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
    if path.basename(args.result_dir) == "paper2video":
        save_dir = path.join(save_dir, path.basename(args.result_dir))
    else: save_dir = path.join(save_dir, path.basename(args.result_dir))
    
    save_path = path.join(save_dir, "video_arena.json")
    os.makedirs(save_dir, exist_ok=True)
    if path.exists(save_path):
        with open(save_path, 'r') as f: arena_score_list = json.load(f)
    else: arena_score_list = []
    
    ## path
    gt_dir, result_dir = args.gt_dir, args.result_dir
    groundtruth_list = sort_by_leading_number([path.join(gt_dir, name) for name in os.listdir(gt_dir)])
    result_list = sort_by_leading_number([path.join(result_dir, name) for name in os.listdir(result_dir)])
    
    ## Generated v.s GT (1)
    for index in tqdm(len(result_list)):
        if path.basename(args.result_dir) == "paper2video":
            test_video_path = path.join(result_list[index], "3_merage.mp4")
        elif path.basename(args.result_dir) == 'veo3':
            test_video_path = result_list[index]
        else:
            test_video_path = path.join(result_list[index], "result.mp4")
            
        if path.exists(test_video_path) is False: continue
        gt_video_path = path.join(groundtruth_list[index], "gt_presentation_video.mp4")
        if path.exists(gt_video_path) is False: 
            gt_video_path = path.join(groundtruth_list[index], "raw_video.mp4")
            if path.exists(gt_video_path) is False: continue
        result = eval_gemini(gt_video_path, test_video_path)
        if result is None: continue
        
        pat = r"\[(?:A|B)\]"
        m = re.findall(pat, result, flags=re.I)
        score = 0
        if m[0][1] == "B": score += 1
        
        result = eval_gemini(test_video_path, gt_video_path)
        if result is None: continue
        
        pat = r"\[(?:A|B)\]"
        m = re.findall(pat, result, flags=re.I)
        if m[0][1] == "A": score += 1
        
        arena_score_list.append({
            "data_idx": index,
            "score": score/2
        })
        with open(save_path, 'w') as f: json.dump(arena_score_list, f, indent=4)
