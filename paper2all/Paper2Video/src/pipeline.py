'''
    1. (LLM) slide generation 
    2. (VLM) subtitle and cursor prompt generation
    3. TTS->audio; GUI&WhisperX Grounding->cursor;
    4. Talking Gen: local-[hallo2, fantasy, ...], api-[HeyGen]
    5. Merage
'''
import cv2
import pdb
import json
import time
import shutil
import asyncio
import os, sys
import argparse
import subprocess
from os import path
from pdf2image import convert_from_path


from speech_gen import tts_per_slide
from subtitle_render import add_subtitles
from talking_gen import talking_gen_per_slide
from cursor_gen import cursor_gen_per_sentence
# from slide_code_gen import latex_code_gen
from slide_code_gen_select_improvement import latex_code_gen_upgrade
from cursor_render import render_video_with_cursor_from_json
from subtitle_cursor_prompt_gen import subtitle_cursor_gen

from wei_utils import get_agent_config


# os.environ["GEMINI_API_KEY"] = ""
# os.environ["OPENAI_API_KEY"] = ""

def copy_folder(src_dir, dst_dir):
    if not os.path.exists(src_dir): raise FileNotFoundError(f"no such dir: {src_dir}")
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    shutil.copytree(src_dir, dst_dir)

def str2list(s): return [int(x) for x in s.split(',')]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paper2Video Generation Pipeline')
    parser.add_argument('--result_dir', type=str, default='./result/zeyu')
    parser.add_argument('--model_name_t', type=str, default='gpt-4.1') 
    parser.add_argument('--model_name_v', type=str, default='gpt-4.1') 
    parser.add_argument('--model_name_talking', type=str, default='hallo2')
    parser.add_argument('--paper_latex_root', type=str, default='./assets/demo/latex_proj')
    parser.add_argument('--ref_img', type=str, default='./assets/demo/zeyu.png')
    parser.add_argument('--ref_audio', type=str, default='./assets/demo/zeyu.wav')
    parser.add_argument('--ref_text', type=str, default=None)
    parser.add_argument('--gpu_list', type=str2list, default="")
    parser.add_argument('--if_tree_search', type=bool, default=True)
    parser.add_argument('--beamer_templete_prompt', type=str, default=None)
    parser.add_argument('--stage', type=str, default="[\"0\"]") 
    parser.add_argument('--talking_head_env', type=str, default="") 
    # slide+subtitle: 1; 
    # tts+cusor: 2; 
    # talking-head: 3: 
    # all: 0
    args = parser.parse_args()
    stage = json.loads(args.stage)
    print("start", "stage:", stage, args.gpu_list)
    
    cursor_img_path = "./cursor_image/red.png"
    os.makedirs(args.result_dir, exist_ok=True) # result dir
    agent_config_t = get_agent_config(args.model_name_t) # LLM
    agent_config_v = get_agent_config(args.model_name_v) # VLM
    copy_latex_proj_path = path.join(args.result_dir, path.basename(args.paper_latex_root))
    if path.exists(copy_latex_proj_path) is False:
        copy_folder(args.paper_latex_root, copy_latex_proj_path)
    args.paper_latex_root = copy_latex_proj_path
    
    if path.exists(path.join(args.result_dir, "sat.json")) is True:
        with open(path.join(args.result_dir, "sat.json"), 'r') as f: 
            time_second = json.load(f)
    else: time_second = {}
        
    if path.exists(path.join(args.result_dir, "token.json")) is True:
        with open(path.join(args.result_dir, "token.json"), 'r') as f: 
            token_usage = json.load(f)
    else: token_usage = {}
    
    ## Step 1: Slide Generation
    slide_latex_path = path.join(args.paper_latex_root, "slides.tex")
    slide_image_dir = path.join(args.result_dir, 'slide_imgs')
    os.makedirs(slide_image_dir, exist_ok=True)
    
    start1_time = time.time() # start time
    if "1" in stage or  "0" in stage:
        prompt_path = "./prompts/slide_beamer_prompt.txt"
        if args.if_tree_search is True: 
            usage_slide, beamer_path = latex_code_gen_upgrade(prompt_path=prompt_path, tex_dir=args.paper_latex_root, beamer_save_path=slide_latex_path, 
                                                            model_config_ll=agent_config_t, model_config_vl=agent_config_v, beamer_temp_name=args.beamer_templete_prompt)
        else:
            paper_latex_path = path.join(args.paper_latex_root, "main.tex") 
            # usage_slide = latex_code_gen(prompt_path=prompt_path, tex_dir=args.paper_latex_root, tex_path=paper_latex_path, beamer_save_path=slide_latex_path, model_config=agent_config_t)
            
        slide_imgs = convert_from_path(beamer_path, dpi=400)
        for i, img in enumerate(slide_imgs): img.save(path.join(slide_image_dir, f"{i+1}.png")) # save slides as images
        if args.model_name_t not in token_usage.keys(): 
            token_usage[args.model_name_t] = [usage_slide]
        else: token_usage[args.model_name_t].append(usage_slide)
        step1_time =  time.time()
        time_second["slide_gen"] = [step1_time-start1_time]
        print("Slide Generation", step1_time-start1_time)
    
    ## Step 2: Subtitle and Cursor Prompt Generation
    start_time = time.time() # start time
    subtitle_cursor_save_path = path.join(args.result_dir, 'subtitle_w_cursor.txt')
    cursor_save_path = path.join(args.result_dir, 'cursor.json')

    speech_save_dir = path.join(args.result_dir, 'audio')
    if "2" in stage or  "0" in stage:
        prompt_path = "./prompts/slide_subtitle_cursor_prompt.txt"
        subtitle, usage_subtitle = subtitle_cursor_gen(slide_image_dir, prompt_path, agent_config_v)
        with open(subtitle_cursor_save_path, 'w') as f: f.write(subtitle)
        if args.model_name_v not in token_usage.keys(): 
            token_usage[args.model_name_v] = [usage_subtitle]
        else: token_usage[args.model_name_v].append(usage_subtitle)
        step2_time =  time.time()
        time_second["subtitle_cursor_prompt_gen"] = [step2_time-start_time]
        print("Subtitle and Cursor Prompt Generation", step2_time-start_time)

        ## Step 3-1: Speech Generation
        # tts_per_slide(model_type='f5', script_path=subtitle_cursor_save_path, 
        #             speech_save_dir=speech_save_dir, ref_audio=args.ref_audio, ref_text=args.ref_text)  
        step3_1_time =  time.time()
        time_second["tts"] = [step3_1_time-step2_time]
        print("Speech Generation", step3_1_time-step2_time)
        
        ## Step 3-2: Cursor Generation
        os.environ["PYTHONHASHSEED"] = "random"        
        cursor_token = cursor_gen_per_sentence(script_path=subtitle_cursor_save_path, slide_img_dir=slide_image_dir, 
                                slide_audio_dir=speech_save_dir, cursor_save_path=cursor_save_path, gpu_list=args.gpu_list)
        token_usage["cursor"] = cursor_token
        step3_2_time =  time.time()
        time_second["cursor_gen"] = [step3_2_time-step3_1_time]
        print("Cursor Generation", step3_2_time-step3_1_time)
    
    ## Step 4: Talking Video Generation
    start_time = time.time() # start time
    if "3" in stage or  "0" in stage:
        talking_save_dir = path.join(args.result_dir, 'talking_{}'.format(args.model_name_talking))
        talking_inference_input = []
        audio_path_list = [path.join(speech_save_dir, name) for name in os.listdir(speech_save_dir)]
        for audio_path in audio_path_list: talking_inference_input.append([args.ref_img, audio_path])
        talking_gen_per_slide(args.model_name_talking, talking_inference_input, talking_save_dir, args.gpu_list, env_path=args.talking_head_env)
        step4_time =  time.time()
        time_second["talking_gen"] = [step4_time-start_time]
        print("Cursor Generation", step4_time-start_time)
    
        ## Step5: Merage
        # merage talking and slides
        tmp_merage_dir = path.join(args.result_dir, "merage")
        tmp_merage_1 = path.join(args.result_dir, "1_merage.mp4")
        image_size = cv2.imread(path.join(slide_image_dir, '1.png')).shape
        if args.model_name_talking == 'hallo2':
            size = max(image_size[0]//6, image_size[1]//6)
            width, height = size, size
        num_slide = len(os.listdir(slide_image_dir))
        print(args.ref_img.split("/")[-1].split(".")[0])
        merage_cmd =  ["./1_merage.bash", slide_image_dir, talking_save_dir, tmp_merage_dir,
                    str(width), str(height), str(num_slide), tmp_merage_1, args.ref_img.split("/")[-1].replace(".png", "")]
        out = subprocess.run(merage_cmd, text=True)
        # render cursor
        cursor_size = size//6
        tmp_merage_2 = path.join(args.result_dir, "2_merage.mp4")
        render_video_with_cursor_from_json(video_path=tmp_merage_1, out_video_path=tmp_merage_2, 
                                        json_path=cursor_save_path, cursor_img_path=cursor_img_path, 
                                        transition_duration=0.1, cursor_size=cursor_size)
        # render subtitle
        front_size = size//10
        tmp_merage_3 = path.join(args.result_dir, "3_merage.mp4")
        add_subtitles(tmp_merage_2, tmp_merage_3, size//10)
        step5_time =  time.time()
        time_second["merage"] = [step5_time-step4_time]
        print("Merage", step5_time-step4_time)
        
    # sat. save
    time_second = {"slide_gen": [step1_time-start1_time, usage_slide], 
                   "subtitle_cursor_prompt_gen": [step2_time-step1_time, usage_subtitle],
                   "tts": step3_1_time-step2_time, "cursor_gen": step3_2_time-step3_1_time, 
                   "talking_gen": step4_time-step3_2_time, "merage": step5_time-step4_time}
    with open(path.join(args.result_dir, "sat.json"), 'w') as f: json.dump(time_second, f, indent=4)
    with open(path.join(args.result_dir, "token.json"), 'w') as f: json.dump(token_usage, f, indent=4)
