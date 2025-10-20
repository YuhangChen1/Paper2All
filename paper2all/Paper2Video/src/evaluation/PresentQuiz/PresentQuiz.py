import random
import string
import yaml
import PIL
import tempfile
import io
import argparse
from os import path
from camel.models import ModelFactory
from math import ceil
from openai import OpenAI
from camel.messages import BaseMessage
from utils.src.model_utils import parse_pdf
from urllib.parse import unquote
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM
from pytorch_fid.fid_score import compute_statistics_of_path
import pytorch_fid.fid_score as fid
from PIL import Image
from httpx import Timeout
from docling.document_converter import DocumentConverter, PdfFormatOption
import re
import shutil
import pytesseract
from utils.wei_utils import account_token
from camel.types import ModelPlatformType, ModelType
from marker.models import create_model_dict
from camel.configs import ChatGPTConfig
from camel.agents import ChatAgent
from jinja2 import Environment, StrictUndefined
from utils.src.utils import get_json_from_response
from pathlib import Path
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from collections import defaultdict
from camel.configs import ChatGPTConfig, QwenConfig, VLLMConfig, OpenRouterConfig, GeminiConfig

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

import math
import base64
import requests
from io import BytesIO
from PIL import Image

import torch
import json
import os
import pickle as pkl
import numpy as np
from transformers import AltCLIPProcessor, AltCLIPModel
from pathlib import Path
from typing import List
from moviepy.editor import VideoFileClip


os.environ["GEMINI_API_KEY"] = ""

def compute_accuracy(predicted, ground_truth, aspects):
    """
    Parameters
    ----------
    predicted : dict
        {question: {'answer': <letter>, 'reference': ...}, ...}
    ground_truth : dict
        {question: '<letter>. full answer', ...}
    aspects : dict
        {question: '<aspect name>', ...}

    Returns
    -------
    overall_accuracy : float
    aspect_summary : dict
        {
          '<aspect name>': {
              'total':    <int>,   # questions in this aspect
              'correct':  <int>,   # correctly answered questions
              'accuracy': <float>  # correct / total (0–1)
          },
          ...
        }
    """
    correct_global = 0
    total_global   = len(ground_truth)

    total_by_aspect   = defaultdict(int)
    correct_by_aspect = defaultdict(int)

    for q, pred_info in predicted.items():
        letter_pred = pred_info['answer']
        aspect = aspects.get(q, 'Unknown')
        total_by_aspect[aspect] += 1

        if q in ground_truth:
            letter_gt = ground_truth[q].split('.')[0].strip()

            if len(letter_pred) > 0:
                letter_pred = letter_pred[0].upper()
            if letter_pred == letter_gt:
                correct_global += 1
                correct_by_aspect[aspect] += 1

    overall_accuracy = correct_global / total_global if total_global else 0.0

    # Build the per-aspect dictionary
    aspect_summary = {}
    for aspect, total in total_by_aspect.items():
        correct = correct_by_aspect[aspect]
        acc     = correct / total if total else 0.0
        aspect_summary[aspect] = {
            'total':   total,
            'correct': correct,
            'accuracy': acc
        }

    return overall_accuracy, aspect_summary

def eval_qa_get_answer(video_input, questions, answers, aspects, agent_config, input_type='video'):
    agent_name = f'answer_question_from_{input_type}'
    with open(f"prompt/{agent_name}.yaml", "r") as f: config = yaml.safe_load(f)

    actor_model = ModelFactory.create(
            model_platform=agent_config['model_platform'],
            model_type=agent_config['model_type'],
            model_config_dict=agent_config['model_config'],)

    actor_sys_msg = config['system_prompt']

    actor_agent = ChatAgent(system_message=actor_sys_msg, model=actor_model, message_window_size=None,)
    actor_agent.reset()
    
    jinja_env = Environment(undefined=StrictUndefined)
    template = jinja_env.from_string(config["template"])
    with open(video_input, "rb") as f: video_bytes = f.read()
    if input_type == 'video':
        prompt = template.render(**{'questions': questions,})
         
        clip = VideoFileClip(video_input)
        duration = clip.duration  
        msg = BaseMessage.make_user_message(
            role_name="User",
            content=prompt+"The video length is {}, you should NOT reference the timesteps if it exceeds video length".format(str(duration)),
            video_bytes=video_bytes,
            video_detail="low")
        response = actor_agent.step(msg)
        agent_answers = get_json_from_response(response.msgs[0].content)

    input_token, output_token = account_token(response)
    accuracy, aspect_accuracy = compute_accuracy(agent_answers, answers, aspects)
    return accuracy, aspect_accuracy, agent_answers, input_token, output_token

def run_qa_metric(question_path, video_path, result_path, test_model):
    if test_model == "gemini":
        agent_config = {
                            "model_type": ModelType.GEMINI_2_5_FLASH,
                            "model_config": GeminiConfig().as_dict(),
                            "model_platform": ModelPlatformType.GEMINI,
                        } 
    overall_qa_result = {"qa_result": {}}

    qa_dict = json.load(open(question_path, 'r'))
    detail_qa, understanding_qa = qa_dict['detail'], qa_dict['understanding']
    input_token_all, output_token_all =0, 0
    detail_accuracy, detail_aspect_accuracy, detail_agent_answers, input_token, output_token = eval_qa_get_answer(
            video_input=video_path,
            questions=detail_qa['questions'],
            answers=detail_qa['answers'],
            aspects=detail_qa['aspects'],
            agent_config=agent_config,
            input_type='video')
    input_token_all += input_token
    output_token_all += output_token
    understanding_accuracy, understanding_aspect_accuracy, understanding_agent_answers, input_token, output_token = eval_qa_get_answer(
            video_input=video_path,
            questions=understanding_qa['questions'],
            answers=understanding_qa['answers'],
            aspects=understanding_qa['aspects'],
            agent_config=agent_config,
            input_type='video')
    input_token_all += input_token
    output_token_all += output_token
    overall_qa_result['qa_result'][test_model] = {
            'detail_accuracy': detail_accuracy,
            'detail_aspect_accuracy': detail_aspect_accuracy,
            'detail_agent_answers': detail_agent_answers,
            'understanding_accuracy': understanding_accuracy,
            'understanding_aspect_accuracy': understanding_aspect_accuracy,
            'understanding_agent_answers': understanding_agent_answers}
    all_models_in_file = list(overall_qa_result['qa_result'].keys())
    detail_accs = []
    understanding_accs = []
    for m in all_models_in_file:
        detail_accs.append(overall_qa_result['qa_result'][m]['detail_accuracy'])
        understanding_accs.append(overall_qa_result['qa_result'][m]['understanding_accuracy'])

    avg_detail_accuracy = float(np.mean(detail_accs)) if detail_accs else 0.0
    avg_understanding_accuracy = float(np.mean(understanding_accs)) if understanding_accs else 0.0

    overall_qa_result['avg_detail_accuracy'] = avg_detail_accuracy
    overall_qa_result['avg_understanding_accuracy'] = avg_understanding_accuracy

    # Finally, overwrite the same JSON file with the updated results
    with open(result_path, 'w') as f: json.dump(overall_qa_result, f, indent=4)
    print(detail_accuracy, detail_aspect_accuracy, detail_agent_answers, input_token, output_token)

_num_at_start = re.compile(r'^\s*["\']?(\d+)')
def sort_by_leading_number(paths: List[str]) -> List[str]:
    def key(p: str):
        name = Path(p).name  
        m = _num_at_start.match(name)
        return (int(m.group(1)) if m else float('inf'), name)
    return sorted(paths, key=key)
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_dir", default="/path/to/result")
    parser.add_argument("-g", "--data_dir", default="/path/to/data")
    parser.add_argument("-s", "--save_dir", default="/path/to/data")
    args = parser.parse_args()
    ## mkdirs
    save_dir = args.save_dir
    if path.basename(args.result_dir) == "paper2video":
        save_dir = path.join(save_dir, path.basename(args.result_dir))
    else: save_dir = path.join(save_dir, path.basename(args.result_dir))
    
    save_path = path.join(save_dir, "qa_result")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    
    ## run test
    gt_dir, result_dir = args.data_dir, args.result_dir
    groundtruth_list = sort_by_leading_number([path.join(gt_dir, name) for name in os.listdir(gt_dir)])
    if path.basename(args.result_dir) == "human_made": result_list = [] # from dataset
    else: result_list = sort_by_leading_number([path.join(result_dir, name) for name in os.listdir(result_dir)])
    
    start, end = 1, 100
    for index in range(start, end):
        qa_json_path = path.join(groundtruth_list[index], "4o-mini_qa.json")
        
        ## paper2video
        if path.basename(args.result_dir) == 'paper2video':
            if without_presenter_flag is False:
                test_video_path = path.join(result_list[index], "3_merage.mp4")
            else:
                test_video_path = path.join(result_list[index], "1_merage.mp4")
            if path.exists(test_video_path) is False: continue
        ## human made as baseline
        elif path.basename(args.result_dir) == 'human_made':
            test_video_path = path.join(groundtruth_list[index], "gt_presentation_video.mp4")
            if path.exists(test_video_path) is False:
                test_video_path = path.join(groundtruth_list[index], "raw_video.mp4")
        ## veo3
        elif path.basename(args.result_dir) == 'veo3':
            test_video_path = result_list[index]
        elif path.basename(args.result_dir) == 'wan2.1':
            test_video_path = path.join(result_list[index], "result.mp4")
        ## presentagent
        else:
            test_video_path = path.join(result_list[index], "result.mp4")
        if path.exists(test_video_path) is False: continue
        result_save_path = path.join(save_path, "qa_result_{}.json".format(index)) 
        print("start")
        run_qa_metric(qa_json_path, test_video_path, result_save_path, 'gemini')