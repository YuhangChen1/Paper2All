from dotenv import load_dotenv
import os
from utils.src.utils import ppt_to_images, get_json_from_response
import json
import pptx

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig, QwenConfig
from camel.agents import ChatAgent

from utils.wei_utils import fill_content

from camel.messages import BaseMessage
from PIL import Image
import pickle as pkl
from utils.pptx_utils import *
from utils.critic_utils import *
from utils.wei_utils import *
import importlib
import yaml
import os
import shutil
from datetime import datetime
from jinja2 import Environment, StrictUndefined, Template
import argparse

load_dotenv()

def fill_poster_content(args, actor_config):
    total_input_token, total_output_token = 0, 0
    poster_content = json.load(open(f'contents/{args.model_name}_{args.poster_name}_poster_content_{args.index}.json', 'r'))
    agent_name = 'content_filler_agent'

    with open(f"prompt_templates/{agent_name}.yaml", "r") as f:
        fill_config = yaml.safe_load(f)

    actor_model = ModelFactory.create(
        model_platform=actor_config['model_platform'],
        model_type=actor_config['model_type'],
        model_config_dict=actor_config['model_config'],
    )

    actor_sys_msg = fill_config['system_prompt']

    actor_agent = ChatAgent(
        system_message=actor_sys_msg,
        model=actor_model,
        message_window_size=10,
    )

    ckpt = pkl.load(open(f'checkpoints/{args.model_name}_{args.poster_name}_ckpt_{args.index}.pkl', 'rb'))
    logs = ckpt['logs']
    outline = ckpt['outline']

    sections = list(outline.keys())
    sections = [s for s in sections if s != 'meta']

    jinja_env = Environment(undefined=StrictUndefined)

    template = jinja_env.from_string(fill_config["template"])
    content_logs = {}

    for section_index in range(len(sections)):
        section_name = sections[section_index]
        section_code = logs[section_name][-1]['code']

        print(f'Filling content for {section_name}')

        jinja_args = {
            'content_json': poster_content[section_name],
            'function_docs': documentation,
            'existing_code': section_code
        }

        prompt = template.render(**jinja_args)
        if section_index == 0:
            existing_code = ''
        else:
            existing_code = content_logs[sections[section_index - 1]][-1]['concatenated_code']
        content_logs[section_name] = fill_content(
            actor_agent, 
            prompt, 
            3, 
            existing_code
        )

        shutil.copy('poster.pptx', f'tmp/content_poster_<{section_name}>.pptx')

        if content_logs[section_name][-1]['error'] is not None:
            raise Exception(f'Error in filling content for {section_name}: {content_logs[section_name][-1]["error"]}')
        
        total_input_token += content_logs[section_name][-1]['cumulative_tokens'][0]
        total_output_token += content_logs[section_name][-1]['cumulative_tokens'][1]

    ppt_to_images(f'tmp/content_poster_<{sections[-1]}>.pptx', 'tmp/content_preview')

    ckpt = {
        'logs': logs,
        'content_logs': content_logs,
        'outline': outline,
        'total_input_token': total_input_token,
        'total_output_token': total_output_token
    }

    pkl.dump(ckpt, open(f'checkpoints/{args.model_name}_{args.poster_name}_content_ckpt_{args.index}.pkl', 'wb'))

    return total_input_token, total_output_token

def stylize_poster(args, actor_config):
    total_input_token, total_output_token = 0, 0
    poster_content = json.load(open(f'contents/{args.model_name}_{args.poster_name}_poster_content_{args.index}.json', 'r'))
    agent_name = 'style_agent'

    with open(f"prompt_templates/{agent_name}.yaml", "r") as f:
        style_config = yaml.safe_load(f)

    actor_model = ModelFactory.create(
        model_platform=actor_config['model_platform'],
        model_type=actor_config['model_type'],
        model_config_dict=actor_config['model_config'],
    )

    actor_sys_msg = style_config['system_prompt']

    actor_agent = ChatAgent(
        system_message=actor_sys_msg,
        model=actor_model,
        message_window_size=10,
    )

    ckpt = pkl.load(open(f'checkpoints/{args.model_name}_{args.poster_name}_content_ckpt_{args.index}.pkl', 'rb'))
    content_logs = ckpt['content_logs']
    outline = ckpt['outline']

    sections = list(outline.keys())
    sections = [s for s in sections if s != 'meta']

    jinja_env = Environment(undefined=StrictUndefined)

    template = jinja_env.from_string(style_config["template"])
    style_logs = {}

    for section_index in range(len(sections)):
        section_name = sections[section_index]
        section_outline = json.dumps(outline[section_name])
        section_code = content_logs[section_name][-1]['code']

        print(f'Stylizing for {section_name}')

        img_ratio_json = get_img_ratio_in_section(poster_content[section_name])

        jinja_args = {
            'content_json': poster_content[section_name],
            'function_docs': documentation,
            'existing_code': section_code,
            'image_ratio': img_ratio_json,
        }

        prompt = template.render(**jinja_args)
        if section_index == 0:
            existing_code = ''
        else:
            existing_code = style_logs[sections[section_index - 1]][-1]['concatenated_code']
        style_logs[section_name] = stylize(
            actor_agent, 
            prompt, 
            args.max_retry, 
            existing_code
        )

        shutil.copy('poster.pptx', f'tmp/style_poster_<{section_name}>.pptx')

        if style_logs[section_name][-1]['error'] is not None:
            raise Exception(f'Error in stylizing for {section_name}')
        
        total_input_token += style_logs[section_name][-1]['cumulative_tokens'][0]
        total_output_token += style_logs[section_name][-1]['cumulative_tokens'][1]

    ppt_to_images(f'tmp/style_poster_<{sections[-1]}>.pptx', 'tmp/style_preview')
    ckpt = {
        'logs': ckpt['logs'],
        'content_logs': content_logs,
        'style_logs': style_logs,
        'outline': outline,
        'total_input_token': total_input_token,
        'total_output_token': total_output_token
    }

    with open(f'checkpoints/{args.model_name}_{args.poster_name}_style_ckpt_{args.index}.pkl', 'wb') as f:
        pkl.dump(ckpt, f)

    return total_input_token, total_output_token

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--poster_name', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='4o')
    parser.add_argument('--poster_path', type=str, required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--max_retry', type=int, default=3)
    args = parser.parse_args()

    actor_config = get_agent_config(args.model_name)

    if args.poster_name is None:
        args.poster_name = args.poster_path.split('/')[-1].replace('.pdf', '').replace(' ', '_')

    fill_total_input_token, fill_total_output_token = fill_poster_content(args, actor_config)
    style_total_input_token, style_total_output_token = stylize_poster(args, actor_config)

    total_input_token = fill_total_input_token + style_total_input_token
    total_output_token = fill_total_output_token + style_total_output_token

    print(f'Token consumption: {total_input_token} -> {total_output_token}')