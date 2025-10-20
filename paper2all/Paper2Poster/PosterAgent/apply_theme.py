from dotenv import load_dotenv
from utils.src.utils import ppt_to_images, get_json_from_response
import json
import shutil

from camel.models import ModelFactory
from camel.agents import ChatAgent

from utils.wei_utils import *

from camel.messages import BaseMessage
from PIL import Image
import pickle as pkl
from utils.pptx_utils import *
from utils.critic_utils import *
import yaml
from jinja2 import Environment, StrictUndefined
from pdf2image import convert_from_path
import argparse

load_dotenv()

def poster_apply_theme(args, actor_config, critic_config):
    total_input_token, total_output_token = 0, 0
    extract_input_token, extract_output_token = 0, 0
    gen_input_token, gen_output_token = 0, 0
    non_overlap_ckpt = pkl.load(open(f'checkpoints/{args.model_name}_{args.poster_name}_non_overlap_ckpt_{args.index}.pkl', 'rb'))
    non_overlap_code = non_overlap_ckpt['final_code_by_section']
    sections = list(non_overlap_code.keys())
    sections = [s for s in sections if s != 'meta']
    template_img = convert_from_path(args.template_path)[0]
    image_bytes = io.BytesIO()
    template_img.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    # Reload the image from memory as a standard PIL.Image.Image
    template_img = Image.open(image_bytes)


    title_actor_agent_name = 'theme_agent_title'
    with open(f"prompt_templates/{title_actor_agent_name}.yaml", "r") as f:
        title_theme_actor_config = yaml.safe_load(f)

    section_actor_agent_name = 'theme_agent_section'
    with open(f"prompt_templates/{section_actor_agent_name}.yaml", "r") as f:
        section_theme_actor_config = yaml.safe_load(f)

    title_actor_model = ModelFactory.create(
        model_platform=actor_config['model_platform'],
        model_type=actor_config['model_type'],
        model_config_dict=actor_config['model_config'], # [Optional] the config for model
    )

    title_actor_sys_msg = title_theme_actor_config['system_prompt']

    title_actor_agent = ChatAgent(
        system_message=title_actor_sys_msg,
        model=title_actor_model,
        message_window_size=10, # [Optional] the length for chat memory
    )

    section_actor_model = ModelFactory.create(
        model_platform=actor_config['model_platform'],
        model_type=actor_config['model_type'],
        model_config_dict=actor_config['model_config'], # [Optional] the config for model
    )

    section_actor_sys_msg = section_theme_actor_config['system_prompt']

    section_actor_agent = ChatAgent(
        system_message=section_actor_sys_msg,
        model=section_actor_model,
        message_window_size=10, # [Optional] the length for chat memory
    )

    critic_model = ModelFactory.create(
        model_platform=critic_config['model_platform'],
        model_type=critic_config['model_type'],
        model_config_dict=critic_config['model_config'],
    )

    critic_sys_msg = 'You are a helpful assistant.'

    critic_agent = ChatAgent(
        system_message=critic_sys_msg,
        model=critic_model,
        message_window_size=None,
    )

    theme_aspects = {
        'background': ['background'],
        'title': ['title_author', 'title_author_border'],
        'section': ['section_body', 'section_title', 'section_border']
    }

    theme_styles = {}
    for aspect in theme_aspects.keys():
        theme_styles[aspect] = {}

    for aspect, prompt_types in theme_aspects.items():
        for prompt_type in prompt_types:
            print(f'Getting style for {prompt_type}')
            with open(f"prompt_templates/theme_templates/theme_{prompt_type}.txt", "r") as f:
                prompt = f.read()
            msg = BaseMessage.make_user_message(
                role_name="User",
                content=prompt,
                image_list=[template_img],
            )

            critic_agent.reset()
            response = critic_agent.step(msg)
            input_token, output_token = account_token(response)
            total_input_token += input_token
            total_output_token += output_token
            extract_input_token += input_token
            extract_output_token += output_token
            theme_style = get_json_from_response(response.msgs[0].content)
            theme_styles[aspect][prompt_type] = theme_style

    if 'fontStyle' in theme_styles['section']['section_body']:
        del theme_styles['section']['section_body']['fontStyle']

    outline_path = f'outlines/{args.model_name}_{args.poster_name}_outline_{args.index}.json'
    outline = json.load(open(outline_path, 'r'))
    outline_skeleton = {}
    for key, val in outline.items():
        if key == 'meta':
            continue
        if not 'subsections' in val:
            outline_skeleton[key] = {
                'section': key
            }
        else:
            for subsection_name, subsection_dict in val['subsections'].items():
                outline_skeleton[subsection_dict['name']] = {
                    'section': key
                }

    for key in outline_skeleton.keys():
        if 'title' in key.lower() or 'author' in key.lower():
            outline_skeleton[key]['style'] = theme_styles['section']['section_title']
        else:
            outline_skeleton[key]['style'] = theme_styles['section']['section_body']

    outline_skeleton_list = []
    for section in sections[1:]:
        # append all subsections whose section key is the current section
        for key, val in outline_skeleton.items():
            if val['section'] == section:
                outline_skeleton_list.append({key: val})

    theme_logs = {}
    theme_code = {}
    concatenated_code = {}

    # Title
    jinja_env = Environment(undefined=StrictUndefined)

    title_actor_template = jinja_env.from_string(title_theme_actor_config["template"])

    # Title section
    print(f'Processing section {sections[0]}')
    curr_title_code = non_overlap_code[sections[0]]
    for style in ['background', 'title']:
        for sub_style in theme_styles[style].keys():
            print(f'    Applying theme for {sub_style}')
            jinja_args = {
                'style_json': {sub_style: theme_styles[style][sub_style]},
                'function_docs': documentation,
                'existing_code': curr_title_code
            }
            actor_prompt = title_actor_template.render(**jinja_args)
            log = apply_theme(title_actor_agent, actor_prompt, args.max_retry, existing_code='')
            if log[-1]['error'] is not None:
                raise Exception(log[-1]['error'])

            input_token, output_token = log[-1]['cumulative_tokens']
            total_input_token += input_token
            total_output_token += output_token
            gen_input_token += input_token
            gen_output_token += output_token
            
            shutil.copy('poster.pptx', f'tmp/theme_poster_<{sections[0]}>_<{style}>_<{sub_style}>.pptx')

            if not style in theme_logs:
                theme_logs[style] = {}

            theme_logs[style][sub_style] = log
            curr_title_code = log[-1]['code']

    theme_code[sections[0]] = curr_title_code
    concatenated_code[sections[0]] = log[-1]['concatenated_code']

    # Remaining sections

    jinja_env = Environment(undefined=StrictUndefined)

    section_actor_template = jinja_env.from_string(section_theme_actor_config["template"])

    prev_section = None
    for style_dict in outline_skeleton_list:
        curr_subsection = list(style_dict.keys())[0]
        curr_section = style_dict[curr_subsection]['section']
        section_index = sections.index(curr_section)
        print(f'Processing section {curr_section}')
        if prev_section != curr_section:
            prev_section = curr_section
            curr_section_code = non_overlap_code[curr_section]
        print(f'    Applying theme for {curr_subsection}')
        jinja_args = {
            'style_json': json.dumps({curr_subsection: style_dict[curr_subsection]['style']}, indent=4),
            'function_docs': documentation,
            'existing_code': curr_section_code
        }
        actor_prompt = section_actor_template.render(**jinja_args)
        existing_code = concatenated_code[sections[section_index - 1]]
        log = apply_theme(section_actor_agent, actor_prompt, args.max_retry, existing_code=existing_code)
        if log[-1]['error'] is not None:
            raise Exception(log[-1]['error'])

        input_token, output_token = log[-1]['cumulative_tokens']
        total_input_token += input_token
        total_output_token += output_token
        gen_input_token += input_token
        gen_output_token += output_token

        shutil.copy('poster.pptx', f'tmp/theme_poster_<{curr_section}>_<{curr_subsection}>.pptx')

        if not style in theme_logs:
            theme_logs[style] = {}

        theme_logs[style][sub_style] = log
        curr_section_code = log[-1]['code']

        theme_code[curr_section] = curr_section_code
        concatenated_code[curr_section] = log[-1]['concatenated_code']

    ppt_to_images(f'poster.pptx', 'tmp/theme_preview')

    result_dir = f'results/{args.poster_name}/{args.model_name}/{args.index}'
    shutil.copy('poster.pptx', f'{result_dir}/theme_poster.pptx')
    ppt_to_images(f'poster.pptx', f'{result_dir}/theme_poster_preview')


    ckpt = {
        'theme_styles': theme_styles,
        'theme_logs': theme_logs,
        'theme_code': theme_code,
        'concatenated_code': concatenated_code,
        'total_input_token': total_input_token,
        'total_output_token': total_output_token,
        'extract_input_token': extract_input_token,
        'extract_output_token': extract_output_token,
        'gen_input_token': gen_input_token,
        'gen_output_token': gen_output_token
    }

    pkl.dump(ckpt, open(f'checkpoints/{args.model_name}_{args.poster_name}_theme_ckpt.pkl', 'wb'))

    return total_input_token, total_output_token

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--poster_name', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='4o')
    parser.add_argument('--poster_path', type=str, required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--template_path', type=str)
    parser.add_argument('--max_retry', type=int, default=3)
    args = parser.parse_args()

    actor_config = get_agent_config(args.model_name)
    critic_config = get_agent_config(args.model_name)

    if args.poster_name is None:
        args.poster_name = args.poster_path.split('/')[-1].replace('.pdf', '').replace(' ', '_')
    
    input_token, output_token = poster_apply_theme(args, actor_config, critic_config)

    print(f'Token consumption: {input_token} -> {output_token}')