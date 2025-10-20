from dotenv import load_dotenv
from utils.src.utils import ppt_to_images, get_json_from_response
import json

from camel.models import ModelFactory
from camel.agents import ChatAgent

from utils.wei_utils import *

from camel.messages import BaseMessage
from PIL import Image
import pickle as pkl
from utils.pptx_utils import *
from utils.critic_utils import *
import yaml
import argparse
import shutil
from jinja2 import Environment, StrictUndefined

load_dotenv()

MAX_ATTEMPTS = 5

def deoverflow(args, actor_config, critic_config):
    total_input_token, total_output_token = 0, 0
    style_ckpt = pkl.load(open(f'checkpoints/{args.model_name}_{args.poster_name}_style_ckpt_{args.index}.pkl', 'rb'))
    logs_ckpt = pkl.load(open(f'checkpoints/{args.model_name}_{args.poster_name}_ckpt_{args.index}.pkl', 'rb'))

    style_logs = style_ckpt['style_logs']
    sections = list(style_logs.keys())
    sections = [s for s in sections if s != 'meta']

    slide_width = style_ckpt['outline']['meta']['width']
    slide_height = style_ckpt['outline']['meta']['height']

    content = json.load(open(f'contents/{args.model_name}_{args.poster_name}_poster_content_{args.index}.json', 'r'))
    outline = logs_ckpt['outline']
    
    name_to_hierarchy = get_hierarchy(outline, 1)

    critic_agent_name = 'critic_overlap_agent'
    with open(f"prompt_templates/{critic_agent_name}.yaml", "r") as f:
        deoverflow_critic_config = yaml.safe_load(f)

    actor_agent_name = 'actor_editor_agent'

    with open(f"prompt_templates/{actor_agent_name}.yaml", "r") as f:
        deoverflow_actor_config = yaml.safe_load(f)

    actor_model = ModelFactory.create(
        model_platform=actor_config['model_platform'],
        model_type=actor_config['model_type'],
        model_config_dict=actor_config['model_config'],
    )

    actor_sys_msg = deoverflow_actor_config['system_prompt']

    actor_agent = ChatAgent(
        system_message=actor_sys_msg,
        model=actor_model,
        message_window_size=10,
    )

    critic_model = ModelFactory.create(
        model_platform=critic_config['model_platform'],
        model_type=critic_config['model_type'],
        model_config_dict=critic_config['model_config'],
    )

    critic_sys_msg = deoverflow_critic_config['system_prompt']

    critic_agent = ChatAgent(
        system_message=critic_sys_msg,
        model=critic_model,
        message_window_size=None,
    )

    jinja_env = Environment(undefined=StrictUndefined)

    actor_template = jinja_env.from_string(deoverflow_actor_config["template"])
    critic_template = jinja_env.from_string(deoverflow_critic_config["template"])

    critic_logs = {}
    actor_logs = {}
    img_logs = {}

    # Load neg and pos examples
    neg_img = Image.open('overflow_example/neg.jpg')
    pos_img = Image.open('overflow_example/pos.jpg')

    for section_index in range(len(sections)):
        section_name = sections[section_index]
        section_code = style_logs[section_name][-1]['code']

        if 'subsections' in content[section_name]:
            subsections = list(content[section_name]['subsections'].keys())
        else:
            subsections = [section_name]

        log = []

        for leaf_section in subsections:
            if leaf_section in outline:
                leaf_name = outline[leaf_section]['name']
            else:
                leaf_name = outline[section_name]['subsections'][leaf_section]['name']
            num_rounds = 0
            while True:
                print(f"Section: {section_name}, Leaf Section: {leaf_section}, Round: {num_rounds}")
                num_rounds += 1
                if num_rounds > MAX_ATTEMPTS:
                    break
                poster = create_poster(slide_width, slide_height)
                add_blank_slide(poster)
                save_presentation(poster, file_name='poster.pptx')
                curr_location, zoomed_in_img, zoomed_in_img_path = get_snapshot_from_section(
                    leaf_section, 
                    section_name,
                    name_to_hierarchy, 
                    leaf_name, 
                    section_code
                )

                if not leaf_section in img_logs:
                    img_logs[leaf_section] = []
                img_logs[leaf_section].append(zoomed_in_img)

                jinja_args = {
                    'content_json': content[leaf_section] if leaf_section in content else content[section_name]['subsections'][leaf_section],
                    'existing_code': section_code,
                }

                critic_prompt = critic_template.render(**jinja_args)

                critic_msg = BaseMessage.make_user_message(
                    role_name="User",
                    content=critic_prompt,
                    image_list=[neg_img, pos_img, zoomed_in_img],
                )

                critic_agent.reset()
                response = critic_agent.step(critic_msg)
                resp = response.msgs[0].content
                input_token, output_token = account_token(response)
                total_input_token += input_token
                total_output_token += output_token
                if not leaf_section in critic_logs:
                    critic_logs[leaf_section] = []

                critic_logs[leaf_section].append(response)

                if type(resp) == str:
                    if resp in ['NO', 'NO.', '"NO"', "'NO'"]:
                        break
                
                feedback = get_json_from_response(resp)
                print(feedback)
                jinja_args = {
                    'content_json': content[leaf_section] if leaf_section in content else content[section_name]['subsections'][leaf_section],
                    'function_docs': documentation,
                    'existing_code': section_code,
                    'suggestion_json': feedback,
                }

                actor_prompt = actor_template.render(**jinja_args)

                log = edit_code(actor_agent, actor_prompt, 3, existing_code='')
                if log[-1]['error'] is not None:
                    raise Exception(log[-1]['error'])

                input_token = log[-1]['cumulative_tokens'][0]
                output_token = log[-1]['cumulative_tokens'][1]
                total_input_token += input_token
                total_output_token += output_token
                
                section_code = log[-1]['code']
                
                if not leaf_section in actor_logs:
                    actor_logs[leaf_section] = []

                actor_logs[leaf_section].append(log)
            if len(log) > 0:
                style_logs[section_name].append(log[-1])

    final_code = ''
    for section in sections:
        final_code += style_logs[section][-1]['code'] + '\n'

    run_code_with_utils(final_code, utils_functions)
    ppt_to_images(f'poster.pptx', 'tmp/non_overlap_preview')
    
    result_dir = f'results/{args.poster_name}/{args.model_name}/{args.index}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    shutil.copy('poster.pptx', f'{result_dir}/non_overlap_poster.pptx')
    ppt_to_images(f'poster.pptx', f'{result_dir}/non_overlap_poster_preview')

    final_code_by_section = {}
    for section in sections:
        final_code_by_section[section] = style_logs[section][-1]['code']

    non_overlap_ckpt = {
        'critic_logs': critic_logs,
        'actor_logs': actor_logs,
        'img_logs': img_logs,
        'name_to_hierarchy': name_to_hierarchy,
        'final_code': final_code,
        'final_code_by_section': final_code_by_section,
        'total_input_token': total_input_token,
        'total_output_token': total_output_token
    }

    pkl.dump(non_overlap_ckpt, open(f'checkpoints/{args.model_name}_{args.poster_name}_non_overlap_ckpt_{args.index}.pkl', 'wb'))

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
    critic_config = get_agent_config(args.model_name)

    if args.poster_name is None:
        args.poster_name = args.poster_path.split('/')[-1].replace('.pdf', '').replace(' ', '_')
    
    input_token, output_token = deoverflow(args, actor_config, critic_config)

    print(f'Token consumption: {input_token} -> {output_token}')