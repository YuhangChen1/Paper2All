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
from concurrent.futures import ThreadPoolExecutor
import copy

load_dotenv()

MAX_ATTEMPTS = 5

def process_leaf_section(
    leaf_section,
    section_name,
    outline,
    content,
    style_logs,
    critic_logs,
    actor_logs,
    img_logs,
    slide_width,
    slide_height,
    name_to_hierarchy,
    critic_template,
    actor_template,
    critic_agent,
    actor_agent,
    neg_img,
    pos_img,
    MAX_ATTEMPTS,
    documentation,
    total_input_token,
    total_output_token,
):
    """
    Handles the logic for a single leaf_section within a section_name.

    Returns a dictionary of updated logs and tokens.
    """
    section_code = style_logs[section_name][-1]['code']  # current code for this section
    log = []
    leaf_name = None
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
        empty_poster_path = f'tmp/empty_poster_{section_name}_{leaf_section}.pptx'
        save_presentation(poster, file_name=empty_poster_path)

        curr_location, zoomed_in_img, zoomed_in_img_path = get_snapshot_from_section(
            leaf_section,
            section_name,
            name_to_hierarchy,
            leaf_name,
            section_code,
            empty_poster_path
        )

        if leaf_section not in img_logs:
            img_logs[leaf_section] = []
        img_logs[leaf_section].append(zoomed_in_img)

        jinja_args = {
            'content_json': content[leaf_section] if leaf_section in content
                           else content[section_name]['subsections'][leaf_section],
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

        # Track tokens
        input_token, output_token = account_token(response)
        total_input_token += input_token
        total_output_token += output_token

        if leaf_section not in critic_logs:
            critic_logs[leaf_section] = []
        critic_logs[leaf_section].append(response)

        # Stop condition
        if isinstance(resp, str):
            if resp in ['NO', 'NO.', '"NO"', "'NO'"]:
                break

        feedback = get_json_from_response(resp)
        print(feedback)

        jinja_args = {
            'content_json': content[leaf_section] if leaf_section in content
                           else content[section_name]['subsections'][leaf_section],
            'function_docs': documentation,
            'existing_code': section_code,
            'suggestion_json': feedback,
        }

        actor_prompt = actor_template.render(**jinja_args)

        leaf_log = edit_code(actor_agent, actor_prompt, 3, existing_code='')
        if leaf_log[-1]['error'] is not None:
            raise Exception(leaf_log[-1]['error'])

        # Track tokens
        in_tok = leaf_log[-1]['cumulative_tokens'][0]
        out_tok = leaf_log[-1]['cumulative_tokens'][1]
        total_input_token += in_tok
        total_output_token += out_tok

        section_code = leaf_log[-1]['code']

        if leaf_section not in actor_logs:
            actor_logs[leaf_section] = []
        actor_logs[leaf_section].append(leaf_log)

        log.extend(leaf_log)

    return {
        "section_code": section_code,
        "log": log,
        "img_logs": img_logs,
        "critic_logs": critic_logs,
        "actor_logs": actor_logs,
        "total_input_token": total_input_token,
        "total_output_token": total_output_token,
    }


def process_section(
    section_name,
    content,
    outline,
    sections,
    style_logs,
    critic_logs,
    actor_logs,
    img_logs,
    slide_width,
    slide_height,
    name_to_hierarchy,
    critic_template,
    actor_template,
    critic_agent,
    actor_agent,
    neg_img,
    pos_img,
    MAX_ATTEMPTS,
    documentation,
    total_input_token,
    total_output_token,
):
    """
    Handles processing of a single section and its subsections (leaf sections).
    Returns updated logs and token counters for this section.
    """
    results_per_leaf = []

    # Grab the current code for this section
    section_code = style_logs[section_name][-1]['code']

    # Determine which leaf sections to process
    if 'subsections' in content[section_name]:
        subsections = list(content[section_name]['subsections'].keys())
    else:
        subsections = [section_name]

    all_logs_for_section = []

    for leaf_section in subsections:
        # Process this leaf section
        leaf_result = process_leaf_section(
            leaf_section,
            section_name,
            outline,
            content,
            style_logs,
            critic_logs,
            actor_logs,
            img_logs,
            slide_width,
            slide_height,
            name_to_hierarchy,
            critic_template,
            actor_template,
            critic_agent,
            actor_agent,
            neg_img,
            pos_img,
            MAX_ATTEMPTS,
            documentation,
            total_input_token,
            total_output_token,
        )

        # Update logs/tokens
        section_code = leaf_result["section_code"]
        all_logs_for_section.extend(leaf_result["log"])
        img_logs = leaf_result["img_logs"]
        critic_logs = leaf_result["critic_logs"]
        actor_logs = leaf_result["actor_logs"]
        total_input_token = leaf_result["total_input_token"]
        total_output_token = leaf_result["total_output_token"]

    # If we have any logs from the last leaf in this section, append them
    if all_logs_for_section:
        style_logs[section_name].append(all_logs_for_section[-1])

    # Return updated state for merging back in the main thread
    return {
        "section_name": section_name,
        "style_logs": style_logs,
        "critic_logs": critic_logs,
        "actor_logs": actor_logs,
        "img_logs": img_logs,
        "total_input_token": total_input_token,
        "total_output_token": total_output_token
    }

def parallel_by_sections(
    sections,
    content,
    outline,
    style_logs,
    critic_logs,
    actor_logs,
    img_logs,
    slide_width,
    slide_height,
    name_to_hierarchy,
    critic_template,
    actor_template,
    critic_agent,
    actor_agent,
    neg_img,
    pos_img,
    MAX_ATTEMPTS,
    documentation,
    total_input_token,
    total_output_token,
    max_workers=4
):
    """
    Main entry point to parallelize processing across sections.

    Returns the merged logs and token counters after processing all sections in parallel.
    """
    # Because we’ll be modifying dictionaries (like style_logs, etc.),
    # it can be safer to create a copy for the workers, then merge results
    # after. (Below is a simple approach—depending on your scale, consider
    # explicit concurrency controls or a database-backed approach.)
    
    # Summaries from each future
    results = []

    # We’ll store fresh copies for each section to avoid concurrency collisions
    # on dictionary updates. If the data is large, you might want a more
    # sophisticated synchronization or partition approach rather than naive copies.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for section_name in sections:
            # Make shallow copies or deep copies of logs
            _style_logs = copy.deepcopy(style_logs)
            _critic_logs = copy.deepcopy(critic_logs)
            _actor_logs = copy.deepcopy(actor_logs)
            _img_logs = copy.deepcopy(img_logs)
            
            futures.append(executor.submit(
                process_section,
                section_name,
                content,
                outline,
                sections,
                _style_logs,
                _critic_logs,
                _actor_logs,
                _img_logs,
                slide_width,
                slide_height,
                name_to_hierarchy,
                critic_template,
                actor_template,
                critic_agent,
                actor_agent,
                neg_img,
                pos_img,
                MAX_ATTEMPTS,
                documentation,
                total_input_token,
                total_output_token
            ))

        for future in futures:
            results.append(future.result())

    # The code below merges the results.  The method of merging depends on how
    # you prefer to aggregate.  For a minimal approach, we’ll pick the logs from
    # each section, then overwrite or update them in the main dictionaries.
    
    for res in results:
        sec_name = res["section_name"]
        # Overwrite or merge logs as needed
        style_logs[sec_name] = res["style_logs"][sec_name]
        critic_logs.update(res["critic_logs"])
        actor_logs.update(res["actor_logs"])
        img_logs.update(res["img_logs"])
        total_input_token = res["total_input_token"]
        total_output_token = res["total_output_token"]

    return style_logs, critic_logs, actor_logs, img_logs, total_input_token, total_output_token


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

    style_logs, critic_logs, actor_logs, img_logs, total_input_token, total_output_token = parallel_by_sections(
        sections=sections,
        content=content,
        outline=outline,
        style_logs=style_logs,
        critic_logs=critic_logs,
        actor_logs=actor_logs,
        img_logs=img_logs,
        slide_width=slide_width,
        slide_height=slide_height,
        name_to_hierarchy=name_to_hierarchy,
        critic_template=critic_template,
        actor_template=actor_template,
        critic_agent=critic_agent,
        actor_agent=actor_agent,
        neg_img=neg_img,
        pos_img=pos_img,
        MAX_ATTEMPTS=MAX_ATTEMPTS,
        documentation=documentation,
        total_input_token=total_input_token,
        total_output_token=total_output_token,
        max_workers=100,  # or however many worker threads you want
    )

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