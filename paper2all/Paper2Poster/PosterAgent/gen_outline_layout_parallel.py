from dotenv import load_dotenv
import os
import json
import copy
import yaml
import logging
import time
from jinja2 import Environment, StrictUndefined

from utils.src.utils import ppt_to_images, get_json_from_response

from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.messages import BaseMessage

from utils.pptx_utils import *
from utils.wei_utils import *

import pickle as pkl
import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
import concurrent.futures
import sys

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(threadName)s: %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

IMAGE_SCALE_RATIO_MIN = 50
IMAGE_SCALE_RATIO_MAX = 40
TABLE_SCALE_RATIO_MIN = 100
TABLE_SCALE_RATIO_MAX = 80

def layout_process_section_wrapped(
    sections,
    new_outline,
    init_template,
    new_section_template,
    init_actor_sys_msg,
    new_section_actor_sys_msg,
    actor_config,
    documentation,
    max_retry,
    slide_width,
    slide_height
):
    logs = {}
    parallel_results = {}
    total_input_token, total_output_token = 0, 0

    # Switch from ThreadPoolExecutor to ProcessPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = []

        for section_index in range(len(sections)):
            if section_index == 0:
                sys_msg = init_actor_sys_msg
                prompt_template = init_template
            else:
                sys_msg = new_section_actor_sys_msg
                prompt_template = new_section_template

            actor_model = ModelFactory.create(
                model_platform=actor_config['model_platform'],
                model_type=actor_config['model_type'],
                model_config_dict=actor_config['model_config'],
            )

            future = executor.submit(
                layout_process_section,
                section_index,
                sections,
                new_outline,
                prompt_template,
                documentation,
                sys_msg,
                actor_model,
                10,
                max_retry,
                slide_width,
                slide_height
            )
            futures.append(future)

        # Collect results as processes complete
        for future in as_completed(futures):
            section_index, section_logs, in_toks, out_toks = future.result()

            # Store logs by section index
            parallel_results[section_index] = section_logs

            # Update token counters
            total_input_token += in_toks
            total_output_token += out_toks

    # Merge results back into `logs`
    for section_index, section_logs in parallel_results.items():
        curr_section = sections[section_index]
        logs[curr_section] = section_logs

    return logs, total_input_token, total_output_token

def create_agent_fn(sys_msg, agent_model, window_size=10):
    agent = ChatAgent(
        system_message=sys_msg,
        model=agent_model,
        message_window_size=window_size,
    )
    return agent

def layout_h2_process_section(
    section, 
    outline_no_sub_locations, 
    h2_actor_template,
    create_h2_actor_agent,  # If you need a fresh agent for each thread
):
    """
    Run the logic for a single section. 
    Returns a tuple containing:
      - section name (or id),
      - updated subsection-location dict,
      - input token count,
      - output token count
    """
    print(f'Generating h2 for section {section}...', flush=True)

    # 1) Create the prompt
    section_outline = {section: outline_no_sub_locations[section]}
    section_jinja_args = {
        'section_outline': json.dumps(section_outline, indent=4),
    }
    section_prompt = h2_actor_template.render(**section_jinja_args)

    # 2) Prepare a fresh agent or reuse existing (thread-safe?) agent
    #    If your h2_actor_agent is not thread-safe, instantiate a new one here:
    h2_actor_agent = create_h2_actor_agent()
    h2_actor_agent.reset()

    # 3) Get response
    response = h2_actor_agent.step(section_prompt)
    input_token, output_token = account_token(response)

    # 4) Parse JSON
    subsection_location = get_json_from_response(response.msgs[0].content)

    # 5) Create a dict from the sub-locations
    sec_bbox = outline_no_sub_locations[section]['location']
    subsection_location_dict = {}
    for k, v in subsection_location.items():
        subsection_location_dict[k] = {
            'left': v['location'][0],
            'top': v['location'][1],
            'width': v['location'][2],
            'height': v['location'][3]
        }

    # 6) Validate and possibly revise
    is_valid, revised = validate_and_adjust_subsections(sec_bbox, subsection_location_dict)
    if not is_valid:
        # Try once more
        is_valid, revised = validate_and_adjust_subsections(sec_bbox, revised)
        assert is_valid, "Failed to adjust subsections to fit section"
        final_sub_loc = revised
    else:
        final_sub_loc = subsection_location

    # Return all data needed by the main thread
    return section, final_sub_loc, input_token, output_token

def layout_process_section(
    section_index,
    sections,
    new_outline,
    new_section_template,
    documentation,
    sys_msg,
    agent_model,
    window_size,
    max_retry,
    slide_width,
    slide_height
):
    """
    Runs the 'gen_layout' logic for a single section_index.

    Returns a tuple:
        (section_index, updated_log, input_tokens, output_tokens)
    """
    curr_section = sections[section_index]
    print(f'Generating h1 layout for section {curr_section}...')

    # Build outline JSON just for current section
    new_section_outline = {curr_section: new_outline[curr_section]}
    if section_index == 0:
        new_section_outline = {'meta': new_outline['meta'], curr_section: new_outline[curr_section]}
    new_section_jinja_args = {
        'json_outline': new_section_outline,
        'function_docs': documentation,
        'file_name': f'poster_{section_index}.pptx'
    }

    # Render prompt
    new_section_prompt = new_section_template.render(**new_section_jinja_args)
    
    existing_code = ''  # Or fetch from a stable location that is not dependent on real-time results

    # Call gen_layout
    section_logs = gen_layout_parallel(
        create_agent_fn(
            sys_msg,
            agent_model,
            window_size
        ), 
        new_section_prompt, 
        max_retry, 
        existing_code=existing_code,
        slide_width=slide_width,
        slide_height=slide_height,
        tmp_name=section_index
    )
    
    if section_logs[-1]['error'] is not None:
        print(f'Failed to generate layout for section {curr_section}.')
        return None

    in_toks, out_toks = section_logs[-1]['cumulative_tokens']
    return (section_index, section_logs, in_toks, out_toks)

def get_outline_location(outline, subsection=False):
    outline_location = {}
    for k, v in outline.items():
        if k == 'meta':
            continue
        outline_location[k] = {
            'location': v['location'],
        }
        if subsection:
            if 'subsections' in v:
                outline_location[k]['subsections'] = get_outline_location(v['subsections'])
    return outline_location

def apply_outline_location(outline, location, subsection=False):
    new_outline = {}
    for k, v in outline.items():
        if k == 'meta':
            new_outline[k] = v
            continue
        new_outline[k] = copy.deepcopy(v)
        new_outline[k]['location'] = location[k]['location']
        if subsection:
            if 'subsections' in v:
                new_outline[k]['subsections'] = apply_outline_location(v['subsections'], location[k]['subsections'])

    return new_outline

def fill_location(outline, section_name, location_dict):
    new_outline = copy.deepcopy(outline)
    if 'subsections' not in new_outline[section_name]:
        return new_outline
    for k, v in new_outline[section_name]['subsections'].items():
        v['location'] = location_dict[k]['location']
    return new_outline

def recover_name_and_location(outline_no_name, outline):
    new_outline = copy.deepcopy(outline_no_name)
    for k, v in outline_no_name.items():
        if k == 'meta':
            continue
        new_outline[k]['name'] = outline[k]['name']
        if type(new_outline[k]['location']) == list:
            new_outline[k]['location'] = {
                'left': v['location'][0],
                'top': v['location'][1],
                'width': v['location'][2],
                'height': v['location'][3]
            }
        if 'subsections' in v:
            for k_sub, v_sub in v['subsections'].items():
                new_outline[k]['subsections'][k_sub]['name'] = outline[k]['subsections'][k_sub]['name']
                if type(new_outline[k]['subsections'][k_sub]['location']) == list:
                    new_outline[k]['subsections'][k_sub]['location'] = {
                        'left': v_sub['location'][0],
                        'top': v_sub['location'][1],
                        'width': v_sub['location'][2],
                        'height': v_sub['location'][3]
                    }
    return new_outline


def validate_and_adjust_subsections(section_bbox, subsection_bboxes):
    """
    Validate that the given subsections collectively occupy the entire section.
    If not, return an adjusted version that fixes the layout.
    
    We assume all subsections are intended to be stacked vertically with no gaps,
    spanning the full width of the section.

    :param section_bbox: dict with keys ["left", "top", "width", "height"]
    :param subsection_bboxes: dict of subsection_name -> bounding_box (each also
                              with keys ["left", "top", "width", "height"])
    :return: (is_valid, revised_subsections)
             where is_valid is True/False,
             and revised_subsections is either the same as subsection_bboxes if valid,
             or a new dict of adjusted bounding boxes if invalid.
    """

    # Helper functions
    def _right(bbox):
        return bbox["left"] + bbox["width"]
    
    def _bottom(bbox):
        return bbox["top"] + bbox["height"]
    
    section_left = section_bbox["left"]
    section_top = section_bbox["top"]
    section_right = section_left + section_bbox["width"]
    section_bottom = section_top + section_bbox["height"]

    # Convert dictionary to a list of (subsection_name, bbox) pairs
    items = list(subsection_bboxes.items())
    if not items:
        # No subsections is definitely not valid if we want to fill the section
        return False, None

    # Sort subsections by their 'top' coordinate
    items_sorted = sorted(items, key=lambda x: x[1]["top"])

    # ---------------------------
    # Step 1: Validate
    # ---------------------------
    # We'll check:
    # 1. left/right boundaries match the section for each subsection
    # 2. The first subsection's top == section_top
    # 3. The last subsection's bottom == section_bottom
    # 4. Each pair of consecutive subsections lines up exactly
    #    (previous bottom == current top) with no gap or overlap.

    is_valid = True

    # Check left/right for each
    for name, bbox in items_sorted:
        if bbox["left"] != section_left or _right(bbox) != section_right:
            is_valid = False
            break

    # Check alignment for the first and last
    if is_valid:
        first_sub_name, first_sub_bbox = items_sorted[0]
        if first_sub_bbox["top"] != section_top:
            is_valid = False

    if is_valid:
        last_sub_name, last_sub_bbox = items_sorted[-1]
        if _bottom(last_sub_bbox) != section_bottom:
            is_valid = False

    # Check consecutive alignment
    if is_valid:
        for i in range(len(items_sorted) - 1):
            _, current_bbox  = items_sorted[i]
            _, next_bbox     = items_sorted[i + 1]
            if _bottom(current_bbox) != next_bbox["top"]:
                is_valid = False
                break

    # If everything passed, we return
    if is_valid:
        return True, subsection_bboxes

    # ---------------------------
    # Step 2: Revise
    # ---------------------------
    # We will adjust all subsection bboxes so that they occupy
    # the entire section exactly, preserving each original bbox's
    # height *ratio* if possible.

    # 2a. Compute total original height (in the order of sorted items)
    original_heights = [bbox["height"] for _, bbox in items_sorted]
    total_original_height = sum(original_heights)

    # Avoid divide-by-zero if somehow there's a 0 height
    if total_original_height <= 0:
        # Fallback: split the section equally among subsections
        # to avoid zero or negative heights
        chunk_height = section_bbox["height"] / len(items_sorted)
        scale_heights = [chunk_height] * len(items_sorted)
    else:
        # Scale each original height by the ratio of
        # (section total height / sum of original heights)
        scale = section_bbox["height"] / total_original_height
        scale_heights = [h * scale for h in original_heights]

    # 2b. Assign bounding boxes top->bottom, ensuring no gap
    revised = {}
    current_top = section_top
    for i, (name, original_bbox) in enumerate(items_sorted):
        revised_height = scale_heights[i]
        # If there's floating error, we can clamp in the last iteration
        # so that the bottom exactly matches section_bottom.
        # But for simplicity, we'll keep it straightforward unless needed.

        revised[name] = {
            "left": section_left,
            "top": current_top,
            "width": section_bbox["width"],
            "height": revised_height
        }
        # Update current_top for next subsection
        current_top += revised_height

    # Due to potential float rounding, we can enforce the last subsection
    # to exactly end at section_bottom:
    last_name = items_sorted[-1][0]
    # Recompute the actual bottom after the above assignment
    new_bottom = revised[last_name]["top"] + revised[last_name]["height"]
    diff = new_bottom - section_bottom
    if abs(diff) > 1e-9:
        # Adjust the last subsection's height
        revised[last_name]["height"] -= diff

    # Return the revised dictionary
    return False, revised

def filter_image_table(args, filter_config):
    images = json.load(open(f'images_and_tables/{args.poster_name}_images.json', 'r'))
    tables = json.load(open(f'images_and_tables/{args.poster_name}_tables.json', 'r'))
    doc_json = json.load(open(f'contents/{args.model_name}_{args.poster_name}_raw_content.json', 'r'))
    agent_filter = 'image_table_filter_agent'
    with open(f"prompt_templates/{agent_filter}.yaml", "r") as f:
        config_filter = yaml.safe_load(f)

    image_information = {}
    for k, v in images.items():
        image_information[k] = copy.deepcopy(v)
        image_information[k]['min_width'] = v['width'] // IMAGE_SCALE_RATIO_MIN
        image_information[k]['min_height'] = v['height'] // IMAGE_SCALE_RATIO_MIN
        image_information[k]['max_width'] = v['width'] // IMAGE_SCALE_RATIO_MAX
        image_information[k]['max_height'] = v['height'] // IMAGE_SCALE_RATIO_MAX

    table_information = {}
    for k, v in tables.items():
        table_information[k] = copy.deepcopy(v)
        table_information[k]['min_width'] = v['width'] // TABLE_SCALE_RATIO_MIN
        table_information[k]['min_height'] = v['height'] // TABLE_SCALE_RATIO_MIN
        table_information[k]['max_width'] = v['width'] // TABLE_SCALE_RATIO_MAX
        table_information[k]['max_height'] = v['height'] // TABLE_SCALE_RATIO_MAX

    filter_actor_sys_msg = config_filter['system_prompt']

    filter_model = ModelFactory.create(
        model_platform=filter_config['model_platform'],
        model_type=filter_config['model_type'],
        model_config_dict=filter_config['model_config'],
    )
    filter_actor_agent = ChatAgent(
        system_message=filter_actor_sys_msg,
        model=filter_model,
        message_window_size=10, # [Optional] the length for chat memory
    )

    filter_jinja_args = {
        'json_content': doc_json,
        'table_information': table_information,
        'image_information': image_information,
    }
    jinja_env = Environment(undefined=StrictUndefined)
    filter_prompt = jinja_env.from_string(config_filter["template"])
    response = filter_actor_agent.step(filter_prompt.render(**filter_jinja_args))
    input_token, output_token = account_token(response)
    response_json = get_json_from_response(response.msgs[0].content)
    table_information = response_json['table_information']
    image_information = response_json['image_information']
    json.dump(images, open(f'images_and_tables/{args.poster_name}_images_filtered.json', 'w'), indent=4)
    json.dump(tables, open(f'images_and_tables/{args.poster_name}_tables_filtered.json', 'w'), indent=4)

    return input_token, output_token

def gen_outline_layout(args, actor_config, critic_config):
    poster_log_path = f'log/{args.model_name}_{args.poster_name}_poster_{args.index}'
    if not os.path.exists(poster_log_path):
        os.mkdir(poster_log_path)
    total_input_token, total_output_token = 0, 0
    consumption_log = {
        'outline': [],
        'h1_actor': [],
        'h2_actor': [],
        'h1_critic': [],
        'gen_layout': []
    }
    jinja_env = Environment(undefined=StrictUndefined)
    outline_file_path = f'outlines/{args.model_name}_{args.poster_name}_outline_{args.index}.json'
    agent_name = 'poster_planner_new'
    agent_init_name = 'layout_agent_init_parallel'
    agent_new_section_name = 'layout_agent_new_section_parallel'
    h1_critic_name = 'critic_layout_hierarchy_1'
    h2_actor_name = 'actor_layout_hierarchy_2'

    doc_json = json.load(open(f'contents/{args.model_name}_{args.poster_name}_raw_content.json', 'r'))
    filtered_table_information = json.load(open(f'images_and_tables/{args.poster_name}_tables_filtered.json', 'r'))
    filtered_image_information = json.load(open(f'images_and_tables/{args.poster_name}_images_filtered.json', 'r'))

    with open(f"prompt_templates/{agent_name}.yaml", "r") as f:
        planner_config = yaml.safe_load(f)

    with open(f"prompt_templates/{agent_init_name}.yaml", "r") as f:
        config_init = yaml.safe_load(f)

    with open(f"prompt_templates/{agent_new_section_name}.yaml", "r") as f:
        config_new_section = yaml.safe_load(f)

    with open(f"prompt_templates/{h1_critic_name}.yaml", "r") as f:
        config_h1_critic = yaml.safe_load(f)

    with open(f"prompt_templates/{h2_actor_name}.yaml", "r") as f:
        config_h2_actor = yaml.safe_load(f)

    planner_model = ModelFactory.create(
        model_platform=actor_config['model_platform'],
        model_type=actor_config['model_type'],
        model_config_dict=actor_config['model_config'],
    )

    planner_agent = ChatAgent(
        system_message=planner_config['system_prompt'],
        model=planner_model,
        message_window_size=10,
    )

    outline_template = jinja_env.from_string(planner_config["template"])

    planner_jinja_args = {
        'json_content': doc_json,
        'table_information': filtered_table_information,
        'image_information': filtered_image_information,
    }

    actor_model = ModelFactory.create(
        model_platform=actor_config['model_platform'],
        model_type=actor_config['model_type'],
        model_config_dict=actor_config['model_config'],
    )

    init_actor_sys_msg = config_init['system_prompt']

    def create_init_actor_agent():
        actor_model = ModelFactory.create(
            model_platform=actor_config['model_platform'],
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
        )
        init_actor_agent = ChatAgent(
            system_message=init_actor_sys_msg,
            model=actor_model,
            message_window_size=10,
        )
        return init_actor_agent

    new_section_actor_sys_msg = config_new_section['system_prompt']

    def create_new_section_actor_agent():
        actor_model = ModelFactory.create(
            model_platform=actor_config['model_platform'],
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
        )
        new_section_actor_agent = ChatAgent(
            system_message=new_section_actor_sys_msg,
            model=actor_model,
            message_window_size=10,
        )
        return new_section_actor_agent

    h1_critic_model = ModelFactory.create(
        model_platform=critic_config['model_platform'],
        model_type=critic_config['model_type'],
        model_config_dict=critic_config['model_config'],
    )

    h1_critic_sys_msg = config_h1_critic['system_prompt']

    h1_critic_agent = ChatAgent(
        system_message=h1_critic_sys_msg,
        model=h1_critic_model,
        message_window_size=None,
    )

    h1_pos_example = Image.open('h1_example/h1_pos.jpg')
    h1_neg_example = Image.open('h1_example/h1_neg.jpg')

    h2_actor_model = ModelFactory.create(
        model_platform=actor_config['model_platform'],
        model_type=actor_config['model_type'],
        model_config_dict=actor_config['model_config'],
    )

    h2_actor_sys_msg = config_h2_actor['system_prompt']

    def create_h2_actor_agent():
        h2_actor_model = ModelFactory.create(
            model_platform=actor_config['model_platform'],
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
        )
        h2_actor_agent = ChatAgent(
            system_message=h2_actor_sys_msg,
            model=h2_actor_model,
            message_window_size=10,
        )
        return h2_actor_agent

    
    init_template = jinja_env.from_string(config_init["template"])
    new_section_template = jinja_env.from_string(config_new_section["template"])
    h1_critic_template = jinja_env.from_string(config_h1_critic["template"])

    attempt = 0
    while True:
        print(f'Generating outline attempt {attempt}...', flush=True)
        planner_prompt = outline_template.render(**planner_jinja_args)
        planner_agent.reset()
        response = planner_agent.step(planner_prompt)
        outline = get_json_from_response(response.msgs[0].content)
        input_token, output_token = account_token(response)
        sections = list(outline.keys())
        sections = [x for x in sections if x != 'meta']
        slide_width = outline['meta']['width']
        slide_height = outline['meta']['height']
        name_to_hierarchy = get_hierarchy(outline)
        consumption_log['outline'].append((input_token, output_token))
        total_input_token += input_token
        total_output_token += output_token
        init_outline = {'meta': outline['meta'], sections[0]: outline[sections[0]]}

        new_outline = outline

        init_jinja_args = {
            'json_outline': init_outline,
            'function_docs': documentation
        }

        init_prompt = init_template.render(**init_jinja_args)

        # hierarchy 1 only
        outline_location = get_outline_location(outline, subsection=False)

        logs, layout_cumulative_input_token, layout_cumulative_output_token = layout_process_section_wrapped(
            sections,
            new_outline,
            init_template,
            new_section_template,
            init_actor_sys_msg,
            new_section_actor_sys_msg,
            actor_config,
            documentation,
            args.max_retry,
            slide_width,
            slide_height
        )

        concatenated_code = utils_functions
        for section_index in range(len(sections)):
            section = sections[section_index]
            concatenated_code += '\n' + logs[section][-1]['code']
            presentation_object_name = logs[section][-1]['output'].replace('\n', '')
            concatenated_code += '\n' + f'save_presentation({presentation_object_name}, file_name="poster_{section_index + 1}.pptx")'

        concatenated_code += f'''
name_to_hierarchy = {name_to_hierarchy}
identifier = "parallel"
poster_path = "poster_{section_index + 1}.pptx"
get_visual_cues(name_to_hierarchy, identifier, poster_path)
'''
        output, error = run_code_with_utils(concatenated_code, utils_functions)
        if error is not None:
            print(error, flush=True)
            attempt += 1
            continue

        consumption_log['h1_actor'].append((layout_cumulative_input_token, layout_cumulative_output_token))
        total_input_token += layout_cumulative_input_token
        total_output_token += layout_cumulative_output_token

        h1_path = f'tmp/poster_<parallel>_hierarchy_1.pptx'
        h2_path = f'tmp/poster_<parallel>_hierarchy_2.pptx'

        h1_filled_path = f'tmp/poster_<parallel>_hierarchy_1_filled.pptx'
        h2_filled_path = f'tmp/poster_<parallel>_hierarchy_2_filled.pptx'

        ppt_to_images(h1_path, 'tmp/layout_h1')
        ppt_to_images(h2_path, 'tmp/layout_h2')
        ppt_to_images(h1_filled_path, 'tmp/layout_h1_filled')
        ppt_to_images(h2_filled_path, 'tmp/layout_h2_filled')

        h1_img = Image.open('tmp/layout_h1/slide_0001.jpg')
        h2_img = Image.open('tmp/layout_h2/slide_0001.jpg')
        h1_filled_img = Image.open('tmp/layout_h1_filled/slide_0001.jpg')
        h2_filled_img = Image.open('tmp/layout_h2_filled/slide_0001.jpg')

        h1_critic_msg = BaseMessage.make_user_message(
            role_name='User',
            content=h1_critic_template.render(),
            image_list=[h1_neg_example, h1_pos_example, h1_filled_img]
        )

        outline_bbox_dict = {}
        for k, v in outline_location.items():
            outline_bbox_dict[k] = v['location']

        bbox_check_result = check_bounding_boxes(
            outline_bbox_dict, 
            new_outline['meta']['width'], 
            new_outline['meta']['height']
        )

        if len(bbox_check_result) != 0:
            print(bbox_check_result, flush=True)
            attempt += 1
            continue

        h1_critic_agent.reset()
        response = h1_critic_agent.step(h1_critic_msg)
        input_token, output_token = account_token(response)
        consumption_log['h1_critic'].append((input_token, output_token))
        total_input_token += input_token
        total_output_token += output_token
        if response.msgs[0].content == 'T':
            print('Blank area detected.', flush=True)
            attempt += 1
            continue

        print('Sucessfully generated outline.', flush=True)

        break

    outline_bbox_dict = {}
    for k, v in outline_location.items():
        outline_bbox_dict[k] = v['location']

    # Generate subsection locations
    outline_no_sub_locations = copy.deepcopy(new_outline)
    if 'meta' in outline_no_sub_locations:
        outline_no_sub_locations.pop('meta')

    for k, v in outline_no_sub_locations.items():
        if 'subsections' in v:
            subsections = v['subsections']
            for k_sub, v_sub in subsections.items():
                del v_sub['location']
                del v_sub['name']

    h2_actor_template = jinja_env.from_string(config_h2_actor["template"])

    h2_cumulative_input_token = 0
    h2_cumulative_output_token = 0
    
    updated_sections = []

    with ThreadPoolExecutor() as executor:
        # Kick off all tasks
        future_to_section = {
            executor.submit(
                layout_h2_process_section, 
                section, 
                outline_no_sub_locations, 
                h2_actor_template,
                create_h2_actor_agent  # pass the factory function
            ): section
            for section in sections
        }

        # Gather results as they complete
        for future in concurrent.futures.as_completed(future_to_section):
            section = future_to_section[future]
            sec, final_sub_loc, in_toks, out_toks = future.result()
            
            # Accumulate token usage
            h2_cumulative_input_token += in_toks
            h2_cumulative_output_token += out_toks
            
            # Stash the final sub-loc for merging
            updated_sections.append((sec, final_sub_loc))

    # Now merge each updated subsection location back into outline_no_sub_locations
    for (section, final_sub_loc) in updated_sections:
        outline_no_sub_locations = fill_location(
            outline_no_sub_locations, 
            section, 
            final_sub_loc
        )

    consumption_log['h2_actor'].append((h2_cumulative_input_token, h2_cumulative_output_token))
    total_input_token += h2_cumulative_input_token
    total_output_token += h2_cumulative_output_token

    outline_no_sub_locations['meta'] = outline['meta']
    outline_no_sub_locations_with_name = recover_name_and_location(outline_no_sub_locations, new_outline)
    new_outline = outline_no_sub_locations_with_name

    ### Outline finalized, actually generate layout

    logs = {}

    gen_layout_cumulative_input_token = 0
    gen_layout_cumulative_output_token = 0
    init_outline = {'meta': outline['meta'], sections[0]: outline[sections[0]]}

    new_outline = outline

    init_jinja_args = {
        'json_outline': init_outline,
        'function_docs': documentation
    }

    outline_location = get_outline_location(outline, subsection=False)
    logs = {}

    # We'll store all updated logs here, keyed by section_index.
    parallel_results = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for section_index in range(len(sections)):
            if section_index == 0:
                create_agent_fn = create_init_actor_agent
                prompt_template = init_template
            else:
                create_agent_fn = create_new_section_actor_agent
                prompt_template = new_section_template
            future = executor.submit(
                layout_process_section,
                section_index,
                sections,
                new_outline,
                prompt_template,
                documentation,
                create_agent_fn,
                args.max_retry,
                name_to_hierarchy,
                slide_width,
                slide_height
            )
            futures.append(future)

        # Collect the results as they come in
        for future in concurrent.futures.as_completed(futures):
            try:
                section_index, section_logs, in_toks, out_toks = future.result()
                
                # Store these logs in a dictionary keyed by the section index
                parallel_results[section_index] = section_logs

                # Update token counters
                gen_layout_cumulative_input_token += in_toks
                gen_layout_cumulative_output_token += out_toks

            except Exception as exc:
                print(f"[ERROR] A section failed: {exc}", flush=True)
                # Possibly re-raise if you want to stop everything on error
                # raise

    # After all tasks complete, merge the results back into `logs`
    for section_index, section_logs in parallel_results.items():
        curr_section = sections[section_index]
        logs[curr_section] = section_logs

    concatenated_code = utils_functions
    for section_index in range(len(sections)):
        section = sections[section_index]
        concatenated_code += '\n' + logs[section][-1]['code']
        concatenated_code += '\n' + f'save_presentation(presentation, file_name="poster_{section_index + 1}.pptx")'

    concatenated_code += f'''
name_to_hierarchy = {name_to_hierarchy}
identifier = "parallel"
poster_path = "poster_{section_index + 1}.pptx"
get_visual_cues(name_to_hierarchy, identifier, poster_path)
'''
    output, error = run_code(concatenated_code)
    if error is not None:
        print(f'Failed to generate layout for section {curr_section}.')

    consumption_log['h1_actor'].append((layout_cumulative_input_token, layout_cumulative_output_token))
    total_input_token += gen_layout_cumulative_input_token
    total_output_token += gen_layout_cumulative_output_token

    h1_path = f'tmp/poster_<parallel>_hierarchy_1.pptx'
    h2_path = f'tmp/poster_<parallel>_hierarchy_2.pptx'

    h1_filled_path = f'tmp/poster_<parallel>_hierarchy_1_filled.pptx'
    h2_filled_path = f'tmp/poster_<parallel>_hierarchy_2_filled.pptx'

    ppt_to_images(h1_path, 'tmp/layout_h1')
    ppt_to_images(h2_path, 'tmp/layout_h2')
    ppt_to_images(h1_filled_path, 'tmp/layout_h1_filled')
    ppt_to_images(h2_filled_path, 'tmp/layout_h2_filled')

    h1_img = Image.open('tmp/layout_h1/slide_0001.jpg')
    h2_img = Image.open('tmp/layout_h2/slide_0001.jpg')
    h1_filled_img = Image.open('tmp/layout_h1_filled/slide_0001.jpg')
    h2_filled_img = Image.open('tmp/layout_h2_filled/slide_0001.jpg')


    ckpt = {
        'logs': logs,
        'outline': new_outline,
        'name_to_hierarchy': name_to_hierarchy,
        'consumption_log': consumption_log,
        'total_input_token': total_input_token,
        'total_output_token': total_output_token,
    }

    with open(f'checkpoints/{args.model_name}_{args.poster_name}_ckpt_{args.index}.pkl', 'wb') as f:
        pkl.dump(ckpt, f)

    json.dump(
        new_outline,
        open(outline_file_path, "w"),
        ensure_ascii=False,
        indent=4,
    )

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

    input_token, output_token = filter_image_table(args, actor_config)
    print(f'Token consumption: {input_token} -> {output_token}', flush=True)

    input_token, output_token = gen_outline_layout(args, actor_config, critic_config)
    print(f'Token consumption: {input_token} -> {output_token}', flush=True)