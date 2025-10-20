from dotenv import load_dotenv
import os
import json
import copy
import yaml
from jinja2 import Environment, StrictUndefined

from utils.src.utils import ppt_to_images, get_json_from_response

from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.messages import BaseMessage

from utils.pptx_utils import *
from utils.wei_utils import *

import pickle as pkl
import argparse

load_dotenv()

IMAGE_SCALE_RATIO_MIN = 50
IMAGE_SCALE_RATIO_MAX = 40
TABLE_SCALE_RATIO_MIN = 100
TABLE_SCALE_RATIO_MAX = 80

def compute_tp(raw_content_json):
    total_length = 0
    for section in raw_content_json['sections']:
        total_length += len(section['content'])

    for i in range(len(raw_content_json['sections'])):
        raw_content_json['sections'][i]['tp'] = len(raw_content_json['sections'][i]['content']) / total_length
        raw_content_json['sections'][i]['text_len'] = len(raw_content_json['sections'][i]['content'])

def compute_gp(table_info, image_info):
    total_area = 0
    for k, v in table_info.items():
        total_area += v['figure_size']

    for k, v in image_info.items():
        total_area += v['figure_size']

    for k, v in table_info.items():
        v['gp'] = v['figure_size'] / total_area

    for k, v in image_info.items():
        v['gp'] = v['figure_size'] / total_area

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
    images = json.load(open(f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/{args.poster_name}_images.json', 'r'))
    tables = json.load(open(f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/{args.poster_name}_tables.json', 'r'))
    doc_json = json.load(open(f'contents/<{args.model_name_t}_{args.model_name_v}>_{args.poster_name}_raw_content.json', 'r'))
    agent_filter = 'image_table_filter_agent'
    with open(f"utils/prompt_templates/{agent_filter}.yaml", "r", encoding="utf-8") as f:
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

    if args.model_name_t.startswith('vllm_qwen'):
        filter_model = ModelFactory.create(
            model_platform=filter_config['model_platform'],
            model_type=filter_config['model_type'],
            model_config_dict=filter_config['model_config'],
            url=filter_config['url'],
        )
    else:
        filter_model = ModelFactory.create(
            model_platform=filter_config['model_platform'],
            model_type=filter_config['model_type'],
            model_config_dict=filter_config['model_config'],
        )

    filter_actor_agent = ChatAgent(
        system_message=filter_actor_sys_msg,
        model=filter_model,
        message_window_size=10,
    )

    filter_jinja_args = {
        'json_content': doc_json,
        'table_information': json.dumps(table_information, indent=4),
        'image_information': json.dumps(image_information, indent=4),
    }
    jinja_env = Environment(undefined=StrictUndefined)
    filter_prompt = jinja_env.from_string(config_filter["template"])
    filter_actor_agent.reset()
    response = filter_actor_agent.step(filter_prompt.render(**filter_jinja_args))
    input_token, output_token = account_token(response)
    response_json = get_json_from_response(response.msgs[0].content)
    table_information = response_json['table_information']
    image_information = response_json['image_information']
    json.dump(images, open(f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/{args.poster_name}_images_filtered.json', 'w'), indent=4)
    json.dump(tables, open(f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/{args.poster_name}_tables_filtered.json', 'w'), indent=4)

    return input_token, output_token

def gen_outline_layout_v2(args, actor_config):
    total_input_token, total_output_token = 0, 0
    agent_name = 'poster_planner_new_v2'
    doc_json = json.load(open(f'contents/<{args.model_name_t}_{args.model_name_v}>_{args.poster_name}_raw_content.json', 'r'))
    filtered_table_information = json.load(open(f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/{args.poster_name}_tables_filtered.json', 'r'))
    filtered_image_information = json.load(open(f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/{args.poster_name}_images_filtered.json', 'r'))

    filtered_table_information_captions = {}
    filtered_image_information_captions = {}

    for k, v in filtered_table_information.items():
        filtered_table_information_captions[k] = {
            v['caption']
        }

    for k, v in filtered_image_information.items():
        filtered_image_information_captions[k] = {
            v['caption']
        }

    with open(f"utils/prompt_templates/{agent_name}.yaml", "r", encoding="utf-8") as f:
        planner_config = yaml.safe_load(f)

    compute_tp(doc_json)

    jinja_env = Environment(undefined=StrictUndefined)
    outline_template = jinja_env.from_string(planner_config["template"])
    planner_jinja_args = {
        'json_content': doc_json,
        'table_information': filtered_table_information_captions,
        'image_information': filtered_image_information_captions,
    }

    if args.model_name_t.startswith('vllm_qwen'):
        planner_model = ModelFactory.create(
            model_platform=actor_config['model_platform'],
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
            url=actor_config['url'],
        )
    else:
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

    print(f'Generating outline...')
    planner_prompt = outline_template.render(**planner_jinja_args)
    planner_agent.reset()
    response = planner_agent.step(planner_prompt)
    input_token, output_token = account_token(response)
    total_input_token += input_token
    total_output_token += output_token

    figure_arrangement = get_json_from_response(response.msgs[0].content)

    print(f'Figure arrangement: {json.dumps(figure_arrangement, indent=4)}')

    arranged_images = {}
    arranged_tables = {}
    assigned_images = set()
    assigned_tables = set()
    
    for section_name, figure in figure_arrangement.items():
        if 'image' in figure:
            image_id = str(figure['image'])
            if image_id in assigned_images:
                continue
            if image_id in filtered_image_information:
                arranged_images[image_id] = filtered_image_information[image_id]
                assigned_images.add(image_id)
        if 'table' in figure:
            table_id = str(figure['table'])
            if table_id in assigned_tables:
                continue
            if table_id in filtered_table_information:
                arranged_tables[table_id] = filtered_table_information[table_id]
                assigned_tables.add(table_id)
    
    compute_gp(arranged_tables, arranged_images)

    # Obtain panel input
    paper_panels = []
    for i in range(len(doc_json['sections'])):
        section = doc_json['sections'][i]
        panel = {}
        panel['panel_id'] = i
        panel['section_name'] = section['title']
        panel['tp'] = section['tp']
        panel['text_len'] = section['text_len']
        panel['gp'] = 0
        panel['figure_size'] = 0
        panel['figure_aspect'] = 1
        if section['title'] in figure_arrangement:
            curr_arrangement = figure_arrangement[section['title']]
            if 'table' in curr_arrangement:
                table_id = str(curr_arrangement['table'])
                if table_id in arranged_tables:
                    panel['gp'] = arranged_tables[table_id]['gp']
                    panel['figure_size'] = arranged_tables[table_id]['figure_size']
                    panel['figure_aspect'] = arranged_tables[table_id]['figure_aspect']
            elif 'image' in curr_arrangement:
                image_id = str(curr_arrangement['image'])
                if image_id in arranged_images:
                    panel['gp'] = arranged_images[image_id]['gp']
                    panel['figure_size'] = arranged_images[image_id]['figure_size']
                    panel['figure_aspect'] = arranged_images[image_id]['figure_aspect']

        paper_panels.append(panel)

    return total_input_token, total_output_token, paper_panels, figure_arrangement

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
    agent_init_name = 'layout_agent_init'
    agent_new_section_name = 'layout_agent_new_section'
    h1_critic_name = 'critic_layout_hierarchy_1'
    h2_actor_name = 'actor_layout_hierarchy_2'

    doc_json = json.load(open(f'contents/{args.model_name}_{args.poster_name}_raw_content.json', 'r'))
    filtered_table_information = json.load(open(f'images_and_tables/{args.poster_name}_tables_filtered.json', 'r'))
    filtered_image_information = json.load(open(f'images_and_tables/{args.poster_name}_images_filtered.json', 'r'))

    with open(f"utils/prompt_templates/{agent_name}.yaml", "r", encoding="utf-8") as f:
        planner_config = yaml.safe_load(f)

    with open(f"utils/prompt_templates/{agent_init_name}.yaml", "r", encoding="utf-8") as f:
        config_init = yaml.safe_load(f)

    with open(f"utils/prompt_templates/{agent_new_section_name}.yaml", "r", encoding="utf-8") as f:
        config_new_section = yaml.safe_load(f)

    with open(f"utils/prompt_templates/{h1_critic_name}.yaml", "r", encoding="utf-8") as f:
        config_h1_critic = yaml.safe_load(f)

    with open(f"utils/prompt_templates/{h2_actor_name}.yaml", "r", encoding="utf-8") as f:
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

    init_actor_agent = ChatAgent(
        system_message=init_actor_sys_msg,
        model=actor_model,
        message_window_size=10,
    )

    new_section_actor_sys_msg = config_new_section['system_prompt']
    new_section_actor_agent = ChatAgent(
        system_message=new_section_actor_sys_msg,
        model=actor_model,
        message_window_size=10,
    )

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

    h1_pos_example = Image.open('assets/h1_example/h1_pos.jpg')
    h1_neg_example = Image.open('assets/h1_example/h1_neg.jpg')

    h2_actor_model = ModelFactory.create(
        model_platform=actor_config['model_platform'],
        model_type=actor_config['model_type'],
        model_config_dict=actor_config['model_config'],
    )

    h2_actor_sys_msg = config_h2_actor['system_prompt']

    h2_actor_agent = ChatAgent(
        system_message=h2_actor_sys_msg,
        model=h2_actor_model,
        message_window_size=10,
    )

    attempt = 0
    while True:
        print(f'Generating outline attempt {attempt}...')
        planner_prompt = outline_template.render(**planner_jinja_args)
        planner_agent.reset()
        response = planner_agent.step(planner_prompt)
        input_token, output_token = account_token(response)
        consumption_log['outline'].append((input_token, output_token))
        total_input_token += input_token
        total_output_token += output_token

        outline = get_json_from_response(response.msgs[0].content)
        name_to_hierarchy = get_hierarchy(outline)

        sections = list(outline.keys())
        sections = [x for x in sections if x != 'meta']
        init_template = jinja_env.from_string(config_init["template"])
        new_section_template = jinja_env.from_string(config_new_section["template"])
        h1_critic_template = jinja_env.from_string(config_h1_critic["template"])
        init_outline = {'meta': outline['meta'], sections[0]: outline[sections[0]]}

        new_outline = outline

        init_jinja_args = {
            'json_outline': init_outline,
            'function_docs': documentation
        }

        init_prompt = init_template.render(**init_jinja_args)

        # hierarchy 1 only
        outline_location = get_outline_location(outline, subsection=False)
        logs = {}
        curr_section = sections[0]

        layout_cumulative_input_token = 0
        layout_cumulative_output_token = 0

        print('Generating h1 layout...\n')
        print(f'Generating h1 layout for section {curr_section}...')
        logs[curr_section] = gen_layout(
            init_actor_agent, 
            init_prompt, 
            args.max_retry, 
            name_to_hierarchy, 
            visual_identifier=curr_section
        )

        if logs[curr_section][-1]['error'] is not None:
            raise ValueError(f'Failed to generate layout for section {curr_section}.')
        
        layout_cumulative_input_token += logs[curr_section][-1]['cumulative_tokens'][0]
        layout_cumulative_output_token += logs[curr_section][-1]['cumulative_tokens'][1]

        for section_index in range(1, len(sections)):
            curr_section = sections[section_index]
            print(f'generating h1 layout for section {curr_section}...')
            new_section_outline = {curr_section: new_outline[curr_section]}
            new_section_jinja_args = {
                'json_outline': new_section_outline,
                'function_docs': documentation
            }
            new_section_prompt = new_section_template.render(**new_section_jinja_args)

            logs[curr_section] = gen_layout(
                new_section_actor_agent, 
                new_section_prompt, 
                args.max_retry, 
                name_to_hierarchy, 
                visual_identifier=curr_section,
                existing_code = logs[sections[section_index - 1]][-1]['concatenated_code']
            )
            if logs[curr_section][-1]['error'] is not None:
                raise ValueError(f'Failed to generate layout for section {curr_section}.')
            
            layout_cumulative_input_token += logs[curr_section][-1]['cumulative_tokens'][0]
            layout_cumulative_output_token += logs[curr_section][-1]['cumulative_tokens'][1]

        consumption_log['h1_actor'].append((layout_cumulative_input_token, layout_cumulative_output_token))
        total_input_token += layout_cumulative_input_token
        total_output_token += layout_cumulative_output_token

        h1_path = f'tmp/poster_<{sections[-1]}>_hierarchy_1.pptx'
        h2_path = f'tmp/poster_<{sections[-1]}>_hierarchy_2.pptx'

        h1_filled_path = f'tmp/poster_<{sections[-1]}>_hierarchy_1_filled.pptx'
        h2_filled_path = f'tmp/poster_<{sections[-1]}>_hierarchy_2_filled.pptx'

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
            print(bbox_check_result)
            attempt += 1
            continue

        h1_critic_agent.reset()
        response = h1_critic_agent.step(h1_critic_msg)
        input_token, output_token = account_token(response)
        consumption_log['h1_critic'].append((input_token, output_token))
        total_input_token += input_token
        total_output_token += output_token
        if response.msgs[0].content == 'T':
            print('Blank area detected.')
            attempt += 1
            continue

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
    
    for section in sections:
        while True:
            print(f'generating h2 for section {section}...')
            section_outline = {section: outline_no_sub_locations[section]}
            section_jinja_args = {
                'section_outline': json.dumps(section_outline, indent=4),
            }

            section_prompt = h2_actor_template.render(**section_jinja_args)

            h2_actor_agent.reset()
            response = h2_actor_agent.step(section_prompt)
            input_token, output_token = account_token(response)
            h2_cumulative_input_token += input_token
            h2_cumulative_output_token += output_token
            subsection_location = get_json_from_response(response.msgs[0].content)

            sec_bbox = outline_no_sub_locations[section]['location']
            subsection_location_dict = {}
            for k, v in subsection_location.items():
                subsection_location_dict[k] = {
                    'left': v['location'][0],
                    'top': v['location'][1],
                    'width': v['location'][2],
                    'height': v['location'][3]
                }

            is_valid, revised = validate_and_adjust_subsections(sec_bbox, subsection_location_dict)
            if not is_valid:
                is_valid, revised = validate_and_adjust_subsections(sec_bbox, revised)
                assert is_valid, "Failed to adjust subsections to fit section"
                outline_no_sub_locations = fill_location(outline_no_sub_locations, section, revised)
            else:
                outline_no_sub_locations = fill_location(outline_no_sub_locations, section, subsection_location)
            break

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
    curr_section = sections[0]

    init_outline = {'meta': new_outline['meta'], sections[0]: new_outline[sections[0]]}

    init_jinja_args = {
        'json_outline': init_outline,
        'function_docs': documentation
    }

    init_prompt = init_template.render(**init_jinja_args)
    logs[curr_section] = gen_layout(
        init_actor_agent, 
        init_prompt, 
        args.max_retry, 
        name_to_hierarchy, 
        visual_identifier=curr_section
    )

    if logs[curr_section][-1]['error'] is not None:
        raise ValueError(f'Failed to generate layout for section {curr_section}.')

    gen_layout_cumulative_input_token += logs[curr_section][-1]['cumulative_tokens'][0]
    gen_layout_cumulative_output_token += logs[curr_section][-1]['cumulative_tokens'][1]

    for section_index in range(1, len(sections)):
        curr_section = sections[section_index]
        print(f'generating section {curr_section}...')
        new_section_outline = {curr_section: new_outline[curr_section]}
        new_section_jinja_args = {
            'json_outline': new_section_outline,
            'function_docs': documentation
        }
        new_section_prompt = new_section_template.render(**new_section_jinja_args)

        logs[curr_section] = gen_layout(
            new_section_actor_agent, 
            new_section_prompt, 
            args.max_retry, 
            name_to_hierarchy, 
            visual_identifier=curr_section,
            existing_code = logs[sections[section_index - 1]][-1]['concatenated_code']
        )
        if logs[curr_section][-1]['error'] is not None:
            raise ValueError(f'Failed to generate layout for section {curr_section}.')
        
        gen_layout_cumulative_input_token += logs[curr_section][-1]['cumulative_tokens'][0]
        gen_layout_cumulative_output_token += logs[curr_section][-1]['cumulative_tokens'][1]

    consumption_log['gen_layout'].append((gen_layout_cumulative_input_token, gen_layout_cumulative_output_token))
    total_input_token += gen_layout_cumulative_input_token
    total_output_token += gen_layout_cumulative_output_token

    h1_path = f'tmp/poster_<{sections[-1]}>_hierarchy_1.pptx'
    h2_path = f'tmp/poster_<{sections[-1]}>_hierarchy_2.pptx'

    h1_filled_path = f'tmp/poster_<{sections[-1]}>_hierarchy_1_filled.pptx'
    h2_filled_path = f'tmp/poster_<{sections[-1]}>_hierarchy_2_filled.pptx'

    ppt_to_images(h1_path, f'{poster_log_path}/layout_h1')
    ppt_to_images(h2_path, f'{poster_log_path}/layout_h2')
    ppt_to_images(h1_filled_path, f'{poster_log_path}/layout_h1_filled')
    ppt_to_images(h2_filled_path, f'{poster_log_path}/layout_h2_filled')

    h1_img = Image.open(f'{poster_log_path}/layout_h1/slide_0001.jpg')
    h2_img = Image.open(f'{poster_log_path}/layout_h2/slide_0001.jpg')
    h1_filled_img = Image.open(f'{poster_log_path}/layout_h1_filled/slide_0001.jpg')
    h2_filled_img = Image.open(f'{poster_log_path}/layout_h2_filled/slide_0001.jpg')

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
    print(f'Token consumption: {input_token} -> {output_token}')

    input_token, output_token = gen_outline_layout(args, actor_config, critic_config)
    print(f'Token consumption: {input_token} -> {output_token}')