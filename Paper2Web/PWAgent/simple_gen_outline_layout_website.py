from dotenv import load_dotenv
import os
import json
import copy
import yaml
from jinja2 import Environment, StrictUndefined
import sys

# Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.src.utils import get_json_from_response
from camel.models import ModelFactory
from camel.agents import ChatAgent
from utils.wei_utils import *

import argparse
from datetime import datetime
import glob
import re

load_dotenv()

def find_project_root():
    from pathlib import Path
    
    # 首先从当前文件路径查找
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "utils" / "prompt_templates").exists():
            return parent
    
    # 如果找不到，尝试从Python路径中找
    import sys
    for path in sys.path:
        path_obj = Path(path)
        if (path_obj / "utils" / "prompt_templates").exists():
            return path_obj
    
    raise FileNotFoundError("can not find project root, please make sure utils/prompt_templates directory exists")

def save_llm_interaction_log(log_dir, stage, input_prompt, model_response, website_name):

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/{website_name}_{stage}_{timestamp}.txt"
    
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"🚀 Simple Gen Outline Layout Website - {stage}\n")
        f.write(f"⏰ TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"📄 NAME: {website_name}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("📝 THE PROMPT:\n")
        f.write("-" * 60 + "\n")
        f.write(input_prompt + "\n\n")
        
        f.write("🤖 THE RETURN RESULT:\n")
        f.write("-" * 60 + "\n")
        f.write(model_response + "\n\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"  - INPUT: {len(input_prompt)} TOKENS\n")
        f.write(f"  - OUTPUT: {len(model_response)} TOKENS\n")
    

    return log_filename

def find_files_smart(website_name, file_type="original"):

    if file_type == "original":
        # 查找原始文件
        images_pattern = f'<*>_images_and_tables/{website_name}_images.json'
        tables_pattern = f'<*>_images_and_tables/{website_name}_tables.json'
    else:
        # 查找过滤后的文件
        images_pattern = f'<*>_images_and_tables/{website_name}_images_filtered.json'
        tables_pattern = f'<*>_images_and_tables/{website_name}_tables_filtered.json'
    
    doc_pattern1 = f'contents/<*>_{website_name}_raw_content.json'
    doc_pattern2 = f'<*>_{website_name}_raw_content.json'
    
    # 查找文件
    images_files = glob.glob(images_pattern)
    tables_files = glob.glob(tables_pattern)
    doc_files = glob.glob(doc_pattern1) + glob.glob(doc_pattern2)
    
    if not images_files:
        raise FileNotFoundError(f"{images_pattern}")
    if not tables_files:
        raise FileNotFoundError(f"{tables_pattern}")
    if not doc_files:
        raise FileNotFoundError(f" {doc_pattern1} OR {doc_pattern2}")
    
    # 返回找到的第一个匹配文件
    return images_files[0], tables_files[0], doc_files[0]

def filter_image_table(args, filter_config):



    try:
        images_path, tables_path, doc_path = find_files_smart(args.website_name)
        

        
        # 从实际路径中提取目录，用于保存过滤后的文件
        images_dir = os.path.dirname(images_path)
        
        images = json.load(open(images_path, 'r'))
        tables = json.load(open(tables_path, 'r'))
        doc_json = json.load(open(doc_path, 'r'))

        
    except FileNotFoundError as e:
        print(f"❌ {e}")

        raise
    
    # 
    agent_filter = 'website_image_table_filter_agent'
    # 使用绝对路径访问提示词文件
    project_root = find_project_root()
    template_path = project_root / "utils" / "prompt_templates" / f"{agent_filter}.yaml"
    with open(template_path, "r") as f:
        config_filter = yaml.safe_load(f)

    # 
    image_information = {}
    for k, v in images.items():
        image_information[k] = copy.deepcopy(v)
        # 

    table_information = {}
    for k, v in tables.items():
        table_information[k] = copy.deepcopy(v)
        # 

    filter_actor_sys_msg = config_filter['system_prompt']

    if args.model_name_t.startswith('vllm_qwen'):
        filter_model = ModelFactory.create(
            model_platform=filter_config['model_platform'],
            model_type=filter_config['model_type'],
            model_config_dict=filter_config['model_config'],
            url=filter_config['url'],
        )
    elif args.model_name_t.startswith('openrouter'):
        # 直接使用 OpenRouterModel 处理 OpenRouter 模型
        from camel.models.openrouter_model import OpenRouterModel
        filter_model = OpenRouterModel(
            model_type=filter_config['model_type'],
            model_config_dict=filter_config['model_config'],
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
    
    # 渲染完整的输入提示词
    rendered_prompt = filter_prompt.render(**filter_jinja_args)
    
    response = filter_actor_agent.step(rendered_prompt)
    input_token, output_token = account_token(response)
    
    # 保存大模型交互日志
    model_response = response.msgs[0].content
    save_llm_interaction_log("simple_gen_logs", "filter_image_table", rendered_prompt, model_response, args.website_name)
    
    response_json = get_json_from_response(model_response)
    

    if isinstance(response_json, dict):
        print(f"   - response_json: {list(response_json.keys())}")
    else:


        raise ValueError(f"{type(response_json)}")
    
    # 
    table_information = response_json['table_information']
    image_information = response_json['image_information']
    
    # 💾 保存过滤后的文件到实际找到的目录
    filtered_images_path = os.path.join(images_dir, f'{args.website_name}_images_filtered.json')
    filtered_tables_path = os.path.join(images_dir, f'{args.website_name}_tables_filtered.json')
    
    json.dump(image_information, open(filtered_images_path, 'w'), indent=4)
    json.dump(table_information, open(filtered_tables_path, 'w'), indent=4)
    

    return input_token, output_token

def gen_outline_layout_website_simple(args, actor_config):

    total_input_token, total_output_token = 0, 0
    


    try:
        filtered_images_path, filtered_tables_path, doc_path = find_files_smart(args.website_name, "filtered")
        

        
        doc_json = json.load(open(doc_path, 'r'))
        filtered_table_information = json.load(open(filtered_tables_path, 'r'))
        filtered_image_information = json.load(open(filtered_images_path, 'r'))
        
    except FileNotFoundError as e:

        raise

    # 
    agent_name = 'website_planner_agent'
    
    # LLM
    filtered_table_information_captions = {}
    filtered_image_information_captions = {}

    # 
    print(f"Debug: filtered_table_information type: {type(filtered_table_information)}")
    print(f"Debug: filtered_table_information content: {filtered_table_information}")
    print(f"Debug: filtered_image_information type: {type(filtered_image_information)}")
    print(f"Debug: filtered_image_information content: {filtered_image_information}")

    # 
    if isinstance(filtered_table_information, dict):
        for k, v in filtered_table_information.items():
            if isinstance(v, dict) and 'caption' in v:
                filtered_table_information_captions[k] = {v['caption']}
            else:
                print(f"Warning: table {k} has unexpected format: {v}")
    elif isinstance(filtered_table_information, list):
        # 
        print("Converting filtered_table_information from list to dict format")
        temp_dict = {}
        for i, item in enumerate(filtered_table_information):
            if isinstance(item, dict) and 'caption' in item:
                temp_dict[str(i)] = item
        filtered_table_information = temp_dict
        filtered_table_information_captions = {k: {v['caption']} for k, v in filtered_table_information.items()}
    else:
        print(f"Warning: filtered_table_information is not a dict or list: {filtered_table_information}")
        filtered_table_information_captions = {}

    # 
    if isinstance(filtered_image_information, dict):
        for k, v in filtered_image_information.items():
            if isinstance(v, dict) and 'caption' in v:
                filtered_image_information_captions[k] = {v['caption']}
            else:
                print(f"Warning: image {k} has unexpected format: {v}")
    elif isinstance(filtered_image_information, list):
        # 
        print("Converting filtered_image_information from list to dict format")
        temp_dict = {}
        for i, item in enumerate(filtered_image_information):
            if isinstance(item, dict) and 'caption' in item:
                temp_dict[str(i)] = item
        filtered_image_information = temp_dict
        filtered_image_information_captions = {k: {v['caption']} for k, v in filtered_image_information.items()}
    else:
        print(f"Warning: filtered_image_information is not a dict or list: {filtered_image_information}")
        filtered_image_information_captions = {}

    # 使用帮助函数获取项目根目录
    project_root = find_project_root()
    template_path = project_root / "utils" / "prompt_templates" / f"{agent_name}.yaml"
    with open(template_path, "r") as f:
        planner_config = yaml.safe_load(f)

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
    elif args.model_name_t.startswith('openrouter'):
        # 直接使用 OpenRouterModel 处理 OpenRouter 模型
        from camel.models.openrouter_model import OpenRouterModel
        planner_model = OpenRouterModel(
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
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

    print(f'Generating website outline...')
    planner_prompt = outline_template.render(**planner_jinja_args)
    planner_agent.reset()
    
    response = planner_agent.step(planner_prompt)
    input_token, output_token = account_token(response)
    total_input_token += input_token
    total_output_token += output_token

    # 保存大模型交互日志
    model_response = response.msgs[0].content
    save_llm_interaction_log("simple_gen_logs", "gen_outline_layout", planner_prompt, model_response, args.website_name)

    figure_arrangement = get_json_from_response(model_response)

    print(f'Figure arrangement: {json.dumps(figure_arrangement, indent=4)}')

    # /
    arranged_images = {}
    arranged_tables = {}
    assigned_images = set()
    assigned_tables = set()
    
    print(f" ...")
    print(f"   - : {len(filtered_image_information)}")
    print(f"   - : {len(filtered_table_information)}")
    
    for section_name, figure in figure_arrangement.items():
        print(f"   - : {section_name}")
        
        # 
        if 'images' in figure:
            for image_id in figure['images']:
                image_id_str = str(image_id)
                print(f"     - : {image_id_str}")
                if image_id_str in assigned_images:
                    print(f"         {image_id_str} ")
                    continue
                if image_id_str in filtered_image_information:
                    arranged_images[image_id_str] = filtered_image_information[image_id_str]
                    assigned_images.add(image_id_str)
                    print(f"         {image_id_str}")
                else:
                    print(f"         {image_id_str} ")
        elif 'image' in figure:
            # 
            image_id = str(figure['image'])
            print(f"     - : {image_id}")
            if image_id in assigned_images:
                print(f"         {image_id} ")
                continue
            if image_id in filtered_image_information:
                arranged_images[image_id] = filtered_image_information[image_id]
                assigned_images.add(image_id)
                print(f"         {image_id}")
            else:
                print(f"         {image_id} ")
        
        # 
        if 'tables' in figure:
            for table_id in figure['tables']:
                table_id_str = str(table_id)
                print(f"     - : {table_id_str}")
                if table_id_str in assigned_tables:
                    print(f"         {table_id_str} ")
                    continue
                if table_id_str in filtered_table_information:
                    arranged_tables[table_id_str] = filtered_table_information[table_id_str]
                    assigned_tables.add(table_id_str)
                    print(f"         {table_id_str}")
                else:
                    print(f"         {table_id_str} ")
        elif 'table' in figure:
            # 
            table_id = str(figure['table'])
            print(f"     - : {table_id}")
            if table_id in assigned_tables:
                print(f"         {table_id} ")
                continue
            if table_id in filtered_table_information:
                arranged_tables[table_id] = filtered_table_information[table_id]
                assigned_tables.add(table_id)
                print(f"         {table_id}")
            else:
                print(f"         {table_id} ")
    
    print(f" :")
    print(f"   - : {len(arranged_images)}")
    print(f"   - : {len(arranged_tables)}")
    
    # 
    website_pages = []
    for i in range(len(doc_json['sections'])):
        section = doc_json['sections'][i]
        page = {}
        page['page_id'] = i
        page['section_name'] = section['title']
        page['content'] = section['content']  # 
        page['text_len'] = len(section['content'])  # 
        
        # 
        if section['title'] in figure_arrangement:
            curr_arrangement = figure_arrangement[section['title']]
            
            # 
            if 'tables' in curr_arrangement:
                page['tables'] = curr_arrangement['tables']
            elif 'table' in curr_arrangement:
                page['table'] = curr_arrangement['table']
            
            # 
            if 'images' in curr_arrangement:
                page['images'] = curr_arrangement['images']
            elif 'image' in curr_arrangement:
                page['image'] = curr_arrangement['image']

        website_pages.append(page)

    # 
    website_outline = {
        'meta': {
            'project_name': args.website_name,
            'website_title': doc_json['meta'].get('website_title', ''),
            'authors': doc_json['meta'].get('authors', ''),
            'affiliations': doc_json['meta'].get('affiliations', ''),
            'total_pages': len(website_pages),
            'generated_by': 'simple_gen_outline_layout_website'
        },
        'pages': website_pages,
        'figure_arrangement': figure_arrangement,
        'arranged_images': arranged_images,
        'arranged_tables': arranged_tables
    }
    
    os.makedirs('website_outlines', exist_ok=True)
    json.dump(website_outline, open(f'website_outlines/<{args.model_name_t}_{args.model_name_v}>_{args.website_name}_website_outline.json', 'w'), indent=4)

    return total_input_token, total_output_token, website_pages, figure_arrangement

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple Website Outline Generation Pipeline (No TP/GP)')
    parser.add_argument('--website_name', type=str, default=None, help='')
    parser.add_argument('--model_name_t', type=str, default='openrouter_qwen_7b', help='')
    parser.add_argument('--model_name_v', type=str, default='v1', help='')
    parser.add_argument('--paper_path', type=str, required=True, help='PDF')
    parser.add_argument('--index', type=int, default=0, help='')
    args = parser.parse_args()

    # 
    args.model_name_v = 'v1'

    actor_config = get_agent_config(args.model_name_t)

    if args.website_name is None:
        args.website_name = args.paper_path.split('/')[-1].replace('.pdf', '').replace(' ', '_')

    print(f'Processing simple website outline for: {args.website_name}')
    
    # 
    print('Step 1 Filtering images and tables')
    input_token, output_token = filter_image_table(args, actor_config)
    print(f'Filtering token consumption: {input_token} -> {output_token}')

    # 
    print('Step 2 Generating simple website outline and layout')
    input_token, output_token, website_pages, figure_arrangement = gen_outline_layout_website_simple(args, actor_config)
    print(f'Outline generation token consumption: {input_token} -> {output_token}')
    
    print(f'Simple website outline generated successfully!')
    print(f'Total pages planned: {len(website_pages)}')
    print(f'Total images arranged: {len(figure_arrangement)}')
    print(f'Note: No TP/GP calculations performed - simplified approach')
