from dotenv import load_dotenv
import os
import json
import yaml
import sys

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.src.utils import get_json_from_response
from camel.models import ModelFactory
from camel.agents import ChatAgent
from utils.wei_utils import *

import argparse
import glob
from jinja2 import Environment, StrictUndefined

load_dotenv()

def find_project_root():
    from pathlib import Path
    
    # First search from current file path
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "utils" / "prompt_templates").exists():
            return parent
    
    # If not found, try searching from Python paths
    import sys
    for path in sys.path:
        path_obj = Path(path)
        if (path_obj / "utils" / "prompt_templates").exists():
            return path_obj
    
    raise FileNotFoundError("Cannot find project root directory, please ensure utils/prompt_templates directory exists")

def find_content_file_smart(website_name):
    """Intelligently find research content files, ignoring model name differences"""
    content_pattern1 = f'contents/<*>_{website_name}_raw_content.json'
    content_pattern2 = f'<*>_{website_name}_raw_content.json'
    
    content_files = glob.glob(content_pattern1) + glob.glob(content_pattern2)
    
    if not content_files:
        raise FileNotFoundError(f"Research content file not found, search patterns: {content_pattern1} or {content_pattern2}")
    
    return content_files[0]

def extract_important_info(args, actor_config):
    # Intelligently find research content files
    print(f"Intelligently searching for research content files...")
    try:
        raw_content_path = find_content_file_smart(args.website_name)
        print(f"Successfully found file: {raw_content_path}")
    except FileNotFoundError as e:
        print(f"File search failed: {e}")
        print("Please ensure you have run the first stage parse_raw.py")
        return None, None, None  # Fix: return 3 values
    
    with open(raw_content_path, 'r', encoding='utf-8') as f:
        raw_content = json.load(f)
    
    print(f"Loading content from: {raw_content_path}")
    print(f"Found {len(raw_content['sections'])} sections")
    
    # Load prompt template
    agent_name = 'extract_important_info_agent'
    # Use absolute path to access prompt template file
    project_root = find_project_root()
    template_path = project_root / "utils" / "prompt_templates" / f"{agent_name}.yaml"
    with open(template_path, "r") as f:
        extract_config = yaml.safe_load(f)
    
    # Prepare input data
    input_data = {
        'paper_title': raw_content['meta']['website_title'],
        'authors': raw_content['meta']['authors'],
        'affiliations': raw_content['meta']['affiliations'],
        'sections': raw_content['sections']
    }
    
    # Initialize LLM model
    if args.model_name_t.startswith('vllm_qwen'):
        extract_model = ModelFactory.create(
            model_platform=actor_config['model_platform'],
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
            url=actor_config['url'],
        )
    elif args.model_name_t.startswith('openrouter'):
        # Directly use OpenRouterModel for OpenRouter models
        from camel.models.openrouter_model import OpenRouterModel
        extract_model = OpenRouterModel(
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
        )
    else:
        extract_model = ModelFactory.create(
            model_platform=actor_config['model_platform'],
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
        )
    
    # Create extraction agent
    extract_agent = ChatAgent(
        system_message=extract_config['system_prompt'],
        model=extract_model,
        message_window_size=10,
    )
    
    # Generate extraction prompt
    jinja_env = Environment(undefined=StrictUndefined)
    extract_template = jinja_env.from_string(extract_config["template"])
    extract_jinja_args = {
        'input_data': input_data
    }
    
    print(f'Generating extraction prompt...')
    extract_prompt = extract_template.render(**extract_jinja_args)
    extract_agent.reset()
    response = extract_agent.step(extract_prompt)
    input_token, output_token = account_token(response)
    
    # Parse LLM response
    important_info = get_json_from_response(response.msgs[0].content)
    
    print(f'Information extraction completed!')
    print(f'Extracted information types: {list(important_info.keys())}')
    
    # Save results
    output_path = f'contents/<{args.model_name_t}_{args.model_name_v}>_{args.website_name}_important_info.json'
    os.makedirs('contents', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(important_info, f, indent=4, ensure_ascii=False)
    
    print(f'Results saved to: {output_path}')
    
    return input_token, output_token, important_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Important Information from Paper')
    parser.add_argument('--website_name', type=str, default=None, help='Website name (defaults to paper filename)')
    parser.add_argument('--model_name_t', type=str, default='openrouter_qwen_7b', help='Text model name for extraction')
    parser.add_argument('--model_name_v', type=str, default='v1', help='Version identifier')
    parser.add_argument('--paper_path', type=str, required=True, help='PDF paper file path')
    args = parser.parse_args()
    
    # Set default website name if not provided
    if args.website_name is None:
        args.website_name = args.paper_path.split('/')[-1].replace('.pdf', '').replace(' ', '_')
    
    print(f'Website name: {args.website_name}')
    
    # Get agent configuration
    actor_config = get_agent_config(args.model_name_t)
    
    # Extract important information
    input_token, output_token, important_info = extract_important_info(args, actor_config)
    
    if important_info:
        print(f'Extraction completed successfully!')
        print(f'Token usage: {input_token} -> {output_token}')
        print(f'Extracted information:')
        for info_type, content in important_info.items():
            if isinstance(content, list):
                print(f'   - {info_type}: {len(content)} items')
            else:
                print(f'   - {info_type}: {type(content).__name__}')
    else:
        print(f'Extraction failed')
