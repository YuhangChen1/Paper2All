#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple End-to-End Website Generator v1
Use LLM to generate complete single-file HTML websites
"""

import os
import sys
import json
import argparse
import shutil
import glob
import random
from pathlib import Path
from jinja2 import Environment, StrictUndefined
import yaml

# Add Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def find_project_root():
    """Find project root directory (directory containing utils folder)"""
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
    
    raise FileNotFoundError("Unable to find project root directory, please ensure utils/prompt_templates directory exists")

from camel.agents import ChatAgent
from camel.configs import OpenRouterConfig
from camel.types import ModelType, ModelPlatformType
from camel.models import ModelFactory


def load_random_html_template():
    """
    Randomly select an HTML template file from utils/template directory and return its content
    """
    # Get project root directory
    project_root = find_project_root()
    template_dir = project_root / 'utils' / 'template'
    
    # Check if template directory exists
    if not template_dir.exists():
        raise FileNotFoundError(f"Template directory does not exist: {template_dir}")
    
    # Find all HTML files
    html_files = list(template_dir.glob('*.html'))
    
    if not html_files:
        raise FileNotFoundError(f"No HTML files found in template directory: {template_dir}")
    
    # Randomly select a template file
    selected_template = random.choice(html_files)
    print(f'Randomly selected HTML template: {selected_template.name}')
    
    # Read template content
    with open(selected_template, 'r', encoding='utf-8') as f:
        template_content = f.read()
    
    if not template_content.strip():
        raise ValueError(f"Selected template file is empty: {selected_template.name}")
    
    return template_content


def find_files_smart_end_to_end(website_name):
    """
    Intelligently find files required for end-to-end generation, ignoring model name differences
    """
    # Find research content file
    content_pattern1 = f'contents/<*>_{website_name}_raw_content.json'
    content_pattern2 = f'<*>_{website_name}_raw_content.json'
    
    # Find website outline file
    outline_pattern = f'website_outlines/<*>_{website_name}_website_outline.json'
    
    # Find important info file
    important_pattern1 = f'contents/<*>_{website_name}_important_info.json'
    important_pattern2 = f'<*>_{website_name}_important_info.json'
    
    # Search for files
    content_files = glob.glob(content_pattern1) + glob.glob(content_pattern2)
    outline_files = glob.glob(outline_pattern)
    important_files = glob.glob(important_pattern1) + glob.glob(important_pattern2)
    
    if not content_files:
        raise FileNotFoundError(f"Research content file not found, search patterns: {content_pattern1} or {content_pattern2}")
    if not outline_files:
        raise FileNotFoundError(f"Website outline file not found, search pattern: {outline_pattern}")
    if not important_files:
        raise FileNotFoundError(f"Important info file not found, search patterns: {important_pattern1} or {important_pattern2}")
    
    return content_files[0], outline_files[0], important_files[0]

def load_yaml_config(file_path):
    """Load YAML configuration"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load YAML config: {e}")
        return None


def load_json_data(file_path):
    """Load JSON data"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load JSON: {e}")
        return None


def get_agent_config(model_name):
    """Get agent configuration"""
    print(f"Loading model configuration: {model_name}")
    
    if model_name == 'openrouter_qwen3_14b':
        config = {
            'model_type': 'qwen/qwen3-14b',
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    elif model_name == 'openrouter_qwen2_5_vl_32b':
        config = {
            'model_type': 'qwen/qwen2.5-vl-32b-instruct:free',
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"Model configuration:")
    print(f"   - model_type: {config['model_type']}")
    print(f"   - model_platform: {config['model_platform']}")
    print(f"   - model_config: {config['model_config']}")
    return config


def create_generator_agent(model_name):
    """Create generator agent"""
    try:
        print(f"Initializing generator agent...")
        # Use unified configuration, directly call function from utils.wei_utils
        from utils.wei_utils import get_agent_config as get_unified_agent_config
        agent_config = get_unified_agent_config(model_name)
        print(f"Agent configuration: {agent_config}")
        
        # Set OpenRouter API key
        if 'OPENROUTER_API_KEY' in os.environ:
            os.environ['OPENAI_API_KEY'] = os.environ['OPENROUTER_API_KEY']
            print(f"Set OPENAI_API_KEY from OPENROUTER_API_KEY")
        
        # Bypass ModelFactory compatibility issues, directly use OpenRouterModel
        print(f"Creating model instance...")
        if model_name.startswith('openrouter'):
            from camel.models.openrouter_model import OpenRouterModel
            model = OpenRouterModel(
                model_type=agent_config['model_type'],
                model_config_dict=agent_config['model_config'],
                api_key=os.environ.get('OPENROUTER_API_KEY')
            )
        elif model_name.startswith('vllm_'):
            model = ModelFactory.create(
                model_platform=agent_config['model_platform'],
                model_type=agent_config['model_type'],
                model_config_dict=agent_config['model_config'],
                url=agent_config.get('url'),
            )
        else:
            model = ModelFactory.create(
                model_platform=agent_config['model_platform'],
                model_type=agent_config['model_type'],
                model_config_dict=agent_config['model_config'],
            )
        print(f"Model instance created: {type(model)}")
        
        # Create ChatAgent
        print(f"Creating ChatAgent...")
        agent = ChatAgent(
            model=model,
            system_message="You are an expert web developer and UI/UX designer specializing in creating beautiful, modern, and interactive academic project websites."
        )
        print(f"ChatAgent created: {type(agent)}")
        return agent
        
    except Exception as e:
        print(f"Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        return None


def account_token(response):
    """Account tokens"""
    try:
        input_tokens = response.input_tokens or 0
        output_tokens = response.output_tokens or 0
        return input_tokens, output_tokens
    except:
        return 0, 0


def generate_website_end_to_end(args, research_content, visual_assets, important_info):
    """
    Generate website end-to-end using LLM
    """
    print(f'Starting end-to-end website generation...')
    
    # Use absolute path to access template file - use v0 version to support multiple templates
    project_root = find_project_root()
    generator_config_path = project_root / 'utils' / 'prompt_templates' / 'simple_end_to_end_website_generator_v0.yaml'
    generator_config = load_yaml_config(str(generator_config_path))
    if not generator_config:
        raise Exception("Unable to load generator configuration file")
    
    # Randomly load HTML template
    print('Loading random HTML template...')
    html_template = load_random_html_template()
    
    # Create generator agent
    generator_agent = create_generator_agent(args.model_name_t)
    if not generator_agent:
        raise Exception("Unable to create generator agent")
    
    # Configure Jinja template engine and parameters
    jinja_env = Environment(undefined=StrictUndefined)
    generator_template = jinja_env.from_string(generator_config["template"])
    generator_jinja_args = {
        'research_content': research_content,
        'visual_assets': visual_assets,
        'important_info': important_info,
        'html_template': html_template  # Add randomly selected HTML template
    }
    
    generator_prompt = generator_template.render(**generator_jinja_args)
    
    # Maximum 5 retries
    max_retries = 5
    retry_count = 0
    website_code = None
    total_input_token = 0
    total_output_token = 0
    
    # Pre-define log file path
    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)
    log_file = f'{log_dir}/{args.website_name}_simple_end_to_end_llm_response.txt'
    
    while retry_count < max_retries:
        try:
            # Call LLM
            if retry_count > 0:
                print(f'Retry attempt {retry_count + 1}...')
            
            generator_agent.reset()
            response = generator_agent.step(generator_prompt)
            
            # Check if response is valid
            if response is None:
                raise Exception("API call returned None response")
            if not hasattr(response, 'msgs') or not response.msgs:
                raise Exception("API response has no msgs field or msgs is empty")
            if response.msgs[0] is None:
                raise Exception("First message in API response is None")
                
            content = response.msgs[0].content
            
            if content is None:
                raise Exception("API response content is None")
            
            # Account tokens
            current_input_token, current_output_token = account_token(response)
            total_input_token += current_input_token
            total_output_token += current_output_token
    
            # Save LLM response
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== LLM Response (Attempt {retry_count + 1}) ===\n")
                f.write(content)
                f.write("\n\n=== End of Response ===\n")
            
            print(f'LLM response received successfully! (Attempt {retry_count + 1})')
            print(f'LLM response saved to: {log_file}')
            
            # Parse JSON
            import re
            
            # Extract and parse JSON
            try:
                # Try to extract JSON from code block
                json_match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1)
                    website_code = json.loads(json_content)
                else:
                    # Try to parse content directly as JSON
                    website_code = json.loads(content)
                
                # Validate required keys exist - now only need HTML file
                required_keys = ['index.html']
                missing_keys = [key for key in required_keys if key not in website_code or not website_code[key].strip()]
                
                if missing_keys:
                    print(f'Generated code is missing required files: {missing_keys}')
                    raise Exception(f"Generated website code is missing required files: {missing_keys}")
                
                print(f'Successfully parsed JSON format, contains single-file HTML content')
                break  # Parsing successful and contains all required files, exit retry loop
            except json.JSONDecodeError as e:
                print(f'JSON parsing failed (Attempt {retry_count + 1}): {e}')
                retry_count += 1
                if retry_count >= max_retries:
                    print(f'Failed to parse JSON after {max_retries} attempts')
                    print(f'Response preview: {content[:200]}...')
                    # Log failure details
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n\n=== JSON Parsing Failed After {max_retries} Attempts ===\n")
                        f.write(f"Error: {e}\n")
                        f.write(f"Full response:\n{content}")
                    raise Exception(f"LLM failed to generate valid JSON after {max_retries} attempts, see log: {log_file}")
                continue
                
        except Exception as e:
            print(f'LLM call failed (Attempt {retry_count + 1}): {e}')
            retry_count += 1
            if retry_count >= max_retries:
                print(f'Failed to call LLM after {max_retries} attempts')
                # Log error
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n\n=== LLM Call Failed After {max_retries} Attempts ===\n")
                    f.write(f"Error: {e}\n")
                raise Exception(f"LLM call failed after {max_retries} attempts: {e}")
            continue
    
    # Log single HTML file generation result
    with open(log_file, 'a', encoding='utf-8') as f:
        html_content = website_code.get('index.html', '')
        f.write(f"Single-file HTML total length: {len(html_content)} characters\n")
        f.write(f"\n=== Complete HTML File Content ===\n")
        f.write(html_content)
    
    print(f'Generation result statistics:')
    html_content = website_code.get('index.html', '')
    print(f'   Complete HTML file: {len(html_content)} characters')
    
    # Analyze CSS and JS content in HTML file
    css_count = html_content.count('<style>') + html_content.count('<style ')
    js_count = html_content.count('<script>') + html_content.count('<script ')
    print(f'   Embedded CSS blocks: {css_count}')
    print(f'   Embedded JS blocks: {js_count}')
    
    # Validate HTML file content is sufficient
    html_content = website_code.get('index.html', '')
    if not html_content or len(html_content.strip()) < 1000:
        print(f'Warning: Generated HTML content is too short or empty, may have issues')
        return total_input_token, total_output_token, None
    
    # Check if HTML file contains necessary tag structure
    if not ('<html' in html_content and '</html>' in html_content):
        print(f'Warning: Generated HTML lacks basic structure tags')
        return total_input_token, total_output_token, None
    
    return total_input_token, total_output_token, website_code


def save_website_files(args, website_code):
    """
    Save single HTML file (with embedded CSS and JavaScript)
    """
    print("Saving single-file HTML website...")
    
    # Create output directory
    output_dir = f'generated_website_{args.website_name}_simple'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save single HTML file (containing all CSS and JS content)
    html_file = os.path.join(output_dir, 'index.html')
    html_content = website_code.get('index.html', '')
    
    if not html_content:
        print(f'Error: Missing index.html content')
        raise Exception("Generated website code is missing index.html content")
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f'Successfully saved single-file HTML: {html_file}')
    print(f'   File size: {len(html_content):,} characters')
    
    # Analyze HTML file structure statistics
    style_tags = html_content.count('<style>') + html_content.count('<style ')
    script_tags = html_content.count('<script>') + html_content.count('<script ')
    print(f'   Embedded CSS blocks: {style_tags} blocks')
    print(f'   Embedded JS blocks: {script_tags} blocks')
    
    # Intelligently find existing image and table directories (supports directories generated by different models)
    print(f'Intelligently searching for image and table directories...')
    
    # Find all possible image directory patterns
    possible_patterns = [
        f'{args.model_name_t}_{args.model_name_v}_images_and_tables',  # Current model directory
        '*_images_and_tables',  # Wildcard pattern
    ]
    
    source_images_dir_pattern = None
    
    # Try to find existing directory
    for pattern in possible_patterns:
        if '*' in pattern:
            # Use glob to find matching directories
            import glob
            matching_dirs = glob.glob(pattern)
            if matching_dirs:
                source_images_dir_pattern = matching_dirs[0]  # Use first matching directory
                print(f'Found image directory: {source_images_dir_pattern}')
                break
        else:
            # Directly check if directory exists
            if os.path.exists(pattern):
                source_images_dir_pattern = pattern
                print(f'Found image directory: {source_images_dir_pattern}')
                break
    
    if not source_images_dir_pattern:
        raise FileNotFoundError(f"No image and table directory found, search patterns: {possible_patterns}")
    
    target_images_dir = os.path.join(output_dir, os.path.basename(source_images_dir_pattern))
    
    if os.path.exists(target_images_dir):
        print(f'Removing existing directory: {target_images_dir}')
        shutil.rmtree(target_images_dir)
    
    print(f'Copying image and table directory: {source_images_dir_pattern} -> {target_images_dir}')
    shutil.copytree(source_images_dir_pattern, target_images_dir)
    print(f'Image and table directory copied successfully: {target_images_dir}')
    
    print(f'Single-file website generation complete! Saved to: {output_dir}')
    return output_dir


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simple end-to-end website generator')
    parser.add_argument('--paper_path', required=True, help='PDF file path')
    parser.add_argument('--model_name_t', required=True, help='Text model name')
    parser.add_argument('--model_name_v', required=True, help='Vision model name')
    parser.add_argument('--website_name', required=True, help='Website name')
    
    args = parser.parse_args()
    
    try:
        print(f'Starting simple end-to-end website generation')
        print(f'Paper path: {args.paper_path}')
        print(f'Text model: {args.model_name_t}')
        print(f'Vision model: {args.model_name_v}')
        print(f'Website name: {args.website_name}')
        
        # Validate paper path
        if not os.path.exists(args.paper_path):
            raise FileNotFoundError(f"Paper file not found: {args.paper_path}")
        
        # Intelligently find files
        print(f'Intelligently searching for required files...')
        try:
            research_content_file, visual_assets_file, important_info_file = find_files_smart_end_to_end(args.website_name)
            
            print(f'Successfully found files:')
            print(f'   - Research content: {research_content_file}')
            print(f'   - Website outline: {visual_assets_file}')
            if important_info_file:
                print(f'   - Important info: {important_info_file}')
            
            # Load files
            research_content = load_json_data(research_content_file)
            visual_assets_raw = load_json_data(visual_assets_file)
            
        except FileNotFoundError as e:
            print(f'File search failed: {e}')
            print(f'Please ensure previous two stages have been completed')
            raise
        
        # Build visual_assets data structure
        visual_assets = {
            "meta": {
                "title": research_content.get("meta", {}).get("website_title", ""),
                "authors": research_content.get("meta", {}).get("authors", ""),
                "affiliations": research_content.get("meta", {}).get("affiliations", ""),
                "project_name": args.website_name
            },
            "images": [],
            "tables": []
        }
        
        # Process outline data
        if "pages" in visual_assets_raw:
            for page in visual_assets_raw["pages"]:
                if "images" in page:
                    if isinstance(page["images"], list):
                        for img_id in page["images"]:
                            # Get from arranged_images
                            if "arranged_images" in visual_assets_raw and str(img_id) in visual_assets_raw["arranged_images"]:
                                img_info = visual_assets_raw["arranged_images"][str(img_id)]
                                visual_assets["images"].append({
                                    "id": str(img_id),
                                    "src": img_info.get("image_path", ""),
                                    "alt": img_info.get("caption", ""),
                                    "web_width": img_info.get("width", 800),
                                    "web_height": img_info.get("height", 600)
                                })
                    else:
                        # Single image
                        img_id = page["images"]
                        if "arranged_images" in visual_assets_raw and str(img_id) in visual_assets_raw["arranged_images"]:
                            img_info = visual_assets_raw["arranged_images"][str(img_id)]
                            visual_assets["images"].append({
                                "id": str(img_id),
                                "src": img_info.get("image_path", ""),
                                "alt": img_info.get("caption", ""),
                                "web_width": img_info.get("width", 800),
                                "web_height": img_info.get("height", 600)
                            })
                
                if "tables" in page:
                    if isinstance(page["tables"], list):
                        for table_id in page["tables"]:
                            # Get from arranged_tables
                            if "arranged_tables" in visual_assets_raw and str(table_id) in visual_assets_raw["arranged_tables"]:
                                table_info = visual_assets_raw["arranged_tables"][str(table_id)]
                                visual_assets["tables"].append({
                                    "id": str(table_id),
                                    "src": table_info.get("table_path", ""),
                                    "alt": table_info.get("caption", "")
                                })
                    else:
                        # Single table
                        table_id = page["tables"]
                        if "arranged_tables" in visual_assets_raw and str(table_id) in visual_assets_raw["arranged_tables"]:
                            table_info = visual_assets_raw["arranged_tables"][str(table_id)]
                            visual_assets["tables"].append({
                                "id": str(table_id),
                                "src": table_info.get("table_path", ""),
                                "alt": table_info.get("caption", "")
                            })
        
        # Load important info file (must exist)
        if not important_info_file:
            raise FileNotFoundError(f"Important info file path is empty")
        if not os.path.exists(important_info_file):
            raise FileNotFoundError(f"Important info file does not exist: {important_info_file}")
        
        important_info = load_json_data(important_info_file)
        if not important_info:
            raise Exception(f"Unable to load important info file: {important_info_file}")
        
        print(f'Data loaded successfully:')
        print(f'   - Content sections: {len(research_content.get("sections", []))} sections')
        print(f'   - Visual assets: {len(visual_assets.get("images", []))} images, {len(visual_assets.get("tables", []))} tables')
        print(f'   - Important info: {len(important_info.get("important_info", []))} items/URLs')
        
        # Generate website
        input_token, output_token, website_code = generate_website_end_to_end(
            args, research_content, visual_assets, important_info
        )
        
        if website_code is None:
            print(f'Website generation failed')
            return
        
        # Save website files
        output_dir = save_website_files(args, website_code)
        
        print(f'Single-file website generated successfully!')
        print(f'Token usage: {input_token:,} -> {output_token:,}')
        print(f'Output directory: {output_dir}')
        print(f'Please open index.html to view the single-file website!')
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
