from dotenv import load_dotenv
import sys
import os
# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.src.utils import get_json_from_response
from utils.src.model_utils import parse_pdf
import json
import random

from camel.models import ModelFactory
from camel.agents import ChatAgent
from tenacity import retry, stop_after_attempt
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from pathlib import Path

import PIL

from marker.models import create_model_dict

from utils.wei_utils import *

from utils.pptx_utils import *
from utils.critic_utils import *
import torch
from jinja2 import Template
import re
import argparse

load_dotenv()
IMAGE_RESOLUTION_SCALE = 5.0

pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

@retry(stop=stop_after_attempt(5))
def parse_raw(args, actor_config, version=2):
    """
    Parse raw PDF document and generate JSON format data for website content
    
    This function first attempts to parse PDF document using docling, and falls back to marker parser if the parsing result is too short.
    Then uses LLM model to generate structured website content based on parsed document content, including metadata and section content.
    Finally validates the generated content and saves it as JSON file.
    
    Args:
        args: Command line arguments object, containing attributes like website_path (website path) and website_name (website name)
        actor_config (dict): LLM model configuration information, including model platform, type, configuration dictionary and optional URL
        version (int, optional): Prompt template version to use, defaults to 2
        
    Returns:
        tuple: Tuple containing the following three elements:
            - input_token (int): Number of tokens input to the model
            - output_token (int): Number of tokens output by the model
            - raw_result: Raw result object from docling document conversion
    """
    raw_source = args.website_path
    markdown_clean_pattern = re.compile(r"<!--[\s\S]*?-->")

    raw_result = doc_converter.convert(raw_source)

    raw_markdown = raw_result.document.export_to_markdown()
    text_content = markdown_clean_pattern.sub("", raw_markdown)

    # If docling parsing result is too short, fall back to marker parser
    if len(text_content) < 500:
        print('\nParsing with docling failed, using marker instead\n')
        parser_model = create_model_dict(device='cuda', dtype=torch.float16)
        text_content, rendered = parse_pdf(raw_source, model_lst=parser_model, save_file=False)

    # Select corresponding prompt template based on version
    if version == 1:
        template = Template(open("utils/prompts/gen_website_raw_content.txt").read())
    elif version == 2:
        template = Template(open("utils/prompts/gen_website_raw_content_v2_enhanced.txt").read())

    # Create corresponding model instance based on model name
    if args.model_name_t.startswith('vllm_qwen'):
        actor_model = ModelFactory.create(
            model_platform=actor_config['model_platform'],
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
            url=actor_config['url'],
        )
    elif args.model_name_t.startswith('openrouter'):
        # Directly use OpenRouterModel for OpenRouter models
        from camel.models.openrouter_model import OpenRouterModel
        actor_model = OpenRouterModel(
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
        )
    else:
        actor_model = ModelFactory.create(
            model_platform=actor_config['model_platform'],
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
        )

    actor_sys_msg = 'You are the author of the paper, and you will create a website for the paper.'

    actor_agent = ChatAgent(
        system_message=actor_sys_msg,
        model=actor_model,
        message_window_size=10,
        token_limit=actor_config.get('token_limit', None)
    )

    # Create log directory
    os.makedirs('log', exist_ok=True)
    
    # Loop requesting LLM to generate content until non-empty response is obtained
    while True:
        prompt = template.render(
            markdown_document=text_content, 
        )
        actor_agent.reset()
        response = actor_agent.step(prompt)
        input_token, output_token = account_token(response)

        # Save LLM's raw response to log file
        raw_response = response.msgs[0].content
        log_filename = f'log/{args.website_name}_llm_response_{len(os.listdir("log")) + 1}.txt'
        
        with open(log_filename, 'w', encoding='utf-8') as f:
            f.write("=== LLM PROMPT ===\n")
            f.write(prompt)
            f.write("\n\n=== LLM RESPONSE ===\n")
            f.write(raw_response)
            f.write("\n\n=== RESPONSE LENGTH ===\n")
            f.write(f"Response length: {len(raw_response)} characters\n")
        
        print(f'LLM response saved to: {log_filename}')
        print(f'Response length: {len(raw_response)} characters')
        print(f'Response content preview:')
        print("=" * 80)
        print(raw_response[:1000])
        if len(raw_response) > 1000:
            print("... (content too long, only showing first 1000 characters)")
        print("=" * 80)

        content_json = get_json_from_response(response.msgs[0].content)

        # Add JSON parsing debug information
        print(f"\nJSON parsing result:")
        print(f"Parsed content type: {type(content_json)}")
        print(f"Parsed content length: {len(content_json) if isinstance(content_json, dict) else 'N/A'}")
        if isinstance(content_json, dict):
            print(f"Top-level keys: {list(content_json.keys())}")
            if 'sections' in content_json:
                print(f"Number of sections: {len(content_json['sections'])}")
        print("=" * 80)

        if len(content_json) > 0:
            break
        print('Error: Empty response, retrying...')
        if args.model_name_t.startswith('vllm_qwen'):
            text_content = text_content[:80000]

    # If number of sections exceeds 9, randomly select some sections to control content volume
    if len(content_json['sections']) > 9:
        # First 2 sections + randomly select 5 sections + last 2 sections
        selected_sections = content_json['sections'][:2] + random.sample(content_json['sections'][2:-2], 5) + content_json['sections'][-2:]
        content_json['sections'] = selected_sections

    # Check if meta information is complete
    if 'meta' not in content_json:
        print(f"Ouch! Missing meta information in response")
        print('Trying again...')
        raise
    
    meta = content_json['meta']
    if 'website_title' not in meta or 'authors' not in meta or 'affiliations' not in meta:
        print(f"Ouch! Incomplete meta information: {meta}")
        print('Trying again...')
        raise

    # Validate that each section contains required title and content fields
    for section in content_json['sections']:
        # Check if section has title
        if type(section) != dict or not 'title' in section:
            print(f"Ouch! Section missing title: {section}")
            print('Trying again...')
            raise
        
        # Check if section has content
        if 'content' not in section:
            print(f"Ouch! Section missing content: {section}")
            print('Trying again...')
            raise
        
        # Check if content is empty
        if not section['content'] or len(section['content'].strip()) == 0:
            print(f"Ouch! Section content is empty: {section}")
            print('Trying again...')
            raise

    print(f"Content validation passed!")
    print(f"   - Paper title: {meta['website_title']}")
    print(f"   - Authors: {meta['authors']}")
    print(f"   - Affiliations: {meta['affiliations']}")
    print(f"   - Number of sections: {len(content_json['sections'])}")

    # Save parsed content to JSON file
    os.makedirs('contents', exist_ok=True)
    json.dump(content_json, open(f'contents/<{args.model_name_t}_{args.model_name_v}>_{args.website_name}_raw_content.json', 'w'), indent=4)
    return input_token, output_token, raw_result


def gen_image_and_table(args, conv_res):
    input_token, output_token = 0, 0
    raw_source = args.website_path

    output_dir = Path(f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/{args.website_name}')

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = args.website_name

    # Save page images
    for page_no, page in conv_res.document.pages.items():
        page_no = page.page_no
        page_image_filename = output_dir / f"{doc_filename}-{page_no}.png"
        with page_image_filename.open("wb") as fp:
            page.image.pil_image.save(fp, format="PNG")

    # Save images of figures and tables
    table_counter = 0
    picture_counter = 0
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = (
                output_dir / f"{doc_filename}-table-{table_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

        if isinstance(element, PictureItem):
            picture_counter += 1
            element_image_filename = (
                output_dir / f"{doc_filename}-picture-{picture_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

    # Save markdown with embedded pictures
    md_filename = output_dir / f"{doc_filename}-with-images.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

    # Save markdown with externally referenced pictures
    md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

    # Save HTML with externally referenced pictures
    html_filename = output_dir / f"{doc_filename}-with-image-refs.html"
    conv_res.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)

    tables = {}

    table_index = 1
    for table in conv_res.document.tables:
        caption = table.caption_text(conv_res.document)
        if len(caption) > 0:
            table_img_path = f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/{args.website_name}/{args.website_name}-table-{table_index}.png'
            table_img = PIL.Image.open(table_img_path)
            tables[str(table_index)] = {
                'caption': caption,
                'table_path': table_img_path,
                'width': table_img.width,
                'height': table_img.height,
                'figure_size': table_img.width * table_img.height,
                'figure_aspect': table_img.width / table_img.height,
            }

        table_index += 1

    images = {}
    image_index = 1
    for image in conv_res.document.pictures:
        caption = image.caption_text(conv_res.document)
        if len(caption) > 0:
            image_img_path = f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/{args.website_name}/{args.website_name}-picture-{image_index}.png'
            image_img = PIL.Image.open(image_img_path)
            images[str(image_index)] = {
                'caption': caption,
                'image_path': image_img_path,
                'width': image_img.width,
                'height': image_img.height,
                'figure_size': image_img.width * image_img.height,
                'figure_aspect': image_img.width / image_img.height,
            }
        image_index += 1

    json.dump(images, open(f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/{args.website_name}_images.json', 'w'), indent=4)
    json.dump(tables, open(f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/{args.website_name}_tables.json', 'w'), indent=4)

    return input_token, output_token, images, tables

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--website_name', type=str, default=None, help='Website name')
    parser.add_argument('--model_name', type=str, default='4o', help='Model name')
    parser.add_argument('--paper_path', type=str, required=True, help='Paper PDF path')
    parser.add_argument('--index', type=int, default=0, help='Index')
    args = parser.parse_args()

    # Add missing attributes
    args.model_name_t = args.model_name
    args.model_name_v = 'v1'
    args.website_name = args.website_name  # Already set
    args.website_path = args.paper_path    # Compatibility

    agent_config = get_agent_config(args.model_name)

    if args.website_name is None:
        args.website_name = args.paper_path.split('/')[-1].replace('.pdf', '').replace(' ', '_')

    # Parse raw content
    input_token, output_token, raw_result = parse_raw(args, agent_config)

    # Generate images and tables
    img_input_token, img_output_token, images, tables = gen_image_and_table(args, raw_result)
    
    # Accumulate token consumption
    total_input_token = input_token + img_input_token
    total_output_token = output_token + img_output_token

    print(f'Total token consumption: {total_input_token} -> {total_output_token}')
    print(f'  - Parse raw content: {input_token} -> {output_token}')
    print(f'  - Generate images and tables: {img_input_token} -> {img_output_token}')
    print(f'  - Images generated: {len(images)}')
    print(f'  - Tables generated: {len(tables)}')
