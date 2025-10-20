# pragent/backend/blog_pipeline.py

from tqdm.asyncio import tqdm
import asyncio
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from openai import AsyncOpenAI
import re
import os
import json
import pytesseract
from PIL import Image
import asyncio

from pragent.backend.agents import setup_client, BlogGeneratorAgent, FigureDescriberAgent, BlogIntegratorAgent, call_text_llm_api,call_text_llm_api_with_token_count
from pragent.backend.data_loader import load_plain_text, load_paired_image_paths
from pragent.backend.text_processor import summarize_long_text
from .prompts import (
    TEXT_GENERATOR_PROMPT, TEXT_GENERATOR_PROMPT_CHINESE,
    TWITTER_RICH_TEXT_PROMPT_ENGLISH, TWITTER_TEXT_ONLY_PROMPT_ENGLISH,
    TWITTER_RICH_TEXT_PROMPT_CHINESE, TWITTER_TEXT_ONLY_PROMPT_CHINESE,
    XIAOHONGSHU_PROMPT_ENGLISH, XIAOHONGSHU_PROMPT_CHINESE,
    XIAOHONGSHU_TEXT_ONLY_PROMPT_ENGLISH, XIAOHONGSHU_TEXT_ONLY_PROMPT_CHINESE,
    BASELINE_PROMPT_ENGLISH, BASELINE_PROMPT_CHINESE,
    GENERIC_RICH_PROMPT_CHINESE,GENERIC_RICH_PROMPT_ENGLISH,
    GENERIC_TEXT_ONLY_PROMPT_CHINESE,GENERIC_TEXT_ONLY_PROMPT_ENGLISH,
    BASELINE_FEWSHOT_PROMPT_ENGLISH, BASELINE_FEWSHOT_PROMPT_CHINESE
)
TOKEN_THRESHOLD = 8000

PROMPT_MAPPING = {
    ('twitter', 'rich', 'en'): TWITTER_RICH_TEXT_PROMPT_ENGLISH,
    ('twitter', 'text_only', 'en'): TWITTER_TEXT_ONLY_PROMPT_ENGLISH,
    ('twitter', 'rich', 'zh'): TWITTER_RICH_TEXT_PROMPT_CHINESE,
    ('twitter', 'text_only', 'zh'): TWITTER_TEXT_ONLY_PROMPT_CHINESE,
    ('xiaohongshu', 'rich', 'en'): XIAOHONGSHU_PROMPT_ENGLISH,
    ('xiaohongshu', 'rich', 'zh'): XIAOHONGSHU_PROMPT_CHINESE,
    ('xiaohongshu', 'text_only', 'en'): XIAOHONGSHU_TEXT_ONLY_PROMPT_ENGLISH,
    ('xiaohongshu', 'text_only', 'zh'): XIAOHONGSHU_TEXT_ONLY_PROMPT_CHINESE,
    ('generic', 'rich', 'en'): GENERIC_RICH_PROMPT_ENGLISH,
    ('generic', 'text_only', 'en'): GENERIC_TEXT_ONLY_PROMPT_ENGLISH,
    ('generic', 'rich', 'zh'): GENERIC_RICH_PROMPT_CHINESE,
    ('generic', 'text_only', 'zh'): GENERIC_TEXT_ONLY_PROMPT_CHINESE,
}


# Asynchronous OCR helper function
async def ocr_image_to_text(image_path: str) -> str:
    """
    Performs OCR on an image file to extract text asynchronously.
    """
    if not Path(image_path).exists():
        return ""
    try:
        # pytesseract is a blocking library, so we run it in a thread pool
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None, 
            lambda: pytesseract.image_to_string(Image.open(image_path))
        )
        return text.strip()
    except Exception as e:
        tqdm.write(f"[!] OCR failed for {image_path}: {e}")
        return ""


async def generate_text_blog(
    txt_path: str, api_key: str, text_api_base: str, model: str, language: str,
    disable_qwen_thinking: bool = False, ablation_mode: str = "none"
) -> Tuple[str, str]:
    """
    Generates a structured, factual blog DRAFT in the specified language. (Stage 1)
    """
    async with setup_client(api_key, text_api_base) as client:
        if not client:
            return "Error: API client configuration failed.", None
        
        paper_text = await load_plain_text(txt_path)
        if not paper_text:
            return "Error: Could not load text file.", None

        text_for_generation = ""
        if len(paper_text) > TOKEN_THRESHOLD: 
            if ablation_mode == 'no_hierarchical_summary':
                tqdm.write(f"[*] ABLATION (no_hierarchical_summary): Truncating text to {TOKEN_THRESHOLD} characters.")
                text_for_generation = paper_text[:TOKEN_THRESHOLD]
            else:
                summarized_text = await summarize_long_text(
                    paper_text,
                    model,
                    client,
                    disable_qwen_thinking=disable_qwen_thinking
                )
                if summarized_text.startswith("Error:"):
                    summarized_text = paper_text[:TOKEN_THRESHOLD]
                text_for_generation = summarized_text
        else:
            text_for_generation = paper_text
        
        if ablation_mode in ['no_logical_draft', 'stage2']:
            ablation_reason = "no_logical_draft" if ablation_mode != 'stage2' else 'stage2'
            tqdm.write(f"[*] ABLATION ({ablation_reason}): Skipping structured draft generation.")
            return text_for_generation, text_for_generation

        draft_prompt = TEXT_GENERATOR_PROMPT_CHINESE if language == 'zh' else TEXT_GENERATOR_PROMPT
        generator = BlogGeneratorAgent(draft_prompt, model)
        generated_blog_draft = await generator.run(
            client, 
            text_for_generation, 
            disable_qwen_thinking=disable_qwen_thinking
        )
        return generated_blog_draft, text_for_generation


async def generate_final_post(
    blog_draft: str,
    source_paper_text: str,
    assets_dir: Optional[str],
    text_api_key: str,
    vision_api_key: str,
    text_api_base: str,
    vision_api_base: str,
    vision_model: str,
    text_model: str,
    platform: str,
    language: str,
    post_format: str,
    description_cache_dir: Optional[str] = None,
    pdf_hash: Optional[str] = None,
    disable_qwen_thinking: bool = False,
    ablation_mode: str = "none"
) -> Optional[Tuple[str, Optional[List[Dict]]]]:
    effective_platform = platform
    if ablation_mode == 'no_platform_adaptation':
        tqdm.write(f"[*] ABLATION (no_platform_adaptation): Using generic prompts instead of '{platform}' specific prompts.")
        effective_platform = 'generic'

    prompt_format = 'rich' if post_format == 'description_only' else post_format
    prompt_key = (effective_platform, prompt_format, language)
    selected_prompt = PROMPT_MAPPING.get(prompt_key)
    
    if not selected_prompt:
        tqdm.write(f"[!] Warning: No prompt found for configuration: {prompt_key}. Falling back to generic prompt.")
        generic_fallback_key = ('generic', prompt_format, language)
        selected_prompt = PROMPT_MAPPING.get(generic_fallback_key)
        if not selected_prompt:
            return f"Error: No prompt found for configuration: {prompt_key} or generic fallback.", None

    tqdm.write(f"\n--- Generating final post for: Platform='{effective_platform}', Format='{post_format}', Language='{language}' ---")

    items_with_descriptions = []
    if post_format in ['rich', 'description_only'] and assets_dir and Path(assets_dir).is_dir():
        all_items = load_paired_image_paths(Path(assets_dir))
        all_items = all_items[:50]  # Limit to first 50 items to avoid overloading the model
        if all_items:
            cache_file_path = None
            if description_cache_dir and pdf_hash:
                sanitized_model_name = re.sub(r'[\\/:"*?<>|]', '_', vision_model)
                cache_dir = Path(description_cache_dir) / pdf_hash
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file_path = cache_dir / f"{sanitized_model_name}.json"

            if cache_file_path and cache_file_path.exists() and ablation_mode not in ['no_visual_analysis', 'stage2']:
                tqdm.write(f"[✓] Cache hit! Loading all descriptions from {cache_file_path}")
                with cache_file_path.open('r', encoding='utf-8') as f:
                    items_with_descriptions = json.load(f)
            
            else:
                # Trigger this ablation also for 'stage2'
                if ablation_mode in ['no_visual_analysis', 'stage2']:
                    ablation_reason = "no_visual_analysis" if ablation_mode != 'stage2' else 'stage2'
                    tqdm.write(f"[*] ABLATION ({ablation_reason}): Using OCR on caption images instead of vision model.")
                    temp_items_with_desc = []
                    
                    ocr_tasks = [ocr_image_to_text(item['caption_path']) for item in all_items]
                    ocr_results = await asyncio.gather(*ocr_tasks)

                    for i, item in enumerate(all_items):
                        caption_content = ocr_results[i]
                        if caption_content:
                            item['description'] = caption_content
                            temp_items_with_desc.append(item)
                    items_with_descriptions = temp_items_with_desc
                else:
                    # Full pipeline: use vision model
                    tqdm.write(f"--- Cache miss. Describing {len(all_items)} new figures using model '{vision_model}'... ---")
                    async with setup_client(vision_api_key, vision_api_base) as vision_client:
                        if not vision_client:
                            return "Error: Vision API client configuration failed.", None
                        
                        describer = FigureDescriberAgent(model=vision_model)
                        description_tasks = [
                            describer.run(
                                vision_client, 
                                item['item_path'], 
                                item['caption_path'],
                                disable_qwen_thinking=disable_qwen_thinking
                            ) for item in all_items
                        ]
                        descriptions = await asyncio.gather(*description_tasks)
                        
                        temp_items_with_desc = []
                        for i, item in enumerate(all_items):
                            if not descriptions[i].startswith("Error:"):
                                item['description'] = descriptions[i]
                                temp_items_with_desc.append(item)
                        items_with_descriptions = temp_items_with_desc

                # Prevent caching for 'stage2' as well
                if cache_file_path and ablation_mode not in ['no_visual_analysis', 'stage2']:
                    tqdm.write(f"[*] Saving all descriptions to cache file: {cache_file_path}")
                    with cache_file_path.open('w', encoding='utf-8') as f:
                        json.dump(items_with_descriptions, f, ensure_ascii=False, indent=4)
                elif cache_file_path and ablation_mode in ['no_visual_analysis', 'stage2']:
                    ablation_reason = "no_visual_analysis" if ablation_mode != 'stage2' else 'stage2'
                    tqdm.write(f"[*] ABLATION ({ablation_reason}): Description caching is disabled for this mode to avoid saving OCR results.")

    items_with_descriptions = items_with_descriptions[:20]
    if post_format in ['rich', 'description_only'] and not items_with_descriptions:
        return f"Error: '{post_format}' format requires images, but none were found/described.", None

    async with setup_client(text_api_key, text_api_base) as text_client:
        if not text_client: return "Error: Text API client configuration failed.", None
        
        if ablation_mode in ['no_visual_integration', 'stage2'] and post_format in ['rich', 'description_only']:
            ablation_reason = "no_visual_integration" if ablation_mode != 'stage2' else 'stage2'
            tqdm.write(f"[*] ABLATION ({ablation_reason}): Generating text first, then appending all figures at the end.")
            
            integrator = BlogIntegratorAgent(selected_prompt, model=text_model)
            text_only_post = await integrator.run(
                local_client=text_client, 
                blog_text=blog_draft, 
                items_with_descriptions=[],
                source_text=source_paper_text,
                disable_qwen_thinking=disable_qwen_thinking
            )

            if not text_only_post or text_only_post.startswith("Error:"):
                return f"Blog integration failed for text-only part: {text_only_post}", None

            final_blog_content = text_only_post
            assets_for_packaging = []
            for i, item_data in enumerate(items_with_descriptions):
                if post_format == 'rich':
                    new_asset_filename = f"img_{i}{Path(item_data['item_path']).suffix}"
                    alt_text = f"Figure {i}"
                    new_markdown_tag = f"\n\n![{alt_text}](./img/{new_asset_filename})"
                    assets_for_packaging.append({'src_path': item_data['item_path'], 'dest_name': new_asset_filename, 'new_index': i})
                    final_blog_content += new_markdown_tag
                elif post_format == 'description_only':
                    alt_text_description = item_data.get('description', f'Figure {i}').strip().replace('\n', ' ')
                    new_markdown_tag = f"\n\n![{alt_text_description}]()"
                    final_blog_content += new_markdown_tag
            
            return final_blog_content, assets_for_packaging if assets_for_packaging else None

        integrator = BlogIntegratorAgent(selected_prompt, model=text_model)
        final_post_with_placeholders = await integrator.run(
            local_client=text_client, 
            blog_text=blog_draft, 
            items_with_descriptions=items_with_descriptions, 
            source_text=source_paper_text,
            disable_qwen_thinking=disable_qwen_thinking
        )

    if not final_post_with_placeholders or final_post_with_placeholders.startswith("Error:"):
        return f"Blog integration failed: {final_post_with_placeholders}", None

    found_indices = re.findall(r'\[FIGURE_PLACEHOLDER_(\d+)\]', final_post_with_placeholders)
    final_blog_content = final_post_with_placeholders
    assets_for_packaging = []
    
    if found_indices:
        items_map = {i: item for i, item in enumerate(items_with_descriptions)}
        for new_index, original_index_str in enumerate(found_indices):
            original_index = int(original_index_str)
            item_data = items_map.get(original_index)
            if not item_data: continue
            
            placeholder_to_replace = f"[FIGURE_PLACEHOLDER_{original_index}]"
            
            if post_format == 'rich':
                new_asset_filename = f"img_{new_index}{Path(item_data['item_path']).suffix}"
                alt_text = f"Figure {new_index}" 
                new_markdown_tag = f"![{alt_text}](./img/{new_asset_filename})"
                assets_for_packaging.append({'src_path': item_data['item_path'], 'dest_name': new_asset_filename, 'new_index': new_index})
            elif post_format == 'description_only':
                alt_text_description = item_data.get('description', f'Figure {new_index}').strip().replace('\n', ' ')
                new_markdown_tag = f"![{alt_text_description}]()"
            else:
                new_markdown_tag = ""
            final_blog_content = final_blog_content.replace(placeholder_to_replace, new_markdown_tag, 1)
    
    final_blog_content = re.sub(r'\[FIGURE_PLACEHOLDER_(\d+)\]', '', final_blog_content)

    if post_format == 'rich':
        return final_blog_content, assets_for_packaging
    else:
        return final_blog_content, None


async def generate_baseline_post(
    paper_text: str,
    api_key: str,
    api_base: str,
    model: str,
    platform: str,
    language: str,
    disable_qwen_thinking: bool = False,
    mode: str = 'original',
    assets_dir: Optional[str] = None
) -> Tuple[str, List[Dict], int]:
    """
    Generates a post using a simple, single-prompt baseline method.
    """
    tqdm.write(f"\n--- Generating baseline post (mode: {mode}) for: Platform='{platform}', Language='{language}' ---")
    
    async with setup_client(api_key, api_base) as client:
        if not client:
            return "Error: API client configuration failed.", [], 0

        if mode == 'fewshot':
            prompt_template = BASELINE_FEWSHOT_PROMPT_CHINESE if language == 'zh' else BASELINE_FEWSHOT_PROMPT_ENGLISH
        else:
            prompt_template = BASELINE_PROMPT_CHINESE if language == 'zh' else BASELINE_PROMPT_ENGLISH
            
        user_prompt = prompt_template.format(paper_text=paper_text[:20000], platform=platform.capitalize())
        system_prompt = "You are an assistant that summarizes academic papers for social media."
        
        text_post, think_token_count = await call_text_llm_api_with_token_count(
            local_client=client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            disable_qwen_thinking=disable_qwen_thinking
        )

        if text_post.startswith("Error:"):
            return text_post, [], think_token_count
            
        final_post = text_post
        assets_for_packaging = []
        if mode == 'with_figure' and assets_dir and Path(assets_dir).is_dir():
            tqdm.write(f"[*] Attaching top 3 figures/tables for 'with_figure' baseline...")
            
            paired_item_dirs = [
                d for d in Path(assets_dir).rglob('paired_*') 
                if d.is_dir() and (d.name.startswith('paired_figure_') or d.name.startswith('paired_table_'))
            ]
            def get_global_sort_key(dir_path: Path):
                page_num = -1
                item_type = ''
                item_index = -1

                try:

                    page_match = re.search(r'page_(\d+)', dir_path.parts[-2])
                    if page_match:
                        page_num = int(page_match.group(1))
                except (IndexError, ValueError):
                    pass 

                item_match = re.search(r'paired_(figure|table)_(\d+)', dir_path.name)
                if item_match:
                    item_type = item_match.group(1)
                    item_index = int(item_match.group(2))
                
                return (page_num, item_index)

            sorted_dirs = sorted(paired_item_dirs, key=get_global_sort_key)
            
            all_items = []

            for item_dir in sorted_dirs:
                item_type = 'figure' if 'figure' in item_dir.name else 'table'
                
                item_file = next(
                    (f for f in item_dir.iterdir() if f.is_file() and f.name.startswith(item_type) and 'caption' not in f.name),
                    None
                )
                if item_file:
                    all_items.append(item_file)


            selected_items = all_items[:3]
            
            if selected_items:
                final_post += "\n\n--- Key Figures & Tables ---\n"
                for i, item_path in enumerate(selected_items):
                    new_asset_filename = f"img_{i}{item_path.suffix}"
                    alt_text = "Table" if "table" in item_path.parent.name else "Figure"
                    alt_text += f" {i+1}"
                    
                    final_post += f"\n![{alt_text}](./img/{new_asset_filename})"
                    assets_for_packaging.append({'src_path': str(item_path), 'dest_name': new_asset_filename})
                tqdm.write(f"[✓] Appended {len(selected_items)} items (figures/tables) to the post.")
            else:
                tqdm.write("[!] Warning: 'with_figure' mode was selected, but no paired items were found.")

        return final_post, assets_for_packaging, think_token_count
