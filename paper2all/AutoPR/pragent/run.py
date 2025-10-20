# run.py
import argparse
import asyncio
import os
import shutil
from pathlib import Path
import traceback
import time
import re
import hashlib
import json
from typing import Optional
from tqdm.asyncio import tqdm

from pragent.backend.text_pipeline import pipeline as run_text_extraction
from pragent.backend.figure_table_pipeline import run_figure_extraction
from pragent.backend.blog_pipeline import generate_text_blog, generate_final_post, generate_baseline_post

def get_pdf_hash(file_path: Path) -> str:
    """Calculates the SHA256 hash of a file's content."""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def create_output_package(base_dir: Path, md_content: str, assets: list):
    """
    Creates a folder with the markdown file and an 'img' subfolder for assets.
    """
    tqdm.write(f"[*] Packaging final post at: {base_dir}")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    filename = "markdown.md"
    (base_dir / filename).write_text(md_content, encoding="utf-8")

    if assets:
        assets_dir = base_dir / "img"
        assets_dir.mkdir(exist_ok=True)
        for asset_info in assets:
            src_path = Path(asset_info['src_path'])
            dest_path = assets_dir / asset_info['dest_name']
            if src_path.exists():
                shutil.copy(src_path, dest_path)
        tqdm.write(f"[*] Copied {len(assets)} assets to {assets_dir}")
    else:
        tqdm.write("[*] No assets to package for this post.")


async def process_single_project(project_path: Path, args: argparse.Namespace, platform: str, language: str):
    """
    Runs the full pipeline for a single project folder.
    """
    project_name = project_path.name
    post_format = args.post_format
    
    # Adjust project name and output information according to ablation experiment settings
    ablation_mode = args.ablation
    if ablation_mode != 'none':
        project_name = f"{project_path.name}_ablation_{ablation_mode}"
        tqdm.write("\n" + "="*80)
        tqdm.write(f"🚀 Starting ABLATION processing for project: {project_path.name}")
        tqdm.write(f"   (Ablation Mode: {ablation_mode}, Platform: {platform.capitalize()}, Format: {post_format}, Language: {language.upper()})")
        tqdm.write("="*80)
    else:
        tqdm.write("\n" + "="*80)
        tqdm.write(f"🚀 Starting processing for project: {project_name} (Platform: {platform.capitalize()}, Format: {post_format}, Language: {language.upper()})")
        tqdm.write("="*80)

    pdf_files = list(project_path.glob('*.pdf'))
    if not pdf_files:
        tqdm.write(f"[!] No PDF file found in '{project_path}'. Skipping this project.")
        return
    pdf_path = pdf_files[0]

    session_id = f"session_{int(time.time())}_{project_name}"
    work_dir = Path(args.output_dir) / ".temp" / session_id
    final_output_dir = Path(args.output_dir) / project_name
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        tqdm.write("\n--- Stage 1/4: Extracting Text from PDF ---")
        txt_output_path = work_dir / f"{project_name}.txt"
        # Pass the ablation_mode parameter
        await run_text_extraction(str(pdf_path), str(txt_output_path), ablation_mode=ablation_mode)
        if not txt_output_path.exists():
            tqdm.write(f"[!] Text extraction failed for {pdf_path.name}. Skipping.")
            return
        tqdm.write(f"[✓] Text extracted successfully.")

        tqdm.write("\n--- Stage 2/4: Extracting Figures ---")
        pdf_hash = None
        paired_dir = None
        if post_format in ['rich', 'description_only']:
            cached_figures_path = None
            if args.cache_dir:
                pdf_hash = get_pdf_hash(pdf_path)
                cached_figures_path = args.cache_dir / "figures" / pdf_hash
                if cached_figures_path.exists() and any(cached_figures_path.iterdir()):
                    tqdm.write(f"[✓] Cache hit for PDF figures '{pdf_path.name}'. Using cached data.")
                    paired_dir = str(cached_figures_path)
            
            if not paired_dir:
                if args.cache_dir:
                        tqdm.write(f"[*] Cache miss for PDF figures '{pdf_path.name}'. Running extraction.")
                extraction_work_dir = work_dir / "figure_extraction"
                extraction_work_dir.mkdir()
                paired_dir = run_figure_extraction(str(pdf_path), str(extraction_work_dir), args.model_path)
                
                if paired_dir and cached_figures_path:
                    tqdm.write(f"[*] Saving extracted figures to cache at: {cached_figures_path}")
                    shutil.copytree(paired_dir, cached_figures_path)
            
            has_figures = paired_dir and any(Path(paired_dir).rglob('paired_*'))
            if not has_figures:
                tqdm.write(f"[!] Warning: No successfully paired figures were found for format '{post_format}'.")
                tqdm.write(f"[*] Automatically switching to 'text_only' format.")
                post_format = 'text_only'
                paired_dir = None
            else:
                tqdm.write(f"[✓] Paired figures found.")
        else:
            tqdm.write("[*] Skipping figure extraction for 'text_only' format.")

        tqdm.write("\n--- Stage 3/4: Generating Structured Blog Draft ---")
        blog_draft, source_paper_text = await generate_text_blog(
            txt_path=str(txt_output_path),
            api_key=args.text_api_key,
            text_api_base=args.text_api_base,
            model=args.text_model,
            language=language,
            disable_qwen_thinking=args.disable_qwen_thinking,
            ablation_mode=ablation_mode
        )
        if not blog_draft or blog_draft.startswith("Error:"):
            tqdm.write(f"[!] Failed to generate blog draft. Error: {blog_draft}. Skipping.")
            return
        tqdm.write("[✓] Structured draft generated successfully.")

        tqdm.write("\n--- Stage 4/4: Generating Final Platform-Specific Post ---")
        description_cache_dir = args.cache_dir / "descriptions" if args.cache_dir else None
        final_post, assets = await generate_final_post(
            blog_draft=blog_draft,
            source_paper_text=source_paper_text,
            assets_dir=paired_dir,
            text_api_key=args.text_api_key,
            vision_api_key=args.vision_api_key,
            text_model=args.text_model,
            text_api_base=args.text_api_base,
            vision_model=args.vision_model,
            vision_api_base=args.vision_api_base,
            platform=platform,
            language=language,
            post_format=post_format,
            pdf_hash=pdf_hash,
            description_cache_dir=str(description_cache_dir) if description_cache_dir else None,
            disable_qwen_thinking=args.disable_qwen_thinking,
            ablation_mode=ablation_mode # <-- Pass the ablation mode parameter
        )
        if not final_post or final_post.startswith("Error:"):
            tqdm.write(f"[!] Failed to generate final post. Error: {final_post}. Skipping.")
            return
        tqdm.write("[✓] Final post generated successfully.")

        create_output_package(final_output_dir, final_post, assets)
        tqdm.write(f"\n✅ Successfully completed processing for project: {project_path.name} (mode: {ablation_mode})")

    except Exception as e:
        tqdm.write(f"\n[!!!] An unexpected error occurred while processing {project_path.name}: {e}")
        traceback.print_exc()
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir)
            tqdm.write(f"[*] Cleaned up temporary directory: {work_dir}")


async def process_baseline_project(
    project_path: Path,
    args: argparse.Namespace,
    platform: str,
    language: str,
    log_lock: Optional[asyncio.Lock] = None,
    log_data: Optional[dict] = None,
    log_file_path: Optional[Path] = None
):
    baseline_mode = args.baseline_mode
    project_name = f"{project_path.name}_baseline_{baseline_mode}"
    
    tqdm.write("\n" + "="*80)
    tqdm.write(f"🚀 Starting BASELINE processing for project: {project_path.name}")
    tqdm.write(f"   (Mode: {baseline_mode}, Platform: {platform.capitalize()}, Language: {language.upper()})")
    tqdm.write("="*80)

    pdf_files = list(project_path.glob('*.pdf'))
    if not pdf_files:
        tqdm.write(f"[!] No PDF file found in '{project_path}'. Skipping.")
        return
    pdf_path = pdf_files[0]

    session_id = f"session_{int(time.time())}_{project_name}"
    work_dir = Path(args.output_dir) / ".temp" / session_id
    final_output_dir = Path(args.output_dir) / project_name
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        tqdm.write("\n--- Stage 1/3: Extracting Text from PDF ---")
        txt_output_path = work_dir / f"{project_name}.txt"
        await run_text_extraction(str(pdf_path), str(txt_output_path))
        if not txt_output_path.exists():
            tqdm.write(f"[!] Text extraction failed for {pdf_path.name}. Skipping.")
            return
        tqdm.write(f"[✓] Text extracted successfully.")
        
        paper_text = txt_output_path.read_text(encoding="utf-8")

        paired_dir = None
        if baseline_mode == 'with_figure':
            tqdm.write("\n--- Stage 2/3: Extracting Figures (for baseline) ---")
            
            pdf_hash = None
            if args.cache_dir:
                pdf_hash = get_pdf_hash(pdf_path)
                cached_figures_path = args.cache_dir / "figures" / pdf_hash
                if cached_figures_path.exists() and any(cached_figures_path.iterdir()):
                    tqdm.write(f"[✓] Cache hit for PDF figures '{pdf_path.name}'. Using cached data.")
                    paired_dir = str(cached_figures_path)
            
            if not paired_dir:
                if args.cache_dir:
                    tqdm.write(f"[*] Cache miss for PDF figures '{pdf_path.name}'. Running extraction.")
                
                extraction_work_dir = work_dir / "figure_extraction"
                extraction_work_dir.mkdir()
                extracted_data_dir = run_figure_extraction(str(pdf_path), str(extraction_work_dir), args.model_path)
                
                if extracted_data_dir and any(Path(extracted_data_dir).iterdir()):
                    paired_dir = extracted_data_dir
                    if args.cache_dir and cached_figures_path:
                        tqdm.write(f"[*] Saving extracted figures to cache at: {cached_figures_path}")
                        shutil.copytree(extracted_data_dir, cached_figures_path)
                else:
                     tqdm.write("[!] Warning: Figure extraction failed or found no figures.")

        else:
            tqdm.write("\n--- Stage 2/3: Skipping Figure Extraction ---")

        tqdm.write("\n--- Stage 3/3: Generating Baseline Post ---")
        baseline_post, assets, think_token_count = await generate_baseline_post(
            paper_text=paper_text,
            api_key=args.text_api_key,
            api_base=args.text_api_base,
            model=args.text_model,
            platform=platform,
            language=language,
            disable_qwen_thinking=args.disable_qwen_thinking,
            mode=baseline_mode,
            assets_dir=paired_dir
        )
        
        tqdm.write(f"[*] 'Thinking' tokens used (baseline): {think_token_count}")

        log_key = f"{project_path.name}_{baseline_mode}"
        if args.log_think_tokens and log_lock and log_data is not None and log_file_path:
            async with log_lock:
                log_data[log_key] = {
                    "think_tokens": think_token_count,
                    "model": args.text_model,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                }
                log_file_path.write_text(json.dumps(log_data, indent=4, ensure_ascii=False))
                tqdm.write(f"[*] Logged think token count for '{log_key}' to {log_file_path.name}")
        
        if not baseline_post or baseline_post.startswith("Error:"):
            tqdm.write(f"[!] Failed to generate baseline post. Error: {baseline_post}. Skipping.")
            return
        tqdm.write("[✓] Baseline post generated successfully.")

        create_output_package(final_output_dir, baseline_post, assets)
        tqdm.write(f"\n✅ Successfully completed baseline processing. Output saved to: {final_output_dir}")

    except Exception as e:
        tqdm.write(f"\n[!!!] An unexpected error occurred during baseline processing for {project_name}: {e}")
        traceback.print_exc()
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir)
            tqdm.write(f"[*] Cleaned up temporary directory: {work_dir}")


async def main():
    """
    Main function to parse arguments and run the batch processing.
    """
    parser = argparse.ArgumentParser(description="PRAgent (Advanced): Batch process PDF projects based on folder names.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing the project subfolders.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where the final posts will be saved.")
    
    parser.add_argument("--model-path", type=str, default="pragent/model/doclayout_yolo_docstructbench_imgsz1024.pt", help="Path to the YOLO model for document layout analysis.")

    parser.add_argument("--text-api-key", type=str, default=None, help="Your API Key for text models.")
    parser.add_argument("--vision-api-key", type=str, default=None, help="Your API Key for vision models. If not provided, it defaults to the text_api_key.")
    parser.add_argument("--text-api-base", type=str, default=None, help="The base URL for the text model API.")
    parser.add_argument("--vision-api-base", type=str, default=None, help="The base URL for the vision model API. If not provided, it defaults to the text_api_base.")

    parser.add_argument("--text-model", type=str, default="gpt-4o", help="Model for text generation tasks.")
    parser.add_argument("--vision-model", type=str, default="gpt-4o", help="Model for vision-related tasks.")
    parser.add_argument("--concurrency", type=int, default=1, help="Maximum number of concurrent projects to process.")

    parser.add_argument(
        "--baseline-mode",
        type=str,
        default=None,
        choices=["original", "fewshot", "with_figure"],
        help="If specified, run a specific baseline generation process instead of the full pipeline."
    )

    parser.add_argument("--log-think-tokens", action="store_true", help="Enable logging of 'think' tokens to a JSON file in the output directory.")

    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional. Directory to cache reusable assets like extracted figures and AI-generated descriptions.")
    parser.add_argument("--post-format", type=str, default="rich", choices=["rich", "description_only", "text_only"], help="Specify the output format for the final post. 'rich': full text and images. 'description_only': text with image descriptions embedded. 'text_only': just the text.")

    parser.add_argument("--disable-qwen-thinking", action="store_true", help="Disable the 'thinking' mode for Qwen models by setting enable_thinking=False.")

    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=[
            "none",
            "no_logical_draft",
            "no_visual_analysis",
            "no_visual_integration",
            "no_hierarchical_summary",
            "no_platform_adaptation",
            "stage2"
        ],
        help="Specify an ablation study mode. 'stage2' combines no_logical_draft, no_visual_analysis, and no_visual_integration."
    )

    args = parser.parse_args()

    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()

    # API Key and Base URL handling
    args.text_api_key = args.text_api_key or os.getenv("OPENAI_API_KEY")
    args.text_api_base = args.text_api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    
    if not args.text_api_key:
        print("Error: OPENAI_API_KEY is not configured. Please pass it as an argument or set it in your .env file.")
        return

    if args.vision_api_key is None:
        args.vision_api_key = args.text_api_key
    if args.vision_api_base is None:
        args.vision_api_base = args.text_api_base
    print(f"[*] Using Text API Base: {args.text_api_base}")
    print(f"[*] Using Vision API Base: {args.vision_api_base}")

    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        (args.cache_dir / "figures").mkdir(exist_ok=True)
        (args.cache_dir / "descriptions").mkdir(exist_ok=True)
        print(f"[*] Using unified cache at: {args.cache_dir}")

    semaphore = asyncio.Semaphore(args.concurrency)
    input_path = Path(args.input_dir)
    if not input_path.is_dir():
        print(f"Error: Input directory not found at '{args.input_dir}'")
        return

    project_folders = [d for d in input_path.iterdir() if d.is_dir()]
    if not project_folders:
        print(f"No project subfolders found in '{args.input_dir}'.")
        return

    print(f"Found {len(project_folders)} project folder(s) to process with a concurrency of {args.concurrency}.")
    if args.baseline_mode:
        print(f"--- RUNNING IN BASELINE MODE ({args.baseline_mode}) ---")
        if args.log_think_tokens:
            print("[*] 'Think' token logging is ENABLED.")
    elif args.ablation != 'none':
        print(f"--- RUNNING IN ABLATION MODE ({args.ablation}) ---")
    else:
        print(f"--- RUNNING IN ADVANCED MODE (Format: {args.post_format}) ---")
        
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    log_lock = None
    log_file_path = None
    log_data = {}
    if args.baseline_mode and args.log_think_tokens:
        log_lock = asyncio.Lock()
        log_file_path = Path(args.output_dir) / "think_token_log.json"
        if log_file_path.exists():
            try:
                log_data = json.loads(log_file_path.read_text(encoding="utf-8"))
                print(f"[*] Loaded existing token log from: {log_file_path}")
            except json.JSONDecodeError:
                print(f"[!] Warning: Could not parse existing log file. Starting fresh.")
        else:
            print(f"[*] Token log file will be created at: {log_file_path}")

    async def process_with_semaphore(coro):
        async with semaphore:
            await coro

    tasks = []
    for project_path in project_folders:
        project_name = project_path.name
        
        output_project_name = project_name
        if args.baseline_mode:
            output_project_name = f"{project_name}_baseline_{args.baseline_mode}"
        elif args.ablation != 'none':
            output_project_name = f"{project_name}_ablation_{args.ablation}"
        
        output_project_dir = Path(args.output_dir) / output_project_name
        
        log_key = f"{project_name}_{args.baseline_mode}" if args.baseline_mode else None
        if args.baseline_mode and args.log_think_tokens and log_key in log_data:
            tqdm.write(f"[*] Skipping '{project_name}' (mode: {args.baseline_mode}): Result already in token log.")
            continue
        elif output_project_dir.exists() and any(output_project_dir.iterdir()):
            tqdm.write(f"[*] Skipping '{output_project_name}': Output directory already exists.")
            continue
            
        folder_name = project_path.name
        platform, language = None, None

        if folder_name.isdigit():
            platform = "twitter"
            language = "en"
        elif re.search('[a-zA-Z]', folder_name):
            platform = "xiaohongshu"
            language = "zh"
        else:
            tqdm.write(f"[*] Skipping folder '{folder_name}' as its name is neither purely numeric nor contains English letters.")
            continue

        if args.baseline_mode:
            coro = process_baseline_project(project_path, args, platform, language, log_lock, log_data, log_file_path)
        else:
            coro = process_single_project(project_path, args, platform, language)
        
        tasks.append(process_with_semaphore(coro))
        
    if tasks:
        original_count = len(project_folders)
        skipped_count = original_count - len(tasks)
        if skipped_count > 0:
            print(f"[*] Skipped {skipped_count} already completed project(s).")
            
        await tqdm.gather(
            *tasks, 
            desc="Processing Projects", 
            unit="project",
            total=len(tasks)
        )
    else:
        print("[*] All projects have already been processed.")


if __name__ == "__main__":
    asyncio.run(main())