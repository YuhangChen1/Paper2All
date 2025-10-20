# app.py

import gradio as gr
import asyncio
import os
import shutil
from pathlib import Path
import time
import json
import traceback
from typing import List, Dict, Tuple, Optional

from pragent.backend.text_pipeline import pipeline as run_text_extraction
from pragent.backend.figure_table_pipeline import run_figure_extraction
from pragent.backend.blog_pipeline import generate_text_blog, generate_final_post
from pragent.backend.agents import setup_client, call_text_llm_api

import base64
import mimetypes
import re

FORMAT_PROMPT_TEMPLATE = '''
You are an expert in structuring social media content. Your task is to convert a post written in Markdown format into a structured JSON format. The JSON structure depends on the target platform.

**Platform:** {platform}
**Markdown Content:**
---
{markdown_text}
---

**Instructions:**
{platform_instructions}
'''

TWITTER_INSTRUCTIONS = '''
Convert the content into a JSON array representing a Twitter thread. Each element in the array is a tweet object.
- Each tweet object must have a "text" key. The text should be plain text, without any Markdown formatting (e.g., no `*`, `#`, `[]()`)
- If a tweet is associated with an image, add an "image_index" key with the corresponding zero-based index from the provided asset list. For example, if the first image in the Markdown `![...](img_0.png)` is used, its index is 0.
- Ensure the thread flows logically. Split the text into multiple tweets if necessary.

**Asset List (for reference):**
{asset_list}

**JSON Output Format:**
[
  {{ "text": "Text of the first tweet.", "image_index": 0 }},
  {{ "text": "Text of the second tweet." }},
  {{ "text": "Text of the third tweet.", "image_index": 1 }}
]
'''

XIAOHONGSHU_INSTRUCTIONS = '''
Convert the content into a single JSON object for a Xiaohongshu post.
- The JSON object must have a "title" key. Extract the main title from the Markdown (usually the first H1/H2 heading). The title should be plain text.
- The JSON object must have a "body" key containing the main text content, with emojis. The body text should be plain text, without any Markdown formatting (e.g., no `*`, `#`, `[]()`)
- The JSON object must have an "image_indices" key, which is an array of all image indexes used in the post, in the order they appear.

**Asset List (for reference):**
{asset_list}

**JSON Output Format:**
{{
  "title": "Your Catchy Title Here",
  "body": "The full body text of the post...",
  "image_indices": [0, 1, 2, 3]
}}
'''

def image_to_base64(path: str) -> str:

    try:

        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            mime_type = "image/jpeg" 
            if path.lower().endswith(".png"):
                mime_type = "image/png"
            else:
                mime_type = "image/jpeg"
        
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:{mime_type};base64,{encoded_string}" 
    except Exception as e:
        print(f"[!] Error converting image to base64: {e}")
        return "" 

LOGO_PATH = "pragent/logo/logo.png"
LOGO_BASE64 = ""
if os.path.exists(LOGO_PATH):
    LOGO_BASE64 = image_to_base64(LOGO_PATH)
else:
    print(f"[!] Warning: Logo file not found at {LOGO_PATH}")

async def format_post_for_display(
    markdown_text: str,
    assets: Optional[List[Dict]],
    platform: str,
    client,
    model: str
) -> Optional[Dict]:
    if platform == 'twitter':
        instructions = TWITTER_INSTRUCTIONS
    elif platform == 'xiaohongshu':
        instructions = XIAOHONGSHU_INSTRUCTIONS
    else:
        return None

    asset_str = "No assets provided."
    if assets:
        asset_str = "\n".join([f"- Index {i}: {asset['dest_name']}" for i, asset in enumerate(assets)])

    prompt = FORMAT_PROMPT_TEMPLATE.format(
        platform=platform.capitalize(),
        markdown_text=markdown_text,
        platform_instructions=instructions.format(asset_list=asset_str),
    )

    system_prompt = "You are a content formatting expert. Output only valid JSON."
    response_str = ""
    try:
        response_str = await call_text_llm_api(client, system_prompt, prompt, model)
        json_str = None
        
        match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", response_str)
        if match:
            json_str = match.group(1)
        else:
            json_str = response_str
        return json.loads(json_str.strip())

    except Exception as e:
        print(f"[!] Error formatting post for display: {e}")
        traceback.print_exc()
        return None

def render_twitter_thread(thread_data: List[Dict], assets: List[str]) -> str:
    html_parts = []
    for i, tweet in enumerate(thread_data):
        text_html = tweet.get("text", "").replace("\n", "<br>")
        image_html = ""
        if "image_index" in tweet and tweet["image_index"] < len(assets):
            img_idx = tweet["image_index"]
            img_path = assets[img_idx]
            base64_string = image_to_base64(img_path)
            image_html = f'<div class="tweet-image-container"><img src="{base64_string}" class="tweet-image"></div>'

        tweet_html = f'''
        <div class="tweet-row">
            <div class="avatar-container">
                <img src="{LOGO_BASE64}" class="avatar">
            </div>
            <div class="tweet-content">
                <div class="user-info">
                    <strong>PRAgent</strong> <span>@pr_agent</span>
                </div>
                <div class="tweet-text">{text_html}</div>
                {image_html}
            </div>
        </div>
        '''
        html_parts.append(tweet_html)

    return "".join(html_parts)

def render_xiaohongshu_post(post_data: Dict, assets: List[str]) -> str:
    """V6 - Final Version: Returns ONLY pure HTML structure."""
    title_html = f"<h2 class='xhs-title'>{post_data.get('title', '')}</h2>"
    body_text = post_data.get('body', '').replace('\n', '<br>')
    body_html = f"<div class='xhs-body'>{body_text}</div>"
    
    gallery_html = ""
    if "image_indices" in post_data and post_data["image_indices"]:
        image_indices = post_data["image_indices"]
        # Fix: Remove duplicate indices to prevent carousel showing duplicate images, while preserving order.
        unique_indices = list(dict.fromkeys(image_indices))
        valid_assets = [assets[i] for i in unique_indices if i < len(assets)]
        
        if valid_assets:
            # We still need a unique ID for the observer to find it
            carousel_id = f"carousel_{int(time.time() * 1000)}"
            
            slides_html = ""
            for i, img_path in enumerate(valid_assets):
                base64_string = image_to_base64(img_path)
                slides_html += f'<div class="carousel-slide"><div class="carousel-numbertext">{i + 1} / {len(valid_assets)}</div><img src="{base64_string}"></div>'
            
            arrows_html = ""
            if len(valid_assets) > 1:
                arrows_html = '<a class="prev">&#10094;</a><a class="next">&#10095;</a>'

            gallery_html = f'<div class="carousel-container" id="{carousel_id}">{slides_html}{arrows_html}</div>'

    return f"{gallery_html}{title_html}{body_html}"

async def process_pdf(
    pdf_file,
    text_api_key,
    vision_api_key,
    base_url,
    text_model,
    vision_model,
    platform,
    language,
    progress=gr.Progress(track_tqdm=True)
):
    # Use text_api_key for vision_api_key if it's not provided
    vision_api_key = vision_api_key or text_api_key

    if not all([pdf_file, text_api_key, vision_api_key, base_url, text_model, vision_model, platform, language]):
        raise gr.Error("Please fill in all required fields and upload a PDF.")

    work_dir = None
    try:
        session_id = f"session_{int(time.time())}"
        work_dir = Path(".temp_output") / session_id
        work_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_path = Path(work_dir) / Path(pdf_file.name).name
        shutil.copy(pdf_file.name, pdf_path)
        final_assets = []
        
        yield gr.update(value="🚀 **Processing...** Please wait.", visible=True), gr.update(value="", visible=False), gr.update(visible=False)
        progress(0.1, desc="Step 1/5: Extracting text from PDF...")
        txt_output_path = work_dir / "paper.txt"
        await run_text_extraction(str(pdf_path), str(txt_output_path))
        if not txt_output_path.exists():
            raise gr.Error("Failed to extract text from the PDF.")

        progress(0.3, desc="Step 2/5: Extracting figures from PDF...")
        extraction_work_dir = work_dir / "figure_extraction"
        extraction_work_dir.mkdir()
        paired_dir = await run_figure_extraction(str(pdf_path), str(extraction_work_dir), progress=progress)
        if not paired_dir or not any(Path(paired_dir).iterdir()):
            raise gr.Error("Failed to extract any figures from the PDF.")

        progress(0.5, desc="Step 3/5: Generating structured text draft...")
        blog_draft, source_paper_text = await generate_text_blog(
            txt_path=str(txt_output_path),
            api_key=text_api_key,
            text_api_base=base_url,
            model=text_model,
            language=language
        )
        if not blog_draft or blog_draft.startswith("Error:"):
            raise gr.Error(f"Failed to generate blog draft: {blog_draft}")
        
        progress(0.7, desc="Step 4/5: Generating final post with vision analysis...")
        final_post_md, assets_info = await generate_final_post(
            blog_draft=blog_draft,
            source_paper_text=source_paper_text,
            assets_dir=paired_dir,
            text_api_key=text_api_key,
            vision_api_key=vision_api_key,
            text_api_base=base_url,
            vision_api_base=base_url,
            text_model=text_model,
            vision_model=vision_model,
            platform=platform,
            language=language,
            post_format='rich'
        )
        if not final_post_md or final_post_md.startswith("Error:"):
            raise gr.Error(f"Failed to generate final post: {final_post_md}")

        post_content_dir = work_dir / "post"
        post_content_dir.mkdir()

        if assets_info:
            for asset in assets_info:
                dest_path = post_content_dir / Path(asset['src_path']).name
                shutil.copy(asset['src_path'], dest_path)
                # The path for rendering needs to be the absolute path to the copied file
                absolute_path_str = str(dest_path.resolve()).replace('\\', '/')
                final_assets.append(absolute_path_str)
        
        (post_content_dir / "post.md").write_text(final_post_md, encoding='utf-8')
        
        progress(0.9, desc="Step 5/5: Formatting for rich display...")
        async with setup_client(text_api_key, base_url) as client:
            structured_data = await format_post_for_display(
                final_post_md, assets_info, platform, client, text_model
            )
        if not structured_data:
            raise gr.Error("Failed to format post for display.")

        (post_content_dir / "post.json").write_text(json.dumps(structured_data, indent=2, ensure_ascii=False), encoding='utf-8')

        if platform == 'twitter':
            final_html = render_twitter_thread(structured_data, final_assets)
        else: # xiaohongshu
            final_html = render_xiaohongshu_post(structured_data, final_assets)

        zip_filename_base = f"PRAgent_post_{platform}_{session_id}"
        zip_path = shutil.make_archive(
            base_name=str(work_dir / zip_filename_base),
            format='zip',
            root_dir=str(work_dir),
            base_dir="post"
        )
        
        yield gr.update(value="✅ **Done!**"), gr.update(value=final_html, visible=True), gr.update(value=zip_path, visible=True)

    except Exception as e:
        traceback.print_exc()
        error_html = f"<h2>Error:</h2><pre>{traceback.format_exc()}</pre>"
        yield gr.update(value=f"❌ An error occurred: {e}"), gr.update(value=error_html, visible=True), gr.update(visible=False)
    finally:
        # Cleanup is disabled to prevent race conditions with Gradio's reloader
        # and to allow inspection of generated files.
        pass
        # if work_dir and work_dir.exists():
        #     shutil.rmtree(work_dir)



CUSTOM_CSS = '''
/* --- Twitter Style --- */
.tweet-row {
    display: flex; 
    align-items: flex-start; 
    padding: 16px;
    border: 1px solid #e1e8ed;
    border-radius: 15px;
    margin-bottom: 12px;
    background-color: #ffffff;
}
.avatar-container {
    flex-shrink: 0; 
    margin-right: 12px; 
}
.avatar {
    width: 48px; 
    height: 48px; 
    border-radius: 50%; 
    object-fit: cover;
}
.tweet-content {
    width: 100%; 
}
.user-info {
    font-size: 15px;
    font-weight: bold;
}
.user-info span {
    color: #536471; 
    font-weight: normal;
}
.tweet-text {
    font-size: 15px;
    line-height: 1.5;
    color: #0f1419;
    margin-top: 4px;
    word-wrap: break-word; 
}
.tweet-image-container {
    margin-top: 12px;
}
.tweet-image {
    width: 100%;
    border-radius: 15px;
    border: 1px solid #ddd; 
    display: block;
}

/* --- Xiaohongshu Style  --- */
.xhs-title { font-size: 20px; font-weight: bold; color: #333; margin-bottom: 10px; }
.xhs-body { font-size: 16px; line-height: 1.8; color: #555; word-wrap: break-word; }

#output_container {
    border: 2px dashed #ccc;
    padding: 20px;
    min-height: 100px;
    border-radius: 15px;
}
.carousel-container { position: relative; max-width: 100%; margin: auto; overflow: hidden; border-radius: 10px; }
.carousel-slide { display: none; animation: fade 0.5s ease-in-out; }
.carousel-slide:first-child { display: block; }
.carousel-slide img { width: 100%; display: block; }
.prev, .next { cursor: pointer; position: absolute; top: 50%; width: auto; padding: 16px; margin-top: -22px; color: white; font-weight: bold; font-size: 20px; transition: 0.3s ease; border-radius: 0 3px 3px 0; user-select: none; background-color: rgba(0,0,0,0.3); }
.next { right: 0; border-radius: 3px 0 0 3px; }
.prev:hover, .next:hover { background-color: rgba(0,0,0,0.6); }
.carousel-numbertext { color: #f2f2f2; font-size: 12px; padding: 8px 12px; position: absolute; top: 0; background-color: rgba(0,0,0,0.5); border-radius: 0 0 5px 0; }
@keyframes fade { from {opacity: .4} to {opacity: 1}}
'''

ACTIVATE_CAROUSEEL_JS = '''
() => {
    // We use a small 100ms delay to ensure Gradio has finished updating the HTML DOM
    setTimeout(() => {
        const container = document.getElementById('output_container');
        if (container) {
            const carousel = container.querySelector('.carousel-container');
            // Check if a carousel exists and hasn't been initialized yet
            if (carousel && !carousel.dataset.initialized) {
                console.log("PRAgent Carousel Script: JS listener has found and is activating the carousel ->", carousel.id);

                let slideIndex = 1;
                const slides = carousel.getElementsByClassName("carousel-slide");
                const prevButton = carousel.querySelector(".prev");
                const nextButton = carousel.querySelector(".next");
                if (slides.length === 0) return;

                const showSlides = () => {
                    if (slideIndex > slides.length) { slideIndex = 1; }
                    if (slideIndex < 1) { slideIndex = slides.length; }
                    for (let i = 0; i < slides.length; i++) {
                        slides[i].style.display = "none";
                    }
                    slides[slideIndex - 1].style.display = "block";
                };

                if (prevButton) {
                    prevButton.addEventListener('click', () => { slideIndex--; showSlides(); });
                }
                if (nextButton) {
                    nextButton.addEventListener('click', () => { slideIndex++; showSlides(); });
                }

                showSlides(); // Show the first slide
                carousel.dataset.initialized = 'true'; // Mark as initialized to prevent re-activation
            }
        }
    }, 100);
}
'''

with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:

    demo.queue()
    gr.Markdown("# 🚀 PRAgent: Paper to Social Media Post")
    gr.Markdown("Upload a research paper PDF, and I will generate a social media post for Twitter or Xiaohongshu, complete with images and platform-specific styling.")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_upload = gr.File(label="Upload PDF Paper", file_types=[".pdf"])
            
            with gr.Accordion("Advanced Settings", open=True):
                text_api_key_input = gr.Textbox(label="Text API Key", type="password", placeholder="Required: sk-...")
                vision_api_key_input = gr.Textbox(label="Vision API Key (Optional)", type="password", placeholder="Optional: If not provided, Text API Key will be used")
                base_url_input = gr.Textbox(label="API Base URL")
                text_model_input = gr.Textbox(label="Text Model")
                vision_model_input = gr.Textbox(label="Vision Model")

            platform_select = gr.Radio(["twitter", "xiaohongshu"], label="Target Platform", value="twitter")
            language_select = gr.Radio([("English", "en"), ("Chinese", "zh")], label="Language", value="en")

            generate_btn = gr.Button("✨ Generate Post", variant="primary")
        
        with gr.Column(scale=2):
            status_text = gr.Markdown("Idle. Please upload a file and click generate.", visible=True)
            output_container = gr.HTML(elem_id="output_container")
            download_button = gr.File(label="Download Post & Images", visible=False)

    click_event = generate_btn.click(
        fn=process_pdf,
        inputs=[
            pdf_upload,
            text_api_key_input,
            vision_api_key_input,
            base_url_input,
            text_model_input,
            vision_model_input,
            platform_select,
            language_select
        ],
        outputs=[status_text, output_container, download_button]
    )

    click_event.then(
        fn=None, 
        inputs=None,
        outputs=None,
        js=ACTIVATE_CAROUSEEL_JS 
    )

if __name__ == "__main__":
    # Create the hidden temp directory
    Path(".temp_output").mkdir(exist_ok=True)
    demo.launch()