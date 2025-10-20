# pragent/backend/text_pipeline.py

import asyncio
import sys
import os
from pathlib import Path
import aiofiles.os 
from tqdm.asyncio import tqdm
from pragent.backend.pdf2html import convert_pdf_to_text_only_html
from pragent.backend.html2txt import convert_html_to_txt

async def pipeline(pdf_path: str, output_txt_path: str, ablation_mode: str = "none"):
    """
    Defines the complete ASYNCHRONOUS conversion flow from PDF to TXT.
    The ablation_mode parameter is accepted but the primary logic for summarization
    ablation is handled downstream in blog_pipeline.py.
    """
    tqdm.write("--- PDF to TXT Conversion Pipeline Started ---")
    
    pdf_file = Path(pdf_path)
    intermediate_html_path = pdf_file.with_suffix(".temp.html")

    tqdm.write("\n--- Step 1/3: Converting PDF to HTML ---")
    if not await convert_pdf_to_text_only_html(pdf_path, str(intermediate_html_path)):
        tqdm.write("[!] PDF to HTML conversion failed. Aborting pipeline.", file=sys.stderr)
        return

    tqdm.write(f"\n--- Step 2/3: Converting HTML to TXT ---")
    if not await convert_html_to_txt(str(intermediate_html_path), output_txt_path):
        tqdm.write("[!] HTML to TXT conversion failed. Aborting pipeline.", file=sys.stderr)
    else:
        tqdm.write(f"\n[✓] Success! Final text file saved to: {output_txt_path}")

    tqdm.write(f"\n--- Step 3/3: Cleaning up temporary files ---")
    try:
        await aiofiles.os.remove(intermediate_html_path)
        tqdm.write(f"[*] Temporary file '{intermediate_html_path.name}' deleted successfully.")
    except OSError as e:
        tqdm.write(f"[!] Error deleting temporary file: {e}", file=sys.stderr)
        
    tqdm.write("\n--- Pipeline Finished ---")