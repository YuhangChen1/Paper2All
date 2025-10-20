# pdf2html.py
import fitz  
from pathlib import Path
import sys
from bs4 import BeautifulSoup
import asyncio
import aiofiles
from tqdm.asyncio import tqdm
def convert_pdf_sync(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        tqdm.write(f"[*] Successfully opened PDF file: {pdf_path}")
    except Exception as e:
        tqdm.write(f"[!] Error: Could not open PDF file. {e}", file=sys.stderr)
        return ""
    full_html_content = ""
    for page in doc:
        full_html_content += page.get_text("html")
    doc.close()
    soup = BeautifulSoup(full_html_content, "lxml")
    for img_tag in soup.find_all("img"):
        img_tag.decompose()
    
    return soup.prettify()

async def convert_pdf_to_text_only_html(pdf_path: str, output_path: str) -> bool:
    cleaned_html = await asyncio.to_thread(convert_pdf_sync, pdf_path)
    if not cleaned_html:
        return False
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
            await f.write(cleaned_html)
        return True
    except Exception as e:
        tqdm.write(f"[!] Error: Could not write HTML file. {e}", file=sys.stderr)
        return False
