# html2txt.py
from bs4 import BeautifulSoup
import sys
import aiofiles 
from tqdm.asyncio import tqdm
async def convert_html_to_txt(html_file_path: str, output_txt_path: str) -> bool:
    try:
        async with aiofiles.open(html_file_path, 'r', encoding='utf-8') as f:
            html_from_file = await f.read()
    except FileNotFoundError:
        tqdm.write(f"[!] Error: Intermediate HTML file not found '{html_file_path}'.", file=sys.stderr)
        return False
    except Exception as e:
        tqdm.write(f"[!] Error reading HTML file: {e}", file=sys.stderr)
        return False

    soup = BeautifulSoup(html_from_file, "lxml")
    paragraphs = soup.find_all('p')

    extracted_lines = [p.get_text(separator=" ", strip=True) for p in paragraphs if p.get_text(strip=True)]
    tqdm.write(f"[*] Text extraction complete, found {len(extracted_lines)} valid lines of text.")

    try:
        full_text_content = "\n".join(extracted_lines)
        async with aiofiles.open(output_txt_path, 'w', encoding='utf-8') as f:
            await f.write(full_text_content)
        return True
    except Exception as e:
        tqdm.write(f"[!] Error writing to TXT file: {e}", file=sys.stderr)
        return False
