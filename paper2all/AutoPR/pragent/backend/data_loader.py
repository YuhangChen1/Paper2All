# data_loader.py
import asyncio
import aiofiles
from pathlib import Path
import re
from typing import List, Dict
from tqdm.asyncio import tqdm
async def load_plain_text(txt_path: str) -> str:
    """Asynchronously load plain text content from a .txt file."""
    try:
        async with aiofiles.open(txt_path, mode='r', encoding='utf-8') as f:
            return await f.read()
    except Exception as e:
        tqdm.write(f"[!] Error reading text file '{txt_path}': {e}")
        return ""

def load_paired_image_paths(base_dir: Path) -> List[Dict]:
    """
    Recursively scan 'paired_*' folders and load the paths of the main image and its caption image.
    """
    items = []
    if not base_dir.is_dir():
        tqdm.write(f"[!] Error: Could not find the base folder for paired results: {base_dir}")
        return items

    tqdm.write(f"[*] Recursively loading image-text pairs from {base_dir}...")
    
    item_dirs = sorted(
        [d for d in base_dir.rglob('paired_*') if d.is_dir()],
        key=lambda p: p.name  
    )

    for item_dir in item_dirs:
        item_files = list(item_dir.glob('*.jpg'))
        if len(item_files) < 2:
            continue

        main_item_path, caption_path = None, None
        for f in item_files:
            if "caption" in f.name:
                caption_path = f
            else:
                main_item_path = f
        
        if main_item_path and caption_path:
            items.append({
                "type": "figure" if "figure" in item_dir.name else "table",
                "item_path": str(main_item_path.resolve()),
                "caption_path": str(caption_path.resolve()),
            })
            
    tqdm.write(f"[*] Loading complete, found {len(items)} image-text pairs.")
    return items