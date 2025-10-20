# loader.py
import fitz
from PIL import Image
from typing import List
from tqdm.asyncio import tqdm
class ImagePDFLoader:
    def __init__(self, file_path: str, dpi: int = 250):
        self.file_path = file_path
        self.dpi = dpi

    def load(self) -> List[Image.Image]:
        images = []
        try:
            doc = fitz.open(self.file_path)
            for page in doc:
                zoom_matrix = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                pix = page.get_pixmap(matrix=zoom_matrix, alpha=False)
                if pix.width > 0 and pix.height > 0:
                    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(image)
            doc.close()
        except Exception as e:
            tqdm.write(f"Error during PDF processing: {e}")
            return []
        return images
