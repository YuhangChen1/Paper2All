# yolo.py
import os
from PIL import Image
from doclayout_yolo import YOLOv10
from tqdm.asyncio import tqdm
CLASS_NAMES = {
    0: "title",
    1: "plain_text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption_above",
    7: "table_caption_below",
    8: "formula",
    9: "formula_caption",
}

def extract_and_save_layout_components(image_path, model_path, save_base_dir="./cropped_results", imgsz=1024, conf=0.2, device="cuda:0"):
    """
    Extract document layout components from an image and save screenshots by category.

    Args:
        image_path (str): Input image path
        model_path (str): Model weight path (.pt)
        save_base_dir (str): Root directory to save screenshots
        imgsz (int): The size of the input image (will be scaled to this size)
        conf (float): Confidence threshold for detection boxes
        device (str): The computing device to use, such as 'cuda:0' or 'cpu'
    """
    model = YOLOv10(model_path)
    image = Image.open(image_path)
    det_results = model.predict(image_path, imgsz=imgsz, conf=conf, device=device)

    result = det_results[0]
    boxes = result.boxes.xyxy.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()
    scores = result.boxes.conf.cpu().tolist()

    for idx, (box, cls_id, score) in enumerate(zip(boxes, classes, scores)):
        cls_id = int(cls_id)
        class_name = CLASS_NAMES.get(cls_id, f"cls{cls_id}")
        save_dir = os.path.join(save_base_dir, class_name)
        os.makedirs(save_dir, exist_ok=True)
        x1, y1, x2, y2 = map(int, box)
        cropped = image.crop((x1, y1, x2, y2))
        if cropped.mode == 'RGBA':
            cropped = cropped.convert('RGB')
        save_path = os.path.join(save_dir, f"{class_name}_{idx}_score{score:.2f}.jpg")
        cropped.save(save_path)
    tqdm.write(f"Saved a total of {len(boxes)} screenshots, saved by category in {save_base_dir}/")