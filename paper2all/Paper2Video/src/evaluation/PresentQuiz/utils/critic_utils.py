from PIL import Image
import io
import json

def crop_image(image, x:float, y:float, width:float, height:float):
    """Crop the image based on the normalized coordinates.
    Return the cropped image.
    This has the effect of zooming in on the image crop.

    Args:
        image (PIL.Image.Image): the input image
        x (float): the horizontal coordinate of the upper-left corner of the box
        y (float): the vertical coordinate of that corner
        width (float): the box width
        height (float): the box height

    Returns:
        cropped_img (PIL.Image.Image): the cropped image
        
    Example:
        image = Image.open("sample_img.jpg")
        cropped_img = crop_image(image, 0.2, 0.3, 0.5, 0.4)
        display(cropped_img)
    """
    
    # get height and width of image
    w, h = image.size
    
    # limit the range of x and y
    x = min(max(0, x), 1)
    y = min(max(0, y), 1)
    x2 = min(max(0, x+width), 1)
    y2 = min(max(0, y+height), 1)
    
    cropped_img = image.crop((x*w, y*h, x2*w, y2*h))

    buffer = io.BytesIO()
    cropped_img.save(buffer, format="JPEG")
    buffer.seek(0)  # Reset buffer position

    # Load as a JpegImageFile
    jpeg_image = Image.open(buffer)
    return jpeg_image


def zoom_in_image_by_bbox(image, box, padding=0.01):
    """A simple wrapper function to crop the image based on the bounding box.
    The zoom factor cannot be too small. Minimum is 0.1

    Args:
        image (PIL.Image.Image): the input image
        box (List[float]): the bounding box in the format of [x, y, w, h]
        padding (float, optional): The padding for the image crop, outside of the bounding box. Defaults to 0.05.

    Returns:
        cropped_img (PIL.Image.Image): the cropped image
        
    Example:
        image = Image.open("sample_img.jpg")
        annotated_img, boxes = detection(image, "bus")
        cropped_img = zoom_in_image_by_bbox(image, boxes[0], padding=0.1)
        display(cropped_img)
    """
    assert padding >= 0.01, "The padding should be at least 0.01"
    x, y, w, h = box
    x, y, w, h = x-padding, y-padding, w+2*padding, h+2*padding
    return crop_image(image, x, y, w, h)


def parse_inch_string(inch_str: str) -> float:
    """
    Convert a string like '12.0 Inches' into a float (12.0).
    """
    return float(inch_str.replace(" Inches", "").strip())

def convert_pptx_bboxes_to_image_space(bbox_dict, slide_width_in, slide_height_in):
    """
    Convert each PPTX bounding box (in inches) to normalized image coords.

    bbox_dict format example:
    {
      'TitleAndAuthor': {
         'left': '12.0 Inches', 'top': '1.0 Inches',
         'width': '24.0 Inches', 'height': '2.0 Inches'
      },
      ...
    }

    Returns a dictionary with the same keys, but values as [x_norm, y_norm, w_norm, h_norm].
    """
    result = {}
    for label, box in bbox_dict.items():
        left_in   = parse_inch_string(box['left'])
        top_in    = parse_inch_string(box['top'])
        width_in  = parse_inch_string(box['width'])
        height_in = parse_inch_string(box['height'])

        x_norm = left_in / slide_width_in
        y_norm = top_in  / slide_height_in
        w_norm = width_in  / slide_width_in
        h_norm = height_in / slide_height_in

        result[label] = [x_norm, y_norm, w_norm, h_norm]
    return result

def convert_pptx_bboxes_json_to_image_json(bbox_json_str, slide_width_in, slide_height_in):
    """
    Convert bounding boxes (in inches) from a JSON string to normalized image coords [0..1].

    Args:
        bbox_json_str (str): JSON text of the bounding box dictionary you provided.
                             Example of the structure (in JSON):
                             {
                                 "TitleAndAuthor": {
                                    "left": "12.0 Inches",
                                    "top": "1.0 Inches",
                                    "width": "24.0 Inches",
                                    "height": "2.0 Inches"
                                 },
                                 "Abstract-Section Title": { ... },
                                 ...
                             }
        slide_width_in (float): The total slide width in inches.
        slide_height_in (float): The total slide height in inches.

    Returns:
        str: A JSON string, where each key maps to [x_norm, y_norm, w_norm, h_norm].
    """

    def parse_inch_string(inch_str: str) -> float:
        """Helper to parse '12.0 Inches' -> 12.0 (float)."""
        return float(inch_str.replace(" Inches", "").strip())

    # 1) Parse the incoming JSON string to a Python dict
    if type(bbox_json_str) == str:
        bbox_dict = json.loads(bbox_json_str)
    else:
        bbox_dict = bbox_json_str

    # 2) Convert each bounding box to normalized coordinates [x, y, w, h]
    normalized_bboxes = {}
    for label, box in bbox_dict.items():
        left_in   = parse_inch_string(box['left'])
        top_in    = parse_inch_string(box['top'])
        width_in  = parse_inch_string(box['width'])
        height_in = parse_inch_string(box['height'])

        x_norm = left_in / slide_width_in
        y_norm = top_in  / slide_height_in
        w_norm = width_in  / slide_width_in
        h_norm = height_in / slide_height_in

        normalized_bboxes[label] = [x_norm, y_norm, w_norm, h_norm]

    # 3) Return as a JSON string
    return normalized_bboxes

