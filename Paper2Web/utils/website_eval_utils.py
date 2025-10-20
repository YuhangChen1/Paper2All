import random
import string
import yaml
import os

def get_template_path(template_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return os.path.join(project_root, "utils", "prompt_templates", template_name)
import PIL
import tempfile
import io
from camel.models import ModelFactory
from math import ceil
from openai import OpenAI
from camel.messages import BaseMessage
from utils.src.model_utils import parse_pdf
from urllib.parse import unquote
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM
from pytorch_fid.fid_score import compute_statistics_of_path
import pytorch_fid.fid_score as fid
from PIL import Image
from httpx import Timeout
from docling.document_converter import DocumentConverter, PdfFormatOption
import re
import shutil
import pytesseract
from utils.wei_utils import account_token
from camel.types import ModelPlatformType, ModelType
from marker.models import create_model_dict
from camel.configs import ChatGPTConfig
from camel.agents import ChatAgent
from jinja2 import Environment, StrictUndefined
from utils.src.utils import get_json_from_response
from pathlib import Path
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from collections import defaultdict

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

import math
import base64
import requests
from io import BytesIO
from PIL import Image

import torch
import json
import os
import pickle as pkl
import numpy as np
from transformers import AltCLIPProcessor, AltCLIPModel

def pil_to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    """
    Convert a PIL.Image to a base-64 data URI suitable for
    the OpenAI/vLLM 'image_url' block.
    fmt = 'PNG' (lossless) or 'JPEG' (smaller, 0-100 quality).
    """
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.save(buf, format="JPEG", quality=90)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:{mime};base64,{b64}"

def md_to_blocks(
    md: str,
    base_dir=''
):
    blocks, pos = [], 0
    pat = re.compile(r'!\[.*?\]\((.*?)\)', re.DOTALL)

    for m in pat.finditer(md):
        # --- text before this image ---------------------------------------
        txt = md[pos : m.start()].strip()
        if txt:
            blocks.append({"type": "text", "text": txt})

        # --- the image itself ---------------------------------------------
        img_path = unquote(m.group(1))
        img_path = os.path.join(base_dir, img_path)

        blocks.append({"type": "image_url", "image_url": {"url": pil_to_data_uri(Image.open(img_path), fmt="PNG")}})
        pos = m.end()

    # --- any trailing text -------------------------------------------------
    tail = md[pos:].strip()
    if tail:
        blocks.append({"type": "text", "text": tail})

    return blocks

def compute_vlm_ppl(content):
    VLLM_BASE_URL = "http://localhost:7000/v1"
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

    client = OpenAI(
        api_key="EMPTY",            # vLLM ignores auth
        base_url=VLLM_BASE_URL,
        timeout=Timeout(5000)
    )

    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{
            "role": "user",
            "content": content,
        }],
        temperature=0.0,
        max_tokens=1, 
        logprobs=0,
        extra_body={
            "prompt_logprobs": 1,
            "echo": True 
        }
    )

    lp_list = resp.to_dict()["prompt_logprobs"]   # list[dict]
    total_lp = 0.0
    n_text   = 0

    for token_entry in lp_list:
        if not token_entry:
            continue
        # find the sub-entry with rank==1 (the real token)
        token_info = next(v for v in token_entry.values() if v["rank"] == 1)
        tok, lp = token_info["decoded_token"], token_info["logprob"]

        # skip image sentinels / padding
        if re.fullmatch(r"<\|?image[^>]*\|?>", tok):
            continue

        total_lp += lp
        n_text   += 1

    return math.exp(-total_lp / n_text)

def compute_interleaved_ppl(paper_name, website_method):
    base_dir = f'eval_website_markdown/{paper_name}/{website_method}'
    with open(os.path.join(base_dir, f'{paper_name}-with-image-refs.md'), 'r') as f:
        md = f.read()
    parts = md_to_blocks(md, base_dir)
    while True:
        try:
            return compute_vlm_ppl(parts)
        except:
            parts = parts[:-1]
            continue


def get_visual_ppl(image, text):

    img_uri = pil_to_data_uri(image, fmt="PNG")
    content = [
        {"type": "text",      "text": text},
        {"type": "image_url", "image_url": {"url": img_uri}},
    ]

    return compute_vlm_ppl(content)

def estimate_visual_tokens(
    images,
    *,
    resized_height: int | None = None,
    resized_width: int | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
):
    """Return per‑image *visual‑token* counts for **Qwen‑2.5‑VL**.

    Token count = ⌈H/28⌉ × ⌈W/28⌉ after the model’s resizing rules. The helper
    mirrors those rules so your offline estimate aligns with server billing.
    """
    counts = []

    for img in images:
        h, w = img.height, img.width
        # manual resize overrides (rarely used)
        if resized_height and resized_width:
            h, w = resized_height, resized_width
        # area‑based resize to respect min/max tokens
        if min_pixels and h * w < min_pixels:
            scale = (min_pixels / (h * w)) ** 0.5
            h, w = int(h * scale), int(w * scale)
        if max_pixels and h * w > max_pixels:
            scale = (max_pixels / (h * w)) ** 0.5
            h, w = int(h * scale), int(w * scale)
        # round each side to multiple of 28
        h = ceil(h / 28) * 28
        w = ceil(w / 28) * 28
        counts.append((h // 28) * (w // 28))

    return counts

def image_memory_size(img: Image.Image, fmt="JPEG"):
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.tell()

def truncate_images_to_fit(
    images,
    *,
    max_ctx: int,
    **resize_kwargs,
):
    """Drop **later** images until total visual tokens ≤ *max_ctx*.

    Chronology‑preserving version: keeps the earliest images intact and
    trims the tail when necessary.
    """

    tokens = estimate_visual_tokens(images, **resize_kwargs)
    max_size = 45 * 1024 * 1024  # 45 MB
    total_size = 0
    keep = []
    total = 0
    for img, n_tok in zip(images, tokens):  # iterate in original order
        if total + n_tok > max_ctx:
            break  # stop adding once budget exceeded – we drop the rest
        img_size = image_memory_size(img)
        if total_size + img_size > max_size:
            break
        keep.append(img)
        total += n_tok
    return keep


def compute_website_image_ppl(images):
    max_ctx = 128_000  # max visual tokens for Qwen2.5-VL
    truncated_images = truncate_images_to_fit(images, max_ctx=max_ctx)
    img_uris = [pil_to_data_uri(image, fmt="PNG") for image in truncated_images]
    content = [
        {"type": "image_url", "image_url": {"url": img_uri}} for img_uri in img_uris
    ]

    return compute_vlm_ppl(content)


def compute_clip_embeddings(folder, model, processor, device):
    """
    Loads each image in `folder`, encodes it with the CLIP model,
    and returns a list (or array) of embeddings, shape (N, D).
    """
    model.eval()
    embeddings = []

    # Gather all image files
    image_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not image_files:
        print(f"No valid images found in {folder}")
        return np.array([])

    for filename in image_files:
        img_path = os.path.join(folder, filename)
        image = Image.open(img_path).convert('RGB')

        # Preprocess for CLIP
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Encode and get the image embeddings
        with torch.no_grad():
            clip_emb = model.get_image_features(**inputs)
            # Move to CPU and convert to NumPy
            clip_emb = clip_emb[0].cpu().numpy()
            embeddings.append(clip_emb)

    return np.array(embeddings)  # shape: (N, D)

def compute_clip_embedding(input_data, model, processor, device='cuda', input_type=None):
    """
    Compute a CLIP embedding for either an image or text.

    Parameters
    ----------
    input_data : str or PIL.Image.Image
        - If a string: treated as a file path to an image (if file exists) or as a text prompt.
        - If a PIL.Image.Image: treated as an image.
    model : CLIPModel
        The loaded CLIP model (e.g., from Hugging Face).
    processor : CLIPProcessor
        The corresponding CLIP processor for tokenization/preprocessing.
    device : torch.device
        The device to run inference on.
    input_type : {'image', 'text', None}, optional
        Force the mode; if `None` (default) the function will try to infer from `input_data`.

    Returns
    -------
    np.ndarray
        A 1D NumPy array of length D (the CLIP embedding dimension).
    """
    model.eval()

    # Decide mode
    if input_type == "image":
        mode = "image"
    elif input_type == "text":
        mode = "text"
    else:
        # auto-detect
        if isinstance(input_data, Image.Image):
            mode = "image"
        elif isinstance(input_data, str) and os.path.isfile(input_data):
            mode = "image"
        else:
            mode = "text"

    # Preprocess + encode
    with torch.no_grad():
        if mode == "image":
            if isinstance(input_data, str):
                image = Image.open(input_data).convert("RGB")
            else:
                image = input_data.convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            features = model.get_image_features(**inputs)

        else:  # text mode
            # CLIP expects a list of strings
            texts = [input_data] if isinstance(input_data, str) else list(input_data)
            inputs = processor(
                text=texts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=processor.tokenizer.model_max_length,
            ).to(device)
            features = model.get_text_features(**inputs)

        # extract, move to CPU, convert to numpy
        emb = features[0].cpu().numpy()

    return emb

def compute_average_l2_distance(emb1, emb2):
    """
    Computes the average L2 distance across all pairs in emb1 x emb2.
    - emb1 shape: (N1, D)
    - emb2 shape: (N2, D)
    Returns a single float: mean of all pairwise distances.
    """
    distances = []
    for e1 in emb1:
        for e2 in emb2:
            dist = np.linalg.norm(e1 - e2)  # L2 distance
            distances.append(dist)
    return np.mean(distances) if distances else float('nan')

def compute_cosine_similarity(e1, e2):
    """
    Computes the cosine similarity between two vectors.
    - e1 shape: (D,)
    - e2 shape: (D,)
    Returns a single float: cosine similarity.
    """
    dot = np.dot(e1, e2)
    norm_e1 = np.linalg.norm(e1)
    norm_e2 = np.linalg.norm(e2)
    return dot / (norm_e1 * norm_e2 + 1e-8)  # avoid division by zero

def compute_average_cosine_similarity(emb1, emb2):
    """
    Computes the average cosine similarity across all pairs in emb1 x emb2.
    - emb1 shape: (N1, D)
    - emb2 shape: (N2, D)
    Returns a single float: mean of all pairwise similarities.
    """
    similarities = []
    for e1 in emb1:
        for e2 in emb2:
            # Cosine similarity = (e1 · e2) / (||e1|| * ||e2||)
            dot = np.dot(e1, e2)
            norm_e1 = np.linalg.norm(e1)
            norm_e2 = np.linalg.norm(e2)
            cos_sim = dot / (norm_e1 * norm_e2 + 1e-8)
            similarities.append(cos_sim)
    return np.mean(similarities) if similarities else float('nan')

def compare_folders_with_clip(folder1, folder2):
    """
    Loads a CLIP model from Hugging Face,
    gets embeddings for each folder,
    and computes both average L2 distance and average cosine similarity.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name="openai/clip-vit-base-patch32"
    model_name = "BAAI/AltCLIP"
    model = AltCLIPModel.from_pretrained(model_name).to('cuda')
    processor = AltCLIPProcessor.from_pretrained(model_name)

    # Compute embeddings
    emb1 = compute_clip_embeddings(folder1, model, processor, device)
    emb2 = compute_clip_embeddings(folder2, model, processor, device)

    if emb1.size == 0 or emb2.size == 0:
        print("One of the folders had no valid images. Comparison not possible.")
        return None, None

    # Average L2 Distance
    avg_l2 = compute_average_l2_distance(emb1, emb2)

    # Average Cosine Similarity
    avg_cos_sim = compute_average_cosine_similarity(emb1, emb2)

    return avg_l2, avg_cos_sim

def convert_folder_to_grayscale(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = Image.open(input_path).convert('L').convert('RGB')  # grayscale + 3 channels
            img.save(output_path)

def compute_fid_with_grayscale(reference_website_folder, generated_website_img_folder, clip=False):
    # Step 1: Create grayscale versions in tmp/
    tmp_ref = 'tmp/ref_gray'
    tmp_gen = 'tmp/gen_gray'

    if os.path.exists('tmp/ref_gray'):
        shutil.rmtree('tmp/ref_gray')

    if os.path.exists('tmp/gen_gray'):
        shutil.rmtree('tmp/gen_gray')
    os.makedirs(tmp_ref)
    os.makedirs(tmp_gen)

    convert_folder_to_grayscale(reference_website_folder, tmp_ref)
    convert_folder_to_grayscale(generated_website_img_folder, tmp_gen)

    if clip:
        return compare_folders_with_clip(tmp_ref, tmp_gen)

    # Step 2: Compute FID
    model = fid.InceptionV3([fid.InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to('cuda')
    m1, s1 = compute_statistics_of_path(tmp_ref, model, 1, 2048, 'cuda')
    m2, s2 = compute_statistics_of_path(tmp_gen, model, 1, 2048, 'cuda')
    fid_score = fid.calculate_frechet_distance(m1, s1, m2, s2)

    return fid_score

def compute_fid(reference_website_folder, generated_website_img_folder, clip=False):
    if clip:
        return compare_folders_with_clip(reference_website_folder, generated_website_img_folder)
    model = fid.InceptionV3([fid.InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to('cuda')

    m1, s1 = compute_statistics_of_path(reference_website_folder, model, 1, 2048, 'cuda')
    m2, s2 = compute_statistics_of_path(generated_website_img_folder, model, 1, 2048, 'cuda')

    fid_score = fid.calculate_frechet_distance(
        m1, s1, m2, s2
    )

    return fid_score


def get_website_text(website_path):
    markdown_clean_pattern = re.compile(r"<!--[\s\S]*?-->")
    converter = DocumentConverter()
    raw_result = converter.convert(website_path)

    raw_markdown = raw_result.document.export_to_markdown()
    text_content = markdown_clean_pattern.sub("", raw_markdown)
    if len(text_content) < 500:
        print('\nParsing with docling failed, using marker instead\n')
        parser_model = create_model_dict(device='cuda', dtype=torch.float16)
        text_content, rendered = parse_pdf(website_path, model_lst=parser_model, save_file=False)
    return text_content

def qwen2_vl_ppl(
    image: Image.Image,
    text: str,
    *,
    vllm_url: str = "http://localhost:8000/v1/chat/completions",
    model: str   = "Qwen/Qwen2-VL-7B",     # whatever name you passed to vLLM
) -> float:
    """
    Compute PPL(text | image) with a Qwen2-VL-7B model served by vLLM.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.
    text : str
        Prompt text that follows the image.
    vllm_url : str, default "http://localhost:8000/v1/chat/completions"
        The full URL of the vLLM chat endpoint.
    model : str, default "Qwen2-VL-7B"
        Model name as registered when you launched vLLM.

    Returns
    -------
    float
        Per-token perplexity of `text` conditioned on `image`.
    """

    # 1) Encode the image as base64‑PNG
    buf = BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # 2) Build a multimodal chat message: image first, then text
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                },
                {
                    "type": "text",
                    "text": text
                }
            ],
        }
    ]

    # 3) Ask vLLM to echo the prompt and give log‑probs
    payload = {
        "model":       model,
        "messages":    messages,
        "temperature": 0.0,
        "max_tokens":  0,    # no generation – just evaluate prompt
        "echo":        True,
        "logprobs":    1
    }

    resp = requests.post(vllm_url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # 4) Extract prompt‑token log‑probs
    token_logps = data["choices"][0]["logprobs"]["token_logprobs"]

    # Ignore special tokens & image placeholders (returned as None)
    valid = [lp for lp in token_logps if lp is not None]
    if not valid:
        raise ValueError("No valid text tokens found in logprobs")

    # 5) Perplexity = exp( − average logp )
    return math.exp(-sum(valid) / len(valid))

def get_ppl(
    text: str,
    model_name: str = "meta-llama/Llama-2-7b-hf",
    stride: int = 512,
) -> float:
    """Compute perplexity for arbitrarily long *text* using a sliding‑window approach.

    Parameters
    ----------
    text : str
        The input string (any length).
    model_name : str, optional
        HF Hub id of the model to use, by default "meta-llama/Llama-2-7b-hf".
    stride : int, optional
        Overlap between successive windows. 512 tends to work well for most
        Transformer LMs with a 2 k context. Increase it for higher accuracy at
        the cost of more compute.

    Returns
    -------
    float
        Per‑token perplexity under the given model.
    """
    # Load tokenizer / model once per call (cache makes subsequent calls cheap)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",  # place on GPU if available
    )
    model.eval()

    # Encode the whole string in one shot
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    # Model context length (e.g. 2048 for Llama‑2)
    max_len = model.config.max_position_embeddings

    # --- Short input: fits in a single window --------------------------------
    if input_ids.size(0) <= max_len:
        with torch.no_grad():
            out = model(input_ids.unsqueeze(0).to(model.device), labels=input_ids.unsqueeze(0).to(model.device))
        return torch.exp(out.loss).item()

    # --- Long input: sliding window with overlap -----------------------------
    nlls = []  # negative‑log‑likelihoods (already multiplied by #tokens scored)
    for i in range(0, input_ids.size(0), stride):
        begin_loc = max(i + stride - max_len, 0)
        end_loc = min(i + stride, input_ids.size(0))
        trg_len = end_loc - i  # tokens we actually score in this window

        ids_chunk = input_ids[begin_loc:end_loc]
        labels = ids_chunk.clone()
        labels[:-trg_len] = -100  # mask out purely‑context tokens

        with torch.no_grad():
            out = model(ids_chunk.unsqueeze(0).to(model.device), labels=labels.unsqueeze(0).to(model.device))
            nll = out.loss * trg_len  # make additive so we can sum across windows
        nlls.append(nll)

        if end_loc == input_ids.size(0):
            break

    ppl = torch.exp(torch.stack(nlls).sum() / input_ids.size(0))
    return ppl.item()

def extract_text_from_image(image_path):
    """
    Open an image file and use Tesseract OCR to extract text.
    :param image_path: Path to the image file
    :return: Extracted text as a string
    """
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in `text` according to OpenAI's tokenizer.
    
    :param text: The input string you want to measure.
    :param model: Which model’s encoding to mimic (defaults to “gpt-4o”).
                  Common choices: "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini".
    :return: The number of tokens.
    """
    # Grab the right encoder for the model; falls back to the nearest base if needed
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # All chat models use the cl100k_base encoding
        enc = tiktoken.get_encoding("cl100k_base")
    
    return len(enc.encode(text))

def count_words(text):
    """
    Count the number of words in a given text string.
    :param text: Input text
    :return: Number of words found
    """
    # Use a regex to find word-like sequences
    words = re.findall(r"\w+", text)
    return len(words)


def count_words_in_image(image_path):
    """
    Extract text from an image and count its words.
    :param image_path: Path to the image file
    :return: Word count (int)
    """
    text = extract_text_from_image(image_path)
    return count_words(text)

def count_tokens_in_image(image_path, model="gpt-4o"):
    """
    Extract text from an image and count its tokens.
    :param image_path: Path to the image file
    :param model: Which model’s encoding to mimic (defaults to “gpt-4o”).
                  Common choices: "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini".
    :return: Token count (int)
    """
    text = extract_text_from_image(image_path)
    return count_tokens(text, model=model)

def png_to_optimized_jpeg(img: Image.Image,
                          max_size=(2048, 2048),
                          quality=80) -> BytesIO:
    """
    Take a PNG PIL Image, downsample it to fit within max_size (preserving aspect
    ratio), then JPEG-compress it at the given quality into a BytesIO buffer.
    
    Args:
      img:     PIL.Image opened from your .png
      max_size: (width, height) ceiling for downsampling
      quality: JPEG quality 1–95 (higher = better quality / larger file)
    
    Returns:
      BytesIO containing the JPEG bytes.
    """
    # 1) Downsample in place (preserves aspect ratio)
    img_copy = img.copy()
    img_copy.thumbnail(max_size, resample=Image.LANCZOS)
    
    # 2) Convert to RGB (drop alpha) and save with compression
    rgb = img_copy.convert("RGB")
    buf = BytesIO()
    rgb.save(
        buf,
        format="JPEG",
        quality=quality,        # try 80–90 for minimal artifacts
        optimize=True,          # runs an extra pass to squeeze out redundant data
        progressive=True        # allows incremental render in browsers/viewers
    )
    buf.seek(0)
    return buf

def get_answers_and_remove_answers(questions):
    question_only, answers, aspects = {}, {}, {}
    for key, val in questions.items():
        question_only[key] = {
            'question': val['question'],
            'options': val['options']
        }
        answers[key] = val['answer']
        aspects[key] = val['aspect']
    return question_only, answers, aspects

def open_folder_images(
    folder_path,
    paper_name,
    return_path=False,
    format='png',
    max_size=(700, 700),
    quality=80
):
    """
    Opens all PNG images in folder_path named '{paper_name}-{index}.png',
    starting from index=1 up to the first missing, and returns them
    either as file-paths (if return_path=True) or as PIL.Image objects.
    
    If img_format!='png', each PNG is downsampled to fit within max_size
    (preserving aspect ratio), converted to RGB, and saved into an
    in-memory JPEG with the given quality, optimize and progressive flags.
    """
    images = []
    index = 1

    while True:
        png_name = f"{paper_name}-{index}.png"
        path = os.path.join(folder_path, png_name)
        if not os.path.isfile(path):
            break

        if format == 'png':
            if return_path:
                images.append(path)
            else:
                images.append(Image.open(path))
        else:
            # 1) Load and downsample
            with Image.open(path) as im:
                thumb = im.copy()
                thumb.thumbnail(max_size, resample=Image.LANCZOS)

                # 2) Convert & compress to JPEG in-memory
                rgb = thumb.convert("RGB")
                buf = BytesIO()
                rgb.save(
                    buf,
                    format="JPEG",
                    quality=quality,        # e.g. 80–90
                    optimize=True,          # extra pass to strip redundant data
                    progressive=True        # for incremental rendering
                )
                buf.seek(0)

                if return_path:
                    # we return a tuple of (fake-jpg-filename, buffer)
                    jpg_name = png_name.rsplit('.', 1)[0] + '.jpg'
                    images.append((jpg_name, buf))
                else:
                    images.append(Image.open(buf))

        index += 1

    return images

def ensure_under_limit_pil(img, max_bytes: int = 10 * 1024 * 1024) -> Image.Image:
    # Ensure RGB mode for JPEG compatibility
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    # Try saving at decreasing qualities until under the limit
    for quality in (90, 80, 70, 60, 50):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        new_raw = buf.getvalue()
        if len(new_raw) <= max_bytes:
            return Image.open(io.BytesIO(new_raw))

    # Fallback: resize by half and save at low quality
    w, h = img.size
    img_resized = img.resize((w // 2, h // 2), Image.LANCZOS)
    buf = io.BytesIO()
    img_resized.save(buf, format="JPEG", quality=50)
    new_raw = buf.getvalue()
    if len(new_raw) > max_bytes:
        raise RuntimeError("Could not reduce image under size limit")

    return Image.open(io.BytesIO(new_raw))

def eval_qa_get_answer(website_input, questions, answers, aspects, input_type, agent_config):
    agent_name = f'answer_question_from_{input_type}'
    with open(get_template_path(f"{agent_name}.yaml"), "r") as f:
        config = yaml.safe_load(f)

    if agent_config['model_platform'].is_vllm:
        actor_model = ModelFactory.create(
            model_platform=agent_config['model_platform'],
            model_type=agent_config['model_type'],
            model_config_dict=agent_config['model_config'],
            url=agent_config['url'],
        )
    else:
        actor_model = ModelFactory.create(
            model_platform=agent_config['model_platform'],
            model_type=agent_config['model_type'],
            model_config_dict=agent_config['model_config'],
        )

    actor_sys_msg = config['system_prompt']

    actor_agent = ChatAgent(
        system_message=actor_sys_msg,
        model=actor_model,
        message_window_size=None,
    )

    actor_agent.reset()

    jinja_env = Environment(undefined=StrictUndefined)

    template = jinja_env.from_string(config["template"])

    if input_type == 'text':
        prompt = template.render(**{
            'questions': questions,
            'website_text': website_input,
        })
        response = actor_agent.step(prompt)
        agent_answers = get_json_from_response(response.msgs[0].content)
    elif input_type == 'image':
        if 'max_images' in agent_config:
            max_images = agent_config['max_images']
        else:
            max_images = len(website_input)
        prompt = template.render(**{
            'questions': questions,
        })
        msg = BaseMessage.make_user_message(
            role_name="User",
            content=prompt,
            image_list=website_input[:max_images],
        )
        response = actor_agent.step(msg)
        agent_answers = get_json_from_response(response.msgs[0].content)

    input_token, output_token = account_token(response)

    accuracy, aspect_accuracy = compute_accuracy(agent_answers, answers, aspects)

    return accuracy, aspect_accuracy, agent_answers, input_token, output_token
    

def compute_accuracy(predicted, ground_truth, aspects):

    correct_global = 0
    total_global   = len(ground_truth)

    total_by_aspect   = defaultdict(int)
    correct_by_aspect = defaultdict(int)

    for q, pred_info in predicted.items():
        letter_pred = pred_info['answer']
        ref = pred_info.get('reference', 'NA')

        # Count this question toward its aspect, even if NA or missing gt
        aspect = aspects.get(q, 'Unknown')
        total_by_aspect[aspect] += 1

        if letter_pred == 'NA' or ref == 'NA':
            continue  # automatically wrong

        if q in ground_truth:
            letter_gt = ground_truth[q].split('.')[0].strip()

            if len(letter_pred) > 0:
                letter_pred = letter_pred[0].upper()
            if letter_pred == letter_gt:
                correct_global += 1
                correct_by_aspect[aspect] += 1

    overall_accuracy = correct_global / total_global if total_global else 0.0

    # Build the per-aspect dictionary
    aspect_summary = {}
    for aspect, total in total_by_aspect.items():
        correct = correct_by_aspect[aspect]
        acc     = correct / total if total else 0.0
        aspect_summary[aspect] = {
            'total':   total,
            'correct': correct,
            'accuracy': acc
        }

    return overall_accuracy, aspect_summary

def shuffle_question_options(question_data):

    # Make a deep copy so we do not modify the original data
    new_data = deepcopy(question_data)
    
    # Loop over each question
    for q_key, q_content in new_data.items():
        original_options = q_content.get("options", [])
        original_answer = q_content.get("answer", "")
        
        # Extract the text portion of the original answer.
        # We assume that each option (and the answer) has the format "X. <option text>"
        if ". " in original_answer:
            orig_letter, orig_text = original_answer.split(". ", 1)
        else:
            # If format not as expected, use the whole answer string
            orig_text = original_answer
        
        # Remove the letter prefixes from each option to obtain a list of option texts.
        option_texts = []
        for opt in original_options:
            if ". " in opt:
                _, text = opt.split(". ", 1)
            else:
                text = opt
            option_texts.append(text)
        
        # Shuffle the list of option texts
        random.shuffle(option_texts)
        
        # Reassign new letter labels (A, B, C, etc.) to the shuffled options.
        new_options = []
        correct_answer_new = None
        letters = list(string.ascii_uppercase)
        for idx, text in enumerate(option_texts):
            new_opt = f"{letters[idx]}. {text}"
            new_options.append(new_opt)
            # When the option's text matches the original answer text, update the answer field.
            if text == orig_text:
                correct_answer_new = new_opt
        
        # Fallback in case no match is found (should not happen if data is consistent)
        if correct_answer_new is None:
            correct_answer_new = original_answer
        
        # Update the question entry with the new options and answer.
        q_content["options"] = new_options
        q_content["answer"] = correct_answer_new

    return new_data

def png_to_pdf(input_path: str, output_path: str) -> None:

    with Image.open(input_path) as img:
        # Convert image to RGB if it has an alpha channel
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1])  # use alpha channel as mask
            img = background
        else:
            img = img.convert("RGB")

        img.save(output_path, "PDF", resolution=200.0)

def extract_images_and_sections(md):
    parts = re.split(r'(## [^\n]+)', md)
    records = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        content = parts[i+1]
        # Find all image paths
        images = re.findall(r'!\[.*?\]\((.*?)\)', content)
        if images:
            # Remove lines that are image markdown
            lines = content.splitlines()
            cleaned = [
                line for line in lines
                if not re.match(r'!\[.*?\]\(.*?\)', line.strip())
            ]
            section_text = "\n".join(cleaned).strip()
            for img in images:
                records.append({
                    'section': header,
                    'image_path': unquote(img),
                    'section_text': section_text
                })

    return records

def gen_eval_markdown(paper_name, website_method, website_path, figure_count_only=False):
    model_name="openai/clip-vit-base-patch32"
    model_name = "BAAI/AltCLIP"
    model = AltCLIPModel.from_pretrained(model_name).to('cuda')
    processor = AltCLIPProcessor.from_pretrained(model_name)

    # create a uniquely‐named file in your system temp dir (or specify dir="tmp")
    with tempfile.NamedTemporaryFile(suffix=".pdf", prefix="website_", dir="tmp", delete=False) as tf:
        unique_pdf = tf.name

    if website_method != 'paper':
        # convert into that file
        png_to_pdf(website_path, unique_pdf)
        website_path = unique_pdf
    IMAGE_RESOLUTION_SCALE = 5.0
    agent_name = f'image_captioner'
    with open(get_template_path(f"{agent_name}.yaml"), "r") as f:
        config = yaml.safe_load(f)
    actor_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict=ChatGPTConfig().as_dict(), # [Optional] the config for model
    )

    actor_sys_msg = config['system_prompt']

    actor_agent = ChatAgent(
        system_message=actor_sys_msg,
        model=actor_model,
        message_window_size=None,
    )
    jinja_env = Environment(undefined=StrictUndefined)

    template = jinja_env.from_string(config["template"])
    prompt = template.render()

    raw_source = website_path
    converter = DocumentConverter()
    raw_result = converter.convert(raw_source)
    raw_markdown = raw_result.document.export_to_markdown()

    output_dir = Path(f'eval_website_markdown/{paper_name}/{website_method}')
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    conv_res = doc_converter.convert(raw_source)

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = paper_name

    # Save images of figures and tables
    table_counter = 0
    picture_counter = 0
    for element, _level in list(conv_res.document.iterate_items()):
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = (
                output_dir / f"table-{table_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

        if isinstance(element, PictureItem):
            picture_counter += 1
            element_image_filename = (
                output_dir / f"picture-{picture_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")


    md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
    markdown = conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)



    images = {}
    images_and_text = extract_images_and_sections(markdown)
    if figure_count_only:
        return len(images_and_text)
    for res in images_and_text:
        image_path = os.path.join('eval_website_markdown', paper_name, website_method, res['image_path'])
        image_img = Image.open(image_path)
        section_text = res['section_text']
        image_clip_embedding = compute_clip_embedding(image_img, model, processor)
        section_text_clip_embedding = compute_clip_embedding(section_text, model, processor)
        msg = BaseMessage.make_user_message(
            role_name="User",
            content=prompt,
            image_list=[image_img],
        )
        response = actor_agent.step(msg)
        images[res['image_path']] = {
            'image_clip_embedding': image_clip_embedding,
            'section_text_clip_embedding': section_text_clip_embedding,
            'section_text': section_text,
            'LLM_caption': response.msgs[0].content,
        }
        actor_agent.reset()

    def replace_with_caption(match):
        # match.group(1) is the URL‐encoded path
        path = match.group(1)
        # lookup the caption (fallback to empty string if missing)
        caption = images.get(path.replace('%20', ' '), {}).get("LLM_caption", "")
        return f"Image: {caption}"

    # perform the replacement
    new_md = re.sub(
        r'!\[.*?\]\((.*?)\)',   # find ![…](path)
        replace_with_caption,   # callback to build replacement
        markdown
    )

    pkl.dump(images, open(f'eval_website_markdown/{paper_name}/{website_method}/images.pkl', 'wb'))
    with open(f'eval_website_markdown/{paper_name}/{website_method}/markdown_with_images.md', 'w') as f:
        f.write(new_md)

    website_text = get_website_text(website_path)

    return images, website_text, markdown, new_md

def get_questions(paper_text, mode, model_type):
    from dotenv import load_dotenv
    load_dotenv()
    agent_name = f'generate_question_{mode}'
    with open(get_template_path(f"{agent_name}.yaml"), "r",encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 首先，创建配置字典
    model_config = ChatGPTConfig().as_dict()
    # 然后，从中移除 OpenRouter 不支持的 'logit_bias' 参数
    model_config.pop('logit_bias', None)

    # 最后，将清理过的配置字典传入模型工厂
    actor_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENROUTER,
        model_type=model_type,
        model_config_dict=model_config,
    )

    actor_sys_msg = config['system_prompt']

    actor_agent = ChatAgent(
        system_message=actor_sys_msg,
        model=actor_model,
        message_window_size=10,
    )

    jinja_env = Environment(undefined=StrictUndefined)

    template = jinja_env.from_string(config["template"])
    question_generation_prompt = template.render(**{
        'document_markdown': paper_text,
    })
    response = actor_agent.step(question_generation_prompt)
    questions = get_json_from_response(response.msgs[0].content)
    questions = shuffle_question_options(questions)

    return questions

def eval_vlm_as_judge_aspect(website_image_list, agent_config, eval_aspect):
    judge_model = ModelFactory.create(
        model_platform=agent_config['model_platform'],
        model_type=agent_config['model_type'],
        model_config_dict=agent_config['model_config'],
    )

    judge_name = f'{eval_aspect}_judge'
    with open(get_template_path(f"{judge_name}.yaml"), "r") as f:
        judge_config = yaml.safe_load(f)
    
    judge_sys_msg = judge_config['system_prompt']
    judge_agent = ChatAgent(
        system_message=judge_sys_msg,
        model=judge_model,
        message_window_size=None,
    )
    jinja_env = Environment(undefined=StrictUndefined)
    template = jinja_env.from_string(judge_config["template"])
    prompt = template.render()

    judge_message = BaseMessage.make_user_message(
        role_name="User",
        content=prompt,
        image_list=website_image_list,
    )

    response = judge_agent.step(judge_message)
    return get_json_from_response(response.msgs[0].content)

def eval_vlm_as_judge(website_image_list, agent_config, aspect=None):
    aspects = [
        'aesthetic_element',
        'aesthetic_engagement',
        'aesthetic_layout',
        'information_low_level',
        'information_logic',
        'information_content',
    ]

    if aspect == 'aesthetic':
        aspects = [
            'aesthetic_element',
            'aesthetic_engagement',
            'aesthetic_layout',
        ]
    elif aspect == 'information':
        aspects = [
            'information_low_level',
            'information_logic',
            'information_content',
        ]

    results = {}
    for aspect in aspects:
        results[aspect] = eval_vlm_as_judge_aspect(website_image_list, agent_config, aspect)
    
    return results

def evaluate_website_completeness(html_file_path, paper_name):

    from bs4 import BeautifulSoup
    import re
    
    print(f'🔍 开始评估网站完整性: {html_file_path}')
    
    # 读取HTML文件
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # 解析HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 初始化评估结果
    completeness_scores = {}
    detailed_results = {}
    
    # 1. 论文标题检查 (Title)
    title_score, title_details = check_title_completeness(soup, paper_name)
    completeness_scores['title'] = title_score
    detailed_results['title'] = title_details
    
    # 2. 作者信息检查 (Authors)
    authors_score, authors_details = check_authors_completeness(soup)
    completeness_scores['authors'] = authors_score
    detailed_results['authors'] = authors_details
    
    # 3. 摘要检查 (Abstract)
    abstract_score, abstract_details = check_abstract_completeness(soup)
    completeness_scores['abstract'] = abstract_score
    detailed_results['abstract'] = abstract_details
    
    # 4. 核心贡献检查 (Contributions)
    contributions_score, contributions_details = check_contributions_completeness(soup)
    completeness_scores['contributions'] = contributions_score
    detailed_results['contributions'] = contributions_details
    
    # 5. 实验设置检查 (Experiments Setup)
    experiments_setup_score, experiments_setup_details = check_experiments_setup_completeness(soup)
    completeness_scores['experiments_setup'] = experiments_setup_score
    detailed_results['experiments_setup'] = experiments_setup_details
    
    # 6. 实验结果检查 (Experiments Results)
    experiments_results_score, experiments_results_details = check_experiments_results_completeness(soup)
    completeness_scores['experiments_results'] = experiments_results_score
    detailed_results['experiments_results'] = experiments_results_details
    
    # 7. 引文格式检查 (Citation Format)
    citation_score, citation_details = check_citation_completeness(soup)
    completeness_scores['citation'] = citation_score
    detailed_results['citation'] = citation_details
    
    # 8. 视频/演示检查 (Video/Demo)
    video_score, video_details = check_video_demo_completeness(soup)
    completeness_scores['video_demo'] = video_score
    detailed_results['video_demo'] = video_details
    
    # 计算总体完整性分数
    overall_completeness = sum(completeness_scores.values()) / len(completeness_scores)
    
    # 构建最终结果
    result = {
        'overall_completeness': overall_completeness,
        'component_scores': completeness_scores,
        'detailed_results': detailed_results,
        'summary': {
            'total_components': len(completeness_scores),
            'components_present': sum(1 for score in completeness_scores.values() if score > 0),
            'completeness_percentage': overall_completeness * 100
        }
    }
    

    
    return result

def check_title_completeness(soup, paper_name):

    score = 0
    details = {'found': False, 'title_text': '', 'issues': []}
    
    # 检查 <title> 标签
    title_tag = soup.find('title')
    title_text = title_tag.get_text().strip() if title_tag else ""
    if title_text:
        details['title_text'] = title_text
        if len(title_text) > 5:
            score += 0.5
            details['found'] = True
    
    # 检查 <h1> 标签
    h1_tag = soup.find('h1')
    h1_text = h1_tag.get_text().strip() if h1_tag else ""
    if h1_text and len(h1_text) > 5:
        score += 0.5
        details['found'] = True
    
    # 检查是否包含论文名称关键词（空串兜底）
    paper_name_lower = paper_name.lower().replace('_', ' ')
    if (paper_name_lower in (title_text or "").lower()) or (paper_name_lower in (h1_text or "").lower()):
        score += 0.5
    
    if score == 0:
        details['issues'].append('COULD NOT FIND VALID PAPER TITLE')
    
    return min(score, 1.0), details

def check_authors_completeness(soup):

    score = 0
    details = {'found': False, 'authors_count': 0, 'linked_authors': 0, 'issues': []}
    
    # 查找作者相关信息
    author_patterns = [
        r'author[s]?', r'by', r'contributors?', r'team',
        r'研究人员', r'作者', r'贡献者'
    ]
    
    authors_found = []
    linked_authors = 0
    
    # 检查各种可能的作者标签
    for pattern in author_patterns:
        # 查找包含作者关键词的标签
        elements = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
        for element in elements:
            parent = element.parent
            if parent:
                # 查找相邻的链接
                links = parent.find_all('a')
                for link in links:
                    link_text = link.get_text().strip()
                    if link_text and len(link_text) > 1:
                        authors_found.append(link_text)
                        if link.get('href') and link.get('href') != '#':
                            linked_authors += 1
    
    
    details['authors_count'] = len(authors_found)
    details['linked_authors'] = linked_authors
    
    # 评分逻辑
    if len(authors_found) > 0:
        score += 0.5
        details['found'] = True
    
    if linked_authors > 0:
        score += 0.5
    
    if score == 0:
        details['issues'].append('COULD NOT FIND VALID AUTHORS')
    
    return min(score, 1.0), details

def check_abstract_completeness(soup):
    """检查摘要完整性"""
    score = 0
    details = {'found': False, 'abstract_text': '', 'issues': []}
    
    # 查找摘要相关内容
    abstract_patterns = [
        r'abstract', r'summary', r'overview', r'introduction',
        r'摘要', r'概述', r'简介'
    ]
    
    abstract_text = ""
    
    for pattern in abstract_patterns:
        # 查找标题
        headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for header in headers:
            if re.search(pattern, header.get_text(), re.IGNORECASE):

                content = get_content_after_header(header)
                if content and len(content) > 50:
                    score += 1.0
                    details['found'] = True
                    details['abstract_text'] = content[:200] + "..." if len(content) > 200 else content
                    break
        
        if details['found']:
            break
    
    if score == 0:
        details['issues'].append('COULD NOT FIND VALID ABSTRACT')
    
    return min(score, 1.0), details

def get_content_after_header(header):

    content = ""
    current = header
    
    # 向上查找，找到包含标题的section或div容器
    container = header.find_parent(['section', 'div', 'main'])
    if not container:

        current = header
    

    while current:

        if current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] and current != header:
            break

        if current.name in ['p', 'span', 'div'] and current != header:
            text = current.get_text().strip()
            if text:
                content += text + " "
        
        # 移动到下一个元素
        current = current.find_next_sibling()
        
        # 如果到达容器末尾，停止
        if current and container and current == container.find_next_sibling():
            break
    
    return content.strip()

def check_contributions_completeness(soup):

    score = 0
    details = {'found': False, 'contributions_count': 0, 'issues': []}
    
    # 查找贡献相关内容
    contribution_patterns = [
        r'contribution[s]?', r'key\s+contribution[s]?', r'novelty',
        r'贡献', r'创新点', r'主要贡献'
    ]
    
    contributions_found = []
    
    for pattern in contribution_patterns:
        # 查找包含贡献关键词的标签
        elements = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
        for element in elements:
            parent = element.parent
            if parent:
                # 查找列表项
                list_items = parent.find_all(['li', 'p'])
                for item in list_items:
                    item_text = item.get_text().strip()
                    if item_text and len(item_text) > 10:
                        contributions_found.append(item_text)
    
    details['contributions_count'] = len(contributions_found)
    
    if len(contributions_found) > 0:
        score += 1.0
        details['found'] = True
    
    if score == 0:
        details['issues'].append('未找到核心贡献列表')
    
    return min(score, 1.0), details

def check_experiments_setup_completeness(soup):

    score = 0
    details = {'found': False, 'setup_elements': [], 'issues': []}
    
    # 查找实验设置相关内容
    setup_patterns = [
        r'experiment[s]?', r'setup', r'dataset[s]?', r'parameter[s]?',
        r'实验', r'设置', r'数据集', r'参数'
    ]
    
    setup_elements = []
    
    for pattern in setup_patterns:
        elements = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
        for element in elements:
            parent = element.parent
            if parent:
                # 查找相关描述
                desc_elements = parent.find_all(['p', 'li', 'div'])
                for desc in desc_elements:
                    desc_text = desc.get_text().strip()
                    if desc_text and len(desc_text) > 20:
                        setup_elements.append(desc_text)
    
    details['setup_elements'] = setup_elements[:3]  # 只保存前3个
    
    if len(setup_elements) > 0:
        score += 1.0
        details['found'] = True
    
    if score == 0:
        details['issues'].append('COULD NOT FIND VALID EXPERIMENT SETUP')
    
    return min(score, 1.0), details

def check_experiments_results_completeness(soup):
    """检查实验结果完整性"""
    score = 0
    details = {'found': False, 'images_count': 0, 'results_text': '', 'issues': []}
    
    # 查找图片
    images = soup.find_all('img')
    details['images_count'] = len(images)
    
    if len(images) > 0:
        score += 0.5
    
    # 查找结果相关内容
    results_patterns = [
        r'result[s]?', r'performance', r'evaluation', r'accuracy',
        r'结果', r'性能', r'评估', r'准确率'
    ]
    
    results_text = ""
    
    for pattern in results_patterns:
        elements = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
        for element in elements:
            parent = element.parent
            if parent:
                # 获取相关文本
                text_elements = parent.find_all(['p', 'li', 'div'])
                for text_elem in text_elements:
                    text = text_elem.get_text().strip()
                    if text and len(text) > 20:
                        results_text += text + " "
    
    details['results_text'] = results_text[:200] + "..." if len(results_text) > 200 else results_text
    
    if len(results_text) > 50:
        score += 0.5
    
    if score == 0:
        details['issues'].append('COUYLD NOT FIND VALID EXPERIMENT RESULTS')
    
    return min(score, 1.0), details

def check_citation_completeness(soup):
    score = 0
    details = {'found': False, 'citation_text': '', 'issues': []}
    
    # 查找引文相关内容
    citation_patterns = [
        r'@inproceedings', r'@article', r'@misc', r'bibtex',
        r'citation', r'cite', r'reference',
        r'引用', r'参考文献'
    ]
    
    citation_text = ""
    
    for pattern in citation_patterns:
        # 查找代码块
        code_blocks = soup.find_all(['pre', 'code'])
        for code in code_blocks:
            code_text = code.get_text()
            if re.search(pattern, code_text, re.IGNORECASE):
                citation_text = code_text
                score += 1.0
                details['found'] = True
                details['citation_text'] = citation_text[:200] + "..." if len(citation_text) > 200 else citation_text
                break
    
    if score == 0:
        details['issues'].append('COULD NOT FIND VALID CITATION')
    
    return min(score, 1.0), details

def check_video_demo_completeness(soup):
    """检查视频/演示完整性"""
    score = 0
    details = {'found': False, 'video_elements': [], 'issues': []}
    

    video_elements = []
    

    videos = soup.find_all('video')
    if videos:
        video_elements.extend(['<video> tag found'])
        score += 0.5

    iframes = soup.find_all('iframe')
    for iframe in iframes:
        src = iframe.get('src', '')
        if 'youtube' in src.lower() or 'vimeo' in src.lower():
            video_elements.append(f'Video iframe: {src}')
            score += 0.5
    
    # 查找视频相关文本
    video_patterns = [
        r'video', r'demo', r'demonstration', r'presentation',
        r'视频', r'演示', r'展示'
    ]
    
    for pattern in video_patterns:
        elements = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
        if elements:
            video_elements.append(f'Video-related text: {pattern}')
            score += 0.5
            break
    
    details['video_elements'] = video_elements
    
    if score == 0:
        details['issues'].append('未找到视频或演示内容')
    
    return min(score, 1.0), details

def evaluate_website_connectivity(html_file_path, paper_name):
    from bs4 import BeautifulSoup
    import re
    from urllib.parse import urlparse, urljoin
    
    # 读取HTML文件
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # 解析HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 初始化评估结果
    connectivity_scores = {}
    detailed_results = {}
    
    # 1. 代码链接检查 (Code Links)
    code_score, code_details = check_code_connectivity(soup)
    connectivity_scores['code_links'] = code_score
    detailed_results['code_links'] = code_details
    
    # 2. 论文链接检查 (Paper Links)
    paper_score, paper_details = check_paper_connectivity(soup)
    connectivity_scores['paper_links'] = paper_score
    detailed_results['paper_links'] = paper_details
    
    # 3. 作者主页链接检查 (Author Homepage Links)
    author_score, author_details = check_author_connectivity(soup)
    connectivity_scores['author_homepage'] = author_score
    detailed_results['author_homepage'] = author_details
    
    # 4. 实验室主页链接检查 (Lab Homepage Links)
    lab_score, lab_details = check_lab_connectivity(soup)
    connectivity_scores['lab_homepage'] = lab_score
    detailed_results['lab_homepage'] = lab_details
    
    # 5. 项目/数据链接检查 (Project/Data Links)
    project_score, project_details = check_project_data_connectivity(soup)
    connectivity_scores['project_data'] = project_score
    detailed_results['project_data'] = project_details
    
    # 6. 相关工作链接检查 (Related Work Links)
    related_score, related_details = check_related_work_connectivity(soup)
    connectivity_scores['related_work'] = related_score
    detailed_results['related_work'] = related_details
    
    # 7. 总体链接质量检查 (Overall Link Quality)
    overall_link_score, overall_link_details = check_overall_link_quality(soup)
    connectivity_scores['overall_link_quality'] = overall_link_score
    detailed_results['overall_link_quality'] = overall_link_details
    
    # 计算总体连通性分数
    overall_connectivity = sum(connectivity_scores.values()) / len(connectivity_scores)
    
    # 构建最终结果
    result = {
        'overall_connectivity': overall_connectivity,
        'component_scores': connectivity_scores,
        'detailed_results': detailed_results,
        'summary': {
            'total_components': len(connectivity_scores),
            'components_present': sum(1 for score in connectivity_scores.values() if score > 0),
            'connectivity_percentage': overall_connectivity * 100
        }
    }
    
    
    return result

def check_code_connectivity(soup):
    score = 0
    details = {'found': False, 'code_links': [], 'issues': []}
    
    # 代码托管平台关键词
    code_platforms = [
        'github.com', 'gitlab.com', 'bitbucket.org', 'sourceforge.net',
        'code.google.com', 'git.code.tencent.com', 'gitee.com'
    ]
    
    # 代码相关关键词
    code_keywords = [
        'code', 'implementation', 'source', 'repository', 'download',
        '代码', '实现', '源码', '仓库', '下载'
    ]
    
    code_links = []
    
    # 查找所有链接
    links = soup.find_all('a', href=True)
    for link in links:
        href = link.get('href', '').lower()
        link_text = link.get_text().strip().lower()
        
        # 检查是否指向代码托管平台
        for platform in code_platforms:
            if platform in href:
                code_links.append({
                    'platform': platform,
                    'url': href,
                    'text': link.get_text().strip()
                })
                score += 0.5
                break
        
        # 检查链接文本是否包含代码关键词
        for keyword in code_keywords:
            if keyword in link_text:
                if href and href != '#' and href != 'javascript:void(0)':
                    code_links.append({
                        'platform': 'text_keyword',
                        'url': href,
                        'text': link.get_text().strip()
                    })
                    score += 0.5
                    break
    
    details['code_links'] = code_links
    details['found'] = len(code_links) > 0
    
    if score == 0:
        details['issues'].append('COULDNOT FIND CODE LINKS')
    
    return min(score, 1.0), details

def check_paper_connectivity(soup):
    """检查论文链接连通性"""
    score = 0
    details = {'found': False, 'paper_links': [], 'issues': []}
    
    # 论文平台关键词
    paper_platforms = [
        'arxiv.org', 'papers.nips.cc', 'proceedings.mlr.press', 'openreview.net',
        'ieeexplore.ieee.org', 'dl.acm.org', 'link.springer.com', 'sciencedirect.com',
        'researchgate.net', 'scholar.google.com', 'semanticscholar.org'
    ]
    
    # 论文相关关键词
    paper_keywords = [
        'paper', 'pdf', 'download', 'read', 'view', 'full paper',
        '论文', 'PDF', '下载', '阅读', '查看', '完整论文'
    ]
    
    paper_links = []
    
    # 查找所有链接
    links = soup.find_all('a', href=True)
    for link in links:
        href = link.get('href', '').lower()
        link_text = link.get_text().strip().lower()
        
        # 检查是否指向论文平台
        for platform in paper_platforms:
            if platform in href:
                paper_links.append({
                    'platform': platform,
                    'url': href,
                    'text': link.get_text().strip()
                })
                score += 0.5
                break
        
        # 检查链接文本是否包含论文关键词
        for keyword in paper_keywords:
            if keyword in link_text:
                if href and href != '#' and href != 'javascript:void(0)':
                    paper_links.append({
                        'platform': 'text_keyword',
                        'url': href,
                        'text': link.get_text().strip()
                    })
                    score += 0.5
                    break
    
    details['paper_links'] = paper_links
    details['found'] = len(paper_links) > 0
    
    if score == 0:
        details['issues'].append('未找到论文相关链接')
    
    return min(score, 1.0), details

def check_author_connectivity(soup):

    score = 0
    details = {'found': False, 'author_links': [], 'issues': []}
    
    # 作者相关关键词
    author_keywords = [
        'author', 'researcher', 'professor', 'phd', 'student',
        '作者', '研究员', '教授', '博士', '学生'
    ]
    
    # 学术主页关键词
    academic_keywords = [
        'homepage', 'personal', 'profile', 'bio', 'about',
        '主页', '个人', '简介', '简历', '关于'
    ]
    
    author_links = []
    
    # 查找所有链接
    links = soup.find_all('a', href=True)
    for link in links:
        href = link.get('href', '').lower()
        link_text = link.get_text().strip().lower()
        
        # 检查链接文本是否包含作者关键词
        is_author_link = False
        for keyword in author_keywords:
            if keyword in link_text:
                is_author_link = True
                break
        
        # 检查链接文本是否包含学术主页关键词
        is_academic_link = False
        for keyword in academic_keywords:
            if keyword in link_text:
                is_academic_link = True
                break
        
        # 检查链接是否指向个人主页
        if is_author_link or is_academic_link:
            if href and href != '#' and href != 'javascript:void(0)':
                # 检查是否是有效的个人主页链接
                if any(domain in href for domain in ['.edu', '.ac.', 'university', 'institute', 'lab']):
                    author_links.append({
                        'type': 'academic_homepage',
                        'url': href,
                        'text': link.get_text().strip()
                    })
                    score += 0.5
                else:
                    author_links.append({
                        'type': 'personal_link',
                        'url': href,
                        'text': link.get_text().strip()
                    })
                    score += 0.3
    
    details['author_links'] = author_links
    details['found'] = len(author_links) > 0
    
    if score == 0:
        details['issues'].append('COULDNOT FIND AUTHOR LINKS')
    
    return min(score, 1.0), details

def check_lab_connectivity(soup):

    score = 0
    details = {'found': False, 'lab_links': [], 'issues': []}
    
    # 实验室相关关键词
    lab_keywords = [
        'lab', 'laboratory', 'group', 'team', 'research group', 'research team',
        '实验室', '研究组', '团队', '研究团队'
    ]
    
    # 机构相关关键词
    institution_keywords = [
        'university', 'institute', 'college', 'school', 'department',
        '大学', '学院', '研究所', '学校', '系'
    ]
    
    lab_links = []
    
    # 查找所有链接
    links = soup.find_all('a', href=True)
    for link in links:
        href = link.get('href', '').lower()
        link_text = link.get_text().strip().lower()
        
  
        is_lab_link = False
        for keyword in lab_keywords:
            if keyword in link_text:
                is_lab_link = True
                break
        

        is_institution_link = False
        for keyword in institution_keywords:
            if keyword in link_text:
                is_institution_link = True
                break
        
        if is_lab_link or is_institution_link:
            if href and href != '#' and href != 'javascript:void(0)':
                lab_links.append({
                    'type': 'lab_or_institution',
                    'url': href,
                    'text': link.get_text().strip()
                })
                score += 1.0
    
    details['lab_links'] = lab_links
    details['found'] = len(lab_links) > 0
    
    if score == 0:
        details['issues'].append('COULD NOT FIND LAB OR INSTITUTION LINKS')
    
    return min(score, 1.0), details

def check_project_data_connectivity(soup):
    score = 0
    details = {'found': False, 'project_links': [], 'issues': []}
    
    # 项目/数据相关关键词
    project_keywords = [
        'project', 'dataset', 'data', 'demo', 'download', 'resource',
        '项目', '数据集', '数据', '演示', '下载', '资源'
    ]
    

    data_platforms = [
        'kaggle.com', 'data.world', 'zenodo.org', 'figshare.com',
        'datadryad.org', 'dataverse.org', 'opendata.aws'
    ]
    
    project_links = []
    
    # 查找所有链接
    links = soup.find_all('a', href=True)
    for link in links:
        href = link.get('href', '').lower()
        link_text = link.get_text().strip().lower()
        
 
        for platform in data_platforms:
            if platform in href:
                project_links.append({
                    'platform': platform,
                    'url': href,
                    'text': link.get_text().strip()
                })
                score += 0.5
                break
        

        for keyword in project_keywords:
            if keyword in link_text:
                if href and href != '#' and href != 'javascript:void(0)':
                    project_links.append({
                        'platform': 'text_keyword',
                        'url': href,
                        'text': link.get_text().strip()
                    })
                    score += 0.5
                    break
    
    details['project_links'] = project_links
    details['found'] = len(project_links) > 0
    
    if score == 0:
        details['issues'].append('COULD NOT FIND PROJECT OR DATA LINKS')
    
    return min(score, 1.0), details

def check_related_work_connectivity(soup):
    score = 0
    details = {'found': False, 'related_links': [], 'issues': []}
    
    # 相关工作相关关键词
    related_keywords = [
        'related work', 'previous work', 'prior work', 'literature', 'reference',
        '相关工作', '先前工作', '文献', '参考文献'
    ]
    
    # 引用相关关键词
    citation_keywords = [
        'cite', 'citation', 'reference', 'bibtex', 'doi',
        '引用', '引文', '参考文献', 'DOI'
    ]
    
    related_links = []
    
    # 查找所有链接
    links = soup.find_all('a', href=True)
    for link in links:
        href = link.get('href', '').lower()
        link_text = link.get_text().strip().lower()
        
        # 检查链接文本是否包含相关工作关键词
        for keyword in related_keywords:
            if keyword in link_text:
                if href and href != '#' and href != 'javascript:void(0)':
                    related_links.append({
                        'type': 'related_work',
                        'url': href,
                        'text': link.get_text().strip()
                    })
                    score += 0.5
                    break
        
        # 检查链接文本是否包含引用关键词
        for keyword in citation_keywords:
            if keyword in link_text:
                if href and href != '#' and href != 'javascript:void(0)':
                    related_links.append({
                        'type': 'citation',
                        'url': href,
                        'text': link.get_text().strip()
                    })
                    score += 0.5
                    break
    
    details['related_links'] = related_links
    details['found'] = len(related_links) > 0
    
    if score == 0:
        details['issues'].append('')
    
    return min(score, 1.0), details

def check_overall_link_quality(soup):
    score = 0
    details = {'total_links': 0, 'valid_links': 0, 'external_links': 0, 'issues': []}
    
    # 查找所有链接
    links = soup.find_all('a', href=True)
    total_links = len(links)
    valid_links = 0
    external_links = 0
    
    for link in links:
        href = link.get('href', '')
        
        # 检查链接是否有效
        if href and href != '#' and href != 'javascript:void(0)':
            valid_links += 1
            
            # 检查是否是外部链接
            if href.startswith('http'):
                external_links += 1
    
    details['total_links'] = total_links
    details['valid_links'] = valid_links
    details['external_links'] = external_links
    
    # 评分逻辑
    if total_links > 0:
        # 有效链接比例
        valid_ratio = valid_links / total_links
        if valid_ratio >= 0.8:
            score += 0.5
        elif valid_ratio >= 0.6:
            score += 0.3
        
        # 外部链接数量
        if external_links >= 5:
            score += 0.5
        elif external_links >= 3:
            score += 0.3
        elif external_links >= 1:
            score += 0.1
    
    if score == 0:
        details['issues'].append('THE  WEBSITE DOES NOT HAVE ENOUGH LINKS')
    
    return min(score, 1.0), details

def evaluate_website_interactivity(html_file_path, css_file_path=None, js_file_path=None):

    from bs4 import BeautifulSoup
    import re
    
    print(f'🎮 {html_file_path}')
    
    # 读取HTML文件
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # 解析HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 初始化评估结果
    interactivity_scores = {}
    detailed_results = {}
    
    # 1. 技术栈CSS/JS存在性检查
    tech_stack_score, tech_stack_details = check_tech_stack_existence(
        soup, css_file_path, js_file_path
    )
    interactivity_scores['tech_stack'] = tech_stack_score
    detailed_results['tech_stack'] = tech_stack_details
    
    # 2. 前端框架检测
    framework_score, framework_details = check_frontend_frameworks(soup, css_file_path, js_file_path)
    interactivity_scores['frameworks'] = framework_score
    detailed_results['frameworks'] = framework_details
    
    # 3. 动态元素计数
    dynamic_elements_score, dynamic_elements_details = check_dynamic_elements(soup)
    interactivity_scores['dynamic_elements'] = dynamic_elements_score
    detailed_results['dynamic_elements'] = dynamic_elements_details
    
    # 4. 动态效果检查
    dynamic_effects_score, dynamic_effects_details = check_dynamic_effects(css_file_path)
    interactivity_scores['dynamic_effects'] = dynamic_effects_score
    detailed_results['dynamic_effects'] = dynamic_effects_details
    
    # 5. 滚动深度和交互性
    scroll_interaction_score, scroll_interaction_details = check_scroll_interaction(soup, js_file_path)
    interactivity_scores['scroll_interaction'] = scroll_interaction_score
    detailed_results['scroll_interaction'] = scroll_interaction_details
    
    # 6. 响应式设计检查
    responsive_score, responsive_details = check_responsive_design(css_file_path)
    interactivity_scores['responsive_design'] = responsive_score
    detailed_results['responsive_design'] = responsive_details
    
    # 计算总体交互性分数
    overall_interactivity = sum(interactivity_scores.values()) / len(interactivity_scores)
    
    # 构建最终结果
    result = {
        'overall_interactivity': overall_interactivity,
        'component_scores': interactivity_scores,
        'detailed_results': detailed_results,
        'summary': {
            'total_components': len(interactivity_scores),
            'components_present': sum(1 for score in interactivity_scores.values() if score > 0),
            'interactivity_percentage': overall_interactivity * 100
        }
    }
    
    
    return result

def check_tech_stack_existence(soup, css_file_path, js_file_path):
    score = 0
    details = {
        'css_linked': False, 'css_file_exists': False, 'css_content_length': 0,
        'js_linked': False, 'js_file_exists': False, 'js_content_length': 0,
        'issues': []
    }
    
    # 检查CSS链接
    css_links = soup.find_all('link', rel='stylesheet')
    if css_links:
        details['css_linked'] = True
        score += 0.3
    
    # 检查CSS文件是否存在且有内容
    if css_file_path and os.path.exists(css_file_path):
        details['css_file_exists'] = True
        try:
            with open(css_file_path, 'r', encoding='utf-8') as f:
                css_content = f.read()
                details['css_content_length'] = len(css_content)
                if len(css_content.strip()) > 100:  # 至少100字符的CSS内容
                    score += 0.3
                else:
                    details['issues'].append('')
        except Exception as e:
                details['issues'].append(f'{e}')
    else:
            details['issues'].append('')
    
    # 检查JavaScript链接
    js_links = soup.find_all('script', src=True)
    inline_js = soup.find_all('script', src=False)
    
    if js_links or inline_js:
        details['js_linked'] = True
        score += 0.3
    
    # 检查JavaScript文件是否存在且有内容
    if js_file_path and os.path.exists(js_file_path):
        details['js_file_exists'] = True
        try:
            with open(js_file_path, 'r', encoding='utf-8') as f:
                js_content = f.read()
                details['js_content_length'] = len(js_content)
                if len(js_content.strip()) > 100:  # 至少100字符的JS内容
                    score += 0.3
                else:
                    details['issues'].append('')
        except Exception as e:
                details['issues'].append(f' {e}')
    else:
            details['issues'].append('')
    
    if score == 0:
        details['issues'].append('')
    
    return min(score, 1.0), details

def check_frontend_frameworks(soup, css_file_path, js_file_path):
    """检测前端框架和UI库"""
    score = 0
    details = {
        'frameworks_detected': [], 'ui_libraries': [], 'css_frameworks': [],
        'issues': []
    }
    
    # 检查HTML中的框架标识
    html_content = str(soup).lower()
    
    # React检测
    if 'react' in html_content or 'data-react' in html_content:
        details['frameworks_detected'].append('React')
        score += 0.3
    
    # Vue检测
    if 'vue' in html_content or 'data-v-' in html_content:
        details['frameworks_detected'].append('Vue')
        score += 0.3
    
    # Angular检测
    if 'angular' in html_content or 'ng-' in html_content:
        details['frameworks_detected'].append('Angular')
        score += 0.3
    
    # 检查CSS文件中的框架
    if css_file_path and os.path.exists(css_file_path):
        try:
            with open(css_file_path, 'r', encoding='utf-8') as f:
                css_content = f.read().lower()
                
                # Bootstrap检测
                if 'bootstrap' in css_content or '.container' in css_content or '.row' in css_content:
                    details['ui_libraries'].append('Bootstrap')
                    score += 0.2
                
                # Tailwind CSS检测
                if 'tailwind' in css_content or 'tw-' in css_content:
                    details['ui_libraries'].append('Tailwind CSS')
                    score += 0.2
                
                # Foundation检测
                if 'foundation' in css_content or '.foundation' in css_content:
                    details['ui_libraries'].append('Foundation')
                    score += 0.2
                
                # Material Design检测
                if 'material' in css_content or 'mdl-' in css_content:
                    details['ui_libraries'].append('Material Design')
                    score += 0.2
                
                # Bulma检测
                if 'bulma' in css_content or '.bulma' in css_content:
                    details['ui_libraries'].append('Bulma')
                    score += 0.2
        except Exception as e:
            details['issues'].append(f'CSS框架检测错误: {e}')
    
    # 检查JavaScript文件中的框架
    if js_file_path and os.path.exists(js_file_path):
        try:
            with open(js_file_path, 'r', encoding='utf-8') as f:
                js_content = f.read().lower()
                
                # jQuery检测
                if 'jquery' in js_content or '$(' in js_content:
                    details['ui_libraries'].append('jQuery')
                    score += 0.2
                
                # D3.js检测
                if 'd3' in js_content or 'd3.' in js_content:
                    details['ui_libraries'].append('D3.js')
                    score += 0.2
                
                # Chart.js检测
                if 'chart' in js_content or 'chartjs' in js_content:
                    details['ui_libraries'].append('Chart.js')
                    score += 0.2
        except Exception as e:
            details['issues'].append(f'{e}')
    
    if score == 0:
        details['issues'].append('')
    
    return min(score, 1.0), details



def check_dynamic_effects(css_file_path):
    """检查CSS中是否存在动态效果"""
    score = 0
    details = {
        'hover_effects': False, 'animations': False, 'transitions': False,
        'keyframes': False, 'transforms': False, 'issues': []
    }
    
    if not css_file_path or not os.path.exists(css_file_path):
        details['issues'].append('CSS文件不存在')
        return score, details
    
    try:
        with open(css_file_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
            
            # 检查:hover伪类
            if ':hover' in css_content:
                details['hover_effects'] = True
                score += 0.2
            
            # 检查动画
            if 'animation:' in css_content or 'animation-name:' in css_content:
                details['animations'] = True
                score += 0.2
            
            # 检查过渡效果
            if 'transition:' in css_content:
                details['transitions'] = True
                score += 0.2
            
            # 检查关键帧动画
            if '@keyframes' in css_content:
                details['keyframes'] = True
                score += 0.2
            
            # 检查变换效果
            if 'transform:' in css_content or 'transform:' in css_content:
                details['transforms'] = True
                score += 0.2
            
    except Exception as e:
        details['issues'].append(f' {e}')
    
    if score == 0:
        details['issues'].append('')
    
    return min(score, 1.0), details

def check_scroll_interaction(soup, js_file_path):
    score = 0
    details = {
        'scroll_events': False, 'scroll_indicators': False, 'smooth_scroll': False,
        'scroll_animations': False, 'issues': []
    }
    
    # 检查HTML中的滚动相关元素
    html_content = str(soup).lower()
    
    # 检查滚动指示器
    if 'scroll' in html_content or 'progress' in html_content:
        details['scroll_indicators'] = True
        score += 0.2
    
    # 检查JavaScript文件中的滚动交互
    if js_file_path and os.path.exists(js_file_path):
        try:
            with open(js_file_path, 'r', encoding='utf-8') as f:
                js_content = f.read().lower()
                
                # 检查滚动事件监听器
                if 'scroll' in js_content and ('addEventListener' in js_content or 'onscroll' in js_content):
                    details['scroll_events'] = True
                    score += 0.3
                
                # 检查平滑滚动
                if 'smooth' in js_content and 'scroll' in js_content:
                    details['smooth_scroll'] = True
                    score += 0.3
                
                # 检查滚动动画
                if 'scroll' in js_content and ('animate' in js_content or 'animation' in js_content):
                    details['scroll_animations'] = True
                    score += 0.2
                
        except Exception as e:
            details['issues'].append(f': {e}')
    
    if score == 0:
        details['issues'].append('')
    
    return min(score, 1.0), details

def check_responsive_design(css_file_path):
    """检查响应式设计"""
    score = 0
    details = {
        'media_queries': False, 'flexbox': False, 'grid': False, 'viewport': False,
        'issues': []
    }
    
    if not css_file_path or not os.path.exists(css_file_path):
        details['issues'].append('')
        return score, details
    
    try:
        with open(css_file_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
            
            # 检查媒体查询
            if '@media' in css_content:
                details['media_queries'] = True
                score += 0.3
            
            # 检查Flexbox
            if 'display: flex' in css_content or 'display:flex' in css_content:
                details['flexbox'] = True
                score += 0.3
            
            # 检查CSS Grid
            if 'display: grid' in css_content or 'display:grid' in css_content:
                details['grid'] = True
                score += 0.3
            
            # 检查视口设置
            if 'viewport' in css_content or 'width=device-width' in css_content:
                details['viewport'] = True
                score += 0.1
            
    except Exception as e:
        details['issues'].append(f'{e}')
    
    if score == 0:
        details['issues'].append('')
    
    return min(score, 1.0), details

def eval_website_completeness_llm(website_image_list, agent_config, html_file_path, paper_name):

    from bs4 import BeautifulSoup
    import tiktoken
    
    # 创建4o模型
    judge_model = ModelFactory.create(
        model_platform=agent_config['model_platform'],
        model_type=agent_config['model_type'],
        model_config_dict=agent_config['model_config'],
    )
    
    # 加载completeness prompt模板
    with open(get_template_path("website_completeness.yaml"), "r") as f:
        judge_config = yaml.safe_load(f)
    
    # 创建agent
    judge_sys_msg = judge_config['system_prompt']
    judge_agent = ChatAgent(
        system_message=judge_sys_msg,
        model=judge_model,
        message_window_size=None,
    )
    

    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # 解析HTML获取关键信息
    soup = BeautifulSoup(html_content, 'html.parser')
    

    def count_tokens(text, model="gpt-4o"):
        """计算文本的token数量"""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except:
   
            return len(text) // 4  # 粗略估算：1 token ≈ 4 字符
    

    prompt_template = judge_config["template"]
    prompt_template_tokens = count_tokens(prompt_template)
    
    # 预留其他内容的token空间
    reserved_tokens = 1000 
    
    available_tokens = 15000 - prompt_template_tokens - reserved_tokens
    
    # 处理HTML内容
    if count_tokens(html_content) <= available_tokens:
        html_for_prompt = html_content
        html_truncated = False
    else:
        html_for_prompt = truncate_html_by_tokens(html_content, available_tokens)
        print("HTML已截断")
        html_truncated = True
    
    # 构建评估提示
    evaluation_prompt = f"""
    {judge_config["template"]}
    
    {paper_name}
    
    HTML内容{'（已截断以适应token限制）' if html_truncated else ''}:
    {html_for_prompt}
    
    """
    

    judge_message = BaseMessage.make_user_message(
        role_name="User",
        content=evaluation_prompt,
        image_list=website_image_list,
    )
    
    # 调用4o模型生成输出
    response = judge_agent.step(judge_message)
    result = get_json_from_response(response.msgs[0].content)

    detailed_evaluation = result.get('detailed_evaluation', {})
    criteria_summary = result.get('criteria_summary', {})

    criteria_scores = {}
    for criterion_name, criterion_data in detailed_evaluation.items():
        if isinstance(criterion_data, dict) and 'found' in criterion_data:
            criteria_scores[criterion_name] = {
                'found': criterion_data.get('found', False),
                'details': criterion_data.get('details', ''),
                'score': 1.0 if criterion_data.get('found', False) else 0.0
            }
    
    return {
        'completeness_score': result.get('score', 0),  # 1-5分
        'reason': result.get('reason', ''),
        'evaluation_method': 'LLM-based (HTML + Image)',
        'model_used': '4o',
        'detailed_evaluation': detailed_evaluation,
        'criteria_summary': {
            'total_criteria': criteria_summary.get('total_criteria', 8),
            'criteria_met': criteria_summary.get('criteria_met', 0),
            'criteria_missing': criteria_summary.get('criteria_missing', 8)
        },
        'criteria_scores': criteria_scores,
        'html_analysis': {
            'title_present': bool(soup.find('title')),
            'html_truncated': html_truncated,
            'total_tokens_used': count_tokens(evaluation_prompt),
            'html_tokens_used': count_tokens(html_for_prompt)
        }
    }

def eval_website_connectivity_llm(website_image_list, agent_config, html_file_path, paper_name):

    from bs4 import BeautifulSoup
    import tiktoken
    
    # 创建4o模型
    judge_model = ModelFactory.create(
        model_platform=agent_config['model_platform'],
        model_type=agent_config['model_type'],
        model_config_dict=agent_config['model_config'],
    )
    
    # 加载connectivity prompt模板
    with open(get_template_path("website_connectivity.yaml"), "r") as f:
        judge_config = yaml.safe_load(f)
    
    # 创建agent
    judge_sys_msg = judge_config['system_prompt']
    judge_agent = ChatAgent(
        system_message=judge_sys_msg,
        model=judge_model,
        message_window_size=None,
    )
    
    # 读取HTML内容
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # 解析HTML获取链接信息
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 分析链接
    all_links = soup.find_all('a', href=True)
    external_links = [link for link in all_links if link.get('href', '').startswith('http')]
    internal_links = [link for link in all_links if not link.get('href', '').startswith('http')]
    
    # 计算token数量并处理HTML内容
    def count_tokens(text, model="gpt-4o"):
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except:

            return len(text) // 4  # 粗略估算：1 token ≈ 4 字符
    
    # 计算各部分token数量
    prompt_template = judge_config["template"]
    prompt_template_tokens = count_tokens(prompt_template)
    
    # 预留其他内容的token空间
    reserved_tokens = 1000  #
    
    # 计算可用于HTML的token数量
    available_tokens = 15000 - prompt_template_tokens - reserved_tokens
    
    # 处理HTML内容
    if count_tokens(html_content) <= available_tokens:
        # HTML内容在限制范围内，使用完整内容
        html_for_prompt = html_content
        html_truncated = False
    else:
        # HTML内容超出限制，需要截断
        html_for_prompt = truncate_html_by_tokens(html_content, available_tokens)
        html_truncated = True
    
    # 构建评估提示
    evaluation_prompt = f"""
    {judge_config["template"]}
    
    论文名称: {paper_name}
    
    HTML内容{'（已截断以适应token限制）' if html_truncated else ''}:
    {html_for_prompt}
    
    """
    
    # 创建包含图像的消息
    judge_message = BaseMessage.make_user_message(
        role_name="User",
        content=evaluation_prompt,
        image_list=website_image_list,
    )
    
    # 调用4o模型生成输出
    response = judge_agent.step(judge_message)
    result = get_json_from_response(response.msgs[0].content)
    
    # 严格按照YAML模板的输出格式解析
    return {
        'connectivity_score': result.get('score', 0),  # 1-5分
        'reason': result.get('reason', ''),
        'evaluation_method': 'LLM-based (HTML + Image)',
        'model_used': '4o',
        'detailed_evaluation': result.get('detailed_evaluation', {}),
        'criteria_summary': result.get('criteria_summary', {}),
        'link_analysis': {
            'total_links': len(all_links),
            'external_links': len(external_links),
            'internal_links': len(internal_links),
            'external_link_examples': [link.get('href') for link in external_links[:5]]
        },
        'html_analysis': {
            'html_truncated': html_truncated,
            'total_tokens_used': count_tokens(evaluation_prompt),
            'html_tokens_used': count_tokens(html_for_prompt)
        }
    }

def eval_website_interactivity_llm(website_image_list,agent_config, html_file_path, css_file_path=None, js_file_path=None):
    """
    使用LLM评估网站交互性 - 基于HTML、CSS、JS和图像
    """
    from bs4 import BeautifulSoup
    import tiktoken
    
    # 创建4o模型
    judge_model = ModelFactory.create(
        model_platform=agent_config['model_platform'],
        model_type=agent_config['model_type'],
        model_config_dict=agent_config['model_config'],
    )
    
    # 加载interactivity prompt模板（基于当前文件所在目录构造绝对路径）
    _this_dir = os.path.dirname(__file__)
    _prompt_path = os.path.join(_this_dir, 'prompt_templates', 'website_interactivity.yaml')
    with open(_prompt_path, "r") as f:
        judge_config = yaml.safe_load(f)
    
    # 创建agent
    judge_sys_msg = judge_config['system_prompt']
    judge_agent = ChatAgent(
        system_message=judge_sys_msg,
        model=judge_model,
        message_window_size=None,
    )
    
    # 读取HTML内容
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # 解析HTML以便进行本地动态元素统计
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 读取CSS文件（如果存在）
    css_content = ""
    if css_file_path and os.path.exists(css_file_path):
        with open(css_file_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
    
    # 读取JS文件（如果存在）
    js_content = ""
    if js_file_path and os.path.exists(js_file_path):
        with open(js_file_path, 'r', encoding='utf-8') as f:
            js_content = f.read()
    
    # 计算token数量并处理HTML内容
    def count_tokens(text, model="gpt-4o"):
        """计算文本的token数量"""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except:
            # 如果tiktoken不可用，使用简单的字符数估算
            return len(text) // 4  # 粗略估算：1 token ≈ 4 字符
    
    # 计算各部分token数量
    prompt_template = judge_config["template"]
    prompt_template_tokens = count_tokens(prompt_template)
    
    # 预留其他内容的token空间
    reserved_tokens = 1000  # 为其他文本内容预留空间
    
    # 计算可用于HTML的token数量（CSS和JS不截断，完整输入）
    print("count_tokens(css_content):"+str(count_tokens(css_content)))
    print("count_tokens(js_content):"+str(count_tokens(js_content)))
    available_tokens = 15000 - prompt_template_tokens - reserved_tokens - count_tokens(css_content) - count_tokens(js_content)
    
    # 处理HTML内容
    if count_tokens(html_content) <= available_tokens:
        # HTML内容在限制范围内，使用完整内容
        html_for_prompt = html_content
        html_truncated = False
    else:
        print("超出限制了")
        # HTML内容超出限制，需要截断
        html_for_prompt = truncate_html_by_tokens(html_content, available_tokens)
        print("HTML已截断")
        html_truncated = True
    
    # 构建评估提示
    evaluation_prompt = f"""
    {judge_config["template"]}
    
    HTML内容{'（已截断以适应token限制）' if html_truncated else ''}:
    {html_for_prompt}
    
    CSS内容:
    {css_content if css_content else '未找到CSS文件'}
    
    JavaScript内容:
    {js_content if js_content else '未找到JavaScript文件'}
    
    请基于HTML结构、CSS样式、JavaScript功能和提供的图像截图来评估网站交互性。
    """
    
    judge_message = BaseMessage.make_user_message(
        role_name="User",
        content=evaluation_prompt,
        image_list=website_image_list,
    )
    
    # 调用模型生成输出
    response = judge_agent.step(judge_message)
    
    # 计算并打印token使用情况
    input_token, output_token = account_token(response)
    print(f"📊 网站交互性评估 - 输入token: {input_token}, 输出token: {output_token}, 总计: {input_token + output_token}")
    
    result = get_json_from_response(response.msgs[0].content)

    # 1) css和js的技术栈检查
    css_js_tech_stack = check_css_js_technology_stack(soup, css_content, js_content)

    css_js_score = 0 
    if css_js_tech_stack['found'] == True:
        css_js_score=1

    # 2) 基于 LLM 的 four-criteria 计分（detailed_evaluation）
    detailed = (result or {}).get('detailed_evaluation', {}) or {}
    llm_found_count = 0
    for key, val in detailed.items():
        try:
            if isinstance(val, dict) and bool(val.get('found')):
                llm_found_count += 1
        except Exception:
            pass

    # 0~1->1, 2->2, 3->3, 4->4, 5->5（最小1分）
    score_llm = max(1, min(5, llm_found_count+css_js_score))

    # 3) rulebase的动态规则检查
    details = {
        'interactive_elements': {},
        'total_count': 0,
        'issues': []
    }

    # 可按需调整的动态元素集合
    dynamic_elements = {
        'button': '按钮',
        'input': '输入框',
        'form': '表单',
        'select': '下拉选择框',
        # 'textarea': '文本域',
        'video': '视频',
        'audio': '音频',
        'canvas': '画布',
        # 'svg': 'SVG图形',
        'a': '链接',
        'details': '可展开详情',
        # 'summary': '摘要',
        'dialog': '对话框',
        # 'menu': '菜单',
        'nav': '导航',
    }

    element_counts = {}
    total_count = 0

    for element_type, element_name in dynamic_elements.items():
        try:
            # --- 查找元素的逻辑 ---
            if element_type == 'a':
                # 对 <a> 标签使用您自定义的 class 查找逻辑
                elements = soup.find_all(
                    'a',
                    class_=re.compile(r'(button|btn)', re.IGNORECASE)
                )
            else:
                # 对所有其他元素，使用标准的标签名查找
                elements = soup.find_all(element_type)
            
            count = len(elements)

        except Exception as e:
            print(f"查找 {element_name} 时出错: {e}")
            count = 0

        if count > 0:
            # --- 核心修改：为不同元素类型生成示例列表 ---
            examples_list = []
            for item in elements: # 使用更具描述性的变量名
                example_text = ""
                if element_type == 'a':
                    # a 标签的示例：显示 class
                    example_text = f"Class: {', '.join(item.get('class', ['N/A']))}"
                
                elif element_type == 'video':
                    # video 标签的示例：查找 src 属性
                    # 首先检查 <video> 标签自身是否有 src 属性
                    if item.get('src'):
                        example_text = f"Source: {item.get('src')}"
                    else:
                        # 如果没有，则查找其内部的 <source> 标签
                        source_tag = item.find('source')
                        if source_tag and source_tag.get('src'):
                            example_text = f"Source: {source_tag.get('src')}"
                        else:
                            example_text = "Source: 未找到"
                
                else:
                    # 其他所有标签的默认示例：获取文本内容
                    example_text = (item.get_text(strip=True))[:50]
                
                examples_list.append(example_text)

            # 将结果存入字典
            element_counts[element_type] = {
                'name': element_name,
                'count': count,
                'examples': examples_list
            }
            total_count += count

    details['interactive_elements'] = element_counts
    details['total_count'] = total_count

    # 0~5=>1; 6~10=>2; 11~15=>3; 16~20=>4; >=21=>5
    if total_count <= 2:
        score_dynamic = 1
    elif total_count <= 5:
        score_dynamic = 2
    elif total_count <= 10:
        score_dynamic = 3
    elif total_count <= 15:
        score_dynamic = 4
    else:
        score_dynamic = 5

    # 4) 最终得分为两者较小值
    final_score = min(score_llm, score_dynamic)

    # 兼容旧字段并扩展新明细
    return {
        'interactivity_score': final_score,
        'reason': result.get('reason', ''),
        'evaluation_method': 'LLM-based (HTML + CSS + JS)',
        'model_used': agent_config.get('model_type', ''),
        'detailed_evaluation': detailed,
        'score_llm': score_llm,
        'score_dynamic': score_dynamic,
        'dynamic_total_count': total_count,
        'dynamic_elements_breakdown': {
            'dynamic_elements_breakdown': element_counts,
        },
        'css_js_technology_stack': css_js_tech_stack,
        'html_analysis': {
            'html_truncated': html_truncated,
            'total_tokens_used': count_tokens(evaluation_prompt),
            'html_tokens_used': count_tokens(html_for_prompt),
            'css_tokens_used': count_tokens(css_content),
            'js_tokens_used': count_tokens(js_content)
        }
    }


# ----------------------------
# 技术栈和框架检测函数
# ----------------------------

def check_css_js_technology_stack(soup, css_content, js_content):
    """
    检查CSS/JS技术栈是否存在
    
    Args:
        soup: BeautifulSoup解析的HTML对象
        css_content: CSS文件内容
        js_content: JS文件内容
    
    Returns:
        dict: 包含found状态和详细信息的字典
    """
    details = {
        'found': False,
        'css_linked': False,
        'js_linked': False,
        'css_external_links': [],
        'js_external_links': [],
        'css_inline_styles': False,
        'js_inline_scripts': False,
        'issues': []
    }
    
    # 1. 检查外部CSS链接
    css_links = soup.find_all('link', rel='stylesheet')
    for link in css_links:
        href = link.get('href', '')
        if href:
            details['css_external_links'].append(href)
            details['css_linked'] = True
    
    # 2. 检查外部JS链接
    js_scripts = soup.find_all('script', src=True)
    for script in js_scripts:
        src = script.get('src', '')
        if src:
            details['js_external_links'].append(src)
            details['js_linked'] = True
    
    # 3. 检查内联CSS样式
    style_tags = soup.find_all('style')
    if style_tags:
        details['css_inline_styles'] = True
        for style in style_tags:
            if style.get_text().strip():
                details['css_linked'] = True
                break
    
    # 4. 检查内联JS脚本
    script_tags = soup.find_all('script')
    for script in script_tags:
        if not script.get('src') and script.get_text().strip():
            details['js_inline_scripts'] = True
            details['js_linked'] = True
            break
    
    # 5. 检查提供的CSS/JS文件内容
    if len(css_content.strip()) > 0:
        details['css_linked'] = True
    
    if len(js_content.strip()) > 0:
        details['js_linked'] = True
    
    # 6. 判断是否找到技术栈
    if details['css_linked'] and details['js_linked']:
        details['found'] = True
    elif details['css_linked'] or details['js_linked']:
        details['found'] = True  # 至少有一种技术栈存在
        if not details['css_linked']:
            details['issues'].append('COULD NOT FIND CSS TECHNOLOGY STACK')
        if not details['js_linked']:
            details['issues'].append('COULD NOT FIND JAVASCRIPT TECHNOLOGY STACK')
    else:
        details['issues'].append('COULD NOT FIND TECHNOLOGY STACK')
    
    return details




def check_dynamic_elements(soup):


    score = 0.0
    details = {
        'interactive_elements': {},
        'total_count': 0,
        'issues': []
    }

    dynamic_elements = {
        'button': '按钮',
        'input': '输入框',
        'form': '表单',
        'select': '下拉选择框',
        # 'textarea': '文本域',
        'video': '视频',
        'audio': '音频',
        'canvas': '画布',
        # 'svg': 'SVG图形',
        'a': '链接',
        'details': '可展开详情',
        # 'summary': '摘要',
        'dialog': '对话框',
        # 'menu': '菜单',
        'nav': '导航',
    }

    element_counts = {}
    total_count = 0

    for element_type, element_name in dynamic_elements.items():
        try:

            if element_type == 'a':
    
                elements = soup.find_all(
                    'a',
                    class_=re.compile(r'(button|btn)', re.IGNORECASE)
                )
            else:
  
                elements = soup.find_all(element_type)
            
            count = len(elements)

        except Exception as e:
            count = 0

        if count > 0:

            examples_list = []
            for item in elements:
                example_text = ""
                if element_type == 'a':
            
                    example_text = f"Class: {', '.join(item.get('class', ['N/A']))}"
                
                elif element_type == 'video':
           
                    if item.get('src'):
                        example_text = f"Source: {item.get('src')}"
                    else:
                   
                        source_tag = item.find('source')
                        if source_tag and source_tag.get('src'):
                            example_text = f"Source: {source_tag.get('src')}"
                        else:
                            example_text = "Source: 未找到"
                
                else:

                    example_text = (item.get_text(strip=True))[:50]
                
                examples_list.append(example_text)

            # 将结果存入字典
            element_counts[element_type] = {
                'name': element_name,
                'count': count,
                'examples': examples_list
            }
            total_count += count

    details['interactive_elements'] = element_counts
    details['total_count'] = total_count

    # 档位评分（与既有逻辑保持一致）
    if total_count >= 10:
        score = 1.0
    elif total_count >= 7:
        score = 0.8
    elif total_count >= 5:
        score = 0.6
    elif total_count >= 3:
        score = 0.4
    elif total_count >= 1:
        score = 0.2
    else:
        score = 0.0
        details['issues'].append('COULDNOT_FIND_INTERACTIVE_ELEMENTS')

    return min(score, 1.0), details

def truncate_html_by_tokens(html_content, max_tokens):

    import tiktoken
    from bs4 import BeautifulSoup
    
    def count_tokens(text, model="gpt-4o"):
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except:
            return len(text) // 4
    
    # 解析HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 优先保留的元素（按重要性排序）
    priority_elements = ['title', 'h1', 'h2', 'h3', 'meta', 'link']
    
    # 构建截断后的HTML
    truncated_html = ""
    current_tokens = 0
    
    # 首先添加DOCTYPE和html标签
    if html_content.startswith('<!DOCTYPE'):
        doctype_end = html_content.find('>') + 1
        truncated_html += html_content[:doctype_end] + '\n'
        current_tokens += count_tokens(html_content[:doctype_end])
    
    # 添加html开始标签
    html_start = html_content.find('<html')
    if html_start != -1:
        html_tag_end = html_content.find('>', html_start) + 1
        truncated_html += html_content[html_start:html_tag_end] + '\n'
        current_tokens += count_tokens(html_content[html_start:html_tag_end])
    
    # 添加head部分（优先保留）
    head_start = html_content.find('<head')
    if head_start != -1:
        head_end = html_content.find('</head>') + 7
        head_content = html_content[head_start:head_end]
        if current_tokens + count_tokens(head_content) <= max_tokens:
            truncated_html += head_content + '\n'
            current_tokens += count_tokens(head_content)
    
    # 添加body开始标签
    body_start = html_content.find('<body')
    if body_start != -1:
        body_tag_end = html_content.find('>', body_start) + 1
        truncated_html += html_content[body_start:body_tag_end] + '\n'
        current_tokens += count_tokens(html_content[body_start:body_tag_end])
    
    # 按优先级添加body内容
    body_content = soup.find('body')
    if body_content:
        for element in body_content.children:
            if element.name is None:  # 文本节点
                continue
                
            element_str = str(element)
            element_tokens = count_tokens(element_str)
            
            # 检查是否超出token限制
            if current_tokens + element_tokens <= max_tokens:
                truncated_html += element_str + '\n'
                current_tokens += element_tokens
            else:
                # 如果单个元素就超出限制，尝试截断元素内容
                if element.name in ['div', 'p', 'section'] and element.get_text():
                    # 截断文本内容
                    text_content = element.get_text()
                    available_tokens = max_tokens - current_tokens - 100  # 预留标签的token
                    
                    # 简单截断文本
                    truncated_text = truncate_text_by_tokens(text_content, available_tokens)
                    
                    # 重建元素（方案A：使用 new_tag 代替不可调用的 copy()）
                    new_tag = soup.new_tag(element.name, **dict(element.attrs))
                    new_tag.string = truncated_text
                    truncated_element = str(new_tag)
                    
                    if current_tokens + count_tokens(truncated_element) <= max_tokens:
                        truncated_html += truncated_element + '\n'
                        current_tokens += count_tokens(truncated_element)
                
                break
    
    # 添加结束标签
    truncated_html += '</body>\n</html>'
    
    return truncated_html

def truncate_text_by_tokens(text, max_tokens):
    import tiktoken
    from bs4 import BeautifulSoup
    
    def count_tokens(text, model="gpt-4o"):
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except:
            return len(text) // 4
    
    if count_tokens(text) <= max_tokens:
        return text
    
    # 二分查找合适的截断点
    left, right = 0, len(text)
    best_length = 0
    
    while left <= right:
        mid = (left + right) // 2
        truncated = text[:mid]
        tokens = count_tokens(truncated)
        
        if tokens <= max_tokens:
            best_length = mid
            left = mid + 1
        else:
            right = mid - 1
    
    # 在句子边界截断
    truncated_text = text[:best_length]
    last_sentence = truncated_text.rfind('.')
    if last_sentence > 0 and last_sentence > best_length * 0.8:  # 如果句子边界在80%范围内
        truncated_text = truncated_text[:last_sentence + 1]
    
    return truncated_text + " [CUT"