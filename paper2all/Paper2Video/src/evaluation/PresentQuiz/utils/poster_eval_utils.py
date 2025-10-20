import random
import string
import yaml
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

def compute_interleaved_ppl(paper_name, poster_method):
    base_dir = f'eval_poster_markdown/{paper_name}/{poster_method}'
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


def compute_poster_image_ppl(images):
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

def compute_fid_with_grayscale(reference_poster_folder, generated_poster_img_folder, clip=False):
    # Step 1: Create grayscale versions in tmp/
    tmp_ref = 'tmp/ref_gray'
    tmp_gen = 'tmp/gen_gray'

    if os.path.exists('tmp/ref_gray'):
        shutil.rmtree('tmp/ref_gray')

    if os.path.exists('tmp/gen_gray'):
        shutil.rmtree('tmp/gen_gray')
    os.makedirs(tmp_ref)
    os.makedirs(tmp_gen)

    convert_folder_to_grayscale(reference_poster_folder, tmp_ref)
    convert_folder_to_grayscale(generated_poster_img_folder, tmp_gen)

    if clip:
        return compare_folders_with_clip(tmp_ref, tmp_gen)

    # Step 2: Compute FID
    model = fid.InceptionV3([fid.InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to('cuda')
    m1, s1 = compute_statistics_of_path(tmp_ref, model, 1, 2048, 'cuda')
    m2, s2 = compute_statistics_of_path(tmp_gen, model, 1, 2048, 'cuda')
    fid_score = fid.calculate_frechet_distance(m1, s1, m2, s2)

    return fid_score

def compute_fid(reference_poster_folder, generated_poster_img_folder, clip=False):
    if clip:
        return compare_folders_with_clip(reference_poster_folder, generated_poster_img_folder)
    model = fid.InceptionV3([fid.InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to('cuda')

    m1, s1 = compute_statistics_of_path(reference_poster_folder, model, 1, 2048, 'cuda')
    m2, s2 = compute_statistics_of_path(generated_poster_img_folder, model, 1, 2048, 'cuda')

    fid_score = fid.calculate_frechet_distance(
        m1, s1, m2, s2
    )

    return fid_score


def get_poster_text(poster_path):
    markdown_clean_pattern = re.compile(r"<!--[\s\S]*?-->")
    converter = DocumentConverter()
    raw_result = converter.convert(poster_path)

    raw_markdown = raw_result.document.export_to_markdown()
    text_content = markdown_clean_pattern.sub("", raw_markdown)
    if len(text_content) < 500:
        print('\nParsing with docling failed, using marker instead\n')
        parser_model = create_model_dict(device='cuda', dtype=torch.float16)
        text_content, rendered = parse_pdf(poster_path, model_lst=parser_model, save_file=False)
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

def eval_qa_get_answer(poster_input, questions, answers, aspects, input_type, agent_config):
    agent_name = f'answer_question_from_{input_type}'
    with open(f"utils/prompt_templates/{agent_name}.yaml", "r") as f:
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
            'poster_text': poster_input,
        })
        response = actor_agent.step(prompt)
        agent_answers = get_json_from_response(response.msgs[0].content)
    elif input_type == 'image':
        if 'max_images' in agent_config:
            max_images = agent_config['max_images']
        else:
            max_images = len(poster_input)
        prompt = template.render(**{
            'questions': questions,
        })
        msg = BaseMessage.make_user_message(
            role_name="User",
            content=prompt,
            image_list=poster_input[:max_images],
        )
        response = actor_agent.step(msg)
        agent_answers = get_json_from_response(response.msgs[0].content)

    input_token, output_token = account_token(response)

    accuracy, aspect_accuracy = compute_accuracy(agent_answers, answers, aspects)

    return accuracy, aspect_accuracy, agent_answers, input_token, output_token
    

def compute_accuracy(predicted, ground_truth, aspects):
    """
    Parameters
    ----------
    predicted : dict
        {question: {'answer': <letter>, 'reference': ...}, ...}
    ground_truth : dict
        {question: '<letter>. full answer', ...}
    aspects : dict
        {question: '<aspect name>', ...}

    Returns
    -------
    overall_accuracy : float
    aspect_summary : dict
        {
          '<aspect name>': {
              'total':    <int>,   # questions in this aspect
              'correct':  <int>,   # correctly answered questions
              'accuracy': <float>  # correct / total (0–1)
          },
          ...
        }
    """
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
    """
    Shuffle the order of the options for each question in the question_data.
    Also updates the "answer" field so that it uses the new letter corresponding
    to the correct option.
    
    Parameters:
        question_data (dict): A dictionary where keys are question identifiers (e.g., "Question 1")
                              and values are dictionaries containing at least the keys "options" (a list
                              of option strings) and "answer" (a string matching one of the options).
    
    Returns:
        dict: A new dictionary with the same structure as question_data but with options shuffled
              and answers updated.
    """
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
    """
    Convert a PNG image to a PDF file.

    Args:
        input_path: Path to the source .png file.
        output_path: Path where the resulting .pdf will be saved.
    """
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

def gen_eval_markdown(paper_name, poster_method, poster_path, figure_count_only=False):
    model_name="openai/clip-vit-base-patch32"
    model_name = "BAAI/AltCLIP"
    model = AltCLIPModel.from_pretrained(model_name).to('cuda')
    processor = AltCLIPProcessor.from_pretrained(model_name)

    # create a uniquely‐named file in your system temp dir (or specify dir="tmp")
    with tempfile.NamedTemporaryFile(suffix=".pdf", prefix="poster_", dir="tmp", delete=False) as tf:
        unique_pdf = tf.name

    if poster_method != 'paper':
        # convert into that file
        png_to_pdf(poster_path, unique_pdf)
        poster_path = unique_pdf
    IMAGE_RESOLUTION_SCALE = 5.0
    agent_name = f'image_captioner'
    with open(f"utils/prompt_templates/{agent_name}.yaml", "r") as f:
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

    raw_source = poster_path
    converter = DocumentConverter()
    raw_result = converter.convert(raw_source)
    raw_markdown = raw_result.document.export_to_markdown()

    output_dir = Path(f'eval_poster_markdown/{paper_name}/{poster_method}')
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

    # # Save markdown with embedded pictures
    # md_filename = output_dir / f"{doc_filename}-with-images.md"
    # conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

    # Save markdown with externally referenced pictures
    md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
    markdown = conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

    # # Save HTML with externally referenced pictures
    # html_filename = output_dir / f"{doc_filename}-with-image-refs.html"
    # conv_res.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)

    images = {}
    images_and_text = extract_images_and_sections(markdown)
    if figure_count_only:
        return len(images_and_text)
    for res in images_and_text:
        image_path = os.path.join('eval_poster_markdown', paper_name, poster_method, res['image_path'])
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

    pkl.dump(images, open(f'eval_poster_markdown/{paper_name}/{poster_method}/images.pkl', 'wb'))
    with open(f'eval_poster_markdown/{paper_name}/{poster_method}/markdown_with_images.md', 'w') as f:
        f.write(new_md)

    poster_text = get_poster_text(poster_path)

    return images, poster_text, markdown, new_md

def get_questions(paper_text, mode, model_type):
    from dotenv import load_dotenv
    load_dotenv()
    agent_name = f'generate_question_{mode}'
    with open(f"utils/prompt_templates/{agent_name}.yaml", "r") as f:
        config = yaml.safe_load(f)

    actor_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=model_type,
        model_config_dict=ChatGPTConfig().as_dict(), # [Optional] the config for model
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

def eval_vlm_as_judge_aspect(poster_image_list, agent_config, eval_aspect):
    judge_model = ModelFactory.create(
        model_platform=agent_config['model_platform'],
        model_type=agent_config['model_type'],
        model_config_dict=agent_config['model_config'],
    )

    judge_name = f'{eval_aspect}_judge'
    with open(f"utils/prompt_templates/{judge_name}.yaml", "r") as f:
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
        image_list=poster_image_list,
    )

    response = judge_agent.step(judge_message)
    return get_json_from_response(response.msgs[0].content)

def eval_vlm_as_judge(poster_image_list, agent_config, aspect=None):
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
        results[aspect] = eval_vlm_as_judge_aspect(poster_image_list, agent_config, aspect)
    
    return results