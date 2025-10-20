import glob
import json
import multiprocessing
import os
import re
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

import torch
from FlagEmbedding import BGEM3FlagModel
from jinja2 import Template
from tqdm import tqdm

os.environ['OPENAI_API_KEY'] = 'Your key here'

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, root_dir)

import src.llms as llms
from src.induct import SlideInducter
from src.model_utils import (
    get_image_embedding,
    get_image_model,
    images_cosine_similarity,
    parse_pdf,
    prs_dedup,
)
from src.multimodal import ImageLabler
from src.presentation import Picture, Presentation, SlidePage
from src.utils import Config, older_than, pexists, pjoin, ppt_to_images

markdown_clean_pattern = re.compile(r"!\[.*?\]\((.*?)\)")
device_count = torch.cuda.device_count()


def rm_folder(folder: str):
    try:
        shutil.rmtree(folder)
    except:
        for i in os.listdir(folder):
            try:
                rm_folder(pjoin(folder, i))
            except:
                pass


def process_filetype(file_type: str, func: callable, thread_num: int, topic="*"):
    folders = glob.glob(f"data/{topic}/{file_type}/*")
    progress_bar = tqdm(total=len(folders), desc=f"processing {file_type}")

    def process_folder(folder, *args, **kwargs):
        try:
            func(folder, *args, **kwargs)
        except Exception as e:
            print(f"process {file_type} folder {folder} failed: {e}")
            traceback.print_exc()
        finally:
            progress_bar.update(1)

    with ThreadPoolExecutor(thread_num) as executor:
        list(executor.map(process_folder, folders, range(len(folders))))

    progress_bar.close()


def parse_pdfs(pdf_folders: list[str], idx: int):
    # require numpy==1.26.0, which is conflict with other packages
    from marker.models import create_model_dict

    model = create_model_dict(device=idx % device_count, dtype=torch.float16)
    for pdf_folder in pdf_folders:
        if not older_than(pdf_folder + "/original.pdf"):
            continue
        if not pexists(pjoin(pdf_folder, "source.md")):
            text_content = parse_pdf(
                pdf_folder + "/original.pdf",
                pdf_folder,
                model,
            )
            if len(text_content) < 512:
                rm_folder(pdf_folder)
                continue


def prepare_pdf_folder(pdf_folder: str, rank: int):
    image_model = get_image_model(f"cuda:{rank % device_count}")
    if not pexists(pjoin(pdf_folder, "source.md")):
        return
    if not pexists(pjoin(pdf_folder, "image_caption.json")):
        images_embeddings = get_image_embedding(pdf_folder, *image_model)
        images = [pjoin(pdf_folder, image) for image in images_embeddings]
        if len(images_embeddings) == 0:
            rm_folder(pdf_folder)
            return
        similarity_matrix = images_cosine_similarity(list(images_embeddings.values()))
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] > 0.85:
                    if pexists(images[i]):
                        os.remove(images[i])
                    break
        images = [image for image in images if pexists(image)]
        image_stats = {}
        caption_prompt = open("prompts/caption.txt").read()
        for image in images:
            image_stats[image] = llms.vision_model(caption_prompt, image)
            print(image_stats[image])
        with open(pjoin(pdf_folder, "image_caption.json"), mode="w") as f:
            json.dump(image_stats, f, indent=4, ensure_ascii=False)

    if not pexists(pjoin(pdf_folder, "refined_doc.json")):
        text_content = open(pjoin(pdf_folder, "source.md")).read()
        text_content = markdown_clean_pattern.sub("", text_content)
        template = Template(open("prompts/document_refine.txt").read())
        doc_json = llms.language_model(
            template.render(markdown_document=text_content), return_json=True
        )
        json.dump(
            doc_json,
            open(pjoin(pdf_folder, "refined_doc.json"), "w"),
            indent=4,
            ensure_ascii=False,
        )


def filter_slide(slide: SlidePage):
    num_pictures = len(list(slide.shape_filter(Picture)))
    num_shapes = len(slide.shapes)
    if num_shapes > 10:
        return True
    if num_shapes - num_pictures < 2:
        return True
    if slide.real_idx != 0 and num_pictures > 2:
        return True

def I_dont_want_to_filter_slide(slide: SlidePage):
    return False

def check_consistency(slides: list[SlidePage], ppt_folder: str, image_model):
    original_embeddings = get_image_embedding(
        pjoin(ppt_folder, "original_slides"), *image_model
    )
    rebuild_embeddings = get_image_embedding(
        pjoin(ppt_folder, "source_slides"), *image_model
    )
    for slide in slides:
        if (
            torch.cosine_similarity(
                original_embeddings[f"slide_{slide.real_idx:04d}.jpg"],
                rebuild_embeddings[f"slide_{slide.slide_idx:04d}.jpg"],
                dim=-1,
            )
            < 0.9
        ):
            raise ValueError(f"slide {slide.real_idx} in {ppt_folder} is inconsistent")
    return True


def prepare_ppt_folder(ppt_folder: str, text_model: BGEM3FlagModel, image_model):
    if pexists(ppt_folder + "/source.pptx") or not older_than(
        ppt_folder + "/original.pptx"
    ):
        return
    config = Config(rundir=ppt_folder, debug=False)
    presentation = Presentation.from_file(ppt_folder + "/original.pptx", config=config)
    if not os.path.exists(pjoin(ppt_folder, "original_slides")):
        ppt_to_images(presentation.source_file, pjoin(ppt_folder, "original_slides"))
    ppt_image_folder = pjoin(ppt_folder, "source_slides")
    shutil.rmtree(ppt_image_folder, ignore_errors=True)
    shutil.copytree(pjoin(ppt_folder, "original_slides"), ppt_image_folder)

    removed_slides = prs_dedup(presentation, text_model)
    for slide in [slide for slide in presentation.slides if I_dont_want_to_filter_slide(slide)]:
        removed_slides.append(slide)
        presentation.slides.remove(slide)

    for slide in removed_slides:
        os.remove(pjoin(ppt_image_folder, f"slide_{slide.real_idx:04d}.jpg"))
    for err_idx, _ in presentation.error_history:
        os.remove(pjoin(ppt_image_folder, f"slide_{err_idx:04d}.jpg"))
    assert len(presentation) == len(
        [i for i in os.listdir(ppt_image_folder) if i.endswith(".jpg")]
    )
    for i, slide in enumerate(presentation.slides, 1):
        slide.slide_idx = i
        os.rename(
            pjoin(ppt_image_folder, f"slide_{slide.real_idx:04d}.jpg"),
            pjoin(ppt_image_folder, f"slide_{slide.slide_idx:04d}.jpg"),
        )

    check_consistency(presentation.slides, ppt_folder, image_model)
    ImageLabler(presentation, config).caption_images()
    presentation.save(pjoin(ppt_folder, "source.pptx"))
    presentation.save(pjoin(ppt_folder, "template.pptx"), layout_only=True)
    ppt_to_images(
        pjoin(ppt_folder, "template.pptx"),
        pjoin(ppt_folder, "template_images"),
    )
    os.remove(pjoin(ppt_folder, "template.pptx"))


def prepare_induction(induct_id: int, wait: bool = False):
    induct_llms = [
        (llms.qwen2_5, llms.qwen_vl),
        (llms.gpt4o, llms.gpt4o),
        (llms.qwen_vl, llms.qwen_vl),
    ]

    def do_induct(llm: list[llms.LLM], ppt_folder: str, rank: int):
        if not older_than(pjoin(ppt_folder, "source.pptx"), wait=wait):
            return
        llms.language_model = llm[0]
        llms.vision_model = llm[1]
        config = Config(rundir=ppt_folder)
        ppt_image_folder = pjoin(ppt_folder, "source_slides")
        template_image_folder = pjoin(ppt_folder, "template_images")
        image_model = get_image_model(f"cuda:{rank % device_count}")
        presentation = Presentation.from_file(pjoin(ppt_folder, "source.pptx"), config)
        ImageLabler(presentation, config).caption_images()
        slide_inducter = SlideInducter(
            presentation, ppt_image_folder, template_image_folder, config, image_model
        )
        slide_inducter.content_induct()

    for folder in tqdm(sorted(glob.glob("data/*/pptx/*")), desc="prepare induction"):
        do_induct(induct_llms[induct_id], folder, 0)


if __name__ == "__main__":
    if sys.argv[1] == "prepare_ppt":
        text_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=0)
        image_model = get_image_model(0)
        for ppt_folder in tqdm(glob.glob("data/*/pptx/*"), desc="prepare ppt"):
            prepare_ppt_folder(ppt_folder, text_model, image_model)
    elif sys.argv[1] == "prepare_induction":
        prepare_induction(int(sys.argv[2]))
    elif sys.argv[1] == "parse_pdf":
        multiprocessing.set_start_method("spawn", force=True)
        num_process = int(sys.argv[2])
        with ProcessPoolExecutor(max_workers=num_process) as executor:
            folders = glob.glob("data/*/pdf/*")
            subfolders = [[] for _ in range(num_process)]
            for idx, folder in enumerate(folders):
                subfolders[idx % num_process].append(folder)
            list(executor.map(parse_pdfs, subfolders, range(num_process)))
    elif sys.argv[1] == "prepare_pdf":
        prepare_pdf_folder = partial(prepare_pdf_folder)
        process_filetype("pdf", prepare_pdf_folder, int(sys.argv[2]))
