import json
import os
import random
import shutil
import tempfile
from collections import defaultdict
from glob import glob
from typing import Literal

import func_argparse
import pytorch_fid.fid_score as fid
import torch
from jinja2 import Template
from pytorch_fid.fid_score import compute_statistics_of_path
from rich import print
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import llms
from presentation import Picture, Presentation, SlidePage
from utils import Config, pexists, pjoin

fid.tqdm = lambda x: x
judges = [
    (llms.gpt4o, llms.gpt4o, "gpt4o"),
    (llms.qwen2_5, llms.intern_vl, "qwen+intern"),
    (llms.qwen2_5, llms.qwen_vl, "Qwen"),
    (llms.qwen_vl, llms.qwen_vl, "qwen_vl"),
    (llms.intern_vl, llms.intern_vl, "intern_vl"),
]
DEVICES = torch.cuda.device_count()


def get_ppl(slide: SlidePage, model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast):
    ppl = []
    text = slide.to_text()
    if len(text) == 0:
        return ppl
    tokenized = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(tokenized.input_ids, labels=tokenized.input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
        ppl.append(perplexity.item())
    return ppl


def eval_general(presentations: list[Presentation], evals: dict[str, list[int]]):
    for prs in presentations:
        if prs.source_file in evals["pages"]:
            continue
        evals["pages"][prs.source_file] = len(prs)
        evals["characters"][prs.source_file] = sum(
            [len(slide.to_text()) for slide in prs.slides]
        )
        evals["figures"][prs.source_file] = sum(
            [len(list(slide.shape_filter(Picture))) for slide in prs.slides]
        )


def eval_feature(
    presentations: list[Presentation],
    evals: dict,
    setting: str,
):
    device = f"cuda:{random.randint(0, DEVICES - 1)}"
    print("start scoring ppl")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    for prs in tqdm(presentations):
        try:
            if prs.source_file in evals["ppl"]:
                continue
            if (
                prs.source_file
                == "data/culture/pptx/ChemBio-in-the-HUB-public/PPTCrew_wo_SchemaInduction/SSRN-id2933553_Management of Systems Engineering and Technical Assistance of DARPA Research Programs/final.pptx"
            ):
                continue
            ppl = []
            for slide in prs.slides:
                ppl.extend(get_ppl(slide, model, tokenizer))
            if len(ppl) == 0:
                continue
            evals["ppl"][prs.source_file] = sum(ppl) / len(ppl)
        except Exception as e:
            print(e, "\n", "happended in ", prs.source_file)

    model = fid.InceptionV3([fid.InceptionV3.BLOCK_INDEX_BY_DIM[64]]).to(device)
    for ppt_folder in tqdm(sorted(glob(f"data/*/pptx/*/"))):
        if ppt_folder in evals["fid"]:
            continue
        source_folder = pjoin(ppt_folder, "source_slides")
        m1, s1 = compute_statistics_of_path(source_folder, model, 128, 64, device)
        try:
            with tempfile.TemporaryDirectory(prefix="ppteval_fid_") as temp_dir:
                for result_folder in glob(
                    pjoin(ppt_folder, f"final_images/{setting}/*")
                ):
                    folder_base = os.path.basename(result_folder)
                    for image_file in os.listdir(result_folder):
                        image_path = os.path.join(result_folder, image_file)
                        temp_image_path = os.path.join(
                            temp_dir, folder_base + "_" + image_file
                        ).replace(" ", "_")
                        shutil.copyfile(image_path, temp_image_path)
                if len(os.listdir(temp_dir)) < 10:
                    continue
                m2, s2 = compute_statistics_of_path(temp_dir, model, 32, 64, device)

                evals["fid"][ppt_folder] = fid.calculate_frechet_distance(
                    m1, s1, m2, s2
                )
        except Exception as e:
            print(e, "\n", "happended in ", ppt_folder, "on:", setting)


def merge_evals(folders: list[str], evals: dict):
    for folder in folders:
        sub_eval = json.load(open(pjoin(folder, "evals.json")))
        for dimension in ["content", "vision", "logic"]:
            evals[dimension] |= sub_eval[dimension]
    return evals


def slide_score(slide_folder: str):
    eval_file = pjoin(slide_folder, "evals.json")
    evals = defaultdict(dict)
    if pexists(eval_file):
        evals |= json.load(open(eval_file))
    text_scorer = Template(open("prompts/ppteval_content.txt", "r").read())
    vision_scorer = Template(open("prompts/ppteval_style.txt", "r").read())
    style_descriptor = open("prompts/ppteval_describe_style.txt", "r").read()
    content_descriptor = open("prompts/ppteval_describe_content.txt", "r").read()
    for slide_image in glob(pjoin(slide_folder, "slide_*.jpg")):
        slide_descr = slide_image.replace(".jpg", ".json")
        if not os.path.exists(slide_descr):
            style_descr = llms.vision_model(style_descriptor, slide_image)
            content_descr = llms.vision_model(content_descriptor, slide_image)
            json.dump(
                {"content": content_descr, "style": style_descr},
                open(slide_descr, "w"),
                indent=4,
            )
        else:
            descr = json.load(open(slide_descr))
            style_descr = descr["style"]
            content_descr = descr["content"]
        if slide_image not in evals["vision"]:
            evals["vision"][slide_image] = llms.language_model(
                vision_scorer.render(descr=style_descr), return_json=True
            )
        if slide_image not in evals["content"]:
            evals["content"][slide_image] = llms.language_model(
                text_scorer.render(descr=content_descr), return_json=True
            )


def pres_score(prs_source: str):
    if "/pptx/" in prs_source:  # ours
        source, setting, pdf, _ = prs_source.rsplit("/", 3)
        slide_folder = os.path.join(source, "final_images", setting, pdf)
    else:  # baseline
        slide_folder = os.path.dirname(prs_source)
    eval_file = pjoin(slide_folder, "evals.json")
    evals = defaultdict(dict)
    if pexists(eval_file):
        try:
            evals |= json.load(open(eval_file))
        except:
            pass
    evals.pop("logic", None)  # ? debug

    slide_descr = pjoin(slide_folder, "extracted.json")
    if not pexists(slide_descr):
        config = Config("/tmp")
        presentation = Presentation.from_file(prs_source, config)
        ppt_extractor = Template(open("prompts/ppteval_extract.txt", "r").read())
        extracted = llms.language_model(
            ppt_extractor.render(presentation=presentation.to_text()),
            return_json=True,
        )
        json.dump(extracted, open(slide_descr, "w"), indent=4)
    else:
        extracted = json.load(open(slide_descr))
    if presentation.source_file not in evals["logic"]:
        logic_scorer = Template(open("ppteval_coherence.txt", "r").read())
        evals["logic"][presentation.source_file] = llms.language_model(
            logic_scorer.render(
                background_information=extracted.pop("metadata"),
                logical_structure=extracted,
            ),
            return_json=True,
        )
    json.dump(evals, open(eval_file, "w"), indent=4)


# ppt eval
def eval_experiment(
    setting: str,
    general_eval: bool = False,
    feature_eval: bool = False,
    ppt_eval: bool = False,
):
    assert setting != "*"
    llms.language_model, llms.vision_model, judge_name = judges[0]
    print(f"evaluating {setting} under {judge_name}")
    print(
        "eval config :",
        f"general_eval: {general_eval}, feature_eval: {feature_eval}, ppt_eval: {ppt_eval}",
    )
    eval_file = f"data/evals/{setting}_{judge_name}.json"
    eval_stats = defaultdict(dict)
    if pexists(eval_file):
        eval_stats |= json.load(open(eval_file))
    config = Config("/tmp")
    prs_files = glob(f"data/*/pptx/*/{setting}/*/final.pptx")
    # filename dimension score
    print("start evaluation")
    if general_eval or feature_eval:
        presentations = [Presentation.from_file(i, config) for i in prs_files]
    if general_eval:
        eval_general(presentations, eval_stats)

    if feature_eval:
        eval_feature(presentations, eval_stats, setting)

    if ppt_eval:
        slide_image_folders = glob(f"data/*/pptx/*/final_images/{setting}/*")
        for presentation in prs_files:
            pres_score(presentation)
        eval_stats = merge_evals(slide_image_folders, eval_stats)
    json.dump(eval_stats, open(eval_file, "w"), indent=4)


def eval_baseline(
    setting: str,
    model: Literal["Qwen2.5", "gpt-4o"],
    general_eval: bool = False,
    feature_eval: bool = False,
    ppt_eval: bool = False,
):
    evals = defaultdict(dict)
    prs_files = glob(f"data/*/pdf/*/{setting}/{model}/final.pptx")
    slide_folders = [os.path.dirname(i) for i in prs_files]

    if general_eval or feature_eval:
        config = Config("/tmp")
        presentations = [Presentation.from_file(i, config) for i in prs_files]

    if general_eval:
        eval_general(presentations, evals)
    if feature_eval:
        eval_feature(presentations, evals, setting, fid_eval=False)
    if ppt_eval:
        for slide_folder in slide_folders:
            slide_score(slide_folder)
        for presentation in prs_files:
            pres_score(presentation)

    merge_evals(slide_folders, evals)
    json.dump(evals, open(f"data/evals/{setting}_{model}.json", "w"), indent=4)


if __name__ == "__main__":
    func_argparse.main(
        eval_experiment,
        eval_baseline,
        pres_score,
        slide_score,
    )
