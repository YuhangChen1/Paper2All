import json
import os
import sys
import shutil
from functools import partial
from glob import glob
from time import sleep
from typing import Type

os.environ['OPENAI_API_KEY'] = 'Your key here'

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, root_dir)

import func_argparse
import torch

import src.llms as llms
from src.experiment.ablation import (
    PPTCrew_wo_Decoupling,
    PPTCrew_wo_HTML,
    PPTCrew_wo_LayoutInduction,
    PPTCrew_wo_SchemaInduction,
    PPTCrew_wo_Structure,
)
from src.experiment.preprocess import process_filetype
from src.model_utils import get_text_model
from src.multimodal import ImageLabler
from src.pptgen import PPTCrew
from src.presentation import Presentation
from src.utils import Config, older_than, pbasename, pexists, pjoin, ppt_to_images

# language_model vision_model
EVAL_MODELS = [
    (llms.qwen2_5, llms.qwen_vl),
    (llms.gpt4o, llms.gpt4o),
    (llms.qwen_vl, llms.qwen_vl),
]

# ablation
# 0: w/o layout induction
# 1: w/o schema induction
# 2: w/o decoupling
# 3: w/o html
# 4: with gpt4o template
# 5: w/o structure information
# 6: retry 5 times

AGENT_CLASS = {
    -1: PPTCrew,
    0: PPTCrew_wo_LayoutInduction,
    1: PPTCrew_wo_SchemaInduction,
    2: PPTCrew_wo_Decoupling,
    3: PPTCrew_wo_HTML,
    4: PPTCrew,
    5: PPTCrew_wo_Structure,
    6: PPTCrew,
}


def get_setting(setting_id: int, ablation_id: int):
    assert ablation_id in AGENT_CLASS, f"ablation_id {ablation_id} not in {AGENT_CLASS}"
    assert (
        ablation_id == -1 or setting_id == 0
    ), "ablation_id == -1 only when setting_id != 0"
    language_model, vision_model = EVAL_MODELS[setting_id]
    agent_class = AGENT_CLASS.get(ablation_id)
    llms.language_model = language_model
    llms.vision_model = vision_model
    if ablation_id == -1:
        setting_name = "PPTCrew-" + llms.get_simple_modelname(
            [language_model, vision_model]
        )
    elif ablation_id == 6:
        setting_name = "PPTCrew_retry_5"
        agent_class = partial(agent_class, retry_times=5)
    else:
        setting_name = agent_class.__name__
    model_identifier = llms.get_simple_modelname(
        [llms.language_model, llms.vision_model]
    )
    if ablation_id == 4:
        setting_name = "PPTCrew_with_gpt4o"
        model_identifier = "gpt-4o+gpt-4o"
    return agent_class, setting_name, model_identifier


def do_generate(
    genclass: Type[PPTCrew],
    setting: str,
    model_identifier: str,
    debug: bool,
    ppt_folder: str,
    thread_id: int,
    num_slides: int = 12,
):
    app_config = Config(rundir=ppt_folder, debug=debug)
    text_model = get_text_model(f"cuda:{thread_id % torch.cuda.device_count()}")
    presentation = Presentation.from_file(
        pjoin(ppt_folder, "source.pptx"),
        app_config,
    )
    ImageLabler(presentation, app_config).caption_images()
    induct_cache = pjoin(
        app_config.RUN_DIR, "template_induct", model_identifier, "induct_cache.json"
    )
    if not older_than(induct_cache, wait=True):
        print(f"induct_cache not found: {induct_cache}")
        return
    slide_induction = json.load(open(induct_cache))
    try:
        pptgen: PPTCrew = genclass(text_model).set_reference(presentation, slide_induction)
    except:
        print("set_reference failed")
        pptgen: PPTCrew = genclass(text_model).set_reference(presentation, slide_induction)
        
    topic = ppt_folder.split("/")[1]
    for pdf_folder in glob(f"data/{topic}/pdf/*"):
        app_config.set_rundir(pjoin(ppt_folder, setting, pbasename(pdf_folder)))
        if pexists(pjoin(app_config.RUN_DIR, "history")):
            continue
        images = json.load(
            open(pjoin(pdf_folder, "image_caption.json"), "r"),
        )
        doc_json = json.load(
            open(pjoin(pdf_folder, "refined_doc.json"), "r"),
        )
        pptgen.generate_pres(app_config, images, num_slides, doc_json)


def generate_pres(
    setting_id: int = 0,
    setting_name: str = None,
    ablation_id: int = -1,
    thread_num: int = 8,
    debug: bool = False,
    topic: str = "*",
    num_slides: int = 12,
):
    agent_class, setting, model_identifier = get_setting(setting_id, ablation_id)
    setting = setting_name or setting
    print("generating slides using:", setting)
    generate = partial(
        do_generate,
        agent_class,
        setting,
        model_identifier,
        debug,
        num_slides=num_slides,
    )
    process_filetype("pptx", generate, thread_num, topic)


def pptx2images(settings: str = "*"):
    while True:
        for folder in glob(f"data/*/pptx/*/{settings}/*/history"):
            folder = os.path.dirname(folder)
            pptx = pjoin(folder, "final.pptx")
            ppt_folder, setting, pdf = folder.rsplit("/", 2)
            dst = pjoin(ppt_folder, "final_images", setting, pdf)

            if not pexists(pptx):
                if pexists(dst):
                    print(f"remove {dst}")
                    shutil.rmtree(dst)
                continue

            older_than(pptx)
            if pexists(dst):
                continue
            try:
                ppt_to_images(pptx, dst)
            except:
                print("pptx to images failed")
        sleep(60)
        print("keep scanning for new pptx")


if __name__ == "__main__":
    func_argparse.main(
        generate_pres,
        pptx2images,
    )
