# rebuild the pptx from saved code steps.jsonl
import os
import shutil
import sys
from copy import deepcopy
from glob import glob

import func_argparse
import jsonlines
import tqdm

from apis import CodeExecutor, HistoryMark
from presentation import Presentation
from utils import Config, pjoin, ppt_to_images

config = Config("/tmp")
code_executor = CodeExecutor(0)


def rebuild_pptx(agent_steps: str, prs: Presentation):
    slides = []
    steps = list(jsonlines.open(agent_steps))
    if len(steps) == 0:
        os.remove(agent_steps)
        raise ValueError(f"Jump {agent_steps} as no steps")
    if steps[-1][0] != HistoryMark.API_CALL_CORRECT:
        raise ValueError(f"Jump {agent_steps} as last step is failed")
    for mark, slide_idx, actions in steps:
        if mark != HistoryMark.API_CALL_CORRECT:
            continue
        slides.append(deepcopy(prs.slides[slide_idx - 1]))  # slide_idx starts from 1
        feedback = code_executor.execute_actions(actions, slides[-1])
        assert feedback is None, feedback
    return slides


def rebuild_all(
    setting: str = "*", topic: str = "*", out_filename: str = "rebuild.pptx"
):
    for folder in tqdm.tqdm(glob(f"data/{topic}/pptx/*")):
        prs = Presentation.from_file(pjoin(folder, "source.pptx"), config)
        pptx_container = deepcopy(prs)
        for agent_steps in glob(pjoin(folder, setting, "*", "agent_steps.jsonl")):
            dst = pjoin(os.path.dirname(agent_steps), out_filename)
            if os.path.exists(dst):
                continue
            try:
                pptx_container.slides = rebuild_pptx(agent_steps, prs)
                pptx_container.save(dst)
            except Exception as e:
                continue


if __name__ == "__main__":
    if len(sys.argv) != 1:
        func_argparse.main(rebuild_all)

    else:
        shutil.rmtree("./test", ignore_errors=True)
        os.makedirs("./test", exist_ok=True)

        source_folder = (
            "data/education/pptx/Open Science - PhD Human Rights - 2021 - module 3"
        )
        setting = "PPTCrew-Qwen2.5+Qwen2.5+Qwen2-VL"
        pdf = "37-105-1-PB (3)"

        prs = Presentation.from_file(pjoin(source_folder, "source.pptx"), config)
        container = deepcopy(prs)
        container.slides = rebuild_pptx(
            pjoin(source_folder, setting, pdf, "agent_steps.jsonl"), prs
        )
