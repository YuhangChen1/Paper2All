import glob
import json
import os
import re
import subprocess

import PyPDF2
from tqdm import tqdm

import llms
from utils import pexists, ppt_to_images

slides = """
Slides should include a title page. Following slides should contain an informative slide title
and short, concise bullet points. Longer slides should be broken up into multiple slides.
"""

convert_to_latex = (
    "Summarize the following input in a {} style."
    "Style parameters: {}"
    "Format the output document as a latex file:\n"
    "Input: {}\n\n"
    "Output:"
)

sure_prompt = (
    f"Given the input text, extract the document title and authors."
    "For each section in the given input text, extract the most important sentences."
    "Format the output using the following json template:\n"
    "{}\n\n"
    "Input: {}\n"
    "Output:"
)


internal_representation = """{
    "Document Title": "TITLE",
    "Document Authors": ["AUTHOR 1", "AUTHOR 2", "...", "AUTHOR N"],
    "SECTION TITLE 1": {
        "Content": [
            "SENTENCE 1",
            "SENTENCE 2",
            "...",
            "SENTENCE N"
        ]
    },
    "SECTION TITLE 2": {
        "Content": [
            "SENTENCE 1",
            "SENTENCE 2",
            "...",
            "SENTENCE N"
        ]
    },
    "...": {},
    "SECTION TITLE N": {
        "Content": [
            "SENTENCE 1",
            "SENTENCE 2",
            "...",
            "SENTENCE N"
        ]
    }
}"""


def replace_mentions_of_figures(latex, figure_dir):
    latex = latex.split("\n")
    for i in range(len(latex)):
        paragraph = latex[i]
        matches = re.findall(r"\\includegraphics.*?{([^}]+)}", paragraph)
        for match in matches:
            if pexists(match):
                continue
            if match == os.path.basename(match):
                if pexists(os.path.join(figure_dir, match)):
                    latex[i] = paragraph.replace(match, f"{figure_dir}/{match}")
                    continue
            raise ValueError(f"Figure {match} not found")
    return "\n".join(latex)


def kctv_gen_ppt(doc_dir):
    # Take input doc
    pdf = doc_dir.split("/")[-1]
    input_json = json.load(open(doc_dir + "/refined_doc.json"))
    model_name = llms.get_simple_modelname(llms.language_model)
    output_base = os.path.join(doc_dir, "kctv", model_name)
    os.makedirs(output_base, exist_ok=True)

    if os.path.exists(os.path.join(output_base, "slide_images")):
        return

    prompt = sure_prompt.format(internal_representation, input_json)
    gpt_response = llms.language_model(prompt, return_json=True)

    with open(
        os.path.join(output_base, "final.json"),
        "w",
        encoding="utf-8",
    ) as fout:
        json.dump(gpt_response, fout, indent=4)

    latex_prompt = convert_to_latex.format("slide", slides, gpt_response)
    gpt_latex = llms.language_model(
        latex_prompt,
    )
    gpt_latex = gpt_latex.strip().removeprefix("```latex").removesuffix("```")
    gpt_latex = replace_mentions_of_figures(gpt_latex, doc_dir)
    with open(os.path.join(output_base, "final.tex"), "w") as f:
        with open(f.name, "w") as fout:
            fout.write(gpt_latex.replace("\\ ", " "))
        subprocess.run(
            ["pdflatex", f.name],
            timeout=30,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        assert len(PyPDF2.PdfReader(open("final.pdf", "rb")).pages) > 1
        os.rename("final.pdf", os.path.join(output_base, "final.pdf"))
        ppt_to_images(
            os.path.join(output_base, "final.pdf"),
            os.path.join(output_base, "slide_images"),
        )


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor

    llms.language_model = llms.gpt4o

    def process_pdf_folder(pdf_folder):
        try:
            kctv_gen_ppt(pdf_folder)
            print("success generated ppt for ", pdf_folder)
        except Exception as e:
            print(e)

    pdf_folders = glob.glob("data/*/pdf/*")
    for i in pdf_folders:
        process_pdf_folder(i)

    with ThreadPoolExecutor() as executor:
        list(
            tqdm(executor.map(process_pdf_folder, pdf_folders), total=len(pdf_folders))
        )
    os.system("make clean")
