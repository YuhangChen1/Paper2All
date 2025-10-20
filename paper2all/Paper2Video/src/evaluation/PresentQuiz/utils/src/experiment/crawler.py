import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from itertools import product
from time import sleep

import aiohttp
import jsonlines
import PyPDF2
import requests
from tqdm import tqdm

from presentation import Presentation
from utils import Config, tenacity

topics = [
    "culture",
    "education",
    "science",
    "society",
    "technology",
]

BANNED_LICENSES = [
    "unknown",
    "cc-by-nc-3.0",
    "cc-by-nc-4.0",
    "cc-by-nc-nd-1.0",
    "cc-by-nc-nd-4.0",
    "cc-by-nd-1.0",
    "cc-by-nd-2.5",
    "cc-by-nd-4.0",
    "other-open",
    "other-at",
    "other-pd",
    "other-closed",
]


@tenacity
def search_zenodo(
    sort: str,
    query: str = None,
    page: int = 1,
    filetype: str = "pptx",
    size: int = 500,
) -> dict:
    url = "https://zenodo.org/api/records"
    params = {
        "size": size,
        "page": page,
        "access_right": "open",
        "file_type": filetype,
        "sort": sort,
    }
    if query is not None:
        params["q"] = query
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


@tenacity
def iter_zenodo(
    sort: str,
    query: str = None,
    maxpage: int = None,
    filetype: str = "pptx",
    page: int = 1,
):
    while True:
        if maxpage is not None and page > maxpage:
            break
        for record in search_zenodo(sort, query, page, filetype, 500)["hits"]["hits"]:
            yield record, page
        page += 1
        if page > 20:
            break


def ppt_validate(filename: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        config = Config(rundir=temp_dir, debug=False)
        try:
            presentation = Presentation.from_file(filename, config)
        except Exception as e:
            return False
        num_images = len(os.listdir(config.IMAGE_DIR))
        if num_images > 3 * len(presentation) or num_images == 0:
            return False
    if len(presentation) < 8 or len(presentation) > 64:
        return False
    if len(presentation.error_history) > len(presentation) / 2:
        return False
    layout_count = defaultdict(int)

    for slide in presentation.slides:
        layout_count[slide.slide_layout_name] += 1
    if sum(layout_count.values()) / len(layout_count) < 2:
        return False
    return True


def pdf_validate(filename: str):
    try:
        with open(filename, "rb") as f:
            num_pages = len(PyPDF2.PdfReader(f).pages)
    except:
        return False
    if num_pages > 16:
        return False
    return True


async def download_file(
    session: aiohttp.ClientSession, filepath: str, url: str, pbar: tqdm = None
) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    async with session.get(
        url, params={"access_token": os.environ["ZENODO_TOKEN"]}
    ) as response:
        if response.status == 200:
            with open(filepath, "wb") as f:
                f.write(await response.read())
        elif response.status != 404:
            raise Exception(f"Failed to download {filepath}: {response.status}")
    if pbar is not None:
        pbar.update(1)


def collect_links(jsonl_file: str) -> None:
    page = 1
    progress_bar = None
    with jsonlines.open(jsonl_file, mode="w") as writer:
        while progress_bar is None or progress_bar.n < progress_bar.total:
            try:
                results = search_zenodo(page=page, sort="-mostrecent")
            except Exception as e:
                tqdm.write(f"Error {e}, current page: {page}")
                continue
            if progress_bar is None:
                progress_bar = tqdm(
                    initial=(page - 1) * 500,
                    total=results["hits"]["total"],
                    desc="Processing pages",
                )
            progress_bar.update(len(results["hits"]["hits"]))
            records = []
            for record in results["hits"]["hits"]:
                for file in record["files"]:
                    if not file["key"].endswith(".pptx"):
                        continue
                    license = record["metadata"].get("license", {"id": "unknown"})["id"]
                    if license == "unknow":
                        continue
                    records.append(
                        {
                            "filename": file["key"],
                            "size": file["size"],
                            "url": file["links"]["self"],
                            "license": license,
                            "title": record["title"],
                            "created": record["created"],
                            "updated": record["updated"],
                            "doi": record.get("doi", "unknown"),
                            "checksum": file["checksum"],
                        }
                    )
            writer.write_all(records)
            page += 1


async def download_links(jsonl_file: str) -> None:
    async with aiohttp.ClientSession() as session:
        with jsonlines.open(jsonl_file) as reader:
            tasks = list(reader)
        progress_bar = tqdm(total=len(tasks), desc="Downloading files")
        task_iter = iter(tasks)
        coroutines = []
        while True:
            while len(coroutines) < 80:
                task = next(task_iter, None)
                if task is None:
                    break
                dirname = f"zenodo-pptx/pptx/{task['license']}/{task['created'][:4]}/"
                basename = f"{task['checksum'][4:]}-{task['filename']}"
                filepath = dirname + basename
                try:
                    open("/tmp/" + basename, "wb").close()
                except:
                    filepath = dirname + basename[:240] + ".pptx"
                if os.path.exists(filepath):
                    progress_bar.update(1)
                    continue
                coroutines.append(
                    download_file(session, filepath, task["url"], progress_bar)
                )

            start_time = time.time()
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    tqdm.write(f"Error {result}")
            if len(coroutines) % 80 != 0:
                return
            coroutines.clear()
            elapsed_time = time.time() - start_time
            sleep(max(60 - elapsed_time, 0))


async def gather_files(topics: list[str], num_results: int) -> None:
    session = aiohttp.ClientSession()
    filetypes = ["pptx", "pdf"]
    progress_bar = tqdm(
        total=len(topics) * len(filetypes) * num_results, desc="Gathering files"
    )
    writer = jsonlines.open(f"data/datastats.jsonl", mode="a")
    existed = defaultdict(list)
    for i in jsonlines.open(f"data/datastats.jsonl"):
        existed[i["topic"] + i["filetype"]].append(i)
    for topic, filetype in product(topics, filetypes):
        selected = existed.get(topic + filetype, [])
        progress_bar.set_description(f"Gathering {topic} {filetype}")
        progress_bar.update(len(selected))
        page = 1 if len(selected) == 0 else selected[-1]["page"] + 1
        for record, page in iter_zenodo(
            query=topic, filetype=filetype, sort="mostviewed", page=page
        ):
            if len(selected) >= num_results:
                break
            license = record["metadata"].get("license", {"id": "unknown"})["id"]
            if license in BANNED_LICENSES:
                continue
            for file in record["files"]:
                if not file["key"].endswith(f".{filetype}"):
                    continue
                filepath = f"zenodo-pptx/{filetype}/{license}/{record['created'][:4]}/{file['checksum'][4:]}-{file['key']}"
                dst = f"data/{topic}/{filetype}/{file['key'].rsplit('.')[0]}/original.{filetype}"
                if os.path.exists(dst):
                    continue
                os.makedirs(os.path.dirname(dst))
                if os.path.exists(filepath):
                    shutil.copy(filepath, dst)
                else:
                    try:
                        await download_file(session, dst, file["links"]["self"])
                    except:
                        continue
                if (filetype == "pptx" and not ppt_validate(dst)) or (
                    filetype == "pdf" and not pdf_validate(dst)
                ):
                    shutil.rmtree(os.path.dirname(dst))
                    continue
                selected.append(
                    {
                        "filename": file["key"],
                        "size": file["size"],
                        "url": file["links"]["self"],
                        "license": license,
                        "title": record["title"],
                        "created": record["created"],
                        "updated": record["updated"],
                        "doi": record.get("doi", "unknown"),
                        "checksum": file["checksum"],
                        "page": page,
                        "topic": topic,
                        "filetype": filetype,
                    }
                )
                writer.write(selected[-1])
                progress_bar.update(1)
        progress_bar.update(num_results - len(selected))


def verify_md5(filepath):
    expected_md5 = os.path.basename(filepath).split("-")[0]
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
    if hash_md5.hexdigest() != expected_md5:
        print("find incorrect file: ", filepath)
        return filepath
    return None


def check_files_md5_parallel(directory: str, num_workers: int = 8):
    filepaths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pptx") or file.endswith(".pdf"):
                filepaths.append(os.path.join(root, file))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        result = list(executor.map(verify_md5, filepaths))
    with jsonlines.open("incorrect_files.jsonl", mode="w") as writer:
        for i in result:
            if i is not None:
                writer.write(i)


def dataset_stat():
    pdf_stat = {}
    ppt_stat = {}
    tempdir = tempfile.TemporaryDirectory()
    config = Config()
    config.set_rundir(tempdir.name)
    for topic in topics:
        markdown_contents = {
            f: len(open(f, "r").read()) for f in glob(f"data/{topic}/pdf/*/*.md")
        }
        pdf_stat |= markdown_contents
        avg_pdf_text_len = sum(markdown_contents.values()) / len(markdown_contents)
        num_images = 0
        for pdf_folder in glob(f"data/{topic}/pdf/*"):
            images = json.load(open(os.path.join(pdf_folder, "image_caption.json")))
            num_images += len(images)
        avg_pdf_images = num_images / len(markdown_contents)
        ppt_text_len = 0
        ppt_pages = 0
        ppt_images = 0
        num_ppts = 10
        for ppt_folder in glob(f"data/{topic}/pptx/*"):
            presentation = Presentation.from_file(
                os.path.join(ppt_folder, "source.pptx"), config
            )
            ppt_stat[ppt_folder] = sum(
                [len(slide.to_text()) for slide in presentation.slides]
            )

            ppt_text_len += ppt_stat[ppt_folder]
            ppt_pages += len(presentation)
            ppt_images += len(os.listdir(os.path.join(ppt_folder, "images")))

        avg_ppt_pages = ppt_pages / num_ppts
        avg_ppt_text_len = ppt_text_len / num_ppts
        avg_ppt_images = ppt_images / num_ppts
        print(
            "topic",
            "avg_pdf_text_len",
            "avg_pdf_images",
            "avg_ppt_pages",
            "avg_ppt_images",
            "avg_ppt_text_len",
        )
        print(
            f"{topic}: {avg_pdf_text_len:.2f}, {avg_pdf_images:.2f}, {avg_ppt_pages:.2f}, {avg_ppt_images:.2f}, {avg_ppt_text_len:.2f}"
        )

    json.dump(
        {"pdf": pdf_stat, "ppt": ppt_stat}, open("data/eval/stat.json", "w"), indent=4
    )


if __name__ == "__main__":
    jsonl = "zenodo-pptx/filestats.jsonl"
    if sys.argv[1] == "collect":
        collect_links(jsonl)
    elif sys.argv[1] == "download":
        asyncio.run(download_links(jsonl))
    elif sys.argv[1] == "gather":
        asyncio.run(gather_files(topics, 100))
    elif sys.argv[1] == "check":
        check_files_md5_parallel("zenodo-pptx", int(sys.argv[2]))
