#!/usr/bin/env python3
"""Download PRBench from the Hugging Face Hub and rebuild the local layout.

The script consumes the refactored PRBench dataset that stores per-sample
metadata together with inline images and checklist text. PDF files remain as
separate assets referenced through the `pdf_file` column. The reconstruction
process restores the original evaluation directory layout (figures, PDFs,
YAML checklists, and metadata JSON files).

Usage example:

```bash
python download_and_reconstruct_prbench.py \
    --repo-id yzweak/PRBench \
    --subset core \
    --output-dir eval
```
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import hf_hub_download
from tqdm import tqdm


DEFAULT_REPO_ID = "yzweak/PRBench"
DEFAULT_OUTPUT_DIR = Path("eval_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download PRBench and reconstruct the benchmark layout")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="Hugging Face dataset repository ID or local dataset path")
    parser.add_argument("--subset", type=str, default="core", choices=["core", "full"], help="Dataset split to download")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Destination directory for the reconstructed benchmark")
    parser.add_argument("--revision", type=str, default=None, help="Optional dataset revision (branch, tag, or commit)")
    parser.add_argument("--hf-cache", type=Path, default=None, help="Optional Hugging Face cache directory override")
    parser.add_argument("--local-assets", type=Path, default=None, help="Optional path containing `papers/` for offline PDF retrieval")
    parser.add_argument("--overwrite", action="store_true", help="Allow deleting an existing output directory before reconstruction")
    return parser.parse_args()


def load_split(args: argparse.Namespace) -> Dataset:
    repo_path = Path(args.repo_id)
    cache_dir = str(args.hf_cache) if args.hf_cache else None
    if repo_path.exists():
        dataset_dict_marker = repo_path / "dataset_dict.json"
        if dataset_dict_marker.exists():
            ds = load_from_disk(str(repo_path))
            if isinstance(ds, DatasetDict):
                return ds[args.subset]
            raise ValueError(f"Expected a DatasetDict at {repo_path}, found {type(ds)}")
        return load_dataset(str(repo_path.resolve()), split=args.subset, cache_dir=cache_dir)
    return load_dataset(args.repo_id, split=args.subset, revision=args.revision, cache_dir=cache_dir)


def ensure_output_dirs(root: Path, overwrite: bool) -> Dict[str, Path]:
    if root.exists() and overwrite:
        import shutil
        shutil.rmtree(root)
    data_dir = root / "data"
    fine_dir = data_dir / "Fine_grained_evaluation"
    twitter_dir = data_dir / "twitter_figure"
    xhs_dir = data_dir / "xhs_figure"
    input_dir = root / "input_dir"

    for directory in (data_dir, fine_dir, twitter_dir, xhs_dir, input_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "data": data_dir,
        "fine": fine_dir,
        "twitter": twitter_dir,
        "xhs": xhs_dir,
        "input": input_dir,
    }


def resolve_pdf(pdf_path: str, args: argparse.Namespace, cache: Dict[str, Path]) -> Optional[Path]:
    if not pdf_path:
        return None
    if pdf_path in cache:
        return cache[pdf_path]

    if args.local_assets:
        candidate = args.local_assets / pdf_path
        if candidate.exists():
            cache[pdf_path] = candidate
            return candidate

    downloaded = Path(hf_hub_download(args.repo_id, filename=pdf_path, repo_type="dataset", revision=args.revision, cache_dir=str(args.hf_cache) if args.hf_cache else None))
    cache[pdf_path] = downloaded
    return downloaded


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_text(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


def save_image(image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    format_hint = getattr(image, "format", None)
    if not format_hint and dest.suffix:
        format_hint = dest.suffix.lstrip(".").upper()
    image.save(dest, format=format_hint or None)


def reconstruct(args: argparse.Namespace) -> None:
    dataset = load_split(args)
    dirs = ensure_output_dirs(args.output_dir, args.overwrite)
    pdf_cache: Dict[str, Path] = {}

    metadata_records: List[Dict] = []
    core_records: List[Dict] = []
    seen_pdf: Dict[str, Path] = {}

    for row in tqdm(dataset, desc="Reconstructing records"):
        origin_str = row.get("origin_data") or "{}"
        try:
            origin_data = json.loads(origin_str)
        except json.JSONDecodeError:
            origin_data = {}

        arxiv_id = row.get("arxiv_id") or ""
        post_id = row.get("id") or ""
        platform = (row.get("platform_source") or "").upper()
        markdown = row.get("markdown_content") or ""
        image_paths = row.get("image_paths") or []
        images = row.get("images") or []
        pdf_ref = row.get("pdf_file") or ""
        is_core = bool(row.get("is_core"))

        record_entry = {
            "title": row.get("title") or "",
            "arxiv_id": arxiv_id,
            "PDF_path": "",
            "platform_source": row.get("platform_source") or "",
            "id": post_id,
            "figure_path": list(image_paths),
            "markdown_content": markdown,
            "origin_data": origin_data,
        }
        metadata_records.append(record_entry)
        if is_core:
            core_records.append(dict(record_entry))

        # Restore PDF and YAML checklist.
        target_paper_dir = dirs["fine"] / arxiv_id
        pdf_source = resolve_pdf(pdf_ref, args, pdf_cache)
        if pdf_source:
            pdf_name = Path(pdf_ref).name
            dest_pdf = target_paper_dir / pdf_name
            if pdf_ref not in seen_pdf:
                data = pdf_source.read_bytes()
                write_bytes(dest_pdf, data)
                seen_pdf[pdf_ref] = dest_pdf
            # Mirror into input_dir/{post_id}
            if post_id:
                input_pdf_dir = dirs["input"] / post_id
                if not (input_pdf_dir / pdf_name).exists():
                    write_bytes(input_pdf_dir / pdf_name, dest_pdf.read_bytes())

        yaml_text = row.get("yaml_content") or ""
        if yaml_text:
            yaml_dest = target_paper_dir / "Factual_accuracy" / "checklist.yaml"
            write_text(yaml_dest, yaml_text)

        # Restore images.
        for img, rel in zip(images, image_paths):
            if rel:
                dest_root = dirs["twitter"] if platform == "TWITTER" else dirs["xhs"]
                dest_path = dest_root / rel
                save_image(img, dest_path)

    data_dir = dirs["data"]
    write_json = lambda path, payload: write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
    write_json(data_dir / "academic_promotion_data.json", metadata_records)
    write_json(data_dir / "academic_promotion_data_core.json", core_records)

    print("\nReconstruction finished")
    print(f"Output directory: {args.output_dir}")
    print(f"Records reconstructed: {len(metadata_records)} (core subset: {len(core_records)})")


def main() -> None:
    args = parse_args()
    reconstruct(args)


if __name__ == "__main__":
    main()
