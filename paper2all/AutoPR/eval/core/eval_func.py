# eval/core/eval_func.py

import os
import re
import statistics

import yaml
import aiofiles
from typing import Any, Dict, Optional, List, Tuple
from collections import Counter
import random

from openai import AsyncOpenAI

from rouge_score import rouge_scorer
from bert_score import score as bert_score_calculator
from eval.core.datatype import PromotionDataItem, FineGrainedChecklist, ImageHandlingStrategy, BaseEvalType
from eval.core.llm_interface import call_llm_api
from eval.core.utils import extract_text_from_pdf


PLATFORM_NAME_MAPPING = {
    "XHS_NOTE": "Xiaohongshu",
    "TWITTER": "Twitter"
}

BERT_SCORE_CONFIGS = {
    "XHS_NOTE": {
        "lang": "zh",
        "model_path": "eval/bert_model/bert-base-chinese"
    },
    "TWITTER": {
        "lang": "en",
        "model_path": "eval/bert_model/bert-base-uncased"
    }
}

# Centralized prompt snippets for image association
IMAGE_ASSOCIATION_PROMPTS = {
    "single_note": {
        "real": "\n\n#### Image Association\nThe social media post's text will be provided. I will also provide one or more images separately. The intended position of these images within the post text will be indicated by Markdown-style placeholders. You must associate these placeholders with the corresponding images to understand their context and narrative function.",
        "placeholder": "\n\n#### Image Context\nThe social media post's text contains placeholders like `![...](...)` that indicate where an image is intended to be. You should evaluate the post based on the text and the described placement of these visual elements, even though the actual images are not provided."
    },
    "preference": {
        "real": "\n\n#### Image Association\nThe text for each post contains explicit tags, such as **[Image A1]** or **[Image B2]**. These tags correspond to the sequence of images provided to you. This means that the image sequence you received is [Image A1, Image A2, ..., Image B1, Image B2, ...]. Use these tags to understand which image belongs to which part of the text.",
        "placeholder": "\n\n#### Image Context\nThe text for each post contains explicit tags, such as **[Image A1]** or **[Image B2]**. These placeholders indicate where images are intended to be. You must evaluate the posts based on the text and the described placement of these visual elements, even though the actual images are not provided."
    }
}

def _get_image_association_prompt(strategy: ImageHandlingStrategy, base_type: str) -> str:
    """Generates the appropriate image association prompt snippet based on the handling strategy."""
    if strategy == ImageHandlingStrategy.NONE:
        return ""
    
    prompt_key = "real" if strategy == ImageHandlingStrategy.REAL_IMAGES else "placeholder"
    
    if base_type in [BaseEvalType.SINGLE_NOTE, BaseEvalType.FINE_GRAINED]:
        return IMAGE_ASSOCIATION_PROMPTS["single_note"].get(prompt_key, "")
    elif base_type == BaseEvalType.PREFERENCE:
        return IMAGE_ASSOCIATION_PROMPTS["preference"].get(prompt_key, "")
    return ""


def _get_full_image_path(item: PromotionDataItem, relative_img_path: str) -> str:
    """Constructs the full path to an image based on the item's platform source."""
    if item.is_pr_test:
        return relative_img_path

    if item.platform_source == "TWITTER":
        base_folder = "eval/data/twitter_figure"
    else:  # Default to Xiaohongshu
        base_folder = "eval/data/xhs_figure"
    return os.path.join(base_folder, relative_img_path)


async def evaluate_single_note(
        client: AsyncOpenAI,
        item_data: PromotionDataItem,
        instruction: str,
        model: str = "gemini-1.5-flash-latest",
        include_images: ImageHandlingStrategy = ImageHandlingStrategy.NONE,
        include_pdf: bool = False,
        response_schema: Optional[Dict[str, Any]] = None,
        n_samples: int = 1,
        force_json_format_in_prompt: bool = False
) -> Dict[str, Any]:
    
    # Dynamically generate and append image association prompt
    image_prompt_snippet = _get_image_association_prompt(include_images, BaseEvalType.SINGLE_NOTE)
    instruction += image_prompt_snippet
    
    image_paths = []
    if include_images == ImageHandlingStrategy.REAL_IMAGES:
        for img_path in item_data.image_links:
            full_img_path = _get_full_image_path(item_data, img_path)
            image_paths.append(full_img_path)

    pdf_text_content = None
    if include_pdf and item_data.PDF_path:
        # Assuming PDF path might be relative or absolute, adjust as needed. Here, constructing full path.
        pdf_full_path = f'eval/data/Fine_grained_evaluation/{item_data.arxiv_id}/{item_data.arxiv_id}.pdf'
        if os.path.exists(pdf_full_path):
             pdf_text_content = await extract_text_from_pdf(pdf_full_path)
        else:
            print(f"WARNING: PDF file not found at {pdf_full_path}")


    platform_name = PLATFORM_NAME_MAPPING.get(item_data.platform_source, "post")
    formatted_instruction = instruction.format(platform_source=platform_name)

    full_content_message = formatted_instruction
    if pdf_text_content:
        full_content_message += f"\n\n--- Reference Paper Content ---\n{pdf_text_content}\n"
    full_content_message += f"\n--- Social Media Post Content ---\n{item_data.markdown_content}"

    all_assessments = await call_llm_api(
        client=client,
        full_content_message=full_content_message,
        image_paths=image_paths,
        model=model,
        response_schema=response_schema,
        n=n_samples,
        force_json_format_in_prompt=force_json_format_in_prompt # Pass through
    )
    if len(all_assessments) == 1 and isinstance(all_assessments[0], dict) and all_assessments[0].get('status') == 'failed':
        return all_assessments[0]

    return {
        "status": "completed",
        "n_samples": n_samples,
        "assessments": all_assessments
    }

async def _run_single_preference_pass(
    client: AsyncOpenAI,
    note_a: PromotionDataItem,
    note_b: PromotionDataItem,
    instruction: str, model: str, include_images: ImageHandlingStrategy, include_pdf: bool,
    response_schema: Optional[Dict[str, Any]],
    force_json_format_in_prompt: bool,
    n_samples: int = 1
) -> List[Tuple[Optional[str], Optional[Dict[str, Any]]]]:
    """Runs one comparison and returns a list of results, one for each sample."""
    
    # Dynamically generate and append image association prompt
    image_prompt_snippet = _get_image_association_prompt(include_images, BaseEvalType.PREFERENCE)
    instruction += image_prompt_snippet
    
    image_paths = []
    note_a_content = note_a.markdown_content
    note_b_content = note_b.markdown_content

    if include_images != ImageHandlingStrategy.NONE:
        img_a_count = 0
        for img_path in note_a.image_links:
            if include_images == ImageHandlingStrategy.REAL_IMAGES:
                image_paths.append(_get_full_image_path(note_a, img_path))
            note_a_content = re.sub(r'!\[.*?\]\(.*?\)', f'**[Image A{img_a_count + 1}]**', note_a_content, 1)
            img_a_count += 1
        
        img_b_count = 0
        for img_path in note_b.image_links:
            if include_images == ImageHandlingStrategy.REAL_IMAGES:
                image_paths.append(_get_full_image_path(note_b, img_path))
            note_b_content = re.sub(r'!\[.*?\]\(.*?\)', f'**[Image B{img_b_count + 1}]**', note_b_content, 1)
            img_b_count += 1

    pdf_text_content = None
    if include_pdf and note_a.PDF_path:
        pdf_full_path = f'eval/data/Fine_grained_evaluation/{note_a.arxiv_id}/{note_a.arxiv_id}.pdf'
        if os.path.exists(pdf_full_path):
            pdf_text_content = await extract_text_from_pdf(pdf_full_path)

    platform_name = PLATFORM_NAME_MAPPING.get(note_a.platform_source, "post")
    base_prompt = instruction.format(
        platform_source=platform_name,
        post_a_content=note_a_content,
        post_b_content=note_b_content
    )
    full_content_message = base_prompt
    if pdf_text_content:
        full_content_message += f"\n\n--- Reference Paper Content ---\n{pdf_text_content}"

    llm_outputs = await call_llm_api(client, full_content_message, image_paths, model, response_schema=response_schema, n=n_samples, force_json_format_in_prompt=force_json_format_in_prompt)

    results = []
    for output in llm_outputs:
        if isinstance(output, dict) and output.get("status") == "failed":
            results.append((None, output))
            continue

        winner = output.get("preference", "tie").lower()
        if "a" in winner:
            results.append(("note_a", output))
        elif "b" in winner:
            results.append(("note_b", output))
        else:
            results.append(("tie", output))
    return results


async def evaluate_preference(
        client: AsyncOpenAI,
        pr_test_item: PromotionDataItem, original_item: PromotionDataItem,
        instruction: str, model: str, include_images: ImageHandlingStrategy, include_pdf: bool,
        response_schema: Optional[Dict[str, Any]] = None,
        n_samples: int = 1, enable_rotation: bool = False,
        force_json_format_in_prompt: bool = False
) -> Dict[str, Any]:

    votes = Counter()
    raw_outcomes = []
    items = {"pr_test": pr_test_item, "original": original_item}

    pass_kwargs = {
        "client": client, "instruction": instruction, "model": model, 
        "include_images": include_images, "include_pdf": include_pdf,
        "response_schema": response_schema, "force_json_format_in_prompt": force_json_format_in_prompt
    }

    if not enable_rotation:
        pass_results = await _run_single_preference_pass(note_a=pr_test_item, note_b=original_item, n_samples=n_samples, **pass_kwargs)
        for i, (winner_identity, assessment) in enumerate(pass_results):
            votes[winner_identity] += 1
            raw_outcomes.append({"sample": i + 1, "assignment": {"note_a": "pr_test", "note_b": "original"}, "winner": winner_identity, "explanation": assessment.get("explanation", "N/A") if assessment else "Error"})
        
        final_votes = Counter({"pr_test": votes["note_a"], "original": votes["note_b"], "tie": votes["tie"]})

    else: # Rotation enabled
        n1 = n_samples // 2
        n2 = n_samples - n1
        final_votes = Counter()
        
        if n1 > 0:
            results_ab = await _run_single_preference_pass(note_a=items['pr_test'], note_b=items['original'], n_samples=n1, **pass_kwargs)
            for i, (winner, assessment) in enumerate(results_ab):
                mapped_winner = {"note_a": "pr_test", "note_b": "original"}.get(winner, "tie")
                final_votes[mapped_winner] += 1
                raw_outcomes.append({"sample": i + 1, "assignment": {"note_a": "pr_test", "note_b": "original"}, "winner": mapped_winner, "explanation": assessment.get("explanation", "N/A") if assessment else "Error"})

        if n2 > 0:
            results_ba = await _run_single_preference_pass(note_a=items['original'], note_b=items['pr_test'], n_samples=n2, **pass_kwargs)
            for i, (winner, assessment) in enumerate(results_ba):
                mapped_winner = {"note_a": "original", "note_b": "pr_test"}.get(winner, "tie")
                final_votes[mapped_winner] += 1
                raw_outcomes.append({"sample": n1 + i + 1, "assignment": {"note_a": "original", "note_b": "pr_test"}, "winner": mapped_winner, "explanation": assessment.get("explanation", "N/A") if assessment else "Error"})

    pr_votes = final_votes.get('pr_test', 0)
    original_votes = final_votes.get('original', 0)
    
    final_decision = "tie"
    if pr_votes > original_votes:
        final_decision = "pr_test"
    elif original_votes > pr_votes:
        final_decision = "original"

    return { "status": "completed", "final_decision": final_decision, "vote_summary": dict(final_votes), "raw_outcomes": raw_outcomes }


async def evaluate_fine_grained(
        client: AsyncOpenAI,
        item_data: PromotionDataItem,
        eval_criteria_base_path: str,
        criteria_subdir: str,
        instruction: str, model: str, include_images: ImageHandlingStrategy, include_pdf: bool,
        response_schema: Optional[Dict[str, Any]] = None,
        n_samples: int = 1,
        force_json_format_in_prompt: bool = False
) -> Dict[str, Any]:
    if not item_data.arxiv_id:
        return {"status": "failed", "error": "arxiv_id is missing for fine-grained evaluation."}

    checklist_path = os.path.join(eval_criteria_base_path, item_data.arxiv_id, criteria_subdir, "checklist.yaml")
    try:
        async with aiofiles.open(checklist_path, 'r', encoding='utf-8') as f:
            checklist_data = yaml.safe_load(await f.read())
        checklist = FineGrainedChecklist.model_validate(checklist_data)
    except Exception as e:
        return {"status": "failed", "error": f"Failed to load or parse checklist {checklist_path}: {e}"}

    # image_prompt_snippet = _get_image_association_prompt(include_images, BaseEvalType.FINE_GRAINED)
    
    image_paths = []
    if include_images == ImageHandlingStrategy.REAL_IMAGES:
        for img_path in item_data.image_links:
            full_img_path = _get_full_image_path(item_data, img_path)
            image_paths.append(full_img_path)
    
    pdf_text_content = None
    if include_pdf:
        # For fine-grained, the PDF should be with the criteria data
        pdf_full_path = os.path.join(eval_criteria_base_path, item_data.arxiv_id, f"{item_data.arxiv_id}.pdf")
        if os.path.exists(pdf_full_path):
            pdf_text_content = await extract_text_from_pdf(pdf_full_path)
        else:
             print(f"WARNING: PDF not found for fine-grained eval at {pdf_full_path}")


    all_samples_results = {crit.description: [] for crit in checklist.checklist}

    for criterion in checklist.checklist:
        platform_name = PLATFORM_NAME_MAPPING.get(item_data.platform_source, "post")
        
        final_instruction = instruction
        
        final_instruction = final_instruction.format(
            platform_source=platform_name,
            description=criterion.description,
            max_score=criterion.max_score
        )
        full_content_message = final_instruction
        if pdf_text_content:
            full_content_message += f"\n\n--- Reference Paper Content ---\n{pdf_text_content}\n"
        full_content_message += f"\n--- Social Media Post Content ---\n{item_data.markdown_content}"

        llm_outputs = await call_llm_api(client, full_content_message, image_paths, model, response_schema=response_schema, n=n_samples, force_json_format_in_prompt=force_json_format_in_prompt)
        
        for llm_output in llm_outputs:
            if isinstance(llm_output, dict) and "score" in llm_output and "explanation" in llm_output:
                try:
                    score = int(llm_output["score"])
                    if 0 <= score <= criterion.max_score:
                        all_samples_results[criterion.description].append({"score": score, "explanation": llm_output["explanation"]})
                    else:
                        all_samples_results[criterion.description].append({"error": "Score out of range"})
                except (ValueError, TypeError):
                    all_samples_results[criterion.description].append({"error": "Invalid score format"})
            else:
                all_samples_results[criterion.description].append({"error": f"LLM output error: {llm_output}"})

    aggregated_results = []
    total_obtained_score = 0
    total_possible_score = sum(item.max_score for item in checklist.checklist)
    
    for criterion in checklist.checklist:
        criterion_samples = all_samples_results[criterion.description]
        valid_samples = [s for s in criterion_samples if "error" not in s]
        raw_scores = [s["score"] for s in valid_samples]
        explanations = [s["explanation"] for s in valid_samples]
        
        avg_score = statistics.mean(raw_scores) if raw_scores else 0
        
        aggregated_results.append({
            "description": criterion.description,
            "max_score": criterion.max_score,
            "score": avg_score,
            "raw_scores": raw_scores,
            "explanations": explanations,
            "num_valid_samples": len(valid_samples)
        })
        total_obtained_score += avg_score

    normalized_score = (total_obtained_score / total_possible_score) if total_possible_score > 0 else 0
    
    return {
        "status": "completed",
        "fine_grained_assessment": {
            "checklist_name": checklist.name,
            "criteria_results": aggregated_results,
            "total_possible_score": total_possible_score,
            "total_obtained_score": total_obtained_score,
            "normalized_score": normalized_score,
            "n_samples": n_samples
        }
    }


async def evaluate_traditional_metrics(
        pr_test_item: PromotionDataItem,
        original_item: PromotionDataItem
) -> Dict[str, Any]:
    """Calculates ROUGE and BERT scores between a new PR note and the original note."""
    try:
        candidate = pr_test_item.markdown_content
        reference = original_item.markdown_content

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference, candidate)
        formatted_rouge = {
            "ROUGE-1": {"precision": rouge_scores["rouge1"].precision, "recall": rouge_scores["rouge1"].recall, "fmeasure": rouge_scores["rouge1"].fmeasure},
            "ROUGE-2": {"precision": rouge_scores["rouge2"].precision, "recall": rouge_scores["rouge2"].recall, "fmeasure": rouge_scores["rouge2"].fmeasure},
            "ROUGE-L": {"precision": rouge_scores["rougeL"].precision, "recall": rouge_scores["rougeL"].recall, "fmeasure": rouge_scores["rougeL"].fmeasure},
        }

        platform = original_item.platform_source
        config = BERT_SCORE_CONFIGS.get(platform)
        if not config:
            return {"status": "failed", "error": f"BERTScore config not found for platform: {platform}"}

        P, R, F1 = bert_score_calculator(
            [candidate], [reference],
            lang=config["lang"],
            verbose=False,
            model_type=config["model_path"]
        )
        formatted_bert = {"precision": P.item(), "recall": R.item(), "f1": F1.item()}

        return {
            "status": "completed",
            "traditional_metrics": {
                "rouge_scores": formatted_rouge,
                "bert_score": formatted_bert
            }
        }
    except Exception as e:
        return {"status": "failed", "error": f"An error occurred during traditional metric calculation: {e}"}