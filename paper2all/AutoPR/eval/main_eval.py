# eval/main_eval.py

import asyncio
import os
import glob
import json
import yaml
import aiofiles
import argparse
import shutil
from collections.abc import Coroutine, Sequence
from typing import Any, Dict, Set, Tuple, List

from tqdm import tqdm
from openai import AsyncOpenAI

from eval.core.datatype import PromotionDataItem, MetricItem, EvaluationConfig, ImageHandlingStrategy
from eval.core.eval_func import (
    evaluate_single_note,
    evaluate_preference,
    evaluate_fine_grained,
    evaluate_traditional_metrics
)

from dotenv import load_dotenv

load_dotenv()



async def _wrapped_evaluate_single_note(client: AsyncOpenAI, item_data: PromotionDataItem, **kwargs) -> tuple[str, Dict[
    str, Any]]:
    result = await evaluate_single_note(client=client, item_data=item_data, **kwargs)
    return item_data.id, result


async def _wrapped_evaluate_preference(client: AsyncOpenAI, pr_test_item: PromotionDataItem,
                                     original_item: PromotionDataItem, **kwargs) -> tuple[str, Dict[str, Any]]:
    result = await evaluate_preference(client=client, pr_test_item=pr_test_item, original_item=original_item, **kwargs)
    return pr_test_item.id, result


async def _wrapped_evaluate_fine_grained(client: AsyncOpenAI, item_data: PromotionDataItem, **kwargs) -> tuple[str, Dict[
    str, Any]]:
    result = await evaluate_fine_grained(client=client, item_data=item_data, **kwargs)
    return item_data.id, result

async def _wrapped_evaluate_traditional_metrics(pr_test_item: PromotionDataItem,
                                             original_item: PromotionDataItem) -> tuple[str, Dict[str, Any]]:
    result = await evaluate_traditional_metrics(pr_test_item=pr_test_item, original_item=original_item)
    return pr_test_item.id, result

class BenchmarkRunner:

    def __init__(self, data_path: str, metric_output_path: str, pr_test_dir: str, concurrency: int = 5):
        self.data_path = data_path
        self.metric_output_path = metric_output_path
        self.pr_test_dir = pr_test_dir 
        self.concurrency = concurrency
        self.promotion_data: dict[str, PromotionDataItem] = {}
        self.completed_evals: Set[Tuple[str, str]] = set()

    async def load_data(self):
        try:
            async with aiofiles.open(self.data_path, 'r', encoding='utf-8') as f:
                raw_data = json.loads(await f.read())
                for item in raw_data:
                    try:
                        promo_item = PromotionDataItem(**item)
                        self.promotion_data[promo_item.id] = promo_item
                    except Exception as e:
                        print(f"WARNING: Skipping item due to validation error: {item.get('id', 'N/A')}. Error: {e}")
                print(f"INFO: Loaded {len(self.promotion_data)} items from {self.data_path}.")
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while loading promotion data: {e}")


    async def load_completed_tasks(self):
        try:
            async with aiofiles.open(self.metric_output_path, "r", encoding="utf-8") as f:
                async for line in f:
                    if line.strip():
                        try:
                            completed_item = json.loads(line)
                            item_id = completed_item.get('id')
                            eval_name = completed_item.get('eval_name')
                            status = completed_item.get('status')
                            if item_id and eval_name and status == "completed":
                                self.completed_evals.add((item_id, eval_name))
                        except json.JSONDecodeError:
                            print(f"WARNING: Failed to decode JSON line from metric file: {line.strip()}")
            print(f"INFO: Found {len(self.completed_evals)} completed task entries.")
        except FileNotFoundError:
            print("INFO: No completed tasks file found, starting from scratch.")
        except Exception as e:
            print(f"ERROR: An error occurred while reading completed tasks: {e}")

    def limit_concurrency(self, coroutines: Sequence[Coroutine], concurrency: int) -> list[Coroutine]:
        semaphore = asyncio.Semaphore(concurrency)

        async def with_concurrency_limit(coroutine: Coroutine):
            async with semaphore:
                return await coroutine

        return [with_concurrency_limit(coroutine) for coroutine in coroutines]

    async def run_benchmark(self, config: EvaluationConfig, llm_client: AsyncOpenAI):
        await self.load_data()
        await self.load_completed_tasks()

        tasks = []
        eval_kwargs = {
            "instruction": config.instruction, "model": config.model,
            "response_schema": config.response_schema, "include_images": config.include_images,
            "include_pdf": config.include_pdf,
            "n_samples": config.n_samples,
            "force_json_format_in_prompt": config.force_json_format_in_prompt
        }


        items_to_evaluate: List[PromotionDataItem] = []
        if config.target_data_source == "pr_test":
            print(f"INFO: [{config.eval_name}] Target is 'pr_test'. Loading data from '{self.pr_test_dir}' directory.")
            for test_dir in glob.glob(os.path.join(self.pr_test_dir, "*/")):
                item_id = os.path.basename(os.path.normpath(test_dir))
                original_item = self.promotion_data.get(item_id)
                if not original_item:
                    print(f"WARNING: Skipping pr_test for ID {item_id} as no corresponding original item was found.")
                    continue
                try:
                    with open(os.path.join(test_dir, "markdown.md"), 'r', encoding='utf-8') as f:
                        pr_markdown = f.read()
                    pr_images = glob.glob(os.path.join(test_dir, "img", "*"))

                    pr_item = PromotionDataItem(
                        id=item_id,
                        title=f"PR Test for {original_item.title}",
                        markdown_content=pr_markdown,
                        figure_path=pr_images,
                        arxiv_id=original_item.arxiv_id,
                        PDF_path=original_item.PDF_path,
                        platform_source=original_item.platform_source,
                        is_pr_test=True 
                    )
                    items_to_evaluate.append(pr_item)
                except FileNotFoundError:
                    print(f"WARNING: Skipping pr_test for ID {item_id} because markdown.md was not found.")
                except Exception as e:
                    print(f"ERROR: Failed to load pr_test data for {item_id}: {e}")
        else:
            print(f"INFO: [{config.eval_name}] Target is 'original'. Using data from main JSON file.")
            items_to_evaluate = list(self.promotion_data.values())


        if config.base_type == "preference":
            print(f"INFO: [{config.eval_name}] Setting up preference comparison tasks.")
            preference_kwargs = eval_kwargs.copy()
            preference_kwargs["enable_rotation"] = config.enable_rotation
            
            for test_dir in glob.glob(os.path.join(self.pr_test_dir, "*/")):
                item_id = os.path.basename(os.path.normpath(test_dir))
                if (item_id, config.eval_name) in self.completed_evals:
                    continue

                original_item = self.promotion_data.get(item_id)
                if not original_item: continue

                try:
                    with open(os.path.join(test_dir, "markdown.md"), 'r', encoding='utf-8') as f:
                        pr_markdown = f.read()
                    pr_images = glob.glob(os.path.join(test_dir, "img", "*"))
                    pr_test_item = PromotionDataItem(
                        id=item_id, title=f"PR Test for {original_item.title}",
                        markdown_content=pr_markdown, figure_path=pr_images,
                        arxiv_id=original_item.arxiv_id, PDF_path=original_item.PDF_path,
                        platform_source=original_item.platform_source,
                        is_pr_test=True
                    )
                    tasks.append(_wrapped_evaluate_preference(
                        client=llm_client, pr_test_item=pr_test_item,
                        original_item=original_item, **preference_kwargs
                    ))
                except Exception as e:
                    print(f"ERROR: Failed to prep preference task for {item_id}: {e}")

        elif config.base_type == "traditional_metrics":
            print(f"INFO: [{config.eval_name}] Setting up traditional metrics comparison tasks.")
            for test_dir in glob.glob(os.path.join(self.pr_test_dir, "*/")):
                item_id = os.path.basename(os.path.normpath(test_dir))
                if (item_id, config.eval_name) in self.completed_evals:
                    continue

                original_item = self.promotion_data.get(item_id)
                if not original_item: continue

                try:
                    with open(os.path.join(test_dir, "markdown.md"), 'r', encoding='utf-8') as f:
                        pr_markdown = f.read()
                    pr_images = glob.glob(os.path.join(test_dir, "img", "*"))
                    pr_test_item = PromotionDataItem(
                        id=item_id, title=f"PR Test for {original_item.title}",
                        markdown_content=pr_markdown, figure_path=pr_images,
                        arxiv_id=original_item.arxiv_id, PDF_path=original_item.PDF_path,
                        platform_source=original_item.platform_source,
                        is_pr_test=True
                    )
                    tasks.append(_wrapped_evaluate_traditional_metrics(
                        pr_test_item=pr_test_item,
                        original_item=original_item
                    ))
                except Exception as e:
                    print(f"ERROR: Failed to prep traditional metrics task for {item_id}: {e}")
        else:
            items_to_process = [
                item for item in items_to_evaluate
                if (item.id, config.eval_name) not in self.completed_evals
            ]
            print(f"INFO: [{config.eval_name}] Found {len(items_to_process)} items to evaluate.")

            for item in items_to_process:
                if config.base_type == "single_note":
                    tasks.append(_wrapped_evaluate_single_note(
                        client=llm_client, item_data=item, **eval_kwargs
                    ))
                elif config.base_type == "fine_grained":
                    if not config.criteria_subdir:
                        print(f"ERROR: [{config.eval_name}] 'criteria_subdir' is required for fine_grained evaluation. Skipping.")
                        break

                    fine_grained_kwargs = eval_kwargs.copy()
                    fine_grained_kwargs["eval_criteria_base_path"] = "eval/data/Fine_grained_evaluation/"
                    fine_grained_kwargs["criteria_subdir"] = config.criteria_subdir
                    tasks.append(_wrapped_evaluate_fine_grained(
                        client=llm_client, item_data=item, **fine_grained_kwargs
                    ))

        if not tasks:
            print(f"INFO: [{config.eval_name}] All tasks already completed or no items to process!")
            return

        llm_tasks_with_concurrency_limit = self.limit_concurrency(tasks, self.concurrency)
        async with aiofiles.open(self.metric_output_path, "a", encoding="utf-8") as f:
            for future in tqdm(asyncio.as_completed(llm_tasks_with_concurrency_limit),
                               total=len(llm_tasks_with_concurrency_limit), desc=f"Running eval: {config.eval_name}"):
                evaluated_item_id, result_dict = (None, None)
                try:
                    evaluated_item_id, result_dict = await future
                    metric_item = MetricItem(
                        id=evaluated_item_id, eval_name=config.eval_name,
                        evaluation_results=result_dict, status=result_dict.get("status", "failed"),
                        error=result_dict.get("error") if result_dict.get("status") == "failed" else None
                    )
                    await f.write(metric_item.model_dump_json(by_alias=True) + "\n")
                    await f.flush()
                except Exception as e:
                    error_id = evaluated_item_id if evaluated_item_id else "Unknown"
                    print(f"ERROR: A top-level error occurred while processing task for ID {error_id}: {e}")
        print(f"INFO: Evaluation '{config.eval_name}' completed.")

async def main():
    parser = argparse.ArgumentParser(description="Run Academic Promotion Benchmark from YAML configurations.")
    parser.add_argument("--data-path", type=str, default="eval/data/academic_promotion_data.json",
                        help="Path to the academic_promotion_data.json file.")
    parser.add_argument("--configs-dir", type=str, default="eval/configs",
                        help="Directory containing YAML configuration files for evaluations.")
    parser.add_argument("--metric-output-path", type=str, default="metric_results.jsonl",
                        help="Path to the output metric results file.")
    parser.add_argument("--concurrency", type=int, default=15, help="Number of concurrent LLM requests.")
    parser.add_argument("--run-evals", nargs='+', default=None,
                        help="Run only specific evaluations by their 'eval_name'. E.g., --run-evals 'Overall_Preference_Comparison' 'Factual Accuracy Assessment'")
    parser.add_argument("--reset-metrics", nargs='*',
                        help="""Deletes metric entries. With no arguments, deletes the whole file.
                                With arguments, removes only the specified evaluations by 'eval_name'.""")
    parser.add_argument("--target-data-source", type=str, default="default",
                        choices=["original", "pr_test"],
                        help="Override the 'target_data_source' for all evaluations. 'original' uses the main JSON, 'pr_test' uses the pr_test directory.")
    parser.add_argument("--pr-test-dir", type=str, default="pr_test",
                        help="Path to the directory containing the PR test data.")
    parser.add_argument("--model", type=str, default=None,
                        help="Override the 'model' for all evaluations (e.g., 'gemini-1.5-pro-latest').")
    
    parser.add_argument("--force-json-prompt", action='store_true',
                        help="Force JSON output by injecting formatting instructions into the prompt, for models that don't support native tool use/JSON mode.")

    parser.add_argument("--include-images-override", type=str, default=None,
                        choices=[s.value for s in ImageHandlingStrategy],
                        help="Override the 'include_images' strategy for all applicable evaluations. This will NOT affect evaluations originally set to 'none'.")

    args = parser.parse_args()

    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    if args.reset_metrics is not None:
        if not os.path.exists(args.metric_output_path):
            print(f"INFO: Metric file {args.metric_output_path} does not exist. Nothing to reset.")
        elif len(args.reset_metrics) == 0:
            os.remove(args.metric_output_path)
            print(f"INFO: Existing metrics file {args.metric_output_path} deleted for a full reset.")
        else:
            temp_file_path = args.metric_output_path + ".tmp"
            evals_to_reset = set(args.reset_metrics)
            print(f"INFO: Resetting evaluations: {', '.join(evals_to_reset)}")
            try:
                with open(args.metric_output_path, "r", encoding="utf-8") as infile, \
                     open(temp_file_path, "w", encoding="utf-8") as outfile:
                    for line in infile:
                        try:
                            entry = json.loads(line)
                            if entry.get("eval_name") not in evals_to_reset:
                                outfile.write(line)
                        except json.JSONDecodeError:
                            continue
                shutil.move(temp_file_path, args.metric_output_path)
                print("INFO: Selective reset completed.")
            except Exception as e:
                print(f"ERROR: Failed to perform selective reset: {e}")
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

    all_config_files = sorted(glob.glob(os.path.join(args.configs_dir, "*.yaml")))
    if not all_config_files:
        print(f"ERROR: No YAML config files found in {args.configs_dir}. Nothing to run.")
        return

    configs_to_run: List[EvaluationConfig] = []
    print("--- Scanning available evaluation configs ---")
    for config_path in all_config_files:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                eval_config = EvaluationConfig(**config_data)
                
                # 1. Override target_data_source
                if args.target_data_source != "default" and eval_config.target_data_source != args.target_data_source:
                    print(f"  INFO [{eval_config.eval_name}]: Overriding target_data_source to '{args.target_data_source}'")
                    eval_config.target_data_source = args.target_data_source
                
                # 2. Override model
                if args.model and eval_config.model != args.model:
                    print(f"  INFO [{eval_config.eval_name}]: Overriding model to '{args.model}'")
                    eval_config.model = args.model

                # 3. Override force_json_format_in_prompt
                if args.force_json_prompt and not eval_config.force_json_format_in_prompt:
                    print(f"  INFO [{eval_config.eval_name}]: Enabling force_json_prompt.")
                    eval_config.force_json_format_in_prompt = True
                
                # 4. Override include_images strategy with the special rule
                if args.include_images_override and eval_config.include_images != ImageHandlingStrategy.NONE:
                    override_strategy = ImageHandlingStrategy(args.include_images_override)
                    if eval_config.include_images != override_strategy:
                        print(f"  INFO [{eval_config.eval_name}]: Overriding include_images from '{eval_config.include_images.value}' to '{override_strategy.value}'")
                        eval_config.include_images = override_strategy

                # --- End of override logic ---

                if args.run_evals is None or eval_config.eval_name in args.run_evals:
                    configs_to_run.append(eval_config)
                    print(f"  [+] Queued: {eval_config.eval_name}")
                else:
                    print(f"  [-] Skipped: {eval_config.eval_name}")

        except Exception as e:
            print(f"ERROR: Failed to load or parse config {config_path}. Skipping. Error: {e}")

    if not configs_to_run:
        print("\n--- No evaluations selected to run. Exiting. ---")
        return

    print(f"\n--- Starting benchmark for {len(configs_to_run)} selected evaluation(s) ---")
    for eval_config in configs_to_run:
        print(f"\n--- Running evaluation: {eval_config.eval_name} ---")
        print(f"  - Description: {eval_config.description}")

        runner = BenchmarkRunner(
                data_path=args.data_path,
                metric_output_path=args.metric_output_path,
                pr_test_dir=args.pr_test_dir,  
                concurrency=args.concurrency
                )
        await runner.run_benchmark(config=eval_config, llm_client=client)


if __name__ == "__main__":
    asyncio.run(main())