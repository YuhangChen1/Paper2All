import sys
import os
# Add parent directory to Python path to find utils module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils.website_eval_utils import *
import json
from utils.wei_utils import get_agent_config
import argparse
from dotenv import load_dotenv
import tempfile
import shutil
import glob
import re

load_dotenv()

def _is_timeout_error(err: Exception) -> bool:
    msg = str(err).lower()
    timeout_keywords = [
        'timeout', 'timed out', 'readtimeout', 'gateway timeout', '504',
        'deadline exceeded', 'request timed out'
    ]
    return any(k in msg for k in timeout_keywords)

def _crop_top_quarter(images):
    """
    Crop each image to the top quarter (reducing from bottom up) for downsampling retry after timeout.
    Ensure minimum height is not less than 16 pixels.
    """
    new_images = []
    for img in images:
        try:
            width, height = img.size
            new_h = max(16, int(height * 0.25))
            new_images.append(img.crop((0, 0, width, new_h)))
        except Exception:
            # If a single image fails, keep the original image to avoid overall failure
            new_images.append(img)
    return new_images

def _run_with_image_timeout_retry(eval_callable, website_images, max_retries: int = 3):

    last_err = None
    curr_images = website_images
    for attempt in range(max_retries + 1):
        try:
            return eval_callable(curr_images)
        except Exception as e:
            last_err = e
            if _is_timeout_error(e) and attempt < max_retries:
                # Crop to top quarter and retry
                curr_images = _crop_top_quarter(curr_images)
                print(f"Timeout detected. Retrying with cropped images (attempt {attempt+1}/{max_retries})...")
                continue
            raise

def run_qa_and_update_results(
    args,
    raw_folder,
    gen_website_path,
    save_path,
    single_model_name=None,
    del_model_name=None,
):
    """
    If single_model_name is provided, run QA for that one model only,
    but update an existing JSON file (which already contains the other
    models' results) and re-compute the overall averages.

    If single_model_name is None, run QA for all models in all_model_names
    and write a new JSON file.

    :param raw_folder: Path to folder with 'o3_qa.json'.
    :param gen_website_path: Path to the generated website image.
    :param save_path: Directory where overall_qa_result.json is saved or should be written.
    :param all_model_names: List of model names (e.g. ['vllm_qwen_vl', '4o', 'o3']).
    :param single_model_name: Optional single model name.
    """

    # Load the QA data (questions, answers, aspects)
    qa_dict = json.load(open(os.path.join(raw_folder, 'o3_qa.json'), 'r'))
    detail_qa = qa_dict['detail']
    understanding_qa = qa_dict['understanding']

    # Option A: Single model case
    if single_model_name is not None:
        qa_input_token, qa_output_token = 0, 0
        # Load the existing JSON with all previously computed results
        existing_path = os.path.join(save_path, "overall_qa_result.json")
        with open(existing_path, 'r') as f:
            overall_qa_result = json.load(f)

        if del_model_name is not None:
            # Remove the specified model from the existing results
            if del_model_name in overall_qa_result['qa_result']:
                del overall_qa_result['qa_result'][del_model_name]
                print(f"Removed model {del_model_name} from existing results.")
        
        if single_model_name in overall_qa_result['qa_result']:
            print(f"Model {single_model_name} already evaluated. Skipping.")
            return

        # Evaluate QA for the single_model_name
        print(f"Running QA for single model: {single_model_name}")
        agent_config = get_agent_config(single_model_name)

        if args.judge_version == 'paper':
            website_images = open_folder_images(gen_folder, args.paper_name.replace(' ', '_'), format='jpg')
        else:
            website_images = [Image.open(gen_website_path)]

        website_images = [ensure_under_limit_pil(image) for image in website_images]

        detail_accuracy, detail_aspect_accuracy, detail_agent_answers, input_token, output_token = eval_qa_get_answer(
            poster_input=website_images,
            questions=detail_qa['questions'],
            answers=detail_qa['answers'],
            aspects=detail_qa['aspects'],
            input_type='image',
            agent_config=agent_config
        )
        qa_input_token += input_token
        qa_output_token += output_token
        print('Detail QA accuracy:', detail_accuracy)

        understanding_accuracy, understanding_aspect_accuracy, understanding_agent_answers, input_token, output_token = eval_qa_get_answer(
            poster_input=website_images,
            questions=understanding_qa['questions'],
            answers=understanding_qa['answers'],
            aspects=understanding_qa['aspects'],
            input_type='image',
            agent_config=agent_config
        )
        qa_input_token += input_token
        qa_output_token += output_token
        print('Understanding QA accuracy:', understanding_accuracy)

        # Update QA result for this one model
        # overall_qa_result["qa_result"] is assumed to already have the others
        overall_qa_result['qa_result'][single_model_name] = {
            'detail_accuracy': detail_accuracy,
            'detail_aspect_accuracy': detail_aspect_accuracy,
            'detail_agent_answers': detail_agent_answers,
            'understanding_accuracy': understanding_accuracy,
            'understanding_aspect_accuracy': understanding_aspect_accuracy,
            'understanding_agent_answers': understanding_agent_answers
        }

        # Now re-compute the averages across all models present in the JSON
        # Grab all model entries from overall_qa_result['qa_result']
        all_models_in_file = list(overall_qa_result['qa_result'].keys())
        detail_accs = []
        understanding_accs = []
        for m in all_models_in_file:
            detail_accs.append(overall_qa_result['qa_result'][m]['detail_accuracy'])
            understanding_accs.append(overall_qa_result['qa_result'][m]['understanding_accuracy'])

        avg_detail_accuracy = float(np.mean(detail_accs)) if detail_accs else 0.0
        avg_understanding_accuracy = float(np.mean(understanding_accs)) if understanding_accs else 0.0

        overall_qa_result['avg_detail_accuracy'] = avg_detail_accuracy
        overall_qa_result['avg_understanding_accuracy'] = avg_understanding_accuracy

        # Finally, overwrite the same JSON file with the updated results
        with open(existing_path, 'w') as f:
            json.dump(overall_qa_result, f, indent=4)

        print(f'Input tokens: {qa_input_token}')
        print(f'Output tokens: {qa_output_token}')

        print('Updated overall_qa_result.json with single-model results.')
        print('New average detail accuracy:', avg_detail_accuracy)
        print('New average understanding accuracy:', avg_understanding_accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Paper2Website Evaluation Pipeline - Comprehensive evaluation system for generated academic websites',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate QA pairs for a paper
  python create_paper_questions.py --paper_folder "/path/to/paper" --model_name o3
  
  # Evaluate website quality with QA assessment
  python eval_website_pipeline.py --paper_name "Paper Title" --base_dir "/path/to/papers" --judge_version "v2" --generate_website_image "screenshot.png" --metric qa
  
  # Evaluate website aesthetics
  python eval_website_pipeline.py --paper_name "Paper Title" --base_dir "/path/to/papers" --judge_version "v2" --generate_website_image "screenshot.png" --metric informative_judge
  
  # Evaluate website completeness using LLM
  python eval_website_pipeline.py --paper_name "Paper Title" --base_dir "/path/to/papers" --judge_version "v2" --generate_website_image "screenshot.png" --metric completeness_llm
        """
    )
    parser.add_argument('--paper_name', type=str, required=True,
                       help='Name of the paper directory (the directory name under base_dir containing the paper)')
    parser.add_argument('--base_dir', type=str, default='Paper2Website-data',
                       help='Base directory containing all paper directories')
    parser.add_argument('--judge_version', type=str, required=True,
                       help='Version directory to evaluate (the directory under base_dir/paper_name/containing the image)')
    parser.add_argument('--generate_website_image', type=str, default='screenshot.png',
                       help='Name of the generated website screenshot image file')
    parser.add_argument('--metric', type=str,
                       choices=[
                           'stats',
                           'qa',
                           'informative_judge',
                           'aesthetic_judge',
                           'figure_count',
                           'completeness',
                           'connectivity',
                           'interactivity',
                           'completeness_llm',
                           'connectivity_llm',
                           'interactivity_judge',
                           'dynamic_element',
                       ],
                       default='stats',
                       help='Evaluation metric type: qa, informative_judge, aesthetic_judge, word_count, token_count, figure_count, completeness, connectivity, interactivity, completeness_llm, connectivity_llm, interactivity_judge, dynamic_element')
    parser.add_argument('--fix', type=str, default=None,
                       help='Single model name to evaluate and update existing results (for incremental evaluation)')
    parser.add_argument('--del_model_name', type=str, default=None,
                       help='Model name to delete from existing results before running --fix evaluation')
    parser.add_argument('--is_gt', action='store_true',
                       help='Whether this is evaluation for ground truth website. When specified, CSS and JS files will be included in interactivity_llm evaluation.')
    parser.add_argument('--variant_dir', type=str, default=None,
                       help='When not using --is_gt and not in paper mode, path to a specific variant directory (e.g., arxiv, openrouter_gemini_flash, openrouter_4o, v1). Used to auto-pick html and the lexicographically largest PNG as inputs.')
    
    args = parser.parse_args()

    raw_website_path = f'{args.base_dir}/{args.paper_name}/website.png'
    raw_folder = f'{args.base_dir}/{args.paper_name}'

    gen_website_path = f'{args.base_dir}/{args.paper_name}/{args.judge_version}/{args.generate_website_image}'
    gen_folder = f'{args.base_dir}/{args.paper_name}/{args.judge_version}'

    paper_root_dir = f"{args.base_dir}/{args.paper_name}"
    new_base_dir = f'{paper_root_dir}/{args.judge_version}'

    # Override inputs based on is_gt or variant_dir for non-paper mode
    html_file_path_override = None
    css_file_path_override = None
    js_file_path_override = None

    if args.judge_version != 'paper':
        if args.is_gt:
            # Use ground-truth files located at the paper root directory
            gt_screenshot = os.path.join(raw_folder, 'screenshot.png')
            gt_html = os.path.join(raw_folder, 'page.html')
            gt_css = os.path.join(raw_folder, 'inline_styles.css')
            gt_js = os.path.join(raw_folder, 'inline_scripts.js')

            if os.path.exists(gt_screenshot):
                gen_website_path = gt_screenshot
            if os.path.exists(gt_html):
                html_file_path_override = gt_html
            if os.path.exists(gt_css):
                css_file_path_override = gt_css
            if os.path.exists(gt_js):
                js_file_path_override = gt_js
        elif args.variant_dir:
            # Auto-pick inputs from the specified variant directory
            variant_dir_abs = os.path.abspath(args.variant_dir)
            if os.path.isdir(variant_dir_abs):
                try:
                    png_files = sorted([f for f in os.listdir(variant_dir_abs) if f.lower().endswith('.png')])
                    if png_files:
                        gen_website_path = os.path.join(variant_dir_abs, png_files[-1])
                    index_html = os.path.join(variant_dir_abs, 'index.html')
                    if os.path.exists(index_html):
                        html_file_path_override = index_html
                    else:
                        html_candidates = sorted([f for f in os.listdir(variant_dir_abs) if f.lower().endswith('.html')])
                        if html_candidates:
                            html_file_path_override = os.path.join(variant_dir_abs, html_candidates[-1])
                    style_css = os.path.join(variant_dir_abs, 'style.css')
                    script_js = os.path.join(variant_dir_abs, 'script.js')
                    if os.path.exists(style_css):
                        css_file_path_override = style_css
                    if os.path.exists(script_js):
                        js_file_path_override = script_js
                except Exception:
                    pass

    # Parse evaluation result save directory: eval_result/{variant} under paper directory
    def _resolve_variant_label() -> str:
        if args.is_gt:
            return 'gt'
        if args.judge_version == 'paper':
            return 'paper'
        if args.variant_dir:
            basename = os.path.basename(os.path.abspath(args.variant_dir))
            mapping = {
                'openrouter_gemini_flash': 'gemini',
                'openrouter_4o': '4o',
                'v1': 'original',
                'arxiv': 'arxiv',
            }
            return mapping.get(basename, basename)
        # When no variant, classify as agent
        return 'agent'

    variant_label = _resolve_variant_label()
    save_path = os.path.join(paper_root_dir, 'eval_result', variant_label)
    os.makedirs(save_path, exist_ok=True)

    if args.judge_version == 'paper':
        if args.metric == 'qa' and args.fix is not None:
            overall_qa_result = json.load(open(f'{save_path}/overall_qa_result.json', 'r'))
            if args.fix in overall_qa_result['qa_result']:
                print(f"Model {args.fix} already evaluated. Skipping.")
                exit(0)
        # create a temp folder to store the paper
        # 1) Create a unique temp folder
        temp_dir = tempfile.mkdtemp(prefix="eval_temp", suffix="_data")

        # 2) Build your source directory path, replacing spaces
        paper_slug = args.paper_name.replace(' ', '_')
        source_dir = os.path.join('<4o_vllm_qwen>_images_and_tables', paper_slug)

        # 3) Sequentially copy files named "<paper_slug>-<index>.png"
        index = 1
        while True:
            filename = f"{paper_slug}-{index}.png"
            src_path = os.path.join(source_dir, filename)
            if not os.path.isfile(src_path):
                # stop once the next index is missing
                break
            shutil.copy2(src_path, os.path.join(temp_dir, filename))
            index += 1
            if index > 20 and args.metric != 'word_count' and args.metric != 'token_count':
                break

        gen_folder = temp_dir
        gen_website_path = f'{args.base_dir}/{args.paper_name}/paper.pdf'
        

    print('Evaluating website:', args.paper_name)

    if args.metric == 'qa':
        if args.fix is not None:
            run_qa_and_update_results(
                args,
                raw_folder,
                gen_website_path,
                save_path,
                single_model_name=args.fix,
                del_model_name=args.del_model_name
            )
        else:
            overall_qa_result = {}
            qa_result = {}
            qa_dict = json.load(open(os.path.join(raw_folder, 'o3_qa.json'), 'r'))
            detail_qa = qa_dict['detail']
            understanding_qa = qa_dict['understanding']
            model_names = [
                '4o',
                'gemini-2.5-flash',
                '4o-mini',
                # 'llava',                      
            ]
            if args.judge_version == 'paper':
                website_images = open_folder_images(gen_folder, args.paper_name.replace(' ', '_'))
            else:
                website_images = [Image.open(gen_website_path)]

            website_images = [ensure_under_limit_pil(image) for image in website_images]
            
            for model_name in model_names:
                qa_input_token, qa_output_token = 0, 0
                print('QA model:', model_name)
                agent_config = get_agent_config(model_name)
                detail_accuracy, detail_aspect_accuracy, detail_agent_answers, input_token, output_token = eval_qa_get_answer(
                    poster_input=website_images, 
                    questions=detail_qa['questions'], 
                    answers=detail_qa['answers'], 
                    aspects=detail_qa['aspects'], 
                    input_type='image', 
                    agent_config=agent_config
                )
                print(f'{model_name} Detail QA accuracy:', detail_accuracy)
                qa_input_token += input_token
                qa_output_token += output_token

                understanding_accuracy, understanding_aspect_accuracy, understanding_agent_answers, input_token, output_token = eval_qa_get_answer(
                    poster_input=website_images, 
                    questions=understanding_qa['questions'], 
                    answers=understanding_qa['answers'], 
                    aspects=understanding_qa['aspects'], 
                    input_type='image', 
                    agent_config=agent_config
                )
                print(f'{model_name} Understanding QA accuracy:', understanding_accuracy)
                qa_input_token += input_token
                qa_output_token += output_token

                qa_result[model_name] = {
                    'detail_accuracy': detail_accuracy,
                    'detail_aspect_accuracy': detail_aspect_accuracy,
                    'detail_agent_answers': detail_agent_answers,
                    'understanding_accuracy': understanding_accuracy,
                    'understanding_aspect_accuracy': understanding_aspect_accuracy,
                    'understanding_agent_answers': understanding_agent_answers
                }

                print(f'{model_name} Input tokens:', qa_input_token)
                print(f'{model_name} Output tokens:', qa_output_token)

            # average the results
            avg_detail_accuracy = np.mean([qa_result[model_name]['detail_accuracy'] for model_name in model_names])
            avg_understanding_accuracy = np.mean([qa_result[model_name]['understanding_accuracy'] for model_name in model_names])

            print('Average detail accuracy:', avg_detail_accuracy)
            print('Average understanding accuracy:', avg_understanding_accuracy)

            overall_qa_result['avg_detail_accuracy'] = avg_detail_accuracy
            overall_qa_result['avg_understanding_accuracy'] = avg_understanding_accuracy
            overall_qa_result['qa_result'] = qa_result

            print(f'save_path: {save_path}')
            with open(f'{save_path}/overall_qa_result.json', 'w') as f:
                json.dump(overall_qa_result, f, indent=4)

    elif args.metric == 'word_count':
        if args.judge_version == 'paper':
            # loop through all images in the folder
            image_paths = open_folder_images(gen_folder, args.paper_name.replace(' ', '_'), return_path=True)
            word_count = 0
            for image_path in image_paths:
                # count words in each image
                word_count += count_words_in_image(image_path)
        else:
            word_count = count_words_in_image(gen_website_path)
        # save to json
        with open(f'{save_path}/word_count.json', 'w') as f:
            json.dump({'word_count': word_count}, f, indent=4)

    elif args.metric == 'token_count':
        if args.judge_version == 'paper':
            # loop through all images in the folder
            image_paths = open_folder_images(gen_folder, args.paper_name.replace(' ', '_'), return_path=True)
            token_count = 0
            for image_path in image_paths:
                # count tokens in each image
                token_count += count_tokens_in_image(image_path)
        else:
            token_count = count_tokens_in_image(gen_website_path)
        # save to json
        with open(f'{save_path}/token_count.json', 'w') as f:
            json.dump({'token_count': token_count}, f, indent=4)
    elif args.metric == 'informative_judge':
        agent_config = get_agent_config('gemini-2.5-flash')

        if args.judge_version == 'paper':
            website_images = open_folder_images(gen_folder, args.paper_name.replace(' ', '_'))
        else:
            website_images = [Image.open(gen_website_path)]

        def _judge_call(images):
            return eval_vlm_as_judge(
                poster_image_list=images,
                agent_config=agent_config,
            )
        results = _run_with_image_timeout_retry(_judge_call, website_images)

        aesthetic_aspects = [
            'aesthetic_element',
            'aesthetic_engagement',
            'aesthetic_layout'
        ]

        information_aspects = [
            'information_low_level',
            'information_logic',
            'information_content',
        ]

        # compute average scores for all, for aesthetic, and for information
        overall_average = np.mean([results[aspect]['score'] for aspect in results])
        aesthetic_average = np.mean([results[aspect]['score'] for aspect in results if aspect in aesthetic_aspects])
        information_average = np.mean([results[aspect]['score'] for aspect in results if aspect in information_aspects])

        judge_result = {
            'overall_average': overall_average,
            'aesthetic_average': aesthetic_average,
            'information_average': information_average,
            'results': results
        }

        # save to json
        with open(f'{save_path}/judge_result.json', 'w') as f:
            json.dump(judge_result, f, indent=4)
    elif args.metric == 'aesthetic_judge':  
        agent_config = get_agent_config('4o')

        if args.judge_version == 'paper':
            website_images = open_folder_images(gen_folder, args.paper_name.replace(' ', '_'))
        else:
            website_images = [Image.open(gen_website_path)]

        def _aesthetic_call(images):
            return eval_vlm_as_judge(
                poster_image_list=images,
                agent_config=agent_config,
                aspect='aesthetic'
            )
        results = _run_with_image_timeout_retry(_aesthetic_call, website_images)

        aesthetic_aspects = [
            'aesthetic_element',
            'aesthetic_engagement',
            'aesthetic_layout'
        ]

        aesthetic_average = np.mean([results[aspect]['score'] for aspect in results if aspect in aesthetic_aspects])

        judge_result = {
            'aesthetic_average': aesthetic_average,
            'results': results
        }

        # save to json
        with open(f'{save_path}/aesthetic_judge_result.json', 'w') as f:
            json.dump(judge_result, f, indent=4)

    if args.judge_version == 'paper':
        # remove the temp folder
        shutil.rmtree(temp_dir)
        print(f"Removed temporary folder {temp_dir}")

    elif args.metric == 'completeness':
        print('Evaluating website completeness...')
        
        # Build HTML file path
        html_file_path =  html_file_path_override if html_file_path_override else f'{new_base_dir}/index.html'

        if not os.path.exists(html_file_path):
            print(f'HTML file not found: {html_file_path}')
            exit(1)
        
        # Execute completeness evaluation
        completeness_result = evaluate_website_completeness(
            html_file_path=html_file_path,
            paper_name=args.paper_name
        )
        
        # Save results
        with open(f'{save_path}/completeness_result.json', 'w') as f:
            json.dump(completeness_result, f, indent=4)
        
        print(f'Completeness evaluation completed. Results saved to {save_path}/completeness_result.json')

    elif args.metric == 'connectivity':
        print('Evaluating website connectivity...')
        
        # Build HTML file path
        html_file_path = html_file_path_override if html_file_path_override else f'{new_base_dir}/index.html'
        
        if not os.path.exists(html_file_path):
            print(f'HTML file not found: {html_file_path}')
            exit(1)
        
        # Execute connectivity evaluation
        connectivity_result = evaluate_website_connectivity(
            html_file_path=html_file_path,
            paper_name=args.paper_name
        )
        
        # Save results
        with open(f'{save_path}/connectivity_result.json', 'w') as f:
            json.dump(connectivity_result, f, indent=4)
        
        print(f'Connectivity evaluation completed. Results saved to {save_path}/connectivity_result.json')

    elif args.metric == 'interactivity':
        print('Evaluating website interactivity...')
        
        # Build file path
        html_file_path = html_file_path_override if html_file_path_override else f'{new_base_dir}/index.html'
        css_file_path = css_file_path_override if css_file_path_override else f'{new_base_dir}/style.css'
        js_file_path = js_file_path_override if js_file_path_override else f'{new_base_dir}/script.js'
        
        if not os.path.exists(html_file_path):
            print(f'HTML file not found: {html_file_path}')
            exit(1)
        
        # Execute interactivity evaluation
        interactivity_result = evaluate_website_interactivity(
            html_file_path=html_file_path,
            css_file_path=css_file_path if os.path.exists(css_file_path) else None,
            js_file_path=js_file_path if os.path.exists(js_file_path) else None
        )
        
        # Save results
        with open(f'{save_path}/interactivity_result.json', 'w') as f:
            json.dump(interactivity_result, f, indent=4)
        
        print(f'Interactivity evaluation completed. Results saved to {save_path}/interactivity_result.json')

    elif args.metric == 'completeness_llm':
        print('Evaluating website completeness using LLM...')
        
        # Build HTML file path
        html_file_path = html_file_path_override if html_file_path_override else f'{new_base_dir}/index.html'
        
        if not os.path.exists(html_file_path):
            print(f'HTML file not found: {html_file_path}')
            exit(1)
        
        # Get 4o model configuration
        agent_config = get_agent_config('gemini-2.5-flash')
        
        # Prepare image data
        if args.judge_version == 'paper':
            website_images = open_folder_images(gen_folder, args.paper_name.replace(' ', '_'))
        else:
            website_images = [Image.open(gen_website_path)]

        # Execute LLM-based completeness evaluation
        completeness_llm_result = eval_website_completeness_llm(
            poster_image_list=website_images,
            agent_config=agent_config,
            html_file_path=html_file_path,
            paper_name=args.paper_name
        )
        
        # Save results
        with open(f'{save_path}/completeness_llm_result.json', 'w') as f:
            json.dump(completeness_llm_result, f, indent=4)
        
        print(f'LLM-based completeness evaluation completed. Results saved to {save_path}/completeness_llm_result.json')

    elif args.metric == 'connectivity_llm':
        print('Evaluating website connectivity using LLM...')
        
        # Build HTML file path
        html_file_path = html_file_path_override if html_file_path_override else f'{new_base_dir}/index.html'
        
        if not os.path.exists(html_file_path):
            print(f'HTML file not found: {html_file_path}')
            exit(1)
        
        # Get 4o model configuration
        agent_config = get_agent_config('gemini-2.5-flash')
        
        # Prepare image data
        if args.judge_version == 'paper':
            website_images = open_folder_images(gen_folder, args.paper_name.replace(' ', '_'))
        else:
            website_images = [Image.open(gen_website_path)]

        # Execute LLM-based connectivity evaluation
        connectivity_llm_result = eval_website_connectivity_llm(
            poster_image_list=website_images,
            agent_config=agent_config,
            html_file_path=html_file_path,
            paper_name=args.paper_name
        )
        
        # Save results
        with open(f'{save_path}/connectivity_llm_result.json', 'w') as f:
            json.dump(connectivity_llm_result, f, indent=4)
        
        print(f'LLM-based connectivity evaluation completed. Results saved to {save_path}/connectivity_llm_result.json')

    elif args.metric == 'interactivity_judge':
        print('Evaluating website interactivity using LLM...')
        
        # Build file path
        html_file_path = html_file_path_override if html_file_path_override else f'{new_base_dir}/index.html'
        
        if not os.path.exists(html_file_path):
            print(f'HTML file not found: {html_file_path}')
            exit(1)
        
        # Determine whether to build CSS and JS file paths based on is_gt parameter
        css_file_path = None
        js_file_path = None
        
        if args.is_gt:
            # When evaluating source website, build CSS and JS file paths
            css_file_path = css_file_path_override if css_file_path_override else f'{new_base_dir}/style.css'
            js_file_path = js_file_path_override if js_file_path_override else f'{new_base_dir}/script.js'
            print(f'Ground truth evaluation: CSS path = {css_file_path}, JS path = {js_file_path}')
        else:
            # When evaluating generated website, do not build CSS and JS file paths
            print('Generated website evaluation: CSS and JS files have been included in html')
        
        # Get 4o model configuration
        agent_config = get_agent_config('gemini-2.5-flash')
        
        # Prepare image data
        if args.judge_version == 'paper':
            website_images = open_folder_images(gen_folder, args.paper_name.replace(' ', '_'))
        else:
            website_images = [Image.open(gen_website_path)]

        # Execute LLM-based interactivity evaluation
        interactivity_llm_result = eval_website_interactivity_llm(
            poster_image_list=website_images,
            agent_config=agent_config,
            html_file_path=html_file_path,
            css_file_path=css_file_path if css_file_path and os.path.exists(css_file_path) else None,
            js_file_path=js_file_path if js_file_path and os.path.exists(js_file_path) else None
        )
        
        # Save results
        with open(f'{save_path}/interactivity_llm_result.json', 'w') as f:
            json.dump(interactivity_llm_result, f, indent=4)

        print(f'LLM-based interactivity evaluation completed. Results saved to {save_path}/interactivity_llm_result.json')

    elif args.metric == 'dynamic_element':
        print('Evaluating dynamic elements (HTML interactive components)...')
        
        # Build HTML file path
        html_file_path = html_file_path_override if html_file_path_override else f'{new_base_dir}/page.html'
        if not os.path.exists(html_file_path):
            print(f'HTML file not found: {html_file_path}')
            exit(1)
        
        # Read HTML and parse
        from bs4 import BeautifulSoup
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Count dynamic elements
        dyn_score, dyn_details = check_dynamic_elements(soup)
        
        # Save results
        dynamic_element_result = {
            'dynamic_total_count': dyn_details.get('total_count', 0),
            'dynamic_elements_breakdown': dyn_details.get('interactive_elements', {}),
            'issues': dyn_details.get('issues', []),
        }
        with open(f'{save_path}/dynamic_element_result.json', 'w') as f:
            json.dump(dynamic_element_result, f, indent=4, ensure_ascii=False)
        
        print(f'Dynamic elements evaluation completed. Results saved to {save_path}/dynamic_element_result.json')