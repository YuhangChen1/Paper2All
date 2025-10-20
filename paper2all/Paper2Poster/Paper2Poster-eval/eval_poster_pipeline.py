from utils.poster_eval_utils import *
import json
from utils.wei_utils import get_agent_config
import argparse
from dotenv import load_dotenv
import tempfile
import shutil
import os
import glob
import re

load_dotenv()

def run_qa_and_update_results(
    args,
    raw_folder,
    gen_poster_path,
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
    :param gen_poster_path: Path to the generated poster image.
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

        if args.poster_method == 'paper':
            poster_images = open_folder_images(gen_folder, args.paper_name.replace(' ', '_'), format='jpg')
        else:
            poster_images = [Image.open(gen_poster_path)]

        poster_images = [ensure_under_limit_pil(image) for image in poster_images]

        detail_accuracy, detail_aspect_accuracy, detail_agent_answers, input_token, output_token = eval_qa_get_answer(
            poster_input=poster_images,
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
            poster_input=poster_images,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper_name', type=str)
    parser.add_argument('--base_dir', type=str, default='Paper2Poster-data')
    parser.add_argument('--poster_method', type=str)
    parser.add_argument('--poster_image_name', type=str, default='poster.png', choices=['poster.png'])
    parser.add_argument('--metric', type=str, choices=['stats', 'qa', 'judge', 'word_count', 'token_count', 'figure_count', 'aesthetic_judge'], default='stats')
    parser.add_argument('--fix', type=str, default=None)
    parser.add_argument('--del_model_name', type=str, default=None)
    
    args = parser.parse_args()

    raw_poster_path = f'{args.base_dir}/{args.paper_name}/poster.png'
    raw_folder = f'{args.base_dir}/{args.paper_name}'

    gen_poster_path = f'{args.poster_method}/{args.base_dir}/{args.paper_name}/{args.poster_image_name}'
    gen_folder = f'{args.poster_method}/{args.base_dir}/{args.paper_name}'

    save_path = f'eval_results/{args.paper_name}/{args.poster_method}'
    os.makedirs(save_path, exist_ok=True)

    if args.poster_method == 'paper':
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
        gen_poster_path = f'{args.base_dir}/{args.paper_name}/paper.pdf'
        

    print('Evaluating poster:', args.paper_name)

    if args.metric == 'stats':
        stats_file = os.path.join(save_path, 'stats_result.json')

        # 1) load existing results if there are any
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats_result = json.load(f)
            print(f"Loaded existing stats from {stats_file}")
        else:
            stats_result = {}

        # 2) CLIP similarity
        if 'CLIP_similarity' not in stats_result:
            _, cos_sim = compare_folders_with_clip(raw_folder, gen_folder)
            stats_result['CLIP_similarity'] = cos_sim
            print(f'CLIP similarity: {cos_sim}')
        else:
            print(f"Skipping CLIP similarity (already {stats_result['CLIP_similarity']})")

        # 3) we only need to regenerate markdown+images if any of the text/image metrics is missing
        need_eval = any(k not in stats_result for k in ('textual_ppl', 'mixtual_ppl', 'visual_relevance', 'visual_ppl'))
        if need_eval:                
            images, poster_text, raw_markdown, new_markdown = gen_eval_markdown(
                args.paper_name,
                args.poster_method,
                gen_poster_path
            )

            # textual PPL
            if 'textual_ppl' not in stats_result:
                textual_ppl = get_ppl(poster_text)
                stats_result['textual_ppl'] = textual_ppl
                print(f'Textual PPL: {textual_ppl}')
            else:
                print(f"Skipping textual PPL (already {stats_result['textual_ppl']})")

            # mixtual PPL
            if 'mixtual_ppl' not in stats_result:
                mixtual_ppl = get_ppl(new_markdown)
                stats_result['mixtual_ppl'] = mixtual_ppl
                print(f'Mixtual PPL: {mixtual_ppl}')
            else:
                print(f"Skipping mixtual PPL (already {stats_result['mixtual_ppl']})")

            # visual relevance
            if 'visual_relevance' not in stats_result:
                if images:
                    sims = [
                        compute_cosine_similarity(v['image_clip_embedding'],
                                                v['section_text_clip_embedding'])
                        for v in images.values()
                    ]
                    avg_sim = float(np.mean(sims))
                    stats_result['visual_relevance'] = avg_sim
                    print(f'Average cosine similarity: {avg_sim}')
                else:
                    stats_result['visual_relevance'] = 0.0
                    print('No images found in the poster. Set visual_relevance to 0.')
            else:
                print(f"Skipping visual relevance (already {stats_result['visual_relevance']})")

            if 'visual_ppl' not in stats_result or math.isnan(stats_result['visual_ppl']):
                visual_ppls = []
                for relative_path, v in images.items():
                    image_path = os.path.join('eval_poster_markdown', args.paper_name, args.poster_method, relative_path)
                    image = Image.open(image_path)
                    visual_ppl = get_visual_ppl(image, poster_text)
                    visual_ppls.append(visual_ppl)
                avg_visual_ppl = float(np.mean(visual_ppls))
                stats_result['visual_ppl'] = avg_visual_ppl
                print(f'Average visual PPL: {avg_visual_ppl}')
            else:
                print("All textual and visual metrics already computed; skipping gen_eval_markdown.")

        if 'interleaved_ppl' not in stats_result:
            interleaved_ppl = compute_interleaved_ppl(args.paper_name, args.poster_method)
            stats_result['interleaved_ppl'] = interleaved_ppl
            print(f'Interleaved PPL: {interleaved_ppl}')
        else:
            print(f"Skipping interleaved PPL (already {stats_result['interleaved_ppl']})")
        
        if 'poster_image_ppl' not in stats_result:
            if args.poster_method == 'paper':
                poster_images = open_folder_images(gen_folder, args.paper_name.replace(' ', '_'), format='jpg')
            else:
                poster_images = [Image.open(gen_poster_path)]
            poster_image_ppl = compute_poster_image_ppl(poster_images)
            stats_result['poster_image_ppl'] = poster_image_ppl
            print(f'Poster image PPL: {poster_image_ppl}')
        else:
            print(f"Skipping poster image PPL (already {stats_result['poster_image_ppl']})")

        # 4) write back updated file
        with open(stats_file, 'w') as f:
            json.dump(stats_result, f, indent=4)
        print(f"Updated stats written to {stats_file}")
    elif args.metric == 'figure_count':
        save_file_path = os.path.join(save_path, 'figure_count.json')
        if os.path.exists(save_file_path):
            print(f"Figure count already exists at {save_file_path}. Skipping.")
        else:
            figure_count = gen_eval_markdown(
                args.paper_name,
                args.poster_method,
                gen_poster_path,
                figure_count_only=True
            )
            with open(save_file_path, 'w') as f:
                json.dump({'figure_count': figure_count}, f, indent=4)
            print(f"Figure count saved to {save_file_path}")
    elif args.metric == 'qa':
        if args.fix is not None:
            run_qa_and_update_results(
                args,
                raw_folder,
                gen_poster_path,
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
                'o3',
                '4o-mini'
            ]
            if args.poster_method == 'paper':
                poster_images = open_folder_images(gen_folder, args.paper_name.replace(' ', '_'))
            else:
                poster_images = [Image.open(gen_poster_path)]

            poster_images = [ensure_under_limit_pil(image) for image in poster_images]
            
            for model_name in model_names:
                qa_input_token, qa_output_token = 0, 0
                print('QA model:', model_name)
                agent_config = get_agent_config(model_name)
                detail_accuracy, detail_aspect_accuracy, detail_agent_answers, input_token, output_token = eval_qa_get_answer(
                    poster_input=poster_images, 
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
                    poster_input=poster_images, 
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

            with open(f'{save_path}/overall_qa_result.json', 'w') as f:
                json.dump(overall_qa_result, f, indent=4)

    elif args.metric == 'word_count':
        if args.poster_method == 'paper':
            # loop through all images in the folder
            image_paths = open_folder_images(gen_folder, args.paper_name.replace(' ', '_'), return_path=True)
            word_count = 0
            for image_path in image_paths:
                # count words in each image
                word_count += count_words_in_image(image_path)
        else:
            word_count = count_words_in_image(gen_poster_path)
        # save to json
        with open(f'{save_path}/word_count.json', 'w') as f:
            json.dump({'word_count': word_count}, f, indent=4)

    elif args.metric == 'token_count':
        if args.poster_method == 'paper':
            # loop through all images in the folder
            image_paths = open_folder_images(gen_folder, args.paper_name.replace(' ', '_'), return_path=True)
            token_count = 0
            for image_path in image_paths:
                # count tokens in each image
                token_count += count_tokens_in_image(image_path)
        else:
            token_count = count_tokens_in_image(gen_poster_path)
        # save to json
        with open(f'{save_path}/token_count.json', 'w') as f:
            json.dump({'token_count': token_count}, f, indent=4)
    elif args.metric == 'judge':
        agent_config = get_agent_config('4o')

        if args.poster_method == 'paper':
            poster_images = open_folder_images(gen_folder, args.paper_name.replace(' ', '_'))
        else:
            poster_images = [Image.open(gen_poster_path)]
        
        results = eval_vlm_as_judge(
            poster_image_list=poster_images,
            agent_config=agent_config,
        )

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

        if args.poster_method == 'paper':
            poster_images = open_folder_images(gen_folder, args.paper_name.replace(' ', '_'))
        else:
            poster_images = [Image.open(gen_poster_path)]
        
        results = eval_vlm_as_judge(
            poster_image_list=poster_images,
            agent_config=agent_config,
            aspect='aesthetic'
        )

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

    if args.poster_method == 'paper':
        # remove the temp folder
        shutil.rmtree(temp_dir)
        print(f"Removed temporary folder {temp_dir}")
