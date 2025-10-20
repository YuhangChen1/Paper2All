from utils.poster_eval_utils import *
import argparse
import os
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper_folder', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='o3')
    args = parser.parse_args()

    paper_text = get_poster_text(os.path.join(args.paper_folder, 'paper.pdf'))

    if args.model_name == '4o':
        model_type = ModelType.GPT_4O
    elif args.model_name == 'o3':
        model_type = ModelType.O3
    detail_qa = get_questions(paper_text, 'detail', model_type)
    understanding_qa = get_questions(paper_text, 'understanding', model_type)

    detail_q, detail_a, detail_aspects = get_answers_and_remove_answers(detail_qa)
    understanding_q, understanding_a, understanding_aspects = get_answers_and_remove_answers(understanding_qa)

    final_qa = {}
    detail_qa = {
        'questions': detail_q,
        'answers': detail_a,
        'aspects': detail_aspects,
    }

    understanding_qa = {
        'questions': understanding_q,
        'answers': understanding_a,
        'aspects': understanding_aspects,
    }
    final_qa['detail'] = detail_qa
    final_qa['understanding'] = understanding_qa

    with open(os.path.join(args.paper_folder, f'{args.model_name}_qa.json'), 'w') as f:
        json.dump(final_qa, f, indent=4)