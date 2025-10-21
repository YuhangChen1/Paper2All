import sys
import os
# Add parent directory to Python path to find utils module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils.website_eval_utils import *
import argparse
import json
import glob
import re

def find_processed_markdown(paper_folder):
    """
    Find processed markdown files
    Path structure: paper_folder/result/*_images_and_tables/*/paper_name-with-image-refs.md
    """
    result_dir = os.path.join(paper_folder, 'result')
    if not os.path.exists(result_dir):
        return None
    
    # Find *_images_and_tables directory
    images_tables_dirs = glob.glob(os.path.join(result_dir, '*_images_and_tables'))
    if not images_tables_dirs:
        return None
    
    # Use the first found directory
    images_tables_dir = images_tables_dirs[0]
    
    # Find subdirectories
    subdirs = [d for d in os.listdir(images_tables_dir) 
               if os.path.isdir(os.path.join(images_tables_dir, d))]
    
    if not subdirs:
        return None
    
    # Use the first subdirectory
    subdir = subdirs[0]
    subdir_path = os.path.join(images_tables_dir, subdir)
    
    # Find -with-image-refs.md files
    md_files = glob.glob(os.path.join(subdir_path, '*-with-image-refs.md'))
    
    if md_files:
        return md_files[0]  # Return the first found file
    
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper_folder', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='o3')
    args = parser.parse_args()

    # First check if there are already processed markdown files
    processed_md_file = find_processed_markdown(args.paper_folder)
    
    if processed_md_file:
        print(f"Found processed markdown file: {os.path.basename(processed_md_file)}")
        with open(processed_md_file, 'r', encoding='utf-8') as f:
            paper_text = f.read()
    else:
        # If no processed markdown file is found, look for PDF files
        pdf_files = glob.glob(os.path.join(args.paper_folder, '*.pdf'))
        
        if not pdf_files:
            print(f"PDF file or processed markdown file not found in directory {args.paper_folder}")
            exit(1)
        elif len(pdf_files) > 1:
            print(f"Multiple PDF files found in directory {args.paper_folder}:")
            for i, pdf_file in enumerate(pdf_files, 1):
                print(f"  {i}. {os.path.basename(pdf_file)}")
            print("Will use the first PDF file for processing")
        
        # Use the first found PDF file
        pdf_path = pdf_files[0]
        print(f"Using PDF file: {os.path.basename(pdf_path)}")
        
        paper_text = get_website_text(pdf_path)
    
    # Write extracted text to markdown document for easy viewing of conversion results
    paper_name = os.path.basename(args.paper_folder)
    markdown_path = os.path.join(args.paper_folder, f'{paper_name}_extracted_text.md')
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(f"# {paper_name} - Extracted Text Content\n\n")
        f.write("## Original PDF Conversion Result\n\n")
        f.write("```\n")
        f.write(paper_text)
        f.write("\n```\n")
    
    print(f"Extracted text has been saved to: {markdown_path}")
    
    if args.model_name == '4o':
        model_type = ModelType.GPT_4O
    elif args.model_name == 'o3':
        model_type = "openai/o3"
    else:
        model_type="google/gemini-2.5-flash"
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