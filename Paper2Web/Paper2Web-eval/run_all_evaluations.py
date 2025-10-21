#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper2Website Full Evaluation Script
Automatically run all evaluation metrics and generate complete evaluation reports
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime

def generate_qa_pairs(paper_folder, model_name='o3'):
    """Generate question-answer pairs"""
    cmd = [
        'python', 'create_paper_questions.py',
        '--paper_folder', paper_folder,
        '--model_name', model_name
    ]
    
    print(f"\n{'='*80}")
    print(f"Starting to generate question-answer pairs")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print(f"Question-answer pair generation completed")
            return True, result.stdout
        else:
            print(f"Question-answer pair generation failed")
            print(f"Error output: {result.stderr}")
            return False, result.stderr
            
    except Exception as e:
        print(f"Question-answer pair generation exception: {e}")
        return False, str(e)

def run_evaluation(paper_name, base_dir, judge_version, generate_website_image, metric, fix=None, del_model_name=None):
    """Run a single evaluation metric"""
    cmd = [
        'python', 'eval_website_pipeline.py',
        '--paper_name', paper_name,
        '--base_dir', base_dir,
        '--judge_version', judge_version,
        '--generate_website_image', generate_website_image,
        '--metric', metric
    ]
    
    if fix:
        cmd.extend(['--fix', fix])
    if del_model_name:
        cmd.extend(['--del_model_name', del_model_name])
    
    print(f"\n{'='*80}")
    print(f"Starting to run evaluation metric: {metric}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print(f"{metric} evaluation completed")
            return True, result.stdout
        else:
            print(f"{metric} evaluation failed")
            print(f"Error output: {result.stderr}")
            return False, result.stderr
            
    except Exception as e:
        print(f"{metric} evaluation exception: {e}")
        return False, str(e)

def generate_summary_report(paper_name, judge_version, results):
    """Generate evaluation summary report"""
    save_path = f'eval_results/{paper_name}/{judge_version}'
    # Ensure directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Collect all evaluation results
    summary = {
        'paper_name': paper_name,
        'judge_version': judge_version,
        'evaluation_time': datetime.now().isoformat(),
        'results': {}
    }
    
    # Read each evaluation result file
    result_files = {
        'qa': 'overall_qa_result.json',
        'informative_judge': 'judge_result.json',
        'aesthetic_judge': 'aesthetic_judge_result.json',
        'completeness_llm': 'completeness_llm_result.json',
        'connectivity_llm': 'connectivity_llm_result.json',
        'interactivity_judge': 'interactivity_llm_result.json',
        'word_count': 'word_count.json',
        'token_count': 'token_count.json',
        'figure_count': 'figure_count.json'
    }
    
    for metric, filename in result_files.items():
        file_path = os.path.join(save_path, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    summary['results'][metric] = json.load(f)
            except Exception as e:
                print(f"Unable to read {filename}: {e}")
        else:
            print(f"File does not exist: {filename}")
    
    # Save summary report
    summary_path = os.path.join(save_path, 'evaluation_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation summary report saved: {summary_path}")
    return summary

def print_evaluation_summary(summary):
    """Print evaluation summary"""
    print(f"\n{'='*80}")
    print(f"Evaluation Summary Report")
    print(f"{'='*80}")
    print(f"Paper Name: {summary['paper_name']}")
    print(f"Evaluation Version: {summary['judge_version']}")
    print(f"Evaluation Time: {summary['evaluation_time']}")
    
    results = summary['results']
    
    # QA evaluation results
    if 'qa' in results:
        qa_result = results['qa']
        print(f"\nQA Accuracy Evaluation:")
        print(f"  Average Detail Accuracy: {qa_result.get('avg_detail_accuracy', 'N/A'):.3f}")
        print(f"  Average Understanding Accuracy: {qa_result.get('avg_understanding_accuracy', 'N/A'):.3f}")
    
    # Aesthetic evaluation results
    if 'judge' in results:
        judge_result = results['judge']
        print(f"\nAesthetic Quality Evaluation:")
        print(f"  Overall Average: {judge_result.get('overall_average', 'N/A'):.3f}")
        print(f"  Aesthetic Average: {judge_result.get('aesthetic_average', 'N/A'):.3f}")
        print(f"  Information Average: {judge_result.get('information_average', 'N/A'):.3f}")
    
    # Statistics results
    if 'word_count' in results:
        print(f"\nContent Statistics:")
        print(f"  Word Count: {results['word_count'].get('word_count', 'N/A')}")
    
    if 'token_count' in results:
        print(f"  Token Count: {results['token_count'].get('token_count', 'N/A')}")
    
    if 'figure_count' in results:
        print(f"  Figure Count: {results['figure_count'].get('figure_count', 'N/A')}")
    
    # Website functionality evaluation
    llm_metrics = ['completeness_llm', 'connectivity_llm', 'interactivity_llm']
    for metric in llm_metrics:
        if metric in results:
            result = results[metric]
            metric_name = metric.replace('_llm', '').replace('_', ' ').title()
            print(f"\n{metric_name} Evaluation:")
            if 'score' in result:
                print(f"  Score: {result['score']:.3f}")
            if 'details' in result:
                print(f"  Details: {result['details']}")

def main():
    parser = argparse.ArgumentParser(
        description='Paper2Website Full Evaluation Script - Automatically run all evaluation metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # Run all evaluation metrics (automatically generate QA file)
  python run_all_evaluations.py --paper_name "Paper Title" --base_dir "/path/to/papers" --judge_version "v2" --auto_generate_qa
  
  # Specify screenshot filename
  python run_all_evaluations.py --paper_name "Paper Title" --base_dir "/path/to/papers" --judge_version "v2" --generate_website_image "final_screenshot.png" --auto_generate_qa
  
  # Skip certain evaluation metrics
  python run_all_evaluations.py --paper_name "Paper Title" --base_dir "/path/to/papers" --judge_version "v2" --skip_metrics "word_count,token_count" --auto_generate_qa
  
  # Use different model to generate QA
  python run_all_evaluations.py --paper_name "Paper Title" --base_dir "/path/to/papers" --judge_version "v2" --auto_generate_qa --qa_model "4o"
        """
    )
    
    parser.add_argument('--paper_name', type=str, required=True,
                       help='Paper directory name')
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Base directory path')
    parser.add_argument('--judge_version', type=str, required=True,
                       help='Evaluation version directory')
    parser.add_argument('--generate_website_image', type=str, default='screenshot.png',
                       help='Generated website screenshot filename')
    parser.add_argument('--skip_metrics', type=str, default='',
                       help='Evaluation metrics to skip, comma-separated (e.g., word_count,token_count)')
    parser.add_argument('--fix', type=str, default=None,
                       help='Incremental evaluation: evaluate only specified model')
    parser.add_argument('--del_model_name', type=str, default=None,
                       help='Incremental evaluation: delete specified model results')
    parser.add_argument('--auto_generate_qa', action='store_true',
                       help='Automatically generate question-answer pairs (if not exist)')
    parser.add_argument('--qa_model', type=str, default='o3',
                       help='Model to use for generating question-answer pairs (default: o3)')
    
    args = parser.parse_args()
    
    # All evaluation metrics
    all_metrics = [
        'qa',                    # QA accuracy
        'informative_judge',     # Aesthetic and information quality
        'aesthetic_judge',       # Aesthetic quality
        'completeness_llm',      # Completeness evaluation
        'connectivity_llm',      # Connectivity evaluation
        'interactivity_judge',   # Interactivity evaluation
        'word_count',            # Word count statistics
        'token_count',           # Token statistics
        'figure_count'           # Figure count statistics
    ]
    
    # Parse metrics to skip
    skip_metrics = set(args.skip_metrics.split(',')) if args.skip_metrics else set()
    skip_metrics.discard('')  # Remove empty strings
    
    # Determine metrics to run
    metrics_to_run = [m for m in all_metrics if m not in skip_metrics]
    
    # Check if QA file needs to be generated
    paper_folder = os.path.join(args.base_dir, args.paper_name)
    qa_file_path = os.path.join(paper_folder, 'o3_qa.json')
    
    if 'qa' in metrics_to_run and not os.path.exists(qa_file_path):
        if args.auto_generate_qa:
            print(f"\nQA file not found, starting automatic generation...")
            success, output = generate_qa_pairs(paper_folder, args.qa_model)
            if not success:
                print(f"QA file generation failed, skipping QA evaluation")
                metrics_to_run = [m for m in metrics_to_run if m != 'qa']
            else:
                print(f"QA file generation successful")
        else:
            print(f"\nQA file does not exist: {qa_file_path}")
            print(f"   Use --auto_generate_qa parameter to automatically generate, or manually run:")
            print(f"   python create_paper_questions.py --paper_folder \"{paper_folder}\" --model_name {args.qa_model}")
            print(f"   Skipping QA evaluation...")
            metrics_to_run = [m for m in metrics_to_run if m != 'qa']
    
    print(f"Starting full evaluation")
    print(f"Paper Name: {args.paper_name}")
    print(f"Base Directory: {args.base_dir}")
    print(f"Evaluation Version: {args.judge_version}")
    print(f"Screenshot File: {args.generate_website_image}")
    print(f"Metrics to Run: {', '.join(metrics_to_run)}")
    if skip_metrics:
        print(f"Metrics to Skip: {', '.join(skip_metrics)}")
    
    # Run all evaluation metrics
    results = {}
    success_count = 0
    
    for metric in metrics_to_run:
        success, output = run_evaluation(
            args.paper_name, 
            args.base_dir, 
            args.judge_version, 
            args.generate_website_image, 
            metric,
            args.fix,
            args.del_model_name
        )
        
        results[metric] = {
            'success': success,
            'output': output
        }
        
        if success:
            success_count += 1
    
    # Generate summary report
    print(f"\n{'='*80}")
    print(f"Evaluation Completion Statistics")
    print(f"{'='*80}")
    print(f"Total Metrics: {len(metrics_to_run)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(metrics_to_run) - success_count}")
    print(f"Success Rate: {success_count/len(metrics_to_run)*100:.1f}%")
    
    # Generate detailed report
    summary = generate_summary_report(args.paper_name, args.judge_version, results)
    print_evaluation_summary(summary)
    
    # Display failed items
    failed_metrics = [metric for metric, result in results.items() if not result['success']]
    if failed_metrics:
        print(f"\nFailed Evaluation Metrics:")
        for metric in failed_metrics:
            print(f"  - {metric}")
    
    print(f"\nFull evaluation completed!")

if __name__ == '__main__':
    main()
