#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Website Refiner v2
Test qwen.py section HTML optimization
"""

import os
import sys
import json
import argparse 
import yaml
from pathlib import Path
from bs4 import BeautifulSoup
from jinja2 import Environment, StrictUndefined
import shutil
from datetime import datetime

# Add Python path relative to Paper2Web directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import section optimizer
from section_optimizer import SectionOptimizer

def load_json_data(file_path):
    """Load JSON data"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

def optimize_sections_with_new_logic(mapping_file_path, output_dir, safe_logs_dir=None):
    """
    Optimize sections using new logic from testqwen.py
    
    Args:
        mapping_file_path: Path to mapping file
        output_dir: Output directory
        safe_logs_dir: Safe logs directory
    """
    try:
        print(f"Optimizing sections...")
        if safe_logs_dir:
            print(f"Safe logs directory: {safe_logs_dir}")
        
        # Create section optimizer
        optimizer = SectionOptimizer()
        
        # Optimize all sections
        results = optimizer.optimize_all_sections(mapping_file_path, output_dir, safe_logs_dir)
        
        return results
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return None

# Vision agent section_optimizer.py

def reassemble_html_from_mapping(original_html_path, optimized_mapping_path):
    """
    Reassemble HTML from mapping.json
    Use BeautifulSoup to replace sections
    """
    print(f"Reassembling HTML from mapping.json...")
    
    # Read HTML
    with open(original_html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Load mapping data
    with open(optimized_mapping_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all sections
    all_sections = soup.find_all('section')
    print(f"Found {len(all_sections)} sections in HTML")
    print(f"Found {len(mapping_data)} sections in mapping.json")
    
    replaced_count = 0
    for i, section_data in enumerate(mapping_data):
        try:
            # Get element info
            element_info = section_data.get('element_info', {})
            section_id = element_info.get('id', '')
            section_index = element_info.get('index', i)
            
            print(f"Processing Section {i+1}: ID='{section_id}', Index={section_index}")
            
            # Get optimized HTML
            optimized_html = section_data.get('html_snippet', '')
            if not optimized_html:
                print(f"Section {i+1} has no html_snippet")
                continue
            
            # Find target section
            target_section = None
            
            if section_id:
                # Find by ID
                target_section = soup.find('section', id=section_id)
                if target_section:
                    print(f"Found target section by ID: {section_id}")
            
            if not target_section and section_index < len(all_sections):
                # Find by index
                target_section = all_sections[section_index]
                print(f"Found target section by index: {section_index}")
            
            if target_section:
                # Parse optimized HTML
                new_section_soup = BeautifulSoup(optimized_html, 'html.parser')
                new_section = new_section_soup.find('section')
                
                if new_section:
                    # Replace section
                    target_section.replace_with(new_section)
                    replaced_count += 1
                    print(f"Replaced Section {i+1}")
                else:
                    print(f"No section tag found in optimized HTML")
            else:
                print(f"Target section not found: {section_id or f'index_{section_index}'}")
                
        except Exception as e:
            print(f"Error processing Section {i+1}: {e}")
            continue
    
    print(f"Successfully replaced {replaced_count} sections in HTML")
    return str(soup)

def save_refined_website(output_dir, refined_html, original_script_path, original_style_path):
    """
    Save refined website files to output directory
    """
    print(f"Saving refined website...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save HTML file
    html_file = os.path.join(output_dir, 'index.html')
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(refined_html)
    print(f"Saved HTML: {html_file}")
    
    # Copy JavaScript file
    script_file = os.path.join(output_dir, 'script.js')
    shutil.copy2(original_script_path, script_file)
    print(f"Copied JavaScript: {script_file}")
    
    # Copy CSS file
    style_file = os.path.join(output_dir, 'style.css')
    shutil.copy2(original_style_path, style_file)
    print(f"Copied CSS: {style_file}")
    
    print(f"Refined website saved to: {output_dir}")
    return output_dir

def main():
    """Main function - testqwen.py integration"""
    parser = argparse.ArgumentParser(description='Website Refiner v2 - testqwen.py HTML optimization')
    parser.add_argument('--slicer_dir', required=True, help='Slicer directory path')
    parser.add_argument('--model_name', default='openrouter_qwen2_5_vl_32b', help='Model name for testqwen.py')
    parser.add_argument('--output_dir', required=True, help='Output directory path')
    parser.add_argument('--original_html', required=True, help='Original HTML file path')
    parser.add_argument('--original_script', required=True, help='Original JavaScript file path')
    parser.add_argument('--original_style', required=True, help='Original CSS file path')
    parser.add_argument('--safe_logs_dir', help='Safe logs directory path')
    
    args = parser.parse_args()
    
    try:
        print(f'Starting Website Refiner v2 (testqwen.py integration)')
        print(f'Slicer directory: {args.slicer_dir}')
        print(f'Output directory: {args.output_dir}')
        print(f'Original HTML: {args.original_html}')
        
        # Validate input files
        if not os.path.exists(args.slicer_dir):
            raise FileNotFoundError(f"Slicer directory not found: {args.slicer_dir}")
        if not os.path.exists(args.original_html):
            raise FileNotFoundError(f"HTML file not found: {args.original_html}")
        if not os.path.exists(args.original_script):
            raise FileNotFoundError(f"JavaScript file not found: {args.original_script}")
        if not os.path.exists(args.original_style):
            raise FileNotFoundError(f"CSS file not found: {args.original_style}")
        
        # Check mapping.json file
        mapping_file = os.path.join(args.slicer_dir, 'mapping.json')
        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"mapping.json not found: {mapping_file}")
        
        print(f"Input validation completed")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create logs directory
        log_dir = os.path.join(args.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log file
        log_file = os.path.join(log_dir, 'refinement_log.txt')
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Website Refinement Log (testqwen.py)\n")
            f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Slicer directory: {args.slicer_dir}\n")
            f.write(f"Output directory: {args.output_dir}\n")
            f.write(f"{'='*80}\n")
        
        # Optimize sections
        optimization_results = optimize_sections_with_new_logic(mapping_file, log_dir, args.safe_logs_dir)
        
        if not optimization_results:
            raise Exception("Section optimization failed")
        
        print(f"\nOptimization results:")
        print(f"   - Successful optimizations: {optimization_results['successful_optimizations']}")
        print(f"   - Failed optimizations: {optimization_results['failed_optimizations']}")
        print(f"   - Total sections: {optimization_results['total_sections']}")
        
        # Get optimized mapping
        optimized_mapping_path = optimization_results.get('updated_mapping_path')
        if not optimized_mapping_path or not os.path.exists(optimized_mapping_path):
            raise Exception("Optimized mapping.json not found")
        
        # Reassemble HTML from mapping.json
        print(f"\nReassembling HTML from mapping.json...")
        refined_html = reassemble_html_from_mapping(args.original_html, optimized_mapping_path)
        
        # Save refined website
        output_dir = save_refined_website(
            args.output_dir, 
            refined_html, 
            args.original_script, 
            args.original_style
        )
        
        # Copy optimized mapping.json
        final_mapping_path = os.path.join(args.output_dir, 'mapping_optimized.json')
        shutil.copy2(optimized_mapping_path, final_mapping_path)
        
        print(f"\nRefinement completed successfully!")
        print(f"Output directory: {output_dir}")
        print(f"Log file: {log_file}")
        print(f"Optimized mapping: {final_mapping_path}")
        print(f"Success rate: {optimization_results['successful_optimizations']}/{optimization_results['total_sections']}")
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
