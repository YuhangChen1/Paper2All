#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 v2

"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import subprocess

# Add Python path relative to Paper2Web directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ChatAgent
from camel.agents import ChatAgent

def load_json_data(file_path):
    """JSON"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f" JSON: {e}")
        return None

def check_directory_structure(v0_dir):
    """
    v0
    """
    print(f" v0: {v0_dir}")
    
    if not os.path.exists(v0_dir):
        print(f" v0: {v0_dir}")
        return False
    
    # 
    required_files = ['index.html', 'script.js', 'style.css']
    for file_name in required_files:
        file_path = os.path.join(v0_dir, file_name)
        if not os.path.exists(file_path):
            print(f" : {file_path}")
            return False
    
    # slices
    slices_dir = os.path.join(v0_dir, "slices")
    if not os.path.exists(slices_dir):
        print(f" slices: {slices_dir}")
        return False
    
    # mapping.json
    mapping_file = os.path.join(slices_dir, "mapping.json")
    if not os.path.exists(mapping_file):
        print(f" mapping.json: {mapping_file}")
        return False
    
    print(f" v0")
    return True

def create_version_directory(base_dir, version_number, total_sections):
    """
    
    """
    # v{version_number}tov{version_number+1}
    version_dir = os.path.join(base_dir, f"v{version_number}tov{version_number+1}")
    
    # section
    if total_sections == 1:
        dir_name = "allsection1slice"
    else:
        dir_name = f"{total_sections}section1slice"
    
    # 
    transition_dir = os.path.join(version_dir, dir_name)
    
    # 
    os.makedirs(version_dir, exist_ok=True)
    os.makedirs(transition_dir, exist_ok=True)
    
    print(f" : {version_dir}")
    print(f" : {transition_dir}")
    
    return version_dir, transition_dir

def merge_sections(mapping_data, merge_count):
    """
    sectionsmerge_countsection
    """
    print(f" sections{merge_count}...")
    
    merged_sections = []
    total_sections = len(mapping_data)
    
    for i in range(0, total_sections, merge_count):
        end_index = min(i + merge_count, total_sections)
        section_group = mapping_data[i:end_index]
        
        if len(section_group) == 1:
            # section
            merged_sections.append({
                'type': 'single',
                'sections': section_group,
                'html_content': section_group[0]['html_snippet']
            })
        else:
            # sectionHTML
            merged_html = merge_html_sections(section_group)
            merged_sections.append({
                'type': 'merged',
                'sections': section_group,
                'html_content': merged_html
            })
    
    print(f"  {len(merged_sections)} sections")
    return merged_sections

def merge_html_sections(section_group):
    """
    sectionHTML
    """
    merged_html = ""
    
    for section in section_group:
        # section
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(section['html_snippet'], 'html.parser')
        section_tag = soup.find('section')
        
        if section_tag:
            # section
            section_content = ""
            for child in section_tag.children:
                if child.name:  # 
                    section_content += str(child)
            
            merged_html += section_content + "\n"
    
    # section
    merged_html = f'<section class="merged-section">\n{merged_html}</section>'
    return merged_html

def create_merged_html_file(merged_sections, output_dir, original_html_path):
    """
    HTML
    """
    print(f" HTML...")
    
    # HTML
    with open(original_html_path, 'r', encoding='utf-8') as f:
        original_html = f.read()
    
    # BeautifulSoup
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(original_html, 'html.parser')
    
    # section
    sections = soup.find_all('section')
    
    # sections
    section_index = 0
    for merged_section in merged_sections:
        if section_index < len(sections):
            # section
            new_section = BeautifulSoup(merged_section['html_content'], 'html.parser').find('section')
            if new_section:
                sections[section_index].replace_with(new_section)
                section_index += 1
    
    # HTML
    merged_html_path = os.path.join(output_dir, 'index.html')
    with open(merged_html_path, 'w', encoding='utf-8') as f:
        f.write(str(soup))
    
    print(f" HTML: {merged_html_path}")
    return merged_html_path

def copy_assets(output_dir, original_dir):
    """
    CSSJS
    """
    print(f" ...")
    
    # CSS
    css_src = os.path.join(original_dir, 'style.css')
    css_dst = os.path.join(output_dir, 'style.css')
    if os.path.exists(css_src):
        shutil.copy2(css_src, css_dst)
        print(f" CSS")
    
    # JS
    js_src = os.path.join(original_dir, 'script.js')
    js_dst = os.path.join(output_dir, 'script.js')
    if os.path.exists(js_src):
        shutil.copy2(js_src, js_dst)
        print(f" JavaScript")
    
    print(f" ")

def capture_website_screenshot(html_path, output_dir):
    """
    website_capture_full_page.py
    """
    print(f" ...")
    print(f"   HTML: {html_path}")
    print(f"   : {output_dir}")
    
    # HTML
    if not os.path.exists(html_path):
        print(f" HTML: {html_path}")
        return None
    
    # 
    screenshot_path = os.path.join(output_dir, 'screenshot.png')
    print(f"   : {screenshot_path}")
    
    # 
    os.makedirs(output_dir, exist_ok=True)
    
    # website_capture_full_page.py
    cmd = [
        'python', 'website_capture_full_page.py',
        '--html_path', html_path,
        '--output_path', screenshot_path,
        '--viewport_width', '1920',
        '--viewport_height', '1080'
    ]
    
    try:
        print(f"   : {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        print(f"   : {result.returncode}")
        if result.returncode == 0:
            if os.path.exists(screenshot_path):
                print(f" : {screenshot_path}")
                return screenshot_path
            else:
                print(f" : {screenshot_path}")
                return None
        else:
            print(f" :")
            print(f"   : {result.stdout}")
            print(f"   : {result.stderr}")
            return None
    except Exception as e:
        print(f" : {e}")
        import traceback
        traceback.print_exc()
        return None

def slice_website_auto_merge(html_path, screenshot_path, output_dir, mapping_data, is_first_iteration=False):
    """
    range_website_slicer.py
    1234
    """
    print(f" ...")
    
    # range_website_slicer.py
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # section
    if is_first_iteration and len(mapping_data) > 1:
        print(f" ...")
        # 
        from website_slicer import slice_webpage_into_components
        
        # 
        slices_dir = os.path.join(output_dir, "slices")
        os.makedirs(slices_dir, exist_ok=True)
        
        # website_slicer.py
        try:
            component_data = slice_webpage_into_components(
                html_file_path=html_path,
                full_screenshot_path=screenshot_path,
                output_dir=output_dir,
                target_selector="section"
            )
            
            # image_path
            for item in component_data:
                if 'image_path' in item:
                    # 
                    image_filename = os.path.basename(item['image_path'])
                    # slices
                    item['image_path'] = image_filename
            
            # mapping.json
            final_mapping_path = os.path.join(slices_dir, "mapping.json")
            with open(final_mapping_path, 'w', encoding='utf-8') as f:
                json.dump(component_data, f, ensure_ascii=False, indent=2)
            print(f"  {len(component_data)} sections")
            return True
        except Exception as e:
            print(f" : {e}")
            return False
    
    # section
    from range_website_slicer import slice_webpage_range
    
    total_sections = len(mapping_data)
    if total_sections <= 1:
        # section
        print(f" section")
        # 
        slices_dir = os.path.join(output_dir, "slices")
        os.makedirs(slices_dir, exist_ok=True)
        
        # 
        src_slices_dir = os.path.join(os.path.dirname(html_path), "slices")
        if os.path.exists(src_slices_dir):
            for item in os.listdir(src_slices_dir):
                src_file = os.path.join(src_slices_dir, item)
                dst_file = os.path.join(slices_dir, item)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
        return True
    
    # 
    slices_dir = os.path.join(output_dir, "slices")
    os.makedirs(slices_dir, exist_ok=True)
    
    merged_data = []
    i = 0
    group_index = 0
    
    # 
    while i < total_sections:
        if i + 1 < total_sections:
            # section
            start_index = i
            end_index = i + 1
            print(f" section {start_index}  {end_index}")
            i += 2
        else:
            # section
            start_index = i
            end_index = i
            print(f" section {start_index}")
            i += 1
        
        # 
        temp_output_dir = os.path.join(output_dir, f"temp_group_{group_index}")
        os.makedirs(temp_output_dir, exist_ok=True)
        
        try:
            # range_website_slicer
            slice_webpage_range(
                html_file_path=html_path,
                full_screenshot_path=screenshot_path,
                output_dir=temp_output_dir,
                start_index=start_index,
                end_index=end_index,
                target_selector="section"
            )
            
            # slices
            temp_slices_dir = os.path.join(temp_output_dir, "slices")
            temp_mapping_file = os.path.join(temp_slices_dir, "mapping.json")
            
            if os.path.exists(temp_mapping_file):
                temp_mapping_data = load_json_data(temp_mapping_file)
                if temp_mapping_data and len(temp_mapping_data) > 0:
                    #  image_path
                    for item in temp_mapping_data:
                        if 'image_path' in item:
                            # 
                            image_filename = os.path.basename(item['image_path'])
                            #  slices 
                            item['image_path'] = image_filename
                            print(f" : {item['image_path']}")
                        else:
                            print("   image_path ")

                    merged_data.extend(temp_mapping_data)

                    #  slices 
                    if os.path.exists(temp_slices_dir):
                        #  PNG  slices 
                        for item in os.listdir(temp_slices_dir):
                            if item.endswith('.png'):
                                src_file = os.path.join(temp_slices_dir, item)
                                dst_file = os.path.join(slices_dir, item)
                                if os.path.exists(src_file):
                                    shutil.copy2(src_file, dst_file)
                                    print(f" : {item}")
                                else:
                                    print(f" :  {src_file} ")
                    else:
                        print(f" :  slices  {temp_slices_dir} ")
            # 
            shutil.rmtree(temp_output_dir)
            group_index += 1
            
        except Exception as e:
            print(f" section {start_index}{end_index}: {e}")
            # 
            if os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir)
            return False
    
    # mapping.json
    final_mapping_path = os.path.join(slices_dir, "mapping.json")
    try:
        with open(final_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print(f"  {len(merged_data)} sections")
        return True
    except Exception as e:
        print(f" : {e}")
        return False

def slice_website(html_path, screenshot_path, output_dir):
    """
    website_slicer.py
    """
    print(f" ...")
    
    # website_slicer.py
    cmd = [
        'python', 'website_slicer.py',
        '--html_path', html_path,
        '--screenshot_path', screenshot_path,
        '--output_dir', output_dir,
        '--selector', 'section'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            print(f" : {output_dir}")
            return True
        else:
            print(f" : {result.stderr}")
            return False
    except Exception as e:
        print(f" : {e}")
        return False

def run_refinement_cycle(version_dir, slicer_dir, model_name, log_file, safe_logs_dir=None):
    """
    
    """
    print(f" ...")
    
    # 
    refined_output_dir = os.path.join(version_dir, "refined")
    os.makedirs(refined_output_dir, exist_ok=True)
    
    # website_refiner_v2.py
    cmd = [
        'python', 'website_refiner_v2.py',
        '--slicer_dir', slicer_dir,
        '--model_name', model_name,  # 
        '--output_dir', refined_output_dir,
        '--original_html', os.path.join(version_dir, 'index.html'),
        '--original_script', os.path.join(version_dir, 'script.js'),
        '--original_style', os.path.join(version_dir, 'style.css')
    ]
    
    #  website_refiner_v2.py
    if safe_logs_dir:
        cmd.extend(['--safe_logs_dir', safe_logs_dir])
        print(f" refiner: {safe_logs_dir}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            print(f" ")
            
            # 
            for file_name in ['index.html', 'script.js', 'style.css']:
                src_file = os.path.join(refined_output_dir, file_name)
                dst_file = os.path.join(version_dir, file_name)
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dst_file)
                    print(f" : {file_name}")
            
            # 
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f": {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f": {version_dir}\n")
                f.write(f"{'='*60}\n")
            
            #  qwen
            # safe_logs_dirrefined_output_dir
            print(f" : {refined_output_dir}")
            print(f" ")
            shutil.rmtree(refined_output_dir)
            
            return True
        else:
            print(f" : {result.stderr}")
            print(f"   : {result.stdout}")
            # 
            if os.path.exists(refined_output_dir):
                shutil.rmtree(refined_output_dir)
            return False
    except Exception as e:
        print(f" : {e}")
        # 
        if os.path.exists(refined_output_dir):
            shutil.rmtree(refined_output_dir)
        return False

def get_agent_config(model_name):
    """"""
    print(f" : {model_name}")
    
    if model_name == 'openrouter_qwen3_14b':
        config = {
            'model_type': 'qwen/qwen2.5-vl-32b-instruct:free',
            'model_platform': 'openrouter',
            'model_config': {
                'max_tokens': 32768,  # token
                'temperature': 0.7
            },
        }
        print(f" :")
        print(f"   - model_type: {config['model_type']}")
        print(f"   - model_platform: {config['model_platform']}")
        return config
    elif model_name == 'openrouter_qwen2_5_vl_32b':
        config = {
            'model_type': 'qwen/qwen2.5-vl-32b-instruct:free',
            'model_platform': 'openrouter',
            'model_config': {
                'max_tokens': 32768,  # token
                'temperature': 0.7
            },
        }
        print(f" :")
        print(f"   - model_type: {config['model_type']}")
        print(f"   - model_platform: {config['model_platform']}")
        return config
    else:
        raise ValueError(f": {model_name}")

def create_vision_agent(model_name):
    """"""
    try:
        print(f" ...")
        agent_config = get_agent_config(model_name)
        print(f" : {agent_config}")
        
        # OpenRouter
        if 'OPENROUTER_API_KEY' in os.environ:
            os.environ['OPENAI_API_KEY'] = os.environ['OPENROUTER_API_KEY']
            print(f" OPENAI_API_KEY")
        
        # 
        print(f" ...")
        from camel.models.openrouter_model import OpenRouterModel
        model = OpenRouterModel(
            model_type=agent_config['model_type'],
            model_config_dict=agent_config['model_config'],  # model_config_dict
            api_key=os.environ.get('OPENROUTER_API_KEY')
        )
        print(f" : {type(model)}")
        
        # 
        print(f" ChatAgent...")
        agent = ChatAgent(
            model=model,
            system_message="You are an expert web developer and UI/UX designer specializing in analyzing and optimizing web layouts."
        )
        print(f" ChatAgent: {type(agent)}")
        return agent
    except Exception as e:
        print(f" : {e}")
        return None

def main():
    """"""
    parser = argparse.ArgumentParser(description=' v2')
    parser.add_argument('--base_dir', required=True, help='')
    # version_namev0
    parser.add_argument('--model_name', default='openrouter_qwen2_5_vl_32b', help='')
    parser.add_argument('--max_iterations', type=int, default=5, help='')
    
    args = parser.parse_args()
    
    try:
        print(f'  v2 ')
        print(f' : {args.base_dir}')
        print(f' : {args.model_name}')
        print(f' : {args.max_iterations}')
        
        # v0
        v0_dir = os.path.join(args.base_dir, "v0")
        if not check_directory_structure(v0_dir):
            raise Exception("")
        
        print(f" v0: {v0_dir}")
        
        #  
        safe_logs_base_dir = os.path.join(args.base_dir, 'qwen_analysis_logs')
        os.makedirs(safe_logs_base_dir, exist_ok=True)
        print(f" : {safe_logs_base_dir}")
        
        # t
        log_file = os.path.join(args.base_dir, 'optimization_log.txt')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"\n")
            f.write(f": {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f": {args.model_name}\n")
            f.write(f": {safe_logs_base_dir}\n")
            f.write(f"{'='*60}\n")
        
        # 
        current_version = 0
        iteration_count = 0
        for iteration in range(args.max_iterations):
            print(f"\n{'='*80}")
            print(f"  {iteration + 1} ")
            print(f"{'='*80}")
            
            #  
            iteration_logs_dir = os.path.join(safe_logs_base_dir, f"iteration_{iteration + 1}")
            os.makedirs(iteration_logs_dir, exist_ok=True)
            print(f" {iteration + 1}: {iteration_logs_dir}")
            
            # 
            if current_version == 0:
                # v0
                source_html_path = os.path.join(args.base_dir, 'v0', 'index.html')
                source_css_path = os.path.join(args.base_dir, 'v0', 'style.css')
                source_js_path = os.path.join(args.base_dir, 'v0', 'script.js')
                prev_slicer_dir = os.path.join(args.base_dir, 'v0', 'slices')
                is_first_iteration = True
            else:
                # 
                prev_version_dir = os.path.join(args.base_dir, f"v{current_version-1}tov{current_version}")
                # allsection1slice
                prev_dirs = [d for d in os.listdir(prev_version_dir) if os.path.isdir(os.path.join(prev_version_dir, d))]
                if not prev_dirs:
                    raise Exception(f": {prev_version_dir}")
                
                # 
                last_dir = sorted(prev_dirs)[-1]
                prev_result_dir = os.path.join(prev_version_dir, last_dir)
                source_html_path = os.path.join(prev_result_dir, 'index.html')
                source_css_path = os.path.join(prev_result_dir, 'style.css')
                source_js_path = os.path.join(prev_result_dir, 'script.js')
                prev_slicer_dir = os.path.join(prev_result_dir, 'slices')
                is_first_iteration = False
            
            # 
            if not os.path.exists(source_html_path):
                raise Exception(f"HTML: {source_html_path}")
            
            # mapping.json
            mapping_file = os.path.join(prev_slicer_dir, 'mapping.json')
            if not os.path.exists(mapping_file):
                raise Exception(f"mapping.json: {mapping_file}")
            
            mapping_data = load_json_data(mapping_file)
            if not mapping_data:
                raise Exception("mapping.json")
                
            print(f" mapping.json {len(mapping_data)} sections")
            
            # section
            if len(mapping_data) <= 1:
                print(" sections")
                break
                
            # sections
            section_count = len(mapping_data)
            version_dir = os.path.join(args.base_dir, f"v{current_version}tov{current_version+1}")
            
            # 
            transition_dir_name = f"{section_count}section1slice"
            transition_dir = os.path.join(version_dir, transition_dir_name)
            os.makedirs(transition_dir, exist_ok=True)
            
            # 
            shutil.copy2(source_html_path, os.path.join(transition_dir, 'index.html'))
            if os.path.exists(source_css_path):
                shutil.copy2(source_css_path, os.path.join(transition_dir, 'style.css'))
            if os.path.exists(source_js_path):
                shutil.copy2(source_js_path, os.path.join(transition_dir, 'script.js'))

            # 
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n {iteration + 1} \n")
                f.write(f": {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f": {version_dir}\n")
                f.write(f": {transition_dir}\n")

            # HTML
            screenshot_path = capture_website_screenshot(os.path.join(transition_dir, 'index.html'), transition_dir)
            if not screenshot_path:
                raise Exception("")

            # HTML
            if not slice_website_auto_merge(
                os.path.join(transition_dir, 'index.html'), 
                screenshot_path, 
                transition_dir, 
                mapping_data,
                is_first_iteration
            ):
                raise Exception("")
            
            # slicer.pytransition_dirslicesmapping.jsonslices
            new_slicer_dir = os.path.join(transition_dir, 'slices')
            
            # 
            if not run_refinement_cycle(transition_dir, new_slicer_dir, args.model_name, log_file, iteration_logs_dir):
                raise Exception("")
            
            # 
            current_version += 1
            iteration_count += 1
        
        print(f"\n !")
        print(f"  {current_version} ")
        print(f" : {log_file}")
        
    except Exception as e:
        print(f' : {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
