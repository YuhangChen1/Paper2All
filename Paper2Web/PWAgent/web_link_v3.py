#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Link v3 - Website Iterative Optimization System v3
Fixed duplicate merge bug, implemented correct replacement instead of accumulation logic
- v3 Feature: Rule-based component type determination
- v3 Feature: header/nav only optimized without merging, sections merged in pairs
- v3 Feature: Complete replacement each round instead of accumulation, avoiding duplicate elements
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import math

# Add Python path relative to Paper2Web directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing tools
from website_capture_full_page import capture_full_page_screenshot
from website_slicer import slice_webpage_into_components
from range_website_slicer import slice_webpage_range
from api_integrations_v2 import create_vision_analyzer_v2, create_code_optimizer_v2  # Use v2 version
from image_path_fixer import fix_image_paths_before_screenshot

class WebsiteIterativeOptimizerV3:
    """Website Iterative Optimizer v3 - Fixed duplicate merge bug"""
    
    def __init__(self, v0_dir: str, vision_model: str = "openrouter_qwen2_5_VL_72B",
                 code_model: str = "openrouter_qwen3_coder", max_try: int = 0):
        self.v0_dir = Path(v0_dir)
        self.vision_model = vision_model
        self.code_model = code_model
        self.max_try = max_try
        self.base_dir = self.v0_dir.parent
        self.v02v1_dir = self.base_dir / "v02v1_v3" 
        self.v1_dir = self.base_dir / "v1_v3"       
        
        # Create necessary directories
        self.v02v1_dir.mkdir(exist_ok=True)
        self.v1_dir.mkdir(exist_ok=True)
        
        # Log files
        self.log_file = self.base_dir / "optimization_log_v3.txt"
        self.html_evolution_log = self.base_dir / "html_evolution_log_v3.txt"
        self.init_logging()
        
        # v3 Feature: Store optimization results from previous round to avoid accumulation
        self.current_optimized_headers = []
        self.current_optimized_navs = []
    
    def init_logging(self):
        """Initialize logging"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Website Iterative Optimization System v3\n")
            f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"v0 directory: {self.v0_dir}\n")
            f.write(f"Vision model: {self.vision_model}\n")
            f.write(f"Code model: {self.code_model}\n")
            f.write(f"v3 New Features: Fixed duplicate merge bug + rule-based component classification\n")
            f.write(f"v3 Feature: header/nav only optimized without merging + sections merged in pairs\n")
            f.write(f"v3 Feature: Complete replacement each round to avoid accumulation\n")
            f.write("="*60 + "\n")
        
        # 初始化HTML演进日志
        with open(self.html_evolution_log, 'w', encoding='utf-8') as f:
            f.write(f"HTML Code Evolution Record (v3)\n")
            f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"v3 Feature: Fixed duplicate merge bug, rule-based classification\n")
            f.write(f"Complete HTML code after each optimization round\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message: str, level: str = "INFO"):
        """Write log"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
    
        if level == "SUCCESS":
            prefix = "✅"
        elif level == "ERROR":
            prefix = "❌"
        elif level == "WARNING":
            prefix = "⚠️"
        elif level == "PROGRESS":
            prefix = "🚀"
        elif level == "VISION":
            prefix = "👁️"
        elif level == "CODE":
            prefix = "💻"
        elif level == "SLICE":
            prefix = "🔪"
        elif level == "MERGE":
            prefix = "🔗"
        elif level == "RULE":
            prefix = "📋"
        elif level == "V3_FIX":
            prefix = "🔧"
        else:
            prefix = "ℹ️"
        
        log_msg = f"[{timestamp}] {prefix} {message}"
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    
    def log_html_evolution(self, round_num: int, html_content: str, description: str = ""):
        """Record HTML evolution process"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.html_evolution_log, 'a', encoding='utf-8') as f:
            if description:
                f.write(f"Description: {description}\n")
            f.write("-" * 80 + "\n")
            f.write(html_content)
            f.write("\n" + "=" * 80 + "\n\n")
    
    def ensure_image_directories(self, target_dir: Path, source_dir: Path = None) -> bool:

        if source_dir is None:
            source_dir = self.v0_dir
        
        self.log(f"📂 Check image directory: {target_dir}")
        
        # Find image directories in source directory
        source_image_dirs = []
        for item in source_dir.iterdir():
            if item.is_dir() and any(keyword in item.name.lower() for keyword in ['image', 'table', 'figure', 'img']):
                source_image_dirs.append(item)
        
        if not source_image_dirs:
            self.log("No image directories found in source directory, skip copying")
            return True
        
        # 检查Target directory是否已存在图片目录
        target_image_dirs = []
        for item in target_dir.iterdir():
            if item.is_dir() and any(keyword in item.name.lower() for keyword in ['image', 'table', 'figure', 'img']):
                target_image_dirs.append(item)
        
        # 如果Target directory已有图片目录，检查是否需要更新
        if target_image_dirs:
            self.log(f"Target directory EXITS: {[d.name for d in target_image_dirs]}")
            return True
        
        # Copy image directories
        copied_count = 0
        for src_dir in source_image_dirs:
            dst_dir = target_dir / src_dir.name
            try:
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)
                self.log(f"Copy image directory: {src_dir.name} -> {dst_dir}")
                copied_count += 1
            except Exception as e:
                self.log(f"Failed to copy image directory {src_dir.name}: {e}")
        
        if copied_count > 0:
            return True
        else:
            self.log("No image directories copied successfully")
            return False
    
    def validate_v0_directory(self) -> bool:
        """Validate v0 directory structure"""
        self.log(f"验证v0 directory: {self.v0_dir}")
        
        if not self.v0_dir.exists():
            self.log(f"Error: v0 directory does not exist: {self.v0_dir}")
            return False
        
        # Check required files - v3 supports single-file HTML format
        required_files = ['index.html']  # Only require index.html file
        optional_files = ['script.js', 'style.css']  # CSS/JS can be embedded in HTML
        
        for file_name in required_files:
            file_path = self.v0_dir / file_name
            if not file_path.exists():
                self.log(f"Error: Missing required file: {file_path}")
                return False
        
        # Check optional files and record status
        separate_files_exist = False
        for file_name in optional_files:
            file_path = self.v0_dir / file_name
            if file_path.exists():
                separate_files_exist = True
                self.log(f"Found separate file: {file_name}")
        
        if not separate_files_exist:
            self.log("🔧 v3 Feature: Detected single-file HTML format (CSS/JS embedded)", "V3_FIX")
        

        image_dirs = list(self.v0_dir.glob("*images*")) + list(self.v0_dir.glob("*_images_and_tables"))
        if not image_dirs:
            self.log("Warning: No image directories found, but continue execution")
        else:
            self.log(f"Found image directories: {[str(d) for d in image_dirs]}")
        
        self.log("v0 directory validation passed")
        return True
    
    def run_v0_processing(self) -> Tuple[str, List[Dict]]:
        """Execute v0 stage processing: screenshot and slicing"""
        self.log("Starting v0 stage processing...")
        

        self.log("Check v0 directory image resources...")
        
        # 2. Screenshot
        html_path = str(self.v0_dir / "index.html")
        screenshot_path = str(self.v0_dir / "screenshot.png")
        
        # Fix image paths
        self.log("🖼️  Check and fix image paths...")
        fix_image_paths_before_screenshot(html_path, self.log)
        
        self.log("生成页面Screenshot...")
        success = capture_full_page_screenshot(
            html_file_path=html_path,
            output_image_path=screenshot_path,
            viewport_width=1920,
            viewport_height=1080
        )
        
        if not success:
            raise Exception("")
        
        self.log(f"Screenshot保存至: {screenshot_path}")
        
        # 3. Slice
        self.log("开始页面Slice...")
        component_data = slice_webpage_into_components(
            html_file_path=html_path,
            full_screenshot_path=screenshot_path,
            output_dir=str(self.v0_dir),
            target_selector="header, nav, section"
        )
        
        if not component_data:
            raise Exception("页面Slice失败")
        
        self.log(f"Slice完成，共生成 {len(component_data)} 个Slice")
        
        return screenshot_path, component_data
    
    def create_section_merge_directory(self, section_count: int) -> Path:
        """Create ksection1slice directory"""
        dir_name = f"{section_count}section1slice"
        section_dir = self.v02v1_dir / dir_name
        section_dir.mkdir(exist_ok=True)
        self.log(f"Create directory: {section_dir}")
        return section_dir
    
    def separate_elements_by_type(self, mapping_data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:

        headers = []
        navs = []
        sections = []
        
        for item in mapping_data:
            element_info = item.get('element_info', {})
            tag_name = element_info.get('tag_name', '').lower()
            
            if tag_name == 'header':
                headers.append(item)
            elif tag_name == 'nav':
                navs.append(item)
            elif tag_name == 'section':
                sections.append(item)
            else:
                # Other types default to section processing
                self.log(f"")
                sections.append(item)
        
        return headers, navs, sections
    
    def merge_sections_two_by_two(self, sections: List[Dict], target_count: int) -> List[Dict]:
 
        if target_count == 1:
            # If target is 1, no merging needed, but need to convert data structure to match expected format
            converted_sections = []
            for i, section in enumerate(sections):
                converted_sections.append({
                    'type': 'single',
                    'original_indices': [section['element_info']['index']],
                    'sections': [section],
                    'html_content': section['html_snippet'],  # Convert field name
                    'merged_index': i
                })
                self.log(f"   Convert section: index {section['element_info']['index']}")
            return converted_sections
        

        merged_sections = []
        for i in range(0, len(sections), target_count):
            end_index = min(i + target_count, len(sections))
            section_group = sections[i:end_index]
            
            if len(section_group) == 1:
                # Single section, use directly
                merged_sections.append({
                    'type': 'single',
                    'original_indices': [section_group[0]['element_info']['index']],
                    'sections': section_group,
                    'html_content': section_group[0]['html_snippet'],
                    'merged_index': len(merged_sections)
                })
                self.log(f"   Keep separate section: index {section_group[0]['element_info']['index']}")
            else:
                # Multiple sections, merge HTML content
                merged_html = self.merge_html_sections_clean(section_group)
                original_indices = [s['element_info']['index'] for s in section_group]
                merged_sections.append({
                    'type': 'merged',
                    'original_indices': original_indices,
                    'sections': section_group,
                    'html_content': merged_html,
                    'merged_index': len(merged_sections)
                })
                self.log(f"   Merge sections: indices {original_indices}")
        
        return merged_sections
    
    def merge_html_sections_clean(self, section_group: List[Dict]) -> str:

        from bs4 import BeautifulSoup
        
        # If only one section, return directly
        if len(section_group) == 1:
            return section_group[0]['html_snippet']
        
        # 🔧 Fix 2: Multiple sections truly merged into 1 section
        
        # Collect all section content (excluding section tags themselves)
        merged_content_parts = []
        first_section_attrs = {}  # Save first section's attributes
        
        for i, section in enumerate(section_group):
            html_snippet = section['html_snippet']
            soup = BeautifulSoup(html_snippet, 'html.parser')
            
            # 找到section标签
            section_tag = soup.find('section')
            if section_tag:
                # If first section, save its attributes
                if i == 0:
                    first_section_attrs = dict(section_tag.attrs)
                    # Generate merged ID
                    original_ids = [s['element_info']['index'] for s in section_group]
                    merged_id = f"merged-sections-{'-'.join(map(str, original_ids))}"
                    first_section_attrs['id'] = merged_id
                
                # v3 key fix: Ensure only section content is taken, excluding embedded header/nav
                embedded_headers = section_tag.find_all('header')
                embedded_navs = section_tag.find_all('nav')
                
                if embedded_headers or embedded_navs:
                    # Remove embedded header and nav
                    for embedded_header in embedded_headers:
                        embedded_header.extract()
                    for embedded_nav in embedded_navs:
                        embedded_nav.extract()
                
                # Extract section inner content
                section_inner_html = ''.join(str(child) for child in section_tag.children)
                merged_content_parts.append(section_inner_html)
            else:
                # If no section tag, use original content
                clean_content = html_snippet
                # Remove possible header/nav tags
                soup_clean = BeautifulSoup(clean_content, 'html.parser')
                for header in soup_clean.find_all('header'):
                    header.extract()
                for nav in soup_clean.find_all('nav'):
                    nav.extract()
                merged_content_parts.append(str(soup_clean))
        
        # 🔧 Key fix: Create 1 merged section
        merged_content = '\n'.join(merged_content_parts)
        
        # Build attribute string
        attrs_str = ' '.join([f'{k}="{v}"' for k, v in first_section_attrs.items()])
        
        merged_section = f'<section {attrs_str}>\n{merged_content}\n</section>'
        
        original_indices = [s['element_info']['index'] for s in section_group]
        
        return merged_section
    
    def create_clean_html_file(self, headers: List[Dict], navs: List[Dict], 
                              merged_sections: List[Dict], output_dir: Path) -> str:

        from bs4 import BeautifulSoup
        
        
        # Read v0 version as clean template
        v0_html_path = self.v0_dir / "index.html"
        with open(v0_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Clear existing header, nav, section content
        for header in soup.find_all('header'):
            header.extract()
        for nav in soup.find_all('nav'):
            nav.extract()
        for section in soup.find_all('section'):
            section.extract()
        
        self.log(" Clear original header/nav/section content")
        
        # Find body or main container
        body = soup.find('body')
        main_container = soup.find('main')
        if not main_container:
            main_container = body
        
        if not main_container:
            raise Exception("未Find body or main container")
        
        # v3特性：Insert optimized elements in correct order 
        # 🔧 Fix: header and nav must be inserted at body beginning, maintain correct order
        
        insert_position = 0  # Track current insertion position
        
        # 1. Insert headers to body beginning first
        for i, header_data in enumerate(headers):
            header_soup = BeautifulSoup(header_data['html_snippet'], 'html.parser')
            header_element = header_soup.find('header')
            if header_element:
                # Ensure ID uniqueness
                header_id = header_element.get('id', f'optimized-header-{i}')
                header_element['id'] = header_id
                # 🔧 Key fix: Insert in order to body beginning
                body.insert(insert_position, header_element)
                insert_position += 1
        
        # 2. Insert navs after headers
        for i, nav_data in enumerate(navs):
            nav_soup = BeautifulSoup(nav_data['html_snippet'], 'html.parser')
            nav_element = nav_soup.find('nav')
            if nav_element:
                # Ensure ID uniqueness
                nav_id = nav_element.get('id', f'optimized-nav-{i}')
                nav_element['id'] = nav_id
                # 🔧 Key fix: Insert to current position
                body.insert(insert_position, nav_element)
                insert_position += 1
        
        # 3. Insert merged sections to main container
        for merged_section in merged_sections:
            merged_html = merged_section['html_content']
            sections_soup = BeautifulSoup(merged_html, 'html.parser')
            
            # Handle cases that may contain multiple sections
            all_sections = sections_soup.find_all('section')
            for section_element in all_sections:
                section_id = section_element.get('id', f'section-{len(merged_sections)}')
                section_element['id'] = section_id
                main_container.append(section_element)
                self.log(f"✅ Insert merged section: {section_id}")
        
        # Save HTML file
        html_path = output_dir / "index.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(str(soup))
        
        # Copy CSS and JS files (v3 supports single-file HTML)
        for file_name in ['style.css', 'script.js']:
            src_file = self.v0_dir / file_name
            dst_file = output_dir / file_name
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                self.log(f"✅ Copy separate file: {file_name}")
        
        # Copy image directories
        self.ensure_image_directories(output_dir, self.v0_dir)
        
        self.log(f"🔧 v3 Clean HTML file generation completed: {html_path}", "V3_FIX")
        return str(html_path)
    
    def optimize_elements_separately(self, section_dir: Path, headers: List[Dict], 
                                   navs: List[Dict], sections: List[Dict]) -> Dict:

        self.log("🔧 v3 Feature: Optimize header/nav (no merge) and section (merge) separately", "V3_FIX")
        
        # Ensure image directories exist
        self.ensure_image_directories(section_dir)
        
        # 生成当前阶段的Screenshot
        html_path = section_dir / "index.html"
        screenshot_path = section_dir / "screenshot.png"
        
        # Fix image paths
        self.log("🖼️  Check and fix image paths...")
        fix_image_paths_before_screenshot(str(html_path), self.log)
        
        capture_full_page_screenshot(
            html_file_path=str(html_path),
            output_image_path=str(screenshot_path),
            viewport_width=1920,
            viewport_height=1080
        )
        
        # 生成Slice（包含header、nav、section）
        self.log("Slice网页...", "SLICE")
        component_data = slice_webpage_into_components(
            html_file_path=str(html_path),
            full_screenshot_path=str(screenshot_path),
            output_dir=str(section_dir),
            target_selector="header, nav, section"
        )
        
        # Re-classify component data to match current HTML structure
        current_headers, current_navs, current_sections = self.separate_elements_by_type(component_data)
        
        # v3 Feature: Optimize header (no merge, direct optimization)
        optimized_headers = []
        for header_component in current_headers:
            # 🔧 Fix 1: Use original index instead of enumerate's i
            element_index = header_component['element_info']['index']
            slice_name = f"header_{element_index:02d}"
            optimized_header = self.optimize_single_component(
                header_component, section_dir, slice_name, "Header/Hero"
            )
            if optimized_header:
                optimized_headers.append(optimized_header)
        
        # v3 Feature: Optimize nav (no merge, direct optimization)
        optimized_navs = []
        for nav_component in current_navs:
            # 🔧 Fix 1: Use original index instead of enumerate's i
            element_index = nav_component['element_info']['index']
            slice_name = f"nav_{element_index:02d}"
            optimized_nav = self.optimize_single_component(
                nav_component, section_dir, slice_name, "Navigator"
            )
            if optimized_nav:
                optimized_navs.append(optimized_nav)
        
        # v3 Feature: Optimize section (already merged, now optimize)
        optimized_sections = []
        for section_component in current_sections:
            # 🔧 Fix 1: Use original index instead of enumerate's i
            element_index = section_component['element_info']['index']
            slice_name = f"section_{element_index:02d}"
            optimized_section = self.optimize_single_component(
                section_component, section_dir, slice_name, "Content Block"
            )
            if optimized_section:
                optimized_sections.append(optimized_section)
        
        # Generate new HTML using optimized elements
        self.log("📝 Generate optimized HTML...")
        optimized_html_path = self.create_clean_html_file(
            optimized_headers, optimized_navs, 
            [{'html_content': comp['html_snippet']} for comp in optimized_sections],
            section_dir
        )
        
        # Rename to index_new.html
        new_html_path = section_dir / "index_new.html"
        shutil.copy2(optimized_html_path, new_html_path)
        
        # v3 Feature: Update current optimization results to avoid next accumulation
        self.current_optimized_headers = optimized_headers
        self.current_optimized_navs = optimized_navs
        
        return {
            'optimized_html_path': str(new_html_path),
            'optimized_headers': optimized_headers,
            'optimized_navs': optimized_navs,
            'optimized_sections': optimized_sections
        }
    
    def optimize_single_component(self, component: Dict, section_dir: Path, 
                                 slice_name: str, rule_category: str) -> Optional[Dict]:
        """
        v3特性：Optimize single component using rule-based classification
        
        Args:
            component: Component data
            section_dir: Working directory
            slice_name: Slice名称
            rule_category: Rule-determined category
            
        Returns:
            优化后的Component data
        """
        # 构建Slice文件路径
        slice_image_path = section_dir / "slices" / f"{slice_name}.png"
        
        if not slice_image_path.exists():
            return component
        
        
        # Use v2 API for visual analysis (based on rule classification)
        if not hasattr(self, 'vision_analyzer'):
            self.vision_analyzer = create_vision_analyzer_v2(self.vision_model, self.log)
        
        # v3 Feature: Visual analysis, but category determined by rules
        analysis = self.vision_analyzer.analyze_website_image(str(slice_image_path), component)
        
        # Verify rule classification is correctly applied
        if analysis.get('category') != rule_category:
            analysis['category'] = rule_category
        
        html_content = component['html_snippet']
        
        
        # Use v2 API for code optimization
        if not hasattr(self, 'code_optimizer'):
            self.code_optimizer = create_code_optimizer_v2(self.code_model, self.log)
        
        optimized_html = self.code_optimizer.optimize_html_code(
            html_content, 
            rule_category
        )
        
        # v3 Feature: If optimization fails, use original content
        if not optimized_html or optimized_html == html_content:
            self.log(f"     ⚠️ Code optimization unchanged, keep original")
            return component
        else:
            self.log(f"     ✅ Code optimization completed")
            
        # Return optimized component
        result = component.copy()
        result['html_snippet'] = optimized_html
        return result
    
    def run_iterative_optimization(self) -> str:
        """Run complete iterative optimization process v3"""
        self.log("Start iterative optimization process v3...", "PROGRESS")
        self.log("🔧 v3 Feature: Fixed duplicate merge bug + rule-based classification", "V3_FIX")
        self.log(f"🔧 Max try parameter: {self.max_try}", "V3_FIX")

        # Early exit logic: if max_try is 0, directly copy v0 to v1
        if self.max_try == 0:
            self.log("🔧 v3: max_try=0, skip optimization, directly copy v0 to v1", "V3_FIX")
            self.generate_final_version_v3_direct_copy()
            self.log("Direct copy completed, optimization skipped", "SUCCESS")
            return str(self.v1_dir)

        # 1. v0 stage processing
        self.log("📸 执行v0 stage processing（Screenshot和Slice）...")
        screenshot_path, initial_mapping = self.run_v0_processing()
        
        # v3 Feature: Separate initial elements
        initial_headers, initial_navs, initial_sections = self.separate_elements_by_type(initial_mapping)
        
        # 2. v02v1 iterative optimization
        n_sections = len(initial_sections)     
        # Calculate required iteration count (1, 2, 4, 8, ..., n)
        merge_counts = []
        count = 1
        while count <= n_sections:
            merge_counts.append(count)
            count *= 2

        
        # Initialize current state
        current_headers = initial_headers
        current_navs = initial_navs  
        current_sections = initial_sections
        
        # Execute optimization for each stage
        for i, merge_count in enumerate(merge_counts, 1):

            section_dir = self.create_section_merge_directory(merge_count)
            
            # v3 Feature: sections merged in pairs

            merged_sections = self.merge_sections_two_by_two(current_sections, merge_count)
            
            # v3 Feature: Create clean HTML (header/nav unchanged, sections use merged ones)
            self.log("📝 Create clean HTML file (avoid accumulation)...")
            clean_html_path = self.create_clean_html_file(
                current_headers, current_navs, merged_sections, section_dir
            )
            
            # v3 Feature: Optimize different component types separately
            self.log("🎨 Start AI analysis and optimization (based on rule classification)...")
            optimization_result = self.optimize_elements_separately(
                section_dir, current_headers, current_navs, merged_sections
            )
            
            # v3 Feature: Update current state (complete replacement, no accumulation)
            current_headers = optimization_result['optimized_headers']
            current_navs = optimization_result['optimized_navs']
            current_sections = optimization_result['optimized_sections']
            
            
            # Record HTML code after each optimization round
            try:
                final_html_path = optimization_result['optimized_html_path']
                with open(final_html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                description = f"v3修复: Headers({len(current_headers)}), Navs({len(current_navs)}), 合并Sections({merge_count}), 规则分类, 替换非累积"
                self.log_html_evolution(i, html_content, description)
            except Exception as e:
                self.log(f"⚠️ Failed to record HTML evolution log: {e}", "WARNING")
            

        
        # 3. Generate final version
        self.log("🏁 Generate final v3 version...", "PROGRESS")
        final_optimization_result = optimization_result  # 使用最后一轮的结果
        self.generate_final_version_v3(final_optimization_result['optimized_html_path'])
        
        self.log("Iterative optimization v3 completed!", "SUCCESS")
        self.log("🔧 v3 Fix Summary: Resolved duplicate merge bug, implemented correct replacement logic", "V3_FIX")
        return str(self.v1_dir)
    
    def generate_final_version_v3(self, final_html_path: str):
        """Generate final version到v1目录 v3"""
        self.log("Generate final version v3...")
        
        # Ensure using absolute path
        final_html_path = str(Path(final_html_path).resolve())
        src_dir = Path(final_html_path).parent
        
        
        # Copy main files (v3 supports single-file HTML)
        # Required files
        required_files = ['index_new.html', 'index.html']
        # Optional files (may be embedded in HTML)
        optional_files = ['style.css', 'script.js']
        
        for file_name in required_files:
            src_file = src_dir / file_name
            if file_name == 'index_new.html':
                dst_file = self.v1_dir / 'index.html'  # Rename to index.html
            else:
                dst_file = self.v1_dir / file_name
            
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
        
        for file_name in optional_files:
            src_file = src_dir / file_name
            dst_file = self.v1_dir / file_name
            
            if src_file.exists():
                shutil.copy2(src_file, dst_file)

        
        # Copy image directories
        self.ensure_image_directories(self.v1_dir, src_dir)
        
        # 生成最终Screenshot
        final_screenshot = self.v1_dir / "final_screenshot_v3.png"
        
        # Ensure image directories exist后再Screenshot
        self.ensure_image_directories(self.v1_dir)
        
        # Fix final version image paths
        final_index_path = str(self.v1_dir / "index.html")
        self.log("🖼️  检查和Fix final version image paths...")
        
        if Path(final_index_path).exists():
            fix_image_paths_before_screenshot(final_index_path, self.log)
            
            self.log("📸 生成最终Screenshot...")
            capture_full_page_screenshot(
                html_file_path=final_index_path,
                output_image_path=str(final_screenshot),
                viewport_width=1920,
                viewport_height=1080
            )
        else:
            self.log(f"❌ Final HTML file does not exist: {final_index_path}")
        
        self.log(f"✅ Final version v3 saved to: {self.v1_dir}")

    def generate_final_version_v3_direct_copy(self):
        """Generate final version by directly copying from v0 to v1 directory (no optimization)"""
        self.log("🔧 v3: Direct copy from v0 to v1 (no optimization)...", "V3_FIX")
        self.log(f"Target directory: {self.v1_dir}")

        # Copy main files (v3 supports single-file HTML)
        # Required files
        required_files = ['index.html']
        # Optional files (may be embedded in HTML)
        optional_files = ['style.css', 'script.js']

        for file_name in required_files:
            src_file = self.v0_dir / file_name
            dst_file = self.v1_dir / file_name

            if src_file.exists():
                shutil.copy2(src_file, dst_file)

        for file_name in optional_files:
            src_file = self.v0_dir / file_name
            dst_file = self.v1_dir / file_name

            if src_file.exists():
                shutil.copy2(src_file, dst_file)

        # Copy image directories
        self.ensure_image_directories(self.v1_dir, self.v0_dir)

        # 生成最终Screenshot (直接使用v0版本)
        final_screenshot = self.v1_dir / "final_screenshot_v3.png"

        # Ensure image directories exist后再Screenshot
        self.ensure_image_directories(self.v1_dir)

        # Fix final version image paths
        final_index_path = str(self.v1_dir / "index.html")

        if Path(final_index_path).exists():
            fix_image_paths_before_screenshot(final_index_path, self.log)
            capture_full_page_screenshot(
                html_file_path=final_index_path,
                output_image_path=str(final_screenshot),
                viewport_width=1920,
                viewport_height=1080
            )

        else:
            self.log(f"❌ Final HTML file does not exist: {final_index_path}")

        self.log(f"✅ Direct copy completed, v3 results saved to: {self.v1_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Website Iterative Optimization System v3 ',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """
    )
    
    parser.add_argument(
        '--v0_dir',
        required=True,
        help='v0 directory path (must contain index.html, optionally script.js, style.css and image folders; v3 supports single-file HTML format)'
    )
    
    parser.add_argument(
        '--vision_model',
        default='openrouter_qwen2_5_VL_72B',
        help='Vision analysis model name (for image analysis and suggestion generation)'
    )
    
    parser.add_argument(
        '--code_model',
        default='openrouter_qwen3_coder',
        help='Code optimization model name (for HTML code generation and optimization)'
    )

    parser.add_argument(
        '--max_try',
        type=int,
        default=0,
        help='Maximum number of optimization iterations (0 = no optimization, directly copy v0 to v1)'
    )

    args = parser.parse_args()
    
    try:
        print("="*60)
        print("🚀 Website Iterative Optimization System v3")
        print("🔧 Fixed duplicate merge bug + rule-based classification")
        print("="*60)
        
        # Validate v0 directory
        optimizer = WebsiteIterativeOptimizerV3(args.v0_dir, args.vision_model, args.code_model, args.max_try)
        
        if not optimizer.validate_v0_directory():
            print("❌ v0 directory validation failed")
            sys.exit(1)
        
        # Run iterative optimization
        result_dir = optimizer.run_iterative_optimization()
        
        print(f"\n🎉 Optimization completed!")
        print(f"📁 Results saved in: {result_dir}")
        print(f"🔧 v3 Fix: Resolved duplicate merge bug, implemented rule-based component classification")
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
