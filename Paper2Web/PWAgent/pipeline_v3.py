#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper2Web Pipeline v3 - Complete Pipeline
End-to-end processing from PDF to optimized website

Process:
1. PDF Parsing (parse_raw.py)
2. Website Outline Generation (simple_gen_outline_layout_website.py) 
3. Key Information Extraction (extract_importinfo.py)
4. Website Code Generation (simple_end_to_end_generator_v1.py) -> v0 (single-file HTML)
5. Iterative Optimization (web_link_v3.py) -> v1 (category-based specialized optimization)
"""

import os
import sys
import shutil
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

# Add current directory and parent directory to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(project_root))

# Import functions from all modules
from parse_raw import parse_raw, gen_image_and_table
from simple_gen_outline_layout_website import filter_image_table, gen_outline_layout_website_simple
from extract_importinfo import extract_important_info
from simple_end_to_end_generator_v1 import generate_website_end_to_end, save_website_files, find_files_smart_end_to_end
from web_link_v3 import WebsiteIterativeOptimizerV3
from utils.wei_utils import get_agent_config


class ModelConfig:
    """Model configuration management"""
    
    def __init__(self, 
                 parse_model: str = "openrouter_qwen3_30b_a3b",
                 outline_model: str = "openrouter_qwen3_30b_a3b", 
                 extract_model: str = "openrouter_qwen3_30b_a3b",
                 generator_model: str = "openrouter_qwen3_coder",
                 vision_model: str = "openrouter_qwen2_5_VL_72B",
                 coder_model: str = "openrouter_qwen3_coder"):
        
        self.parse_model = parse_model           # PDF parsing model
        self.outline_model = outline_model       # Outline generation model
        self.extract_model = extract_model       # Information extraction model
        self.generator_model = generator_model   # Code generation model (single-file HTML)
        self.vision_model = vision_model         # Visual analysis model
        self.coder_model = coder_model          # Code optimization model
    
    def get_all_models(self) -> Dict[str, str]:
        """Get all model configurations"""
        return {
            "parse": self.parse_model,
            "outline": self.outline_model,
            "extract": self.extract_model,
            "generator": self.generator_model,
            "vision": self.vision_model,
            "coder": self.coder_model
        }

class CostCalculator:
    """Cost calculator"""
    
    def __init__(self):
        self.costs = {}  # {model_name: {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}}
    
    def add_usage(self, model_name: str, input_tokens: int, output_tokens: int):
        """Add model usage record"""
        if model_name not in self.costs:
            self.costs[model_name] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0
            }
        
        self.costs[model_name]["input_tokens"] += input_tokens
        self.costs[model_name]["output_tokens"] += output_tokens
        
        # Cost calculation removed as requested
        # if model_name in MODEL_PRICING:
        #     pricing = MODEL_PRICING[model_name]
        #     input_cost = (input_tokens / 1_000_000) * pricing["input"]
        #     output_cost = (output_tokens / 1_000_000) * pricing["output"]
        #     self.costs[model_name]["cost"] += input_cost + output_cost
    

    
    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get detailed cost breakdown - disabled as requested"""
        return {
            "breakdown": {},
            "total_cost": 0.0,
            "untracked_models": []
        }

class Pipeline:
    """Paper2Web v3 complete pipeline"""
    
    def __init__(self, pdf_path: str, res_dir: str, model_config: ModelConfig = None, max_try: int = 1):
        self.pdf_path = Path(pdf_path).resolve()
        self.res_dir = Path(res_dir).resolve()
        self.max_try = max(1, max_try)  # Ensure at least 1 optimization round
        
        # Ensure original_cwd points to project root directory (directory containing utils folder)
        current_dir = Path.cwd()
        if current_dir.name == "PWAAgent":
            self.original_cwd = current_dir.parent  # Project root directory
        else:
            self.original_cwd = current_dir
            
        # Generate website name from PDF filename
        self.website_name = self.pdf_path.stem.replace(' ', '_').replace('-', '_')
        
        # Model configuration
        self.model_config = model_config or ModelConfig()
        
        # Cost calculator
        self.cost_calculator = CostCalculator()
        
        # Create result directory
        self.res_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.log_file = self.res_dir / "pipeline_v3.log"
        self.init_logging()
        
    def init_logging(self):
        """Initialize logging system"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("Paper2Web Pipeline v3 Log\n")
            f.write(f"v3 New Features: Single-file HTML generation, based on web_link_v3 optimizer\n")
            f.write(f"Multi-round optimization: Maximum optimization rounds = {self.max_try}\n")
            f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"PDF file: {self.pdf_path}\n")
            f.write(f"Result directory: {self.res_dir}\n")
            f.write(f"Website name: {self.website_name}\n")
            f.write("Model configuration:\n")
            for stage, model in self.model_config.get_all_models().items():
                f.write(f"  - {stage}: {model}\n")
            f.write("="*80 + "\n")
    
    def log(self, message: str, level: str = "INFO"):
        """Record log"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Add different prefixes based on level
        if level == "SUCCESS":
            prefix = "[SUCCESS]"
        elif level == "ERROR":
            prefix = "[ERROR]"
        elif level == "WARNING":
            prefix = "[WARNING]"
        elif level == "PROGRESS":
            prefix = "[PROGRESS]"
        elif level == "COST":
            prefix = "[COST]"
        elif level == "TOKEN":
            prefix = "[TOKEN]"
        else:
            prefix = "[INFO]"
        
        log_msg = f"[{timestamp}] {prefix} {message}"
        print(log_msg)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    
    def create_args_object(self, model_name: str = None, **kwargs) -> argparse.Namespace:
        """Create args object for all modules to use"""
        args = argparse.Namespace()
        
        # Basic parameters
        args.website_name = self.website_name
        args.paper_path = str(self.pdf_path)
        args.website_path = str(self.pdf_path)  # Compatibility
        args.model_name = model_name or self.model_config.parse_model
        args.model_name_t = model_name or self.model_config.parse_model
        args.model_name_v = "v1"
        args.index = 0
        
        # Update custom parameters
        for key, value in kwargs.items():
            setattr(args, key, value)
        
        return args
    
    def stage1_parse_raw(self) -> Tuple[int, int]:
        """Stage 1: PDF parsing and image/table extraction"""
        self.log("Starting Stage 1: PDF parsing and image/table extraction", "PROGRESS")
        
        try:
            # Create args object, use PDF parsing model
            args = self.create_args_object(model_name=self.model_config.parse_model)
            
            # Get model configuration
            actor_config = get_agent_config(self.model_config.parse_model)
            self.log(f"Using Model: {self.model_config.parse_model}")
            
            # Switch to result directory for execution
            os.chdir(self.res_dir)
            
            # Execute PDF parsing (execute in project root directory to access prompt files)
            self.log("Starting PDF document parsing...")
            os.chdir(self.original_cwd)  # Temporarily switch back to project root directory
            input_token, output_token, raw_result = parse_raw(args, actor_config, version=2)
            
            # Move generated contents directory to res directory
            source_contents = self.original_cwd / "contents"
            target_contents = self.res_dir / "contents"
            
            if source_contents.exists():
                if target_contents.exists():
                    # If target directory already exists, move files within it
                    for file in source_contents.iterdir():
                        if file.is_file():
                            target_file = target_contents / file.name
                            if target_file.exists():
                                target_file.unlink()  # Delete existing file
                            shutil.move(str(file), str(target_file))
                    # Delete empty source directory
                    source_contents.rmdir()
                else:
                    # Directly move entire directory
                    shutil.move(str(source_contents), str(target_contents))
                
                self.log(f"Moved contents directory to: {target_contents}")
            
            os.chdir(self.res_dir)  # Switch back to result directory
            
            # Add cost record
            self.cost_calculator.add_usage(self.model_config.parse_model, input_token, output_token)
            
            # Generate images and tables
            self.log("Starting image and table extraction...")
            img_input_token, img_output_token, images, tables = gen_image_and_table(args, raw_result)
            
            # Image/table extraction also uses the same model
            self.cost_calculator.add_usage(self.model_config.parse_model, img_input_token, img_output_token)
            
            total_input = input_token + img_input_token
            total_output = output_token + img_output_token
            
            self.log("Stage 1 complete!", "SUCCESS")
            self.log(f"   Model used: {self.model_config.parse_model}")
            self.log(f"   Token usage: {total_input} -> {total_output}", "TOKEN")
            self.log(f"   Generated images: {len(images)}")
            self.log(f"   Generated tables: {len(tables)}")
            
            return total_input, total_output
            
        except Exception as e:
            self.log(f"Stage 1 failed: {e}", "ERROR")
            raise
        finally:
            # Ensure we return to original directory
            os.chdir(self.original_cwd)
    
    def stage2_simple_gen_outline(self) -> Tuple[int, int]:
        """Stage 2: Website outline generation"""
        self.log("Starting Stage 2: Website outline generation", "PROGRESS")
        
        try:
            # Create args object, use outline generation model
            args = self.create_args_object(model_name=self.model_config.outline_model)
            
            # Get model configuration
            actor_config = get_agent_config(self.model_config.outline_model)
            self.log(f"Using Model: {self.model_config.outline_model}")
            
            # Switch to result directory
            os.chdir(self.res_dir)
            
            # Filter images and tables (execute in res directory, need to add project root to Python path to access prompt files)
            self.log("Starting image and table filtering...")
            
            # Ensure Python path includes project root, so utils module can be accessed
            import sys
            if str(self.original_cwd) not in sys.path:
                sys.path.insert(0, str(self.original_cwd))
            
            # Execute filter_image_table in res directory, so file search can find correct files
            filter_input, filter_output = filter_image_table(args, actor_config)
            
            # Add cost record
            self.cost_calculator.add_usage(self.model_config.outline_model, filter_input, filter_output)
            
            # Generate website outline (continue executing in res directory, because need to find filtered files)
            self.log("Generating website outline...")
            # Continue executing in res directory, don't switch directories, because gen_outline_layout_website_simple needs to find filtered files in res directory
            outline_input, outline_output, website_pages, figure_arrangement = gen_outline_layout_website_simple(args, actor_config)
            
            # Add cost record
            self.cost_calculator.add_usage(self.model_config.outline_model, outline_input, outline_output)
            
            total_input = filter_input + outline_input
            total_output = filter_output + outline_output
            
            self.log("Stage 2 complete!", "SUCCESS")
            self.log(f"   Model used: {self.model_config.outline_model}")
            self.log(f"   Token usage: {total_input} -> {total_output}", "TOKEN")
            self.log(f"   Planned pages: {len(website_pages)}")
            
            return total_input, total_output
            
        except Exception as e:
            self.log(f"Stage 2 failed: {e}", "ERROR")
            raise
        finally:
            # Ensure we return to original directory
            os.chdir(self.original_cwd)
    
    def stage3_extract_info(self) -> Tuple[int, int]:
        """Stage 3: Key information extraction"""
        self.log("Starting Stage 3: Key information extraction", "PROGRESS")
        
        try:
            # Create args object, use information extraction model
            args = self.create_args_object(model_name=self.model_config.extract_model)
            
            # Get model configuration
            actor_config = get_agent_config(self.model_config.extract_model)
            self.log(f"Using Model: {self.model_config.extract_model}")
            
            # Switch to result directory
            os.chdir(self.res_dir)
            
            # Extract important information (execute in res directory, template path issue fixed)
            self.log("Starting key information extraction...")
            # Ensure Python path includes project root directory
            import sys
            if str(self.original_cwd) not in sys.path:
                sys.path.insert(0, str(self.original_cwd))
            
            # Execute in res directory, file search and result saving are in correct location
            input_token, output_token, important_info = extract_important_info(args, actor_config)
            
            # Add cost record
            self.cost_calculator.add_usage(self.model_config.extract_model, input_token, output_token)
            
            if important_info:
                info_count = sum(len(v) if isinstance(v, list) else 1 for v in important_info.values())
                self.log("Stage 3 complete!", "SUCCESS")
                self.log(f"   Model used: {self.model_config.extract_model}")
                self.log(f"   Token usage: {input_token} -> {output_token}", "TOKEN")
                self.log(f"   Extracted info: {info_count} items")
            else:
                self.log("Stage 3: Failed to extract key information, but continuing execution", "WARNING")
                input_token, output_token = 0, 0
            
            return input_token, output_token
            
        except Exception as e:
            self.log(f"Stage 3 failed: {e}", "ERROR")
            raise
        finally:
            # Ensure we return to original directory
            os.chdir(self.original_cwd)
    
    def stage4_generate_v0(self) -> str:
        """Stage 4: Generate v0 single-file HTML website"""
        self.log("Starting Stage 4: Generate v0 single-file HTML website", "PROGRESS")
        
        try:
            # Create args object, use code generation model
            args = self.create_args_object(model_name=self.model_config.generator_model)
            
            self.log(f"Using Model: {self.model_config.generator_model}")
            self.log("v3 Feature: Generate single-file HTML (embedded CSS+JS)")
            
            # Switch to result directory to find files
            os.chdir(self.res_dir)
            
            # Intelligently find required files
            self.log("Finding files required for website generation...")
            research_content_file, visual_assets_file, important_info_file = find_files_smart_end_to_end(self.website_name)
            
            # Load data
            with open(research_content_file, 'r', encoding='utf-8') as f:
                research_content = json.load(f)
            
            with open(visual_assets_file, 'r', encoding='utf-8') as f:
                visual_assets_raw = json.load(f)
            
            # Build visual_assets data structure
            visual_assets = {
                "meta": {
                    "title": research_content.get("meta", {}).get("website_title", ""),
                    "authors": research_content.get("meta", {}).get("authors", ""),
                    "affiliations": research_content.get("meta", {}).get("affiliations", ""),
                    "project_name": self.website_name
                },
                "images": [],
                "tables": []
            }
            
            # Process image and table data
            if "pages" in visual_assets_raw:
                for page in visual_assets_raw["pages"]:
                    # Process images
                    if "images" in page:
                        images_list = page["images"] if isinstance(page["images"], list) else [page["images"]]
                        for img_id in images_list:
                            if "arranged_images" in visual_assets_raw and str(img_id) in visual_assets_raw["arranged_images"]:
                                img_info = visual_assets_raw["arranged_images"][str(img_id)]
                                visual_assets["images"].append({
                                    "id": str(img_id),
                                    "src": img_info.get("image_path", ""),
                                    "alt": img_info.get("caption", ""),
                                    "web_width": img_info.get("width", 800),
                                    "web_height": img_info.get("height", 600)
                                })
                    
                    # Process tables
                    if "tables" in page:
                        tables_list = page["tables"] if isinstance(page["tables"], list) else [page["tables"]]
                        for table_id in tables_list:
                            if "arranged_tables" in visual_assets_raw and str(table_id) in visual_assets_raw["arranged_tables"]:
                                table_info = visual_assets_raw["arranged_tables"][str(table_id)]
                                visual_assets["tables"].append({
                                    "id": str(table_id),
                                    "src": table_info.get("table_path", ""),
                                    "alt": table_info.get("caption", "")
                                })
            
            # Load important information
            important_info = {"important_info": []}
            if important_info_file and Path(important_info_file).exists():
                with open(important_info_file, 'r', encoding='utf-8') as f:
                    important_info = json.load(f)
            
            # Generate website code (execute in res directory, template path issue fixed)
            self.log("Starting single-file HTML website generation...")
            # Ensure Python path includes project root directory
            import sys
            if str(self.original_cwd) not in sys.path:
                sys.path.insert(0, str(self.original_cwd))
            
            # Execute in res directory, so v0 directory will be created in res directory
            input_token, output_token, website_code = generate_website_end_to_end(
                args, research_content, visual_assets, important_info
            )
            
            # Add cost record
            self.cost_calculator.add_usage(self.model_config.generator_model, input_token, output_token)
            
            if website_code is None:
                raise Exception("Single-file HTML website code generation failed")
            
            # Save website files to v0 directory
            self.log("Saving single-file HTML to v0 directory...")
            v0_dir = save_website_files(args, website_code)
            
            # Rename directory to v0
            actual_v0_dir = self.res_dir / "v0"
            if Path(v0_dir).name != "v0":
                if actual_v0_dir.exists():
                    shutil.rmtree(actual_v0_dir)
                shutil.move(v0_dir, actual_v0_dir)
                v0_dir = str(actual_v0_dir)
            
            self.log("Stage 4 complete!", "SUCCESS")
            self.log(f"   Model used: {self.model_config.generator_model}")
            self.log(f"   Token usage: {input_token} -> {output_token}", "TOKEN")
            self.log(f"   v0 directory: {v0_dir}")
            self.log(f"   Generated single-file HTML: includes embedded CSS and JavaScript")
            
            return v0_dir
            
        except Exception as e:
            self.log(f"Stage 4 failed: {e}", "ERROR")
            raise
        finally:
            # Switch back to original directory
            os.chdir(self.original_cwd)
    
    def stage5_iterative_optimization(self, v0_dir: str) -> str:
        """Stage 5: Multi-round iterative optimization (using v3 optimizer - supports custom optimization rounds)"""
        self.log(f"Starting Stage 5: Multi-round iterative optimization (total {self.max_try} rounds)", "PROGRESS")
        
        # Switch to result directory
        os.chdir(self.res_dir)
        
        try:
            current_dir = v0_dir
            final_dir = None
            
            # Multi-round optimization loop
            for round_num in range(1, self.max_try + 1):
                self.log(f"Starting round {round_num}/{self.max_try} optimization", "PROGRESS")
                
                # Use relative path (because already in res directory)
                current_relative_path = Path(current_dir).name if Path(current_dir).parent == self.res_dir else current_dir
                
                self.log(f"Initializing website optimizer v3, input directory: {current_relative_path}")
                if round_num == 1:
                    self.log(f"   v3 New features: Rule-based component classification, correct slice merging logic")
                    self.log(f"   Visual analysis model: {self.model_config.vision_model}")
                    self.log(f"   Code optimization model: {self.model_config.coder_model}")
            
                # WebsiteIterativeOptimizerV3 accepts separate vision_model and code_model parameters
                optimizer = WebsiteIterativeOptimizerV3(
                    v0_dir=current_relative_path,
                    vision_model=self.model_config.vision_model,
                    code_model=self.model_config.coder_model
                )
            
                # Validate input directory
                if not optimizer.validate_v0_directory():
                    raise Exception(f"Round {round_num} optimization input directory validation failed: {current_relative_path}")
            
                # Run current round optimization
                self.log(f"Starting round {round_num} optimization process...")
                result_dir = optimizer.run_iterative_optimization()
            
                # Try to extract token usage from optimizer log
                self.log(f"Recording round {round_num} optimization v3 cost (Note: this is an estimate)", "COST")
                try:
                    # Read optimizer log, try to extract API call information
                    if optimizer.log_file.exists():
                        with open(optimizer.log_file, 'r', encoding='utf-8') as f:
                            optimizer_logs = f.read()
                        
                        # Estimate vision analysis and code optimization token usage
                        # Use simple estimation method here
                        vision_calls = optimizer_logs.count("Starting visual analysis")
                        code_calls = optimizer_logs.count("Starting code optimization")
                        
                        # Estimate token usage per call (based on empirical values)
                        estimated_vision_input_per_call = 1000  # Estimated value
                        estimated_vision_output_per_call = 200
                        estimated_code_input_per_call = 2000
                        estimated_code_output_per_call = 1500
                        
                        vision_input_tokens = vision_calls * estimated_vision_input_per_call
                        vision_output_tokens = vision_calls * estimated_vision_output_per_call
                        code_input_tokens = code_calls * estimated_code_input_per_call
                        code_output_tokens = code_calls * estimated_code_output_per_call
                        
                        # Record cost
                        self.cost_calculator.add_usage(self.model_config.vision_model, vision_input_tokens, vision_output_tokens)
                        self.cost_calculator.add_usage(self.model_config.coder_model, code_input_tokens, code_output_tokens)
                        
                        self.log(f"   Round {round_num} estimated visual analysis calls: {vision_calls} times")
                        self.log(f"   Round {round_num} estimated code optimization calls: {code_calls} times")
                        self.log(f"   Round {round_num} estimated token usage (visual): {vision_input_tokens} -> {vision_output_tokens}", "TOKEN")
                        self.log(f"   Round {round_num} estimated token usage (code): {code_input_tokens} -> {code_output_tokens}", "TOKEN")
                        
                except Exception as token_error:
                    self.log(f"Round {round_num} unable to extract token usage: {token_error}", "WARNING")
            
                # Append optimizer log content to main log
                if optimizer.log_file.exists():
                    with open(optimizer.log_file, 'r', encoding='utf-8') as f:
                        optimizer_logs = f.read()
                    
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write("\n" + "="*80 + "\n")
                        f.write(f"Round {round_num} iterative optimization v3 detailed log:\n")
                        f.write("="*80 + "\n")
                        f.write(optimizer_logs)
                
                self.log(f"Round {round_num} optimization complete!", "SUCCESS")
                self.log(f"   Round {round_num} optimization result: {result_dir}")
                
                # Update input directory for next round
                final_dir = result_dir
                
                # If there's a next round, use current result as next round's input
                if round_num < self.max_try:
                    # Rename v1 directory to v{round_num}_temp as next round's input
                    next_round_input = self.res_dir / f"v{round_num}_temp"
                    if next_round_input.exists():
                        shutil.rmtree(next_round_input)
                    
                    # Copy v1 directory to next round's temporary input directory
                    shutil.copytree(result_dir, next_round_input)
                    current_dir = str(next_round_input)
                    
                    self.log(f"Preparing round {round_num + 1} optimization, input directory: {next_round_input.name}")
                else:
                    # Last round, rename to final version
                    final_version_dir = self.res_dir / f"v{self.max_try}"
                    if str(result_dir) != str(final_version_dir):
                        if final_version_dir.exists():
                            shutil.rmtree(final_version_dir)
                        shutil.move(result_dir, final_version_dir)
                        final_dir = str(final_version_dir)
                    
                    self.log(f"All {self.max_try} rounds of optimization complete! Final version: {final_version_dir.name}")
            
            # Clean up temporary files
            for round_num in range(1, self.max_try):
                temp_dir = self.res_dir / f"v{round_num}_temp"
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    self.log(f"Cleaned up temporary directory: {temp_dir.name}")

            # Additional cleanup: delete intermediate product directory v02v1_v3 (from v3 optimizer's working directory)
            try:
                intermediate_dir = self.res_dir / "v02v1_v3"
                if intermediate_dir.exists():
                    shutil.rmtree(intermediate_dir)
                    self.log(f"Cleaned up intermediate directory: {intermediate_dir.name}")
            except Exception as cleanup_err:
                # Don't block process, just log warning
                self.log(f"Failed to clean up intermediate directory v02v1_v3: {cleanup_err}", "WARNING")
            
            self.log("Stage 5 complete!", "SUCCESS")
            self.log(f"   Multi-round optimization ({self.max_try} rounds) overall result: {final_dir}")
            
            return final_dir
            
        except Exception as e:
            self.log(f"Stage 5 failed: {e}", "ERROR")
            raise
        finally:
            # Switch back to original directory
            os.chdir(self.original_cwd)
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run complete pipeline"""
        start_time = datetime.now()
        self.log("Starting Paper2Web v3 complete pipeline", "PROGRESS")
        
        try:
            # Validate PDF file
            if not self.pdf_path.exists():
                raise FileNotFoundError(f"PDF file does not exist: {self.pdf_path}")
            
            total_tokens = {"input": 0, "output": 0}
            
            # Stage 1: PDF parsing
            input_1, output_1 = self.stage1_parse_raw()
            total_tokens["input"] += input_1
            total_tokens["output"] += output_1
            
            # Stage 2: Website outline generation  
            input_2, output_2 = self.stage2_simple_gen_outline()
            total_tokens["input"] += input_2
            total_tokens["output"] += output_2
            
            # Stage 3: Key information extraction
            input_3, output_3 = self.stage3_extract_info()
            total_tokens["input"] += input_3
            total_tokens["output"] += output_3
            
            # Stage 4: Generate v0 single-file HTML website
            v0_dir = self.stage4_generate_v0()
            
            # Stage 5: Multi-round iterative optimization
            final_optimized_dir = self.stage5_iterative_optimization(v0_dir)
            
            # Complete
            end_time = datetime.now()
            duration = end_time - start_time
            
            # Calculate cost details
            cost_breakdown = self.cost_calculator.get_cost_breakdown()
            total_cost = cost_breakdown["total_cost"]
            
            result = {
                "success": True,
                "pdf_path": str(self.pdf_path),
                "res_dir": str(self.res_dir),
                "website_name": self.website_name,
                "model_config": self.model_config.get_all_models(),
                "max_try": self.max_try,
                "v0_dir": v0_dir,
                "final_optimized_dir": final_optimized_dir,
                "v1_dir": final_optimized_dir,  # Compatibility field
                "total_tokens": total_tokens,
                "cost_breakdown": cost_breakdown,
                "total_cost": total_cost,
                "duration": str(duration),
                "log_file": str(self.log_file)
            }
            
            self.log("Paper2Web v3 pipeline executed successfully!")
            self.log(f"   Total duration: {duration}")
            self.log(f"   Total token usage: {total_tokens['input']} -> {total_tokens['output']}")
            self.log(f"   Optimization rounds: {self.max_try} rounds")
            self.log(f"   v0 directory: {v0_dir} (single-file HTML)")
            self.log(f"   Final optimization result: {final_optimized_dir} ({self.max_try} rounds of optimized single-file HTML)")
            
            # Output cost details
            self.log("Cost details:")
            self.log(f"   Total cost: ${total_cost:.4f}")
            for model_name, details in cost_breakdown["breakdown"].items():
                self.log(f"   - {model_name}:")
                self.log(f"     Input tokens: {details['input_tokens']:,}")
                self.log(f"     Output tokens: {details['output_tokens']:,}")
                self.log(f"     Cost: ${details['cost']:.4f}")
            
            if cost_breakdown["untracked_models"]:
                self.log("The following models are not included in billing:")
                for model in cost_breakdown["untracked_models"]:
                    self.log(f"   - {model}")
            
            self.log(f"   Log file: {self.log_file}")
            
            return result
            
        except Exception as e:
            self.log(f"Pipeline execution failed: {e}")
            import traceback
            self.log(f"Error details:\n{traceback.format_exc()}")
            
            # Return cost information even if failed
            cost_breakdown = self.cost_calculator.get_cost_breakdown()
            
            return {
                "success": False,
                "error": str(e),
                "pdf_path": str(self.pdf_path),
                "res_dir": str(self.res_dir),
                "cost_breakdown": cost_breakdown,
                "total_cost": cost_breakdown["total_cost"],
                "log_file": str(self.log_file)
            }


def create_argument_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Paper2Web Pipeline v3 - Complete pipeline from PDF to optimized single-file HTML website',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Optional model configuration parameters
    parser.add_argument('--parse_model', default="openrouter_qwen3_30b_a3b", 
                       help='PDF parsing model (default: openrouter_qwen3_30b_a3b)')
    parser.add_argument('--outline_model', default="openrouter_qwen3_30b_a3b",
                       help='Outline generation model (default: openrouter_qwen3_30b_a3b)')
    parser.add_argument('--extract_model', default="openrouter_qwen3_30b_a3b",
                       help='Information extraction model (default: openrouter_qwen3_30b_a3b)')
    parser.add_argument('--generator_model', default="openrouter_qwen3_coder",
                       help='Single-file HTML generation model (default: openrouter_qwen3_coder)')
    parser.add_argument('--vision_model', default="openrouter_qwen2_5_VL_72B",
                       help='Visual analysis model (default: openrouter_qwen2_5_VL_72B)')
    parser.add_argument('--coder_model', default="openrouter_qwen3_coder",
                       help='Code optimization model (default: openrouter_qwen3_coder)')
    
    # Optional direct input parameters (for non-interactive execution)
    parser.add_argument('--pdf_path', help='PDF file path')
    parser.add_argument('--res_dir', help='Result directory')
    parser.add_argument('--auto_confirm', action='store_true', help='Auto confirm, no user input required')
    
    # Multi-round optimization parameters
    parser.add_argument('--max_try', type=int, default=1, 
                       help='Maximum optimization rounds (default: 1, i.e. v0→v1; set to 2 for v0→v1→v2)')
    
    return parser


def main():
    """Main function"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    print("="*80)
    print("Paper2Web Pipeline v3 - Complete pipeline from PDF to optimized single-file HTML website")
    print("v3 New Features: Single-file HTML generation + Multi-round iterative optimization")
    print("="*80)
    
    try:
        # Create model configuration
        model_config = ModelConfig(
            parse_model=args.parse_model,
            outline_model=args.outline_model,
            extract_model=args.extract_model,
            generator_model=args.generator_model,
            vision_model=args.vision_model,
            coder_model=args.coder_model
        )
        
        # Display model configuration
        print("\nModel configuration:")
        for stage, model in model_config.get_all_models().items():
            print(f"   - {stage}: {model}")
        
        # Get PDF path
        if args.pdf_path:
            pdf_path = args.pdf_path
        else:
            while True:
                pdf_path = input("\nPlease enter PDF file path: ").strip()
                if not pdf_path:
                    print("PDF path cannot be empty, please try again")
                    continue
                
                pdf_file = Path(pdf_path)
                if not pdf_file.exists():
                    print(f"PDF file does not exist: {pdf_path}")
                    continue
                
                if not pdf_file.suffix.lower() == '.pdf':
                    print(f"File is not PDF format: {pdf_path}")
                    continue
                
                break
        
        # Get result directory
        if args.res_dir:
            res_dir = args.res_dir
        else:
            while True:
                res_dir = input("Please enter result directory: ").strip()
                if not res_dir:
                    print("Result directory cannot be empty, please try again")
                    continue
                
                try:
                    res_path = Path(res_dir)
                    res_path.mkdir(parents=True, exist_ok=True)
                    break
                except Exception as e:
                    print(f"Cannot create directory {res_dir}: {e}")
                    continue
        
        print(f"\nConfiguration confirmation:")
        print(f"   PDF file: {pdf_path}")
        print(f"   Result directory: {res_dir}")
        print(f"   Optimization rounds: {args.max_try} rounds")
        
        if not args.auto_confirm:
            max_try_info = f"({args.max_try} rounds of optimization will be performed)" if args.max_try > 1 else ""
            confirm = input(f"\nStart executing v3 pipeline{max_try_info}? (y/N): ").strip().lower()
            if confirm not in ['y', 'yes']:
                print("User cancelled execution")
                return
        
        # Create and run pipeline
        pipeline = Pipeline(pdf_path, res_dir, model_config, args.max_try)
        result = pipeline.run_full_pipeline()
        
        # Display results
        print("\n" + "="*80)
        if result["success"]:
            print("v3 pipeline executed successfully!")
            print(f"   Website name: {result['website_name']}")
            print(f"   Optimization rounds: {result['max_try']} rounds")
            print(f"   v0 version: {result['v0_dir']} (initial single-file HTML)")
            print(f"   Final version: {result['final_optimized_dir']} ({result['max_try']} rounds of optimized single-file HTML)")
            print(f"   Execution time: {result['duration']}")
            print(f"   Token usage: {result['total_tokens']['input']:,} -> {result['total_tokens']['output']:,}")
            
            # Display cost information
            cost_breakdown = result["cost_breakdown"]
            total_cost = result["total_cost"]
            print(f"\nTotal cost: ${total_cost:.4f}")
            
            if cost_breakdown["breakdown"]:
                print("   Cost breakdown:")
                for model_name, details in cost_breakdown["breakdown"].items():
                    print(f"     - {model_name}: ${details['cost']:.4f}")
                    print(f"       Input: {details['input_tokens']:,} tokens, Output: {details['output_tokens']:,} tokens")
            
            if cost_breakdown["untracked_models"]:
                print("   The following models are not included in billing:")
                for model in cost_breakdown["untracked_models"]:
                    print(f"     - {model}")
            
            print(f"\n   Detailed log: {result['log_file']}")
            final_version_name = Path(result['final_optimized_dir']).name
            print(f"\nYou can open index.html in the {final_version_name} directory to view the final single-file HTML website after {result['max_try']} rounds of optimization!")
        else:
            print("v3 pipeline execution failed!")
            print(f"   Error message: {result['error']}")
            
            # Display cost information even if failed
            if "total_cost" in result and result["total_cost"] > 0:
                print(f"   Cost incurred: ${result['total_cost']:.4f}")
            
            print(f"   Detailed log: {result['log_file']}")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\nUser interrupted execution")
    except Exception as e:
        print(f"Program execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
