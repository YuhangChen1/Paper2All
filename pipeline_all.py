#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Paper Processing Pipeline
Integrate Paper2Web, Paper2Poster, AutoPR three projects
Only supports OPENROUTER_API_KEY, does not support other APIs
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
import json
import time
from typing import Dict, List, Optional


class UnifiedPipeline:
    """Unified Pipeline Manager"""

    def __init__(self, input_dir: str, output_dir: str, openrouter_api_key: str,
                 pdf_path: str = None, poster_width_inches: int = 48, poster_height_inches: int = 36):
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.specified_pdf_path = pdf_path
        self.openrouter_api_key = openrouter_api_key
        self.poster_width_inches = poster_width_inches
        self.poster_height_inches = poster_height_inches


        os.environ['OPENROUTER_API_KEY'] = self.openrouter_api_key


        self.p2e_root = Path(__file__).parent.resolve()


        self.pdf_path = self._find_pdf_file()


        self.output_dir.mkdir(parents=True, exist_ok=True)


        self.project_name = self.pdf_path.stem.replace(' ', '_').replace('-', '_')

    def _find_pdf_file(self) -> Path:
        if self.specified_pdf_path:
            pdf_path = Path(self.specified_pdf_path).resolve()
            if not pdf_path.exists():
                raise FileNotFoundError()
            if pdf_path.suffix.lower() != '.pdf':
                raise ValueError()

            return pdf_path

        # Auto search logic
        print()

        # Check if input_dir exists
        if not self.input_dir.exists():
            raise FileNotFoundError()

        # Find all PDF files
        pdf_files = []
        if self.input_dir.is_file():
            # If input_dir is a PDF file itself
            if self.input_dir.suffix.lower() == '.pdf':
                pdf_files.append(self.input_dir)
        else:
            # Find PDF files in subdirectories of input_dir
            for item in self.input_dir.iterdir():
                if item.is_dir():
                    # Find paper.pdf in subdirectories
                    paper_pdf = item / "paper.pdf"
                    if paper_pdf.exists() and paper_pdf.is_file():
                        pdf_files.append(paper_pdf)
                elif item.is_file() and item.suffix.lower() == '.pdf':
                    # Find PDF files directly in input_dir
                    pdf_files.append(item)

        if not pdf_files:
            raise FileNotFoundError()

        # Use the first found PDF file
        selected_pdf = pdf_files[0]
        return selected_pdf

    def run_paper2web(self) -> Dict:

        try:
            # Switch to Paper2Web directory
            paper2web_dir = self.p2e_root / "Paper2Web"
            os.chdir(paper2web_dir)

            # Set result directory (in website subdirectory of specified output directory)
            website_output_dir = self.output_dir / "website"

            # Build command
            cmd = [
                "python", "PWAgent/pipeline_v3.py",
                "--pdf_path", str(self.pdf_path),
                "--res_dir", str(website_output_dir),
                "--max_try", "0", 
                "--auto_confirm"
            ]



            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

            if result.returncode == 0:
                return {"success": True, "output_dir": str(website_output_dir), "stdout": result.stdout}
            else:
                return {"success": False, "error": result.stderr, "stdout": result.stdout}

        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            # Return to original directory
            os.chdir(self.p2e_root)

    def run_paper2poster(self) -> Dict:


        try:
            # Switch to Paper2Poster directory
            paper2poster_dir = self.p2e_root / "paper2all"/"Paper2Poster"
            os.chdir(paper2poster_dir)

            # Set output directory (in poster subdirectory of specified output directory)
            poster_output_dir = self.output_dir / "poster"

            # Build command - use openrouter_4o model, use user-specified dimensions
            cmd = [
                "python", "-m", "PosterAgent.new_pipeline",
                "--poster_path", str(self.pdf_path),
                "--model_name_t", "openrouter_4o",
                "--model_name_v", "openrouter_4o",
                "--poster_width_inches", str(self.poster_width_inches),
                "--poster_height_inches", str(self.poster_height_inches)
            ]



            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

            if result.returncode == 0:

                # Find generated poster directory
                generated_posters_dir = paper2poster_dir / "<openrouter_4o_openrouter_4o>_generated_posters"
                if generated_posters_dir.exists():
                    # Copy to specified output directory
                    for item in generated_posters_dir.iterdir():
                        if item.is_dir() and self.project_name in item.name:
                            dest_dir = poster_output_dir / item.name
                            if dest_dir.exists():
                                shutil.rmtree(dest_dir)
                            shutil.copytree(item, dest_dir)
                            break

                return {"success": True, "output_dir": str(poster_output_dir), "stdout": result.stdout}
            else:
                return {"success": False, "error": result.stderr, "stdout": result.stdout}

        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            # Return to original directory
            os.chdir(self.p2e_root)

    def run_autopr(self, model_path: str = None) -> Dict:

        try:
            # Set default model path
            if model_path is None:
                model_path = str(self.p2e_root / "DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt")

            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                return {"success": False, "error": f"model do not exist: {model_path}"}

            # Set output directory (in PR subdirectory of specified output directory)
            pr_output_dir = self.output_dir / "PR"
            pr_output_dir.mkdir(exist_ok=True)

            # Switch to AutoPR directory
            autopr_dir = self.p2e_root / "AutoPR"
            os.chdir(autopr_dir)

            # Build command - use user-provided input_dir directly
            cmd = [
                "python", "pragent/run.py",
                "--model-path", model_path,
                "--input-dir", str(self.input_dir),
                "--output-dir", str(pr_output_dir),
                "--text-api-key", self.openrouter_api_key,
                "--vision-api-key", self.openrouter_api_key,
                "--text-api-base", "https://openrouter.ai/api/v1",
                "--vision-api-base", "https://openrouter.ai/api/v1"
            ]



            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

            if result.returncode == 0:

                return {"success": True, "output_dir": str(pr_output_dir), "stdout": result.stdout}
            else:

                return {"success": False, "error": result.stderr, "stdout": result.stdout}

        except Exception as e:

            return {"success": False, "error": str(e)}
        finally:
            # Return to original directory
            os.chdir(self.p2e_root)

    def run_pipeline(self, model_choice: List[int]) -> Dict:

        results = {
            "paper2web": None,
            "paper2poster": None,
            "autopr": None,
            "start_time": time.time()
        }

        for choice in model_choice:
            if choice == 1:
                results["paper2web"] = self.run_paper2web()
            elif choice == 2:
                results["paper2poster"] = self.run_paper2poster()
            elif choice == 3:
                results["autopr"] = self.run_autopr()

        results["end_time"] = time.time()
        results["total_time"] = results["end_time"] - results["start_time"]

        return results


def main():

    parser = argparse.ArgumentParser(
        description="Paper2Web、Paper2Poster、AutoPR",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Enter the directory path, including the subdirectories of the papers (format: 12345/ or paper_name/, each subdirectory contains paper.pdf)"
    )

    parser.add_argument(
        "--pdf-path",
        type=str,
        default=None,
        help="Optional: Specify the exact path of the PDF file. If not provided, the first PDF file in the input-dir will be automatically selected."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output dir"
    )

    parser.add_argument(
        "--model-choice",
        type=int,
        nargs='+',
        choices=[1, 2, 3],
        default=[1, 2, 3],
        help="Select the module to run (1=Paper2Web, 2=Paper2Poster, 3=AutoPR). By default, all modules will be run."
    )

    parser.add_argument(
        "--autopr-model-path",
        type=str,
        default=None,
        help="AutoPR model path, default relative to p2e directory under DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt"
    )

    parser.add_argument(
        "--poster-width-inches",
        type=int,
        default=48,
        help="Paper2Poster poster width (inches), default 48"
    )

    parser.add_argument(
        "--poster-height-inches",
        type=int,
        default=36,
        help="Paper2Poster poster height (inches), default 36"
    )

    args = parser.parse_args()

    # Check environment variables
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set environment variable: export OPENROUTER_API_KEY=your_api_key_here")
        return

    print("="*80)
    print("🚀 Starting Unified Paper Processing Pipeline")
    print("="*80)
    print(f"📄 Input PDF: {args.pdf_path}")
    print(f"📂 Output Directory: {args.output_dir}")
    print(f"🔧 Selected Modules: {args.model_choice}")
    print(f"🔑 API Key: {'*' * len(openrouter_api_key)}")

    try:
        # Create pipeline instance
        pipeline = UnifiedPipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            openrouter_api_key=openrouter_api_key,
            pdf_path=args.pdf_path,
            poster_width_inches=args.poster_width_inches,
            poster_height_inches=args.poster_height_inches
        )

        # Run pipeline
        results = pipeline.run_pipeline(args.model_choice)

        # Display results
        print("\n" + "="*80)
        print("📊 Pipeline Execution Results")
        print("="*80)

        for module_name in ["paper2web", "paper2poster", "autopr"]:
            choice_num = {"paper2web": 1, "paper2poster": 2, "autopr": 3}[module_name]
            if choice_num in args.model_choice:
                result = results[module_name]
                if result and result.get("success"):
                    print(f"✅ {module_name.upper()}: Success")
                    if "output_dir" in result:
                        print(f"   📂 Output Directory: {result['output_dir']}")
                else:
                    print(f"❌ {module_name.upper()}: Failed")
                    if "error" in result:
                        print(f"   💥 Error: {result['error']}")

        print(f"\n⏱️ Total Time: {results['total_time']:.2f} seconds")
        print("="*80)

    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
