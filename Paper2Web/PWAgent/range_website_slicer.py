#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Range Website Slicer - Slice webpage by range
Supports webpage slicing based on specified section index ranges
Used to implement section merging functionality in iterative optimization
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from playwright.sync_api import sync_playwright, Page, Browser, Locator
from PIL import Image

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def slice_webpage_range(html_file_path: str, full_screenshot_path: str, output_dir: str, 
                       start_index: int, end_index: int, target_selector: str = "section") -> List[Dict[str, str]]:
    """
    Slice webpage by specified range
    
    Args:
        html_file_path: HTML file path
        full_screenshot_path: Full page screenshot path
        output_dir: Output directory
        start_index: Starting section index (inclusive)
        end_index: Ending section index (inclusive)
        target_selector: CSS selector
    
    Returns:
        List[Dict[str, str]]: Mapping relationship containing image paths and html snippets
    """
    
    # Validate input files
    if not os.path.exists(html_file_path):
        raise FileNotFoundError(f"HTML file does not exist: {html_file_path}")
    
    if not os.path.exists(full_screenshot_path):
        raise FileNotFoundError(f"Screenshot file does not exist: {full_screenshot_path}")
    
    # Create output directory
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create slices subdirectory
    slices_dir = os.path.join(output_dir, "slices")
    if os.path.exists(slices_dir):
        # Clean existing slices directory
        for item in os.listdir(slices_dir):
            item_path = os.path.join(slices_dir, item)
            if os.path.isfile(item_path):
                os.unlink(item_path)
        print(f"Cleaned existing directory: {slices_dir}")
    else:
        os.makedirs(slices_dir, exist_ok=True)
        print(f"Created slices directory: {slices_dir}")
    
    # Load full page screenshot
    print(f"Loading full page screenshot: {full_screenshot_path}")
    try:
        full_image = Image.open(full_screenshot_path)
        print(f"Screenshot dimensions: {full_image.size[0]}x{full_image.size[1]}px")
    except Exception as e:
        raise Exception(f"Unable to load screenshot: {e}")
    
    component_data = []
    
    try:
        with sync_playwright() as p:
            # Launch Chromium browser
            browser: Browser = p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--disable-gpu'
                ]
            )
            
            # Create browser context
            context = browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            
            # Create page
            page: Page = context.new_page()
            
            # Build file URL
            html_file_uri = f"file://{os.path.abspath(html_file_path)}"
            print(f"Loading HTML page: {html_file_uri}")
            
            # Load HTML page
            page.goto(html_file_uri)
            
            # Wait for page to load
            print("Waiting for page to load...")
            page.wait_for_load_state('networkidle', timeout=30000)
            page.wait_for_timeout(2000)
            
            # Find target elements
            print(f"Finding elements: {target_selector}")
            elements: Locator = page.locator(target_selector)
            element_count = elements.count()
            
            if element_count == 0:
                print(f"Warning: No '{target_selector}' elements found")
                print("Trying 'section' selector")
                elements = page.locator('section')
                element_count = elements.count()
                
                if element_count == 0:
                    print("Trying generic div selector")
                    elements = page.locator('div[class*="section"], div[id*="section"], div')
                    element_count = elements.count()
            
            print(f"Found {element_count} elements")
            
            # Validate index range
            if start_index < 0 or end_index >= element_count or start_index > end_index:
                raise ValueError(f"Invalid index range: [{start_index}, {end_index}], total elements: {element_count}")
            
            print(f"Slice range: [{start_index}, {end_index}]")
            
            # Process elements in specified range
            slice_index = 0
            for i in range(start_index, end_index + 1):
                try:
                    element = elements.nth(i)
                    
                    # Get element position and size information
                    element_info = element.evaluate("""
                        (element) => {
                            const rect = element.getBoundingClientRect();
                            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
                            const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;
                            return {
                                x: rect.left + scrollLeft,
                                y: rect.top + scrollTop,
                                width: rect.width,
                                height: rect.height,
                                tagName: element.tagName.toLowerCase(),
                                id: element.id || '',
                                className: element.className || ''
                            };
                        }
                    """)
                    
                    x, y, width, height = element_info['x'], element_info['y'], element_info['width'], element_info['height']
                    tag_name = element_info['tagName']
                    element_id = element_info['id']
                    element_class = element_info['className']
                    
                    print(f"Processing element {i} (Slice {slice_index}): {tag_name} (ID: {element_id}, Class: {element_class})")
                    print(f"    Position: ({x}, {y}), Size: {width}x{height}")
                    
                    # Check element dimensions
                    if width <= 0 or height <= 0:
                        print(f"    Skipping: Invalid element dimensions")
                        continue
                    
                    # Crop element area
                    crop_box = (int(x), int(y), int(x + width), int(y + height))
                    
                    # Ensure crop area is within image bounds
                    img_width, img_height = full_image.size
                    crop_box = (
                        max(0, min(crop_box[0], img_width)),
                        max(0, min(crop_box[1], img_height)),
                        max(0, min(crop_box[2], img_width)),
                        max(0, min(crop_box[3], img_height))
                    )
                    
                    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                        print(f"    Skipping: Invalid crop area")
                        continue
                    
                    cropped_image = full_image.crop(crop_box)
                    
                    # Save slice image
                    slice_filename = f"section_{slice_index:02d}.png"
                    slice_path = os.path.join(slices_dir, slice_filename)
                    cropped_image.save(slice_path, "PNG")
                    print(f"    Saved slice: {slice_filename}")
                    
                    # Get HTML snippet
                    html_snippet = element.evaluate('(element) => element.outerHTML')
                    print(f"    HTML length: {len(html_snippet)} characters")
                    
                    # Add to result data
                    component_data.append({
                        "image_path": os.path.abspath(slice_path),
                        "html_snippet": html_snippet,
                        "element_info": {
                            "index": i,  # Original index
                            "slice_index": slice_index,  # Slice index
                            "tag_name": tag_name,
                            "id": element_id,
                            "class": element_class,
                            "position": {"x": x, "y": y},
                            "dimensions": {"width": width, "height": height}
                        }
                    })
                    
                    slice_index += 1
                    
                except Exception as e:
                    print(f"Error processing element {i}: {e}")
                    continue
            
            # Close browser
            browser.close()
            
    except Exception as e:
        print(f"Error during slicing process: {e}")
        raise
    
    # Save mapping file
    mapping_path = os.path.join(slices_dir, "mapping.json")
    print(f"Saving mapping file: {mapping_path}")
    
    try:
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(component_data, f, ensure_ascii=False, indent=2)
        print(f"Mapping file saved successfully, contains {len(component_data)} slices")
    except Exception as e:
        print(f"Failed to save mapping file: {e}")
        raise
    
    return component_data


def validate_inputs(html_path: str, screenshot_path: str, output_dir: str, start_index: int, end_index: int) -> bool:
    """
    Validate input parameters
    """
    # Check HTML file
    if not os.path.exists(html_path):
        print(f"Error: HTML file does not exist: {html_path}")
        return False
    
    if not html_path.lower().endswith('.html'):
        print(f"Warning: File extension is not .html: {html_path}")
    
    # Check screenshot file
    if not os.path.exists(screenshot_path):
        print(f"Error: Screenshot file does not exist: {screenshot_path}")
        return False
    
    # Check output directory permissions
    output_parent = os.path.dirname(output_dir) if os.path.dirname(output_dir) else '.'
    if not os.access(output_parent, os.W_OK):
        print(f"Error: Output directory is not writable: {output_parent}")
        return False
    
    # Check index range
    if start_index < 0:
        print(f"Error: Start index cannot be negative: {start_index}")
        return False
    
    if end_index < start_index:
        print(f"Error: End index cannot be less than start index: start={start_index}, end={end_index}")
        return False
    
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Range-based webpage slicing tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  python range_website_slicer.py --html_path ./index.html --screenshot_path ./screenshot.png --output_dir ./range_slices --start_index 0 --end_index 1
  python range_website_slicer.py --html_path ./index.html --screenshot_path ./screenshot.png --output_dir ./range_slices --start_index 2 --end_index 3 --selector "div.section"
        """
    )
    
    parser.add_argument(
        '--html_path',
        required=True,
        help='HTML file path'
    )
    
    parser.add_argument(
        '--screenshot_path',
        required=True,
        help='Full page screenshot path'
    )
    
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Output directory path'
    )
    
    parser.add_argument(
        '--start_index',
        type=int,
        required=True,
        help='Starting section index (inclusive)'
    )
    
    parser.add_argument(
        '--end_index',
        type=int,
        required=True,
        help='Ending section index (inclusive)'
    )
    
    parser.add_argument(
        '--selector',
        default="section",
        help='CSS selector to locate HTML elements (default: "section")'
    )
    
    args = parser.parse_args()
    
    try:
        print("Range-based webpage slicing tool")
        print(f"HTML file: {args.html_path}")
        print(f"Screenshot file: {args.screenshot_path}")
        print(f"Output directory: {args.output_dir}")
        print(f"Index range: [{args.start_index}, {args.end_index}]")
        print(f"CSS selector: {args.selector}")
        print("-" * 50)
        
        # Validate inputs
        if not validate_inputs(args.html_path, args.screenshot_path, args.output_dir, 
                             args.start_index, args.end_index):
            sys.exit(1)
        
        # Execute slicing
        component_data = slice_webpage_range(
            html_file_path=args.html_path,
            full_screenshot_path=args.screenshot_path,
            output_dir=args.output_dir,
            start_index=args.start_index,
            end_index=args.end_index,
            target_selector=args.selector
        )
        
        print("\nSlicing complete!")
        print(f"Result statistics:")
        print(f"   - Slice count: {len(component_data)}")
        print(f"   - Output directory: {os.path.abspath(args.output_dir)}")
        print(f"   - Mapping file: slices/mapping.json")
        
        # Display generated files
        print(f"\nGenerated files:")
        slices_dir = os.path.join(args.output_dir, "slices")
        if os.path.exists(slices_dir):
            print(f"    slices/")
            for item in os.listdir(slices_dir):
                item_path = os.path.join(slices_dir, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path)
                    if item.endswith('.png'):
                        print(f"       {item} ({size} bytes)")
                    elif item.endswith('.json'):
                        print(f"       {item} ({size} bytes)")
        
    except KeyboardInterrupt:
        print("\nUser interrupted operation")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()