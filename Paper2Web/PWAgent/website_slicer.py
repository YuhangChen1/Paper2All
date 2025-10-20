# Installation: pip install playwright pillow && playwright install

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Website Slicing Tool

Slice webpages into components for analysis and optimization
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from playwright.sync_api import sync_playwright, Page, Browser, Locator
from PIL import Image


def slice_webpage_into_components(html_file_path: str, full_screenshot_path: str, output_dir: str, target_selector: str = "header, nav, section") -> List[Dict[str, str]]:
    # Validate input files
    if not os.path.exists(html_file_path):
        raise FileNotFoundError(f"HTML file does not exist: {html_file_path}")
    
    if not os.path.exists(full_screenshot_path):
        raise FileNotFoundError(f"Full page screenshot does not exist: {full_screenshot_path}")
    
    # A. Prepare output directory
    print(f"📁 Preparing output directory: {output_dir}")
    
    # Prepare slice directory
    slices_dir = os.path.join(output_dir, "slices")
    if os.path.exists(slices_dir):
        # Clean existing slice files
        for item in os.listdir(slices_dir):
            item_path = os.path.join(slices_dir, item)
            if os.path.isfile(item_path):
                os.unlink(item_path)
        print(f"🧹 Cleaned existing slice directory: {slices_dir}")
    else:
        # Create new slice directory
        os.makedirs(slices_dir, exist_ok=True)
        print(f"Created slice directory: {slices_dir}")
    
    print(f"Output directory ready: {output_dir}")
    print(f"Slice directory ready: {slices_dir}")
    
    # Load full page screenshot
    print(f"Loading full page screenshot: {full_screenshot_path}")
    try:
        full_image = Image.open(full_screenshot_path)
        print(f"Screenshot dimensions: {full_image.size[0]}x{full_image.size[1]}px")
    except Exception as e:
        raise Exception(f"Failed to load screenshot: {e}")
    
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
            
            # Navigate to HTML page
            page.goto(html_file_uri)
            
            # Wait for page to fully load
            print("Waiting for page to load...")
            page.wait_for_load_state('networkidle', timeout=30000)
            page.wait_for_timeout(2000)
            
            # B. Find target elements
            print(f"Finding target elements: {target_selector}")
            elements: Locator = page.locator(target_selector)
            element_count = elements.count()
            
            if element_count == 0:
                print(f"Warning: No elements found matching selector '{target_selector}'")
                print("Trying default selector 'header, nav, section'")
                elements = page.locator('header, nav, section')
                element_count = elements.count()
                
                if element_count == 0:
                    print("Trying broader selector: header, nav, section, div")
                    elements = page.locator('header, nav, section, div[class*="section"], div[id*="section"], div')
                    element_count = elements.count()
            
            print(f"Found {element_count} sliceable elements")
            
            # C. Process each element
            for i in range(element_count):
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
                    
                    print(f"  Slice {i + 1}/{element_count}: <{tag_name}> (ID: {element_id}, Class: {element_class[:30]}...)")
                    print(f"     Position: ({x}, {y}), Size: {width}x{height}px")
                    
                    # Check if element dimensions are valid
                    if width <= 0 or height <= 0:
                        print(f"     Skipping element with invalid dimensions")
                        continue
                    
                    # Crop image
                    crop_box = (int(x), int(y), int(x + width), int(y + height))
                    cropped_image = full_image.crop(crop_box)
                    
                    # Generate more appropriate filename based on element type
                    if tag_name == 'header':
                        slice_filename = f"header_{i:02d}.png"
                    elif tag_name == 'nav':
                        slice_filename = f"nav_{i:02d}.png"
                    elif tag_name == 'section':
                        slice_filename = f"section_{i:02d}.png"
                    else:
                        slice_filename = f"element_{i:02d}.png"
                    
                    slice_path = os.path.join(slices_dir, slice_filename)
                    cropped_image.save(slice_path, "PNG")
                    print(f"     Saved slice: {slice_filename}")
                    
                    # Extract HTML code
                    html_snippet = element.evaluate('(element) => element.outerHTML')
                    print(f"     HTML code: {len(html_snippet)} characters")
                    
                    # Save component data
                    component_data.append({
                        "image_path": os.path.abspath(slice_path),
                        "html_snippet": html_snippet,
                        "element_info": {
                            "index": i,
                            "tag_name": tag_name,
                            "id": element_id,
                            "class": element_class,
                            "position": {"x": x, "y": y},
                            "dimensions": {"width": width, "height": height}
                        }
                    })
                    
                except Exception as e:
                    print(f"     Error processing element {i}: {e}")
                    continue
            
            # Close browser
            browser.close()
            
    except Exception as e:
        print(f"Slicing process failed: {e}")
        raise
    
    # D. Sort slice data by page order (based on Y coordinate)
    print("Sorting slices by page order...")
    component_data.sort(key=lambda x: x['element_info']['position']['y'])
    
    # Renumber slice files and indices
    print("Renumbering slice files...")
    for new_index, component in enumerate(component_data):
        old_path = component['image_path']
        old_filename = os.path.basename(old_path)
        
        # Generate new filename based on element type
        tag_name = component['element_info']['tag_name']
        if tag_name == 'header':
            new_filename = f"header_{new_index:02d}.png"
        elif tag_name == 'nav':
            new_filename = f"nav_{new_index:02d}.png"
        elif tag_name == 'section':
            new_filename = f"section_{new_index:02d}.png"
        else:
            new_filename = f"element_{new_index:02d}.png"
        
        new_path = os.path.join(slices_dir, new_filename)
        
        # Rename file
        if old_filename != new_filename and os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"  Renamed: {old_filename} -> {new_filename}")
        
        # Update component data
        component['image_path'] = new_path
        component['element_info']['index'] = new_index
    
    # Save mapping.json
    mapping_path = os.path.join(slices_dir, "mapping.json")
    print(f"Saving mapping file: {mapping_path}")
    
    try:
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(component_data, f, ensure_ascii=False, indent=2)
        print(f"Successfully generated {len(component_data)} slices (including header, nav, sections)")
        
        # Print slice summary
        element_types = {}
        for component in component_data:
            tag_name = component['element_info']['tag_name']
            element_types[tag_name] = element_types.get(tag_name, 0) + 1
        
        print("Slice summary:")
        for tag_name, count in element_types.items():
            print(f"  - {tag_name}: {count} items")
            
    except Exception as e:
        print(f"Failed to save mapping file: {e}")
        raise
    
    return component_data


def slice_webpage_complete(html_file_path: str, full_screenshot_path: str, output_dir: str) -> List[Dict[str, str]]:
    """
    Completely slice webpage, including all major elements (header, nav, section, etc.)
    This is the recommended slicing function to ensure no important page elements are missed
    
    Args:
        html_file_path: HTML file path
        full_screenshot_path: Full page screenshot path  
        output_dir: Output directory
        
    Returns:
        List[Dict[str, str]]: Slice data arranged by page order
    """
    print("Starting complete webpage slicing (including header, nav, section)...")
    
    # Use selector containing all major elements
    complete_selector = "header, nav, main, section, article, aside, footer"
    
    return slice_webpage_into_components(
        html_file_path=html_file_path,
        full_screenshot_path=full_screenshot_path, 
        output_dir=output_dir,
        target_selector=complete_selector
    )


def validate_inputs(html_path: str, screenshot_path: str, output_dir: str) -> bool:
    """
    Validate input parameters
    
    Args:
        html_path: HTML file path
        screenshot_path: Screenshot file path
        output_dir: Output directory
    
    Returns:
        bool: Whether validation passed
    """
    # Check HTML file
    if not os.path.exists(html_path):
        print(f"HTML file not found: {html_path}")
        return False
    
    if not html_path.lower().endswith('.html'):
        print(f"Warning: File extension is not .html: {html_path}")
    
    # Check screenshot file
    if not os.path.exists(screenshot_path):
        print(f"Screenshot file not found: {screenshot_path}")
        return False
    
    # Check output directory permissions
    output_parent = os.path.dirname(output_dir) if os.path.dirname(output_dir) else '.'
    if not os.access(output_parent, os.W_OK):
        print(f"Output directory not writable: {output_parent}")
        return False
    
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Website slicing tool for component-based analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  python website_slicer.py --html_path ./index.html --screenshot_path ./screenshot.png --output_dir ./components
  python website_slicer.py --html_path ./index.html --screenshot_path ./screenshot.png --output_dir ./components --selector "div.section"
        """
    )
    
    parser.add_argument(
        '--html_path',
        required=True,
        help='HTML file path (required)'
    )
    
    parser.add_argument(
        '--screenshot_path',
        required=True,
        help='Full page screenshot path (required)'
    )
    
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Output directory path (required)'
    )
    
    parser.add_argument(
        '--selector',
        default="header, nav, section",
        help='CSS selector to specify HTML elements to slice (default: "header, nav, section")'
    )
    
    parser.add_argument(
        '--complete',
        action='store_true',
        help='Use complete slicing mode, including all major page elements (header, nav, main, section, etc.)'
    )
    
    args = parser.parse_args()
    
    try:
        print("Starting webpage slicing")
        print(f"HTML file: {args.html_path}")
        print(f"Screenshot file: {args.screenshot_path}")
        print(f"Output directory: {args.output_dir}")
        
        if args.complete:
            print("Slice mode: Complete mode (including all major page elements)")
        else:
            print(f"Slice selector: {args.selector}")
        
        print("-" * 50)
        
        # Validate inputs
        if not validate_inputs(args.html_path, args.screenshot_path, args.output_dir):
            sys.exit(1)
        
        # Execute slicing
        if args.complete:
            # Use complete slice mode
            component_data = slice_webpage_complete(
                html_file_path=args.html_path,
                full_screenshot_path=args.screenshot_path,
                output_dir=args.output_dir
            )
        else:
            # Use custom selector
            component_data = slice_webpage_into_components(
                html_file_path=args.html_path,
                full_screenshot_path=args.screenshot_path,
                output_dir=args.output_dir,
                target_selector=args.selector
            )
        
        print("\nSlicing complete!")
        print(f"Result summary:")
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
        
        # Display other files
        print(f"\nOther files:")
        for item in os.listdir(args.output_dir):
            if item != "slices":
                item_path = os.path.join(args.output_dir, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path)
                    if item.endswith('.html'):
                        print(f"    {item} ({size} bytes)")
                    elif item.endswith('.css'):
                        print(f"    {item} ({size} bytes)")
                    elif item.endswith('.js'):
                        print(f"    {item} ({size} bytes)")
                    elif item.endswith('.png'):
                        print(f"    {item} ({size} bytes)")
                    else:
                        print(f"    {item} ({size} bytes)")
        
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