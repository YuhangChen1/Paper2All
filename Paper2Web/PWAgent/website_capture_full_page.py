# Installation: pip install playwright && playwright install

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Website Full Page Screenshot Capture Tool
Use Playwright to capture full page screenshots of HTML files
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
from playwright.sync_api import sync_playwright, Page, Browser


def capture_full_page_screenshot(html_file_path: str, output_image_path: str, viewport_width: int = 1920, viewport_height: int = 1080) -> bool:
    # Check if HTML file exists
    if not os.path.exists(html_file_path):
        raise FileNotFoundError(f"HTML file not found: {html_file_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        with sync_playwright() as p:
            # Launch Chromium browser
            browser: Browser = p.chromium.launch(
                headless=True,  # Run in headless mode
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
                viewport={'width': viewport_width, 'height': viewport_height},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            
            # Create new page
            page: Page = context.new_page()
            
            # Build file URL
            html_file_uri = f"file://{os.path.abspath(html_file_path)}"
            print(f"Loading HTML: {html_file_uri}")
            
            # Navigate to HTML file
            page.goto(html_file_uri)
            
            # Wait for page to load completely
            print("Waiting for page to load...")
            page.wait_for_load_state('networkidle', timeout=30000)  # 30 seconds timeout
            
            # Wait for JavaScript to execute
            page.wait_for_timeout(2000)  # 2 seconds
            
            print("Page loaded successfully")
            
            # Calculate page dimensions
            page_height = page.evaluate("() => document.documentElement.scrollHeight")
            viewport_height = page.viewport_size['height']
            
            print(f"Page height: {page_height}px, Viewport height: {viewport_height}px")
            
            # Scroll through the entire page to ensure all content is loaded
            scroll_steps = max(1, page_height // viewport_height)
            for step in range(scroll_steps + 1):
                scroll_position = min(step * viewport_height, page_height)
                page.evaluate(f"window.scrollTo(0, {scroll_position})")
                page.wait_for_timeout(500)  # 500ms
                print(f"Scrolling: {step + 1}/{scroll_steps + 1} (position: {scroll_position}px)")
            
            # Return to top
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(1000)
            
            # Wait 10 seconds for final rendering
            print("Waiting 10 seconds for final rendering...")
            page.wait_for_timeout(10000)
            
            print("Taking screenshot...")
            
            # Capture full page screenshot
            page.screenshot(
                path=output_image_path,
                full_page=True,  # Capture entire page
                type='png'  # PNG format
            )
            
            print(f"Screenshot saved: {output_image_path}")
            
            # Get page information
            page_title = page.title()
            page_width, page_height = page.evaluate("() => [document.documentElement.scrollWidth, document.documentElement.scrollHeight]")
            
            print(f"Page information:")
            print(f"   - Title: {page_title}")
            print(f"   - Width: {page_width}px")
            print(f"   - Height: {page_height}px")
            print(f"   - Viewport: {viewport_width}x{viewport_height}")
            
            # Close browser
            browser.close()
            
            return True
            
    except Exception as e:
        print(f"Screenshot capture failed: {e}")
        return False


def validate_file_paths(html_path: str, output_path: str) -> bool:
    """
    Validate input file paths
    
    Args:
        html_path: HTML file path
        output_path: Output image file path
    
    Returns:
        bool: Whether paths are valid
    """
    # Check HTML file
    if not os.path.exists(html_path):
        print(f"HTML file not found: {html_path}")
        return False
    
    if not html_path.lower().endswith('.html'):
        print(f"Warning: File should end with .html: {html_path}")
    
    # Check output directory permissions
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.access(output_dir, os.W_OK):
        print(f"Output directory not writable: {output_dir}")
        return False
    
    return True


def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(
        description='Website Full Page Screenshot Capture Tool - Use Playwright to capture full page screenshots of HTML files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python website_capture_full_page.py --html_path ./index.html --output_path ./screenshot.png
  python website_capture_full_page.py --html_path ./website/index.html --output_path ./output/screenshot.png --viewport_width 1366 --viewport_height 768
        """
    )
    
    parser.add_argument(
        '--html_path',
        required=True,
        help='HTML file path (required)'
    )
    
    parser.add_argument(
        '--output_path',
        required=True,
        help='Output image file path (required)'
    )
    
    parser.add_argument(
        '--viewport_width',
        type=int,
        default=1920,
        help='Viewport width (default: 1920)'
    )
    
    parser.add_argument(
        '--viewport_height',
        type=int,
        default=1080,
        help='Viewport height (default: 1080)'
    )
    
    args = parser.parse_args()
    
    try:
        print("Starting screenshot capture...")
        print(f"HTML file: {args.html_path}")
        print(f"Output path: {args.output_path}")
        print(f"Viewport: {args.viewport_width}x{args.viewport_height}")
        print("-" * 50)
        
        # Validate file paths
        if not validate_file_paths(args.html_path, args.output_path):
            sys.exit(1)
        
        # Capture screenshot
        success = capture_full_page_screenshot(
            html_file_path=args.html_path,
            output_image_path=args.output_path,
            viewport_width=args.viewport_width,
            viewport_height=args.viewport_height
        )
        
        if success:
            print("Screenshot capture completed successfully!")
            print(f"Screenshot saved to: {os.path.abspath(args.output_path)}")
        else:
            print("Screenshot capture failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()