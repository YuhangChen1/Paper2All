#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup

class ImagePathFixer:
    """Image path fixer"""
    
    def __init__(self, logger=None):
        self.logger = logger
        # Supported image formats
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg'}
    
    def log(self, message: str, level: str = "INFO"):
        """Record log"""
        if self.logger:
            self.logger(message, level)
        else:
            print(f"Image: {message}")
    
    def find_images_in_directory(self, directory: Path) -> Dict[str, Path]:
        """
        Find all image files in specified directory and subdirectories
        
        Args:
            directory: Directory to search
            
        Returns:
            Dict[filename, full_path]: Image file mapping
        """
        image_files = {}
        
        try:
            # Recursively search for all image files
            for file_path in directory.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                    filename = file_path.name
                    # If duplicate names exist, keep the first one found
                    if filename not in image_files:
                        image_files[filename] = file_path
                    else:
                        self.log(f"Warning: Found duplicate image file: {filename}")
                        # Can choose more intelligent duplicate handling strategy
        except Exception as e:
            self.log(f"Error searching for image files: {e}", "ERROR")
        
        return image_files
    
    def extract_filename_from_path(self, img_path: str) -> str:
        """Extract filename from image path"""
        # Handle various path formats
        path = img_path.strip()
        
        # Remove query parameters and anchors
        if '?' in path:
            path = path.split('?')[0]
        if '#' in path:
            path = path.split('#')[0]
        
        # Extract filename
        filename = Path(path).name
        return filename
    
    def calculate_relative_path(self, html_file: Path, image_file: Path) -> str:
        """Calculate relative path from HTML file to image file"""
        try:
            # Get HTML file directory
            html_dir = html_file.parent
            
            # Calculate relative path
            relative_path = os.path.relpath(image_file, html_dir)
            
            # Ensure forward slashes are used (suitable for HTML)
            relative_path = relative_path.replace('\\', '/')
            
            return relative_path
        except Exception as e:
            self.log(f"Error calculating relative path: {e}", "ERROR")
            return str(image_file)  # Return absolute path as fallback
    
    def fix_html_image_paths(self, html_file_path: str, backup: bool = True) -> Tuple[bool, int]:
        """
        Fix image paths in HTML file
        
        Args:
            html_file_path: HTML file path
            backup: Whether to backup original file
            
        Returns:
            Tuple[success, number_of_fixed_images]
        """
        html_path = Path(html_file_path)
        
        if not html_path.exists():
            self.log(f"HTML file does not exist: {html_file_path}", "ERROR")
            return False, 0
        
        self.log(f"Starting to fix image paths: {html_path.name}")
        
        try:
            # Read HTML file
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Backup original file
            if backup:
                backup_path = html_path.with_suffix('.html.backup')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                self.log(f"Backed up original file: {backup_path.name}")
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all images in HTML directory and subdirectories
            html_dir = html_path.parent
            available_images = self.find_images_in_directory(html_dir)
            
            if not available_images:
                self.log("No image files found in directory")
                return True, 0
            
            self.log(f"Found {len(available_images)} image files")
            
            # Find all image tags
            img_tags = soup.find_all('img')
            fixed_count = 0
            
            for img_tag in img_tags:
                src = img_tag.get('src', '')
                if not src:
                    continue
                
                # Extract filename
                filename = self.extract_filename_from_path(src)
                
                if filename in available_images:
                    # Found matching image file
                    image_file = available_images[filename]
                    
                    # Calculate correct relative path
                    correct_path = self.calculate_relative_path(html_path, image_file)
                    
                    # Update if path is different
                    if src != correct_path:
                        old_src = src
                        img_tag['src'] = correct_path
                        fixed_count += 1
                        self.log(f"Fixed path: {filename}")
                        self.log(f"   Old path: {old_src}")
                        self.log(f"   New path: {correct_path}")
                    else:
                        self.log(f"Path is correct: {filename}")
                else:
                    self.log(f"Image file not found: {filename}")
            
            # Save fixed HTML
            if fixed_count > 0:
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(str(soup))
                self.log(f"Saved fixed HTML file")
            
            self.log(f"Image path fixing completed, fixed {fixed_count} images")
            return True, fixed_count
            
        except Exception as e:
            self.log(f"Error fixing image paths: {e}", "ERROR")
            return False, 0
    
    def validate_image_paths(self, html_file_path: str) -> Tuple[bool, List[str]]:
        """
        Validate if image paths in HTML file are correct
        
        Args:
            html_file_path: HTML file path
            
        Returns:
            Tuple[all_paths_valid, invalid_path_list]
        """
        html_path = Path(html_file_path)
        
        if not html_path.exists():
            return False, [f"HTML file does not exist: {html_file_path}"]
        
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            img_tags = soup.find_all('img')
            
            invalid_paths = []
            html_dir = html_path.parent
            
            for img_tag in img_tags:
                src = img_tag.get('src', '')
                if not src:
                    continue
                
                # Build complete path for image file
                if src.startswith('http://') or src.startswith('https://'):
                    # Network image, skip validation
                    continue
                
                img_file = html_dir / src
                if not img_file.exists():
                    invalid_paths.append(src)
            
            return len(invalid_paths) == 0, invalid_paths
            
        except Exception as e:
            return False, [f"Error during validation: {e}"]


def fix_image_paths_before_screenshot(html_file_path: str, logger=None) -> bool:
    """
    Convenience function to fix image paths before screenshot
    
    Args:
        html_file_path: HTML file path
        logger: Logger function
        
    Returns:
        bool: Whether successfully fixed
    """
    fixer = ImagePathFixer(logger)
    success, count = fixer.fix_html_image_paths(html_file_path, backup=False)
    
    if success and count > 0:
        if logger:
            logger(f"Fixed {count} image paths", "SUCCESS")
        else:
            print(f"Fixed {count} image paths")
    
    return success


def validate_image_paths_in_html(html_file_path: str, logger=None) -> bool:
    """
    Convenience function to validate image paths in HTML file
    
    Args:
        html_file_path: HTML file path
        logger: Logger function
        
    Returns:
        bool: Whether all image paths are valid
    """
    fixer = ImagePathFixer(logger)
    all_valid, invalid_paths = fixer.validate_image_paths(html_file_path)
    
    if not all_valid:
        if logger:
            logger(f"Found {len(invalid_paths)} invalid image paths", "WARNING")
            for path in invalid_paths:
                logger(f"   Invalid path: {path}", "WARNING")
        else:
            print(f"Found {len(invalid_paths)} invalid image paths")
            for path in invalid_paths:
                print(f"   Invalid path: {path}")
    
    return all_valid


# Test function
def test_image_path_fixer():
    """Test image path fixing functionality"""
    print("Testing image path fixer module...")
    
    # Add specific test logic here
    fixer = ImagePathFixer()
    print(f"Image path fixer initialized successfully")
    print(f"Supported image formats: {fixer.image_extensions}")
    
    print("Image path fixer module test completed")


if __name__ == '__main__':
    test_image_path_fixer()
