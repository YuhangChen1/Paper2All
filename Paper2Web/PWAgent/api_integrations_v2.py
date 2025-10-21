#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import yaml
import base64
import requests
from pathlib import Path
from typing import Dict, Optional, Any

# Add path relative to Paper2Web directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class VisionAnalysisAPIV2:
    """Vision Analysis API v2 - Rule-based component type determination"""
    
    def __init__(self, model_name: str = "openrouter_qwen2_5_VL_72B", logger=None):
        self.model_name = model_name
        self.api_key = os.environ.get('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1"
        self.logger = logger
        
        # Load prompt template
        self.prompt_template = self.load_prompt_template()
    
    def log(self, message: str):
        """Log message"""
        if self.logger:
            self.logger(message)
        else:
            print(message)
    
    def load_prompt_template(self) -> Dict[str, Any]:
        """Load visual analysis prompt template"""
        template_path = Path(__file__).parent.parent / "utils" / "prompt_templates" / "website_visual_analysis_v2.yaml"
        
        # Check if file exists
        if not template_path.exists():
            error_msg = f"❌ Fatal error: Visual analysis prompt template file not found: {template_path}"
            print(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_data = yaml.safe_load(f)
                
            # Validate template data integrity
            if not isinstance(template_data, dict):
                error_msg = f"❌ Fatal error: Invalid prompt template format, must be dictionary: {template_path}"
                print(error_msg)
                raise ValueError(error_msg)
            
            # Check required fields
            required_fields = ["system_message", "user_message_template"]
            missing_fields = [field for field in required_fields if field not in template_data]
            if missing_fields:
                error_msg = f"❌ Fatal error: Prompt template missing required fields {missing_fields}: {template_path}"
                print(error_msg)
                raise ValueError(error_msg)
            
            print(f"✅ Successfully loaded visual analysis prompt template v2: {template_path}")
            return template_data
            
        except yaml.YAMLError as e:
            error_msg = f"❌ Fatal error: Prompt template YAML format error: {e} in {template_path}"
            print(error_msg)
            raise yaml.YAMLError(error_msg)
        except Exception as e:
            error_msg = f"❌ Fatal error: Unable to load visual analysis prompt template: {e} from {template_path}"
            print(error_msg)
            raise Exception(error_msg)
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64"""
        try:
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise Exception(f"Image encoding failed: {e}")
    
    def determine_category_by_rule(self, component: Dict) -> str:
        """
        v2 new feature: Determine component type based on rules instead of AI
        
        Args:
            component: Component data containing element_info field
            
        Returns:
            str: Component type (Header/Hero | Navigator | Content Block | Component/Card)
        """
        element_info = component.get('element_info', {})
        tag_name = element_info.get('tag_name', '').lower()
        element_id = element_info.get('id', '').lower()
        element_class = element_info.get('class', '').lower()
        
        # Rule-based judgment based on tag name
        if tag_name == 'header':
            self.log(f"Rule judgment: {tag_name} -> Header/Hero")
            return "Header/Hero"
        elif tag_name == 'nav':
            self.log(f"Rule judgment: {tag_name} -> Navigator")  
            return "Navigator"
        elif tag_name == 'section':
            # For section, check if ID and class have special hints
            if 'nav' in element_id or 'nav' in element_class:
                self.log(f"Rule judgment: {tag_name} (contains navigation style) -> Navigator")
                return "Navigator"
            elif 'header' in element_id or 'header' in element_class or 'hero' in element_id or 'hero' in element_class:
                self.log(f"Rule judgment: {tag_name} (contains header style) -> Header/Hero")
                return "Header/Hero"
            elif 'card' in element_id or 'card' in element_class or 'component' in element_id:
                self.log(f"Rule judgment: {tag_name} (contains card style) -> Component/Card")
                return "Component/Card"
            else:
                self.log(f"Rule judgment: {tag_name} (general content) -> Content Block")
                return "Content Block"
        else:
            # Other tags default to content block
            self.log(f"Rule judgment: {tag_name} (other) -> Content Block")
            return "Content Block"
    
    def analyze_website_image(self, image_path: str, component: Dict, max_retries: int = 3) -> Dict[str, Any]:
        """
        Analyze website image using vision model, v2 version always uses rule-based category judgment
        
        Args:
            image_path: Image file path
            component: Component data for rule-based judgment
            max_retries: Maximum retry count, default 3
            
        Returns:
            Dict containing analysis result: {"is_needed_to_fix": bool, "category": str, "fix_suggest": str}
        """
        # v2 feature: Always use rule-based category judgment
        rule_based_category = self.determine_category_by_rule(component)
        
        # Check if file exists (check only once)
        if not os.path.exists(image_path):
            self.log(f"Image file does not exist: {image_path}")
            return {
                "is_needed_to_fix": True,
                "category": rule_based_category,  # Use rule-based classification
                "fix_suggest": f"Image file does not exist: {image_path}"
            }
        
        # Encode image (encode only once)
        try:
            base64_image = self.encode_image_to_base64(image_path)
            image_size = len(base64_image)
            self.log(f"Image encoding completed, size: {image_size} characters")
        except Exception as e:
            self.log(f"Image encoding failed: {e}")
            return {
                "is_needed_to_fix": True,
                "category": rule_based_category,  # Use rule-based classification
                "fix_suggest": f"Image encoding failed: {str(e)}"
            }
        
        for attempt in range(max_retries):
            try:
                attempt_info = f"Attempt {attempt + 1}/{max_retries}"
                self.log(f"🔄 {attempt_info} - Starting visual analysis: {os.path.basename(image_path)} (rule classification: {rule_based_category})")
                
                # Build request
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                # Select API configuration based on model type
                if "openrouter_qwen2_5_VL_72B" in self.model_name:
                    model_type = "qwen/qwen2.5-vl-72b-instruct"
                elif "qwen2.5-vl" in self.model_name or "qwen2_5_vl" in self.model_name:
                    model_type = "qwen/qwen2.5-vl-32b-instruct:free"
                else:
                    model_type = "qwen/qwen2.5-vl-72b-instruct"  # Default to 72B version
                
                # Record content sent to model (detailed recording only on first attempt)
                system_message = self.prompt_template.get("system_message", "You are an expert web developer.")
                user_text = self.prompt_template.get("user_message_template", "Analyze this website screenshot.")
                
                if attempt == 0:
                    self.log("=" * 80)
                    self.log("🤖 Content sent to visual analysis model v2:")
                    self.log(f"Model: {model_type}")
                    # Only save first 100 characters of system prompt
                    system_preview = system_message[:100] + "..." if len(system_message) > 100 else system_message
                    self.log(f"System prompt: {system_preview}")
                    # Only save first 100 characters of user prompt
                    user_preview = user_text[:100] + "..." if len(user_text) > 100 else user_text
                    self.log(f"User prompt: {user_preview}")
                    self.log("=" * 80)
                
                payload = {
                    "model": model_type,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_message
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": user_text
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 99999,
                    "temperature": 0.2 + (attempt * 0.05)  # Slightly increase randomness with each retry
                }
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code != 200:
                    error_msg = f"{attempt_info} - API fail: {response.status_code} - {response.text}"
                    self.log(f"❌ {error_msg}")
                    
                    # If not the last attempt, continue retrying
                    if attempt < max_retries - 1:
                        import time
                        wait_time = (attempt + 1) * 2
                        time.sleep(wait_time)
                        continue
                    else:
                        # Last attempt failed, return default response
                        self.log(f"💥 All retries failed, return default analysis result")
                        return {
                            "is_needed_to_fix": True,
                            "category": rule_based_category,  # Use rule-determined category
                        }
                
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Record complete model response
                self.log("=" * 80)
                self.log(f"Complete response: {content}")
                self.log("=" * 80)
                
                # Parse JSON response but force use rule-classified category
                analysis_result = self._parse_vision_response(content, attempt_info)
                
                # v2 Feature: Force use rule-determined category，忽略AI的category判断
                analysis_result["category"] = rule_based_category
                
                # Verify result validity
                if self._is_valid_vision_result(analysis_result):
                    self.log("=" * 80)
                    self.log(f"Needs fixing: {analysis_result.get('is_needed_to_fix', 'Unknown')}")
                    self.log(f"Fix suggestion: {analysis_result.get('fix_suggest', 'No suggestion')}")
                    self.log("=" * 80)
                    return analysis_result
                else:
                    # If not the last attempt, continue retrying
                    if attempt < max_retries - 1:
                        self.log(f"⚠️ {attempt_info} - Analysis result invalid, preparing to retry...")
                        continue
                    else:
                        # Last attempt failed, return default response
                        self.log(f"💥 All retries returned invalid results, return default analysis result")
                        return {
                            "is_needed_to_fix": True,
                            "category": rule_based_category,  # Use rule-determined category
                            "fix_suggest": "Analysis result parsing failed, suggest manual page check"
                        }
                        
            except Exception as e:
                error_msg = f"{attempt_info} - Visual analysis API call exception: {str(e)}"
                self.log(f"❌ {error_msg}")
                
                # If not the last attempt, continue retrying
                if attempt < max_retries - 1:
                    import time
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    # Last attempt failed, return default response
                    self.log(f"💥 All retries failed with exception, return default analysis result")
                    return {
                        "is_needed_to_fix": True,
                        "category": rule_based_category,  # Use rule-determined category
                        "fix_suggest": f"API fail: {str(e)}"
                    }
    
    def _parse_vision_response(self, content: str, attempt_info: str) -> Dict[str, Any]:
        """解析视觉分析响应"""
        try:
            # Try to parse JSON directly
            return json.loads(content)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON part
            try:
                # Find JSON code block
                start = content.find('```json')
                if start != -1:
                    start += 7  # Skip ```json
                    end = content.find('```', start)
                    if end != -1:
                        json_content = content[start:end].strip()
                        return json.loads(json_content)
                    else:
                        raise json.JSONDecodeError("End marker not found", content, 0)
                else:
                    # Try to find content surrounded by braces
                    start = content.find('{')
                    end = content.rfind('}')
                    if start != -1 and end != -1:
                        json_content = content[start:end+1]
                        return json.loads(json_content)
                    else:
                        raise json.JSONDecodeError("JSON format not found", content, 0)
            except json.JSONDecodeError:
                self.log(f"⚠️ {attempt_info} - JSON解析失败，使用文本分析: {content}")
                # Simple text analysis as fallback
                is_needed = "fix" in content.lower() or "improve" in content.lower() or "issue" in content.lower()
                
                return {
                    "is_needed_to_fix": is_needed,
                    "category": "Content Block",  # Default value, will be overridden by rules
                    "fix_suggest": content
                }
    
    def _is_valid_vision_result(self, result: Dict[str, Any]) -> bool:
        if not isinstance(result, dict):
            return False
        
        # Ensure required fields are included
        if "is_needed_to_fix" not in result:
            result["is_needed_to_fix"] = True
        
        if "category" not in result:
            result["category"] = "Content Block"  # Default category, will be overridden by rules
        
        if "fix_suggest" not in result:
            result["fix_suggest"] = "Requires further analysis"
            
        # Verify field types
        if not isinstance(result.get("is_needed_to_fix"), bool):
            # Try to convert
            fix_needed = result.get("is_needed_to_fix")
            if isinstance(fix_needed, str):
                result["is_needed_to_fix"] = fix_needed.lower() in ["true", "yes", "1"]
            else:
                result["is_needed_to_fix"] = True
        
        if not isinstance(result.get("fix_suggest"), str):
            result["fix_suggest"] = str(result.get("fix_suggest", ""))
        
        # v2 Feature: category will be overridden by rule judgment，所以这里不需要验证
            
        return True


class CodeOptimizationAPIV2:
    """Code optimization API call class v2 - supports rule-based category prompt selection"""
    
    def __init__(self, model_name: str = "openrouter_qwen3_coder", logger=None):
        self.model_name = model_name
        self.api_key = os.environ.get('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1"
        self.logger = logger
        

        self.prompt_templates = self.load_all_prompt_templates()
    
    def log(self, message: str):
        """Record log"""
        if self.logger:
            self.logger(message)
        else:
            print(message)
    
    def load_all_prompt_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load all code optimization prompt templates"""
        template_dir = Path(__file__).parent.parent / "utils" / "prompt_templates"
        
        # Define mapping from category to template files
        category_template_mapping = {
            "Navigator": "navigator.html_optimization.yaml",
            "Header/Hero": "head_html_optimization.yaml", 
            "Content Block": "content_block_code_optimization.yaml",
            "Component/Card": "component_card_html_optimization.yaml"
        }
        
        templates = {}
        
        for category, template_filename in category_template_mapping.items():
            template_path = template_dir / template_filename
            
            try:
                # Check if file exists
                if not template_path.exists():
                    error_msg = f"❌ Fatal error: Code optimization prompt template file does not exist: {template_path}"
                    print(error_msg)
                    raise FileNotFoundError(error_msg)
                
                with open(template_path, 'r', encoding='utf-8') as f:
                    template_data = yaml.safe_load(f)
                    
                # Verify template data integrity
                if not isinstance(template_data, dict):
                    error_msg = f"❌ Fatal error: Prompt template format invalid, must be dictionary format: {template_path}"
                    print(error_msg)
                    raise ValueError(error_msg)
                
                # Check required fields
                required_fields = ["system_message", "user_message_template"]
                missing_fields = [field for field in required_fields if field not in template_data]
                if missing_fields:
                    error_msg = f" {missing_fields}: {template_path}"
                    print(error_msg)
                    raise ValueError(error_msg)
                
                templates[category] = template_data
                
            except yaml.YAMLError as e:
                raise yaml.YAMLError(error_msg)
            except Exception as e:
                raise Exception(error_msg)
        
        return templates
    
    def optimize_html_code(self, html_content: str, fix_suggest: str, category: str = "Content Block", max_retries: int = 3) -> str:
        """
        Use code model to optimize HTML code, v2 version enhanced error handling
        
        Args:
            html_content: Original HTML code
            fix_suggest: Optimization suggestion
            category: Component classification (Navigator | Header/Hero | Content Block | Component/Card)
            max_retries: Maximum retry count, default 3
            
        Returns:
            Optimized HTML code
        """
        # Ensure category is valid
        if category not in self.prompt_templates:
            category = "Content Block"
        
        # Select corresponding prompt template
        prompt_template = self.prompt_templates[category]
        
        for attempt in range(max_retries):
            try:
                attempt_info = f"{attempt + 1}/{max_retries} try "
                
                # Build request
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                # Select based on model
                if "openrouter_qwen3_coder" in self.model_name:
                    model_type = "qwen/qwen3-coder"
                elif "qwen3" in self.model_name.lower():
                    model_type = "qwen/qwen-2.5-coder-32b-instruct"
                else:
                    model_type = "qwen/qwen3-coder"  # 默认使用qwen3-coder
                
                # Build prompt
                system_message = prompt_template.get("system_message", "You are an expert web developer.")
                user_message = prompt_template.get("user_message_template", "").format(
                    fix_suggest=fix_suggest,
                    html_content=html_content
                )
                
                # Record content sent to model (detailed recording only on first attempt)
                if attempt == 0:
                    self.log("=" * 80)
                    self.log(f"Model: {model_type}")
                    self.log(f"Using template: {category}")
                    # Only save first 100 characters of system prompt
                    system_preview = system_message[:100] + "..." if len(system_message) > 100 else system_message
                    self.log(f"System prompt: {system_preview}")
                    self.log("👥 User message content:")
                    fix_preview = fix_suggest[:100] + "..." if len(fix_suggest) > 100 else fix_suggest
                    self.log(f"Optimization suggestion: {fix_preview}")
                    # Only save first 100 characters of HTML code
                    html_preview = html_content[:100] + "..." if len(html_content) > 100 else html_content
                    self.log(f"📄 Original HTML code: {html_preview}")
                    self.log("=" * 80)
                
                payload = {
                    "model": model_type,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_message
                        },
                        {
                            "role": "user",
                            "content": user_message
                        }
                    ],
                    "max_tokens": 99999,
                    "temperature": 0.1 + (attempt * 0.1)  # Slightly increase randomness with each retry
                }
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                
                if response.status_code != 200:
                    self.log(f"❌ {error_msg}")
                    
                    # If not the last attempt, continue retrying
                    if attempt < max_retries - 1:
                        import time
                        wait_time = (attempt + 1) * 2  # Exponential backoff：2秒、4秒、6秒
                        time.sleep(wait_time)
                        continue
                    else:
                        # v2特性：如果API失败，返回原始代码而不是抛异常
                        return html_content
                
                result = response.json()
                optimized_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Record complete model response
                self.log("=" * 80)
                self.log("Complete response:")
                self.log(optimized_content)
                self.log("=" * 80)
                
                # Clean response content, extract HTML code
                cleaned_content = self._extract_html_from_response(optimized_content)
                
                # v2 Enhancement: Stricter HTML validation
                if not self._is_valid_html_v2(cleaned_content, html_content):
                    
                    # If not the last attempt, continue retrying
                    if attempt < max_retries - 1:
                        self.log(f"🔄 Preparing to retry...")
                        continue
                    else:
                        # 最后一次尝试失败，返回原始代码
                        self.log(f"💥 All retries returned invalid HTML, return original code")
                        return html_content
                
                # 成功获得有效的HTML
                self.log("=" * 80)
                self.log("✅ Optimized HTML code:")
                self.log(cleaned_content)
                self.log("=" * 80)
                
                return cleaned_content
                
            except Exception as e:
                error_msg = f"{attempt_info} - Code optimization API call exception: {str(e)}"
                self.log(f"❌ {error_msg}")
                
                # If not the last attempt, continue retrying
                if attempt < max_retries - 1:
                    import time
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
        
                    return html_content
    
    def _extract_html_from_response(self, content: str) -> str:
        """Extract HTML code from response"""
        if "```html" in content:
            start = content.find("```html") + 7
            end = content.find("```", start)
            if end != -1:
                return content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end != -1:
                return content[start:end].strip()
        return content.strip()
    
    def _is_valid_html_v2(self, content: str, original_content: str) -> bool:
        """v2 Enhanced HTML validation"""
        # Basic HTML tag check
        has_valid_tags = ("<header" in content or "<nav" in content or "<section" in content or 
                         "<div" in content or "<article" in content or "<main" in content)
        
        if not has_valid_tags:
            return False
        
        # Length check: optimized content should not be too short (relative to original content)
        min_length = max(50, len(original_content) * 0.3)  # 至少50字符或原始长度的30%
        if len(content) < min_length:
            return False
        
        # Check if empty or contains only whitespace
        if not content.strip():
            return False
        
        return True


# 工厂函数
def create_vision_analyzer_v2(model_name: str = "openrouter_qwen2_5_VL_72B", logger=None) -> VisionAnalysisAPIV2:
    """Create visual analyzer instance v2"""
    return VisionAnalysisAPIV2(model_name, logger)


def create_code_optimizer_v2(model_name: str = "openrouter_qwen3_coder", logger=None) -> CodeOptimizationAPIV2:
    """Create code optimizer instance v2"""
    return CodeOptimizationAPIV2(model_name, logger)


# 测试函数
def test_apis_v2():
    """Test API functionality v2"""
    print("Test API integration v2...")
    
    # 测试视觉分析
    vision_api = create_vision_analyzer_v2()
    print(f"Visual analysis API v2 initialized successfully: {vision_api.model_name}")
    
    # Test rule classification
    test_component = {
        "element_info": {
            "tag_name": "header",
            "id": "main-header", 
            "class": "hero-section"
        }
    }
    category = vision_api.determine_category_by_rule(test_component)
    
    # 测试代码优化
    code_api = create_code_optimizer_v2()
    print(f"Code optimization API v2 initialized successfully: {code_api.model_name}")
    
    print("API v2 test completed")


if __name__ == '__main__':
    test_apis_v2()