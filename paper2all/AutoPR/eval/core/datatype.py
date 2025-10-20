# eval/core/datatype.py

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class BaseEvalType(str, Enum):
    SINGLE_NOTE = "single_note"
    PREFERENCE = "preference"
    FINE_GRAINED = "fine_grained"
    TRADITIONAL_METRICS = "traditional_metrics"

# Enum to define how images are handled in the evaluation
class ImageHandlingStrategy(str, Enum):
    NONE = "none" # No images are processed or mentioned
    REAL_IMAGES = "real" # Real image data is sent to the model
    TEXT_PLACEHOLDERS = "placeholder" # Only text placeholders (e.g., [Image 1]) are included in the prompt


class EvaluationConfig(BaseModel):
    eval_name: str
    base_type: BaseEvalType
    description: str
    target_data_source: str = "original"
    instruction: Optional[str] = None
    model: Optional[str] = "gemini-1.5-flash-latest"
    response_schema: Optional[Dict[str, Any]] = None
    
    include_images: ImageHandlingStrategy = ImageHandlingStrategy.NONE 
    
    include_pdf: bool = False
    criteria_subdir: Optional[str] = None
    n_samples: int = Field(default=1, description="Number of samples to run for each evaluation.")
    enable_rotation: bool = Field(default=False, description="For preference evaluation, enables rotating the order of items.")
    
    force_json_format_in_prompt: bool = Field(default=False, description="If true, injects JSON format instructions into the prompt instead of using native tool/JSON mode.")


class PromotionDataItem(BaseModel):
    id: str
    title: str
    arxiv_id: Optional[str] = None
    PDF_path: Optional[str] = None
    platform_source: str = Field(default="XHS_NOTE")
    image_links: List[str] = Field(..., alias='figure_path')
    markdown_content: str
    origin_data: Optional[Dict[str, Any]] = None
    is_pr_test: bool = Field(default=False, exclude=True)


class MetricItem(BaseModel):
    id: str
    eval_name: str
    evaluation_results: Dict[str, Any]
    status: str = "completed"
    error: Optional[str] = None

class ChecklistItem(BaseModel):
    description: str
    max_score: int

class FineGrainedChecklist(BaseModel):
    name: str
    checklist: List[ChecklistItem]