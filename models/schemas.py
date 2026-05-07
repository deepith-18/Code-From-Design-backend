"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel
from typing import List, Optional, Dict


class UIElement(BaseModel):
    """Represents a detected UI element"""
    type: str  # 'button', 'input', 'text', 'container', 'image'
    x: int
    y: int
    width: int
    height: int
    text: Optional[str] = None
    placeholder: Optional[str] = None
    confidence: float  # 0.0 to 1.0


class GeneratedCode(BaseModel):
    """Generated code for different formats"""
    html: str
    css: str
    react: str


class AnalysisResponse(BaseModel):
    """Response from the analysis endpoint"""
    valid: bool
    message: str
    elements: Optional[List[UIElement]] = None
    generated_code: Optional[GeneratedCode] = None
    preview_data: Optional[Dict] = None


class ValidationResult(BaseModel):
    """Result of image validation"""
    is_ui_design: bool
    confidence: float
    reason: str
    metrics: Dict[str, float]