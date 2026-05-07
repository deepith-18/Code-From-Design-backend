"""
UI Element Detector
Extracts UI elements from a validated UI design image
"""
import cv2
import numpy as np
from typing import List
from models.schemas import UIElement
from utils.image_processor import ImageProcessor


class UIDetector:
    """
    Detects UI elements in a design image
    
    Detection Strategy:
    1. Find all rectangular regions (contours)
    2. Classify by size and position:
       - Large regions → containers
       - Medium regions → buttons
       - Small tall regions → inputs
       - Text regions → labels/headings
    3. Extract text using OCR (optional, simplified here)
    4. Create structured representation
    """
    
    def __init__(self):
        self.processor = ImageProcessor()
    
    def detect_elements(self, image: np.ndarray) -> List[UIElement]:
        """
        Main detection method
        
        Args:
            image: OpenCV image (BGR)
            
        Returns:
            List of detected UI elements
        """
        # Preprocess image
        processed = self.processor.preprocess_for_detection(image)
        height, width = processed['height'], processed['width']
        
        # Find all contours
        contours = self._find_contours(processed['dilated'])
        
        # Filter and classify contours
        elements = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            image_area = width * height
            
            # Skip very small or very large regions
            area_ratio = area / image_area
            if area_ratio < 0.001 or area_ratio > 0.85:
                continue
            
            # Classify element based on size and aspect ratio
            element_type = self._classify_element(x, y, w, h, width, height)
            
            # Extract text if it's a text region (simplified - uses placeholder)
            text = self._extract_text(image, x, y, w, h, element_type)
            
            # Calculate confidence based on contour quality
            confidence = self._calculate_confidence(contour, w, h)
            
            # Create UI element
            element = UIElement(
                type=element_type,
                x=x,
                y=y,
                width=w,
                height=h,
                text=text if text else None,
                placeholder=self._generate_placeholder(element_type),
                confidence=confidence
            )
            
            elements.append(element)
        
        # Sort elements by position (top to bottom, left to right)
        elements = self._sort_elements(elements)
        
        # Group related elements (forms, navigation)
        elements = self._group_elements(elements, width, height)
        
        return elements
    
    def _find_contours(self, edges: np.ndarray) -> list:
        """Find contours in edge-detected image"""
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours
    
    def _classify_element(self, x: int, y: int, w: int, h: int, 
                         img_width: int, img_height: int) -> str:
        """
        Classify UI element based on dimensions and position
        
        Classification rules:
        - Container: Large area (>15% of image)
        - Header: Top region, wide
        - Button: Small/medium, roughly square or slightly wide
        - Input: Narrow height, medium width
        - Text: Small area
        """
        area_ratio = (w * h) / (img_width * img_height)
        aspect_ratio = w / (h + 1)
        position_ratio = y / img_height
        
        # Container: Large area
        if area_ratio > 0.15:
            return 'container'
        
        # Header: Top of page, wide
        if position_ratio < 0.15 and aspect_ratio > 3:
            return 'header'
        
        # Input field: Narrow, horizontal
        if 20 < h < 60 and aspect_ratio > 2 and aspect_ratio < 10:
            return 'input'
        
        # Button: Moderate size, roughly square to slightly wide
        if 25 < h < 80 and 0.5 < aspect_ratio < 5 and area_ratio > 0.005:
            return 'button'
        
        # Image placeholder: Square or vertical
        if aspect_ratio < 1.5 and area_ratio > 0.02:
            return 'image'
        
        # Default: Text or label
        return 'text'
    
    def _extract_text(self, image: np.ndarray, x: int, y: int, 
                     w: int, h: int, element_type: str) -> str:
        """
        Extract text from region
        
        Note: For production, integrate Tesseract OCR
        For this demo, we use intelligent placeholders
        """
        # In production: Use pytesseract.image_to_string()
        # For demo: Return semantic placeholders
        
        if element_type == 'button':
            return 'Click Here'
        elif element_type == 'input':
            return ''
        elif element_type == 'text':
            return 'Sample Text'
        elif element_type == 'header':
            return 'Page Title'
        
        return None
    
    def _generate_placeholder(self, element_type: str) -> str:
        """Generate placeholder text for inputs"""
        placeholders = {
            'input': 'Enter text here...',
            'button': '',
            'text': '',
            'container': '',
            'header': '',
            'image': ''
        }
        return placeholders.get(element_type, '')
    
    def _calculate_confidence(self, contour, w: int, h: int) -> float:
        """
        Calculate confidence score for detected element
        Based on contour regularity
        """
        # Get contour perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Ideal perimeter for a rectangle
        ideal_perimeter = 2 * (w + h)
        
        # Calculate similarity (closer to 1.0 is better)
        if ideal_perimeter > 0:
            similarity = min(perimeter, ideal_perimeter) / max(perimeter, ideal_perimeter)
        else:
            similarity = 0.5
        
        # Confidence is similarity clamped to [0.5, 1.0]
        return max(0.5, min(1.0, similarity))
    
    def _sort_elements(self, elements: List[UIElement]) -> List[UIElement]:
        """
        Sort elements by visual position (top to bottom, left to right)
        """
        return sorted(elements, key=lambda e: (e.y // 50, e.x))
    
    def _group_elements(self, elements: List[UIElement], 
                       width: int, height: int) -> List[UIElement]:
        """
        Group related elements (e.g., form fields, navigation items)
        This is a simplified version - in production, use spatial clustering
        """
        # For now, just return elements as-is
        # In production: Implement DBSCAN clustering for spatial grouping
        return elements