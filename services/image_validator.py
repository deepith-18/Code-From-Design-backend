"""
Image Validation Service
Determines if an uploaded image is a UI design or not
"""
import cv2
import numpy as np
from models.schemas import ValidationResult
from utils.image_processor import ImageProcessor


class ImageValidator:
    """
    Validates whether an image is a UI design using multiple heuristics
    
    Strategy:
    1. Check for human faces (UI designs shouldn't have faces)
    2. Analyze color distribution (UI designs have limited palettes)
    3. Detect geometric shapes (UI designs have many rectangles)
    4. Analyze edge density (UI designs have clear boundaries)
    """
    
    def __init__(self):
        self.processor = ImageProcessor()
        
        # Thresholds for validation (tuned through testing)
        self.thresholds = {
            'max_faces': 0,  # No faces allowed
            'min_rectangles': 3,  # At least 3 rectangular shapes
            'max_unique_colors': 150,  # Limited color palette
            'min_edge_density': 0.05,  # Minimum edge density
            'max_edge_density': 0.40,  # Maximum edge density (photos have more)
        }
    
    def validate(self, image: np.ndarray) -> ValidationResult:
        """
        Main validation method
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            ValidationResult with is_ui_design boolean and explanation
        """
        # Initialize metrics
        metrics = {}
        reasons = []
        
        # 1. Face Detection Check
        num_faces = self.processor.detect_faces(image)
        metrics['num_faces'] = num_faces
        
        if num_faces > self.thresholds['max_faces']:
            return ValidationResult(
                is_ui_design=False,
                confidence=0.95,
                reason="Image contains human faces - this appears to be a photograph, not a UI design.",
                metrics=metrics
            )
        
        # 2. Color Analysis
        color_stats = self.processor.get_color_statistics(image)
        metrics.update(color_stats)
        
        has_limited_palette = color_stats['unique_colors'] < self.thresholds['max_unique_colors']
        
        if not has_limited_palette:
            reasons.append(f"Too many unique colors ({color_stats['unique_colors']})")
        
        # 3. Geometric Shape Detection
        processed = self.processor.preprocess_for_detection(image)
        rectangles = self._detect_rectangles(processed['dilated'])
        metrics['num_rectangles'] = len(rectangles)
        
        has_enough_shapes = len(rectangles) >= self.thresholds['min_rectangles']
        
        if not has_enough_shapes:
            reasons.append(f"Too few rectangular shapes ({len(rectangles)})")
        
        # 4. Edge Density Analysis
        edge_density = self._calculate_edge_density(processed['edges'])
        metrics['edge_density'] = edge_density
        
        has_ui_edge_pattern = (
            self.thresholds['min_edge_density'] <= edge_density <= 
            self.thresholds['max_edge_density']
        )
        
        if not has_ui_edge_pattern:
            reasons.append(f"Edge density ({edge_density:.3f}) outside UI range")
        
        # 5. Calculate overall confidence
        passed_checks = sum([
            num_faces == 0,
            has_limited_palette,
            has_enough_shapes,
            has_ui_edge_pattern
        ])
        
        confidence = passed_checks / 4.0
        
        # Decision logic
        if passed_checks >= 3:
            # Likely a UI design
            return ValidationResult(
                is_ui_design=True,
                confidence=confidence,
                reason="Image appears to be a UI design with geometric shapes and limited colors.",
                metrics=metrics
            )
        else:
            # Not a UI design
            reason_text = "Image does not appear to be a UI design. Issues: " + "; ".join(reasons)
            return ValidationResult(
                is_ui_design=False,
                confidence=1.0 - confidence,
                reason=reason_text,
                metrics=metrics
            )
    
    def _detect_rectangles(self, edges: np.ndarray) -> list:
        """
        Detect rectangular contours in the edge-detected image
        
        Args:
            edges: Edge-detected image
            
        Returns:
            List of rectangle contours
        """
        # Find contours
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        rectangles = []
        image_area = edges.shape[0] * edges.shape[1]
        
        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's approximately a rectangle (4 corners)
            if len(approx) >= 4 and len(approx) <= 6:
                area = cv2.contourArea(contour)
                
                # Filter out very small or very large rectangles
                if 0.001 < (area / image_area) < 0.8:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (not too elongated)
                    aspect_ratio = max(w, h) / (min(w, h) + 1)
                    if aspect_ratio < 10:
                        rectangles.append((x, y, w, h))
        
        return rectangles
    
    def _calculate_edge_density(self, edges: np.ndarray) -> float:
        """
        Calculate the density of edges in the image
        
        UI designs have moderate edge density (not too sparse, not too dense)
        """
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_pixels = np.count_nonzero(edges)
        
        return edge_pixels / total_pixels