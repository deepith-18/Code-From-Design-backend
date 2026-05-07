"""
Image processing utilities for preparing images for analysis
"""
import cv2
import numpy as np
from PIL import Image
import io


class ImageProcessor:
    """Handles image preprocessing and analysis"""
    
    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
        """
        Load image from bytes and convert to OpenCV format
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            OpenCV image (BGR format)
        """
        # Convert bytes to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_bgr
    
    @staticmethod
    def preprocess_for_detection(image: np.ndarray) -> dict:
        """
        Preprocess image for UI element detection
        
        Returns dict with multiple processed versions:
        - grayscale: For edge detection
        - binary: For contour detection
        - edges: Canny edge detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Binary threshold
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological operations to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        return {
            'original': image,
            'grayscale': gray,
            'binary': binary,
            'edges': edges,
            'dilated': dilated,
            'height': image.shape[0],
            'width': image.shape[1]
        }
    
    @staticmethod
    def get_color_statistics(image: np.ndarray) -> dict:
        """
        Analyze color distribution in the image
        UI designs typically have limited color palettes
        """
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate color histograms
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Count unique colors (reduced to 64 colors per channel)
        reduced = image // 64 * 64
        unique_colors = len(np.unique(reduced.reshape(-1, 3), axis=0))
        
        # Calculate color variance
        color_std = np.std(image, axis=(0, 1))
        
        return {
            'unique_colors': unique_colors,
            'color_std_mean': float(np.mean(color_std)),
            'saturation_mean': float(np.mean(s_hist)),
            'value_mean': float(np.mean(v_hist))
        }
    
    @staticmethod
    def detect_faces(image: np.ndarray) -> int:
        """
        Detect faces in the image (to reject human photos)
        Returns the number of faces detected
        """
        try:
            # Load OpenCV's pre-trained face detector
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            return len(faces)
        except Exception:
            # If face detection fails, assume no faces
            return 0