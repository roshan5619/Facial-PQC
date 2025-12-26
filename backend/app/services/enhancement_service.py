# backend/app/services/enhancement_service.py
# COMPLETE FILE - Replace entire content

import cv2
import numpy as np
from app.core.config import settings

class ImageEnhancer:
    """
    Image enhancement service wrapping enhance.py functionality
    """
    
    def __init__(self):
        self.clahe_clip_limit = settings.CLAHE_CLIP_LIMIT
        self.clahe_grid_size = (settings.CLAHE_GRID_SIZE, settings.CLAHE_GRID_SIZE)
        self.gamma = settings.GAMMA_CORRECTION
    
    def gamma_correction(self, image: np.ndarray, gamma: float = None) -> np.ndarray:
        """
        Apply gamma correction to image
        
        Args:
            image: Input grayscale image
            gamma: Gamma value (default from settings)
        
        Returns:
            Gamma corrected image
        """
        if gamma is None:
            gamma = self.gamma
        
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 
            for i in np.arange(0, 256)
        ]).astype("uint8")
        
        return cv2.LUT(image, table)
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply enhancement pipeline to normalize image
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            Enhanced grayscale image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size
        )
        enhanced = clahe.apply(gray)
        
        # Apply gamma correction
        enhanced = self.gamma_correction(enhanced, self.gamma)
        
        return enhanced
    
    def enhance_roi(self, roi_image: np.ndarray) -> np.ndarray:
        """
        Apply enhancement specifically to ROI region
        
        Args:
            roi_image: Input ROI image (should already be grayscale)
        
        Returns:
            Enhanced ROI image
        """
        # Ensure grayscale
        if len(roi_image.shape) == 3:
            roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size
        )
        enhanced = clahe.apply(roi_image)
        
        # Apply gamma correction
        enhanced = self.gamma_correction(enhanced, self.gamma)
        
        return enhanced


# Singleton instance
image_enhancer = ImageEnhancer()