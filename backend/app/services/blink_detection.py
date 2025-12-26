import cv2
import numpy as np
from typing import Tuple, Optional
import time

class BlinkDetector:
    """
    Simple OpenCV-based blink detection using eye cascade classifier
    """
    
    def __init__(self):
        # Load eye cascade classifier
        eye_cascade_paths = [
            cv2.data.haarcascades + "haarcascade_eye.xml",
            "ml_models/haarcascade_eye.xml",
            "./haarcascade_eye.xml"
        ]
        
        self.eye_cascade = None
        for path in eye_cascade_paths:
            try:
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    self.eye_cascade = cascade
                    print(f"✓ Loaded eye cascade from: {path}")
                    break
            except:
                continue
        
        if self.eye_cascade is None:
            # Fallback: download from OpenCV
            print("Downloading eye cascade...")
            import urllib.request
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
            urllib.request.urlretrieve(url, "haarcascade_eye.xml")
            self.eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
        
        # Blink detection state
        self.blink_count = 0
        self.eyes_closed_frames = 0
        self.eyes_open_frames = 0
        self.last_eye_count = 0
        
        # Thresholds
        self.min_blinks_required = 1  # At least 1 blink
        self.frames_for_closed = 2     # Eyes closed for 2 frames
        self.frames_for_open = 2       # Eyes open for 2 frames
    
    def reset(self):
        """Reset blink counter"""
        self.blink_count = 0
        self.eyes_closed_frames = 0
        self.eyes_open_frames = 0
        self.last_eye_count = 0
    
    def detect_eyes(self, frame: np.ndarray) -> int:
        """
        Detect eyes in frame
        Returns: Number of eyes detected
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        return len(eyes)
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame for blink detection
        
        Returns:
            dict: {
                'eyes_detected': int,
                'blink_count': int,
                'blink_detected': bool,
                'status': str,
                'ready_to_capture': bool
            }
        """
        num_eyes = self.detect_eyes(frame)
        
        # State machine for blink detection
        if num_eyes >= 2:
            # Eyes are open
            self.eyes_open_frames += 1
            self.eyes_closed_frames = 0
            
            # If eyes were closed and now open → blink detected
            if self.last_eye_count == 0 and self.eyes_open_frames >= self.frames_for_open:
                self.blink_count += 1
                print(f"✓ Blink detected! Count: {self.blink_count}")
        
        elif num_eyes == 0:
            # Eyes are closed
            self.eyes_closed_frames += 1
            self.eyes_open_frames = 0
        
        self.last_eye_count = num_eyes
        
        # Determine status
        if self.blink_count >= self.min_blinks_required:
            status = f"✓ Blink Confirmed ({self.blink_count} blinks)"
            ready = True
            blink_detected = True
        elif num_eyes >= 2:
            status = "Eyes Open - Please Blink"
            ready = False
            blink_detected = False
        elif num_eyes == 0:
            status = "Eyes Closed..."
            ready = False
            blink_detected = False
        else:
            status = "One Eye Detected"
            ready = False
            blink_detected = False
        
        return {
            'eyes_detected': num_eyes,
            'blink_count': self.blink_count,
            'blink_detected': blink_detected,
            'status': status,
            'ready_to_capture': ready
        }
    
    def is_ready_to_capture(self) -> bool:
        """Check if enough blinks detected"""
        return self.blink_count >= self.min_blinks_required


# Singleton instance
blink_detector = BlinkDetector()