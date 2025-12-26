import os
import cv2
import numpy as np
import torch
import joblib
import torch.nn as nn
from typing import Tuple, Optional, Dict
from sklearn.metrics.pairwise import cosine_similarity
from app.core.config import settings
from app.services.enhancement_service import image_enhancer

# PennyLane for Quantum Circuit
import pennylane as qml

class VQCFaceService:
    """
    Integrates your VQC quantum model for face detection and recognition
    Based on your prediction_new.py
    """
    
    def __init__(self):
        self.models_dir = settings.MODELS_DIR
        self.roi_size = (settings.ROI_SIZE, settings.ROI_SIZE)
        self.n_qubits = settings.N_QUBITS
        self.cosine_threshold = settings.COSINE_THRESHOLD
        # Model objects
        self.vqc_model = None
        self.pca_detection = None
        self.pca_recognition = None
        self.scaler_recognition = None
        self.recognition_db = None
        self.used_paths = None
        self.roi_paths = None  # ADD THIS LINE
        self.face_cascade = None
        # Load Haar Cascade for face detection
        self.load_haar_cascade()
        
        # Load ML models
        self.load_models()
        
        print("✓ VQC Face Service initialized")
    
    def load_haar_cascade(self):
        """Load Haar Cascade classifier"""
        haar_paths = [
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
            os.path.join(self.models_dir, "haarcascade_frontalface_default.xml"),
            "./haarcascade_frontalface_default.xml"
        ]
        
        self.face_cascade = None
        for path in haar_paths:
            if os.path.exists(path):
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    self.face_cascade = cascade
                    print(f"✓ Loaded Haar cascade from: {path}")
                    break
        
        if self.face_cascade is None or self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade")
    
    def load_models(self):
        """Load all ML models"""
        try:
            # Define Quantum Circuit
            dev = qml.device("default.qubit", wires=self.n_qubits)
            
            @qml.qnode(dev, interface="torch")
            def quantum_circuit(inputs, weights):
                qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))
                qml.templates.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
                return [qml.expval(qml.PauliZ(i)) for i in range(2)]
            
            # Define VQC Model
            class VQC(nn.Module):
                def __init__(self):
                    super().__init__()
                    weight_shapes = {"weights": (3, settings.N_QUBITS, 3)}
                    self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
                    self.fc = nn.Linear(2, 2)
                
                def forward(self, x):
                    return self.fc(self.qlayer(x))
            
            # Load VQC model
            vqc_path = os.path.join(self.models_dir, settings.VQC_MODEL_PATH)
            self.vqc_model = VQC()
            self.vqc_model.load_state_dict(
                torch.load(vqc_path, map_location=torch.device("cpu"), weights_only=True)
            )
            self.vqc_model.eval()
            print("✓ VQC model loaded")
            
            # Load PCA for detection (8 components)
            pca_detection_path = os.path.join(self.models_dir, settings.PCA_DETECTION_PATH)
            self.pca_detection = joblib.load(pca_detection_path)
            print(f"✓ Detection PCA loaded: {self.pca_detection.n_components} components")
            
            # Load recognition models (512 components)
            pca_rec_path = os.path.join(self.models_dir, settings.PCA_RECOGNITION_PATH)
            scaler_rec_path = os.path.join(self.models_dir, settings.SCALER_RECOGNITION_PATH)
            db_path = os.path.join(self.models_dir, settings.RECOGNITION_DB_PATH)
            paths_path = os.path.join(self.models_dir, settings.USED_PATHS_PATH)
            roi_paths_path = os.path.join(self.models_dir, settings.ROI_PATHS_PATH)  # ADD THIS
            
            # Check if files exist
            if not os.path.exists(pca_rec_path):
                raise FileNotFoundError(f"PCA recognition file not found: {pca_rec_path}")
            if not os.path.exists(scaler_rec_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_rec_path}")
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Recognition DB file not found: {db_path}")
            if not os.path.exists(paths_path):
                raise FileNotFoundError(f"Used paths file not found: {paths_path}")
            if not os.path.exists(roi_paths_path):
                raise FileNotFoundError(f"ROI paths file not found: {roi_paths_path}")  # ADD THIS
            
            # Load models
            self.pca_recognition = joblib.load(pca_rec_path)
            self.scaler_recognition = joblib.load(scaler_rec_path)
            self.recognition_db = joblib.load(db_path)
            self.used_paths = joblib.load(paths_path)
            self.roi_paths = joblib.load(roi_paths_path)  # ADD THIS LINE
            
            print(f"✓ Recognition PCA loaded: {self.pca_recognition.n_components} components")
            print(f"✓ Recognition database loaded: {self.recognition_db.shape[0]} embeddings")
            print(f"✓ ROI paths loaded: {len(self.roi_paths)} entries")  # ADD THIS
            
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def reload_recognition_models(self):
        """Reload recognition models after database update"""
        try:
            pca_rec_path = os.path.join(self.models_dir, settings.PCA_RECOGNITION_PATH)
            scaler_rec_path = os.path.join(self.models_dir, settings.SCALER_RECOGNITION_PATH)
            db_path = os.path.join(self.models_dir, settings.RECOGNITION_DB_PATH)
            paths_path = os.path.join(self.models_dir, settings.USED_PATHS_PATH)
            roi_paths_path = os.path.join(self.models_dir, settings.ROI_PATHS_PATH)  # ADD THIS
            
            self.pca_recognition = joblib.load(pca_rec_path)
            self.scaler_recognition = joblib.load(scaler_rec_path)
            self.recognition_db = joblib.load(db_path)
            self.used_paths = joblib.load(paths_path)
            self.roi_paths = joblib.load(roi_paths_path)  # ADD THIS
            
            print(f"✓ Models reloaded: {self.recognition_db.shape[0]} embeddings")
            return True
        except Exception as e:
            print(f"✗ Error reloading models: {e}")
            return False
    
    def extract_detection_roi(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], Tuple, np.ndarray]:
        """
        Extract ROI for face detection (8-PCA)
        Returns: (roi_vector, bbox, roi_image)
        """
        # Enhance entire image first
        enhanced_img = image_enhancer.enhance_image(img)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            enhanced_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            # Get largest face
            x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
            roi = enhanced_img[y:y+h, x:x+w]
        else:
            # Fallback to center crop
            h, w = enhanced_img.shape
            center_size = min(h, w) // 2
            y_center, x_center = h // 2, w // 2
            y_start = max(0, y_center - center_size // 2)
            x_start = max(0, x_center - center_size // 2)
            y_end = min(h, y_start + center_size)
            x_end = min(w, x_start + center_size)
            roi = enhanced_img[y_start:y_end, x_start:x_end]
            x, y, w, h = x_start, y_start, center_size, center_size
        
        # Resize ROI
        roi_resized = cv2.resize(roi, self.roi_size, interpolation=cv2.INTER_AREA)
        
        # Apply enhancement to ROI
        roi_enhanced = image_enhancer.enhance_roi(roi_resized)
        
        # Normalize for detection
        roi_normalized = roi_enhanced.flatten().astype(np.float32) / 255.0
        
        return roi_normalized, (x, y, w, h), roi_enhanced
    
    def extract_recognition_roi(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], Tuple]:
        """
        Extract ROI for recognition (512-PCA)
        Returns: (roi_image, bbox)
        """
        # Enhance entire image
        enhanced_img = image_enhancer.enhance_image(img)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            enhanced_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
            roi = enhanced_img[y:y+h, x:x+w]
        else:
            # Fallback to center crop
            h, w = enhanced_img.shape
            center_size = min(h, w) // 2
            y_center, x_center = h // 2, w // 2
            y_start = max(0, y_center - center_size // 2)
            x_start = max(0, x_center - center_size // 2)
            y_end = min(h, y_start + center_size)
            x_end = min(w, x_start + center_size)
            roi = enhanced_img[y_start:y_end, x_start:x_end]
            x, y, w, h = x_start, y_start, center_size, center_size
        
        # Resize ROI
        roi_resized = cv2.resize(roi, self.roi_size, interpolation=cv2.INTER_AREA)
        
        # Apply enhancement to ROI
        roi_enhanced = image_enhancer.enhance_roi(roi_resized)
        
        return roi_enhanced, (x, y, w, h)
    
    def detect_face_vqc(self, roi_vector: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if ROI contains a face using VQC quantum model
        Returns: (is_face, confidence)
        """
        try:
            # Transform with 8-PCA
            features_pca = self.pca_detection.transform([roi_vector])
            
            # Normalize features
            features_norm = (features_pca - features_pca.min()) / (features_pca.max() - features_pca.min() + 1e-10)
            
            # Scale to quantum angles
            features_scaled = features_norm * 2 * np.pi
            
            # Convert to tensor
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
            
            # Run through VQC model
            self.vqc_model.eval()
            with torch.no_grad():
                outputs = self.vqc_model(features_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred = int(torch.argmax(outputs, dim=1).item())
                confidence = float(probs[0][pred].item())
            
            is_face = (pred == 1)  # Assuming 1 = face, 0 = non-face
            
            return is_face, confidence
            
        except Exception as e:
            print(f"Error in VQC detection: {e}")
            return False, 0.0
    
    def match_face_cosine(self, probe_vec: np.ndarray) -> Tuple[int, float, Optional[str]]:
        """
        Match face against recognition database
        Returns: (match_index, similarity, user_path)
        """
        try:
            # Scale and transform with 512-PCA
            probe_scaled = self.scaler_recognition.transform(probe_vec.reshape(1, -1))
            probe_pca = self.pca_recognition.transform(probe_scaled)
            
            # Compute cosine similarity
            similarities = cosine_similarity(probe_pca, self.recognition_db)[0]
            
            # Get best match
            best_idx = int(np.argmax(similarities))
            best_sim = similarities[best_idx]
            
            # Check threshold
            if best_sim < self.cosine_threshold:
                return -1, best_sim, None
            
            # Get matched user path
            matched_path = self.used_paths[best_idx]
            
            return best_idx, best_sim, matched_path
            
        except Exception as e:
            print(f"Error in face matching: {e}")
            return -1, 0.0, None
    
    def image_to_vector(self, img_gray: np.ndarray) -> np.ndarray:
        """Convert grayscale image to feature vector"""
        return img_gray.astype(np.float32).ravel() / 255.0
    
    def process_image_for_authentication(self, image: np.ndarray) -> Dict:
        """
        Complete pipeline: Detection → Recognition → Matching
        
        Returns:
            dict: {
                'success': bool,
                'is_face': bool,
                'match_found': bool,
                'user_id': Optional[int],
                'similarity': float,
                'message': str
            }
        """
        result = {
            'success': False,
            'is_face': False,
            'match_found': False,
            'user_id': None,
            'similarity': 0.0,
            'message': ''
        }
        
        try:
            # Step 1: Extract ROI for detection
            detection_roi_vec, detection_bbox, detection_roi_img = self.extract_detection_roi(image)
            
            if detection_roi_vec is None:
                result['message'] = "Failed to extract ROI"
                return result
            
            # Step 2: VQC Face Detection
            is_face, face_confidence = self.detect_face_vqc(detection_roi_vec)
            result['is_face'] = is_face
            
            if not is_face:
                result['message'] = "No face detected"
                return result
            
            # Step 3: Extract ROI for recognition
            recognition_roi_img, recognition_bbox = self.extract_recognition_roi(image)
            
            if recognition_roi_img is None:
                result['message'] = "Face detected but recognition ROI extraction failed"
                return result
            
            # Step 4: Convert to vector
            probe_vec = self.image_to_vector(recognition_roi_img)
            
            # Step 5: Match against database
            match_idx, similarity, matched_path = self.match_face_cosine(probe_vec)
            
            result['similarity'] = similarity
            
            if match_idx == -1:
                result['message'] = f"Face not recognized (similarity: {similarity:.2f})"
                return result
            
            # Extract user_id from path (assuming format: user__...)
            if matched_path:
                try:
                    # Parse user_id from path
                    # Format: "path/to/user_123_img_1.jpg"
                    filename = os.path.basename(matched_path)
                    user_id_str = filename.split('_')[1]
                    user_id = int(user_id_str)
                    result['user_id'] = user_id
                except:
                    result['user_id'] = match_idx
            
            result['success'] = True
            result['match_found'] = True
            result['message'] = f"Match found! Similarity: {similarity:.2f}"
            
            return result
            
        except Exception as e:
            result['message'] = f"Error processing image: {str(e)}"
            return result


# Singleton instance
vqc_service = VQCFaceService()