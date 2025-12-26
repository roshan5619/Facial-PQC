"""
VQC Service with PQC Integration
Extends VQCFaceService to support Post-Quantum Cryptography for embeddings

This module provides:
- PQC-encrypted face embedding storage and retrieval
- Secure face matching with decrypted embeddings
- Migration support for existing embeddings
"""

import os
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
import joblib
from pathlib import Path

from app.services.vqc_service import vqc_service, VQCFaceService
from app.services.pqc_service import pqc_service
from app.services.pqc_key_manager import pqc_key_manager
from app.services.hybrid_crypto_service import hybrid_crypto_service
from app.core.config import settings


class VQCFaceServicePQC:
    """
    VQC Face Service with Post-Quantum Cryptography
    
    Wraps the existing VQCFaceService to add PQC encryption for:
    - Face embedding storage (encrypted with hybrid NTRU + Kyber)
    - Face embedding retrieval (decrypted for matching)
    - Secure face matching workflow
    
    This class can be used as a drop-in replacement for VQCFaceService
    when PQC protection is required.
    """
    
    def __init__(self, base_service: VQCFaceService = None):
        """
        Initialize PQC-enabled VQC Face Service
        
        Args:
            base_service: Existing VQCFaceService instance (uses global if not provided)
        """
        self.base_service = base_service or vqc_service
        self.models_dir = Path(settings.MODELS_DIR)
        
        # PQC database files
        self.pqc_db_path = self.models_dir / "pqc_recognition_db_cosine_roi.pkl"
        self.pqc_metadata_path = self.models_dir / "pqc_metadata.pkl"
        
        # Cached decrypted embeddings for matching (cleared after use)
        self._decrypted_cache: Dict[int, np.ndarray] = {}
        
        # Check if PQC database exists
        self.pqc_enabled = self.pqc_db_path.exists()
        
        if self.pqc_enabled:
            print("✓ VQC PQC Service initialized with encrypted database")
        else:
            print("⚠ VQC PQC Service initialized (no PQC database found, using unencrypted)")
    
    def is_pqc_enabled(self) -> bool:
        """Check if PQC-encrypted database is available"""
        return self.pqc_enabled
    
    def reload_recognition_models(self) -> bool:
        """Reload recognition models and update PQC status"""
        success = self.base_service.reload_recognition_models()
        self.pqc_enabled = self.pqc_db_path.exists()
        return success
    
    # =========================================================================
    # Embedding Encryption/Decryption
    # =========================================================================
    
    def encrypt_embedding(
        self,
        embedding: np.ndarray,
        user_id: int,
        sign: bool = True
    ) -> bytes:
        """
        Encrypt face embedding for storage
        
        Uses hybrid NTRU + Kyber encryption with optional Dilithium signature.
        
        Args:
            embedding: 512-dim face embedding vector
            user_id: User ID for key lookup
            sign: Whether to add Dilithium signature
            
        Returns:
            bytes: Encrypted embedding
        """
        return hybrid_crypto_service.encrypt_embedding(embedding, user_id, sign=sign)
    
    def decrypt_embedding(
        self,
        encrypted_data: bytes,
        user_id: int,
        verify_signature: bool = True
    ) -> Optional[np.ndarray]:
        """
        Decrypt face embedding for matching
        
        Args:
            encrypted_data: Encrypted embedding bytes
            user_id: User ID for key lookup
            verify_signature: Whether to verify Dilithium signature
            
        Returns:
            np.ndarray: Decrypted embedding, or None if decryption fails
        """
        try:
            return hybrid_crypto_service.decrypt_embedding(
                encrypted_data, user_id, verify_signature
            )
        except Exception as e:
            print(f"Failed to decrypt embedding for user {user_id}: {e}")
            return None
    
    # =========================================================================
    # Face Matching with PQC
    # =========================================================================
    
    def process_image_for_authentication_pqc(
        self,
        image: np.ndarray,
        decrypt_for_matching: bool = True
    ) -> Dict[str, Any]:
        """
        Complete authentication pipeline with PQC support
        
        Process:
        1. VQC face detection
        2. Extract face embedding
        3. If PQC enabled: Decrypt stored embeddings for matching
        4. Match against database
        5. Return result
        
        Args:
            image: Input image (grayscale or BGR)
            decrypt_for_matching: Whether to decrypt PQC embeddings
            
        Returns:
            dict with authentication result
        """
        result = {
            'success': False,
            'is_face': False,
            'match_found': False,
            'user_id': None,
            'similarity': 0.0,
            'message': '',
            'pqc_verified': False
        }
        
        try:
            # Step 1: Extract ROI for detection
            detection_roi_vec, detection_bbox, detection_roi_img = \
                self.base_service.extract_detection_roi(image)
            
            if detection_roi_vec is None:
                result['message'] = "Failed to extract ROI"
                return result
            
            # Step 2: VQC Face Detection
            is_face, face_confidence = self.base_service.detect_face_vqc(detection_roi_vec)
            result['is_face'] = is_face
            
            if not is_face:
                result['message'] = "No face detected"
                return result
            
            # Step 3: Extract ROI for recognition
            recognition_roi_img, recognition_bbox = \
                self.base_service.extract_recognition_roi(image)
            
            if recognition_roi_img is None:
                result['message'] = "Face detected but recognition ROI extraction failed"
                return result
            
            # Step 4: Convert to vector
            probe_vec = self.base_service.image_to_vector(recognition_roi_img)
            
            # Step 5: Match against database
            if self.pqc_enabled and decrypt_for_matching:
                match_idx, similarity, matched_path, user_id = \
                    self._match_face_pqc(probe_vec)
            else:
                match_idx, similarity, matched_path = \
                    self.base_service.match_face_cosine(probe_vec)
                user_id = self._extract_user_id_from_path(matched_path)
            
            result['similarity'] = similarity
            
            if match_idx == -1:
                result['message'] = f"Face not recognized (similarity: {similarity:.2f})"
                return result
            
            result['success'] = True
            result['match_found'] = True
            result['user_id'] = user_id
            result['pqc_verified'] = self.pqc_enabled
            result['message'] = f"Match found! Similarity: {similarity:.2f}"
            
            return result
            
        except Exception as e:
            result['message'] = f"Error processing image: {str(e)}"
            import traceback
            traceback.print_exc()
            return result
    
    def _match_face_pqc(
        self,
        probe_vec: np.ndarray
    ) -> Tuple[int, float, Optional[str], Optional[int]]:
        """
        Match face against PQC-encrypted database
        
        Decrypts embeddings on-the-fly for matching, then clears from memory.
        
        Args:
            probe_vec: Probe face vector
            
        Returns:
            Tuple of (match_index, similarity, path, user_id)
        """
        try:
            # Load PQC encrypted database
            encrypted_embeddings = joblib.load(self.pqc_db_path)
            paths = self.base_service.used_paths
            
            # Scale and transform probe vector
            probe_scaled = self.base_service.scaler_recognition.transform(
                probe_vec.reshape(1, -1)
            )
            probe_pca = self.base_service.pca_recognition.transform(probe_scaled)
            
            best_idx = -1
            best_sim = 0.0
            best_path = None
            best_user_id = None
            
            # Iterate through encrypted embeddings
            for idx, encrypted_emb in enumerate(encrypted_embeddings):
                path = paths[idx] if idx < len(paths) else f"unknown_{idx}"
                
                # Extract user_id from path
                user_id = self._extract_user_id_from_path(path)
                
                if user_id is None:
                    # Skip if we can't determine user_id
                    continue
                
                try:
                    # Check if this is actually encrypted
                    if isinstance(encrypted_emb, bytes):
                        # Decrypt embedding
                        decrypted = self.decrypt_embedding(encrypted_emb, user_id)
                        
                        if decrypted is None:
                            continue
                        
                        # Transform decrypted embedding
                        emb_scaled = self.base_service.scaler_recognition.transform(
                            decrypted.reshape(1, -1)
                        )
                        emb_pca = self.base_service.pca_recognition.transform(emb_scaled)
                    else:
                        # Not encrypted (migration in progress), use directly
                        emb_pca = encrypted_emb.reshape(1, -1) \
                            if len(encrypted_emb.shape) == 1 else encrypted_emb
                    
                    # Compute similarity
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarity = cosine_similarity(probe_pca, emb_pca)[0][0]
                    
                    if similarity > best_sim:
                        best_sim = similarity
                        best_idx = idx
                        best_path = path
                        best_user_id = user_id
                    
                except Exception as e:
                    # Skip this embedding if decryption fails
                    continue
            
            # Check threshold
            if best_sim < self.base_service.cosine_threshold:
                return -1, best_sim, None, None
            
            return best_idx, best_sim, best_path, best_user_id
            
        except Exception as e:
            print(f"Error in PQC face matching: {e}")
            import traceback
            traceback.print_exc()
            return -1, 0.0, None, None
    
    def _extract_user_id_from_path(self, path: str) -> Optional[int]:
        """Extract user ID from image path"""
        if path is None:
            return None
        
        try:
            filename = os.path.basename(path)
            
            # Format: user_123_img_1.jpg
            if 'user_' in filename:
                parts = filename.split('_')
                for i, part in enumerate(parts):
                    if part == 'user' and i + 1 < len(parts):
                        return int(parts[i + 1])
            
            # Format: 123_face_1.jpg
            parts = filename.split('_')
            if parts[0].isdigit():
                return int(parts[0])
            
            return None
            
        except Exception:
            return None
    
    # =========================================================================
    # User Enrollment with PQC
    # =========================================================================
    
    def enroll_user_pqc(
        self,
        user_id: int,
        embeddings: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Enroll user with PQC-encrypted embeddings
        
        Args:
            user_id: User ID
            embeddings: List of face embeddings (usually 3 per user)
            
        Returns:
            dict with enrollment result
        """
        result = {
            'success': False,
            'user_id': user_id,
            'embeddings_encrypted': 0,
            'message': ''
        }
        
        try:
            # Ensure user has PQC keys
            if not pqc_key_manager.user_has_keys(user_id):
                print(f"Generating PQC keys for user {user_id}...")
                user_keys = pqc_key_manager.generate_user_keys(user_id)
                pqc_key_manager.save_user_keys(user_id, user_keys)
            
            encrypted_embeddings = []
            
            for embedding in embeddings:
                encrypted = self.encrypt_embedding(embedding, user_id, sign=True)
                encrypted_embeddings.append(encrypted)
            
            result['success'] = True
            result['embeddings_encrypted'] = len(encrypted_embeddings)
            result['encrypted_data'] = encrypted_embeddings
            result['message'] = f"Successfully encrypted {len(encrypted_embeddings)} embeddings"
            
            return result
            
        except Exception as e:
            result['message'] = f"Enrollment failed: {str(e)}"
            return result
    
    # =========================================================================
    # Delegate methods to base service
    # =========================================================================
    
    def extract_detection_roi(self, *args, **kwargs):
        """Delegate to base service"""
        return self.base_service.extract_detection_roi(*args, **kwargs)
    
    def extract_recognition_roi(self, *args, **kwargs):
        """Delegate to base service"""
        return self.base_service.extract_recognition_roi(*args, **kwargs)
    
    def detect_face_vqc(self, *args, **kwargs):
        """Delegate to base service"""
        return self.base_service.detect_face_vqc(*args, **kwargs)
    
    def match_face_cosine(self, *args, **kwargs):
        """Delegate to base service (unencrypted)"""
        return self.base_service.match_face_cosine(*args, **kwargs)
    
    def image_to_vector(self, *args, **kwargs):
        """Delegate to base service"""
        return self.base_service.image_to_vector(*args, **kwargs)
    
    def process_image_for_authentication(self, *args, **kwargs):
        """
        Process image for authentication
        
        Uses PQC if enabled, otherwise falls back to base service.
        """
        if self.pqc_enabled:
            return self.process_image_for_authentication_pqc(*args, **kwargs)
        else:
            return self.base_service.process_image_for_authentication(*args, **kwargs)


# Singleton instance
vqc_service_pqc = VQCFaceServicePQC()
