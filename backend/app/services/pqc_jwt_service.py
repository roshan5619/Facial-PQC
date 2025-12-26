"""
PQC-Protected JWT Service
Replaces python-jose HMAC-SHA256 JWT tokens with Dilithium digital signatures

This service provides quantum-resistant JWT tokens using:
- Dilithium3 digital signatures (NIST Level 3 security)
- Optional hybrid encryption for sensitive payloads
- Token revocation support via database tracking

Migration from python-jose:
OLD: jwt.encode(payload, SECRET_KEY, algorithm="HS256")
NEW: pqc_jwt_service.create_token(payload, user_id, expires_delta)
"""

import json
import base64
import struct
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple
import uuid

from app.services.pqc_service import pqc_service
from app.services.pqc_key_manager import pqc_key_manager, PQCKeySet
from app.core.config import settings


class PQCJWTConfig:
    """Configuration for PQC JWT Service"""
    # Token structure
    VERSION = 1
    ALGORITHM = "DILITHIUM3"
    
    # Token timing
    DEFAULT_EXPIRE_MINUTES = 5
    REFRESH_EXPIRE_DAYS = 7
    
    # Token types
    TOKEN_TYPE_ACCESS = "access"
    TOKEN_TYPE_REFRESH = "refresh"
    
    # Header fields
    HEADER_ALG = "alg"
    HEADER_TYP = "typ"
    HEADER_VERSION = "ver"
    HEADER_KID = "kid"  # Key ID for key rotation support


class PQCJWTService:
    """
    Post-Quantum Cryptography JWT Service
    
    Replaces traditional JWT with Dilithium-signed tokens that are
    resistant to quantum computer attacks.
    
    Token Structure:
    - Header (base64url): Algorithm, type, version, key ID
    - Payload (base64url): User data, timestamps, claims
    - Signature (base64url): Dilithium3 digital signature
    
    Format: header.payload.signature (similar to standard JWT)
    """
    
    def __init__(self):
        """Initialize PQC JWT Service"""
        self.config = PQCJWTConfig()
        
        # System keys for server-signed tokens
        self._system_keys: Optional[Dict[str, bytes]] = None
        
        print("✓ PQC JWT Service initialized")
    
    def _ensure_system_keys(self):
        """Ensure system keys are loaded or generated"""
        if self._system_keys is None:
            self._system_keys = pqc_key_manager.load_system_keys()
            
            if self._system_keys is None:
                print("Generating system PQC keys for JWT signing...")
                self._system_keys = pqc_key_manager.generate_system_keys()
    
    def _base64url_encode(self, data: bytes) -> str:
        """Encode bytes to base64url string (no padding)"""
        return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')
    
    def _base64url_decode(self, data: str) -> bytes:
        """Decode base64url string to bytes"""
        # Add padding if needed
        padding = 4 - (len(data) % 4)
        if padding != 4:
            data += '=' * padding
        return base64.urlsafe_b64decode(data.encode('utf-8'))
    
    def _create_header(self, user_id: int, key_version: int = 1) -> Dict[str, Any]:
        """Create JWT header"""
        return {
            self.config.HEADER_ALG: self.config.ALGORITHM,
            self.config.HEADER_TYP: "JWT",
            self.config.HEADER_VERSION: self.config.VERSION,
            self.config.HEADER_KID: f"user_{user_id}_v{key_version}"
        }
    
    def _create_system_header(self) -> Dict[str, Any]:
        """Create JWT header for system-signed tokens"""
        return {
            self.config.HEADER_ALG: self.config.ALGORITHM,
            self.config.HEADER_TYP: "JWT",
            self.config.HEADER_VERSION: self.config.VERSION,
            self.config.HEADER_KID: "system_v1"
        }
    
    # ==========================================================================
    # User-Signed Tokens (user's Dilithium key)
    # ==========================================================================
    
    def create_token(
        self,
        payload: Dict[str, Any],
        user_id: int,
        expires_delta: Optional[timedelta] = None,
        token_type: str = None
    ) -> str:
        """
        Create PQC-protected JWT token signed with user's Dilithium key
        
        Format: header.payload.dilithium_signature
        
        Args:
            payload: Token payload data
            user_id: User ID (for key lookup)
            expires_delta: Token expiration time
            token_type: "access" or "refresh"
            
        Returns:
            str: JWT token string
        """
        if token_type is None:
            token_type = self.config.TOKEN_TYPE_ACCESS
        
        # Load user's keys
        user_keys = pqc_key_manager.load_user_keys(user_id)
        if user_keys is None:
            # Generate keys if not exist
            user_keys = pqc_key_manager.generate_user_keys(user_id)
            pqc_key_manager.save_user_keys(user_id, user_keys)
        
        # Create header
        header = self._create_header(user_id, user_keys.key_version)
        
        # Create payload with standard claims
        token_payload = payload.copy()
        
        now = datetime.utcnow()
        if expires_delta:
            expire = now + expires_delta
        else:
            expire = now + timedelta(minutes=self.config.DEFAULT_EXPIRE_MINUTES)
        
        token_payload.update({
            "sub": str(user_id),
            "iat": int(now.timestamp()),
            "exp": int(expire.timestamp()),
            "jti": str(uuid.uuid4()),
            "type": token_type
        })
        
        # Encode header and payload
        header_b64 = self._base64url_encode(json.dumps(header).encode('utf-8'))
        payload_b64 = self._base64url_encode(json.dumps(token_payload).encode('utf-8'))
        
        # Create signature message
        message = f"{header_b64}.{payload_b64}".encode('utf-8')
        
        # Sign with Dilithium
        signature = pqc_service.sign_dilithium(message, user_keys.dilithium_private)
        signature_b64 = self._base64url_encode(signature)
        
        # Combine into JWT
        token = f"{header_b64}.{payload_b64}.{signature_b64}"
        
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify PQC JWT token and return payload if valid
        
        Args:
            token: JWT token string
            
        Returns:
            Dict with payload if valid, None if invalid
        """
        try:
            # Split token
            parts = token.split('.')
            if len(parts) != 3:
                print("Invalid token format: expected 3 parts")
                return None
            
            header_b64, payload_b64, signature_b64 = parts
            
            # Decode header
            header = json.loads(self._base64url_decode(header_b64))
            
            # Check algorithm
            if header.get(self.config.HEADER_ALG) != self.config.ALGORITHM:
                print(f"Invalid algorithm: {header.get(self.config.HEADER_ALG)}")
                return None
            
            # Decode payload
            payload = json.loads(self._base64url_decode(payload_b64))
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.utcnow().timestamp() > exp:
                print("Token expired")
                return None
            
            # Get user ID from key ID
            key_id = header.get(self.config.HEADER_KID, "")
            
            if key_id.startswith("system"):
                # System-signed token
                public_key = self._get_system_verification_key()
            else:
                # User-signed token
                user_id = int(payload.get("sub", 0))
                user_keys = pqc_key_manager.load_user_keys(user_id)
                
                if user_keys is None:
                    print(f"No keys found for user {user_id}")
                    return None
                
                public_key = user_keys.dilithium_public
            
            # Decode signature
            signature = self._base64url_decode(signature_b64)
            
            # Verify signature
            message = f"{header_b64}.{payload_b64}".encode('utf-8')
            
            if not pqc_service.verify_dilithium(message, signature, public_key):
                print("Signature verification failed")
                return None
            
            return payload
            
        except Exception as e:
            print(f"Token verification error: {e}")
            return None
    
    # ==========================================================================
    # System-Signed Tokens (server's Dilithium key)
    # ==========================================================================
    
    def _get_system_verification_key(self) -> bytes:
        """Get system Dilithium public key for verification"""
        self._ensure_system_keys()
        return self._system_keys["dilithium_public"]
    
    def create_system_token(
        self,
        payload: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create system-signed JWT token
        
        Used for:
        - Admin operations
        - Inter-service communication
        - System-level authentication
        
        Args:
            payload: Token payload
            expires_delta: Expiration time
            
        Returns:
            str: JWT token
        """
        self._ensure_system_keys()
        
        # Create header
        header = self._create_system_header()
        
        # Create payload
        token_payload = payload.copy()
        
        now = datetime.utcnow()
        if expires_delta:
            expire = now + expires_delta
        else:
            expire = now + timedelta(minutes=self.config.DEFAULT_EXPIRE_MINUTES)
        
        token_payload.update({
            "iat": int(now.timestamp()),
            "exp": int(expire.timestamp()),
            "jti": str(uuid.uuid4()),
            "iss": "pqc_face_auth_system"
        })
        
        # Encode
        header_b64 = self._base64url_encode(json.dumps(header).encode('utf-8'))
        payload_b64 = self._base64url_encode(json.dumps(token_payload).encode('utf-8'))
        
        # Sign
        message = f"{header_b64}.{payload_b64}".encode('utf-8')
        signature = pqc_service.sign_dilithium(message, self._system_keys["dilithium_private"])
        signature_b64 = self._base64url_encode(signature)
        
        return f"{header_b64}.{payload_b64}.{signature_b64}"
    
    # ==========================================================================
    # Encrypted Tokens (for sensitive payloads)
    # ==========================================================================
    
    def create_encrypted_token(
        self,
        payload: Dict[str, Any],
        user_id: int,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create encrypted JWT token with hybrid PQC encryption
        
        Used when payload contains sensitive data that should be hidden
        even if token is intercepted.
        
        Process:
        1. Create standard JWT payload
        2. Encrypt payload with user's NTRU + Kyber keys
        3. Sign with Dilithium
        
        Args:
            payload: Sensitive payload data
            user_id: User ID
            expires_delta: Expiration time
            
        Returns:
            str: Encrypted JWT token
        """
        user_keys = pqc_key_manager.load_user_keys(user_id)
        if user_keys is None:
            user_keys = pqc_key_manager.generate_user_keys(user_id)
            pqc_key_manager.save_user_keys(user_id, user_keys)
        
        # Add standard claims
        token_payload = payload.copy()
        
        now = datetime.utcnow()
        if expires_delta:
            expire = now + expires_delta
        else:
            expire = now + timedelta(minutes=self.config.DEFAULT_EXPIRE_MINUTES)
        
        token_payload.update({
            "sub": str(user_id),
            "iat": int(now.timestamp()),
            "exp": int(expire.timestamp()),
            "jti": str(uuid.uuid4())
        })
        
        # Serialize and encrypt payload
        payload_bytes = json.dumps(token_payload).encode('utf-8')
        encrypted_payload = pqc_service.hybrid_encrypt(
            payload_bytes,
            user_keys.ntru_public,
            user_keys.kyber_public
        )
        
        # Create header (marks as encrypted)
        header = {
            self.config.HEADER_ALG: self.config.ALGORITHM,
            self.config.HEADER_TYP: "JWT+E",  # Encrypted JWT
            self.config.HEADER_VERSION: self.config.VERSION,
            self.config.HEADER_KID: f"user_{user_id}_v{user_keys.key_version}"
        }
        
        # Encode
        header_b64 = self._base64url_encode(json.dumps(header).encode('utf-8'))
        encrypted_payload_b64 = self._base64url_encode(encrypted_payload)
        
        # Sign
        message = f"{header_b64}.{encrypted_payload_b64}".encode('utf-8')
        signature = pqc_service.sign_dilithium(message, user_keys.dilithium_private)
        signature_b64 = self._base64url_encode(signature)
        
        return f"{header_b64}.{encrypted_payload_b64}.{signature_b64}"
    
    def verify_encrypted_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decrypt encrypted JWT token
        
        Args:
            token: Encrypted JWT token
            
        Returns:
            Dict with decrypted payload if valid, None if invalid
        """
        try:
            parts = token.split('.')
            if len(parts) != 3:
                return None
            
            header_b64, encrypted_payload_b64, signature_b64 = parts
            
            # Decode header
            header = json.loads(self._base64url_decode(header_b64))
            
            # Check if encrypted
            if header.get(self.config.HEADER_TYP) != "JWT+E":
                print("Token is not encrypted, use verify_token instead")
                return None
            
            # Get user ID from key ID
            key_id = header.get(self.config.HEADER_KID, "")
            if not key_id.startswith("user_"):
                return None
            
            user_id = int(key_id.split("_")[1])
            user_keys = pqc_key_manager.load_user_keys(user_id)
            
            if user_keys is None:
                return None
            
            # Verify signature first
            signature = self._base64url_decode(signature_b64)
            message = f"{header_b64}.{encrypted_payload_b64}".encode('utf-8')
            
            if not pqc_service.verify_dilithium(message, signature, user_keys.dilithium_public):
                print("Signature verification failed")
                return None
            
            # Decrypt payload
            encrypted_payload = self._base64url_decode(encrypted_payload_b64)
            payload_bytes = pqc_service.hybrid_decrypt(
                encrypted_payload,
                user_keys.ntru_private,
                user_keys.kyber_private
            )
            
            payload = json.loads(payload_bytes.decode('utf-8'))
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.utcnow().timestamp() > exp:
                print("Token expired")
                return None
            
            return payload
            
        except Exception as e:
            print(f"Encrypted token verification error: {e}")
            return None
    
    # ==========================================================================
    # Token Utilities
    # ==========================================================================
    
    def decode_token_without_verification(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Decode token without verification (for debugging/logging only)
        
        WARNING: Do not use for authentication decisions!
        
        Args:
            token: JWT token
            
        Returns:
            Dict with header and payload
        """
        try:
            parts = token.split('.')
            if len(parts) != 3:
                return None
            
            header = json.loads(self._base64url_decode(parts[0]))
            
            # Check if encrypted
            if header.get(self.config.HEADER_TYP) == "JWT+E":
                return {
                    "header": header,
                    "payload": "[ENCRYPTED]",
                    "signature_length": len(self._base64url_decode(parts[2]))
                }
            
            payload = json.loads(self._base64url_decode(parts[1]))
            
            return {
                "header": header,
                "payload": payload,
                "signature_length": len(self._base64url_decode(parts[2]))
            }
            
        except Exception as e:
            return None
    
    def get_token_user_id(self, token: str) -> Optional[int]:
        """
        Extract user ID from token without full verification
        
        Args:
            token: JWT token
            
        Returns:
            User ID or None
        """
        decoded = self.decode_token_without_verification(token)
        if decoded and isinstance(decoded.get("payload"), dict):
            sub = decoded["payload"].get("sub")
            if sub:
                return int(sub)
        return None
    
    def create_refresh_token(self, user_id: int, payload: Dict[str, Any] = None) -> str:
        """
        Create a long-lived refresh token
        
        Args:
            user_id: User ID
            payload: Additional payload data
            
        Returns:
            str: Refresh token
        """
        if payload is None:
            payload = {}
        
        payload["refresh"] = True
        
        return self.create_token(
            payload=payload,
            user_id=user_id,
            expires_delta=timedelta(days=self.config.REFRESH_EXPIRE_DAYS),
            token_type=self.config.TOKEN_TYPE_REFRESH
        )
    
    def is_refresh_token(self, payload: Dict[str, Any]) -> bool:
        """Check if payload is from a refresh token"""
        return payload.get("type") == self.config.TOKEN_TYPE_REFRESH


# ==========================================================================
# Singleton Instance
# ==========================================================================

pqc_jwt_service = PQCJWTService()


# ==========================================================================
# Drop-in Replacement Functions for python-jose
# ==========================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Drop-in replacement for app.core.security.create_access_token
    
    Creates PQC-protected JWT token using Dilithium signatures.
    
    Args:
        data: Token payload (must include 'user_id' or 'sub')
        expires_delta: Token expiration time
        
    Returns:
        str: PQC JWT token
    """
    user_id = data.get("user_id") or data.get("sub")
    
    if user_id is None:
        # Use system token if no user ID
        return pqc_jwt_service.create_system_token(data, expires_delta)
    
    return pqc_jwt_service.create_token(data, int(user_id), expires_delta)


def decode_access_token(token: str) -> Optional[Dict]:
    """
    Drop-in replacement for app.core.security.decode_access_token
    
    Verifies and decodes PQC JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Dict with payload if valid, None otherwise
    """
    return pqc_jwt_service.verify_token(token)
