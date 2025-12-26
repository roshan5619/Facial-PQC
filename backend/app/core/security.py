"""
Security module with Post-Quantum Cryptography support
Uses Dilithium signatures for JWT tokens instead of HMAC-SHA256

Migration from python-jose:
- create_access_token now uses PQC JWT service with Dilithium signatures
- decode_access_token verifies Dilithium signatures

Backward compatibility:
- Classical JWT creation/verification available via _classical_ prefixed functions
- Automatic fallback to classical if PQC fails
"""

from datetime import datetime, timedelta
from typing import Optional, Dict
from jose import JWTError, jwt # pyright: ignore[reportMissingModuleSource]
from app.core.config import settings
import uuid

# PQC imports
try:
    from app.services.pqc_jwt_service import pqc_jwt_service
    PQC_AVAILABLE = True
except ImportError:
    PQC_AVAILABLE = False
    print("WARNING: PQC JWT service not available, using classical JWT")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create access token using Post-Quantum Cryptography (Dilithium signatures)
    
    Falls back to classical JWT if PQC is not available or user_id not provided.
    
    Args:
        data: Token payload (should include 'user_id' for PQC)
        expires_delta: Token expiration time
        
    Returns:
        str: JWT token string
    """
    # Try PQC first if available and user_id provided
    if PQC_AVAILABLE:
        user_id = data.get("user_id") or data.get("sub")
        
        if user_id is not None:
            try:
                return pqc_jwt_service.create_token(
                    data,
                    int(user_id),
                    expires_delta
                )
            except Exception as e:
                print(f"PQC JWT creation failed, falling back to classical: {e}")
        else:
            # System token (no user_id)
            try:
                return pqc_jwt_service.create_system_token(data, expires_delta)
            except Exception as e:
                print(f"PQC system token failed, falling back to classical: {e}")
    
    # Fallback to classical JWT
    return _classical_create_access_token(data, expires_delta)


def decode_access_token(token: str) -> Optional[Dict]:
    """
    Decode and verify access token
    
    Automatically detects PQC vs classical JWT and verifies accordingly.
    
    Args:
        token: JWT token string
        
    Returns:
        Dict with payload if valid, None if invalid
    """
    # Try PQC verification first
    if PQC_AVAILABLE:
        try:
            # Check if it's a PQC token by trying to decode header
            payload = pqc_jwt_service.verify_token(token)
            if payload is not None:
                return payload
        except Exception:
            pass  # Try classical
    
    # Try classical JWT
    return _classical_decode_access_token(token)


# =============================================================================
# Classical JWT Functions (for backward compatibility)
# =============================================================================

def _classical_create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create classical JWT token using HMAC-SHA256"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "jti": str(uuid.uuid4())
    })
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def _classical_decode_access_token(token: str) -> Optional[Dict]:
    """Decode classical JWT token"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None


# =============================================================================
# PQC-Specific Functions
# =============================================================================

def create_pqc_token(data: dict, user_id: int, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create PQC-protected JWT token with Dilithium signature
    
    This function explicitly uses PQC (no fallback).
    
    Args:
        data: Token payload
        user_id: User ID for key lookup
        expires_delta: Token expiration time
        
    Returns:
        str: PQC JWT token
        
    Raises:
        RuntimeError: If PQC is not available
    """
    if not PQC_AVAILABLE:
        raise RuntimeError("PQC JWT service not available")
    
    return pqc_jwt_service.create_token(data, user_id, expires_delta)


def verify_pqc_token(token: str) -> Optional[Dict]:
    """
    Verify PQC JWT token
    
    Args:
        token: PQC JWT token string
        
    Returns:
        Dict with payload if valid, None if invalid
    """
    if not PQC_AVAILABLE:
        return None
    
    return pqc_jwt_service.verify_token(token)


def create_encrypted_token(data: dict, user_id: int, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create encrypted PQC JWT token
    
    For sensitive payloads that should be encrypted, not just signed.
    
    Args:
        data: Sensitive payload data
        user_id: User ID for encryption keys
        expires_delta: Token expiration time
        
    Returns:
        str: Encrypted PQC JWT token
    """
    if not PQC_AVAILABLE:
        raise RuntimeError("PQC JWT service not available for encrypted tokens")
    
    return pqc_jwt_service.create_encrypted_token(data, user_id, expires_delta)


def is_pqc_available() -> bool:
    """Check if PQC is available"""
    return PQC_AVAILABLE


def get_token_info(token: str) -> Optional[Dict]:
    """
    Get information about a token without full verification
    
    Useful for debugging and logging.
    
    Args:
        token: JWT token
        
    Returns:
        Dict with token info (header, payload preview)
    """
    if PQC_AVAILABLE:
        info = pqc_jwt_service.decode_token_without_verification(token)
        if info:
            return info
    
    # Try to decode classical JWT header/payload
    try:
        parts = token.split('.')
        if len(parts) == 3:
            import base64
            header = base64.urlsafe_b64decode(parts[0] + '==')
            return {"header": header.decode(), "type": "classical_jwt"}
    except Exception:
        pass
    
    return None