"""
PQC Key Management Service
Handles generation, storage, retrieval, and rotation of quantum-safe keys

This service provides:
- Secure key generation for all PQC algorithms (NTRU, Kyber, Dilithium, SPHINCS+)
- Encrypted storage of private keys using Fernet (symmetric encryption)
- Master password protected key hierarchy
- Key rotation capabilities
- User-specific key management
"""

import os
import json
import secrets
import hashlib
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

# Cryptographic imports
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

from app.services.pqc_service import pqc_service


@dataclass
class PQCKeySet:
    """Complete set of PQC keys for a user"""
    user_id: int
    
    # NTRU keys
    ntru_public: bytes
    ntru_private: bytes
    
    # Kyber keys
    kyber_public: bytes
    kyber_private: bytes
    
    # Dilithium keys (for JWT signing)
    dilithium_public: bytes
    dilithium_private: bytes
    
    # SPHINCS+ keys (for critical operations)
    sphincs_public: Optional[bytes] = None
    sphincs_private: Optional[bytes] = None
    
    # Metadata
    created_at: str = ""
    key_version: int = 1
    algorithm_versions: Dict[str, str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if self.algorithm_versions is None:
            self.algorithm_versions = {
                "ntru": "NTRU-HPS-2048-509",
                "kyber": "Kyber768",
                "dilithium": "Dilithium3",
                "sphincs": "SPHINCS+-SHA2-128f-simple"
            }


class PQCKeyManagerConfig:
    """Configuration for key management"""
    # Master key settings
    MASTER_KEY_FILE = "master.key.enc"
    MASTER_SALT_FILE = "master.salt"
    PBKDF2_ITERATIONS = 600000  # NIST recommended minimum
    
    # Key storage structure
    KEY_FILES = {
        "ntru_public": "ntru_public.key",
        "ntru_private": "ntru_private.key.enc",
        "kyber_public": "kyber_public.key",
        "kyber_private": "kyber_private.key.enc",
        "dilithium_public": "dilithium_public.key",
        "dilithium_private": "dilithium_private.key.enc",
        "sphincs_public": "sphincs_public.key",
        "sphincs_private": "sphincs_private.key.enc"
    }
    
    # Metadata file
    METADATA_FILE = "key_metadata.json"
    
    # Key rotation settings
    KEY_VALIDITY_DAYS = 365
    ROTATION_WARNING_DAYS = 30


class PQCKeyManager:
    """
    Post-Quantum Cryptography Key Management Service
    
    Manages the complete lifecycle of PQC keys including:
    - Generation of all key types (NTRU, Kyber, Dilithium, SPHINCS+)
    - Secure encrypted storage with Fernet
    - Master password protection using PBKDF2
    - Per-user key isolation
    - Key rotation and versioning
    """
    
    def __init__(self, keys_dir: str = "data/pqc_keys"):
        """
        Initialize PQC Key Manager
        
        Args:
            keys_dir: Base directory for key storage
        """
        self.keys_dir = Path(keys_dir)
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        self.config = PQCKeyManagerConfig()
        
        # Master key (loaded/generated on first use)
        self._master_fernet: Optional[Fernet] = None
        self._master_password: Optional[bytes] = None
        
        # Key cache for performance (encrypted keys still on disk)
        self._key_cache: Dict[int, PQCKeySet] = {}
        
        print(f"✓ PQC Key Manager initialized: {self.keys_dir}")
    
    # ==========================================================================
    # Master Password Management
    # ==========================================================================
    
    def _get_master_salt_path(self) -> Path:
        """Get path to master salt file"""
        return self.keys_dir / self.config.MASTER_SALT_FILE
    
    def _get_master_key_path(self) -> Path:
        """Get path to encrypted master key file"""
        return self.keys_dir / self.config.MASTER_KEY_FILE
    
    def _derive_key_from_password(self, password: bytes, salt: bytes) -> bytes:
        """
        Derive encryption key from password using PBKDF2
        
        Args:
            password: Master password
            salt: Random salt
            
        Returns:
            bytes: 32-byte key for Fernet
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.config.PBKDF2_ITERATIONS,
            backend=default_backend()
        )
        return base64.urlsafe_b64encode(kdf.derive(password))
    
    def initialize_master_password(self, password: str) -> bool:
        """
        Initialize or verify master password
        
        If no master password exists, creates one.
        If exists, verifies the provided password.
        
        Args:
            password: Master password string
            
        Returns:
            bool: True if initialization/verification successful
        """
        password_bytes = password.encode('utf-8')
        salt_path = self._get_master_salt_path()
        key_path = self._get_master_key_path()
        
        if salt_path.exists() and key_path.exists():
            # Verify existing password
            salt = salt_path.read_bytes()
            derived_key = self._derive_key_from_password(password_bytes, salt)
            
            try:
                fernet = Fernet(derived_key)
                encrypted_marker = key_path.read_bytes()
                decrypted = fernet.decrypt(encrypted_marker)
                
                if decrypted == b"PQC_MASTER_KEY_VALID":
                    self._master_fernet = fernet
                    self._master_password = password_bytes
                    print("✓ Master password verified")
                    return True
                else:
                    print("✗ Master password verification failed")
                    return False
            except Exception as e:
                print(f"✗ Master password incorrect: {e}")
                return False
        else:
            # Initialize new master password
            salt = secrets.token_bytes(32)
            salt_path.write_bytes(salt)
            
            derived_key = self._derive_key_from_password(password_bytes, salt)
            fernet = Fernet(derived_key)
            
            # Store encrypted marker to verify password later
            encrypted_marker = fernet.encrypt(b"PQC_MASTER_KEY_VALID")
            key_path.write_bytes(encrypted_marker)
            
            self._master_fernet = fernet
            self._master_password = password_bytes
            
            print("✓ Master password initialized")
            return True
    
    def _load_or_generate_master_password(self) -> bytes:
        """
        Load existing master password or generate a new one
        
        For automated/testing scenarios where no interactive password is available.
        WARNING: In production, always use initialize_master_password() with user input.
        
        Returns:
            bytes: Master password
        """
        env_password = os.environ.get("PQC_MASTER_PASSWORD")
        
        if env_password:
            password = env_password
        else:
            # Auto-generate for PoC (NOT recommended for production)
            auto_password_path = self.keys_dir / ".auto_master_password"
            
            if auto_password_path.exists():
                password = auto_password_path.read_text().strip()
            else:
                password = secrets.token_urlsafe(32)
                auto_password_path.write_text(password)
                # Set restrictive permissions
                os.chmod(auto_password_path, 0o600)
                print("⚠ Auto-generated master password (for PoC only)")
        
        if self.initialize_master_password(password):
            return password.encode('utf-8')
        else:
            raise RuntimeError("Failed to initialize master password")
    
    def ensure_master_key_initialized(self):
        """Ensure master key is initialized before operations"""
        if self._master_fernet is None:
            self._load_or_generate_master_password()
    
    # ==========================================================================
    # Key Generation
    # ==========================================================================
    
    def generate_user_keys(self, user_id: int, include_sphincs: bool = True) -> PQCKeySet:
        """
        Generate complete PQC key set for a user
        
        Generates:
        - NTRU keypair (primary KEM)
        - Kyber keypair (secondary KEM for hybrid)
        - Dilithium keypair (JWT signing)
        - SPHINCS+ keypair (critical operations) - optional
        
        Args:
            user_id: User identifier
            include_sphincs: Whether to generate SPHINCS+ keys
            
        Returns:
            PQCKeySet: Complete key set for the user
        """
        print(f"Generating PQC keys for user {user_id}...")
        
        # Generate NTRU keypair
        ntru_pub, ntru_priv = pqc_service.generate_ntru_keypair()
        
        # Generate Kyber keypair
        kyber_pub, kyber_priv = pqc_service.generate_kyber_keypair()
        
        # Generate Dilithium keypair
        dilithium_pub, dilithium_priv = pqc_service.generate_dilithium_keypair()
        
        # Generate SPHINCS+ keypair (optional, for critical operations)
        sphincs_pub, sphincs_priv = None, None
        if include_sphincs:
            sphincs_pub, sphincs_priv = pqc_service.generate_sphincs_keypair()
        
        key_set = PQCKeySet(
            user_id=user_id,
            ntru_public=ntru_pub,
            ntru_private=ntru_priv,
            kyber_public=kyber_pub,
            kyber_private=kyber_priv,
            dilithium_public=dilithium_pub,
            dilithium_private=dilithium_priv,
            sphincs_public=sphincs_pub,
            sphincs_private=sphincs_priv
        )
        
        print(f"✓ Generated PQC keys for user {user_id}")
        return key_set
    
    def generate_system_keys(self) -> Dict[str, bytes]:
        """
        Generate system-wide PQC keys (for server-side operations)
        
        Returns:
            Dict with system keys
        """
        system_dir = self.keys_dir / "system"
        system_dir.mkdir(exist_ok=True)
        
        # Generate keys
        ntru_pub, ntru_priv = pqc_service.generate_ntru_keypair()
        kyber_pub, kyber_priv = pqc_service.generate_kyber_keypair()
        dilithium_pub, dilithium_priv = pqc_service.generate_dilithium_keypair()
        sphincs_pub, sphincs_priv = pqc_service.generate_sphincs_keypair()
        
        keys = {
            "ntru_public": ntru_pub,
            "ntru_private": ntru_priv,
            "kyber_public": kyber_pub,
            "kyber_private": kyber_priv,
            "dilithium_public": dilithium_pub,
            "dilithium_private": dilithium_priv,
            "sphincs_public": sphincs_pub,
            "sphincs_private": sphincs_priv
        }
        
        # Save system keys
        self.ensure_master_key_initialized()
        
        for key_name, key_data in keys.items():
            if "private" in key_name:
                # Encrypt private keys
                encrypted = self._master_fernet.encrypt(key_data)
                (system_dir / f"{key_name}.enc").write_bytes(encrypted)
            else:
                # Public keys stored as-is
                (system_dir / f"{key_name}.key").write_bytes(key_data)
        
        # Save metadata
        metadata = {
            "created_at": datetime.utcnow().isoformat(),
            "algorithms": {
                "ntru": "NTRU-HPS-2048-509",
                "kyber": "Kyber768",
                "dilithium": "Dilithium3",
                "sphincs": "SPHINCS+-SHA2-128f-simple"
            }
        }
        (system_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        
        print("✓ Generated system PQC keys")
        return keys
    
    def load_system_keys(self) -> Optional[Dict[str, bytes]]:
        """
        Load system-wide PQC keys
        
        Returns:
            Dict with system keys or None if not found
        """
        system_dir = self.keys_dir / "system"
        
        if not system_dir.exists():
            return None
        
        self.ensure_master_key_initialized()
        
        keys = {}
        
        try:
            # Load public keys
            for key_type in ["ntru", "kyber", "dilithium", "sphincs"]:
                pub_path = system_dir / f"{key_type}_public.key"
                if pub_path.exists():
                    keys[f"{key_type}_public"] = pub_path.read_bytes()
                
                priv_path = system_dir / f"{key_type}_private.enc"
                if priv_path.exists():
                    encrypted = priv_path.read_bytes()
                    keys[f"{key_type}_private"] = self._master_fernet.decrypt(encrypted)
            
            return keys
        except Exception as e:
            print(f"Error loading system keys: {e}")
            return None
    
    # ==========================================================================
    # Key Storage
    # ==========================================================================
    
    def _get_user_keys_dir(self, user_id: int) -> Path:
        """Get directory for user's keys"""
        return self.keys_dir / f"user_{user_id}"
    
    def save_user_keys(self, user_id: int, keys: PQCKeySet) -> bool:
        """
        Save user keys to disk with encryption for private keys
        
        Directory structure:
        data/pqc_keys/
        └── user_{id}/
            ├── ntru_public.key
            ├── ntru_private.key.enc
            ├── kyber_public.key
            ├── kyber_private.key.enc
            ├── dilithium_public.key
            ├── dilithium_private.key.enc
            ├── sphincs_public.key
            ├── sphincs_private.key.enc
            └── key_metadata.json
        
        Args:
            user_id: User identifier
            keys: PQCKeySet to save
            
        Returns:
            bool: True if save successful
        """
        self.ensure_master_key_initialized()
        
        user_dir = self._get_user_keys_dir(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save public keys (unencrypted)
            (user_dir / "ntru_public.key").write_bytes(keys.ntru_public)
            (user_dir / "kyber_public.key").write_bytes(keys.kyber_public)
            (user_dir / "dilithium_public.key").write_bytes(keys.dilithium_public)
            
            if keys.sphincs_public:
                (user_dir / "sphincs_public.key").write_bytes(keys.sphincs_public)
            
            # Save private keys (encrypted)
            encrypted_ntru_priv = self._master_fernet.encrypt(keys.ntru_private)
            (user_dir / "ntru_private.key.enc").write_bytes(encrypted_ntru_priv)
            
            encrypted_kyber_priv = self._master_fernet.encrypt(keys.kyber_private)
            (user_dir / "kyber_private.key.enc").write_bytes(encrypted_kyber_priv)
            
            encrypted_dilithium_priv = self._master_fernet.encrypt(keys.dilithium_private)
            (user_dir / "dilithium_private.key.enc").write_bytes(encrypted_dilithium_priv)
            
            if keys.sphincs_private:
                encrypted_sphincs_priv = self._master_fernet.encrypt(keys.sphincs_private)
                (user_dir / "sphincs_private.key.enc").write_bytes(encrypted_sphincs_priv)
            
            # Save metadata
            metadata = {
                "user_id": user_id,
                "created_at": keys.created_at,
                "key_version": keys.key_version,
                "algorithm_versions": keys.algorithm_versions,
                "has_sphincs": keys.sphincs_public is not None
            }
            (user_dir / self.config.METADATA_FILE).write_text(
                json.dumps(metadata, indent=2)
            )
            
            # Update cache
            self._key_cache[user_id] = keys
            
            print(f"✓ Saved PQC keys for user {user_id}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to save keys for user {user_id}: {e}")
            return False
    
    def load_user_keys(self, user_id: int) -> Optional[PQCKeySet]:
        """
        Load and decrypt user keys from disk
        
        Args:
            user_id: User identifier
            
        Returns:
            PQCKeySet or None if not found
        """
        # Check cache first
        if user_id in self._key_cache:
            return self._key_cache[user_id]
        
        self.ensure_master_key_initialized()
        
        user_dir = self._get_user_keys_dir(user_id)
        
        if not user_dir.exists():
            print(f"No keys found for user {user_id}")
            return None
        
        try:
            # Load public keys
            ntru_public = (user_dir / "ntru_public.key").read_bytes()
            kyber_public = (user_dir / "kyber_public.key").read_bytes()
            dilithium_public = (user_dir / "dilithium_public.key").read_bytes()
            
            sphincs_public = None
            sphincs_public_path = user_dir / "sphincs_public.key"
            if sphincs_public_path.exists():
                sphincs_public = sphincs_public_path.read_bytes()
            
            # Load and decrypt private keys
            encrypted_ntru_priv = (user_dir / "ntru_private.key.enc").read_bytes()
            ntru_private = self._master_fernet.decrypt(encrypted_ntru_priv)
            
            encrypted_kyber_priv = (user_dir / "kyber_private.key.enc").read_bytes()
            kyber_private = self._master_fernet.decrypt(encrypted_kyber_priv)
            
            encrypted_dilithium_priv = (user_dir / "dilithium_private.key.enc").read_bytes()
            dilithium_private = self._master_fernet.decrypt(encrypted_dilithium_priv)
            
            sphincs_private = None
            sphincs_priv_path = user_dir / "sphincs_private.key.enc"
            if sphincs_priv_path.exists():
                encrypted_sphincs_priv = sphincs_priv_path.read_bytes()
                sphincs_private = self._master_fernet.decrypt(encrypted_sphincs_priv)
            
            # Load metadata
            metadata_path = user_dir / self.config.METADATA_FILE
            metadata = {}
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
            
            key_set = PQCKeySet(
                user_id=user_id,
                ntru_public=ntru_public,
                ntru_private=ntru_private,
                kyber_public=kyber_public,
                kyber_private=kyber_private,
                dilithium_public=dilithium_public,
                dilithium_private=dilithium_private,
                sphincs_public=sphincs_public,
                sphincs_private=sphincs_private,
                created_at=metadata.get("created_at", ""),
                key_version=metadata.get("key_version", 1),
                algorithm_versions=metadata.get("algorithm_versions", {})
            )
            
            # Update cache
            self._key_cache[user_id] = key_set
            
            return key_set
            
        except Exception as e:
            print(f"✗ Failed to load keys for user {user_id}: {e}")
            return None
    
    def export_public_keys(self, user_id: int) -> Optional[Dict[str, bytes]]:
        """
        Export only public keys for a user (for key distribution)
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with public keys or None if not found
        """
        keys = self.load_user_keys(user_id)
        
        if keys is None:
            return None
        
        public_keys = {
            "ntru_public": keys.ntru_public,
            "kyber_public": keys.kyber_public,
            "dilithium_public": keys.dilithium_public
        }
        
        if keys.sphincs_public:
            public_keys["sphincs_public"] = keys.sphincs_public
        
        return public_keys
    
    def user_has_keys(self, user_id: int) -> bool:
        """Check if user has PQC keys generated"""
        user_dir = self._get_user_keys_dir(user_id)
        return user_dir.exists() and (user_dir / "ntru_public.key").exists()
    
    # ==========================================================================
    # Key Rotation
    # ==========================================================================
    
    def rotate_user_keys(self, user_id: int) -> Optional[PQCKeySet]:
        """
        Rotate user's PQC keys (generate new keys, archive old ones)
        
        Args:
            user_id: User identifier
            
        Returns:
            PQCKeySet: New key set, or None if rotation failed
        """
        user_dir = self._get_user_keys_dir(user_id)
        
        # Archive old keys if they exist
        if user_dir.exists():
            # Load old metadata for version info
            metadata_path = user_dir / self.config.METADATA_FILE
            old_version = 1
            if metadata_path.exists():
                old_metadata = json.loads(metadata_path.read_text())
                old_version = old_metadata.get("key_version", 1)
            
            # Create archive directory
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            archive_dir = user_dir / f"archive_v{old_version}_{timestamp}"
            archive_dir.mkdir(exist_ok=True)
            
            # Move old keys to archive
            for key_file in user_dir.glob("*.key*"):
                if key_file.is_file():
                    key_file.rename(archive_dir / key_file.name)
            
            if metadata_path.exists():
                metadata_path.rename(archive_dir / metadata_path.name)
            
            print(f"Archived old keys for user {user_id} to {archive_dir}")
            new_version = old_version + 1
        else:
            new_version = 1
        
        # Generate new keys
        has_sphincs = True  # Include SPHINCS+ in rotation
        new_keys = self.generate_user_keys(user_id, include_sphincs=has_sphincs)
        new_keys.key_version = new_version
        
        # Save new keys
        if self.save_user_keys(user_id, new_keys):
            # Clear cache to force reload
            if user_id in self._key_cache:
                del self._key_cache[user_id]
            
            print(f"✓ Rotated keys for user {user_id} to version {new_version}")
            return new_keys
        else:
            return None
    
    def check_key_expiry(self, user_id: int) -> Dict[str, Any]:
        """
        Check if user's keys are approaching expiry
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with expiry information
        """
        user_dir = self._get_user_keys_dir(user_id)
        metadata_path = user_dir / self.config.METADATA_FILE
        
        if not metadata_path.exists():
            return {"status": "not_found", "needs_rotation": False}
        
        metadata = json.loads(metadata_path.read_text())
        created_at = datetime.fromisoformat(metadata.get("created_at", datetime.utcnow().isoformat()))
        
        expiry_date = created_at + timedelta(days=self.config.KEY_VALIDITY_DAYS)
        warning_date = expiry_date - timedelta(days=self.config.ROTATION_WARNING_DAYS)
        now = datetime.utcnow()
        
        days_until_expiry = (expiry_date - now).days
        
        return {
            "status": "ok" if now < warning_date else ("warning" if now < expiry_date else "expired"),
            "created_at": created_at.isoformat(),
            "expiry_date": expiry_date.isoformat(),
            "days_until_expiry": max(0, days_until_expiry),
            "needs_rotation": now >= warning_date,
            "key_version": metadata.get("key_version", 1)
        }
    
    # ==========================================================================
    # Bulk Operations
    # ==========================================================================
    
    def list_all_users_with_keys(self) -> List[int]:
        """
        List all user IDs that have PQC keys
        
        Returns:
            List of user IDs
        """
        user_ids = []
        
        for item in self.keys_dir.iterdir():
            if item.is_dir() and item.name.startswith("user_"):
                try:
                    user_id = int(item.name.split("_")[1])
                    user_ids.append(user_id)
                except (ValueError, IndexError):
                    continue
        
        return sorted(user_ids)
    
    def delete_user_keys(self, user_id: int) -> bool:
        """
        Securely delete user's PQC keys
        
        Args:
            user_id: User identifier
            
        Returns:
            bool: True if deletion successful
        """
        import shutil
        
        user_dir = self._get_user_keys_dir(user_id)
        
        if not user_dir.exists():
            return False
        
        try:
            # Overwrite files before deletion (secure deletion)
            for key_file in user_dir.rglob("*"):
                if key_file.is_file():
                    size = key_file.stat().st_size
                    key_file.write_bytes(secrets.token_bytes(size))
            
            # Remove directory
            shutil.rmtree(user_dir)
            
            # Clear from cache
            if user_id in self._key_cache:
                del self._key_cache[user_id]
            
            print(f"✓ Securely deleted keys for user {user_id}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to delete keys for user {user_id}: {e}")
            return False
    
    def get_key_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored keys
        
        Returns:
            Dict with key statistics
        """
        user_ids = self.list_all_users_with_keys()
        
        stats = {
            "total_users": len(user_ids),
            "users_with_keys": user_ids,
            "system_keys_exist": (self.keys_dir / "system").exists(),
            "keys_directory": str(self.keys_dir),
            "key_expiry_warnings": []
        }
        
        # Check for expiring keys
        for user_id in user_ids:
            expiry_info = self.check_key_expiry(user_id)
            if expiry_info.get("needs_rotation"):
                stats["key_expiry_warnings"].append({
                    "user_id": user_id,
                    "days_until_expiry": expiry_info.get("days_until_expiry"),
                    "status": expiry_info.get("status")
                })
        
        return stats


# ==========================================================================
# Singleton Instance
# ==========================================================================

pqc_key_manager = PQCKeyManager()


# ==========================================================================
# Convenience Functions
# ==========================================================================

def get_user_keys(user_id: int) -> Optional[PQCKeySet]:
    """Get keys for a user (generates if not exist)"""
    keys = pqc_key_manager.load_user_keys(user_id)
    
    if keys is None:
        keys = pqc_key_manager.generate_user_keys(user_id)
        pqc_key_manager.save_user_keys(user_id, keys)
    
    return keys


def get_public_keys(user_id: int) -> Optional[Dict[str, bytes]]:
    """Get public keys for a user"""
    return pqc_key_manager.export_public_keys(user_id)
