"""
Hybrid Cryptography Service
Implements NTRU + Kyber hybrid encryption for maximum quantum resistance

This service provides a unified interface for hybrid post-quantum encryption,
combining two different lattice-based KEMs for defense-in-depth security.

Rationale for Hybrid Approach:
1. Defense in depth: If one algorithm is broken, the other provides security
2. Different mathematical assumptions (NTRU vs Kyber lattice constructions)
3. Industry best practice for critical infrastructure
4. Future-proof against cryptanalytic advances

Security Level: NIST Level 3+ (combined security of both algorithms)
"""

import os
import struct
import hashlib
import secrets
import zlib
from typing import Tuple, Optional, Dict, Any, Union
from datetime import datetime
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend

from app.services.pqc_service import pqc_service
from app.services.pqc_key_manager import pqc_key_manager, PQCKeySet


class HybridCryptoConfig:
    """Configuration for hybrid cryptography operations"""
    # AES-256-GCM parameters
    AES_KEY_SIZE = 32  # 256 bits
    AES_NONCE_SIZE = 12  # 96 bits
    AES_TAG_SIZE = 16  # 128 bits
    
    # Compression
    COMPRESSION_LEVEL = 9  # Max compression
    COMPRESSION_THRESHOLD = 1024  # Only compress if data > 1KB
    
    # Versioning for format changes
    FORMAT_VERSION = 1
    
    # Magic bytes for format identification
    MAGIC_BYTES = b'PQCH'  # PQC Hybrid


class HybridCryptoService:
    """
    Hybrid Post-Quantum Cryptography Service
    
    Provides combined NTRU + Kyber encryption for maximum security against
    both classical and quantum attacks.
    
    Encryption Process:
    1. Generate shared secrets with both NTRU and Kyber KEMs
    2. Combine secrets using XOR (independent security)
    3. Derive symmetric key using HKDF
    4. Encrypt data with AES-256-GCM
    5. Package: version || ntru_ct || kyber_ct || nonce || aes_ct || tag
    
    Decryption Process:
    1. Parse package to extract components
    2. Decapsulate both secrets
    3. Combine and derive symmetric key
    4. Decrypt with AES-256-GCM
    """
    
    def __init__(self):
        """Initialize Hybrid Crypto Service"""
        self.config = HybridCryptoConfig()
        print("✓ Hybrid Crypto Service initialized")
    
    # ==========================================================================
    # Core Hybrid Encryption
    # ==========================================================================
    
    def encrypt(
        self,
        data: bytes,
        ntru_public_key: bytes,
        kyber_public_key: bytes,
        compress: bool = True,
        associated_data: Optional[bytes] = None
    ) -> bytes:
        """
        Encrypt data using NTRU + Kyber hybrid encryption
        
        Args:
            data: Plaintext data to encrypt
            ntru_public_key: NTRU public key
            kyber_public_key: Kyber public key
            compress: Whether to compress data before encryption
            associated_data: Additional authenticated data (AAD)
            
        Returns:
            bytes: Encrypted data package
        """
        # Step 1: Optionally compress data
        if compress and len(data) > self.config.COMPRESSION_THRESHOLD:
            compressed_data = zlib.compress(data, self.config.COMPRESSION_LEVEL)
            is_compressed = True
        else:
            compressed_data = data
            is_compressed = False
        
        # Step 2: Encapsulate with both KEMs
        ntru_ciphertext, ntru_secret = pqc_service.ntru_encapsulate(ntru_public_key)
        kyber_ciphertext, kyber_secret = pqc_service.kyber_encapsulate(kyber_public_key)
        
        # Step 3: Combine shared secrets (XOR for independent security)
        # If one KEM is compromised, the other still provides security
        combined_secret = self._combine_secrets(ntru_secret, kyber_secret)
        
        # Step 4: Derive AES key using HKDF
        aes_key = self._derive_key(
            combined_secret,
            info=b"hybrid_ntru_kyber_aes256gcm"
        )
        
        # Step 5: Encrypt with AES-256-GCM
        nonce = secrets.token_bytes(self.config.AES_NONCE_SIZE)
        aesgcm = AESGCM(aes_key)
        
        # Include KEM ciphertexts in AAD for integrity
        aad = self._create_aad(ntru_ciphertext, kyber_ciphertext, associated_data)
        
        aes_ciphertext = aesgcm.encrypt(nonce, compressed_data, aad)
        
        # Step 6: Package everything
        package = self._pack_encrypted_data(
            ntru_ciphertext=ntru_ciphertext,
            kyber_ciphertext=kyber_ciphertext,
            nonce=nonce,
            aes_ciphertext=aes_ciphertext,
            is_compressed=is_compressed,
            has_aad=associated_data is not None
        )
        
        return package
    
    def decrypt(
        self,
        encrypted_data: bytes,
        ntru_private_key: bytes,
        kyber_private_key: bytes,
        associated_data: Optional[bytes] = None
    ) -> bytes:
        """
        Decrypt hybrid-encrypted data
        
        Args:
            encrypted_data: Encrypted data package
            ntru_private_key: NTRU private key
            kyber_private_key: Kyber private key
            associated_data: Additional authenticated data (must match encryption)
            
        Returns:
            bytes: Decrypted plaintext
        """
        # Step 1: Unpack encrypted data
        components = self._unpack_encrypted_data(encrypted_data)
        
        # Step 2: Decapsulate both secrets
        ntru_secret = pqc_service.ntru_decapsulate(
            components['ntru_ciphertext'],
            ntru_private_key
        )
        kyber_secret = pqc_service.kyber_decapsulate(
            components['kyber_ciphertext'],
            kyber_private_key
        )
        
        # Step 3: Combine secrets
        combined_secret = self._combine_secrets(ntru_secret, kyber_secret)
        
        # Step 4: Derive AES key
        aes_key = self._derive_key(
            combined_secret,
            info=b"hybrid_ntru_kyber_aes256gcm"
        )
        
        # Step 5: Decrypt with AES-256-GCM
        aad = self._create_aad(
            components['ntru_ciphertext'],
            components['kyber_ciphertext'],
            associated_data
        )
        
        aesgcm = AESGCM(aes_key)
        decrypted_data = aesgcm.decrypt(
            components['nonce'],
            components['aes_ciphertext'],
            aad
        )
        
        # Step 6: Decompress if needed
        if components['is_compressed']:
            decrypted_data = zlib.decompress(decrypted_data)
        
        return decrypted_data
    
    # ==========================================================================
    # User-Based Encryption
    # ==========================================================================
    
    def encrypt_for_user(
        self,
        data: bytes,
        user_id: int,
        compress: bool = True
    ) -> bytes:
        """
        Encrypt data for a specific user using their PQC keys
        
        Args:
            data: Plaintext data
            user_id: Target user's ID
            compress: Whether to compress
            
        Returns:
            bytes: Encrypted data
        """
        # Load user's public keys
        public_keys = pqc_key_manager.export_public_keys(user_id)
        
        if public_keys is None:
            # Generate keys if not exist
            key_set = pqc_key_manager.generate_user_keys(user_id)
            pqc_key_manager.save_user_keys(user_id, key_set)
            public_keys = {
                "ntru_public": key_set.ntru_public,
                "kyber_public": key_set.kyber_public
            }
        
        return self.encrypt(
            data,
            public_keys["ntru_public"],
            public_keys["kyber_public"],
            compress=compress
        )
    
    def decrypt_for_user(
        self,
        encrypted_data: bytes,
        user_id: int
    ) -> bytes:
        """
        Decrypt data encrypted for a specific user
        
        Args:
            encrypted_data: Encrypted data package
            user_id: User's ID
            
        Returns:
            bytes: Decrypted data
        """
        # Load user's private keys
        user_keys = pqc_key_manager.load_user_keys(user_id)
        
        if user_keys is None:
            raise ValueError(f"No keys found for user {user_id}")
        
        return self.decrypt(
            encrypted_data,
            user_keys.ntru_private,
            user_keys.kyber_private
        )
    
    # ==========================================================================
    # Face Embedding Encryption
    # ==========================================================================
    
    def encrypt_embedding(
        self,
        embedding,
        user_id: int,
        sign: bool = True
    ) -> bytes:
        """
        Encrypt face embedding with hybrid PQC and optional Dilithium signature
        
        Args:
            embedding: Numpy array or bytes of face embedding
            user_id: User's ID
            sign: Whether to add Dilithium signature for integrity
            
        Returns:
            bytes: Encrypted (and optionally signed) embedding
        """
        import numpy as np
        
        # Load user keys
        user_keys = pqc_key_manager.load_user_keys(user_id)
        if user_keys is None:
            user_keys = pqc_key_manager.generate_user_keys(user_id)
            pqc_key_manager.save_user_keys(user_id, user_keys)
        
        # Serialize embedding
        if isinstance(embedding, np.ndarray):
            embedding_bytes = embedding.astype(np.float32).tobytes()
            shape = embedding.shape
        else:
            embedding_bytes = bytes(embedding)
            shape = (len(embedding_bytes) // 4,)  # Assume float32
        
        # Add shape metadata
        shape_data = struct.pack('>I', len(shape))
        for dim in shape:
            shape_data += struct.pack('>I', dim)
        
        data_with_metadata = shape_data + embedding_bytes
        
        # Encrypt with hybrid PQC
        encrypted = self.encrypt(
            data_with_metadata,
            user_keys.ntru_public,
            user_keys.kyber_public,
            compress=True
        )
        
        # Optionally sign
        if sign:
            signature = pqc_service.sign_dilithium(encrypted, user_keys.dilithium_private)
            
            # Pack: signature_len || signature || encrypted_data
            result = struct.pack('>I', len(signature))
            result += signature
            result += encrypted
            
            # Add header indicating signed
            final = b'PQCS' + result  # PQC Signed
        else:
            final = b'PQCU' + encrypted  # PQC Unsigned
        
        return final
    
    def decrypt_embedding(
        self,
        encrypted_data: bytes,
        user_id: int,
        verify_signature: bool = True
    ):
        """
        Decrypt face embedding
        
        Args:
            encrypted_data: Encrypted embedding package
            user_id: User's ID
            verify_signature: Whether to verify Dilithium signature
            
        Returns:
            np.ndarray: Decrypted embedding
        """
        import numpy as np
        
        user_keys = pqc_key_manager.load_user_keys(user_id)
        if user_keys is None:
            raise ValueError(f"No keys found for user {user_id}")
        
        # Check header
        header = encrypted_data[:4]
        data = encrypted_data[4:]
        
        if header == b'PQCS':
            # Signed data
            sig_len = struct.unpack('>I', data[:4])[0]
            signature = data[4:4 + sig_len]
            encrypted = data[4 + sig_len:]
            
            if verify_signature:
                if not pqc_service.verify_dilithium(encrypted, signature, user_keys.dilithium_public):
                    raise ValueError("Signature verification failed")
        elif header == b'PQCU':
            # Unsigned data
            encrypted = data
        else:
            raise ValueError("Invalid encrypted embedding format")
        
        # Decrypt
        decrypted = self.decrypt(
            encrypted,
            user_keys.ntru_private,
            user_keys.kyber_private
        )
        
        # Parse shape metadata
        offset = 0
        num_dims = struct.unpack('>I', decrypted[offset:offset + 4])[0]
        offset += 4
        
        shape = []
        for _ in range(num_dims):
            dim = struct.unpack('>I', decrypted[offset:offset + 4])[0]
            shape.append(dim)
            offset += 4
        
        # Reconstruct embedding
        embedding_bytes = decrypted[offset:]
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        if shape:
            embedding = embedding.reshape(tuple(shape))
        
        return embedding
    
    # ==========================================================================
    # File Encryption
    # ==========================================================================
    
    def encrypt_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        user_id: int,
        delete_original: bool = False
    ) -> bool:
        """
        Encrypt a file using hybrid PQC
        
        Args:
            input_path: Path to file to encrypt
            output_path: Path for encrypted output
            user_id: User's ID for encryption keys
            delete_original: Whether to securely delete original
            
        Returns:
            bool: True if successful
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        try:
            # Read file
            data = input_path.read_bytes()
            
            # Encrypt
            encrypted = self.encrypt_for_user(data, user_id, compress=True)
            
            # Add file metadata
            metadata = {
                "original_name": input_path.name,
                "original_size": len(data),
                "encrypted_at": datetime.utcnow().isoformat()
            }
            metadata_bytes = str(metadata).encode('utf-8')
            
            # Pack: metadata_len || metadata || encrypted_data
            final = struct.pack('>I', len(metadata_bytes))
            final += metadata_bytes
            final += encrypted
            
            # Write encrypted file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(final)
            
            # Optionally delete original
            if delete_original:
                # Overwrite with random data before deletion
                random_data = secrets.token_bytes(len(data))
                input_path.write_bytes(random_data)
                input_path.unlink()
            
            return True
            
        except Exception as e:
            print(f"File encryption error: {e}")
            return False
    
    def decrypt_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        user_id: int
    ) -> bool:
        """
        Decrypt a file
        
        Args:
            input_path: Path to encrypted file
            output_path: Path for decrypted output
            user_id: User's ID for decryption keys
            
        Returns:
            bool: True if successful
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        try:
            # Read encrypted file
            data = input_path.read_bytes()
            
            # Parse metadata
            metadata_len = struct.unpack('>I', data[:4])[0]
            metadata_bytes = data[4:4 + metadata_len]
            encrypted = data[4 + metadata_len:]
            
            # Decrypt
            decrypted = self.decrypt_for_user(encrypted, user_id)
            
            # Write decrypted file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(decrypted)
            
            return True
            
        except Exception as e:
            print(f"File decryption error: {e}")
            return False
    
    # ==========================================================================
    # Helper Methods
    # ==========================================================================
    
    def _combine_secrets(self, secret1: bytes, secret2: bytes) -> bytes:
        """
        Combine two shared secrets using XOR
        
        XOR provides independent security: even if one secret is compromised,
        the combined secret is still secure as long as the other is safe.
        """
        # Ensure same length (pad shorter with hash if needed)
        len1, len2 = len(secret1), len(secret2)
        
        if len1 != len2:
            if len1 < len2:
                secret1 = secret1 + hashlib.sha256(secret1).digest()[:len2 - len1]
            else:
                secret2 = secret2 + hashlib.sha256(secret2).digest()[:len1 - len2]
        
        return bytes(a ^ b for a, b in zip(secret1, secret2))
    
    def _derive_key(self, secret: bytes, info: bytes) -> bytes:
        """Derive AES key from combined secret using HKDF"""
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=self.config.AES_KEY_SIZE,
            salt=None,
            info=info,
            backend=default_backend()
        )
        return hkdf.derive(secret)
    
    def _create_aad(
        self,
        ntru_ct: bytes,
        kyber_ct: bytes,
        extra_aad: Optional[bytes]
    ) -> bytes:
        """Create Additional Authenticated Data for AES-GCM"""
        aad = hashlib.sha256(ntru_ct).digest()
        aad += hashlib.sha256(kyber_ct).digest()
        
        if extra_aad:
            aad += hashlib.sha256(extra_aad).digest()
        
        return aad
    
    def _pack_encrypted_data(
        self,
        ntru_ciphertext: bytes,
        kyber_ciphertext: bytes,
        nonce: bytes,
        aes_ciphertext: bytes,
        is_compressed: bool,
        has_aad: bool
    ) -> bytes:
        """
        Pack encrypted data into a structured format
        
        Format:
        - Magic bytes (4): 'PQCH'
        - Version (1): Format version
        - Flags (1): Compression, AAD, etc.
        - NTRU ciphertext length (4)
        - NTRU ciphertext
        - Kyber ciphertext length (4)
        - Kyber ciphertext
        - Nonce (12)
        - AES ciphertext (remaining)
        """
        flags = 0
        if is_compressed:
            flags |= 0x01
        if has_aad:
            flags |= 0x02
        
        result = self.config.MAGIC_BYTES
        result += struct.pack('B', self.config.FORMAT_VERSION)
        result += struct.pack('B', flags)
        result += struct.pack('>I', len(ntru_ciphertext))
        result += ntru_ciphertext
        result += struct.pack('>I', len(kyber_ciphertext))
        result += kyber_ciphertext
        result += nonce
        result += aes_ciphertext
        
        return result
    
    def _unpack_encrypted_data(self, data: bytes) -> Dict[str, Any]:
        """Unpack structured encrypted data"""
        offset = 0
        
        # Magic bytes
        magic = data[offset:offset + 4]
        offset += 4
        
        if magic != self.config.MAGIC_BYTES:
            raise ValueError("Invalid encrypted data format")
        
        # Version
        version = struct.unpack('B', data[offset:offset + 1])[0]
        offset += 1
        
        if version != self.config.FORMAT_VERSION:
            raise ValueError(f"Unsupported format version: {version}")
        
        # Flags
        flags = struct.unpack('B', data[offset:offset + 1])[0]
        offset += 1
        
        is_compressed = bool(flags & 0x01)
        has_aad = bool(flags & 0x02)
        
        # NTRU ciphertext
        ntru_len = struct.unpack('>I', data[offset:offset + 4])[0]
        offset += 4
        ntru_ciphertext = data[offset:offset + ntru_len]
        offset += ntru_len
        
        # Kyber ciphertext
        kyber_len = struct.unpack('>I', data[offset:offset + 4])[0]
        offset += 4
        kyber_ciphertext = data[offset:offset + kyber_len]
        offset += kyber_len
        
        # Nonce
        nonce = data[offset:offset + self.config.AES_NONCE_SIZE]
        offset += self.config.AES_NONCE_SIZE
        
        # AES ciphertext
        aes_ciphertext = data[offset:]
        
        return {
            "version": version,
            "is_compressed": is_compressed,
            "has_aad": has_aad,
            "ntru_ciphertext": ntru_ciphertext,
            "kyber_ciphertext": kyber_ciphertext,
            "nonce": nonce,
            "aes_ciphertext": aes_ciphertext
        }
    
    def get_encrypted_size_overhead(self, plaintext_size: int) -> int:
        """
        Calculate the size overhead of hybrid encryption
        
        Args:
            plaintext_size: Size of plaintext in bytes
            
        Returns:
            int: Approximate ciphertext size
        """
        # NTRU-HPS-2048-509 ciphertext: ~699 bytes
        # Kyber768 ciphertext: ~1088 bytes
        # AES-GCM overhead: 16 bytes (tag)
        # Header: ~20 bytes
        # Nonce: 12 bytes
        
        overhead = 699 + 1088 + 16 + 20 + 12
        
        # Account for compression (rough estimate)
        if plaintext_size > self.config.COMPRESSION_THRESHOLD:
            estimated_compressed = int(plaintext_size * 0.6)  # Assume 40% compression
        else:
            estimated_compressed = plaintext_size
        
        return overhead + estimated_compressed


# ==========================================================================
# Singleton Instance
# ==========================================================================

hybrid_crypto_service = HybridCryptoService()


# ==========================================================================
# Convenience Functions
# ==========================================================================

def encrypt_data(
    data: bytes,
    ntru_pubkey: bytes,
    kyber_pubkey: bytes
) -> bytes:
    """Encrypt data using hybrid NTRU + Kyber"""
    return hybrid_crypto_service.encrypt(data, ntru_pubkey, kyber_pubkey)


def decrypt_data(
    encrypted: bytes,
    ntru_privkey: bytes,
    kyber_privkey: bytes
) -> bytes:
    """Decrypt hybrid-encrypted data"""
    return hybrid_crypto_service.decrypt(encrypted, ntru_privkey, kyber_privkey)


def encrypt_for_user(data: bytes, user_id: int) -> bytes:
    """Encrypt data for a specific user"""
    return hybrid_crypto_service.encrypt_for_user(data, user_id)


def decrypt_for_user(encrypted: bytes, user_id: int) -> bytes:
    """Decrypt data for a specific user"""
    return hybrid_crypto_service.decrypt_for_user(encrypted, user_id)
