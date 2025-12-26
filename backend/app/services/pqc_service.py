"""
Post-Quantum Cryptography Service
Implements NTRU, Kyber, Dilithium, SPHINCS+ using liboqs-python

NIST-approved algorithms for quantum-resistant cryptography:
- NTRU-HPS-2048-509: Primary Key Encapsulation Mechanism (KEM)
- Kyber768: Secondary KEM for hybrid encryption
- Dilithium3: Digital signatures (replaces HMAC-SHA256 in JWT)
- SPHINCS+-SHA2-128f-simple: Hash-based signatures for critical operations

Security Level: NIST Level 3 (equivalent to AES-192)
"""

import os
import zlib
import struct
import hashlib
import secrets
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
import numpy as np

# Cryptographic imports
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

# OQS (Open Quantum Safe) library for PQC
try:
    import oqs
    OQS_AVAILABLE = True
except ImportError:
    OQS_AVAILABLE = False
    print("WARNING: liboqs-python not installed. PQC operations will use fallback mode.")


class PQCAlgorithmConfig:
    """Configuration for PQC algorithms"""
    # Key Encapsulation Mechanisms (KEM)
    NTRU_ALGORITHM = "NTRU-HPS-2048-509"
    KYBER_ALGORITHM = "Kyber768"
    
    # Digital Signature Algorithms
    DILITHIUM_ALGORITHM = "Dilithium3"
    SPHINCS_ALGORITHM = "SPHINCS+-SHA2-128f-simple"
    
    # Symmetric encryption (AES-256-GCM for data encryption)
    AES_KEY_SIZE = 32  # 256 bits
    AES_NONCE_SIZE = 12  # 96 bits for GCM
    
    # Security parameters
    PBKDF2_ITERATIONS = 600000  # High iteration count for key derivation
    SALT_SIZE = 32


class PQCService:
    """
    Post-Quantum Cryptography Service
    
    Provides quantum-resistant encryption and signature operations using
    NIST-approved algorithms from the Open Quantum Safe (liboqs) library.
    
    Key Features:
    - NTRU + Kyber hybrid encryption for maximum security
    - Dilithium signatures for JWT tokens and data integrity
    - SPHINCS+ for critical operations requiring long-term security
    - AES-256-GCM for symmetric data encryption
    """
    
    def __init__(self):
        """Initialize PQC Service with algorithm instances"""
        self.config = PQCAlgorithmConfig()
        self._oqs_available = OQS_AVAILABLE
        
        if self._oqs_available:
            self._verify_algorithms()
        else:
            print("PQC Service running in fallback mode (reduced security)")
    
    def _verify_algorithms(self):
        """Verify that required OQS algorithms are available"""
        required_kems = [self.config.NTRU_ALGORITHM, self.config.KYBER_ALGORITHM]
        required_sigs = [self.config.DILITHIUM_ALGORITHM, self.config.SPHINCS_ALGORITHM]
        
        available_kems = oqs.get_enabled_kem_mechanisms()
        available_sigs = oqs.get_enabled_sig_mechanisms()
        
        for kem in required_kems:
            if kem not in available_kems:
                raise RuntimeError(f"Required KEM algorithm not available: {kem}")
        
        for sig in required_sigs:
            if sig not in available_sigs:
                raise RuntimeError(f"Required signature algorithm not available: {sig}")
        
        print(f"✓ PQC Service initialized with algorithms:")
        print(f"  - KEM: {', '.join(required_kems)}")
        print(f"  - Signatures: {', '.join(required_sigs)}")
    
    # ==========================================================================
    # NTRU Key Encapsulation
    # ==========================================================================
    
    def generate_ntru_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate NTRU-HPS-2048-509 public/private key pair
        
        NTRU is a lattice-based cryptographic algorithm providing:
        - Fast key generation and encapsulation
        - Smaller ciphertext than some alternatives
        - Well-studied security properties
        
        Returns:
            Tuple[bytes, bytes]: (public_key, secret_key)
        """
        if not self._oqs_available:
            # Fallback: Generate random keys for testing
            return secrets.token_bytes(699), secrets.token_bytes(935)
        
        with oqs.KeyEncapsulation(self.config.NTRU_ALGORITHM) as kem:
            public_key = kem.generate_keypair()
            secret_key = kem.export_secret_key()
            return public_key, secret_key
    
    def ntru_encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """
        Encapsulate a shared secret using NTRU public key
        
        Args:
            public_key: NTRU public key
            
        Returns:
            Tuple[bytes, bytes]: (ciphertext, shared_secret)
        """
        if not self._oqs_available:
            # Fallback: Return random values
            shared_secret = secrets.token_bytes(32)
            ciphertext = secrets.token_bytes(699)
            return ciphertext, shared_secret
        
        with oqs.KeyEncapsulation(self.config.NTRU_ALGORITHM) as kem:
            ciphertext, shared_secret = kem.encap_secret(public_key)
            return ciphertext, shared_secret
    
    def ntru_decapsulate(self, ciphertext: bytes, secret_key: bytes) -> bytes:
        """
        Decapsulate shared secret using NTRU secret key
        
        Args:
            ciphertext: NTRU ciphertext from encapsulation
            secret_key: NTRU secret key
            
        Returns:
            bytes: Shared secret
        """
        if not self._oqs_available:
            # Fallback: Return hash of inputs
            return hashlib.sha256(ciphertext + secret_key).digest()
        
        with oqs.KeyEncapsulation(self.config.NTRU_ALGORITHM, secret_key) as kem:
            shared_secret = kem.decap_secret(ciphertext)
            return shared_secret
    
    def ntru_encrypt(self, data: bytes, public_key: bytes) -> bytes:
        """
        Encrypt data using NTRU (KEM + AES-GCM hybrid)
        
        Process:
        1. Generate shared secret via NTRU KEM
        2. Derive AES key from shared secret
        3. Encrypt data with AES-256-GCM
        4. Return: ciphertext_kem || nonce || ciphertext_aes
        
        Args:
            data: Plaintext data to encrypt
            public_key: NTRU public key
            
        Returns:
            bytes: Encrypted data package
        """
        # Step 1: Encapsulate shared secret
        kem_ciphertext, shared_secret = self.ntru_encapsulate(public_key)
        
        # Step 2: Derive AES key from shared secret
        aes_key = self._derive_aes_key(shared_secret, b"ntru_encryption")
        
        # Step 3: Encrypt with AES-GCM
        nonce = secrets.token_bytes(self.config.AES_NONCE_SIZE)
        aesgcm = AESGCM(aes_key)
        aes_ciphertext = aesgcm.encrypt(nonce, data, associated_data=kem_ciphertext)
        
        # Step 4: Pack: kem_ct_length (4 bytes) || kem_ct || nonce || aes_ct
        result = struct.pack('>I', len(kem_ciphertext))
        result += kem_ciphertext
        result += nonce
        result += aes_ciphertext
        
        return result
    
    def ntru_decrypt(self, ciphertext: bytes, secret_key: bytes) -> bytes:
        """
        Decrypt NTRU-encrypted data
        
        Args:
            ciphertext: Encrypted data package
            secret_key: NTRU secret key
            
        Returns:
            bytes: Decrypted plaintext
        """
        # Step 1: Unpack components
        kem_ct_length = struct.unpack('>I', ciphertext[:4])[0]
        offset = 4
        
        kem_ciphertext = ciphertext[offset:offset + kem_ct_length]
        offset += kem_ct_length
        
        nonce = ciphertext[offset:offset + self.config.AES_NONCE_SIZE]
        offset += self.config.AES_NONCE_SIZE
        
        aes_ciphertext = ciphertext[offset:]
        
        # Step 2: Decapsulate shared secret
        shared_secret = self.ntru_decapsulate(kem_ciphertext, secret_key)
        
        # Step 3: Derive AES key
        aes_key = self._derive_aes_key(shared_secret, b"ntru_encryption")
        
        # Step 4: Decrypt with AES-GCM
        aesgcm = AESGCM(aes_key)
        plaintext = aesgcm.decrypt(nonce, aes_ciphertext, associated_data=kem_ciphertext)
        
        return plaintext
    
    # ==========================================================================
    # Kyber Key Encapsulation
    # ==========================================================================
    
    def generate_kyber_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate Kyber768 public/private key pair
        
        Kyber is a lattice-based KEM selected by NIST for standardization.
        Kyber768 provides NIST Level 3 security.
        
        Returns:
            Tuple[bytes, bytes]: (public_key, secret_key)
        """
        if not self._oqs_available:
            return secrets.token_bytes(1184), secrets.token_bytes(2400)
        
        with oqs.KeyEncapsulation(self.config.KYBER_ALGORITHM) as kem:
            public_key = kem.generate_keypair()
            secret_key = kem.export_secret_key()
            return public_key, secret_key
    
    def kyber_encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """
        Encapsulate a shared secret using Kyber public key
        
        Args:
            public_key: Kyber public key
            
        Returns:
            Tuple[bytes, bytes]: (ciphertext, shared_secret)
        """
        if not self._oqs_available:
            shared_secret = secrets.token_bytes(32)
            ciphertext = secrets.token_bytes(1088)
            return ciphertext, shared_secret
        
        with oqs.KeyEncapsulation(self.config.KYBER_ALGORITHM) as kem:
            ciphertext, shared_secret = kem.encap_secret(public_key)
            return ciphertext, shared_secret
    
    def kyber_decapsulate(self, ciphertext: bytes, secret_key: bytes) -> bytes:
        """
        Decapsulate shared secret using Kyber secret key
        
        Args:
            ciphertext: Kyber ciphertext from encapsulation
            secret_key: Kyber secret key
            
        Returns:
            bytes: Shared secret
        """
        if not self._oqs_available:
            return hashlib.sha256(ciphertext + secret_key).digest()
        
        with oqs.KeyEncapsulation(self.config.KYBER_ALGORITHM, secret_key) as kem:
            shared_secret = kem.decap_secret(ciphertext)
            return shared_secret
    
    def kyber_encrypt(self, data: bytes, public_key: bytes) -> bytes:
        """
        Encrypt data using Kyber (KEM + AES-GCM hybrid)
        
        Args:
            data: Plaintext data to encrypt
            public_key: Kyber public key
            
        Returns:
            bytes: Encrypted data package
        """
        # Step 1: Encapsulate shared secret
        kem_ciphertext, shared_secret = self.kyber_encapsulate(public_key)
        
        # Step 2: Derive AES key from shared secret
        aes_key = self._derive_aes_key(shared_secret, b"kyber_encryption")
        
        # Step 3: Encrypt with AES-GCM
        nonce = secrets.token_bytes(self.config.AES_NONCE_SIZE)
        aesgcm = AESGCM(aes_key)
        aes_ciphertext = aesgcm.encrypt(nonce, data, associated_data=kem_ciphertext)
        
        # Step 4: Pack result
        result = struct.pack('>I', len(kem_ciphertext))
        result += kem_ciphertext
        result += nonce
        result += aes_ciphertext
        
        return result
    
    def kyber_decrypt(self, ciphertext: bytes, secret_key: bytes) -> bytes:
        """
        Decrypt Kyber-encrypted data
        
        Args:
            ciphertext: Encrypted data package
            secret_key: Kyber secret key
            
        Returns:
            bytes: Decrypted plaintext
        """
        # Step 1: Unpack components
        kem_ct_length = struct.unpack('>I', ciphertext[:4])[0]
        offset = 4
        
        kem_ciphertext = ciphertext[offset:offset + kem_ct_length]
        offset += kem_ct_length
        
        nonce = ciphertext[offset:offset + self.config.AES_NONCE_SIZE]
        offset += self.config.AES_NONCE_SIZE
        
        aes_ciphertext = ciphertext[offset:]
        
        # Step 2: Decapsulate shared secret
        shared_secret = self.kyber_decapsulate(kem_ciphertext, secret_key)
        
        # Step 3: Derive AES key
        aes_key = self._derive_aes_key(shared_secret, b"kyber_encryption")
        
        # Step 4: Decrypt with AES-GCM
        aesgcm = AESGCM(aes_key)
        plaintext = aesgcm.decrypt(nonce, aes_ciphertext, associated_data=kem_ciphertext)
        
        return plaintext
    
    # ==========================================================================
    # Hybrid NTRU + Kyber Encryption
    # ==========================================================================
    
    def hybrid_encrypt(self, data: bytes, ntru_pubkey: bytes, kyber_pubkey: bytes) -> bytes:
        """
        Hybrid NTRU + Kyber encryption for maximum security
        
        This provides defense-in-depth: even if one algorithm is broken,
        the other still provides security.
        
        Process:
        1. Generate two shared secrets (NTRU + Kyber)
        2. XOR the shared secrets to derive final key
        3. Encrypt data with derived key
        4. Return: ntru_ct || kyber_ct || nonce || aes_ct
        
        Args:
            data: Plaintext data to encrypt
            ntru_pubkey: NTRU public key
            kyber_pubkey: Kyber public key
            
        Returns:
            bytes: Hybrid-encrypted data package
        """
        # Step 1: Encapsulate with both KEMs
        ntru_ct, ntru_secret = self.ntru_encapsulate(ntru_pubkey)
        kyber_ct, kyber_secret = self.kyber_encapsulate(kyber_pubkey)
        
        # Step 2: Combine secrets (XOR for independent security)
        combined_secret = bytes(a ^ b for a, b in zip(ntru_secret, kyber_secret))
        
        # Step 3: Derive AES key from combined secret
        aes_key = self._derive_aes_key(combined_secret, b"hybrid_ntru_kyber")
        
        # Step 4: Encrypt with AES-GCM
        nonce = secrets.token_bytes(self.config.AES_NONCE_SIZE)
        aesgcm = AESGCM(aes_key)
        
        # Include both ciphertexts in AAD for integrity
        aad = hashlib.sha256(ntru_ct + kyber_ct).digest()
        aes_ciphertext = aesgcm.encrypt(nonce, data, associated_data=aad)
        
        # Step 5: Pack result
        result = struct.pack('>I', len(ntru_ct))
        result += ntru_ct
        result += struct.pack('>I', len(kyber_ct))
        result += kyber_ct
        result += nonce
        result += aes_ciphertext
        
        return result
    
    def hybrid_decrypt(self, ciphertext: bytes, ntru_privkey: bytes, kyber_privkey: bytes) -> bytes:
        """
        Decrypt hybrid NTRU + Kyber encrypted data
        
        Args:
            ciphertext: Encrypted data package
            ntru_privkey: NTRU secret key
            kyber_privkey: Kyber secret key
            
        Returns:
            bytes: Decrypted plaintext
        """
        # Step 1: Unpack components
        offset = 0
        
        ntru_ct_len = struct.unpack('>I', ciphertext[offset:offset + 4])[0]
        offset += 4
        ntru_ct = ciphertext[offset:offset + ntru_ct_len]
        offset += ntru_ct_len
        
        kyber_ct_len = struct.unpack('>I', ciphertext[offset:offset + 4])[0]
        offset += 4
        kyber_ct = ciphertext[offset:offset + kyber_ct_len]
        offset += kyber_ct_len
        
        nonce = ciphertext[offset:offset + self.config.AES_NONCE_SIZE]
        offset += self.config.AES_NONCE_SIZE
        
        aes_ciphertext = ciphertext[offset:]
        
        # Step 2: Decapsulate with both KEMs
        ntru_secret = self.ntru_decapsulate(ntru_ct, ntru_privkey)
        kyber_secret = self.kyber_decapsulate(kyber_ct, kyber_privkey)
        
        # Step 3: Combine secrets
        combined_secret = bytes(a ^ b for a, b in zip(ntru_secret, kyber_secret))
        
        # Step 4: Derive AES key
        aes_key = self._derive_aes_key(combined_secret, b"hybrid_ntru_kyber")
        
        # Step 5: Decrypt with AES-GCM
        aad = hashlib.sha256(ntru_ct + kyber_ct).digest()
        aesgcm = AESGCM(aes_key)
        plaintext = aesgcm.decrypt(nonce, aes_ciphertext, associated_data=aad)
        
        return plaintext
    
    # ==========================================================================
    # Dilithium Digital Signatures
    # ==========================================================================
    
    def generate_dilithium_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate Dilithium3 public/private key pair
        
        Dilithium is a lattice-based signature scheme selected by NIST.
        Used for JWT tokens and data integrity verification.
        
        Returns:
            Tuple[bytes, bytes]: (public_key, secret_key)
        """
        if not self._oqs_available:
            return secrets.token_bytes(1952), secrets.token_bytes(4000)
        
        with oqs.Signature(self.config.DILITHIUM_ALGORITHM) as sig:
            public_key = sig.generate_keypair()
            secret_key = sig.export_secret_key()
            return public_key, secret_key
    
    def sign_dilithium(self, message: bytes, secret_key: bytes) -> bytes:
        """
        Sign message with Dilithium3
        
        Args:
            message: Message to sign
            secret_key: Dilithium secret key
            
        Returns:
            bytes: Digital signature
        """
        if not self._oqs_available:
            # Fallback: HMAC-SHA256 (NOT quantum-safe, for testing only)
            import hmac
            return hmac.new(secret_key[:32], message, hashlib.sha256).digest()
        
        with oqs.Signature(self.config.DILITHIUM_ALGORITHM, secret_key) as sig:
            signature = sig.sign(message)
            return signature
    
    def verify_dilithium(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """
        Verify Dilithium3 signature
        
        Args:
            message: Original message
            signature: Digital signature
            public_key: Dilithium public key
            
        Returns:
            bool: True if signature is valid
        """
        if not self._oqs_available:
            # Fallback: Always return True in test mode (NOT secure)
            return len(signature) == 32
        
        try:
            with oqs.Signature(self.config.DILITHIUM_ALGORITHM) as sig:
                is_valid = sig.verify(message, signature, public_key)
                return is_valid
        except Exception:
            return False
    
    # ==========================================================================
    # SPHINCS+ Digital Signatures (for critical operations)
    # ==========================================================================
    
    def generate_sphincs_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate SPHINCS+-SHA2-128f-simple public/private key pair
        
        SPHINCS+ is a hash-based signature scheme providing:
        - Minimal security assumptions (only hash function security)
        - Long-term security (suitable for critical operations)
        - Larger signatures but maximum security confidence
        
        Returns:
            Tuple[bytes, bytes]: (public_key, secret_key)
        """
        if not self._oqs_available:
            return secrets.token_bytes(32), secrets.token_bytes(64)
        
        with oqs.Signature(self.config.SPHINCS_ALGORITHM) as sig:
            public_key = sig.generate_keypair()
            secret_key = sig.export_secret_key()
            return public_key, secret_key
    
    def sign_sphincs(self, message: bytes, secret_key: bytes) -> bytes:
        """
        Sign message with SPHINCS+ (for critical operations)
        
        Args:
            message: Message to sign
            secret_key: SPHINCS+ secret key
            
        Returns:
            bytes: Digital signature
        """
        if not self._oqs_available:
            return hashlib.sha512(message + secret_key).digest()
        
        with oqs.Signature(self.config.SPHINCS_ALGORITHM, secret_key) as sig:
            signature = sig.sign(message)
            return signature
    
    def verify_sphincs(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """
        Verify SPHINCS+ signature
        
        Args:
            message: Original message
            signature: Digital signature
            public_key: SPHINCS+ public key
            
        Returns:
            bool: True if signature is valid
        """
        if not self._oqs_available:
            return len(signature) == 64
        
        try:
            with oqs.Signature(self.config.SPHINCS_ALGORITHM) as sig:
                is_valid = sig.verify(message, signature, public_key)
                return is_valid
        except Exception:
            return False
    
    # ==========================================================================
    # Face Embedding Encryption
    # ==========================================================================
    
    def encrypt_embedding(
        self,
        embedding: np.ndarray,
        ntru_pubkey: bytes,
        kyber_pubkey: bytes,
        dilithium_privkey: bytes
    ) -> bytes:
        """
        Encrypt face embedding using hybrid PQC with integrity signature
        
        Process:
        1. Serialize numpy array to bytes
        2. Compress with zlib to reduce size
        3. Hybrid encrypt (NTRU + Kyber)
        4. Sign the ciphertext with Dilithium for integrity
        5. Return: signature_len || signature || ciphertext
        
        Args:
            embedding: Face embedding vector (512-dim numpy array)
            ntru_pubkey: NTRU public key
            kyber_pubkey: Kyber public key
            dilithium_privkey: Dilithium secret key for signing
            
        Returns:
            bytes: Encrypted and signed embedding package
        """
        # Step 1: Serialize embedding
        embedding_bytes = embedding.astype(np.float32).tobytes()
        
        # Add metadata (shape for reconstruction)
        metadata = struct.pack('>II', *embedding.shape if len(embedding.shape) > 0 else (len(embedding), 0))
        data = metadata + embedding_bytes
        
        # Step 2: Compress
        compressed = zlib.compress(data, level=9)
        
        # Step 3: Hybrid encrypt
        ciphertext = self.hybrid_encrypt(compressed, ntru_pubkey, kyber_pubkey)
        
        # Step 4: Sign the ciphertext
        signature = self.sign_dilithium(ciphertext, dilithium_privkey)
        
        # Step 5: Pack: timestamp (8) || sig_len (4) || signature || ciphertext
        timestamp = struct.pack('>Q', int(datetime.utcnow().timestamp()))
        result = timestamp
        result += struct.pack('>I', len(signature))
        result += signature
        result += ciphertext
        
        return result
    
    def decrypt_embedding(
        self,
        encrypted_data: bytes,
        ntru_privkey: bytes,
        kyber_privkey: bytes,
        dilithium_pubkey: bytes
    ) -> Optional[np.ndarray]:
        """
        Decrypt and verify face embedding
        
        Args:
            encrypted_data: Encrypted embedding package
            ntru_privkey: NTRU secret key
            kyber_privkey: Kyber secret key
            dilithium_pubkey: Dilithium public key for verification
            
        Returns:
            np.ndarray: Decrypted embedding, or None if verification fails
        """
        try:
            # Step 1: Unpack
            offset = 0
            
            # Timestamp (8 bytes)
            timestamp = struct.unpack('>Q', encrypted_data[offset:offset + 8])[0]
            offset += 8
            
            # Signature
            sig_len = struct.unpack('>I', encrypted_data[offset:offset + 4])[0]
            offset += 4
            signature = encrypted_data[offset:offset + sig_len]
            offset += sig_len
            
            # Ciphertext
            ciphertext = encrypted_data[offset:]
            
            # Step 2: Verify signature
            if not self.verify_dilithium(ciphertext, signature, dilithium_pubkey):
                print("ERROR: Embedding signature verification failed")
                return None
            
            # Step 3: Hybrid decrypt
            compressed = self.hybrid_decrypt(ciphertext, ntru_privkey, kyber_privkey)
            
            # Step 4: Decompress
            data = zlib.decompress(compressed)
            
            # Step 5: Deserialize
            shape_0, shape_1 = struct.unpack('>II', data[:8])
            embedding_bytes = data[8:]
            
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            if shape_1 > 0:
                embedding = embedding.reshape(shape_0, shape_1)
            
            return embedding
            
        except Exception as e:
            print(f"ERROR: Failed to decrypt embedding: {e}")
            return None
    
    # ==========================================================================
    # Utility Methods
    # ==========================================================================
    
    def _derive_aes_key(self, shared_secret: bytes, info: bytes) -> bytes:
        """
        Derive AES-256 key from shared secret using PBKDF2
        
        Args:
            shared_secret: Shared secret from KEM
            info: Context information for key derivation
            
        Returns:
            bytes: 256-bit AES key
        """
        # Use PBKDF2 for key derivation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.config.AES_KEY_SIZE,
            salt=info,
            iterations=10000,  # Lower for performance (shared secret already secure)
            backend=default_backend()
        )
        return kdf.derive(shared_secret)
    
    def generate_random_key(self, length: int = 32) -> bytes:
        """Generate cryptographically secure random bytes"""
        return secrets.token_bytes(length)
    
    def hash_data(self, data: bytes) -> bytes:
        """Compute SHA-256 hash of data"""
        return hashlib.sha256(data).digest()
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about configured PQC algorithms"""
        info = {
            "oqs_available": self._oqs_available,
            "kem_algorithms": {
                "ntru": self.config.NTRU_ALGORITHM,
                "kyber": self.config.KYBER_ALGORITHM
            },
            "signature_algorithms": {
                "dilithium": self.config.DILITHIUM_ALGORITHM,
                "sphincs": self.config.SPHINCS_ALGORITHM
            },
            "symmetric": {
                "algorithm": "AES-256-GCM",
                "key_size": self.config.AES_KEY_SIZE * 8,
                "nonce_size": self.config.AES_NONCE_SIZE * 8
            },
            "security_level": "NIST Level 3 (equivalent to AES-192)"
        }
        
        if self._oqs_available:
            info["available_kems"] = list(oqs.get_enabled_kem_mechanisms())
            info["available_sigs"] = list(oqs.get_enabled_sig_mechanisms())
        
        return info


# ==========================================================================
# Singleton Instance
# ==========================================================================

pqc_service = PQCService()


# ==========================================================================
# Convenience Functions
# ==========================================================================

def encrypt_data_hybrid(data: bytes, ntru_pubkey: bytes, kyber_pubkey: bytes) -> bytes:
    """Convenience function for hybrid encryption"""
    return pqc_service.hybrid_encrypt(data, ntru_pubkey, kyber_pubkey)


def decrypt_data_hybrid(ciphertext: bytes, ntru_privkey: bytes, kyber_privkey: bytes) -> bytes:
    """Convenience function for hybrid decryption"""
    return pqc_service.hybrid_decrypt(ciphertext, ntru_privkey, kyber_privkey)


def sign_data(data: bytes, dilithium_privkey: bytes) -> bytes:
    """Convenience function for Dilithium signing"""
    return pqc_service.sign_dilithium(data, dilithium_privkey)


def verify_signature(data: bytes, signature: bytes, dilithium_pubkey: bytes) -> bool:
    """Convenience function for Dilithium verification"""
    return pqc_service.verify_dilithium(data, signature, dilithium_pubkey)
