"""
PQC Test Suite
Comprehensive tests for Post-Quantum Cryptography implementation

Tests cover:
- NTRU encryption/decryption
- Kyber encryption/decryption
- Hybrid NTRU + Kyber encryption
- Dilithium signatures
- SPHINCS+ signatures
- Face embedding encryption
- JWT token operations
- Key management
- Migration functionality

Run with: pytest tests/test_pqc.py -v
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import secrets
from datetime import timedelta
from pathlib import Path
import shutil


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def pqc_service():
    """Initialize PQC service"""
    from app.services.pqc_service import pqc_service as service
    return service


@pytest.fixture(scope="module")
def key_manager():
    """Initialize key manager with test directory"""
    from app.services.pqc_key_manager import PQCKeyManager
    
    test_keys_dir = "data/test_pqc_keys"
    manager = PQCKeyManager(keys_dir=test_keys_dir)
    
    yield manager
    
    # Cleanup after tests
    if Path(test_keys_dir).exists():
        shutil.rmtree(test_keys_dir)


@pytest.fixture(scope="module")
def hybrid_service():
    """Initialize hybrid crypto service"""
    from app.services.hybrid_crypto_service import hybrid_crypto_service as service
    return service


@pytest.fixture(scope="module")
def jwt_service():
    """Initialize JWT service"""
    from app.services.pqc_jwt_service import pqc_jwt_service as service
    return service


@pytest.fixture
def test_user_id():
    """Generate unique test user ID"""
    return 10000 + secrets.randbelow(89999)


@pytest.fixture
def test_embedding():
    """Generate random face embedding"""
    return np.random.rand(512).astype(np.float32)


@pytest.fixture
def test_data():
    """Generate random test data"""
    return secrets.token_bytes(1024)


# =============================================================================
# NTRU Tests
# =============================================================================

class TestNTRU:
    """Tests for NTRU key encapsulation"""
    
    def test_ntru_keypair_generation(self, pqc_service):
        """Test NTRU key pair generation"""
        public_key, private_key = pqc_service.generate_ntru_keypair()
        
        assert public_key is not None
        assert private_key is not None
        assert len(public_key) > 0
        assert len(private_key) > 0
        assert public_key != private_key
    
    def test_ntru_encapsulation(self, pqc_service):
        """Test NTRU encapsulation/decapsulation"""
        public_key, private_key = pqc_service.generate_ntru_keypair()
        
        ciphertext, shared_secret = pqc_service.ntru_encapsulate(public_key)
        decapsulated_secret = pqc_service.ntru_decapsulate(ciphertext, private_key)
        
        assert shared_secret == decapsulated_secret
    
    def test_ntru_encrypt_decrypt(self, pqc_service, test_data):
        """Test NTRU encrypt/decrypt cycle"""
        public_key, private_key = pqc_service.generate_ntru_keypair()
        
        ciphertext = pqc_service.ntru_encrypt(test_data, public_key)
        decrypted = pqc_service.ntru_decrypt(ciphertext, private_key)
        
        assert decrypted == test_data
    
    def test_ntru_different_keys_fail(self, pqc_service, test_data):
        """Test that decryption with wrong key fails"""
        public_key1, private_key1 = pqc_service.generate_ntru_keypair()
        public_key2, private_key2 = pqc_service.generate_ntru_keypair()
        
        ciphertext = pqc_service.ntru_encrypt(test_data, public_key1)
        
        # Decryption with wrong key should fail
        with pytest.raises(Exception):
            pqc_service.ntru_decrypt(ciphertext, private_key2)


# =============================================================================
# Kyber Tests
# =============================================================================

class TestKyber:
    """Tests for Kyber key encapsulation"""
    
    def test_kyber_keypair_generation(self, pqc_service):
        """Test Kyber key pair generation"""
        public_key, private_key = pqc_service.generate_kyber_keypair()
        
        assert public_key is not None
        assert private_key is not None
        assert len(public_key) > 0
        assert len(private_key) > 0
    
    def test_kyber_encapsulation(self, pqc_service):
        """Test Kyber encapsulation/decapsulation"""
        public_key, private_key = pqc_service.generate_kyber_keypair()
        
        ciphertext, shared_secret = pqc_service.kyber_encapsulate(public_key)
        decapsulated_secret = pqc_service.kyber_decapsulate(ciphertext, private_key)
        
        assert shared_secret == decapsulated_secret
    
    def test_kyber_encrypt_decrypt(self, pqc_service, test_data):
        """Test Kyber encrypt/decrypt cycle"""
        public_key, private_key = pqc_service.generate_kyber_keypair()
        
        ciphertext = pqc_service.kyber_encrypt(test_data, public_key)
        decrypted = pqc_service.kyber_decrypt(ciphertext, private_key)
        
        assert decrypted == test_data


# =============================================================================
# Hybrid Encryption Tests
# =============================================================================

class TestHybridEncryption:
    """Tests for hybrid NTRU + Kyber encryption"""
    
    def test_hybrid_encrypt_decrypt(self, pqc_service, test_data):
        """Test hybrid encrypt/decrypt cycle"""
        ntru_pub, ntru_priv = pqc_service.generate_ntru_keypair()
        kyber_pub, kyber_priv = pqc_service.generate_kyber_keypair()
        
        ciphertext = pqc_service.hybrid_encrypt(test_data, ntru_pub, kyber_pub)
        decrypted = pqc_service.hybrid_decrypt(ciphertext, ntru_priv, kyber_priv)
        
        assert decrypted == test_data
    
    def test_hybrid_encryption_different_sizes(self, pqc_service):
        """Test hybrid encryption with different data sizes"""
        ntru_pub, ntru_priv = pqc_service.generate_ntru_keypair()
        kyber_pub, kyber_priv = pqc_service.generate_kyber_keypair()
        
        test_sizes = [1, 16, 256, 1024, 4096, 16384]
        
        for size in test_sizes:
            data = secrets.token_bytes(size)
            ciphertext = pqc_service.hybrid_encrypt(data, ntru_pub, kyber_pub)
            decrypted = pqc_service.hybrid_decrypt(ciphertext, ntru_priv, kyber_priv)
            
            assert decrypted == data, f"Failed for size {size}"
    
    def test_hybrid_service_encrypt_decrypt(self, hybrid_service, test_data):
        """Test hybrid service encrypt/decrypt"""
        from app.services.pqc_service import pqc_service
        
        ntru_pub, ntru_priv = pqc_service.generate_ntru_keypair()
        kyber_pub, kyber_priv = pqc_service.generate_kyber_keypair()
        
        encrypted = hybrid_service.encrypt(test_data, ntru_pub, kyber_pub)
        decrypted = hybrid_service.decrypt(encrypted, ntru_priv, kyber_priv)
        
        assert decrypted == test_data


# =============================================================================
# Dilithium Signature Tests
# =============================================================================

class TestDilithium:
    """Tests for Dilithium digital signatures"""
    
    def test_dilithium_keypair_generation(self, pqc_service):
        """Test Dilithium key pair generation"""
        public_key, private_key = pqc_service.generate_dilithium_keypair()
        
        assert public_key is not None
        assert private_key is not None
        assert len(public_key) > 0
        assert len(private_key) > 0
    
    def test_dilithium_sign_verify(self, pqc_service):
        """Test Dilithium sign/verify cycle"""
        public_key, private_key = pqc_service.generate_dilithium_keypair()
        message = b"Test message for Dilithium signature"
        
        signature = pqc_service.sign_dilithium(message, private_key)
        is_valid = pqc_service.verify_dilithium(message, signature, public_key)
        
        assert is_valid is True
    
    def test_dilithium_wrong_message_fails(self, pqc_service):
        """Test that verification fails with wrong message"""
        public_key, private_key = pqc_service.generate_dilithium_keypair()
        message = b"Original message"
        wrong_message = b"Wrong message"
        
        signature = pqc_service.sign_dilithium(message, private_key)
        is_valid = pqc_service.verify_dilithium(wrong_message, signature, public_key)
        
        assert is_valid is False
    
    def test_dilithium_wrong_key_fails(self, pqc_service):
        """Test that verification fails with wrong public key"""
        public_key1, private_key1 = pqc_service.generate_dilithium_keypair()
        public_key2, private_key2 = pqc_service.generate_dilithium_keypair()
        message = b"Test message"
        
        signature = pqc_service.sign_dilithium(message, private_key1)
        is_valid = pqc_service.verify_dilithium(message, signature, public_key2)
        
        assert is_valid is False


# =============================================================================
# SPHINCS+ Tests
# =============================================================================

class TestSPHINCS:
    """Tests for SPHINCS+ digital signatures"""
    
    def test_sphincs_keypair_generation(self, pqc_service):
        """Test SPHINCS+ key pair generation"""
        public_key, private_key = pqc_service.generate_sphincs_keypair()
        
        assert public_key is not None
        assert private_key is not None
    
    def test_sphincs_sign_verify(self, pqc_service):
        """Test SPHINCS+ sign/verify cycle"""
        public_key, private_key = pqc_service.generate_sphincs_keypair()
        message = b"Critical operation requiring SPHINCS+ signature"
        
        signature = pqc_service.sign_sphincs(message, private_key)
        is_valid = pqc_service.verify_sphincs(message, signature, public_key)
        
        assert is_valid is True


# =============================================================================
# Face Embedding Encryption Tests
# =============================================================================

class TestEmbeddingEncryption:
    """Tests for face embedding encryption"""
    
    def test_embedding_encrypt_decrypt(self, pqc_service, key_manager, test_user_id, test_embedding):
        """Test embedding encrypt/decrypt cycle"""
        # Generate keys
        keys = key_manager.generate_user_keys(test_user_id)
        key_manager.save_user_keys(test_user_id, keys)
        
        # Encrypt
        encrypted = pqc_service.encrypt_embedding(
            test_embedding,
            keys.ntru_public,
            keys.kyber_public,
            keys.dilithium_private
        )
        
        # Decrypt
        decrypted = pqc_service.decrypt_embedding(
            encrypted,
            keys.ntru_private,
            keys.kyber_private,
            keys.dilithium_public
        )
        
        assert decrypted is not None
        assert np.allclose(test_embedding, decrypted)
        
        # Cleanup
        key_manager.delete_user_keys(test_user_id)
    
    def test_hybrid_service_embedding(self, hybrid_service, key_manager, test_user_id, test_embedding):
        """Test embedding encryption through hybrid service"""
        # Generate keys
        keys = key_manager.generate_user_keys(test_user_id)
        key_manager.save_user_keys(test_user_id, keys)
        
        # Use hybrid service (which uses the key manager internally)
        # We need to use the same key manager
        from app.services import pqc_key_manager as global_key_manager
        
        # Copy keys to global manager
        global_key_manager.pqc_key_manager.save_user_keys(test_user_id, keys)
        
        # Encrypt
        encrypted = hybrid_service.encrypt_embedding(test_embedding, test_user_id, sign=True)
        
        # Decrypt
        decrypted = hybrid_service.decrypt_embedding(encrypted, test_user_id)
        
        assert decrypted is not None
        assert np.allclose(test_embedding, decrypted)
        
        # Cleanup
        key_manager.delete_user_keys(test_user_id)
        global_key_manager.pqc_key_manager.delete_user_keys(test_user_id)


# =============================================================================
# Key Management Tests
# =============================================================================

class TestKeyManagement:
    """Tests for PQC key management"""
    
    def test_generate_user_keys(self, key_manager, test_user_id):
        """Test user key generation"""
        keys = key_manager.generate_user_keys(test_user_id)
        
        assert keys.user_id == test_user_id
        assert keys.ntru_public is not None
        assert keys.ntru_private is not None
        assert keys.kyber_public is not None
        assert keys.kyber_private is not None
        assert keys.dilithium_public is not None
        assert keys.dilithium_private is not None
    
    def test_save_load_user_keys(self, key_manager, test_user_id):
        """Test saving and loading user keys"""
        # Generate and save
        original_keys = key_manager.generate_user_keys(test_user_id)
        key_manager.save_user_keys(test_user_id, original_keys)
        
        # Clear cache
        key_manager._key_cache.clear()
        
        # Load
        loaded_keys = key_manager.load_user_keys(test_user_id)
        
        assert loaded_keys is not None
        assert loaded_keys.ntru_public == original_keys.ntru_public
        assert loaded_keys.ntru_private == original_keys.ntru_private
        assert loaded_keys.kyber_public == original_keys.kyber_public
        assert loaded_keys.dilithium_public == original_keys.dilithium_public
        
        # Cleanup
        key_manager.delete_user_keys(test_user_id)
    
    def test_export_public_keys(self, key_manager, test_user_id):
        """Test exporting only public keys"""
        keys = key_manager.generate_user_keys(test_user_id)
        key_manager.save_user_keys(test_user_id, keys)
        
        public_keys = key_manager.export_public_keys(test_user_id)
        
        assert public_keys is not None
        assert "ntru_public" in public_keys
        assert "kyber_public" in public_keys
        assert "dilithium_public" in public_keys
        assert "ntru_private" not in public_keys
        
        # Cleanup
        key_manager.delete_user_keys(test_user_id)
    
    def test_user_has_keys(self, key_manager, test_user_id):
        """Test checking if user has keys"""
        assert key_manager.user_has_keys(test_user_id) is False
        
        keys = key_manager.generate_user_keys(test_user_id)
        key_manager.save_user_keys(test_user_id, keys)
        
        assert key_manager.user_has_keys(test_user_id) is True
        
        # Cleanup
        key_manager.delete_user_keys(test_user_id)
    
    def test_key_rotation(self, key_manager, test_user_id):
        """Test key rotation"""
        # Generate initial keys
        original_keys = key_manager.generate_user_keys(test_user_id)
        key_manager.save_user_keys(test_user_id, original_keys)
        
        # Rotate keys
        new_keys = key_manager.rotate_user_keys(test_user_id)
        
        assert new_keys is not None
        assert new_keys.key_version > original_keys.key_version
        assert new_keys.ntru_public != original_keys.ntru_public
        
        # Cleanup
        key_manager.delete_user_keys(test_user_id)
    
    def test_list_users_with_keys(self, key_manager):
        """Test listing users with keys"""
        user_ids = [10001, 10002, 10003]
        
        for user_id in user_ids:
            keys = key_manager.generate_user_keys(user_id)
            key_manager.save_user_keys(user_id, keys)
        
        listed_users = key_manager.list_all_users_with_keys()
        
        for user_id in user_ids:
            assert user_id in listed_users
        
        # Cleanup
        for user_id in user_ids:
            key_manager.delete_user_keys(user_id)


# =============================================================================
# JWT Tests
# =============================================================================

class TestPQCJWT:
    """Tests for PQC JWT tokens"""
    
    def test_create_verify_token(self, jwt_service, key_manager, test_user_id):
        """Test creating and verifying JWT token"""
        # Setup keys
        keys = key_manager.generate_user_keys(test_user_id)
        key_manager.save_user_keys(test_user_id, keys)
        
        # Also save to global key manager
        from app.services.pqc_key_manager import pqc_key_manager as global_km
        global_km.save_user_keys(test_user_id, keys)
        
        payload = {"user_id": test_user_id, "username": "testuser"}
        
        token = jwt_service.create_token(payload, test_user_id)
        
        assert token is not None
        assert "." in token  # JWT format
        
        verified_payload = jwt_service.verify_token(token)
        
        assert verified_payload is not None
        assert int(verified_payload["sub"]) == test_user_id
        
        # Cleanup
        key_manager.delete_user_keys(test_user_id)
        global_km.delete_user_keys(test_user_id)
    
    def test_token_expiration(self, jwt_service, key_manager, test_user_id):
        """Test token expiration"""
        # Setup keys
        keys = key_manager.generate_user_keys(test_user_id)
        key_manager.save_user_keys(test_user_id, keys)
        
        from app.services.pqc_key_manager import pqc_key_manager as global_km
        global_km.save_user_keys(test_user_id, keys)
        
        # Create token with very short expiration (already expired)
        token = jwt_service.create_token(
            {"user_id": test_user_id},
            test_user_id,
            expires_delta=timedelta(seconds=-10)  # Already expired
        )
        
        # Should fail verification
        verified = jwt_service.verify_token(token)
        assert verified is None
        
        # Cleanup
        key_manager.delete_user_keys(test_user_id)
        global_km.delete_user_keys(test_user_id)
    
    def test_encrypted_token(self, jwt_service, key_manager, test_user_id):
        """Test encrypted JWT token"""
        # Setup keys
        keys = key_manager.generate_user_keys(test_user_id)
        key_manager.save_user_keys(test_user_id, keys)
        
        from app.services.pqc_key_manager import pqc_key_manager as global_km
        global_km.save_user_keys(test_user_id, keys)
        
        sensitive_payload = {
            "user_id": test_user_id,
            "secret_data": "sensitive information"
        }
        
        encrypted_token = jwt_service.create_encrypted_token(
            sensitive_payload,
            test_user_id
        )
        
        # Verify and decrypt
        decrypted_payload = jwt_service.verify_encrypted_token(encrypted_token)
        
        assert decrypted_payload is not None
        assert decrypted_payload["secret_data"] == "sensitive information"
        
        # Cleanup
        key_manager.delete_user_keys(test_user_id)
        global_km.delete_user_keys(test_user_id)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete PQC workflows"""
    
    def test_full_user_workflow(self, pqc_service, key_manager, hybrid_service, jwt_service, test_user_id):
        """Test complete user workflow: registration -> login -> authentication"""
        # 1. Registration: Generate keys
        keys = key_manager.generate_user_keys(test_user_id, include_sphincs=True)
        key_manager.save_user_keys(test_user_id, keys)
        
        from app.services.pqc_key_manager import pqc_key_manager as global_km
        global_km.save_user_keys(test_user_id, keys)
        
        # 2. Registration: Encrypt face embeddings
        embeddings = [np.random.rand(512).astype(np.float32) for _ in range(3)]
        encrypted_embeddings = []
        
        for emb in embeddings:
            encrypted = pqc_service.encrypt_embedding(
                emb,
                keys.ntru_public,
                keys.kyber_public,
                keys.dilithium_private
            )
            encrypted_embeddings.append(encrypted)
        
        assert len(encrypted_embeddings) == 3
        
        # 3. Login: Decrypt embedding
        decrypted = pqc_service.decrypt_embedding(
            encrypted_embeddings[0],
            keys.ntru_private,
            keys.kyber_private,
            keys.dilithium_public
        )
        
        assert np.allclose(embeddings[0], decrypted)
        
        # 4. Login: Create JWT
        token = jwt_service.create_token(
            {"user_id": test_user_id, "action": "login"},
            test_user_id
        )
        
        assert token is not None
        
        # 5. Authentication: Verify JWT
        payload = jwt_service.verify_token(token)
        
        assert payload is not None
        assert int(payload["sub"]) == test_user_id
        
        # Cleanup
        key_manager.delete_user_keys(test_user_id)
        global_km.delete_user_keys(test_user_id)
    
    def test_100_users_key_generation(self, key_manager):
        """Test generating keys for 100 users"""
        user_ids = list(range(20001, 20101))  # 100 users
        
        for user_id in user_ids:
            keys = key_manager.generate_user_keys(user_id)
            success = key_manager.save_user_keys(user_id, keys)
            assert success, f"Failed to save keys for user {user_id}"
        
        # Verify all users have keys
        listed = key_manager.list_all_users_with_keys()
        for user_id in user_ids:
            assert user_id in listed, f"User {user_id} not in list"
        
        # Cleanup
        for user_id in user_ids:
            key_manager.delete_user_keys(user_id)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
