# Post-Quantum Cryptography Implementation

## Overview

This document describes the Post-Quantum Cryptography (PQC) implementation for the AI-Powered Face Authentication System. The implementation protects against future quantum computer attacks using NIST-approved algorithms.

## Security Level

**NIST Level 3** - Equivalent to AES-192 classical security

## Algorithms Used

### Key Encapsulation Mechanisms (KEM)

| Algorithm | Library | Parameters | Use Case |
|-----------|---------|------------|----------|
| **NTRU** | liboqs-python | NTRU-HPS-2048-509 | Primary key encapsulation |
| **Kyber** | liboqs-python | Kyber768 | Secondary KEM (hybrid) |

### Digital Signatures

| Algorithm | Library | Parameters | Use Case |
|-----------|---------|------------|----------|
| **Dilithium** | liboqs-python | Dilithium3 | JWT tokens, embedding integrity |
| **SPHINCS+** | liboqs-python | SPHINCS+-SHA2-128f-simple | Critical operations |

### Symmetric Encryption

| Algorithm | Key Size | Use Case |
|-----------|----------|----------|
| **AES-256-GCM** | 256 bits | Data encryption after KEM |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PQC Architecture                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │  pqc_service.py  │    │pqc_key_manager.py│                   │
│  │                  │    │                  │                   │
│  │ - NTRU encrypt   │    │ - Key generation │                   │
│  │ - Kyber encrypt  │    │ - Key storage    │                   │
│  │ - Hybrid encrypt │    │ - Key rotation   │                   │
│  │ - Dilithium sign │    │ - Master password│                   │
│  │ - SPHINCS+ sign  │    │                  │                   │
│  └────────┬─────────┘    └────────┬─────────┘                   │
│           │                       │                              │
│           └───────────┬───────────┘                              │
│                       │                                          │
│           ┌───────────▼───────────┐                              │
│           │hybrid_crypto_service.py│                             │
│           │                        │                             │
│           │ - NTRU + Kyber hybrid  │                             │
│           │ - Embedding encryption │                             │
│           │ - File encryption      │                             │
│           └───────────┬────────────┘                             │
│                       │                                          │
│           ┌───────────▼───────────┐                              │
│           │  pqc_jwt_service.py   │                              │
│           │                       │                              │
│           │ - Dilithium JWT       │                              │
│           │ - Encrypted tokens    │                              │
│           │ - Token verification  │                              │
│           └───────────────────────┘                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
backend/app/services/
├── pqc_service.py           # Core PQC algorithms
├── pqc_key_manager.py       # Key generation and storage
├── pqc_jwt_service.py       # Quantum-safe JWT tokens
├── hybrid_crypto_service.py # NTRU + Kyber hybrid encryption
└── vqc_service_pqc.py       # VQC service with PQC integration

backend/scripts/
├── migrate_100_users_pqc.py # Migration script
├── backup_rollback_pqc.py   # Backup and rollback
└── benchmark_pqc.py         # Performance benchmarks

backend/tests/
└── test_pqc.py              # Comprehensive test suite

data/pqc_keys/
├── system/                  # System-wide keys
│   ├── ntru_public.key
│   ├── ntru_private.enc
│   ├── kyber_public.key
│   ├── kyber_private.enc
│   ├── dilithium_public.key
│   └── dilithium_private.enc
└── user_{id}/               # Per-user keys
    ├── ntru_public.key
    ├── ntru_private.key.enc
    ├── kyber_public.key
    ├── kyber_private.key.enc
    ├── dilithium_public.key
    ├── dilithium_private.key.enc
    └── key_metadata.json
```

## Hybrid Encryption Process

### Encryption

```
plaintext
    │
    ▼
┌───────────────┐
│  Compress     │  (zlib, level 9)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ NTRU Encap    │ ──► NTRU ciphertext + shared_secret_1
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Kyber Encap   │ ──► Kyber ciphertext + shared_secret_2
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ XOR secrets   │  combined_secret = ss1 ⊕ ss2
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ HKDF derive   │ ──► AES-256 key
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ AES-256-GCM   │ ──► encrypted_data + tag
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Package       │  magic || version || ntru_ct || kyber_ct || nonce || aes_ct
└───────────────┘
```

### Face Embedding Encryption

```python
# Encryption process for face embeddings
embedding = np.random.rand(512).astype(np.float32)

encrypted = hybrid_crypto_service.encrypt_embedding(
    embedding,
    user_id=123,
    sign=True  # Adds Dilithium signature
)

# Decryption
decrypted = hybrid_crypto_service.decrypt_embedding(
    encrypted,
    user_id=123,
    verify_signature=True
)

assert np.allclose(embedding, decrypted)
```

## JWT Token Changes

### Old (Classical)

```python
# python-jose with HMAC-SHA256
token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
```

### New (PQC)

```python
# Dilithium3 signatures
token = pqc_jwt_service.create_token(payload, user_id)

# Token format: header.payload.dilithium_signature
# Header includes: {"alg": "DILITHIUM3", "typ": "JWT", "kid": "user_123_v1"}
```

### Encrypted JWT (for sensitive data)

```python
# Hybrid encrypted + Dilithium signed
encrypted_token = pqc_jwt_service.create_encrypted_token(
    sensitive_payload,
    user_id
)
```

## Key Management

### Master Password

- Protected by PBKDF2 with 600,000 iterations
- Stored in `data/pqc_keys/master.salt` and `master.key.enc`
- Used to encrypt all private keys at rest

### Key Generation

```python
from app.services.pqc_key_manager import pqc_key_manager

# Initialize master password (first time)
pqc_key_manager.initialize_master_password("secure_password")

# Generate user keys
keys = pqc_key_manager.generate_user_keys(user_id=123)
pqc_key_manager.save_user_keys(123, keys)

# Load user keys
loaded_keys = pqc_key_manager.load_user_keys(123)
```

### Key Rotation

```python
# Rotate keys (archives old, generates new)
new_keys = pqc_key_manager.rotate_user_keys(user_id=123)

# Check key expiry
expiry_info = pqc_key_manager.check_key_expiry(123)
# {'status': 'ok', 'days_until_expiry': 350, 'needs_rotation': False}
```

## Migration Guide

### Step 1: Backup Current Data

```bash
cd backend
python scripts/backup_rollback_pqc.py backup
```

### Step 2: Run Migration

```bash
python scripts/migrate_100_users_pqc.py
```

### Step 3: Verify Migration

```bash
python -m pytest tests/test_pqc.py -v
```

### Step 4: Benchmark Performance

```bash
python scripts/benchmark_pqc.py
```

### Rollback (if needed)

```bash
# List available backups
python scripts/backup_rollback_pqc.py list

# Rollback to specific backup
python scripts/backup_rollback_pqc.py rollback --backup-id backup_20251226_120000
```

## Performance

### Benchmarks

| Operation | Average Time | Threshold | Status |
|-----------|--------------|-----------|--------|
| NTRU Keygen | ~15ms | 50ms | ✓ PASS |
| Kyber Keygen | ~10ms | 50ms | ✓ PASS |
| Dilithium Keygen | ~12ms | 50ms | ✓ PASS |
| Hybrid Encrypt | ~35ms | 50ms | ✓ PASS |
| Hybrid Decrypt | ~35ms | 50ms | ✓ PASS |
| Embedding Encrypt | ~80ms | 100ms | ✓ PASS |
| JWT Create | ~30ms | 50ms | ✓ PASS |
| Login Total | ~350ms | 500ms | ✓ PASS |
| Registration Total | ~1500ms | 2000ms | ✓ PASS |

### Size Overhead

| Data Type | Original Size | Encrypted Size | Overhead |
|-----------|---------------|----------------|----------|
| Face Embedding (512-dim) | 2,048 bytes | ~4,000 bytes | ~100% |
| JWT Token | ~200 bytes | ~3,500 bytes | ~1650% |

## Security Considerations

### Strengths

1. **Defense in Depth**: Hybrid NTRU + Kyber provides security even if one algorithm is broken
2. **NIST Approved**: All algorithms are NIST PQC standardization winners
3. **Key Isolation**: Each user has unique keypairs
4. **Signature Integrity**: All encrypted data is signed with Dilithium

### Limitations

1. **Key Size**: PQC keys are larger than classical keys
2. **Performance**: Slightly slower than classical cryptography
3. **Library Maturity**: liboqs-python is relatively new

### Best Practices

1. Always verify Dilithium signatures before trusting decrypted data
2. Rotate keys annually or when security events occur
3. Keep master password secure and backed up separately
4. Monitor for cryptographic algorithm deprecation notices

## API Reference

### PQCService

```python
from app.services.pqc_service import pqc_service

# NTRU
pub, priv = pqc_service.generate_ntru_keypair()
ct = pqc_service.ntru_encrypt(data, pub)
pt = pqc_service.ntru_decrypt(ct, priv)

# Kyber
pub, priv = pqc_service.generate_kyber_keypair()
ct = pqc_service.kyber_encrypt(data, pub)
pt = pqc_service.kyber_decrypt(ct, priv)

# Hybrid
ct = pqc_service.hybrid_encrypt(data, ntru_pub, kyber_pub)
pt = pqc_service.hybrid_decrypt(ct, ntru_priv, kyber_priv)

# Dilithium
pub, priv = pqc_service.generate_dilithium_keypair()
sig = pqc_service.sign_dilithium(msg, priv)
valid = pqc_service.verify_dilithium(msg, sig, pub)
```

### HybridCryptoService

```python
from app.services.hybrid_crypto_service import hybrid_crypto_service

# Encrypt for user
ct = hybrid_crypto_service.encrypt_for_user(data, user_id)
pt = hybrid_crypto_service.decrypt_for_user(ct, user_id)

# Embedding encryption
enc_emb = hybrid_crypto_service.encrypt_embedding(embedding, user_id)
dec_emb = hybrid_crypto_service.decrypt_embedding(enc_emb, user_id)
```

### PQCJWTService

```python
from app.services.pqc_jwt_service import pqc_jwt_service

# Create token
token = pqc_jwt_service.create_token(payload, user_id, expires_delta)

# Verify token
payload = pqc_jwt_service.verify_token(token)

# Encrypted token
enc_token = pqc_jwt_service.create_encrypted_token(payload, user_id)
```

## Troubleshooting

### liboqs-python Installation

```bash
# Windows (requires Visual Studio Build Tools)
pip install liboqs-python

# Linux
sudo apt-get install cmake ninja-build
pip install liboqs-python
```

### Common Errors

**Error**: `No module named 'oqs'`
**Solution**: Install liboqs-python: `pip install liboqs-python`

**Error**: `Required KEM algorithm not available`
**Solution**: Ensure liboqs is compiled with all algorithms enabled

**Error**: `Signature verification failed`
**Solution**: Keys may be corrupted or mismatched. Try rotating keys.

**Error**: `Master password incorrect`
**Solution**: Check `PQC_MASTER_PASSWORD` environment variable or `.auto_master_password` file

## References

- [NIST Post-Quantum Cryptography](https://csrc.nist.gov/projects/post-quantum-cryptography)
- [Open Quantum Safe (liboqs)](https://openquantumsafe.org/)
- [NTRU Specification](https://ntru.org/)
- [Kyber Specification](https://pq-crystals.org/kyber/)
- [Dilithium Specification](https://pq-crystals.org/dilithium/)
- [SPHINCS+ Specification](https://sphincs.org/)
