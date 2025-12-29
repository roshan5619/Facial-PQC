# REQUAGNIZE - AI-Powered Face Authentication System

Advanced face recognition system with **Variational Quantum Circuits (VQC)** and **Post-Quantum Cryptography (PQC)** for quantum-safe authentication.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![React](https://img.shields.io/badge/React-18.x-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- 🔐 **Face-based Authentication** - Secure login using facial biometrics
- 👁️ **Liveness Detection** - Blink detection prevents photo spoofing
- 🎓 **Quantum ML (VQC)** - Variational Quantum Circuits for face embeddings
- 🛡️ **Post-Quantum Cryptography** - NTRU + Kyber hybrid encryption (NIST Level 3)
- 🔑 **Quantum-Safe JWT** - Dilithium-signed tokens
- 📊 **Real-time Logging** - Access tracking and analytics
- ✉️ **Email Confirmation** - User verification
- ☁️ **Cloud Storage** - Cloudinary integration

## 🔒 Security Features

| Feature | Technology | Security Level |
|---------|------------|----------------|
| Key Encapsulation | NTRU + Kyber Hybrid | NIST Level 3 |
| Digital Signatures | ML-DSA-65 (Dilithium) | NIST Level 3 |
| Critical Operations | SPHINCS+-SHA2-128f | NIST Level 1 |
| Symmetric Encryption | AES-256-GCM | 256-bit |
| Face Embeddings | PQC Encrypted | Quantum-Safe |

## 🛠️ Technology Stack

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI + Python 3.11 |
| **Frontend** | React 18 + TailwindCSS |
| **Database** | PostgreSQL 15+ |
| **Quantum ML** | PyTorch + PennyLane (VQC) |
| **PQC** | liboqs 0.15.0 + liboqs-python (NTRU, Kyber, ML-DSA) |
| **Storage** | Cloudinary |
| **Deployment** | Docker + Render |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- Docker Desktop (optional)

### Backend Setup

```bash
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend/react-app
npm install
cp .env.example .env
# Edit .env with API URL
npm start
```

### Docker Setup

```bash
docker-compose up --build
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger documentation |
| POST | `/api/auth/login` | Face authentication |
| POST | `/api/auth/logout` | Logout |
| GET | `/api/auth/verify` | Verify token |
| POST | `/api/registration/register` | User registration |
| GET | `/api/admin/users` | List users |
| GET | `/api/admin/stats` | System statistics |

## 📁 Project Structure
```
REQUAGNIZE_PRODUCT/
├── backend/
│   ├── app/
│   │   ├── api/endpoints/        # API routes
│   │   │   ├── auth.py           # Authentication
│   │   │   ├── registration.py   # User registration
│   │   │   ├── admin.py          # Admin dashboard
│   │   │   └── health.py         # Health checks
│   │   ├── core/
│   │   │   ├── config.py         # Configuration
│   │   │   └── security.py       # JWT + PQC security
│   │   ├── db/
│   │   │   ├── database.py       # PostgreSQL connection
│   │   │   └── crud.py           # Database operations
│   │   ├── services/
│   │   │   ├── vqc_service.py        # Quantum ML face recognition
│   │   │   ├── pqc_service.py        # Post-Quantum Cryptography
│   │   │   ├── pqc_key_manager.py    # PQC key management
│   │   │   ├── pqc_jwt_service.py    # Dilithium JWT tokens
│   │   │   ├── hybrid_crypto_service.py # NTRU+Kyber encryption
│   │   │   ├── blink_detection.py    # Liveness detection
│   │   │   └── enhancement_service.py # Image preprocessing
│   │   └── models_orm/           # SQLAlchemy models
│   ├── ml_models/                # Trained models
│   │   ├── vqc_face_model_roi.pth
│   │   └── haarcascade_*.xml
│   ├── scripts/
│   │   ├── migrate_100_users_pqc.py  # PQC migration
│   │   ├── backup_rollback_pqc.py    # Backup/restore
│   │   ├── benchmark_pqc.py          # Performance tests
│   │   └── system_health_check.py    # Health diagnostics
│   └── tests/
│       ├── test_api.py
│       └── test_pqc.py           # PQC test suite
├── frontend/react-app/
│   ├── src/
│   │   ├── components/
│   │   │   └── BlinkCamera.jsx   # Blink detection camera
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   ├── UserLogin.jsx
│   │   │   ├── UserRegistration.jsx
│   │   │   └── AdminDashboard.jsx
│   │   └── services/api.js
│   └── package.json
├── docs/
│   ├── API_DOCS.md
│   ├── PQC_IMPLEMENTATION.md     # PQC documentation
│   ├── KNOWLEDGE_GUIDE.md        # Complete guide
│   └── DOCKER_GUIDE.md
├── docker-compose.yml
└── README.md
```

<<<<<<< Updated upstream
## 🔐 Post-Quantum Cryptography

This system implements **NIST-approved PQC algorithms** to protect against quantum computer attacks:

### Encryption Flow
```
Face Embedding → NTRU Encapsulation → Kyber Encapsulation
                         ↓                    ↓
                   shared_secret_1    shared_secret_2
                         ↓                    ↓
                    combined_secret = XOR(ss1, ss2)
                              ↓
                    AES-256-GCM Encryption
=======
## 🔐 Post-Quantum Cryptography Integration

This system implements **NIST-approved PQC algorithms** using the [liboqs](https://github.com/open-quantum-safe/liboqs) library to protect against future quantum computer attacks.

### 📦 Algorithms Used

| Algorithm | Type | Purpose | NIST Status |
|-----------|------|---------|-------------|
| **NTRU-HPS-2048-509** | KEM | Primary key encapsulation | Round 3 Finalist |
| **Kyber768** | KEM | Secondary key encapsulation | FIPS 203 (ML-KEM) |
| **ML-DSA-65** | Signature | JWT token signing | FIPS 204 (Dilithium) |
| **SPHINCS+-SHA2-128f** | Signature | Critical operations | FIPS 205 |

### 🔄 How It Works - Integration Flow

#### 1. User Registration Flow
```
┌─────────────────────────────────────────────────────────────────────────┐
│  USER REGISTRATION                                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Capture face image + blink detection (liveness check)               │
│                          ↓                                               │
│  2. Extract face embedding using VQC (Variational Quantum Circuit)       │
│                          ↓                                               │
│  3. Generate PQC key pairs for user:                                     │
│     • NTRU keypair (public + private)                                    │
│     • Kyber keypair (public + private)                                   │
│     • ML-DSA keypair (for signatures)                                    │
│                          ↓                                               │
│  4. Encrypt face embedding with hybrid encryption:                       │
│     embedding → NTRU+Kyber → AES-256-GCM → encrypted_embedding          │
│                          ↓                                               │
│  5. Store in database:                                                   │
│     • User info (name, email)                                            │
│     • Encrypted face embedding                                           │
│     • PQC public keys                                                    │
│     • PQC private keys (encrypted with master key)                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 2. Authentication Flow
```
┌─────────────────────────────────────────────────────────────────────────┐
│  USER LOGIN / AUTHENTICATION                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Capture face image + blink detection                                │
│                          ↓                                               │
│  2. Extract face embedding using VQC model                               │
│                          ↓                                               │
│  3. For each registered user:                                            │
│     a. Retrieve encrypted embedding from database                        │
│     b. Decrypt using user's PQC private keys:                            │
│        encrypted_embedding → NTRU+Kyber decap → AES-GCM decrypt          │
│     c. Compare embeddings using cosine similarity                        │
│                          ↓                                               │
│  4. If match found (similarity > threshold):                             │
│     a. Generate JWT token                                                │
│     b. Sign with ML-DSA-65 (quantum-safe signature)                      │
│     c. Return token to client                                            │
│                          ↓                                               │
│  5. Client stores JWT for subsequent API calls                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 3. Hybrid Encryption Detail
```
┌─────────────────────────────────────────────────────────────────────────┐
│  HYBRID ENCRYPTION (NTRU + Kyber)                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ENCRYPTION:                                                             │
│  ───────────                                                             │
│  plaintext_data                                                          │
│       ↓                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐                              │
│  │ NTRU Encapsulate│    │ Kyber Encapsulate│                             │
│  │ (public key)    │    │ (public key)     │                             │
│  └────────┬────────┘    └────────┬─────────┘                             │
│           ↓                      ↓                                       │
│    shared_secret_1        shared_secret_2                                │
│           ↓                      ↓                                       │
│           └──────────┬───────────┘                                       │
│                      ↓                                                   │
│            combined_key = HKDF(ss1 || ss2)                               │
│                      ↓                                                   │
│            ┌─────────────────────┐                                       │
│            │   AES-256-GCM       │                                       │
│            │   Encrypt(data)     │                                       │
│            └─────────────────────┘                                       │
│                      ↓                                                   │
│  OUTPUT: ntru_ciphertext || kyber_ciphertext || nonce || aes_ciphertext  │
│                                                                          │
│  DECRYPTION:                                                             │
│  ───────────                                                             │
│  Parse ciphertext components                                             │
│       ↓                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐                              │
│  │ NTRU Decapsulate│    │ Kyber Decapsulate│                             │
│  │ (private key)   │    │ (private key)    │                             │
│  └────────┬────────┘    └────────┬─────────┘                             │
│           ↓                      ↓                                       │
│    shared_secret_1        shared_secret_2                                │
│           ↓                      ↓                                       │
│           └──────────┬───────────┘                                       │
│                      ↓                                                   │
│            combined_key = HKDF(ss1 || ss2)                               │
│                      ↓                                                   │
│            ┌─────────────────────┐                                       │
│            │   AES-256-GCM       │                                       │
│            │   Decrypt(data)     │                                       │
│            └─────────────────────┘                                       │
│                      ↓                                                   │
│  OUTPUT: original plaintext_data                                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 🛡️ Why Hybrid Encryption?

Using **both NTRU and Kyber** provides **defense-in-depth**:

| Reason | Explanation |
|--------|-------------|
| **Algorithm Diversity** | If one algorithm is broken, the other still protects data |
| **NIST Recommendations** | NIST suggests using multiple algorithms during transition |
| **Future-Proof** | Kyber is NIST standardized; NTRU is battle-tested |
| **No Single Point of Failure** | Attacker must break BOTH algorithms |

### 🔑 Key Management Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PQC KEY HIERARCHY                                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Master Key (derived from PQC_MASTER_PASSWORD)                           │
│       │                                                                  │
│       ├── Encrypted with PBKDF2 (600,000 iterations)                     │
│       │                                                                  │
│       └── Used to encrypt/decrypt User Private Keys                      │
│                  │                                                       │
│                  ├── User 1                                              │
│                  │    ├── ntru_private.key.enc                           │
│                  │    ├── kyber_private.key.enc                          │
│                  │    └── dilithium_private.key.enc                      │
│                  │                                                       │
│                  ├── User 2                                              │
│                  │    └── ...                                            │
│                  │                                                       │
│                  └── User N                                              │
│                       └── ...                                            │
│                                                                          │
│  Storage Location: backend/data/pqc_keys/                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 🖥️ Windows Setup for liboqs

The PQC functionality requires **liboqs** (C library) + **liboqs-python** (Python wrapper).

#### Prerequisites
```powershell
# Install via winget
winget install msys2.msys2
winget install Kitware.CMake

# In MSYS2 terminal, install MinGW toolchain
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-ninja git
```

#### Build liboqs from Source
```powershell
# Clone and build
git clone https://github.com/open-quantum-safe/liboqs.git
cd liboqs
mkdir build && cd build

# Configure (use MSYS2 MinGW shell)
cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX="C:\oqs" -DBUILD_SHARED_LIBS=ON ..

# Build and install
ninja
cmake --install .
```

#### Configure Python Environment
```powershell
# Install Python wrapper
pip install liboqs-python

# Set environment variables (add to your profile)
$env:OQS_INSTALL_PATH = "C:\oqs"
$env:PATH = "C:\oqs\bin;$env:PATH"
```

#### Verify Installation
```powershell
python -c "import oqs; print('KEMs:', oqs.get_enabled_kem_mechanisms()[:3])"
# Should output: KEMs: ('Classic-McEliece-348864', 'Kyber512', 'Kyber768', ...)
```

### 🧪 Testing PQC Integration

```bash
cd backend

# Quick PQC test (all algorithms)
python scripts/quick_pqc_test.py

# Test encryption/decryption with file output
python scripts/test_encryption_output.py
# Output saved to: backend/data/test_output/decrypted_message.txt

# System health check
python scripts/system_health_check.py

# Run full PQC test suite
pytest tests/test_pqc.py -v
>>>>>>> Stashed changes
```

### JWT Token Format
```
Classical:  header.payload.HMAC_signature
<<<<<<< Updated upstream
PQC:        header.payload.Dilithium_signature
=======
PQC:        header.payload.ML-DSA-65_signature (4627 bytes)
>>>>>>> Stashed changes
```

## 📊 Performance

| Operation | Time | Threshold |
|-----------|------|-----------|
| Login Total | ~350ms | < 500ms ✓ |
| Registration | ~1500ms | < 2000ms ✓ |
| Embedding Encrypt | ~80ms | < 100ms ✓ |
| JWT Create | ~30ms | < 50ms ✓ |

## 🧪 Testing

```bash
# Run all tests
cd backend
pytest tests/ -v

# Run PQC tests
pytest tests/test_pqc.py -v

# Run health check
python scripts/system_health_check.py

# Run API tests
python scripts/test_api_endpoints.py
```

## 🌐 Environment Variables

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/requagnize

# Security
SECRET_KEY=your-secret-key-here
PQC_MASTER_PASSWORD=your-pqc-master-password

# Server
DEBUG=False
API_HOST=0.0.0.0
API_PORT=8000

# Face Recognition
COSINE_THRESHOLD=0.5
```

## 📚 Documentation

- [API Documentation](docs/API_DOCS.md)
- [PQC Implementation Guide](docs/PQC_IMPLEMENTATION.md)
- [Complete Knowledge Guide](docs/KNOWLEDGE_GUIDE.md)
- [Docker Guide](docs/DOCKER_GUIDE.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- **Roshan** - *Initial work*

---

Made with ❤️ using Quantum ML and Post-Quantum Cryptography

