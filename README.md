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
| Digital Signatures | Dilithium3 | NIST Level 3 |
| Critical Operations | SPHINCS+ | NIST Level 1 |
| Symmetric Encryption | AES-256-GCM | 256-bit |
| Face Embeddings | PQC Encrypted | Quantum-Safe |

## 🛠️ Technology Stack

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI + Python 3.11 |
| **Frontend** | React 18 + TailwindCSS |
| **Database** | PostgreSQL 15+ |
| **Quantum ML** | PyTorch + PennyLane (VQC) |
| **PQC** | liboqs-python (NTRU, Kyber, Dilithium) |
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
```

### JWT Token Format
```
Classical:  header.payload.HMAC_signature
PQC:        header.payload.Dilithium_signature
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

