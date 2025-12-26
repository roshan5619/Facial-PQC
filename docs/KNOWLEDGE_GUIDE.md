# REQUAGNIZE - Complete Knowledge Guide

## AI-Powered Face Authentication System with Post-Quantum Cryptography

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Setup Guide](#4-setup-guide)
5. [Backend Deep Dive](#5-backend-deep-dive)
6. [Frontend Deep Dive](#6-frontend-deep-dive)
7. [Post-Quantum Cryptography](#7-post-quantum-cryptography)
8. [Database Schema](#8-database-schema)
9. [API Reference](#9-api-reference)
10. [Security Features](#10-security-features)
11. [Testing Guide](#11-testing-guide)
12. [Deployment Guide](#12-deployment-guide)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Project Overview

### What is REQUAGNIZE?

REQUAGNIZE is an AI-powered face authentication system that combines:
- **Variational Quantum Circuits (VQC)** for face recognition
- **Post-Quantum Cryptography (PQC)** for quantum-safe security
- **Blink Detection** for liveness verification
- **Modern Web Interface** for user interaction

### Key Features

| Feature | Description |
|---------|-------------|
| Face Enrollment | Register users with facial biometrics |
| Face Authentication | Login using face recognition |
| Liveness Detection | Blink detection prevents photo spoofing |
| Quantum ML | VQC-based face embedding generation |
| PQC Security | NTRU + Kyber + Dilithium encryption |
| Admin Dashboard | User management interface |

### Use Cases

1. **Secure Access Control** - Building entry systems
2. **Identity Verification** - KYC processes
3. **Attendance Systems** - Automated check-in
4. **Authentication** - Password-free login

---

## 2. Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           REQUAGNIZE SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────┐    │
│  │   FRONTEND  │     │   BACKEND   │     │       DATABASE          │    │
│  │             │     │             │     │                         │    │
│  │  React.js   │────▶│   FastAPI   │────▶│   PostgreSQL            │    │
│  │  Tailwind   │     │   Python    │     │                         │    │
│  │             │     │             │     │  - Users                │    │
│  └─────────────┘     └──────┬──────┘     │  - Embeddings           │    │
│                             │            │  - Sessions             │    │
│                             │            └─────────────────────────┘    │
│                             │                                           │
│                     ┌───────▼───────┐                                   │
│                     │   ML MODELS   │                                   │
│                     │               │                                   │
│                     │  - VQC Face   │                                   │
│                     │  - OpenCV     │                                   │
│                     │  - PennyLane  │                                   │
│                     └───────────────┘                                   │
│                                                                          │
│                     ┌───────────────┐                                   │
│                     │     PQC       │                                   │
│                     │               │                                   │
│                     │  - NTRU       │                                   │
│                     │  - Kyber      │                                   │
│                     │  - Dilithium  │                                   │
│                     └───────────────┘                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Request → Frontend → API Gateway → FastAPI Backend
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │ Authentication  │
                                    │   Middleware    │
                                    └────────┬────────┘
                                             │
                            ┌────────────────┼────────────────┐
                            ▼                ▼                ▼
                     ┌──────────┐     ┌──────────┐     ┌──────────┐
                     │  Auth    │     │  User    │     │  Admin   │
                     │ Endpoint │     │ Endpoint │     │ Endpoint │
                     └────┬─────┘     └────┬─────┘     └────┬─────┘
                          │                │                │
                          └────────────────┼────────────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │   Services   │
                                    │              │
                                    │ - VQC       │
                                    │ - PQC       │
                                    │ - Blink     │
                                    └──────┬───────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │   Database   │
                                    └──────────────┘
```

---

## 3. Technology Stack

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.11+ | Core language |
| FastAPI | 0.104.1 | Web framework |
| SQLAlchemy | 2.0.23 | ORM |
| PostgreSQL | 15+ | Database |
| PennyLane | 0.33.1 | Quantum ML |
| PyTorch | 2.0.1 | Deep learning |
| OpenCV | 4.8.1 | Computer vision |
| liboqs-python | 0.8.0 | Post-quantum crypto |

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.x | UI framework |
| Tailwind CSS | 3.x | Styling |
| Axios | Latest | HTTP client |

### Infrastructure

| Technology | Purpose |
|------------|---------|
| Docker | Containerization |
| Nginx | Reverse proxy |
| Alembic | Database migrations |

---

## 4. Setup Guide

### Prerequisites

```bash
# Required software
- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- Git
```

### Step 1: Clone Repository

```bash
git clone https://github.com/your-repo/REQUAGNIZE_PRODUCT.git
cd REQUAGNIZE_PRODUCT-main
```

### Step 2: Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
DATABASE_URL=postgresql://user:password@localhost:5432/requagnize
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
PQC_MASTER_PASSWORD=your-pqc-master-password
EOF
```

### Step 3: Database Setup

```bash
# Create database
psql -U postgres -c "CREATE DATABASE requagnize;"

# Run migrations
alembic upgrade head
```

### Step 4: Frontend Setup

```bash
# Navigate to frontend
cd ../frontend/react-app

# Install dependencies
npm install

# Create .env file
cat > .env << EOF
REACT_APP_API_URL=http://localhost:8000
EOF
```

### Step 5: Run Development Servers

```bash
# Terminal 1: Backend
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend/react-app
npm start
```

### Step 6: Verify Installation

```bash
# Check backend health
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "timestamp": "2025-12-26T..."}
```

---

## 5. Backend Deep Dive

### Directory Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry
│   │
│   ├── api/
│   │   └── endpoints/
│   │       ├── admin.py        # Admin endpoints
│   │       ├── auth.py         # Authentication endpoints
│   │       ├── health.py       # Health check
│   │       └── registration.py # User registration
│   │
│   ├── core/
│   │   ├── config.py           # Configuration settings
│   │   └── security.py         # Security utilities
│   │
│   ├── db/
│   │   ├── crud.py             # Database operations
│   │   └── database.py         # Database connection
│   │
│   ├── middleware/
│   │   └── rate_limit.py       # Rate limiting
│   │
│   ├── models_orm/
│   │   └── user.py             # SQLAlchemy models
│   │
│   ├── schemas/
│   │   └── user.py             # Pydantic schemas
│   │
│   └── services/
│       ├── blink_detection.py      # Liveness detection
│       ├── cloudinary_service.py   # Image storage
│       ├── database_builder.py     # DB initialization
│       ├── email_service.py        # Email notifications
│       ├── enhancement_service.py  # Image enhancement
│       ├── vqc_service.py          # Quantum ML service
│       │
│       # PQC Services (New)
│       ├── pqc_service.py          # Core PQC operations
│       ├── pqc_key_manager.py      # Key management
│       ├── pqc_jwt_service.py      # PQC JWT tokens
│       ├── hybrid_crypto_service.py # Hybrid encryption
│       └── vqc_service_pqc.py      # VQC with PQC
│
├── ml_models/
│   ├── vqc_face_model_roi.pth  # Trained VQC model
│   └── haarcascade_*.xml       # Face detection cascades
│
├── scripts/
│   ├── migrate_100_users_pqc.py    # User migration
│   ├── backup_rollback_pqc.py      # Backup tools
│   └── benchmark_pqc.py            # Performance tests
│
└── tests/
    ├── test_api.py             # API tests
    └── test_pqc.py             # PQC tests
```

### Key Components

#### 1. Main Application (main.py)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="REQUAGNIZE API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/api/auth")
app.include_router(admin_router, prefix="/api/admin")
app.include_router(registration_router, prefix="/api/registration")
```

#### 2. VQC Service (vqc_service.py)

The VQC service handles face recognition using quantum circuits:

```python
class VQCFaceService:
    def __init__(self):
        self.model = VQCFaceModel()
        self.model.load_state_dict(torch.load("vqc_face_model_roi.pth"))
    
    def process_image_for_authentication(self, image_data):
        # 1. Detect face using Haar Cascade
        # 2. Extract ROI (Region of Interest)
        # 3. Preprocess image
        # 4. Generate embedding using VQC
        # 5. Compare with stored embeddings
        pass
```

#### 3. Security (security.py)

```python
# PQC-enhanced JWT creation
def create_access_token(data: dict, expires_delta: timedelta = None):
    # Try PQC JWT first
    if pqc_available:
        return pqc_jwt_service.create_token(data, user_id)
    # Fallback to classical JWT
    return classical_jwt_encode(data)
```

---

## 6. Frontend Deep Dive

### Directory Structure

```
frontend/react-app/
├── public/
│   └── index.html
│
├── src/
│   ├── App.js                  # Main application
│   ├── index.js                # Entry point
│   │
│   ├── components/
│   │   ├── BlinkCamera.jsx     # Camera with blink detection
│   │   ├── Header.jsx          # Navigation header
│   │   └── ProtectedRoute.jsx  # Auth route wrapper
│   │
│   ├── pages/
│   │   ├── Home.jsx            # Landing page
│   │   ├── UserLogin.jsx       # User login
│   │   ├── UserRegistration.jsx# User signup
│   │   ├── UserDashboard.jsx   # User dashboard
│   │   ├── AdminLogin.jsx      # Admin login
│   │   └── AdminDashboard.jsx  # Admin dashboard
│   │
│   ├── services/
│   │   └── api.js              # API client
│   │
│   └── utils/
│       └── validation.js       # Form validation
│
└── package.json
```

### Key Components

#### 1. BlinkCamera Component

Handles camera feed and blink detection:

```jsx
const BlinkCamera = ({ onCapture, onBlinkDetected }) => {
  const videoRef = useRef(null);
  const [blinkCount, setBlinkCount] = useState(0);
  
  // Start camera
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        videoRef.current.srcObject = stream;
      });
  }, []);
  
  // Capture frame and send to backend
  const captureFrame = () => {
    const canvas = document.createElement('canvas');
    canvas.getContext('2d').drawImage(videoRef.current, 0, 0);
    const imageData = canvas.toDataURL('image/jpeg');
    onCapture(imageData);
  };
  
  return (
    <div>
      <video ref={videoRef} autoPlay />
      <p>Blinks detected: {blinkCount}</p>
    </div>
  );
};
```

#### 2. API Service

```javascript
// services/api.js
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL;

export const api = {
  // Authentication
  login: (imageData) => 
    axios.post(`${API_BASE}/api/auth/login`, { image: imageData }),
  
  // Registration
  register: (userData, imageData) =>
    axios.post(`${API_BASE}/api/registration/enroll`, {
      ...userData,
      image: imageData
    }),
  
  // Admin
  getUsers: (token) =>
    axios.get(`${API_BASE}/api/admin/users`, {
      headers: { Authorization: `Bearer ${token}` }
    })
};
```

---

## 7. Post-Quantum Cryptography

### Overview

Post-Quantum Cryptography (PQC) protects against attacks from quantum computers.

### Algorithms Used

| Algorithm | Type | Security Level | Use Case |
|-----------|------|----------------|----------|
| NTRU-HPS-2048-509 | KEM | NIST Level 3 | Primary encryption |
| Kyber768 | KEM | NIST Level 3 | Secondary encryption |
| Dilithium3 | Signature | NIST Level 3 | JWT tokens |
| SPHINCS+-SHA2-128f | Signature | NIST Level 1 | Critical operations |

### Hybrid Encryption Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    Hybrid Encryption                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  plaintext ──► [NTRU Encap] ──► shared_secret_1              │
│            │                                                  │
│            └─► [Kyber Encap] ──► shared_secret_2             │
│                                                               │
│  combined_secret = shared_secret_1 ⊕ shared_secret_2         │
│                                                               │
│  aes_key = HKDF(combined_secret)                             │
│                                                               │
│  ciphertext = AES-256-GCM(plaintext, aes_key)                │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Key Storage

```
data/pqc_keys/
├── system/                     # System-wide keys
│   ├── ntru_public.key
│   ├── ntru_private.enc       # Encrypted with master password
│   ├── kyber_public.key
│   ├── kyber_private.enc
│   ├── dilithium_public.key
│   └── dilithium_private.enc
│
└── user_{id}/                  # Per-user keys
    ├── ntru_public.key
    ├── ntru_private.key.enc
    ├── kyber_public.key
    ├── kyber_private.key.enc
    ├── dilithium_public.key
    ├── dilithium_private.key.enc
    └── key_metadata.json
```

### PQC JWT Token Format

```
Classical JWT:
header.payload.hmac_signature

PQC JWT:
header.payload.dilithium_signature

Header: {
  "alg": "DILITHIUM3",
  "typ": "JWT",
  "kid": "user_123_v1"
}
```

---

## 8. Database Schema

### Entity Relationship Diagram

```
┌─────────────────────┐
│       users         │
├─────────────────────┤
│ id (PK)             │
│ email (UNIQUE)      │
│ username            │
│ hashed_password     │
│ full_name           │
│ is_active           │
│ is_admin            │
│ face_embedding      │──────┐
│ created_at          │      │
│ updated_at          │      │
└─────────────────────┘      │
                             │
                             ▼
                    ┌─────────────────┐
                    │ face_embeddings │
                    │   (encrypted)   │
                    └─────────────────┘
```

### User Model

```python
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    face_embedding = Column(LargeBinary)  # PQC-encrypted
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
```

---

## 9. API Reference

### Authentication Endpoints

#### POST /api/auth/login

Login with face recognition.

**Request:**
```json
{
  "image": "base64_encoded_image_data"
}
```

**Response:**
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "email": "user@example.com",
    "username": "johndoe"
  }
}
```

#### POST /api/auth/admin/login

Admin login with credentials.

**Request:**
```json
{
  "email": "admin@example.com",
  "password": "admin_password"
}
```

### Registration Endpoints

#### POST /api/registration/enroll

Register new user with face.

**Request:**
```json
{
  "email": "user@example.com",
  "username": "johndoe",
  "full_name": "John Doe",
  "password": "secure_password",
  "images": ["base64_image_1", "base64_image_2", "base64_image_3"]
}
```

**Response:**
```json
{
  "message": "User registered successfully",
  "user_id": 1
}
```

### Admin Endpoints

#### GET /api/admin/users

Get all users (admin only).

**Headers:**
```
Authorization: Bearer <admin_token>
```

**Response:**
```json
{
  "users": [
    {
      "id": 1,
      "email": "user@example.com",
      "username": "johndoe",
      "is_active": true,
      "created_at": "2025-12-26T10:00:00Z"
    }
  ]
}
```

### Health Endpoint

#### GET /health

Check system health.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-26T10:00:00Z",
  "version": "1.0.0"
}
```

---

## 10. Security Features

### Authentication Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   Authentication Flow                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. User opens camera                                        │
│              │                                               │
│              ▼                                               │
│  2. Blink detection (liveness check)                        │
│              │                                               │
│              ▼                                               │
│  3. Capture face image                                       │
│              │                                               │
│              ▼                                               │
│  4. Backend: Face detection (Haar Cascade)                  │
│              │                                               │
│              ▼                                               │
│  5. Backend: Generate VQC embedding                         │
│              │                                               │
│              ▼                                               │
│  6. Backend: Decrypt stored embeddings (PQC)                │
│              │                                               │
│              ▼                                               │
│  7. Backend: Compare embeddings (cosine similarity)         │
│              │                                               │
│              ▼                                               │
│  8. Backend: Generate PQC JWT token                         │
│              │                                               │
│              ▼                                               │
│  9. Return token to frontend                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Security Layers

| Layer | Protection |
|-------|------------|
| **Transport** | HTTPS/TLS 1.3 |
| **API** | Rate limiting, CORS |
| **Authentication** | Face + Blink detection |
| **Authorization** | JWT tokens (Dilithium signed) |
| **Data at Rest** | PQC hybrid encryption |
| **Keys** | Master password + PBKDF2 |

### Rate Limiting

```python
# Configured limits
RATE_LIMITS = {
    "/api/auth/login": "5/minute",
    "/api/registration/enroll": "3/minute",
    "/api/admin/*": "100/minute"
}
```

---

## 11. Testing Guide

### Running Tests

```bash
# Backend tests
cd backend
pytest tests/ -v

# PQC tests
pytest tests/test_pqc.py -v

# API tests
pytest tests/test_api.py -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

### Test Categories

#### Unit Tests

```python
# tests/test_pqc.py
def test_ntru_encryption():
    pub, priv = pqc_service.generate_ntru_keypair()
    plaintext = b"Hello, World!"
    ciphertext = pqc_service.ntru_encrypt(plaintext, pub)
    decrypted = pqc_service.ntru_decrypt(ciphertext, priv)
    assert decrypted == plaintext
```

#### Integration Tests

```python
# tests/test_api.py
def test_login_flow(client):
    # 1. Register user
    response = client.post("/api/registration/enroll", json={...})
    assert response.status_code == 200
    
    # 2. Login with face
    response = client.post("/api/auth/login", json={...})
    assert response.status_code == 200
    assert "access_token" in response.json()
```

### Benchmark Tests

```bash
# Run performance benchmarks
python scripts/benchmark_pqc.py
```

Expected output:
```
PQC Performance Benchmark Results
================================
NTRU Keygen:        15.2ms (threshold: 50ms) ✓
Kyber Keygen:       10.1ms (threshold: 50ms) ✓
Hybrid Encrypt:     35.4ms (threshold: 50ms) ✓
JWT Create:         28.3ms (threshold: 50ms) ✓
Login Total:       342.1ms (threshold: 500ms) ✓
Registration:     1456.8ms (threshold: 2000ms) ✓
```

---

## 12. Deployment Guide

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build -d

# View logs
docker-compose logs -f
```

### Production Configuration

```yaml
# docker-compose.yml (production)
version: '3.8'
services:
  backend:
    build: ./backend
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/requagnize
      - SECRET_KEY=${SECRET_KEY}
      - PQC_MASTER_PASSWORD=${PQC_MASTER_PASSWORD}
    ports:
      - "8000:8000"
    depends_on:
      - db
  
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
  
  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_PASSWORD=${DB_PASSWORD}

volumes:
  postgres_data:
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `SECRET_KEY` | JWT signing key (fallback) | Yes |
| `PQC_MASTER_PASSWORD` | PQC key encryption password | Yes |
| `CLOUDINARY_URL` | Image storage URL | No |
| `SMTP_HOST` | Email server | No |

### Health Checks

```bash
# Check backend
curl http://localhost:8000/health

# Check database
curl http://localhost:8000/health/db

# Check PQC status
curl http://localhost:8000/health/pqc
```

---

## 13. Troubleshooting

### Common Issues

#### Issue: "liboqs not found"

```bash
# Solution: Install liboqs-python
pip install liboqs-python

# Windows: May need Visual Studio Build Tools
# Linux: sudo apt-get install cmake ninja-build
```

#### Issue: "Database connection failed"

```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Verify DATABASE_URL
echo $DATABASE_URL

# Test connection
psql $DATABASE_URL -c "SELECT 1;"
```

#### Issue: "Face not detected"

Possible causes:
- Poor lighting
- Face not centered in frame
- Camera resolution too low

Solutions:
- Ensure good lighting
- Position face in center
- Try different camera

#### Issue: "JWT verification failed"

```bash
# Check if using PQC or classical JWT
curl http://localhost:8000/health/pqc

# If PQC not available, check liboqs installation
python -c "import oqs; print('OK')"
```

#### Issue: "Embedding mismatch"

```bash
# Re-run migration to fix embeddings
python scripts/migrate_100_users_pqc.py

# Or rollback and retry
python scripts/backup_rollback_pqc.py rollback --backup-id <id>
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# In FastAPI
app = FastAPI(debug=True)
```

### Log Files

```bash
# View logs
tail -f logs/app.log

# Error logs
tail -f logs/error.log
```

---

## Quick Reference Card

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/auth/login | Face login |
| POST | /api/auth/admin/login | Admin login |
| POST | /api/registration/enroll | Register user |
| GET | /api/admin/users | List users |
| GET | /health | Health check |

### CLI Commands

```bash
# Start backend
uvicorn app.main:app --reload

# Start frontend
npm start

# Run tests
pytest tests/ -v

# Run migrations
alembic upgrade head

# Backup database
python scripts/backup_rollback_pqc.py backup

# Benchmark PQC
python scripts/benchmark_pqc.py
```

### Performance Targets

| Operation | Target | Actual |
|-----------|--------|--------|
| Login | <500ms | ~350ms |
| Registration | <2000ms | ~1500ms |
| Embedding Encrypt | <100ms | ~80ms |
| JWT Create | <50ms | ~30ms |

---

## Support

For issues and questions:
1. Check this documentation
2. Review troubleshooting section
3. Check logs for error messages
4. Submit issue on GitHub

---

*Last Updated: December 26, 2025*
