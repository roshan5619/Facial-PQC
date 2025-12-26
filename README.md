# Face Authentication Exhibition System

AI-powered face recognition system with liveness detection for exhibition demo.

## Features

- 🔐 Face-based authentication
- 👁️ Blink detection for liveness verification
- 🎓 Quantum ML (VQC) for face detection
- 📊 Real-time access logging
- ✉️ Email confirmation
- ☁️ Cloud storage integration

## Technology Stack

- **Backend**: FastAPI + Python 3.11
- **Frontend**: React + TailwindCSS
- **Database**: PostgreSQL
- **ML**: PyTorch + PennyLane (Quantum ML)
- **Storage**: Cloudinary
- **Deployment**: Docker + Render

## Local Development Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- PostgreSQL 14+
- Docker Desktop

### Backend Setup

```bash
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
python -m uvicorn app.main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
cp .env.example .env
# Edit .env with API URL
npm start
```

## Project Structure
```
face-auth-exhibition/
│
├── backend/                          # Python FastAPI Backend
│   ├── app/
│   │   ├── init.py
│   │   ├── main.py                   # FastAPI app entry point
│   │   │
│   │   ├── api/
│   │   │   ├── init.py
│   │   │   └── endpoints/
│   │   │       ├── init.py
│   │   │       ├── auth.py           # Login endpoint
│   │   │       ├── registration.py   # Self-registration endpoint
│   │   │       ├── admin.py          # Admin dashboard
│   │   │       └── health.py         # Health checks
│   │   │
│   │   ├── core/
│   │   │   ├── init.py
│   │   │   ├── config.py             # Configuration settings
│   │   │   └── security.py           # JWT, password hashing
│   │   │
│   │   ├── db/
│   │   │   ├── init.py
│   │   │   ├── database.py           # PostgreSQL connection
│   │   │   └── crud.py               # Database operations
│   │   │
│   │   ├── models_orm/               # SQLAlchemy models
│   │   │   ├── init.py
│   │   │   └── user.py
│   │   │
│   │   ├── schemas/                  # Pydantic schemas
│   │   │   ├── init.py
│   │   │   └── user.py
│   │   │
│   │   ├── services/                 # Business logic
│   │   │   ├── init.py
│   │   │   ├── blink_detection.py    # NEW: OpenCV blink detection
│   │   │   ├── vqc_service.py        # NEW: Your VQC model wrapper
│   │   │   ├── enhancement_service.py # NEW: Your enhance.py wrapper
│   │   │   ├── database_builder.py   # NEW: Your db_creation2.py wrapper
│   │   │   ├── cloudinary_service.py # Cloudinary integration
│   │   │   └── email_service.py      # Email confirmation
│   │   │
│   │   ├── middleware/
│   │   │   ├── init.py
│   │   │   └── rate_limit.py
│   │   │
│   │   └── utils/
│   │       ├── init.py
│   │       └── helpers.py
│   │
│   ├── ml_models/                    # Your ML models and scripts
│   │   ├── vqc_face_model_roi.pth    # Your quantum model
│   │   ├── haarcascade_frontalface_default.xml
│   │   ├── pca_detection_roi.pkl     # 8-PCA for detection
│   │   ├── new_pca_recognition_cosine_roi.pkl      # 512-PCA
│   │   ├── new_scaler_recognition_cosine_roi.pkl
│   │   ├── new_recognition_db_cosine_roi.pkl       # Embeddings DB
│   │   ├── new_used_paths_cosine_roi.pkl
│   │   │
│   │   ├── prediction_new.py         # Your original code (reference)
│   │   ├── enhance.py                # Your enhancement script
│   │   ├── db_creation2.py           # Your DB builder script
│   │   ├── Face_DB2.py               # Your detection training
│   │   └── Dataset_Preperation.py    # Your data prep
│   │
│   ├── scripts/
│   │   ├── migrate_100_users.py      # Import your 100 users
│   │   └── test_models.py            # Test model loading
│   │
│   ├── tests/
│   │   ├── init.py
│   │   └── test_api.py
│   │
│   ├── Dockerfile                    # Docker configuration
│   ├── requirements.txt              # Python dependencies
│   ├── .env.example                  # Environment template
│   └── .dockerignore
│
├── frontend/                         # React Frontend
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.ico
│   │
│   ├── src/
│   │   ├── components/
│   │   │   ├── BlinkCamera.jsx       # NEW: Blink detection camera
│   │   │   ├── Header.jsx
│   │   │   └── ProtectedRoute.jsx
│   │   │
│   │   ├── pages/
│   │   │   ├── Home.jsx              # Login/Register buttons
│   │   │   ├── UserLogin.jsx         # Blink → Authenticate
│   │   │   ├── UserRegistration.jsx  # Self-enrollment
│   │   │   ├── UserDashboard.jsx     # Success page
│   │   │   ├── AdminLogin.jsx        # Admin panel
│   │   │   └── AdminDashboard.jsx
│   │   │
│   │   ├── services/
│   │   │   └── api.js                # API calls
│   │   │
│   │   ├── utils/
│   │   │   └── validation.js
│   │   │
│   │   ├── App.js
│   │   ├── index.js
│   │   └── index.css
│   │
│   ├── Dockerfile
│   ├── Dockerfile.dev
│   ├── nginx.conf
│   ├── package.json
│   ├── tailwind.config.js
│   └── .env.example
│
├── data/
│   ├── uploads/
│   │   └── enrollment_images/        # Enhanced images (Cloudinary backup)
│   └── database_backups/
│
├── logs/
│   └── .gitkeep
│
├── config/
│   └── nginx.conf                    # Production nginx config
│
├── docs/
│   ├── API_DOCS.md
│   ├── DOCKER_GUIDE.md               # Docker for beginners
│   └── EXHIBITION_CHECKLIST.md
│
├── .github/
│   └── workflows/
│       └── deploy.yml                # GitHub Actions CI/CD
│
├── docker-compose.yml                # Local development
├── .gitignore
├── .env.example
└── README.md
```

