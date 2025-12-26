from pydantic_settings import BaseSettings # pyright: ignore[reportMissingImports]
from typing import Optional
import os

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 5
    
    # Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    
    # Cloudinary
    CLOUDINARY_CLOUD_NAME: str = "dbubrt9k9"
    CLOUDINARY_API_KEY: str = "982782172324256"
    CLOUDINARY_API_SECRET: str = "05Sx8ZVGqR1cvAHmEtUujGLRfqM"
    
    # Email
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM_EMAIL: str = ""
    SMTP_FROM_NAME: str = "Face Auth System"
    # ML Models
    MODELS_DIR: str = "./ml_models"
    VQC_MODEL_PATH: str = "vqc_face_model_roi.pth"
    HAAR_CASCADE_PATH: str = "haarcascade_frontalface_default.xml"  # ADD THIS LINE
    PCA_DETECTION_PATH: str = "pca_detection_roi.pkl"
    
    # Recognition Models (512 PCA components)
    PCA_RECOGNITION_PATH: str = "qcclass_pca_recognition_cosine_roi.pkl"
    SCALER_RECOGNITION_PATH: str = "qcclass_scaler_recognition_cosine_roi.pkl"
    RECOGNITION_DB_PATH: str = "qcclass_recognition_db_cosine_roi.pkl"
    USED_PATHS_PATH: str = "qcclass_used_paths_cosine_roi.pkl"
    ROI_PATHS_PATH: str = "qcclass_roi_paths_cosine_roi.pkl"
    
    # Face Recognition
    ROI_SIZE: int = 128
    N_QUBITS: int = 8
    COSINE_THRESHOLD: float = 0.5
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_GRID_SIZE: int = 8
    GAMMA_CORRECTION: float = 1.2
    
    # Blink Detection
    BLINK_THRESHOLD: int = 2
    EYE_AR_THRESHOLD: float = 0.25
    EYE_AR_CONSEC_FRAMES: int = 3
    
    # Session
    SESSION_TIMEOUT_MINUTES: int = 5
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Ensure directories exist
os.makedirs(os.path.join(settings.MODELS_DIR), exist_ok=True)
os.makedirs("data/uploads/enrollment_images", exist_ok=True)