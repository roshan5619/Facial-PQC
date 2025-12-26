from fastapi import APIRouter, Depends # pyright: ignore[reportMissingImports]
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.services.vqc_service import vqc_service

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "Face Authentication API",
        "version": "1.0.0"
    }

@router.get("/health/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """Detailed health check"""
    try:
        # Check database
        db.execute("SELECT 1")
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    # Check models loaded
    models_status = "healthy" if vqc_service.vqc_model else "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" and models_status == "healthy" else "degraded",
        "components": {
            "database": db_status,
            "ml_models": models_status
        }
    }