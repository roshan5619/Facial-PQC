from fastapi import APIRouter, Depends, HTTPException # pyright: ignore[reportMissingImports]
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.crud import UserCRUD, LogCRUD
from app.services.database_builder import database_builder
from typing import List
from app.schemas.user import UserResponse, AccessLogResponse

router = APIRouter()

@router.get("/users", response_model=List[UserResponse])
async def get_all_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all registered users"""
    users = UserCRUD.get_all_users(db, skip, limit)
    return users

@router.get("/logs", response_model=List[AccessLogResponse])
async def get_access_logs(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get access logs"""
    logs = LogCRUD.get_all_logs(db, skip, limit)
    return logs

@router.get("/stats")
async def get_statistics(db: Session = Depends(get_db)):
    """Get system statistics"""
    from sqlalchemy import func
    from app.models_orm.user import User, AccessLog
    from datetime import datetime, timedelta
    
    # Total users
    total_users = db.query(func.count(User.user_id)).scalar()
    
    # Active users (logged in today)
    today = datetime.utcnow().date()
    logins_today = db.query(func.count(AccessLog.log_id))\
        .filter(
            AccessLog.event_type == 'login',
            AccessLog.success == True,
            func.date(AccessLog.timestamp) == today
        )\
        .scalar()
    
    # Database stats
    db_stats = database_builder.get_database_stats()
    
    return {
        "total_users": total_users,
        "logins_today": logins_today,
        "database_stats": db_stats
    }

@router.post("/rebuild-database")
async def rebuild_database():
    """Manually trigger database rebuild"""
    success = await database_builder.rebuild_database()
    
    if success:
        return {"success": True, "message": "Database rebuilt successfully"}
    else:
        raise HTTPException(
            status_code=500,
            detail="Database rebuild failed"
        )