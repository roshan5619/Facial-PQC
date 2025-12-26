from sqlalchemy.orm import Session
from app.models_orm.user import User, AccessLog, Session as SessionModel
from typing import Optional, List
from datetime import datetime

class UserCRUD:
    @staticmethod
    def create_user(
        db: Session,
        username: str,
        email: str,
        full_name: str,
        image1_url: str,
        image2_url: str,
        image3_url: str
    ) -> User:
        user = User(
            username=username,
            email=email,
            full_name=full_name,
            image1_url=image1_url,
            image2_url=image2_url,
            image3_url=image3_url,
            email_confirmed=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
        return db.query(User).filter(User.user_id == user_id).first()
    
    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        return db.query(User).filter(User.username == username).first()
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def get_all_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
        return db.query(User).offset(skip).limit(limit).all()
    
    @staticmethod
    def delete_user(db: Session, user_id: int) -> bool:
        user = db.query(User).filter(User.user_id == user_id).first()
        if user:
            db.delete(user)
            db.commit()
            return True
        return False

class LogCRUD:
    @staticmethod
    def create_log(
        db: Session,
        user_id: Optional[int],
        username: Optional[str],
        event_type: str,
        ip_address: str = None,
        user_agent: str = None,
        similarity_score: float = None,
        success: bool = True,
        failure_reason: str = None
    ) -> AccessLog:
        log = AccessLog(
            user_id=user_id,
            username=username,
            event_type=event_type,
            ip_address=ip_address,
            user_agent=user_agent,
            similarity_score=str(similarity_score) if similarity_score else None,
            success=success,
            failure_reason=failure_reason
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return log
    
    @staticmethod
    def get_all_logs(db: Session, skip: int = 0, limit: int = 100) -> List[AccessLog]:
        return db.query(AccessLog)\
            .order_by(AccessLog.timestamp.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()