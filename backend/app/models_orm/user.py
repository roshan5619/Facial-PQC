from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, func
from app.db.database import Base

class User(Base):
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    
    # Enrollment images (Cloudinary URLs)
    image1_url = Column(Text)
    image2_url = Column(Text)
    image3_url = Column(Text)
    
    # Status
    is_active = Column(Boolean, default=True)
    email_confirmed = Column(Boolean, default=True)  # Auto-confirm for now
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class AccessLog(Base):
    __tablename__ = "access_logs"
    
    log_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    username = Column(String(100))
    event_type = Column(String(50), nullable=False)  # 'login', 'logout', 'registration'
    
    # Authentication details
    similarity_score = Column(String)
    success = Column(Boolean, default=True)
    failure_reason = Column(Text)
    
    # Request metadata
    ip_address = Column(String)
    user_agent = Column(Text)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class Session(Base):
    __tablename__ = "sessions"
    
    session_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    username = Column(String(100))
    
    # JWT tracking
    token_jti = Column(String(255), unique=True, nullable=False, index=True)
    issued_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime, nullable=False)
    is_revoked = Column(Boolean, default=False)
    
    # Metadata
    ip_address = Column(String)
    user_agent = Column(Text)