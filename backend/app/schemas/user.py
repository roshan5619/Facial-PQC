from pydantic import BaseModel, EmailStr, validator
from typing import Optional
from datetime import datetime

class UserRegistration(BaseModel):
    username: str
    email: EmailStr
    full_name: str
    
    @validator('username')
    def username_valid(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v

class UserResponse(BaseModel):
    user_id: int
    username: str
    email: str
    full_name: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class LoginResponse(BaseModel):
    success: bool
    access_token: Optional[str] = None
    token_type: str = "bearer"
    user_id: Optional[int] = None
    username: Optional[str] = None
    message: str
    similarity_score: Optional[float] = None

class RegistrationResponse(BaseModel):
    success: bool
    message: str
    user_id: Optional[int] = None
    username: Optional[str] = None

class AccessLogResponse(BaseModel):
    log_id: int
    user_id: Optional[int]
    username: Optional[str]
    event_type: str
    similarity_score: Optional[float]
    success: bool
    timestamp: datetime
    
    class Config:
        from_attributes = True