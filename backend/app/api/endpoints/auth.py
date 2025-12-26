from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, Request # pyright: ignore[reportMissingImports]
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials # pyright: ignore[reportMissingImports]
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.crud import UserCRUD, LogCRUD
from app.schemas.user import LoginResponse
from app.services.vqc_service import vqc_service
from app.services.enhancement_service import image_enhancer
from app.core.security import create_access_token, decode_access_token
from datetime import timedelta
import cv2
import numpy as np

router = APIRouter()
security = HTTPBearer()

@router.post("/login", response_model=LoginResponse)
async def face_login(
    request: Request,
    image: UploadFile = File(..., description="Face image after blink detection"),
    db: Session = Depends(get_db)
):
    """
    Face authentication login
    
    Flow:
    1. Receive image (already passed blink detection on frontend)
    2. Enhance image
    3. VQC face detection
    4. Face recognition & matching
    5. Generate JWT token
    6. Log access
    """
    try:
        # Read uploaded image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            LogCRUD.create_log(
                db, None, None, "failed_login",
                ip_address=request.client.host,
                success=False,
                failure_reason="Invalid image"
            )
            return LoginResponse(
                success=False,
                message="Could not read image. Please try again."
            )
        
        # Enhance image (your enhance.py logic)
        enhanced_img = image_enhancer.enhance_image(img)
        
        # Convert to BGR if grayscale
        if len(enhanced_img.shape) == 2:
            enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
        
        # Process image through VQC pipeline
        result = vqc_service.process_image_for_authentication(enhanced_img)
        
        if not result['success']:
            # Log failed attempt
            LogCRUD.create_log(
                db, None, None, "failed_login",
                ip_address=request.client.host,
                similarity_score=result.get('similarity'),
                success=False,
                failure_reason=result.get('message')
            )
            
            return LoginResponse(
                success=False,
                message=result.get('message', 'Authentication failed'),
                similarity_score=result.get('similarity')
            )
        
        # Get user from database
        user_id = result.get('user_id')
        user = UserCRUD.get_user_by_id(db, user_id)
        
        if not user:
            return LoginResponse(
                success=False,
                message="User not found in database"
            )
        
        if not user.is_active:
            return LoginResponse(
                success=False,
                message="User account is inactive"
            )
        
        # Generate JWT token (5 minutes expiry)
        access_token = create_access_token(
            data={
                "sub": user.username,
                "user_id": user.user_id,
                "email": user.email
            },
            expires_delta=timedelta(minutes=5)
        )
        
        # Log successful login
        LogCRUD.create_log(
            db,
            user_id=user.user_id,
            username=user.username,
            event_type="login",
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent"),
            similarity_score=result.get('similarity'),
            success=True
        )
        
        return LoginResponse(
            success=True,
            access_token=access_token,
            token_type="bearer",
            user_id=user.user_id,
            username=user.username,
            message=f"Welcome back, {user.username}!",
            similarity_score=result.get('similarity')
        )
    
    except HTTPException:
        raise
    except Exception as e:
        LogCRUD.create_log(
            db, None, None, "failed_login",
            ip_address=request.client.host,
            success=False,
            failure_reason=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication failed: {str(e)}"
        )


@router.post("/logout")
async def logout(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Logout user"""
    token = credentials.credentials
    payload = decode_access_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    user_id = payload.get("user_id")
    username = payload.get("sub")
    
    # Log logout
    LogCRUD.create_log(
        db, user_id, username, "logout",
        ip_address=request.client.host,
        success=True
    )
    
    return {
        "success": True,
        "message": "Logged out successfully"
    }


@router.get("/verify")
async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Verify if token is valid"""
    token = credentials.credentials
    payload = decode_access_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    user_id = payload.get("user_id")
    user = UserCRUD.get_user_by_id(db, user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {
        "valid": True,
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name
    }