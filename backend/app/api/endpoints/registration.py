# from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status, Request # pyright: ignore[reportMissingImports]
# from sqlalchemy.orm import Session
# from app.db.database import get_db
# from app.db.crud import UserCRUD, LogCRUD
# from app.schemas.user import RegistrationResponse
# from app.services.vqc_service import vqc_service
# from app.services.cloudinary_service import cloudinary_service
# from app.services.database_builder import database_builder
# from app.services.email_service import email_service
# from app.services.enhancement_service import image_enhancer
# import cv2
# import numpy as np
# import os

# router = APIRouter()

# @router.post("/register", response_model=RegistrationResponse)
# async def register_user(
#     request: Request,
#     username: str = Form(...),
#     email: str = Form(...),
#     full_name: str = Form(...),
#     image1: UploadFile = File(...),
#     image2: UploadFile = File(...),
#     image3: UploadFile = File(...),
#     db: Session = Depends(get_db)
# ):
#     """
#     User self-registration with face enrollment
    
#     Flow:
#     1. Check if email/username exists
#     2. Upload 3 images
#     3. Enhance images
#     4. Verify faces with VQC
#     5. Save to Cloudinary
#     6. Save user to database
#     7. Rebuild recognition database
#     8. Send confirmation email
#     """
#     try:
#         # Check if user exists
#         existing_user = UserCRUD.get_user_by_username(db, username)
#         if existing_user:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Username already exists"
#             )
        
#         existing_email = UserCRUD.get_user_by_email(db, email)
#         if existing_email:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Email already registered. Please use a different email."
#             )
        
#         # Process 3 images
#         images = [image1, image2, image3]
#         image_urls = []
#         enhanced_image_paths = []
        
#         # Create temporary user_id (will be replaced after DB insert)
#         temp_user_id = 0
        
#         for idx, image_file in enumerate(images, start=1):
#             # Read image
#             contents = await image_file.read()
#             nparr = np.frombuffer(contents, np.uint8)
#             img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
#             if img is None:
#                 raise HTTPException(
#                     status_code=status.HTTP_400_BAD_REQUEST,
#                     detail=f"Could not read image {idx}"
#                 )
            
#             # Enhance image (your enhance.py logic)
#             enhanced_img = image_enhancer.enhance_image(img)
            
#             # Convert back to BGR for VQC
#             if len(enhanced_img.shape) == 2:
#                 enhanced_img_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
#             else:
#                 enhanced_img_bgr = enhanced_img
            
#             # Verify it's a face using VQC
#             detection_roi_vec, _, _ = vqc_service.extract_detection_roi(enhanced_img_bgr)
#             if detection_roi_vec is None:
#                 raise HTTPException(
#                     status_code=status.HTTP_400_BAD_REQUEST,
#                     detail=f"Could not extract face from image {idx}"
#                 )
            
#             is_face, confidence = vqc_service.detect_face_vqc(detection_roi_vec)
#             if not is_face:
#                 raise HTTPException(
#                     status_code=status.HTTP_400_BAD_REQUEST,
#                     detail=f"No face detected in image {idx}. Please ensure your face is clearly visible."
#                 )
            
#             # Save enhanced image temporarily (will be renamed after getting user_id)
#             temp_filename = f"temp_{username}_img_{idx}.jpg"
#             temp_path = os.path.join("data/uploads/enrollment_images", temp_filename)
#             cv2.imwrite(temp_path, enhanced_img)
#             enhanced_image_paths.append(temp_path)
        
#         # Create user in database FIRST to get user_id
#         user = UserCRUD.create_user(
#             db, username, email, full_name, "", "", ""  # Temporary empty URLs
#         )
        
#         # Now upload to Cloudinary and update user
#         for idx, temp_path in enumerate(enhanced_image_paths, start=1):
#             # Read enhanced image
#             enhanced_bytes = open(temp_path, 'rb').read()
            
#             # Upload to Cloudinary
#             upload_result = cloudinary_service.upload_enrollment_image(
#                 enhanced_bytes, user.user_id, idx, username
#             )
            
#             if not upload_result["success"]:
#                 # Rollback
#                 UserCRUD.delete_user(db, user.user_id)
#                 raise HTTPException(
#                     status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                     detail=f"Failed to upload image {idx} to cloud storage"
#                 )
            
#             image_urls.append(upload_result["url"])
            
#             # Rename and save enhanced image with proper user_id
#             final_filename = f"user_{user.user_id}_img_{idx}.jpg"
#             final_path = os.path.join("data/uploads/enrollment_images", final_filename)
#             os.rename(temp_path, final_path)
        
#         # Update user with image URLs
#         user.image1_url = image_urls[0]
#         user.image2_url = image_urls[1]
#         user.image3_url = image_urls[2]
#         db.commit()
        
#         # Rebuild recognition database (runs your db_creation2.py)
#         rebuild_success = await database_builder.rebuild_database()
        
#         if not rebuild_success:
#             print("⚠️ Warning: Database rebuild failed, but user created")
        
#         # Send confirmation email
#         await email_service.send_registration_confirmation(email, username)
        
#         # Log registration
#         LogCRUD.create_log(
#             db,
#             user_id=user.user_id,
#             username=username,
#             event_type="registration",
#             ip_address=request.client.host,
#             success=True
#         )
        
#         return RegistrationResponse(
#             success=True,
#             message=f"Registration successful! Welcome, {username}. You can now log in with face authentication.",
#             user_id=user.user_id,
#             username=username
#         )
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Registration failed: {str(e)}"
#         )
# backend/app/api/endpoints/registration.py
# COMPLETE FILE - Replace entire content

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status, Request # pyright: ignore[reportMissingImports]
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.crud import UserCRUD, LogCRUD
from app.schemas.user import RegistrationResponse
from app.services.vqc_service import vqc_service
from app.services.cloudinary_service import cloudinary_service
from app.services.database_builder import database_builder
from app.services.email_service import email_service
from app.services.enhancement_service import image_enhancer
import cv2
import numpy as np
import os

router = APIRouter()

@router.post("/register", response_model=RegistrationResponse)
async def register_user(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    full_name: str = Form(...),
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    image3: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    User self-registration with face enrollment
    
    Flow:
    1. Check if email/username exists
    2. Upload 3 images
    3. Enhance images
    4. Extract face ROI using Haar Cascade
    5. Verify faces with VQC
    6. Save ONLY ROI to Cloudinary (not full image)
    7. Save user to database
    8. Rebuild recognition database
    9. Send confirmation email
    """
    try:
        # Check if user exists
        existing_user = UserCRUD.get_user_by_username(db, username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        existing_email = UserCRUD.get_user_by_email(db, email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered. Please use a different email."
            )
        
        # Process 3 images
        images = [image1, image2, image3]
        image_urls = []
        roi_image_paths = []  # Changed: Store ROI paths
        
        for idx, image_file in enumerate(images, start=1):
            # Read image
            contents = await image_file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not read image {idx}"
                )
            
            # Enhance full image first
            enhanced_img = image_enhancer.enhance_image(img)
            
            # Convert to BGR for processing
            if len(enhanced_img.shape) == 2:
                enhanced_img_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
            else:
                enhanced_img_bgr = enhanced_img
            
            # Extract detection ROI for VQC verification
            detection_roi_vec, detection_bbox, detection_roi_img = vqc_service.extract_detection_roi(enhanced_img_bgr)
            
            if detection_roi_vec is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not detect face in image {idx}. Please ensure your face is clearly visible and centered."
                )
            
            # Verify it's a face using VQC
            is_face, confidence = vqc_service.detect_face_vqc(detection_roi_vec)
            
            if not is_face:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"No face detected in image {idx}. Confidence: {confidence:.2f}. Please ensure your face is clearly visible."
                )
            
            # Extract recognition ROI - THIS IS WHAT WE SAVE
            recognition_roi_img, recognition_bbox = vqc_service.extract_recognition_roi(enhanced_img_bgr)
            
            if recognition_roi_img is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not extract face region from image {idx}"
                )
            
            # Save ONLY the ROI (not the full image)
            temp_filename = f"temp_{username}_img_{idx}.jpg"
            temp_path = os.path.join("data/uploads/enrollment_images", temp_filename)
            
            # Save the ROI image (128x128 or configured size)
            cv2.imwrite(temp_path, recognition_roi_img)
            roi_image_paths.append(temp_path)
            
            print(f"✓ Image {idx}: Face detected (confidence: {confidence:.2f}), ROI saved ({recognition_roi_img.shape})")
        
        # Create user in database FIRST to get user_id
        user = UserCRUD.create_user(
            db, username, email, full_name, "", "", ""  # Temporary empty URLs
        )
        
        print(f"✓ User created with ID: {user.user_id}")
        
        # Now upload ROIs to Cloudinary and update user
        for idx, temp_path in enumerate(roi_image_paths, start=1):
            # Read ROI image
            roi_bytes = open(temp_path, 'rb').read()
            
            # Upload ROI to Cloudinary
            upload_result = cloudinary_service.upload_enrollment_image(
                roi_bytes, user.user_id, idx, username
            )
            
            if not upload_result["success"]:
                # Rollback
                UserCRUD.delete_user(db, user.user_id)
                # Delete temp files
                for path in roi_image_paths:
                    if os.path.exists(path):
                        os.remove(path)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to upload image {idx} to cloud storage: {upload_result.get('error', 'Unknown error')}"
                )
            
            image_urls.append(upload_result["url"])
            
            # Rename ROI image with proper user_id
            final_filename = f"user_{user.user_id}_img_{idx}.jpg"
            final_path = os.path.join("data/uploads/enrollment_images", final_filename)
            os.rename(temp_path, final_path)
            
            print(f"✓ Image {idx} uploaded to Cloudinary: {upload_result['url']}")
        
        # Update user with Cloudinary URLs
        user.image1_url = image_urls[0]
        user.image2_url = image_urls[1]
        user.image3_url = image_urls[2]
        db.commit()
        
        print(f"✓ User URLs updated in database")
        
        # Rebuild recognition database (runs your db_creation2.py)
        print(f"🔄 Rebuilding recognition database...")
        rebuild_success = await database_builder.rebuild_database()
        
        if not rebuild_success:
            print("⚠️ Warning: Database rebuild failed, but user created")
        else:
            print("✓ Recognition database rebuilt successfully")
        
        # Send confirmation email
        try:
            await email_service.send_registration_confirmation(email, username)
            print(f"✓ Confirmation email sent to {email}")
        except Exception as e:
            print(f"⚠️ Warning: Email sending failed: {e}")
            # Don't fail registration if email fails
        
        # Log registration
        LogCRUD.create_log(
            db,
            user_id=user.user_id,
            username=username,
            event_type="registration",
            ip_address=request.client.host,
            success=True
        )
        
        print(f"✓ Registration complete for user: {username} (ID: {user.user_id})")
        
        return RegistrationResponse(
            success=True,
            message=f"Registration successful! Welcome, {username}. You can now log in with face authentication.",
            user_id=user.user_id,
            username=username
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )