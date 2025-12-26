import cloudinary # pyright: ignore[reportMissingImports]
import cloudinary.uploader # pyright: ignore[reportMissingImports]
from app.core.config import settings
from typing import Optional, Dict

class CloudinaryService:
    def __init__(self):
        cloudinary.config(
            cloud_name=settings.CLOUDINARY_CLOUD_NAME,
            api_key=settings.CLOUDINARY_API_KEY,
            api_secret=settings.CLOUDINARY_API_SECRET,
            secure=True
        )
    
    def upload_enrollment_image(
        self, 
        image_data: bytes, 
        user_id: int, 
        image_number: int,
        username: str
    ) -> Dict:
        """
        Upload enrollment image to Cloudinary
        
        Args:
            image_data: Image bytes
            user_id: User ID
            image_number: Image number (1, 2, or 3)
            username: Username for folder organization
        
        Returns:
            dict: {url, public_id, success}
        """
        try:
            folder = f"enrollments/{username}_user_{user_id}"
            public_id = f"image_{image_number}"
            
            result = cloudinary.uploader.upload(
                image_data,
                folder=folder,
                public_id=public_id,
                overwrite=True,
                resource_type="image",
                format="jpg"
            )
            
            return {
                "url": result.get("secure_url"),
                "public_id": result.get("public_id"),
                "success": True
            }
        except Exception as e:
            print(f"Cloudinary upload error: {e}")
            return {
                "url": None,
                "public_id": None,
                "success": False,
                "error": str(e)
            }
    
    def delete_user_images(self, user_id: int, username: str) -> bool:
        """Delete all images for a user"""
        try:
            folder = f"enrollments/{username}_user_{user_id}"
            cloudinary.api.delete_resources_by_prefix(folder)
            return True
        except Exception as e:
            print(f"Cloudinary delete error: {e}")
            return False
    
    def get_image_url(self, public_id: str) -> Optional[str]:
        """Get secure URL for an image"""
        try:
            return cloudinary.CloudinaryImage(public_id).build_url(secure=True)
        except Exception as e:
            print(f"Error getting image URL: {e}")
            return None


# Singleton instance
cloudinary_service = CloudinaryService()