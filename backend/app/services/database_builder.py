import os
import sys
import subprocess
from typing import List
from app.core.config import settings
import asyncio

class DatabaseBuilder:
    """
    Wrapper for your db_creation2.py script
    Automatically rebuilds recognition database when new users enroll
    """
    
    def __init__(self):
        self.models_dir = settings.MODELS_DIR
        self.db_creation_script = os.path.join(self.models_dir, "db_creation2.py")
    
    async def rebuild_database(self, image_paths: List[str] = None) -> bool:
        """
        Run db_creation2.py to rebuild recognition database
        
        Args:
            image_paths: Optional list of new image paths to add
        
        Returns:
            bool: Success status
        """
        try:
            print("🔄 Rebuilding recognition database...")
            
            # Check if db_creation2.py exists
            if not os.path.exists(self.db_creation_script):
                print(f"✗ db_creation2.py not found at: {self.db_creation_script}")
                return False
            
            # Run db_creation2.py as subprocess
            # This script will recreate all .pkl files
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                self.db_creation_script,
                cwd=self.models_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                print("✓ Database rebuilt successfully")
                print(stdout.decode())
                
                # Reload models in VQC service
                from app.services.vqc_service import vqc_service
                vqc_service.reload_recognition_models()
                
                return True
            else:
                print(f"✗ Database rebuild failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            print(f"✗ Error rebuilding database: {e}")
            return False
    
    def get_database_stats(self) -> dict:
        """Get current database statistics"""
        try:
            from app.services.vqc_service import vqc_service
            
            return {
                'total_embeddings': vqc_service.recognition_db.shape[0],
                'pca_components': vqc_service.pca_recognition.n_components,
                'threshold': settings.COSINE_THRESHOLD
            }
        except Exception as e:
            return {'error': str(e)}


# Singleton instance
database_builder = DatabaseBuilder()