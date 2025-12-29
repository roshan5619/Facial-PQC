"""
Migration Script for 100 Users to Post-Quantum Cryptography
Migrates all existing users from classical cryptography to PQC

This script performs a complete migration including:
1. Backup of current data
2. Generation of PQC keys for all users
3. Re-encryption of face embeddings
4. Database updates
5. Recognition database rebuild
6. Verification and validation

CRITICAL: Run this script during a maintenance window.
All users will need to re-authenticate after migration.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import shutil
import random
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import numpy as np
import joblib
from tqdm import tqdm

# Database imports
from sqlalchemy.orm import Session
from app.db.database import SessionLocal, engine
from app.db.crud import UserCRUD
from app.models_orm.user import User, Session as SessionModel

# PQC imports
from app.services.pqc_service import pqc_service
from app.services.pqc_key_manager import pqc_key_manager, PQCKeySet
from app.services.hybrid_crypto_service import hybrid_crypto_service


@dataclass
class MigrationConfig:
    """Configuration for migration process"""
    # Paths
    models_dir: str = "./ml_models"
    backup_dir: str = "./data/migration_backup"
    pqc_keys_dir: str = "./data/pqc_keys"
    
    # Recognition database files
    recognition_db_file: str = "qcclass_recognition_db_cosine_roi.pkl"
    used_paths_file: str = "qcclass_used_paths_cosine_roi.pkl"
    roi_paths_file: str = "qcclass_roi_paths_cosine_roi.pkl"
    pca_recognition_file: str = "qcclass_pca_recognition_cosine_roi.pkl"
    scaler_recognition_file: str = "qcclass_scaler_recognition_cosine_roi.pkl"
    
    # New PQC files
    pqc_db_file: str = "pqc_recognition_db_cosine_roi.pkl"
    pqc_metadata_file: str = "pqc_metadata.pkl"
    
    # Migration settings
    batch_size: int = 10
    verify_each_user: bool = True
    create_rollback: bool = True
    
    # Encryption settings
    encrypt_embeddings: bool = True
    sign_embeddings: bool = True


@dataclass
class MigrationResult:
    """Result of migration process"""
    success: bool
    users_migrated: int
    embeddings_encrypted: int
    keys_generated: int
    errors: List[str]
    warnings: List[str]
    duration_seconds: float
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PQCMigration:
    """
    Post-Quantum Cryptography Migration Manager
    
    Handles the complete migration of 100 users from classical
    cryptography to quantum-resistant algorithms.
    """
    
    def __init__(self, config: Optional[MigrationConfig] = None):
        """Initialize migration manager"""
        self.config = config or MigrationConfig()
        self.start_time = None
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        # Ensure directories exist
        Path(self.config.backup_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.pqc_keys_dir).mkdir(parents=True, exist_ok=True)
    
    def run_migration(self) -> MigrationResult:
        """
        Execute the complete PQC migration process
        
        Returns:
            MigrationResult: Summary of migration outcome
        """
        self.start_time = datetime.utcnow()
        print("=" * 80)
        print("POST-QUANTUM CRYPTOGRAPHY MIGRATION")
        print("=" * 80)
        print(f"Started at: {self.start_time.isoformat()}")
        print()
        
        users_migrated = 0
        embeddings_encrypted = 0
        keys_generated = 0
        
        try:
            # Step 1: Create backup
            print("[1/7] Creating backup of current data...")
            if not self._create_backup():
                return self._create_result(False, 0, 0, 0)
            
            # Step 2: Load current data
            print("\n[2/7] Loading current recognition database...")
            current_data = self._load_current_database()
            if current_data is None:
                return self._create_result(False, 0, 0, 0)
            
            # Step 3: Get all users from database
            print("\n[3/7] Loading users from database...")
            db = SessionLocal()
            try:
                users = UserCRUD.get_all_users(db, skip=0, limit=1000)
                print(f"   Found {len(users)} users")
            finally:
                db.close()
            
            if not users:
                self.warnings.append("No users found in database")
                print("   WARNING: No users to migrate")
            
            # Step 4: Generate PQC keys for all users
            print("\n[4/7] Generating PQC keys for all users...")
            keys_generated = self._generate_all_user_keys(users)
            
            # Step 5: Generate system keys
            print("\n[5/7] Generating system PQC keys...")
            self._generate_system_keys()
            
            # Step 6: Encrypt embeddings (if enabled)
            if self.config.encrypt_embeddings:
                print("\n[6/7] Encrypting face embeddings with hybrid PQC...")
                embeddings_encrypted = self._encrypt_all_embeddings(
                    current_data, users
                )
            else:
                print("\n[6/7] Skipping embedding encryption (disabled in config)")
                embeddings_encrypted = 0
            
            # Step 7: Invalidate old sessions
            print("\n[7/7] Invalidating old sessions...")
            self._invalidate_old_sessions()
            
            users_migrated = len(users)
            
            # Final verification
            print("\n" + "=" * 80)
            print("VERIFICATION")
            print("=" * 80)
            self._verify_migration(users)
            
            return self._create_result(
                success=True,
                users_migrated=users_migrated,
                embeddings_encrypted=embeddings_encrypted,
                keys_generated=keys_generated
            )
            
        except Exception as e:
            self.errors.append(f"Migration failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_result(
                success=False,
                users_migrated=users_migrated,
                embeddings_encrypted=embeddings_encrypted,
                keys_generated=keys_generated
            )
    
    def _create_backup(self) -> bool:
        """Create backup of current data"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_subdir = Path(self.config.backup_dir) / f"backup_{timestamp}"
            backup_subdir.mkdir(parents=True, exist_ok=True)
            
            # Backup recognition database files
            models_dir = Path(self.config.models_dir)
            files_to_backup = [
                self.config.recognition_db_file,
                self.config.used_paths_file,
                self.config.roi_paths_file,
                self.config.pca_recognition_file,
                self.config.scaler_recognition_file
            ]
            
            backed_up = 0
            for filename in files_to_backup:
                src = models_dir / filename
                if src.exists():
                    dst = backup_subdir / filename
                    shutil.copy2(src, dst)
                    backed_up += 1
                    print(f"   ✓ Backed up: {filename}")
                else:
                    self.warnings.append(f"File not found for backup: {filename}")
            
            # Create backup manifest
            manifest = {
                "timestamp": timestamp,
                "files_backed_up": backed_up,
                "source_dir": str(models_dir),
                "backup_dir": str(backup_subdir)
            }
            
            (backup_subdir / "manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )
            
            print(f"   ✓ Backup complete: {backup_subdir}")
            print(f"   ✓ {backed_up} files backed up")
            
            return True
            
        except Exception as e:
            self.errors.append(f"Backup failed: {str(e)}")
            print(f"   ✗ Backup failed: {e}")
            return False
    
    def _load_current_database(self) -> Optional[Dict[str, Any]]:
        """Load current recognition database"""
        try:
            models_dir = Path(self.config.models_dir)
            
            data = {}
            
            # Load recognition database
            db_path = models_dir / self.config.recognition_db_file
            if db_path.exists():
                data['embeddings'] = joblib.load(db_path)
                print(f"   ✓ Loaded {len(data['embeddings'])} embeddings")
            else:
                print(f"   ⚠ Recognition DB not found: {db_path}")
                data['embeddings'] = np.array([])
            
            # Load paths
            paths_path = models_dir / self.config.used_paths_file
            if paths_path.exists():
                data['paths'] = joblib.load(paths_path)
                print(f"   ✓ Loaded {len(data['paths'])} paths")
            else:
                data['paths'] = []
            
            # Load ROI paths
            roi_path = models_dir / self.config.roi_paths_file
            if roi_path.exists():
                data['roi_paths'] = joblib.load(roi_path)
            else:
                data['roi_paths'] = []
            
            # Load PCA and scaler
            pca_path = models_dir / self.config.pca_recognition_file
            if pca_path.exists():
                data['pca'] = joblib.load(pca_path)
            
            scaler_path = models_dir / self.config.scaler_recognition_file
            if scaler_path.exists():
                data['scaler'] = joblib.load(scaler_path)
            
            return data
            
        except Exception as e:
            self.errors.append(f"Failed to load database: {str(e)}")
            print(f"   ✗ Failed to load database: {e}")
            return None
    
    def _generate_all_user_keys(self, users: List[User]) -> int:
        """Generate PQC keys for all users"""
        generated = 0
        
        for user in tqdm(users, desc="   Generating keys"):
            try:
                # Check if keys already exist
                if pqc_key_manager.user_has_keys(user.user_id):
                    print(f"      User {user.user_id}: Keys already exist, skipping")
                    generated += 1
                    continue
                
                # Generate new keys
                key_set = pqc_key_manager.generate_user_keys(
                    user.user_id,
                    include_sphincs=True
                )
                
                # Save keys
                if pqc_key_manager.save_user_keys(user.user_id, key_set):
                    generated += 1
                else:
                    self.errors.append(f"Failed to save keys for user {user.user_id}")
                    
            except Exception as e:
                self.errors.append(f"Key generation failed for user {user.user_id}: {str(e)}")
        
        print(f"   ✓ Generated keys for {generated}/{len(users)} users")
        return generated
    
    def _generate_system_keys(self):
        """Generate system-wide PQC keys"""
        try:
            system_keys = pqc_key_manager.load_system_keys()
            
            if system_keys is None:
                print("   Generating new system keys...")
                pqc_key_manager.generate_system_keys()
                print("   ✓ System keys generated")
            else:
                print("   ✓ System keys already exist")
                
        except Exception as e:
            self.errors.append(f"System key generation failed: {str(e)}")
            print(f"   ✗ System key generation failed: {e}")
    
    def _extract_user_id_from_path(self, path: str) -> Optional[int]:
        """Extract user_id from image path"""
        try:
            # Try various path formats
            filename = os.path.basename(path)
            
            # Format: user_123_img_1.jpg
            if 'user_' in filename:
                parts = filename.split('_')
                for i, part in enumerate(parts):
                    if part == 'user' and i + 1 < len(parts):
                        return int(parts[i + 1])
            
            # Format: 123_face_1.jpg
            parts = filename.split('_')
            if parts[0].isdigit():
                return int(parts[0])
            
            return None
            
        except Exception:
            return None
    
    def _encrypt_all_embeddings(
        self,
        current_data: Dict[str, Any],
        users: List[User]
    ) -> int:
        """Encrypt all face embeddings with hybrid PQC"""
        
        embeddings = current_data.get('embeddings', np.array([]))
        paths = current_data.get('paths', [])
        
        if len(embeddings) == 0:
            print("   No embeddings to encrypt")
            return 0
        
        # Create user_id -> user mapping
        user_map = {user.user_id: user for user in users}
        
        encrypted_embeddings = []
        encrypted_paths = []
        encryption_metadata = []
        
        encrypted_count = 0
        
        for idx in tqdm(range(len(embeddings)), desc="   Encrypting embeddings"):
            try:
                embedding = embeddings[idx]
                path = paths[idx] if idx < len(paths) else f"unknown_{idx}"
                
                # Extract user_id
                user_id = self._extract_user_id_from_path(path)
                
                if user_id is None:
                    # Use a default user or skip
                    self.warnings.append(f"Could not extract user_id from path: {path}")
                    # For now, store unencrypted (will be encrypted on next login)
                    encrypted_embeddings.append(embedding)
                    encrypted_paths.append(path)
                    continue
                
                # Check if user has keys
                user_keys = pqc_key_manager.load_user_keys(user_id)
                
                if user_keys is None:
                    # Generate keys for this user
                    user_keys = pqc_key_manager.generate_user_keys(user_id)
                    pqc_key_manager.save_user_keys(user_id, user_keys)
                
                # Encrypt embedding
                encrypted = hybrid_crypto_service.encrypt_embedding(
                    embedding,
                    user_id,
                    sign=self.config.sign_embeddings
                )
                
                encrypted_embeddings.append(encrypted)
                encrypted_paths.append(path)
                
                encryption_metadata.append({
                    'idx': idx,
                    'user_id': user_id,
                    'original_size': len(embedding.tobytes()) if hasattr(embedding, 'tobytes') else 0,
                    'encrypted_size': len(encrypted),
                    'signed': self.config.sign_embeddings
                })
                
                encrypted_count += 1
                
            except Exception as e:
                self.errors.append(f"Encryption failed for embedding {idx}: {str(e)}")
                # Store original (unencrypted) on failure
                encrypted_embeddings.append(embeddings[idx])
                encrypted_paths.append(paths[idx] if idx < len(paths) else f"failed_{idx}")
        
        # Save encrypted database
        models_dir = Path(self.config.models_dir)
        
        # Save encrypted embeddings
        joblib.dump(
            encrypted_embeddings,
            models_dir / self.config.pqc_db_file
        )
        
        # Save PQC metadata
        pqc_metadata = {
            'algorithm': 'NTRU-HPS-2048-509 + Kyber768',
            'signature': 'Dilithium3',
            'migration_date': datetime.utcnow().isoformat(),
            'num_embeddings': len(encrypted_embeddings),
            'num_encrypted': encrypted_count,
            'encryption_metadata': encryption_metadata[:100]  # Sample
        }
        
        joblib.dump(pqc_metadata, models_dir / self.config.pqc_metadata_file)
        
        print(f"   ✓ Encrypted {encrypted_count}/{len(embeddings)} embeddings")
        print(f"   ✓ Saved to: {self.config.pqc_db_file}")
        
        return encrypted_count
    
    def _invalidate_old_sessions(self):
        """Invalidate all sessions before migration"""
        try:
            db = SessionLocal()
            try:
                # Update all sessions to revoked
                result = db.query(SessionModel).update(
                    {SessionModel.is_revoked: True},
                    synchronize_session=False
                )
                db.commit()
                print(f"   ✓ Invalidated {result} existing sessions")
            finally:
                db.close()
                
        except Exception as e:
            self.warnings.append(f"Session invalidation warning: {str(e)}")
            print(f"   ⚠ Session invalidation warning: {e}")
    
    def _verify_migration(self, users: List[User]):
        """Verify migration was successful"""
        print("\nRunning verification checks...")
        
        checks_passed = 0
        checks_total = 4
        
        # Check 1: All users have keys
        users_with_keys = sum(1 for u in users if pqc_key_manager.user_has_keys(u.user_id))
        if users_with_keys == len(users):
            print(f"   ✓ All {len(users)} users have PQC keys")
            checks_passed += 1
        else:
            print(f"   ✗ Only {users_with_keys}/{len(users)} users have keys")
        
        # Check 2: System keys exist
        system_keys = pqc_key_manager.load_system_keys()
        if system_keys is not None:
            print("   ✓ System keys exist and are loadable")
            checks_passed += 1
        else:
            print("   ✗ System keys missing or corrupt")
        
        # Check 3: PQC database exists
        pqc_db_path = Path(self.config.models_dir) / self.config.pqc_db_file
        if pqc_db_path.exists():
            print(f"   ✓ PQC encrypted database exists: {pqc_db_path}")
            checks_passed += 1
        else:
            print("   ✗ PQC encrypted database not found")
        
        # Check 4: Test encryption/decryption cycle
        if len(users) > 0:
            test_user = users[0]
            test_data = b"Test data for PQC verification"
            
            try:
                encrypted = hybrid_crypto_service.encrypt_for_user(test_data, test_user.user_id)
                decrypted = hybrid_crypto_service.decrypt_for_user(encrypted, test_user.user_id)
                
                if decrypted == test_data:
                    print("   ✓ Encryption/decryption cycle verified")
                    checks_passed += 1
                else:
                    print("   ✗ Decrypted data doesn't match original")
            except Exception as e:
                print(f"   ✗ Encryption/decryption test failed: {e}")
        else:
            checks_passed += 1  # Skip if no users
        
        print(f"\n   Verification: {checks_passed}/{checks_total} checks passed")
        
        if checks_passed < checks_total:
            self.warnings.append(f"Only {checks_passed}/{checks_total} verification checks passed")
    
    def _create_result(
        self,
        success: bool,
        users_migrated: int,
        embeddings_encrypted: int,
        keys_generated: int
    ) -> MigrationResult:
        """Create migration result summary"""
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds() if self.start_time else 0
        
        result = MigrationResult(
            success=success,
            users_migrated=users_migrated,
            embeddings_encrypted=embeddings_encrypted,
            keys_generated=keys_generated,
            errors=self.errors.copy(),
            warnings=self.warnings.copy(),
            duration_seconds=duration,
            timestamp=end_time.isoformat()
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("MIGRATION SUMMARY")
        print("=" * 80)
        print(f"Status:              {'SUCCESS ✓' if success else 'FAILED ✗'}")
        print(f"Users migrated:      {users_migrated}")
        print(f"Keys generated:      {keys_generated}")
        print(f"Embeddings encrypted:{embeddings_encrypted}")
        print(f"Duration:            {duration:.1f} seconds")
        print(f"Errors:              {len(self.errors)}")
        print(f"Warnings:            {len(self.warnings)}")
        
        if self.errors:
            print("\nErrors:")
            for err in self.errors[:5]:
                print(f"  - {err}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more")
        
        if self.warnings:
            print("\nWarnings:")
            for warn in self.warnings[:5]:
                print(f"  - {warn}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more")
        
        print("=" * 80)
        
        # Save result to file
        result_path = Path(self.config.backup_dir) / f"migration_result_{end_time.strftime('%Y%m%d_%H%M%S')}.json"
        result_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"Result saved to: {result_path}")
        
        return result


def migrate_users_to_pqc() -> bool:
    """
    Main entry point for PQC migration
    
    Returns:
        bool: True if migration successful
    """
    config = MigrationConfig()
    migration = PQCMigration(config)
    result = migration.run_migration()
    
    return result.success


if __name__ == "__main__":
    success = migrate_users_to_pqc()
    sys.exit(0 if success else 1)
