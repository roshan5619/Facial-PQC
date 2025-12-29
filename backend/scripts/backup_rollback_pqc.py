"""
Backup and Rollback Scripts for PQC Migration
Provides data safety mechanisms for the migration process

Functions:
- Full backup of current cryptographic state
- Rollback to pre-PQC state
- Verification of backup integrity
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import joblib


@dataclass
class BackupManifest:
    """Manifest for backup tracking"""
    backup_id: str
    timestamp: str
    source_dir: str
    backup_dir: str
    files: List[Dict[str, Any]]
    database_backup: bool
    pqc_keys_backup: bool
    total_size_bytes: int
    checksum: str


class PQCBackupManager:
    """
    Backup Manager for PQC Migration
    
    Handles:
    - Complete backup of recognition database
    - PQC key backup
    - Rollback functionality
    - Integrity verification
    """
    
    def __init__(
        self,
        models_dir: str = "./ml_models",
        backup_base_dir: str = "./data/migration_backup",
        pqc_keys_dir: str = "./data/pqc_keys"
    ):
        self.models_dir = Path(models_dir)
        self.backup_base_dir = Path(backup_base_dir)
        self.pqc_keys_dir = Path(pqc_keys_dir)
        
        self.backup_base_dir.mkdir(parents=True, exist_ok=True)
    
    def create_full_backup(self, backup_name: Optional[str] = None) -> Optional[BackupManifest]:
        """
        Create a full backup of the current state
        
        Args:
            backup_name: Optional name for the backup
            
        Returns:
            BackupManifest if successful, None otherwise
        """
        timestamp = datetime.utcnow()
        backup_id = backup_name or f"backup_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        backup_dir = self.backup_base_dir / backup_id
        
        print(f"Creating full backup: {backup_id}")
        print("=" * 60)
        
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            files_backed_up = []
            total_size = 0
            
            # Backup ML models
            ml_backup_dir = backup_dir / "ml_models"
            ml_backup_dir.mkdir(exist_ok=True)
            
            # Files to backup
            model_files = [
                "qcclass_recognition_db_cosine_roi.pkl",
                "qcclass_used_paths_cosine_roi.pkl",
                "qcclass_roi_paths_cosine_roi.pkl",
                "qcclass_pca_recognition_cosine_roi.pkl",
                "qcclass_scaler_recognition_cosine_roi.pkl",
                "pca_detection_roi.pkl",
                "vqc_face_model_roi.pth",
                # PQC files if they exist
                "pqc_recognition_db_cosine_roi.pkl",
                "pqc_metadata.pkl"
            ]
            
            for filename in model_files:
                src = self.models_dir / filename
                if src.exists():
                    dst = ml_backup_dir / filename
                    shutil.copy2(src, dst)
                    
                    file_size = src.stat().st_size
                    file_hash = self._calculate_file_hash(src)
                    
                    files_backed_up.append({
                        "filename": filename,
                        "path": str(src),
                        "size": file_size,
                        "hash": file_hash
                    })
                    
                    total_size += file_size
                    print(f"  ✓ {filename} ({file_size:,} bytes)")
            
            # Backup PQC keys if they exist
            pqc_keys_backed_up = False
            if self.pqc_keys_dir.exists():
                keys_backup_dir = backup_dir / "pqc_keys"
                shutil.copytree(self.pqc_keys_dir, keys_backup_dir)
                pqc_keys_backed_up = True
                
                # Count key files
                key_count = sum(1 for _ in keys_backup_dir.rglob("*.key*"))
                print(f"  ✓ PQC keys ({key_count} files)")
            
            # Calculate overall checksum
            overall_checksum = self._calculate_backup_checksum(backup_dir)
            
            # Create manifest
            manifest = BackupManifest(
                backup_id=backup_id,
                timestamp=timestamp.isoformat(),
                source_dir=str(self.models_dir),
                backup_dir=str(backup_dir),
                files=files_backed_up,
                database_backup=True,
                pqc_keys_backup=pqc_keys_backed_up,
                total_size_bytes=total_size,
                checksum=overall_checksum
            )
            
            # Save manifest
            manifest_path = backup_dir / "manifest.json"
            manifest_path.write_text(json.dumps(asdict(manifest), indent=2))
            
            print("=" * 60)
            print(f"Backup complete: {backup_dir}")
            print(f"Total size: {total_size:,} bytes")
            print(f"Files: {len(files_backed_up)}")
            print(f"Checksum: {overall_checksum[:16]}...")
            
            return manifest
            
        except Exception as e:
            print(f"✗ Backup failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def rollback(self, backup_id: str, verify_first: bool = True) -> bool:
        """
        Rollback to a previous backup state
        
        Args:
            backup_id: ID of backup to restore
            verify_first: Whether to verify backup integrity first
            
        Returns:
            bool: True if rollback successful
        """
        backup_dir = self.backup_base_dir / backup_id
        
        if not backup_dir.exists():
            print(f"✗ Backup not found: {backup_id}")
            return False
        
        print(f"Rolling back to: {backup_id}")
        print("=" * 60)
        
        # Load manifest
        manifest_path = backup_dir / "manifest.json"
        if not manifest_path.exists():
            print("✗ Manifest not found")
            return False
        
        manifest_data = json.loads(manifest_path.read_text())
        
        # Verify backup integrity
        if verify_first:
            print("Verifying backup integrity...")
            if not self.verify_backup(backup_id):
                print("✗ Backup verification failed, aborting rollback")
                return False
            print("✓ Backup verified")
        
        try:
            # Create rollback point (backup current state)
            print("\nCreating rollback point...")
            rollback_point = self.create_full_backup(
                f"pre_rollback_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            )
            
            if rollback_point is None:
                print("✗ Failed to create rollback point, aborting")
                return False
            
            # Restore ML models
            print("\nRestoring ML models...")
            ml_backup_dir = backup_dir / "ml_models"
            
            for file_info in manifest_data['files']:
                src = ml_backup_dir / file_info['filename']
                dst = self.models_dir / file_info['filename']
                
                if src.exists():
                    shutil.copy2(src, dst)
                    print(f"  ✓ Restored: {file_info['filename']}")
            
            # Restore PQC keys if backed up
            if manifest_data.get('pqc_keys_backup'):
                print("\nRestoring PQC keys...")
                keys_backup_dir = backup_dir / "pqc_keys"
                
                if keys_backup_dir.exists():
                    # Remove current keys
                    if self.pqc_keys_dir.exists():
                        shutil.rmtree(self.pqc_keys_dir)
                    
                    # Restore from backup
                    shutil.copytree(keys_backup_dir, self.pqc_keys_dir)
                    print("  ✓ PQC keys restored")
            
            print("=" * 60)
            print(f"✓ Rollback complete to: {backup_id}")
            return True
            
        except Exception as e:
            print(f"✗ Rollback failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def verify_backup(self, backup_id: str) -> bool:
        """
        Verify backup integrity
        
        Args:
            backup_id: ID of backup to verify
            
        Returns:
            bool: True if backup is valid
        """
        backup_dir = self.backup_base_dir / backup_id
        
        if not backup_dir.exists():
            print(f"✗ Backup not found: {backup_id}")
            return False
        
        manifest_path = backup_dir / "manifest.json"
        if not manifest_path.exists():
            print("✗ Manifest not found")
            return False
        
        manifest_data = json.loads(manifest_path.read_text())
        
        all_valid = True
        
        # Verify individual files
        ml_backup_dir = backup_dir / "ml_models"
        
        for file_info in manifest_data['files']:
            backup_file = ml_backup_dir / file_info['filename']
            
            if not backup_file.exists():
                print(f"  ✗ Missing: {file_info['filename']}")
                all_valid = False
                continue
            
            # Verify hash
            current_hash = self._calculate_file_hash(backup_file)
            if current_hash != file_info['hash']:
                print(f"  ✗ Hash mismatch: {file_info['filename']}")
                all_valid = False
            else:
                print(f"  ✓ Verified: {file_info['filename']}")
        
        # Verify overall checksum
        current_checksum = self._calculate_backup_checksum(backup_dir)
        if current_checksum != manifest_data['checksum']:
            print("  ✗ Overall checksum mismatch")
            all_valid = False
        
        return all_valid
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups"""
        backups = []
        
        for item in self.backup_base_dir.iterdir():
            if item.is_dir():
                manifest_path = item / "manifest.json"
                if manifest_path.exists():
                    manifest_data = json.loads(manifest_path.read_text())
                    backups.append({
                        "backup_id": manifest_data['backup_id'],
                        "timestamp": manifest_data['timestamp'],
                        "total_size": manifest_data['total_size_bytes'],
                        "files_count": len(manifest_data['files']),
                        "pqc_keys": manifest_data.get('pqc_keys_backup', False)
                    })
        
        return sorted(backups, key=lambda x: x['timestamp'], reverse=True)
    
    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup"""
        backup_dir = self.backup_base_dir / backup_id
        
        if not backup_dir.exists():
            print(f"Backup not found: {backup_id}")
            return False
        
        try:
            shutil.rmtree(backup_dir)
            print(f"✓ Deleted backup: {backup_id}")
            return True
        except Exception as e:
            print(f"✗ Failed to delete backup: {e}")
            return False
    
    def _calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA-256 hash of a file"""
        sha256 = hashlib.sha256()
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def _calculate_backup_checksum(self, backup_dir: Path) -> str:
        """Calculate overall checksum of backup"""
        sha256 = hashlib.sha256()
        
        # Hash all files in sorted order
        for filepath in sorted(backup_dir.rglob("*")):
            if filepath.is_file() and filepath.name != "manifest.json":
                sha256.update(filepath.name.encode())
                sha256.update(self._calculate_file_hash(filepath).encode())
        
        return sha256.hexdigest()


def create_backup():
    """Create a full backup"""
    manager = PQCBackupManager()
    manifest = manager.create_full_backup()
    return manifest is not None


def rollback_to_backup(backup_id: str):
    """Rollback to a specific backup"""
    manager = PQCBackupManager()
    return manager.rollback(backup_id)


def list_all_backups():
    """List all available backups"""
    manager = PQCBackupManager()
    backups = manager.list_backups()
    
    print("\nAvailable Backups:")
    print("=" * 80)
    
    for backup in backups:
        print(f"\n  ID: {backup['backup_id']}")
        print(f"  Timestamp: {backup['timestamp']}")
        print(f"  Size: {backup['total_size']:,} bytes")
        print(f"  Files: {backup['files_count']}")
        print(f"  PQC Keys: {'Yes' if backup['pqc_keys'] else 'No'}")
    
    return backups


def verify_backup(backup_id: str):
    """Verify a specific backup"""
    manager = PQCBackupManager()
    return manager.verify_backup(backup_id)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PQC Backup Manager")
    parser.add_argument("action", choices=["backup", "rollback", "list", "verify", "delete"])
    parser.add_argument("--backup-id", help="Backup ID for rollback/verify/delete")
    
    args = parser.parse_args()
    
    if args.action == "backup":
        create_backup()
    elif args.action == "rollback":
        if not args.backup_id:
            print("Error: --backup-id required for rollback")
            sys.exit(1)
        rollback_to_backup(args.backup_id)
    elif args.action == "list":
        list_all_backups()
    elif args.action == "verify":
        if not args.backup_id:
            print("Error: --backup-id required for verify")
            sys.exit(1)
        verify_backup(args.backup_id)
    elif args.action == "delete":
        if not args.backup_id:
            print("Error: --backup-id required for delete")
            sys.exit(1)
        manager = PQCBackupManager()
        manager.delete_backup(args.backup_id)
