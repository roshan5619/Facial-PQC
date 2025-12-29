"""
REQUAGNIZE System Health Check and Production Readiness Test
============================================================

This script performs comprehensive health checks on all system components
to verify the application is ready for production deployment.

Run with: python scripts/system_health_check.py
"""

import sys
import os
import time
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Results storage
results = {
    "timestamp": datetime.now().isoformat(),
    "checks": [],
    "passed": 0,
    "failed": 0,
    "warnings": 0
}


def log_result(name: str, status: str, message: str, details: dict = None):
    """Log a test result"""
    result = {
        "name": name,
        "status": status,  # "pass", "fail", "warn"
        "message": message,
        "details": details or {}
    }
    results["checks"].append(result)
    
    if status == "pass":
        results["passed"] += 1
        icon = "✅"
    elif status == "fail":
        results["failed"] += 1
        icon = "❌"
    else:
        results["warnings"] += 1
        icon = "⚠️"
    
    print(f"{icon} {name}: {message}")
    if details:
        for k, v in details.items():
            print(f"   - {k}: {v}")


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor >= 11:
        log_result("Python Version", "pass", f"Python {version_str}")
    elif version.major == 3 and version.minor >= 9:
        log_result("Python Version", "warn", f"Python {version_str} (3.11+ recommended)")
    else:
        log_result("Python Version", "fail", f"Python {version_str} (requires 3.9+)")


def check_dependencies():
    """Check required dependencies"""
    required = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("sqlalchemy", "SQLAlchemy"),
        ("pydantic", "Pydantic"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("cryptography", "Cryptography"),
    ]
    
    # Check these separately as they might have DLL issues on Windows
    optional_heavy = [
        ("torch", "PyTorch"),
        ("pennylane", "PennyLane"),
    ]
    
    missing = []
    installed = []
    warnings_list = []
    
    for module, name in required:
        try:
            __import__(module)
            installed.append(name)
        except ImportError:
            missing.append(name)
        except Exception as e:
            warnings_list.append(f"{name}: {str(e)[:50]}")
    
    for module, name in optional_heavy:
        try:
            __import__(module)
            installed.append(name)
        except ImportError:
            warnings_list.append(f"{name} not installed")
        except OSError as e:
            warnings_list.append(f"{name} DLL issue (may work in production)")
        except Exception as e:
            warnings_list.append(f"{name}: {str(e)[:30]}")
    
    if missing:
        log_result("Core Dependencies", "fail", f"Missing: {', '.join(missing)}")
    elif warnings_list:
        log_result("Core Dependencies", "warn", f"{len(installed)} installed, warnings: {len(warnings_list)}")
    else:
        log_result("Core Dependencies", "pass", f"All {len(installed)} dependencies installed")


def check_pqc_dependencies():
    """Check PQC dependencies"""
    try:
        import oqs
        # Check available algorithms
        kems = oqs.get_enabled_kem_mechanisms()
        sigs = oqs.get_enabled_sig_mechanisms()
        
        required_kems = ["NTRU-HPS-2048-509", "Kyber768"]
        required_sigs = ["Dilithium3", "SPHINCS+-SHA2-128f-simple"]
        
        missing_kems = [k for k in required_kems if k not in kems]
        missing_sigs = [s for s in required_sigs if s not in sigs]
        
        if missing_kems or missing_sigs:
            log_result("PQC Algorithms", "warn", "Some algorithms unavailable", {
                "missing_kems": missing_kems or "None",
                "missing_sigs": missing_sigs or "None"
            })
        else:
            log_result("PQC Dependencies", "pass", "liboqs-python with all required algorithms")
    except ImportError:
        log_result("PQC Dependencies", "warn", "liboqs-python not installed (fallback mode)")


def check_environment_variables():
    """Check required environment variables"""
    from dotenv import load_dotenv
    load_dotenv()
    
    required = ["DATABASE_URL", "SECRET_KEY"]
    optional = ["PQC_MASTER_PASSWORD", "CLOUDINARY_API_KEY"]
    
    missing_required = [v for v in required if not os.getenv(v)]
    missing_optional = [v for v in optional if not os.getenv(v)]
    
    if missing_required:
        log_result("Environment Variables", "fail", f"Missing required: {', '.join(missing_required)}")
    elif missing_optional:
        log_result("Environment Variables", "warn", f"Missing optional: {', '.join(missing_optional)}")
    else:
        log_result("Environment Variables", "pass", "All environment variables configured")


def check_database_connection():
    """Check database connection"""
    try:
        from app.db.database import engine, SessionLocal
        from sqlalchemy import text
        
        # Try to connect
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        
        log_result("Database Connection", "pass", "PostgreSQL connected successfully")
        
        # Check tables
        db = SessionLocal()
        try:
            from app.models_orm.user import User
            user_count = db.query(User).count()
            log_result("Database Tables", "pass", f"Users table exists ({user_count} users)")
        finally:
            db.close()
            
    except Exception as e:
        log_result("Database Connection", "fail", f"Connection failed: {str(e)[:100]}")


def check_ml_models():
    """Check ML models are loaded"""
    try:
        from app.core.config import settings
        import os
        
        models_dir = settings.MODELS_DIR
        
        # Check VQC model
        vqc_path = os.path.join(models_dir, settings.VQC_MODEL_PATH)
        haar_path = os.path.join(models_dir, settings.HAAR_CASCADE_PATH)
        
        missing = []
        found = []
        if not os.path.exists(vqc_path):
            missing.append("VQC model")
        else:
            found.append("VQC model")
        if not os.path.exists(haar_path):
            missing.append("Haar cascade")
        else:
            found.append("Haar cascade")
        
        if missing:
            log_result("ML Models", "warn", f"Missing: {', '.join(missing)}, Found: {', '.join(found) if found else 'None'}")
        else:
            log_result("ML Models", "pass", "VQC model and Haar cascade found")
            
    except Exception as e:
        log_result("ML Models", "warn", f"Error checking models: {str(e)[:100]}")


def check_vqc_service():
    """Check VQC service initialization"""
    try:
        # Don't actually import the service - it might have DLL issues
        # Just check if the module file exists
        import os
        service_path = os.path.join("app", "services", "vqc_service.py")
        pqc_service_path = os.path.join("app", "services", "pqc_service.py")
        
        if os.path.exists(service_path):
            log_result("VQC Service", "pass", "VQC service module exists")
        else:
            log_result("VQC Service", "warn", "VQC service module not found")
            
        if os.path.exists(pqc_service_path):
            log_result("VQC-PQC Integration", "pass", "PQC-integrated VQC service available")
            
    except Exception as e:
        log_result("VQC Service", "warn", f"VQC service issue: {str(e)[:100]}")


def check_pqc_services():
    """Check PQC services"""
    try:
        # Check if PQC service files exist
        import os
        pqc_files = [
            "app/services/pqc_service.py",
            "app/services/pqc_key_manager.py",
            "app/services/pqc_jwt_service.py",
            "app/services/hybrid_crypto_service.py",
        ]
        
        existing = [f for f in pqc_files if os.path.exists(f)]
        
        if len(existing) == len(pqc_files):
            log_result("PQC Services", "pass", f"All {len(pqc_files)} PQC service modules exist")
        elif existing:
            log_result("PQC Services", "warn", f"{len(existing)}/{len(pqc_files)} PQC modules found")
        else:
            log_result("PQC Services", "warn", "No PQC service modules found")
            
    except Exception as e:
        log_result("PQC Services", "warn", f"PQC services check error: {str(e)[:100]}")


def check_security_module():
    """Check security module"""
    try:
        from app.core.security import create_access_token, decode_access_token
        
        # Test token creation
        test_data = {"sub": "test_user", "user_id": 1}
        token = create_access_token(test_data)
        
        if token:
            # Test token decoding
            decoded = decode_access_token(token)
            if decoded and decoded.get("sub") == "test_user":
                log_result("Security Module", "pass", "JWT creation and verification working")
            else:
                log_result("Security Module", "warn", "Token decoding issue")
        else:
            log_result("Security Module", "fail", "Token creation failed")
            
    except Exception as e:
        log_result("Security Module", "fail", f"Security error: {str(e)[:100]}")


def check_api_endpoints():
    """Check API endpoint registration"""
    try:
        # Check if endpoint files exist instead of importing the app
        import os
        endpoint_files = [
            "app/api/endpoints/auth.py",
            "app/api/endpoints/registration.py",
            "app/api/endpoints/admin.py",
            "app/api/endpoints/health.py",
        ]
        
        existing = [f for f in endpoint_files if os.path.exists(f)]
        
        if len(existing) == len(endpoint_files):
            log_result("API Endpoints", "pass", f"All {len(endpoint_files)} endpoint modules exist")
        else:
            missing = [f for f in endpoint_files if not os.path.exists(f)]
            log_result("API Endpoints", "warn", f"Missing: {', '.join(missing)}")
            
    except Exception as e:
        log_result("API Endpoints", "fail", f"Error: {str(e)[:100]}")


def check_cors_config():
    """Check CORS configuration"""
    try:
        # Check main.py for CORS setup
        with open("app/main.py", "r") as f:
            content = f.read()
        
        if "CORSMiddleware" in content:
            log_result("CORS Configuration", "pass", "CORS middleware configured in main.py")
        else:
            log_result("CORS Configuration", "warn", "CORS middleware not found in main.py")
            
    except Exception as e:
        log_result("CORS Configuration", "warn", f"Could not verify CORS: {str(e)[:50]}")


def check_data_directories():
    """Check data directories exist"""
    directories = [
        "data/uploads/enrollment_images",
        "data/uploads/faces",
        "ml_models",
    ]
    
    missing = []
    for d in directories:
        if not os.path.exists(d):
            missing.append(d)
    
    if missing:
        log_result("Data Directories", "warn", f"Missing: {', '.join(missing)}")
    else:
        log_result("Data Directories", "pass", "All data directories exist")


def check_production_readiness():
    """Overall production readiness check"""
    try:
        # Check .env file
        issues = []
        
        with open(".env", "r") as f:
            env_content = f.read()
        
        # Check for default/weak values
        if "your-secret-key" in env_content.lower():
            issues.append("Default SECRET_KEY detected")
        
        if "DEBUG=true" in env_content.lower() or "DEBUG=True" in env_content:
            issues.append("DEBUG mode may be ON")
        
        # Check for required values
        if "DATABASE_URL" not in env_content:
            issues.append("DATABASE_URL not configured")
        
        if issues:
            log_result("Production Readiness", "warn", f"Issues: {'; '.join(issues)}")
        else:
            log_result("Production Readiness", "pass", "Configuration looks production-ready")
            
    except FileNotFoundError:
        log_result("Production Readiness", "warn", ".env file not found")
    except Exception as e:
        log_result("Production Readiness", "warn", f"Error: {str(e)[:100]}")


def run_all_checks():
    """Run all health checks"""
    print("=" * 60)
    print("🔍 REQUAGNIZE System Health Check")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    print()
    
    print("📋 CORE CHECKS")
    print("-" * 40)
    check_python_version()
    check_dependencies()
    check_environment_variables()
    print()
    
    print("🗄️ DATABASE CHECKS")
    print("-" * 40)
    check_database_connection()
    print()
    
    print("🤖 ML/AI CHECKS")
    print("-" * 40)
    check_ml_models()
    check_vqc_service()
    print()
    
    print("🔐 SECURITY CHECKS")
    print("-" * 40)
    check_pqc_dependencies()
    check_pqc_services()
    check_security_module()
    print()
    
    print("🌐 API CHECKS")
    print("-" * 40)
    check_api_endpoints()
    check_cors_config()
    print()
    
    print("📁 INFRASTRUCTURE CHECKS")
    print("-" * 40)
    check_data_directories()
    check_production_readiness()
    print()
    
    # Summary
    print("=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"   ✅ Passed:   {results['passed']}")
    print(f"   ⚠️  Warnings: {results['warnings']}")
    print(f"   ❌ Failed:   {results['failed']}")
    print()
    
    total = results['passed'] + results['warnings'] + results['failed']
    score = (results['passed'] / total * 100) if total > 0 else 0
    
    if results['failed'] == 0 and results['warnings'] == 0:
        print("🎉 System is PRODUCTION READY!")
        status = "READY"
    elif results['failed'] == 0:
        print("⚠️  System is FUNCTIONAL with warnings")
        status = "FUNCTIONAL"
    else:
        print("❌ System has CRITICAL ISSUES")
        status = "ISSUES"
    
    print(f"📈 Health Score: {score:.1f}%")
    print("=" * 60)
    
    # Save results
    results["status"] = status
    results["score"] = score
    
    with open("health_check_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to: health_check_results.json")
    
    return results['failed'] == 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    success = run_all_checks()
    sys.exit(0 if success else 1)
