"""
API Endpoint Test Script
========================

This script tests all API endpoints to verify production readiness.
Run after starting the server with: uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def print_result(name, success, message, response=None):
    icon = "✅" if success else "❌"
    print(f"{icon} {name}: {message}")
    if response and not success:
        print(f"   Status: {response.status_code}")
        try:
            print(f"   Response: {response.json()}")
        except:
            print(f"   Response: {response.text[:200]}")

def test_root():
    """Test root endpoint"""
    try:
        r = requests.get(f"{BASE_URL}/")
        if r.status_code == 200:
            data = r.json()
            if "message" in data and "endpoints" in data:
                print_result("Root Endpoint", True, "API info returned")
                return True
        print_result("Root Endpoint", False, "Unexpected response", r)
        return False
    except Exception as e:
        print_result("Root Endpoint", False, f"Error: {str(e)}")
        return False

def test_health():
    """Test health endpoint"""
    try:
        r = requests.get(f"{BASE_URL}/health")
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "healthy":
                print_result("Health Endpoint", True, f"Status: healthy, Timestamp: {data.get('timestamp', 'N/A')[:19]}")
                return True
        print_result("Health Endpoint", False, "Unexpected response", r)
        return False
    except Exception as e:
        print_result("Health Endpoint", False, f"Error: {str(e)}")
        return False

def test_docs():
    """Test OpenAPI documentation"""
    try:
        r = requests.get(f"{BASE_URL}/docs")
        if r.status_code == 200:
            print_result("API Documentation", True, "/docs accessible")
            return True
        print_result("API Documentation", False, "Not accessible", r)
        return False
    except Exception as e:
        print_result("API Documentation", False, f"Error: {str(e)}")
        return False

def test_openapi():
    """Test OpenAPI schema"""
    try:
        r = requests.get(f"{BASE_URL}/openapi.json")
        if r.status_code == 200:
            data = r.json()
            paths = len(data.get("paths", {}))
            print_result("OpenAPI Schema", True, f"{paths} endpoints documented")
            return True
        print_result("OpenAPI Schema", False, "Not accessible", r)
        return False
    except Exception as e:
        print_result("OpenAPI Schema", False, f"Error: {str(e)}")
        return False

def test_login_no_image():
    """Test login without image (should fail gracefully)"""
    try:
        r = requests.post(f"{BASE_URL}/api/auth/login")
        # Should return 422 (validation error) or similar
        if r.status_code in [422, 400]:
            print_result("Login Validation", True, "Correctly rejects missing image")
            return True
        print_result("Login Validation", False, f"Unexpected status: {r.status_code}", r)
        return False
    except Exception as e:
        print_result("Login Validation", False, f"Error: {str(e)}")
        return False

def test_verify_no_token():
    """Test verify without token (should fail)"""
    try:
        r = requests.get(f"{BASE_URL}/api/auth/verify")
        if r.status_code in [401, 403, 422]:
            print_result("Token Verification", True, "Correctly rejects missing token")
            return True
        print_result("Token Verification", False, f"Unexpected status: {r.status_code}", r)
        return False
    except Exception as e:
        print_result("Token Verification", False, f"Error: {str(e)}")
        return False

def test_admin_unauthorized():
    """Test admin endpoint without auth (should fail)"""
    try:
        r = requests.get(f"{BASE_URL}/api/admin/users")
        if r.status_code in [401, 403]:
            print_result("Admin Authorization", True, "Correctly rejects unauthorized access")
            return True
        print_result("Admin Authorization", False, f"Unexpected status: {r.status_code}", r)
        return False
    except Exception as e:
        print_result("Admin Authorization", False, f"Error: {str(e)}")
        return False

def test_cors_headers():
    """Test CORS headers"""
    try:
        r = requests.options(f"{BASE_URL}/", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })
        cors_header = r.headers.get("Access-Control-Allow-Origin", "")
        if cors_header:
            print_result("CORS Headers", True, f"Allow-Origin: {cors_header}")
            return True
        print_result("CORS Headers", False, "No CORS headers found")
        return False
    except Exception as e:
        print_result("CORS Headers", False, f"Error: {str(e)}")
        return False

def test_response_time():
    """Test response time"""
    try:
        import time
        start = time.time()
        r = requests.get(f"{BASE_URL}/health")
        elapsed = (time.time() - start) * 1000
        
        if elapsed < 500:
            print_result("Response Time", True, f"{elapsed:.1f}ms (< 500ms threshold)")
            return True
        print_result("Response Time", False, f"{elapsed:.1f}ms (exceeds 500ms)")
        return False
    except Exception as e:
        print_result("Response Time", False, f"Error: {str(e)}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🧪 REQUAGNIZE API Endpoint Tests")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print("-" * 60)
    print()
    
    tests = [
        test_root,
        test_health,
        test_docs,
        test_openapi,
        test_login_no_image,
        test_verify_no_token,
        test_admin_unauthorized,
        test_cors_headers,
        test_response_time,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print_result(test.__name__, False, f"Exception: {str(e)}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"📊 Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed! API is ready for production.")
    else:
        print("⚠️ Some tests failed. Review issues above.")
    
    return failed == 0

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to server at", BASE_URL)
        print("   Make sure the server is running with:")
        print("   uvicorn app.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
