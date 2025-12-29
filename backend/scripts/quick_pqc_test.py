"""
Quick PQC Encryption Test - Checks core functionality
======================================================
Tests encryption/decryption without pytest dependency
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import secrets
import traceback

# Track test results
passed = 0
failed = 0
errors = []

def test(name, func):
    """Run a test and track results"""
    global passed, failed, errors
    try:
        result = func()
        if result:
            print(f"✅ PASS: {name}")
            passed += 1
        else:
            print(f"❌ FAIL: {name} - returned False")
            failed += 1
            errors.append((name, "Test returned False"))
    except Exception as e:
        print(f"❌ ERROR: {name}")
        print(f"   {type(e).__name__}: {e}")
        failed += 1
        errors.append((name, f"{type(e).__name__}: {e}"))


def main():
    global passed, failed, errors
    
    print("=" * 60)
    print("  PQC Quick Test - Encryption/Decryption Check")
    print("=" * 60)
    print()
    
    # Import services
    print("Loading services...")
    try:
        from app.services.pqc_service import pqc_service
        print("✓ PQC Service loaded")
    except Exception as e:
        print(f"✗ Failed to load PQC Service: {e}")
        return
    
    try:
        from app.services.hybrid_crypto_service import hybrid_crypto_service
        print("✓ Hybrid Crypto Service loaded")
    except Exception as e:
        print(f"✗ Failed to load Hybrid Crypto Service: {e}")
        return
    
    print()
    print("-" * 60)
    print("NTRU Tests")
    print("-" * 60)
    
    # Test 1: NTRU Key Generation
    def test_ntru_keygen():
        pub, priv = pqc_service.generate_ntru_keypair()
        return pub is not None and priv is not None and len(pub) > 0 and len(priv) > 0
    test("NTRU Key Generation", test_ntru_keygen)
    
    # Test 2: NTRU Encapsulation
    def test_ntru_encap():
        pub, priv = pqc_service.generate_ntru_keypair()
        ct, ss = pqc_service.ntru_encapsulate(pub)
        return ct is not None and ss is not None and len(ss) == 32
    test("NTRU Encapsulation", test_ntru_encap)
    
    # Test 3: NTRU Decapsulation (THIS IS THE KEY TEST)
    def test_ntru_decap():
        pub, priv = pqc_service.generate_ntru_keypair()
        ct, ss1 = pqc_service.ntru_encapsulate(pub)
        ss2 = pqc_service.ntru_decapsulate(ct, priv)
        match = ss1 == ss2
        if not match:
            print(f"   Encap secret: {ss1[:16].hex()}...")
            print(f"   Decap secret: {ss2[:16].hex()}...")
        return match
    test("NTRU Encap/Decap Match", test_ntru_decap)
    
    # Test 4: NTRU Encrypt/Decrypt
    def test_ntru_encrypt():
        pub, priv = pqc_service.generate_ntru_keypair()
        data = b"Test message for NTRU"
        ct = pqc_service.ntru_encrypt(data, pub)
        pt = pqc_service.ntru_decrypt(ct, priv)
        return data == pt
    test("NTRU Encrypt/Decrypt", test_ntru_encrypt)
    
    print()
    print("-" * 60)
    print("Kyber Tests")
    print("-" * 60)
    
    # Test 5: Kyber Key Generation
    def test_kyber_keygen():
        pub, priv = pqc_service.generate_kyber_keypair()
        return pub is not None and priv is not None and len(pub) > 0 and len(priv) > 0
    test("Kyber Key Generation", test_kyber_keygen)
    
    # Test 6: Kyber Encapsulation
    def test_kyber_encap():
        pub, priv = pqc_service.generate_kyber_keypair()
        ct, ss = pqc_service.kyber_encapsulate(pub)
        return ct is not None and ss is not None and len(ss) == 32
    test("Kyber Encapsulation", test_kyber_encap)
    
    # Test 7: Kyber Decapsulation (KEY TEST)
    def test_kyber_decap():
        pub, priv = pqc_service.generate_kyber_keypair()
        ct, ss1 = pqc_service.kyber_encapsulate(pub)
        ss2 = pqc_service.kyber_decapsulate(ct, priv)
        match = ss1 == ss2
        if not match:
            print(f"   Encap secret: {ss1[:16].hex()}...")
            print(f"   Decap secret: {ss2[:16].hex()}...")
        return match
    test("Kyber Encap/Decap Match", test_kyber_decap)
    
    # Test 8: Kyber Encrypt/Decrypt
    def test_kyber_encrypt():
        pub, priv = pqc_service.generate_kyber_keypair()
        data = b"Test message for Kyber"
        ct = pqc_service.kyber_encrypt(data, pub)
        pt = pqc_service.kyber_decrypt(ct, priv)
        return data == pt
    test("Kyber Encrypt/Decrypt", test_kyber_encrypt)
    
    print()
    print("-" * 60)
    print("Hybrid Encryption Tests")
    print("-" * 60)
    
    # Test 9: Hybrid (pqc_service)
    def test_hybrid_pqc():
        ntru_pub, ntru_priv = pqc_service.generate_ntru_keypair()
        kyber_pub, kyber_priv = pqc_service.generate_kyber_keypair()
        data = b"Hybrid test message via PQC service"
        ct = pqc_service.hybrid_encrypt(data, ntru_pub, kyber_pub)
        pt = pqc_service.hybrid_decrypt(ct, ntru_priv, kyber_priv)
        return data == pt
    test("Hybrid via PQC Service", test_hybrid_pqc)
    
    # Test 10: Hybrid (hybrid_crypto_service)
    def test_hybrid_crypto():
        ntru_pub, ntru_priv = pqc_service.generate_ntru_keypair()
        kyber_pub, kyber_priv = pqc_service.generate_kyber_keypair()
        data = b"Hybrid test message via Hybrid Crypto service"
        ct = hybrid_crypto_service.encrypt(data, ntru_pub, kyber_pub, compress=False)
        pt = hybrid_crypto_service.decrypt(ct, ntru_priv, kyber_priv)
        return data == pt
    test("Hybrid via Hybrid Crypto Service", test_hybrid_crypto)
    
    print()
    print("-" * 60)
    print("Signature Tests")
    print("-" * 60)
    
    # Test 11: ML-DSA (Dilithium) signatures
    def test_dilithium():
        pub, priv = pqc_service.generate_dilithium_keypair()
        message = b"Message to sign"
        sig = pqc_service.sign_dilithium(message, priv)
        return pqc_service.verify_dilithium(message, sig, pub)
    test("ML-DSA (Dilithium) Sign/Verify", test_dilithium)
    
    # Test 12: SPHINCS+ signatures
    def test_sphincs():
        pub, priv = pqc_service.generate_sphincs_keypair()
        message = b"Message to sign with SPHINCS+"
        sig = pqc_service.sign_sphincs(message, priv)
        return pqc_service.verify_sphincs(message, sig, pub)
    test("SPHINCS+ Sign/Verify", test_sphincs)
    
    # Summary
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {passed + failed}")
    print()
    
    if errors:
        print("ERRORS FOUND:")
        for name, error in errors:
            print(f"  • {name}: {error}")
        print()
    
    if failed == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        print()
        print("DIAGNOSIS:")
        print("-" * 40)
        print("The failures are likely due to liboqs-python not being installed.")
        print("The fallback mode generates random values that don't match on decryption.")
        print()
        print("To fix: Install liboqs-python (requires building from source on Windows)")
        print("Or: Use WSL2 with: pip install liboqs-python")
    
    return failed == 0


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
