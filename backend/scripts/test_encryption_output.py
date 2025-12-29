#!/usr/bin/env python3
"""
Test Encryption/Decryption Demo
Creates a test message, encrypts it with PQC (NTRU + Kyber hybrid),
decrypts it, and saves the results to an output file.
"""

import sys
import os
from datetime import datetime

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.pqc_service import PQCService
from app.services.hybrid_crypto_service import HybridCryptoService

def main():
    print("=" * 70)
    print("  POST-QUANTUM CRYPTOGRAPHY - ENCRYPTION/DECRYPTION DEMO")
    print("=" * 70)
    print()
    
    # Initialize services
    print("Initializing PQC Services...")
    pqc_service = PQCService()
    hybrid_service = HybridCryptoService()
    print()
    
    # Test message
    test_message = """
================================================================================
                    REQUAGNIZE - FACE AUTHENTICATION SYSTEM
================================================================================

This is a secret test message encrypted using Post-Quantum Cryptography!

Algorithms Used:
  • NTRU-HPS-2048-509: Primary Key Encapsulation Mechanism
  • Kyber768: Secondary Key Encapsulation Mechanism
  • AES-256-GCM: Symmetric encryption for the actual data

The hybrid approach uses BOTH NTRU and Kyber to derive the symmetric key,
providing defense-in-depth against future quantum computer attacks.

Test conducted at: {timestamp}

Security Level: NIST Level 3 (equivalent to AES-192 classical security)

This message demonstrates that the encryption and decryption process works
correctly with the liboqs-python library.

================================================================================
""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    original_bytes = test_message.encode('utf-8')
    
    print("-" * 70)
    print("ORIGINAL MESSAGE")
    print("-" * 70)
    print(test_message)
    print()
    
    # Generate key pairs
    print("-" * 70)
    print("GENERATING PQC KEY PAIRS")
    print("-" * 70)
    
    print("Generating NTRU key pair...")
    ntru_pub, ntru_priv = pqc_service.generate_ntru_keypair()
    print(f"  NTRU Public Key:  {len(ntru_pub)} bytes")
    print(f"  NTRU Private Key: {len(ntru_priv)} bytes")
    
    print("Generating Kyber key pair...")
    kyber_pub, kyber_priv = pqc_service.generate_kyber_keypair()
    print(f"  Kyber Public Key:  {len(kyber_pub)} bytes")
    print(f"  Kyber Private Key: {len(kyber_priv)} bytes")
    print()
    
    # Encrypt
    print("-" * 70)
    print("ENCRYPTING MESSAGE")
    print("-" * 70)
    
    ciphertext = hybrid_service.encrypt(original_bytes, ntru_pub, kyber_pub, compress=True)
    print(f"Original size:   {len(original_bytes)} bytes")
    print(f"Ciphertext size: {len(ciphertext)} bytes")
    print(f"Ciphertext (first 100 bytes hex): {ciphertext[:100].hex()}")
    print()
    
    # Decrypt
    print("-" * 70)
    print("DECRYPTING MESSAGE")
    print("-" * 70)
    
    decrypted_bytes = hybrid_service.decrypt(ciphertext, ntru_priv, kyber_priv)
    decrypted_message = decrypted_bytes.decode('utf-8')
    
    print(f"Decrypted size: {len(decrypted_bytes)} bytes")
    print(f"Messages match: {original_bytes == decrypted_bytes}")
    print()
    
    print("-" * 70)
    print("DECRYPTED MESSAGE")
    print("-" * 70)
    print(decrypted_message)
    print()
    
    # Save to output file
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_output')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'decrypted_message.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  DECRYPTION OUTPUT - REQUAGNIZE PQC SYSTEM\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Encryption: NTRU-HPS-2048-509 + Kyber768 Hybrid\n")
        f.write(f"Original Size: {len(original_bytes)} bytes\n")
        f.write(f"Ciphertext Size: {len(ciphertext)} bytes\n")
        f.write(f"Verification: {'SUCCESS - Messages match!' if original_bytes == decrypted_bytes else 'FAILED'}\n")
        f.write("\n" + "-" * 70 + "\n")
        f.write("DECRYPTED MESSAGE CONTENT:\n")
        f.write("-" * 70 + "\n")
        f.write(decrypted_message)
        f.write("\n" + "=" * 70 + "\n")
    
    print("-" * 70)
    print("OUTPUT FILE SAVED")
    print("-" * 70)
    print(f"Decrypted message saved to: {os.path.abspath(output_file)}")
    print()
    
    # Verify
    print("=" * 70)
    print("  VERIFICATION RESULT")
    print("=" * 70)
    if original_bytes == decrypted_bytes:
        print("✅ SUCCESS! Original and decrypted messages are IDENTICAL!")
        print("   Post-Quantum Cryptography is working correctly.")
    else:
        print("❌ FAILURE! Messages do not match!")
        return 1
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
