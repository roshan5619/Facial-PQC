"""
Test Script for Hybrid PQC Encryption and Decryption
Tests the NTRU + Kyber hybrid encryption scheme
"""

import sys
import os
from datetime import datetime

# Add the app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.pqc_service import pqc_service
from app.services.hybrid_crypto_service import hybrid_crypto_service


def test_encryption_decryption():
    """Test encryption and decryption of a text message"""
    
    print("=" * 60)
    print("  Post-Quantum Cryptography - Encryption/Decryption Test")
    print("=" * 60)
    print()
    
    # =====================================================
    # Step 1: Create the test message
    # =====================================================
    original_message = """
    Hello! This is a secret test message for PQC encryption.
    
    Testing the hybrid NTRU + Kyber encryption system.
    This message demonstrates quantum-resistant cryptography.
    
    Date: {date}
    Purpose: Encryption/Decryption Verification
    Status: Testing Complete Security
    
    Additional test data:
    - Special characters: @#$%^&*()
    - Numbers: 1234567890
    - Unicode: café, naïve, 日本語
    """.format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    print("[1] ORIGINAL MESSAGE:")
    print("-" * 40)
    print(original_message)
    print("-" * 40)
    print()
    
    # Convert to bytes
    message_bytes = original_message.encode('utf-8')
    print(f"Message size: {len(message_bytes)} bytes")
    print()
    
    # =====================================================
    # Step 2: Generate Key Pairs
    # =====================================================
    print("[2] GENERATING PQC KEY PAIRS...")
    print("-" * 40)
    
    # Generate NTRU key pair
    ntru_public_key, ntru_private_key = pqc_service.generate_ntru_keypair()
    print(f"✓ NTRU Public Key:  {len(ntru_public_key)} bytes")
    print(f"✓ NTRU Private Key: {len(ntru_private_key)} bytes")
    
    # Generate Kyber key pair
    kyber_public_key, kyber_private_key = pqc_service.generate_kyber_keypair()
    print(f"✓ Kyber Public Key:  {len(kyber_public_key)} bytes")
    print(f"✓ Kyber Private Key: {len(kyber_private_key)} bytes")
    print()
    
    # =====================================================
    # Step 3: Encrypt the message
    # =====================================================
    print("[3] ENCRYPTING MESSAGE...")
    print("-" * 40)
    
    encrypted_data = hybrid_crypto_service.encrypt(
        data=message_bytes,
        ntru_public_key=ntru_public_key,
        kyber_public_key=kyber_public_key,
        compress=True
    )
    
    print(f"✓ Encrypted data size: {len(encrypted_data)} bytes")
    print(f"✓ Encryption overhead: {len(encrypted_data) - len(message_bytes)} bytes")
    print()
    
    # Show a snippet of encrypted data (hex format)
    print("Encrypted data preview (hex):")
    print(encrypted_data[:100].hex())
    print("...")
    print()
    
    # =====================================================
    # Step 4: Decrypt the message
    # =====================================================
    print("[4] DECRYPTING MESSAGE...")
    print("-" * 40)
    
    decrypted_data = hybrid_crypto_service.decrypt(
        encrypted_data=encrypted_data,
        ntru_private_key=ntru_private_key,
        kyber_private_key=kyber_private_key
    )
    
    decrypted_message = decrypted_data.decode('utf-8')
    
    print(f"✓ Decrypted data size: {len(decrypted_data)} bytes")
    print()
    
    # =====================================================
    # Step 5: Verify the result
    # =====================================================
    print("[5] VERIFICATION:")
    print("-" * 40)
    
    if original_message == decrypted_message:
        print("✓ SUCCESS: Decrypted message matches original!")
        verification_status = "PASSED"
    else:
        print("✗ FAILURE: Messages do not match!")
        verification_status = "FAILED"
    print()
    
    # =====================================================
    # Step 6: Display decrypted message
    # =====================================================
    print("[6] DECRYPTED MESSAGE:")
    print("-" * 40)
    print(decrypted_message)
    print("-" * 40)
    print()
    
    # =====================================================
    # Step 7: Save results to file
    # =====================================================
    output_file = os.path.join(os.path.dirname(__file__), 'decrypted_message_output.txt')
    
    results_content = f"""
================================================================================
           POST-QUANTUM CRYPTOGRAPHY - ENCRYPTION/DECRYPTION RESULTS
================================================================================

Test Executed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Verification Status: {verification_status}

--------------------------------------------------------------------------------
                            ENCRYPTION DETAILS
--------------------------------------------------------------------------------
Algorithm: Hybrid NTRU + Kyber (PQC)
NTRU Variant: NTRU-HPS-2048-509
Kyber Variant: Kyber768
Symmetric: AES-256-GCM

Key Sizes:
  - NTRU Public Key:  {len(ntru_public_key)} bytes
  - NTRU Private Key: {len(ntru_private_key)} bytes
  - Kyber Public Key:  {len(kyber_public_key)} bytes
  - Kyber Private Key: {len(kyber_private_key)} bytes

Data Sizes:
  - Original Message: {len(message_bytes)} bytes
  - Encrypted Data:   {len(encrypted_data)} bytes
  - Decrypted Data:   {len(decrypted_data)} bytes
  - Overhead:         {len(encrypted_data) - len(message_bytes)} bytes

--------------------------------------------------------------------------------
                            ORIGINAL MESSAGE
--------------------------------------------------------------------------------
{original_message}

--------------------------------------------------------------------------------
                            ENCRYPTED DATA (HEX PREVIEW)
--------------------------------------------------------------------------------
{encrypted_data[:200].hex()}
...
[Total {len(encrypted_data)} bytes]

--------------------------------------------------------------------------------
                            DECRYPTED MESSAGE
--------------------------------------------------------------------------------
{decrypted_message}

--------------------------------------------------------------------------------
                            VERIFICATION
--------------------------------------------------------------------------------
Original == Decrypted: {original_message == decrypted_message}
Status: {verification_status}

================================================================================
                              END OF REPORT
================================================================================
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(results_content)
    
    print(f"[7] RESULTS SAVED TO: {output_file}")
    print()
    print("=" * 60)
    print("  Test Complete!")
    print("=" * 60)
    
    return verification_status == "PASSED"


if __name__ == "__main__":
    try:
        success = test_encryption_decryption()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
