"""
PQC Performance Benchmark Suite
Measures performance of post-quantum cryptographic operations

Benchmarks:
- Key generation (NTRU, Kyber, Dilithium, SPHINCS+)
- Encryption/Decryption (single and hybrid)
- Signing/Verification
- Face embedding encryption
- JWT token operations

Performance Targets:
- Login latency: < 500ms
- Registration: < 2000ms
- Embedding encryption: < 100ms
- JWT generation: < 50ms
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import statistics
from typing import Dict, List, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np


@dataclass
class BenchmarkResult:
    """Result of a single benchmark"""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    ops_per_second: float
    passed_threshold: bool
    threshold_ms: float


class PQCBenchmark:
    """
    Post-Quantum Cryptography Benchmark Suite
    
    Measures performance of all PQC operations to ensure
    they meet the defined latency targets.
    """
    
    def __init__(self):
        """Initialize benchmark suite"""
        self.results: List[BenchmarkResult] = []
        
        # Performance thresholds (in milliseconds)
        self.thresholds = {
            "ntru_keygen": 50,
            "kyber_keygen": 50,
            "dilithium_keygen": 50,
            "sphincs_keygen": 100,
            "ntru_encrypt": 20,
            "ntru_decrypt": 20,
            "kyber_encrypt": 20,
            "kyber_decrypt": 20,
            "hybrid_encrypt": 50,
            "hybrid_decrypt": 50,
            "dilithium_sign": 30,
            "dilithium_verify": 30,
            "embedding_encrypt": 100,
            "embedding_decrypt": 100,
            "jwt_create": 50,
            "jwt_verify": 50,
            "login_total": 500,
            "registration_total": 2000
        }
    
    def _run_benchmark(
        self,
        name: str,
        func: Callable,
        iterations: int = 100,
        setup: Callable = None,
        warmup: int = 5
    ) -> BenchmarkResult:
        """
        Run a single benchmark
        
        Args:
            name: Benchmark name
            func: Function to benchmark
            iterations: Number of iterations
            setup: Setup function called before each iteration
            warmup: Number of warmup iterations
        """
        times = []
        
        # Warmup runs
        for _ in range(warmup):
            if setup:
                setup()
            func()
        
        # Actual benchmark
        for _ in range(iterations):
            if setup:
                setup()
            
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            
            times.append((end - start) * 1000)  # Convert to ms
        
        avg_time = statistics.mean(times)
        threshold = self.thresholds.get(name, float('inf'))
        
        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time_ms=sum(times),
            avg_time_ms=avg_time,
            min_time_ms=min(times),
            max_time_ms=max(times),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
            ops_per_second=1000 / avg_time if avg_time > 0 else 0,
            passed_threshold=avg_time <= threshold,
            threshold_ms=threshold
        )
        
        self.results.append(result)
        return result
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all PQC benchmarks"""
        print("=" * 80)
        print("POST-QUANTUM CRYPTOGRAPHY PERFORMANCE BENCHMARKS")
        print("=" * 80)
        print(f"Started at: {datetime.utcnow().isoformat()}")
        print()
        
        # Import PQC services
        from app.services.pqc_service import pqc_service
        from app.services.pqc_key_manager import pqc_key_manager
        from app.services.hybrid_crypto_service import hybrid_crypto_service
        from app.services.pqc_jwt_service import pqc_jwt_service
        
        # Test data
        test_data = b"Test data for encryption benchmark " * 10
        test_embedding = np.random.rand(512).astype(np.float32)
        test_message = b"Test message for signing benchmark"
        
        # =====================================================================
        # Key Generation Benchmarks
        # =====================================================================
        print("\n[1/6] KEY GENERATION BENCHMARKS")
        print("-" * 60)
        
        # NTRU Key Generation
        result = self._run_benchmark(
            "ntru_keygen",
            lambda: pqc_service.generate_ntru_keypair(),
            iterations=50
        )
        self._print_result(result)
        
        # Kyber Key Generation
        result = self._run_benchmark(
            "kyber_keygen",
            lambda: pqc_service.generate_kyber_keypair(),
            iterations=50
        )
        self._print_result(result)
        
        # Dilithium Key Generation
        result = self._run_benchmark(
            "dilithium_keygen",
            lambda: pqc_service.generate_dilithium_keypair(),
            iterations=50
        )
        self._print_result(result)
        
        # SPHINCS+ Key Generation
        result = self._run_benchmark(
            "sphincs_keygen",
            lambda: pqc_service.generate_sphincs_keypair(),
            iterations=20
        )
        self._print_result(result)
        
        # =====================================================================
        # Single Algorithm Encryption Benchmarks
        # =====================================================================
        print("\n[2/6] SINGLE ALGORITHM ENCRYPTION BENCHMARKS")
        print("-" * 60)
        
        # Generate keys for encryption tests
        ntru_pub, ntru_priv = pqc_service.generate_ntru_keypair()
        kyber_pub, kyber_priv = pqc_service.generate_kyber_keypair()
        
        # NTRU Encrypt
        result = self._run_benchmark(
            "ntru_encrypt",
            lambda: pqc_service.ntru_encrypt(test_data, ntru_pub),
            iterations=100
        )
        self._print_result(result)
        
        # NTRU Decrypt
        ntru_ct = pqc_service.ntru_encrypt(test_data, ntru_pub)
        result = self._run_benchmark(
            "ntru_decrypt",
            lambda: pqc_service.ntru_decrypt(ntru_ct, ntru_priv),
            iterations=100
        )
        self._print_result(result)
        
        # Kyber Encrypt
        result = self._run_benchmark(
            "kyber_encrypt",
            lambda: pqc_service.kyber_encrypt(test_data, kyber_pub),
            iterations=100
        )
        self._print_result(result)
        
        # Kyber Decrypt
        kyber_ct = pqc_service.kyber_encrypt(test_data, kyber_pub)
        result = self._run_benchmark(
            "kyber_decrypt",
            lambda: pqc_service.kyber_decrypt(kyber_ct, kyber_priv),
            iterations=100
        )
        self._print_result(result)
        
        # =====================================================================
        # Hybrid Encryption Benchmarks
        # =====================================================================
        print("\n[3/6] HYBRID ENCRYPTION BENCHMARKS (NTRU + Kyber)")
        print("-" * 60)
        
        # Hybrid Encrypt
        result = self._run_benchmark(
            "hybrid_encrypt",
            lambda: pqc_service.hybrid_encrypt(test_data, ntru_pub, kyber_pub),
            iterations=50
        )
        self._print_result(result)
        
        # Hybrid Decrypt
        hybrid_ct = pqc_service.hybrid_encrypt(test_data, ntru_pub, kyber_pub)
        result = self._run_benchmark(
            "hybrid_decrypt",
            lambda: pqc_service.hybrid_decrypt(hybrid_ct, ntru_priv, kyber_priv),
            iterations=50
        )
        self._print_result(result)
        
        # =====================================================================
        # Signature Benchmarks
        # =====================================================================
        print("\n[4/6] DIGITAL SIGNATURE BENCHMARKS")
        print("-" * 60)
        
        # Generate signature keys
        dilithium_pub, dilithium_priv = pqc_service.generate_dilithium_keypair()
        
        # Dilithium Sign
        result = self._run_benchmark(
            "dilithium_sign",
            lambda: pqc_service.sign_dilithium(test_message, dilithium_priv),
            iterations=100
        )
        self._print_result(result)
        
        # Dilithium Verify
        signature = pqc_service.sign_dilithium(test_message, dilithium_priv)
        result = self._run_benchmark(
            "dilithium_verify",
            lambda: pqc_service.verify_dilithium(test_message, signature, dilithium_pub),
            iterations=100
        )
        self._print_result(result)
        
        # =====================================================================
        # Face Embedding Encryption Benchmarks
        # =====================================================================
        print("\n[5/6] FACE EMBEDDING ENCRYPTION BENCHMARKS")
        print("-" * 60)
        
        # Setup: Generate user keys
        test_user_id = 9999
        user_keys = pqc_key_manager.generate_user_keys(test_user_id)
        pqc_key_manager.save_user_keys(test_user_id, user_keys)
        
        # Embedding Encrypt
        result = self._run_benchmark(
            "embedding_encrypt",
            lambda: hybrid_crypto_service.encrypt_embedding(test_embedding, test_user_id, sign=True),
            iterations=50
        )
        self._print_result(result)
        
        # Embedding Decrypt
        encrypted_embedding = hybrid_crypto_service.encrypt_embedding(test_embedding, test_user_id, sign=True)
        result = self._run_benchmark(
            "embedding_decrypt",
            lambda: hybrid_crypto_service.decrypt_embedding(encrypted_embedding, test_user_id),
            iterations=50
        )
        self._print_result(result)
        
        # Cleanup test user
        pqc_key_manager.delete_user_keys(test_user_id)
        
        # =====================================================================
        # JWT Token Benchmarks
        # =====================================================================
        print("\n[6/6] JWT TOKEN BENCHMARKS")
        print("-" * 60)
        
        # Setup: Generate keys for JWT test user
        jwt_test_user_id = 9998
        jwt_user_keys = pqc_key_manager.generate_user_keys(jwt_test_user_id)
        pqc_key_manager.save_user_keys(jwt_test_user_id, jwt_user_keys)
        
        test_payload = {
            "user_id": jwt_test_user_id,
            "username": "benchmark_user"
        }
        
        # JWT Create
        result = self._run_benchmark(
            "jwt_create",
            lambda: pqc_jwt_service.create_token(test_payload, jwt_test_user_id),
            iterations=50
        )
        self._print_result(result)
        
        # JWT Verify
        test_token = pqc_jwt_service.create_token(test_payload, jwt_test_user_id)
        result = self._run_benchmark(
            "jwt_verify",
            lambda: pqc_jwt_service.verify_token(test_token),
            iterations=50
        )
        self._print_result(result)
        
        # Cleanup JWT test user
        pqc_key_manager.delete_user_keys(jwt_test_user_id)
        
        # =====================================================================
        # Simulated Operation Benchmarks
        # =====================================================================
        print("\n[SIMULATED OPERATIONS]")
        print("-" * 60)
        
        # Simulate login operation
        login_total = self._simulate_login_operation(
            pqc_service, pqc_key_manager, pqc_jwt_service
        )
        
        # Simulate registration operation
        registration_total = self._simulate_registration_operation(
            pqc_service, pqc_key_manager, hybrid_crypto_service
        )
        
        # Print summary
        return self._print_summary()
    
    def _simulate_login_operation(
        self,
        pqc_service,
        pqc_key_manager,
        pqc_jwt_service
    ) -> float:
        """Simulate a complete login operation"""
        test_user_id = 9997
        
        # Setup: Create user keys
        user_keys = pqc_key_manager.generate_user_keys(test_user_id)
        pqc_key_manager.save_user_keys(test_user_id, user_keys)
        
        times = []
        
        for _ in range(20):
            start = time.perf_counter()
            
            # 1. Load user keys
            keys = pqc_key_manager.load_user_keys(test_user_id)
            
            # 2. Verify face (simulated - just decrypt an embedding)
            test_embedding = np.random.rand(512).astype(np.float32)
            from app.services.hybrid_crypto_service import hybrid_crypto_service
            encrypted = hybrid_crypto_service.encrypt_embedding(test_embedding, test_user_id, sign=True)
            decrypted = hybrid_crypto_service.decrypt_embedding(encrypted, test_user_id)
            
            # 3. Create JWT token
            token = pqc_jwt_service.create_token({"user_id": test_user_id}, test_user_id)
            
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        # Cleanup
        pqc_key_manager.delete_user_keys(test_user_id)
        
        avg_time = statistics.mean(times)
        threshold = self.thresholds["login_total"]
        
        result = BenchmarkResult(
            name="login_total",
            iterations=20,
            total_time_ms=sum(times),
            avg_time_ms=avg_time,
            min_time_ms=min(times),
            max_time_ms=max(times),
            std_dev_ms=statistics.stdev(times),
            ops_per_second=1000 / avg_time,
            passed_threshold=avg_time <= threshold,
            threshold_ms=threshold
        )
        
        self.results.append(result)
        self._print_result(result)
        
        return avg_time
    
    def _simulate_registration_operation(
        self,
        pqc_service,
        pqc_key_manager,
        hybrid_crypto_service
    ) -> float:
        """Simulate a complete registration operation"""
        times = []
        
        for i in range(10):
            test_user_id = 9900 + i
            start = time.perf_counter()
            
            # 1. Generate user keys
            user_keys = pqc_key_manager.generate_user_keys(test_user_id, include_sphincs=True)
            
            # 2. Save keys
            pqc_key_manager.save_user_keys(test_user_id, user_keys)
            
            # 3. Encrypt 3 face embeddings
            for _ in range(3):
                embedding = np.random.rand(512).astype(np.float32)
                encrypted = hybrid_crypto_service.encrypt_embedding(embedding, test_user_id, sign=True)
            
            end = time.perf_counter()
            times.append((end - start) * 1000)
            
            # Cleanup
            pqc_key_manager.delete_user_keys(test_user_id)
        
        avg_time = statistics.mean(times)
        threshold = self.thresholds["registration_total"]
        
        result = BenchmarkResult(
            name="registration_total",
            iterations=10,
            total_time_ms=sum(times),
            avg_time_ms=avg_time,
            min_time_ms=min(times),
            max_time_ms=max(times),
            std_dev_ms=statistics.stdev(times),
            ops_per_second=1000 / avg_time,
            passed_threshold=avg_time <= threshold,
            threshold_ms=threshold
        )
        
        self.results.append(result)
        self._print_result(result)
        
        return avg_time
    
    def _print_result(self, result: BenchmarkResult):
        """Print a single benchmark result"""
        status = "✓ PASS" if result.passed_threshold else "✗ FAIL"
        
        print(f"  {result.name:25} | "
              f"avg: {result.avg_time_ms:8.2f}ms | "
              f"min: {result.min_time_ms:8.2f}ms | "
              f"max: {result.max_time_ms:8.2f}ms | "
              f"threshold: {result.threshold_ms:6.0f}ms | "
              f"{status}")
    
    def _print_summary(self) -> Dict[str, Any]:
        """Print summary of all benchmarks"""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        passed = sum(1 for r in self.results if r.passed_threshold)
        total = len(self.results)
        
        print(f"\nResults: {passed}/{total} benchmarks passed")
        print()
        
        # Performance categories
        categories = {
            "Key Generation": ["ntru_keygen", "kyber_keygen", "dilithium_keygen", "sphincs_keygen"],
            "Encryption": ["ntru_encrypt", "ntru_decrypt", "kyber_encrypt", "kyber_decrypt"],
            "Hybrid Encryption": ["hybrid_encrypt", "hybrid_decrypt"],
            "Signatures": ["dilithium_sign", "dilithium_verify"],
            "Embeddings": ["embedding_encrypt", "embedding_decrypt"],
            "JWT Tokens": ["jwt_create", "jwt_verify"],
            "Full Operations": ["login_total", "registration_total"]
        }
        
        for category, benchmarks in categories.items():
            cat_results = [r for r in self.results if r.name in benchmarks]
            if cat_results:
                cat_passed = sum(1 for r in cat_results if r.passed_threshold)
                avg_time = statistics.mean([r.avg_time_ms for r in cat_results])
                
                status = "✓" if cat_passed == len(cat_results) else "✗"
                print(f"  {status} {category:20} | "
                      f"{cat_passed}/{len(cat_results)} passed | "
                      f"avg: {avg_time:.2f}ms")
        
        print()
        
        # Failed benchmarks
        failed = [r for r in self.results if not r.passed_threshold]
        if failed:
            print("Failed Benchmarks:")
            for r in failed:
                print(f"  ✗ {r.name}: {r.avg_time_ms:.2f}ms (threshold: {r.threshold_ms}ms)")
        
        print("=" * 80)
        
        # Return summary data
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_benchmarks": total,
            "passed": passed,
            "failed": total - passed,
            "all_passed": passed == total,
            "results": [asdict(r) for r in self.results]
        }


def run_benchmarks() -> Dict[str, Any]:
    """Run all PQC benchmarks"""
    benchmark = PQCBenchmark()
    return benchmark.run_all_benchmarks()


if __name__ == "__main__":
    summary = run_benchmarks()
    
    # Exit with error code if any benchmarks failed
    sys.exit(0 if summary["all_passed"] else 1)
