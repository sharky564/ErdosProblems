import math
import time

def is_prime_mr(n):
    """Deterministic 64-bit Miller-Rabin Primality Test"""
    if n < 2: return False
    if n in (2, 3): return True
    if n % 2 == 0: return False
    
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
        
    for a in[2, 13, 23, 1662803]:
        if n <= a: break
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def sum_digits(x, base):
    res = 0
    while x > 0:
        res += x % base
        x //= base
    return res

def check_n_k_fast(n, k):
    m = n - 1
    N = m + k
    
    print(f"--- O(sqrt(N)) Validation for n={n}, k={k} ---")
    start_time = time.time()
    
    if (2 * N - m).bit_count() + m.bit_count() > 2 * N.bit_count():
        print("[-] FAILED: Condition violated for prime p = 2")
        return False
        
    sqrt_2N = math.isqrt(2 * N)
    
    sieve = bytearray([1]) * (sqrt_2N + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, math.isqrt(sqrt_2N) + 1):
        if sieve[p]:
            sieve[p*p : sqrt_2N+1 : p] = bytearray([0]) * len(range(p*p, sqrt_2N+1, p))
            
    primes =[p for p, is_prime in enumerate(sieve) if is_prime]
    
    for p in primes:
        if p == 2: continue
        if sum_digits(2 * N - m, p) + sum_digits(m, p) > 2 * sum_digits(N, p):
            print(f"[-] FAILED: Condition violated for prime p = {p}")
            return False
            
    limit_d0 = (m + 1) // 2
    max_d1 = N // (sqrt_2N + 1) + 1
    
    for d1 in range(1, max_d1 + 1):
        r = N % d1
        if r < limit_d0:
            for d0 in range(r, limit_d0, d1):
                q = (N - d0) // d1
                if q > sqrt_2N and q > m:
                    if is_prime_mr(q):
                        print(f"[-] FAILED: Condition violated for large prime p = {q}")
                        return False

    print(f"[+] SUCCESS: n={n}, k={k} is mathematically valid! (Checked in {time.time() - start_time:.5f}s)\n")
    return True

if __name__ == "__main__":
    n_val = 23
    k_val = 17609764972
    check_n_k_fast(n_val, k_val)