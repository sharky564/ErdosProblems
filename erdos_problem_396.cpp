#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <execution>
#include <atomic>
#include <ranges>
#include <span>
#include <chrono>
#include <iomanip>

constexpr int NUM_THREADS = 7;
std::vector<uint64_t> primes;

void get_primes(uint64_t limit) 
{
    std::vector<bool> sieve(limit + 1, true);
    sieve[0] = false;
    sieve[1] = false;
    for (uint64_t p = 2; p * p <= limit; ++p) 
    {
        if (sieve[p]) 
        {
            for (uint64_t i = p * p; i <= limit; i += p)
                sieve[i] = false;
        }
    }
    for (uint64_t p = 2; p <= limit; ++p) 
    {
        if (sieve[p])
            primes.push_back(p);
    }
}

bool exact_check(uint64_t n, uint64_t k) 
{
    for (uint64_t p : primes) 
    {
        if (p > 2 * k)
            break;
        uint64_t nu_prod = 0;
        uint64_t nu_comb = 0;
        uint64_t power = p;
        while (true) 
        {
            uint64_t v_n = n / power;
            nu_prod += v_n - (n - k - 1) / power;
            nu_comb += (2 * n) / power - 2 * v_n;
            if (power > (2 * n) / p)
                break;
            power *= p;
        }
        if (nu_prod > nu_comb)
            return false;
    }
    return true;
}

uint64_t solve(uint64_t k) 
{
    const uint64_t CHUNK_SIZE = 5000000;
    const uint64_t BLOCK_SIZE = 32768;
    uint64_t L_batch = 1;
    
    std::atomic<uint64_t> global_min_n{static_cast<uint64_t>(-1)};
    std::atomic<bool> found_in_batch{false};
    auto chunks = std::views::iota(0, NUM_THREADS);

    auto is_valid_c = [](uint64_t c, uint64_t p) {
        uint64_t nu = 0;
        uint64_t temp_c = c;
        while (temp_c >= p && temp_c % p == 0) 
        {
            ++nu;
            temp_c /= p;
        }
        uint64_t carries = 0;
        uint64_t temp = c;
        uint64_t carry = 0;
        while (temp > 0) 
        {
            uint64_t digit = temp % p;
            if (digit * 2 + carry >= p) 
            {
                ++carries;
                carry = 1;
            } 
            else 
            {
                carry = 0;
            }
            temp /= p;
            if (carries >= 1 + nu)
                return true;
        }
        return false;
    };

    while (true) 
    {
        found_in_batch = false;
        std::for_each(std::execution::par, chunks.begin(), chunks.end(), [&](int chunk_id) {
            uint64_t L_chunk = L_batch + chunk_id * CHUNK_SIZE;
            uint64_t R_chunk = L_chunk + CHUNK_SIZE + k - 1;
            
            if (L_chunk > global_min_n.load(std::memory_order_relaxed))
                return;

            uint64_t max_p = std::max<uint64_t>(std::sqrt(2 * R_chunk) + 1, 2 * k);
            
            thread_local std::vector<uint8_t> valid;
            thread_local std::vector<uint64_t> rem;
            valid.resize(BLOCK_SIZE);
            rem.resize(BLOCK_SIZE);

            uint64_t consecutive = 0;

            for (uint64_t block_L = L_chunk; block_L <= R_chunk; block_L += BLOCK_SIZE) 
            {
                uint64_t block_R = std::min(R_chunk, block_L + BLOCK_SIZE - 1);
                uint64_t W_block = block_R - block_L + 1;
                
                for (uint64_t j = 0; j < W_block; ++j) 
                {
                    rem[j] = block_L + j;
                    valid[j] = 1;
                }

                for (uint64_t p : primes) 
                {
                    if (p > max_p)
                        break;
                    
                    uint64_t start_c = (block_L + p - 1) / p;
                    uint64_t start_j = start_c * p - block_L;
                    if (start_j >= W_block)
                        continue;

                    if (p > 2 * k) 
                    {
                        uint64_t c = start_c;
                        for (uint64_t j = start_j; j < W_block; j += p, ++c) 
                        {
                            if (valid[j]) 
                            {
                                bool is_val = true;
                                if (c < p) 
                                {
                                    if (c * 2 < p)
                                        is_val = false;
                                } 
                                else 
                                {
                                    if (!is_valid_c(c, p))
                                        is_val = false;
                                }
                                
                                if (!is_val) 
                                {
                                    valid[j] = 0;
                                } 
                                else 
                                {
                                    uint64_t temp = rem[j];
                                    temp /= p;
                                    if (temp % p == 0) 
                                    {
                                        do { temp /= p; } while (temp % p == 0);
                                    }
                                    rem[j] = temp;
                                }
                            }
                        }
                    } 
                    else 
                    {
                        for (uint64_t j = start_j; j < W_block; j += p) 
                        {
                            if (valid[j]) 
                            {
                                uint64_t temp = rem[j];
                                temp /= p;
                                if (temp % p == 0) 
                                {
                                    do { temp /= p; } while (temp % p == 0);
                                }
                                rem[j] = temp;
                            }
                        }
                    }
                }

                for (uint64_t j = 0; j < W_block; ++j) 
                {
                    if (valid[j] && rem[j] <= 1) 
                    {
                        if (++consecutive >= k + 1) 
                        {
                            uint64_t n = block_L + j;
                            if (n > k && n < global_min_n.load(std::memory_order_relaxed)) 
                            {
                                if (exact_check(n, k)) 
                                {
                                    uint64_t current = global_min_n.load(std::memory_order_relaxed);
                                    while (n < current && !global_min_n.compare_exchange_weak(current, n, std::memory_order_relaxed)) {}
                                    found_in_batch.store(true, std::memory_order_relaxed);
                                }
                            }
                        }
                    } 
                    else 
                    {
                        consecutive = 0;
                    }
                }
            }
        });

        if (found_in_batch)
            return global_min_n.load();
        L_batch += NUM_THREADS * CHUNK_SIZE;
    }
}

int main() 
{
    std::cout << "Generating primes up to 5,000,000..." << std::endl;
    auto start_primes = std::chrono::high_resolution_clock::now();
    get_primes(5000000); 
    auto end_primes = std::chrono::high_resolution_clock::now();
    std::cout << "Primes generated in " 
              << std::chrono::duration<double>(end_primes - start_primes).count() 
              << " seconds.\n\n";

    for (uint64_t k = 1; k <= 20; ++k) {
        auto start = std::chrono::high_resolution_clock::now();
        uint64_t ans = solve(k);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double seconds = elapsed.count();
        double speed = ans / seconds;
        std::cout << "k = " << std::setw(2) << k 
                  << " | min n = " << std::setw(15) << ans 
                  << " | Time: " << std::fixed << std::setprecision(4) << std::setw(12) << seconds << " s"
                  << " | Speed: " << std::fixed << std::setprecision(2) << std::setw(8) << (speed / 1e6) << " M candidates/s\n";
    }
    return 0;
}