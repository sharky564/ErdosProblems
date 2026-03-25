#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <thread>
#include <ranges>
#include <span>
#include <chrono>
#include <iomanip>
#include <bit>
#include <fstream>
#include <filesystem>
#include <sstream>

const unsigned int NUM_THREADS = std::thread::hardware_concurrency(); // 8
const unsigned int NUM_SEQ = 2;
std::vector<uint32_t> primes;
std::vector<uint64_t> prime_magics;
std::vector<uint8_t> prime_shifts;

void get_primes(uint32_t limit) 
{
    std::vector<bool> sieve(limit + 1, true);
    sieve[0] = false;
    sieve[1] = false;
    for (uint32_t p = 2; p * p <= limit; ++p) 
    {
        if (sieve[p]) 
        {
            for (uint32_t i = p * p; i <= limit; i += p)
                sieve[i] = false;
        }
    }
    for (uint32_t p = 2; p <= limit; ++p) 
    {
        if (sieve[p]) 
        {
            primes.push_back(p);
            uint8_t shift = std::bit_width(p) - 1;
            unsigned __int128 magic_128 = ((unsigned __int128)1 << (64 + shift)) / p + 1;
            prime_magics.push_back((uint64_t)magic_128);
            prime_shifts.push_back(shift);
        }
    }
}

uint64_t power(uint64_t base, uint64_t exp, uint64_t mod) 
{
    uint64_t res = 1;
    base %= mod;
    while (exp > 0) 
    {
        if (exp % 2 == 1)
            res = (__uint128_t)res * base % mod;
        base = (__uint128_t)base * base % mod;
        exp /= 2;
    }
    return res;
}

bool miller_rabin(uint64_t n) 
{
    if (n < 2)
        return false;
    if (n == 2 || n == 3)
        return true;
    if (n % 2 == 0)
        return false;

    uint64_t d = n - 1;
    int s = 0;
    while (d % 2 == 0) 
    { 
        d /= 2; 
        s++; 
    }

    static const uint64_t bases[] = {2, 13, 23, 1662803};
    for (uint64_t a : bases) 
    {
        if (n <= a) 
            break;
        uint64_t x = power(a, d, n);
        if (x == 1 || x == n - 1) 
            continue;
        bool composite = true;
        for (int r = 1; r < s; r++) 
        {
            x = (__uint128_t)x * x % n;
            if (x == n - 1) 
            {
                composite = false;
                break;
            }
        }
        if (composite) 
            return false;
    }
    return true;
}

bool is_valid(uint64_t N, uint64_t m) {
    uint64_t sqrt_2N = std::sqrt(2 * N);
    size_t total_primes = primes.size();
    
    if (total_primes > 0 && primes.back() < sqrt_2N) 
    {
        std::cerr << "\n[ERROR] Candidate exceeds prime generation bounds! Need primes up to " << sqrt_2N << "\n";
        std::exit(1);
    }
    
    for (size_t idx = 0; idx < total_primes; ++idx) 
    {
        uint64_t p = primes[idx];
        if (p <= 199)
            continue;
        if (p > sqrt_2N)
            break;
        
        uint64_t magic = prime_magics[idx];
        uint8_t shift = prime_shifts[idx];
        
        if (p > m) {
            uint64_t q_2N = (uint64_t)((((unsigned __int128)(2 * N) * magic) >> 64) >> shift);
            uint64_t r_2N = (2 * N) - q_2N * p;
            if (r_2N >= m) 
                continue;
            
            uint64_t s_2Nm = 0; uint64_t temp = 2 * N - m;
            while (temp) 
            { 
                uint64_t q = (uint64_t)((((unsigned __int128)temp * magic) >> 64) >> shift);
                s_2Nm += temp - q * p; 
                temp = q; 
            }
            
            uint64_t s_N = 0; temp = N;
            while (temp) 
            { 
                uint64_t q = (uint64_t)((((unsigned __int128)temp * magic) >> 64) >> shift);
                s_N += temp - q * p; 
                temp = q; 
            }
            
            if (s_2Nm + m > 2 * s_N) 
                return false;
        } 
        else 
        {
            uint64_t s_m = 0, temp = m;
            while (temp) 
            { 
                uint64_t q = (uint64_t)((((unsigned __int128)temp * magic) >> 64) >> shift);
                s_m += temp - q * p; 
                temp = q; 
            }
            
            uint64_t s_2Nm = 0; temp = 2 * N - m;
            while (temp) 
            { 
                uint64_t q = (uint64_t)((((unsigned __int128)temp * magic) >> 64) >> shift);
                s_2Nm += temp - q * p; 
                temp = q; 
            }
            
            uint64_t s_N = 0; temp = N;
            while (temp) 
            { 
                uint64_t q = (uint64_t)((((unsigned __int128)temp * magic) >> 64) >> shift);
                s_N += temp - q * p; 
                temp = q; 
            }
            
            if (s_2Nm + s_m > 2 * s_N) return false;
        }
    }

    uint64_t limit_d0 = (m + 1) / 2;
    uint64_t max_d1 = N / (sqrt_2N + 1) + 1;
    
    for (uint64_t d1 = 1; d1 <= max_d1; ++d1) 
    {
        uint64_t r = N % d1;
        if (r < limit_d0) 
        {
            for (uint64_t d0 = r; d0 < limit_d0; d0 += d1) 
            {
                uint64_t q = (N - d0) / d1;
                if (q > sqrt_2N && q > m) 
                {
                    if (q % 2 != 0 && q % 3 != 0 && q % 5 != 0 && q % 7 != 0) 
                    {
                        if (miller_rabin(q)) 
                            return false;
                    }
                }
            }
        }
    }
    return true;
}

template<uint64_t p>
inline void check_p_small(uint64_t block_N_L, uint64_t W_block, uint8_t* valid, uint64_t m, uint64_t s_m) 
{
    uint64_t m_0 = (m - 1) / 2;
    for (uint64_t j = 0; j < W_block; ++j) 
    {
        if (valid[j]) 
        {
            uint64_t N = block_N_L + j;
            if constexpr (p > 29) 
            { 
                if (p > m) 
                {
                    uint64_t q = N / p;
                    uint64_t r = N - q * p;
                    if (r <= m_0 && q < p / 2) 
                    {
                        valid[j] = 0;
                        continue;
                    }
                }
            }
            
            uint64_t temp = 2 * N - m;
            uint64_t s_2Nm = 0;
            while (temp > 0) 
            {
                s_2Nm += temp % p; 
                temp /= p;
            }
            
            uint64_t s_N = 0;
            temp = N;
            while (temp > 0) 
            {
                s_N += temp % p;
                temp /= p; 
            }
            
            if (s_2Nm + s_m > 2 * s_N) 
                valid[j] = 0;
        }
    }
}

uint64_t solve_new(uint64_t n, uint64_t start_L) 
{
    if (n == 1)
        return 1;
    uint64_t m = n - 1;
    uint32_t pop_m = std::popcount(m);
    
    uint64_t s_m_arr[200] = {0};
    for (uint64_t p : primes) 
    {
        if (p > 199)
            break;
        uint64_t temp = m;
        uint64_t s = 0;
        while(temp) 
        { 
            s += temp % p; 
            temp /= p; 
        }
        s_m_arr[p] = s;
    }

    const uint64_t CHUNK_SIZE = 10000000;
    const uint64_t BLOCK_SIZE = 32768;
    const uint64_t CHUNKS_PER_BATCH = NUM_THREADS * NUM_SEQ; 
    
    uint64_t L_batch = start_L;
    
    std::atomic<uint64_t> global_min_k{static_cast<uint64_t>(-1)};
    std::atomic<bool> found_in_batch{false};

#define PROCESS_PRIME(p_val) \
    case p_val: check_p_small<p_val>(block_L + m, W_block, valid.data(), m, s_m_arr[p_val]); break;

    auto worker = [&](std::atomic<int>& current_chunk) {
        std::vector<uint8_t> valid(BLOCK_SIZE);
        
        while (true) 
        {
            int chunk_id = current_chunk.fetch_add(1, std::memory_order_relaxed);
            if (chunk_id >= CHUNKS_PER_BATCH)
                break;
            
            uint64_t L_chunk = L_batch + chunk_id * CHUNK_SIZE;
            uint64_t R_chunk = L_chunk + CHUNK_SIZE - 1;
            
            if (L_chunk >= global_min_k.load(std::memory_order_relaxed))
                continue;

            for (uint64_t block_L = L_chunk; block_L <= R_chunk; block_L += BLOCK_SIZE) 
            {
                uint64_t block_R = std::min(R_chunk, block_L + BLOCK_SIZE - 1);
                uint64_t W_block = block_R - block_L + 1;
                
                uint64_t x = block_L + m;
                for (uint64_t j = 0; j < W_block; ++j, ++x) 
                {
                    uint32_t p_N = std::popcount(x);
                    uint32_t p_2Nm = std::popcount(2 * x - m);
                    valid[j] = (p_2Nm + pop_m <= 2 * p_N) ? 1 : 0;
                }

                for (uint64_t p : primes) 
                {
                    if (p == 2)
                        continue;
                    if (p > 199)
                        break;
                    
                    switch (p) 
                    {
                        PROCESS_PRIME(3) PROCESS_PRIME(5) PROCESS_PRIME(7) PROCESS_PRIME(11)
                        PROCESS_PRIME(13) PROCESS_PRIME(17) PROCESS_PRIME(19) PROCESS_PRIME(23)
                        PROCESS_PRIME(29) PROCESS_PRIME(31) PROCESS_PRIME(37) PROCESS_PRIME(41)
                        PROCESS_PRIME(43) PROCESS_PRIME(47) PROCESS_PRIME(53) PROCESS_PRIME(59)
                        PROCESS_PRIME(61) PROCESS_PRIME(67) PROCESS_PRIME(71) PROCESS_PRIME(73)
                        PROCESS_PRIME(79) PROCESS_PRIME(83) PROCESS_PRIME(89) PROCESS_PRIME(97)
                        PROCESS_PRIME(101) PROCESS_PRIME(103) PROCESS_PRIME(107) PROCESS_PRIME(109)
                        PROCESS_PRIME(113) PROCESS_PRIME(127) PROCESS_PRIME(131) PROCESS_PRIME(137)
                        PROCESS_PRIME(139) PROCESS_PRIME(149) PROCESS_PRIME(151) PROCESS_PRIME(157)
                        PROCESS_PRIME(163) PROCESS_PRIME(167) PROCESS_PRIME(173) PROCESS_PRIME(179)
                        PROCESS_PRIME(181) PROCESS_PRIME(191) PROCESS_PRIME(193) PROCESS_PRIME(197)
                        PROCESS_PRIME(199)
                    }
                }
                for (uint64_t j = 0; j < W_block; ++j) 
                {
                    if (valid[j]) 
                    {
                        uint64_t k_cand = block_L + j;
                        if (k_cand >= global_min_k.load(std::memory_order_relaxed))
                            continue;
                        
                        uint64_t N = k_cand + m;
                        if (is_valid(N, m)) 
                        {
                            uint64_t current = global_min_k.load(std::memory_order_relaxed);
                            while (k_cand < current && !global_min_k.compare_exchange_weak(current, k_cand, std::memory_order_relaxed)) {}
                            found_in_batch.store(true, std::memory_order_relaxed);
                        }
                    }
                }
            }
        }
    };

    while (true) 
    {
        found_in_batch = false;
        std::atomic<int> current_chunk{0};
        std::vector<std::thread> threads;
        for (unsigned int i = 0; i < NUM_THREADS; ++i)
            threads.emplace_back(worker, std::ref(current_chunk));
        for (auto& t : threads) t.join();

        if (found_in_batch)
            return global_min_k.load();
        
        L_batch += CHUNKS_PER_BATCH * CHUNK_SIZE;
        
        std::ofstream fout("checkpoint-389.tmp");
        if (fout) 
        {
            fout << n << " " << L_batch << "\n";
            fout.close();
            std::error_code ec;
            std::filesystem::rename("checkpoint-389.tmp", "checkpoint-389-master.txt", ec);
        }
    }
}

int main() {
    std::cout << "Detected " << NUM_THREADS << " logical cores. Using C++ Thread Pool...\n";
    std::cout << "Generating primes up to 200,000,000..." << std::endl;
    auto start_primes = std::chrono::high_resolution_clock::now();
    
    get_primes(200000000); 
    
    auto end_primes = std::chrono::high_resolution_clock::now();
    std::cout << "Primes generated in " 
              << std::chrono::duration<double>(end_primes - start_primes).count() 
              << " seconds.\n\n";

    uint64_t start_n = 1;
    uint64_t start_L = 1;

    if (std::filesystem::exists("checkpoint-389-master.txt")) 
    {
        std::ifstream fin("checkpoint-389-master.txt");
        if (fin >> start_n >> start_L) 
        {
            std::cout << "--> Resuming from Checkpoint: n = " << start_n 
                      << ", L_batch = " << start_L << "\n\n";
        } 
        else 
        {
            start_n = 1; 
            start_L = 1;
        }
    }

    for (uint64_t n = start_n; n <= 30; ++n) 
    {
        auto start = std::chrono::high_resolution_clock::now();
        uint64_t ans = solve_new(n, start_L);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double seconds = elapsed.count();
        
        uint64_t candidates_checked = (ans >= start_L) ? (ans - start_L + 1) : 1;
        double speed = candidates_checked / seconds;
        
        std::ostringstream oss;
        oss << "n = " << std::setw(2) << n 
            << " | min k = " << std::setw(12) << ans 
            << " | Time: " << std::fixed << std::setprecision(4) << std::setw(8) << seconds << " s"
            << " | Speed: " << std::fixed << std::setprecision(2) << std::setw(8) << (speed / 1e6) << " M candidates/s\n";
            
        std::string output_str = oss.str();
        std::cout << output_str;
        std::ofstream results_file("results-389-master.txt", std::ios::app);
        if (results_file)
            results_file << output_str;

        start_L = 1;
        std::ofstream fout("checkpoint-389.tmp");
        if (fout) 
        {
            fout << (n + 1) << " " << 1 << "\n";
            fout.close();
            std::error_code ec;
            std::filesystem::rename("checkpoint-389.tmp", "checkpoint-389-master.txt", ec);
        }
    }
    
    std::error_code ec;
    std::filesystem::remove("checkpoint-389-master.txt", ec);
    
    return 0;
}