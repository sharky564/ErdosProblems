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
            for (uint64_t i : std::views::iota(p * p, limit + 1) | std::views::stride(p))
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
    uint64_t nu_2_prod = std::popcount(n - k - 1) - std::popcount(n) + k + 1;
    if (std::popcount(n) < nu_2_prod) [[unlikely]]
        return false;

    for (uint64_t p : primes) 
    {
        if (p == 2)
            continue;
        if (p > 2 * k)
            break;
            
        uint64_t nu_prod = 0;
        uint64_t nu_comb = 0;
        uint64_t power = p;
        while (true) 
        {
            [[assume(power > 0)]];            
            uint64_t v_n = n / power;
            nu_prod += v_n - (n - k - 1) / power;
            nu_comb += (2 * n) / power - 2 * v_n;
            if (power > (2 * n) / p)
                break;
            power *= p;
        }
        if (nu_prod > nu_comb) [[unlikely]]
            return false;
    }
    return true;
}

template<uint64_t p>
inline void process_prime_p(uint64_t start_c, uint64_t start_j, uint64_t W_block, uint8_t* valid, uint64_t* rem, uint64_t k) {
    if (p > 2 * k) 
    {
        uint64_t c = start_c;
        for (uint64_t j = start_j; j < W_block; j += p, ++c) 
        {
            if (valid[j]) 
            {
                bool is_val = true;
                if (c * 2 < p) 
                {
                    is_val = false;
                } 
                else if (c >= p) 
                {
                    uint64_t nu = 0;
                    uint64_t temp = c;
                    while (temp > 0 && temp % p == 0) 
                    {
                        ++nu;
                        temp /= p;
                    }
                    uint64_t carries = 0;
                    uint64_t carry = 0;
                    while (temp > 0) 
                    {
                        uint64_t digit = temp % p;
                        if (digit * 2 + carry >= p) 
                        {
                            ++carries;
                            carry = 1;
                            if (carries > nu)
                                break;
                        } 
                        else 
                        {
                            carry = 0;
                        }
                        temp /= p;
                    }
                    if (carries <= nu) is_val = false;
                }
                
                if (!is_val) [[unlikely]] 
                {
                    valid[j] = 0;
                } 
                else 
                {
                    uint64_t temp = rem[j] / p;
                    if (temp > 0) 
                    {
                        while (temp % p == 0)
                            temp /= p;
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
                uint64_t temp = rem[j] / p;
                if (temp > 0) 
                {
                    while (temp % p == 0)
                        temp /= p;
                }
                rem[j] = temp;
            }
        }
    }
}

inline void process_p_dyn(uint64_t p, uint64_t start_c, uint64_t start_j, uint64_t W_block, uint8_t* valid, uint64_t* rem, uint64_t k) {
    if (p > 2 * k) 
    {
        uint64_t c = start_c;
        for (uint64_t j = start_j; j < W_block; j += p, ++c) 
        {
            if (valid[j]) 
            {
                bool is_val = true;
                if (c * 2 < p) 
                {
                    is_val = false;
                }
                else if (c >= p) 
                {
                    uint64_t nu = 0;
                    uint64_t temp = c;
                    while (temp > 0 && temp % p == 0) 
                    {
                        ++nu;
                        temp /= p;
                    }
                    uint64_t carries = 0;
                    uint64_t carry = 0;
                    while (temp > 0) 
                    {
                        uint64_t digit = temp % p;
                        if (digit * 2 + carry >= p) 
                        {
                            ++carries;
                            carry = 1;
                            if (carries > nu)
                                break;
                        } 
                        else 
                        {
                            carry = 0;
                        }
                        temp /= p;
                    }
                    if (carries <= nu)
                        is_val = false;
                }
                if (!is_val) [[unlikely]] 
                {
                    valid[j] = 0;
                } 
                else 
                {
                    uint64_t temp = rem[j] / p;
                    if (temp > 0) 
                    {
                        while (temp % p == 0)
                            temp /= p;
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
                uint64_t temp = rem[j] / p;
                if (temp > 0) 
                {
                    while (temp % p == 0)
                        temp /= p;
                }
                rem[j] = temp;
            }
        }
    }
}

uint64_t solve(uint64_t k, uint64_t start_L) 
{
    const uint64_t CHUNK_SIZE = 10000000;
    const uint64_t BLOCK_SIZE = 32768;
    const uint64_t CHUNKS_PER_BATCH = NUM_THREADS * NUM_SEQ; 
    
    uint64_t L_batch = start_L;
    
    std::atomic<uint64_t> global_min_n{static_cast<uint64_t>(-1)};
    std::atomic<bool> found_in_batch{false};

#define PROCESS_PRIME(p_val) \
    case p_val: process_prime_p<p_val>(start_c, start_j, W_block, valid.data(), rem.data(), k); break;

    auto worker = [&](std::atomic<int>& current_chunk){
        std::vector<uint8_t> valid(BLOCK_SIZE);
        std::vector<uint64_t> rem(BLOCK_SIZE);

        while (true) 
        {
            int chunk_id = current_chunk.fetch_add(1, std::memory_order_relaxed);
            if (chunk_id >= CHUNKS_PER_BATCH)
                break;
            
            uint64_t L_chunk = L_batch + chunk_id * CHUNK_SIZE;
            uint64_t R_chunk = L_chunk + CHUNK_SIZE + k - 1;
            
            if (L_chunk > global_min_n.load(std::memory_order_relaxed))
                continue;

            uint64_t max_p = std::max<uint64_t>(std::sqrt(2 * R_chunk) + 1, 2 * k);
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

                    if (p == 2) 
                    {
                        for (uint64_t j = start_j; j < W_block; j += 2) 
                        {
                            if (valid[j]) 
                            {
                                uint64_t temp = rem[j];
                                temp >>= std::countr_zero(temp);
                                rem[j] = temp;
                            }
                        }
                        continue;
                    }

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
                        default:
                            process_p_dyn(p, start_c, start_j, W_block, valid.data(), rem.data(), k);
                            break;
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
        }
    };

    while (true) 
    {
        found_in_batch = false;
        std::atomic<int> current_chunk{0};
        std::vector<std::thread> threads;
        for (unsigned int i = 0; i < NUM_THREADS; ++i)
            threads.emplace_back(worker, std::ref(current_chunk));
        for (auto& t : threads)
            t.join();

        if (found_in_batch)
            return global_min_n.load();

        L_batch += CHUNKS_PER_BATCH * CHUNK_SIZE;
        std::ofstream fout("checkpoint.tmp");
        if (fout) 
        {
            fout << k << " " << L_batch << "\n";
            fout.close();
            std::error_code ec;
            std::filesystem::rename("checkpoint.tmp", "checkpoint.txt", ec);
        }
    }
}

int main() 
{
    std::cout << "Detected " << NUM_THREADS << " logical cores. Using C++ Thread Pool...\n";
    std::cout << "Generating primes up to 200,000,000..." << std::endl;
    auto start_primes = std::chrono::high_resolution_clock::now();
    get_primes(200000000); 
    auto end_primes = std::chrono::high_resolution_clock::now();
    std::cout << "Primes generated in " << std::chrono::duration<double>(end_primes - start_primes).count() 
              << " seconds.\n\n";

    uint64_t start_k = 1;
    uint64_t start_L = 1;

    if (std::filesystem::exists("checkpoint.txt")) 
    {
        std::ifstream fin("checkpoint.txt");
        if (fin >> start_k >> start_L) 
        {
            std::cout << "--> Resuming from Checkpoint: k = " << start_k << ", L_batch = " << start_L << "\n\n";
        } 
        else 
        {
            start_k = 1; 
            start_L = 1;
        }
    }

    for (uint64_t k = start_k; k <= 20; ++k) 
    {
        auto start = std::chrono::high_resolution_clock::now();
        uint64_t ans = solve(k, start_L);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed = end - start;
        double seconds = elapsed.count();
        
        uint64_t candidates_checked = (ans >= start_L) ? (ans - start_L + 1) : 1;
        double speed = candidates_checked / seconds;
    
        std::ostringstream oss;
        oss << "k = " << std::setw(2) << k 
            << " | min n = " << std::setw(15) << ans 
            << " | Time: " << std::fixed << std::setprecision(4) << std::setw(12) << seconds << " s"
            << " | Speed: " << std::fixed << std::setprecision(2) << std::setw(8) << (speed / 1e6) << " M candidates/s\n";
        
        std::string output_str = oss.str();
        std::cout << output_str;
        std::ofstream results_file("results.txt", std::ios::app);
        if (results_file)
            results_file << output_str;

        start_L = ans;
        std::ofstream fout("checkpoint.tmp");
        if (fout) 
        {
            fout << (k + 1) << " " << start_L << "\n";
            fout.close();
            std::error_code ec;
            std::filesystem::rename("checkpoint.tmp", "checkpoint.txt", ec);
        }
    }
    
    std::error_code ec;
    std::filesystem::remove("checkpoint.txt", ec);
    
    return 0;
}