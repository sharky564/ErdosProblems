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
#include <barrier>
#include <immintrin.h>

const unsigned int NUM_THREADS = std::thread::hardware_concurrency(); // 8
const unsigned int NUM_SEQ = 4;

struct alignas(16) PrimeData
{
    uint64_t magic;
    uint32_t p;
    uint8_t shift;
};

std::vector<PrimeData> primes;

void get_primes(uint32_t limit)
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
    for (uint32_t p = 2; p <= limit; ++p)
    {
        if (sieve[p])
        {
            uint8_t shift = std::bit_width(p) - 1;
            unsigned __int128 magic_128 = ((unsigned __int128)1 << (64 + shift)) / p + 1;
            primes.push_back({(uint64_t)magic_128, p, shift});
        }
    }
}

bool exact_check(uint64_t n, uint64_t k)
{
    uint32_t nu_2_prod = std::popcount(n - k - 1) - std::popcount(n) + k + 1;
    if (std::popcount(n) < nu_2_prod) [[unlikely]]
        return false;

    for (const auto& pd : primes)
    {
        uint32_t p = pd.p;
        if (p == 2)
            continue;
        if (p > 2 * k)
            break;

        uint64_t nu_prod = 0, nu_comb = 0, power = p;
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

    for (uint64_t i = 0; i <= k; ++i)
    {
        uint64_t temp = n - i;
        temp >>= std::countr_zero(temp);
        for (const auto& pd : primes)
        {
            uint32_t p = pd.p;
            if (p <= 2 * k)
            {
                while (temp % p == 0)
                    temp /= p;
                continue;
            }
            if (p * p > temp)
            {
                if (temp > 2 * k)
                {
                    uint64_t p_val = temp;
                    uint64_t nu_prod = 1;
                    uint64_t t2 = (n - i) / p_val;
                    while (t2 > 0 && t2 % p_val == 0)
                    {
                        nu_prod++;
                        t2 /= p_val;
                    }

                    uint64_t nu_comb = 0;
                    uint64_t power = p_val;
                    while (true)
                    {
                        uint64_t v_n = n / power;
                        nu_comb += (2 * n) / power - 2 * v_n;
                        if (power > (2 * n) / p_val)
                            break;
                        power *= p_val;
                    }
                    if (nu_prod > nu_comb)
                        return false;
                }
                break;
            }
            if (temp % p == 0)
            {
                uint64_t nu_prod = 0;
                while (temp % p == 0)
                {
                    nu_prod++;
                    temp /= p;
                }

                uint64_t nu_comb = 0;
                uint64_t power = p;
                while (true)
                {
                    uint64_t v_n = n / power;
                    nu_comb += (2 * n) / power - 2 * v_n;
                    if (power > (2 * n) / p)
                        break;
                    power *= p;
                }
                if (nu_prod > nu_comb)
                    return false;
            }
        }
    }
    return true;
}

template<uint64_t p>
inline void process_prime_p(uint64_t start_j, uint64_t W_block, uint64_t* rem, uint64_t k)
{
    for (uint64_t j = start_j; j < W_block; j += p)
    {
        uint64_t temp = rem[j] / p;
        while (true)
        {
            if (temp % p != 0)
                break;
            temp /= p;
        }
        rem[j] = temp;
    }
}

inline void process_p_dyn(uint32_t p, uint64_t magic, uint8_t shift, uint64_t start_j, uint64_t W_block, uint64_t* rem)
{
    for (uint64_t j = start_j; j < W_block; j += p)
    {
        uint64_t temp = (uint64_t)((((unsigned __int128)rem[j] * magic) >> 64) >> shift);
        while (true)
        {
            uint64_t q = (uint64_t)((((unsigned __int128)temp * magic) >> 64) >> shift);
            if (temp - q * p != 0)
                break;
            temp = q;
        }
        rem[j] = temp;
    }
}

uint64_t solve(uint64_t k, uint64_t start_L)
{
    const uint64_t CHUNK_SIZE = 1000000;
    const uint64_t BLOCK_SIZE = 65536;

    std::atomic<uint64_t> current_chunk{0};
    std::atomic<uint64_t> global_min_n{static_cast<uint64_t>(-1)};

    auto worker = [&]() {
        alignas(32) std::vector<uint64_t> rem(BLOCK_SIZE);

        while (true)
        {
            uint64_t chunk_id = current_chunk.fetch_add(1, std::memory_order_relaxed);
            uint64_t L_chunk = start_L + chunk_id * CHUNK_SIZE;
            uint64_t R_chunk = L_chunk + CHUNK_SIZE + k - 1;

            if (L_chunk > global_min_n.load(std::memory_order_relaxed))
                break;

            uint64_t max_p = std::max<uint64_t>(std::sqrt(2 * R_chunk) + 1, 2 * k);
            auto it = std::upper_bound(primes.begin(), primes.end(), max_p,[](uint64_t val, const PrimeData& pd) { return val < pd.p; });
            size_t chunk_total_primes = std::distance(primes.begin(), it);

            for (uint64_t block_L = L_chunk; block_L <= R_chunk; block_L += BLOCK_SIZE)
            {
                uint64_t block_R = std::min(R_chunk, block_L + BLOCK_SIZE - 1);
                uint64_t W_block = block_R - block_L + 1;

                uint64_t x = block_L;
                for (uint64_t j = 0; j < W_block; ++j, ++x) {
                    rem[j] = x >> std::countr_zero(x);
                }

                for (size_t idx = 1; idx < chunk_total_primes; ++idx)
                {
                    uint32_t p = primes[idx].p;
                    uint64_t magic = primes[idx].magic;
                    uint8_t shift = primes[idx].shift;

                    uint64_t num = block_L + p - 1;
                    uint64_t start_c = (uint64_t)((((unsigned __int128)num * magic) >> 64) >> shift);
                    uint64_t start_j = start_c * p - block_L;
                    if (start_j >= W_block) continue;

#define PROCESS_PRIME(p_val) \
    case p_val: process_prime_p<p_val>(start_j, W_block, rem.data(), k); break;

                    switch (p)
                    {
                        PROCESS_PRIME(3) PROCESS_PRIME(5) PROCESS_PRIME(7) PROCESS_PRIME(11)
                        PROCESS_PRIME(13) PROCESS_PRIME(17) PROCESS_PRIME(19) PROCESS_PRIME(23)
                        default:
                            process_p_dyn(p, magic, shift, start_j, W_block, rem.data());
                    }
                }

                uint64_t consecutive = 0;
                for (uint64_t j = 0; j < W_block; ++j)
                {
                    if (rem[j] <= 1)
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

            if (chunk_id > NUM_THREADS && chunk_id % 1000 == 0)
            {
                uint64_t safe_L = start_L + (chunk_id - NUM_THREADS) * CHUNK_SIZE;
                std::ofstream fout("checkpoint-396.tmp");
                if (fout)
                {
                    fout << k << " " << safe_L << "\n";
                    fout.close();
                    std::error_code ec;
                    std::filesystem::rename("checkpoint-396.tmp", "checkpoint-396.txt", ec);
                }
            }
        }
    };

    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < NUM_THREADS; ++i)
        threads.emplace_back(worker);
    for (auto& t : threads)
        t.join();

    return global_min_n.load();
}

int main()
{
    std::cout << "Detected " << NUM_THREADS << " logical cores. Using C++ Thread Pool...\n";
    std::cout << "Generating primes up to 200,000,000...\n";
    auto start_primes = std::chrono::high_resolution_clock::now();
    get_primes(200000000);
    auto end_primes = std::chrono::high_resolution_clock::now();
    std::cout << "Primes generated in " << std::chrono::duration<double>(end_primes - start_primes).count()
              << " seconds.\n\n";

    uint64_t start_k = 1;
    uint64_t start_L = 1;

    if (std::filesystem::exists("checkpoint-396.txt"))
    {
        std::ifstream fin("checkpoint-396.txt");
        if (fin >> start_k >> start_L)
        {
            std::cout << "--> Resuming from checkpoint: k = " << start_k << ", L_batch = " << start_L << "\n\n";
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
        std::ofstream results_file("results-396.txt", std::ios::app);
        if (results_file)
            results_file << output_str;

        start_L = ans;
        std::ofstream fout("checkpoint-396.tmp");
        if (fout)
        {
            fout << (k + 1) << " " << start_L << "\n";
            fout.close();
            std::error_code ec;
            std::filesystem::rename("checkpoint-396.tmp", "checkpoint-396.txt", ec);
        }
    }

    std::error_code ec;
    std::filesystem::remove("checkpoint-396.txt", ec);

    return 0;
}