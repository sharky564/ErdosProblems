#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

constexpr int S_CHUNKS = 32;

const unsigned int NUM_THREADS = std::thread::hardware_concurrency();

struct PrimeData
{
    uint64_t magic;
    uint64_t inv_p;
    uint64_t limit;
    uint32_t p;
    uint8_t shift;
};

struct PrimeFast
{
    uint64_t inv_p;
    uint64_t limit;
};

struct BucketItem
{
    uint32_t p_idx;
    uint32_t offset;
};

struct FastBucket
{
    BucketItem *data;
    uint32_t count;
    uint32_t cap;

    FastBucket()
    {
        cap = 0;
        data = nullptr;
        count = 0;
    }
    ~FastBucket()
    {
        std::free(data);
    }
    FastBucket(const FastBucket &) = delete;
    FastBucket &operator=(const FastBucket &) = delete;

    inline void clear()
    {
        count = 0;
    }

    void reserve(uint32_t new_cap)
    {
        if (new_cap > cap) [[unlikely]]
        {
            cap = new_cap;
            data = static_cast<BucketItem *>(std::realloc(data, cap * sizeof(BucketItem)));
        }
    }

    inline void push_back_unsafe(uint32_t p_idx, uint32_t offset)
    {
        data[count].p_idx = p_idx;
        data[count].offset = offset;
        ++count;
    }
};

struct alignas(64) AlignedAtomic
{
    std::atomic<uint64_t> val{UINT64_MAX};
};

std::vector<PrimeData> primes;
std::vector<PrimeFast> primes_fast;

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
            uint64_t inv = 0;
            if (p % 2 != 0)
            {
                inv = p;
                for (int i = 0; i < 5; ++i)
                    inv *= 2 - p * inv;
            }
            primes.push_back({(uint64_t)magic_128, inv, UINT64_MAX / p, p, shift});
            primes_fast.push_back({inv, UINT64_MAX / p});
        }
    }
}

template <uint64_t K> bool exact_check(uint64_t n)
{
    uint32_t nu_2_prod = std::popcount(n - K - 1) - std::popcount(n) + K + 1;
    if (std::popcount(n) < nu_2_prod) [[unlikely]]
        return false;

    for (const auto &pd : primes)
    {
        uint32_t p = pd.p;
        if (p == 2)
            continue;
        if (p > 2 * K)
            break;

        uint64_t nu_prod = 0, nu_comb = 0, power = p;
        while (true)
        {
            uint64_t v_n = n / power;
            nu_prod += v_n - (n - K - 1) / power;
            nu_comb += (2 * n) / power - 2 * v_n;
            if (power > (2 * n) / p)
                break;
            power *= p;
        }
        if (nu_prod > nu_comb)
            return false;
    }

    for (uint64_t i = 0; i <= K; ++i)
    {
        uint64_t temp = n - i;
        temp >>= std::countr_zero(temp);
        for (const auto &pd : primes)
        {
            uint32_t p = pd.p;
            if (p == 2)
                continue;
            if (p <= 2 * K)
            {
                uint64_t q = temp * pd.inv_p;
                while (q <= pd.limit)
                {
                    temp = q;
                    q = temp * pd.inv_p;
                }
                continue;
            }
            if (p * p > temp)
            {
                if (temp > 2 * K)
                {
                    uint64_t p_val = temp, nu_prod = 1, t2 = (n - i) / p_val;
                    while (t2 > 0 && t2 % p_val == 0)
                    {
                        nu_prod++;
                        t2 /= p_val;
                    }
                    uint64_t nu_comb = 0, power = p_val;
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
            uint64_t q = temp * pd.inv_p;
            if (q <= pd.limit)
            {
                uint64_t nu_prod = 0;
                while (q <= pd.limit)
                {
                    nu_prod++;
                    temp = q;
                    q = temp * pd.inv_p;
                }
                uint64_t nu_comb = 0, power = p;
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

inline void process_p_dyn(uint32_t p, uint64_t inv_p, uint64_t limit, uint32_t &start_j, uint32_t W_block,
                          uint64_t *rem)
{
    uint32_t j = start_j;
    for (; j < W_block; j += p)
    {
        uint64_t val = rem[j], temp = val * inv_p, q = temp * inv_p;
        if (q <= limit) [[unlikely]]
        {
            temp = q;
            while (true)
            {
                q = temp * inv_p;
                if (q > limit)
                    break;
                temp = q;
            }
        }
        rem[j] = temp;
    }
    start_j = j - W_block;
}

template <uint64_t K> uint64_t solve_impl(uint64_t start_L)
{
    const uint64_t SUB_CHUNK_SIZE = 1048576ULL;
    const uint64_t SUPER_CHUNK_SIZE = S_CHUNKS * SUB_CHUNK_SIZE;

    std::atomic<uint64_t> current_chunk{0};
    std::atomic<uint64_t> global_min_n{static_cast<uint64_t>(-1)};
    std::vector<AlignedAtomic> active_chunks(NUM_THREADS);

    auto start_time = std::chrono::high_resolution_clock::now();

    auto worker = [&](uint32_t thread_index) {
        const uint32_t OPT_BLOCK_SIZE = 65536;
        const uint32_t BLOCK_SHIFT = std::countr_zero(OPT_BLOCK_SIZE);
        const uint32_t BLOCK_MASK = OPT_BLOCK_SIZE - 1;
        const uint32_t FAST_BUCKET_SIZE = SUB_CHUNK_SIZE / OPT_BLOCK_SIZE + 1;

        std::vector<uint64_t> rem(OPT_BLOCK_SIZE + 32);
        std::vector<uint32_t> prime_offsets;
        std::vector<uint32_t> active_large_primes;

        FastBucket buckets[FAST_BUCKET_SIZE];

        while (true)
        {
            uint64_t super_chunk_id = current_chunk.fetch_add(1, std::memory_order_relaxed);
            active_chunks[thread_index].val.store(super_chunk_id, std::memory_order_release);

#ifdef BENCHMARK
            if (super_chunk_id >= (10000ULL + S_CHUNKS - 1) / S_CHUNKS)
            {
                active_chunks[thread_index].val.store(UINT64_MAX, std::memory_order_release);
                break;
            }
#endif
            uint64_t L_super = start_L + super_chunk_id * SUPER_CHUNK_SIZE;
            uint64_t R_super = L_super + SUPER_CHUNK_SIZE + K - 1;

            if (L_super > global_min_n.load(std::memory_order_relaxed))
            {
                active_chunks[thread_index].val.store(UINT64_MAX, std::memory_order_release);
                break;
            }

            uint64_t max_p = std::max<uint64_t>(std::sqrt(2 * R_super) + 1, 2 * K);
            auto it = std::upper_bound(primes.begin(), primes.end(), max_p,
                                       [](uint64_t val, const PrimeData &pd) { return val < pd.p; });
            size_t chunk_total_primes = std::distance(primes.begin(), it);

            if (prime_offsets.size() < chunk_total_primes)
                prime_offsets.resize(chunk_total_primes);
            auto it_large = std::upper_bound(primes.begin(), primes.begin() + chunk_total_primes, OPT_BLOCK_SIZE,
                                             [](uint32_t val, const PrimeData &pd) { return val < pd.p; });
            size_t first_large_prime_idx = std::distance(primes.begin(), it_large);

            uint32_t p_thresh = std::max<uint32_t>(2 * K, (uint32_t)std::cbrt(2.0 * R_super) + 2);
            active_large_primes.clear();

            for (size_t idx = first_large_prime_idx; idx < chunk_total_primes; ++idx)
            {
                uint32_t p = primes[idx].p;
                if (p > p_thresh)
                {
                    uint64_t p2 = (uint64_t)p * p;
                    uint64_t c = L_super / p2;
                    if (2 * R_super < (2 * c + 1) * p2)
                        continue; // Unconditionally discard
                }
                active_large_primes.push_back((uint32_t)idx);
                uint64_t num = L_super + p - 1;
                uint64_t start_c =
                    (uint64_t)((((unsigned __int128)num * primes[idx].magic) >> 64) >> primes[idx].shift);
                prime_offsets[idx] = (uint32_t)(start_c * p - L_super);
            }
            uint32_t safe_cap = (uint32_t)active_large_primes.size();
            for (int i = 0; i < FAST_BUCKET_SIZE; ++i)
                buckets[i].reserve(safe_cap);

            for (int sub = 0; sub < S_CHUNKS; ++sub)
            {
                uint64_t L_chunk = L_super + sub * SUB_CHUNK_SIZE;
                uint64_t R_chunk = L_chunk + SUB_CHUNK_SIZE + K - 1;
                if (L_chunk > global_min_n.load(std::memory_order_relaxed))
                    break;

                uint32_t CHUNK_W = (uint32_t)(R_chunk - L_chunk + 1);
                for (int i = 0; i < FAST_BUCKET_SIZE; ++i)
                    buckets[i].clear();

                for (size_t idx = 1; idx < first_large_prime_idx; ++idx)
                {
                    uint32_t p = primes[idx].p;
                    uint64_t num = L_chunk + p - 1;
                    uint64_t start_c =
                        (uint64_t)((((unsigned __int128)num * primes[idx].magic) >> 64) >> primes[idx].shift);
                    prime_offsets[idx] = (uint32_t)(start_c * p - L_chunk);
                }
                for (uint32_t idx : active_large_primes)
                {
                    uint32_t start_j = prime_offsets[idx];
                    uint32_t p = primes[idx].p;

                    while (start_j < SUB_CHUNK_SIZE)
                    {
                        buckets[start_j >> BLOCK_SHIFT].push_back_unsafe(idx, start_j & BLOCK_MASK);
                        start_j += p;
                    }
                    prime_offsets[idx] = start_j - SUB_CHUNK_SIZE;
                    while (start_j < CHUNK_W)
                    {
                        buckets[start_j >> BLOCK_SHIFT].push_back_unsafe(idx, start_j & BLOCK_MASK);
                        start_j += p;
                    }
                }

                uint32_t overlap = 0, j = 0;
                for (uint32_t block_start = 0; block_start < CHUNK_W; block_start += OPT_BLOCK_SIZE)
                {
                    uint32_t block_idx = block_start >> BLOCK_SHIFT;
                    uint64_t block_L = L_chunk + block_start;
                    uint64_t block_R = std::min(R_chunk, block_L + OPT_BLOCK_SIZE - 1);
                    uint32_t num_new = (uint32_t)(block_R - block_L + 1);

                    uint64_t x = block_L;
                    for (uint32_t idx_new = 0; idx_new < num_new; ++idx_new, ++x)
                        rem[overlap + idx_new] = x >> std::countr_zero(x);

                    uint64_t *rem_ptr = rem.data() + overlap;

                    for (size_t idx = 1; idx < first_large_prime_idx; ++idx)
                    {
                        uint32_t p = primes[idx].p, start_j = prime_offsets[idx];
                        if (start_j < num_new)
                        {
                            process_p_dyn(p, primes[idx].inv_p, primes[idx].limit, start_j, num_new, rem_ptr);
                            prime_offsets[idx] = start_j;
                        }
                        else
                        {
                            prime_offsets[idx] = start_j - num_new;
                        }
                    }

                    uint32_t b_count = buckets[block_idx].count;
                    BucketItem *b_items = buckets[block_idx].data;

                    for (uint32_t i = 0; i < b_count; ++i)
                    {
                        if (i + 8 < b_count) [[likely]]
                            __builtin_prefetch(&primes_fast[b_items[i + 8].p_idx], 0);
                        uint32_t p_idx = b_items[i].p_idx, offset = b_items[i].offset;
                        uint64_t inv_p = primes_fast[p_idx].inv_p, limit = primes_fast[p_idx].limit;
                        uint64_t val = rem_ptr[offset], temp = val * inv_p, q = temp * inv_p;

                        if (q <= limit) [[unlikely]]
                        {
                            temp = q;
                            while (true)
                            {
                                q = temp * inv_p;
                                if (q > limit)
                                    break;
                                temp = q;
                            }
                        }
                        rem_ptr[offset] = temp;
                    }

                    uint32_t W_search = overlap + num_new;
                    while (j + K < W_search)
                    {
                        if (rem[j + K] != 1) [[likely]]
                        {
                            j += K + 1;
                        }
                        else
                        {
                            int i = K - 1;
                            while (i >= 0 && rem[j + i] == 1)
                                --i;

                            if (i < 0)
                            {
                                uint64_t n = block_L - overlap + j + K;
                                if (n > K && n < global_min_n.load(std::memory_order_relaxed))
                                {
                                    if (exact_check<K>(n))
                                    {
                                        uint64_t current = global_min_n.load(std::memory_order_relaxed);
                                        while (n < current && !global_min_n.compare_exchange_weak(
                                                                  current, n, std::memory_order_relaxed))
                                        {
                                        }
                                    }
                                }
                                ++j;
                            }
                            else
                            {
                                j += i + 1;
                            }
                        }
                    }

                    if (W_search >= K)
                    {
                        for (uint32_t i = 0; i < K; ++i)
                            rem[i] = rem[W_search - K + i];
                        j -= (W_search - K);
                        overlap = K;
                    }
                    else
                    {
                        overlap = W_search;
                    }
                }
            }

            static std::atomic<bool> is_writing{false};
            if (super_chunk_id > NUM_THREADS && super_chunk_id % std::max(1, 1024 / S_CHUNKS) == 0)
            {
                bool expected = false;
                if (is_writing.compare_exchange_strong(expected, true))
                {
                    uint64_t min_active = super_chunk_id;
                    for (uint32_t t = 0; t < NUM_THREADS; ++t)
                        min_active = std::min(min_active, active_chunks[t].val.load(std::memory_order_acquire));

                    uint64_t safe_L = start_L + min_active * SUPER_CHUNK_SIZE;
                    std::ofstream fout("checkpoint-396.tmp");
                    if (fout)
                    {
                        fout << K << " " << safe_L << "\n";
                        fout.close();
                        std::error_code ec;
                        std::filesystem::rename("checkpoint-396.tmp", "checkpoint-396.txt", ec);

                        double speed = (safe_L - start_L) / std::chrono::duration<double>(
                                                                std::chrono::high_resolution_clock::now() - start_time)
                                                                .count();
                        std::cout << "\r[Checkpoint] k = " << std::setw(2) << K << " | Candidate L = " << safe_L
                                  << " | Speed: " << std::fixed << std::setprecision(2) << (speed / 1e6)
                                  << " M candidates/s   " << std::flush;
                    }
                    is_writing.store(false, std::memory_order_release);
                }
            }
        }
    };

    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < NUM_THREADS; ++i)
        threads.emplace_back(worker, i);
    for (auto &t : threads)
        t.join();

    return global_min_n.load();
}

uint64_t solve(uint64_t k, uint64_t start_L)
{
    switch (k)
    {
    case 1:
        return solve_impl<1>(start_L);
    case 2:
        return solve_impl<2>(start_L);
    case 3:
        return solve_impl<3>(start_L);
    case 4:
        return solve_impl<4>(start_L);
    case 5:
        return solve_impl<5>(start_L);
    case 6:
        return solve_impl<6>(start_L);
    case 7:
        return solve_impl<7>(start_L);
    case 8:
        return solve_impl<8>(start_L);
    case 9:
        return solve_impl<9>(start_L);
    case 10:
        return solve_impl<10>(start_L);
    case 11:
        return solve_impl<11>(start_L);
    case 12:
        return solve_impl<12>(start_L);
    case 13:
        return solve_impl<13>(start_L);
    case 14:
        return solve_impl<14>(start_L);
    case 15:
        return solve_impl<15>(start_L);
    case 16:
        return solve_impl<16>(start_L);
    case 17:
        return solve_impl<17>(start_L);
    case 18:
        return solve_impl<18>(start_L);
    case 19:
        return solve_impl<19>(start_L);
    case 20:
        return solve_impl<20>(start_L);
    default:
        return 0;
    }
}

int main()
{
    std::cout << "Detected " << NUM_THREADS << " logical cores. Using C++ Thread Pool...\n";
    std::cout << "Generating primes up to 200,000,000...\n";
    auto start_primes = std::chrono::high_resolution_clock::now();
    get_primes(200000000);
    std::cout << "Primes generated in "
              << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_primes).count()
              << " seconds.\n\n";

#ifdef BENCHMARK
    uint64_t k = 14;
    uint64_t start_L = 359'000'000'000'000ULL;
    uint64_t total_chunks = ((10000ULL + S_CHUNKS - 1) / S_CHUNKS) * S_CHUNKS;
    uint64_t CHUNK_SIZE = 1048576ULL;

    std::cout << "--- BENCHMARK MODE ---\n";
    std::cout << "Testing S_CHUNKS=" << S_CHUNKS << " | k=" << k
              << " | Workload: " << (total_chunks * CHUNK_SIZE) / 1000000 << " M candidates\n\n";

    for (int run = 1; run <= 5; ++run)
    {
        auto start = std::chrono::high_resolution_clock::now();
        uint64_t ans = solve(k, start_L);
        double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        double speed = (total_chunks * CHUNK_SIZE) / seconds;
        std::cout << "Run " << run << " | Time: " << std::fixed << std::setprecision(4) << std::setw(8) << seconds
                  << " s"
                  << " | Speed: " << std::fixed << std::setprecision(2) << std::setw(8) << (speed / 1e6)
                  << " M candidates/s\n";
    }
#else
    uint64_t start_k = 1, start_L = 1;

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
        double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        uint64_t candidates_checked = (ans >= start_L) ? (ans - start_L + 1) : 1;
        double speed = candidates_checked / seconds;

        std::ostringstream oss;
        oss << "\rk = " << std::setw(2) << k << " | min n = " << std::setw(15) << ans << " | Time: " << std::fixed
            << std::setprecision(4) << std::setw(12) << seconds << " s"
            << " | Speed: " << std::fixed << std::setprecision(2) << std::setw(8) << (speed / 1e6)
            << " M candidates/s\n";

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
#endif
    return 0;
}