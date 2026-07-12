// Erdos Problem #389 solver - fixed-point logarithmic sieve.
//
// For each n, find the least k with C(n+k-1, k) | C(n+2k-1, k). Writing
// N = n+k-1 and m = n-1, the divisibility is equivalent to
//   nu_p((2N-m)!) + nu_p(m!) >= 2*nu_p(N!)  for every prime p,
// i.e. C(N,m) | C(2N-m,N): a product of m consecutive integers ending at N
// against a near-central binomial. The search variable is N; k = N - m.
//
// Soundness of the sieve (Kummer digit analysis): for a prime p | N-i the
// digit-0 addition of (N-m) + N in base p carries freely iff 2i >= m, so only
// run positions i <= K = (m-1)/2 need fully accounted smoothness - those are
// the positions this sieve scans. For them, a carry at digit >= 1 is possible
// iff some base-p digit of floor(x/p) is at least (p+1)/2, which is the same
// criterion the sieve's chunk-wide Kummer skips test for #396-shaped runs,
// and multi-carry digit sums are dominated the same way; hence every accounting
// rule below (log accumulator, presieve patterns, skips, thresholds, prime
// bounds) is a valid relaxation of the #389 condition. Survivors go to the
// exact oracle exact_check389, so reported minima are exact.
//
// Pairing: n = 2K+2 and n = 2K+3 share the same K, so one scan serves both
// (two minima, two oracle calls per survivor, stop when both are found).
// Answers are NOT monotone in n (e.g. n=7's k=984 < n=6's k=2475 - the m!
// term can help), so every pair scans from 1.
//
// Modes: normal sweep (checkpoint-389.txt: "K L min1 min2"; NMAX_RT env,
// default 30) and worker mode `./bin <n> <start_L> <end_L>` printing
// "RESULT n N" for the multi-host coordinator (use "reset_start": true).
// Accumulator configs (default u8 scale-4 / -DSCALE3 / -DACC16) and the PGO
// recipe are documented in the companion erdos_396_v11.cpp; the ceilings
// there apply to N.

#ifndef MAXP
#define MAXP 500'000'000
#endif
#ifndef KMAX
#define KMAX 20
#endif
#ifndef RUNS
#define RUNS 5
#endif
#ifndef BENCH_CANDS
#define BENCH_CANDS 10485760000ULL
#endif
#ifndef BLOCKSZ
#define BLOCKSZ 65536
#endif
// Chunk size is chosen at RUNTIME per solve: 2^25 while the per-chunk prime
// set is small (bucket working set dominates), 2^26 once it is large (per-prime
// setup dominates). The bucket item packing is sized once for the maximum.
// CHUNK_SWITCH_P: use the big chunk when the sieving prime bound reaches this.
// Env override CHUNK_LOG=25|26 forces either side for A/B without recompiling.
#ifndef CHUNK_SWITCH_P
#define CHUNK_SWITCH_P 8000000.0
#endif
#ifndef BUCKET_RESERVE
#define BUCKET_RESERVE (BLOCKSZ / 2)
#endif

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

static unsigned int detect_threads()
{
    if (const char *e = std::getenv("NT")) { int v = std::atoi(e); if (v > 0) return (unsigned)v; }
    return std::thread::hardware_concurrency();
}
const unsigned int NUM_THREADS = detect_threads();

constexpr uint32_t OPT_BLOCK_SIZE = BLOCKSZ;
constexpr uint32_t BLOCK_SHIFT = std::countr_zero(OPT_BLOCK_SIZE);
constexpr uint32_t BLOCK_MASK = OPT_BLOCK_SIZE - 1;
static_assert((OPT_BLOCK_SIZE & (OPT_BLOCK_SIZE - 1)) == 0, "block size must be a power of two");
static_assert(OPT_BLOCK_SIZE >= 16384, "p^2 exclusion proof needs p > (2n)^(1/4); keep blocks >= 16384");

// ACCUMULATOR CONFIGS - pick one at compile time. All run the same sieve;
// they trade table width and rounding budget against the range ceiling.
//   (default)  u8,  LOG_SCALE 4,  LOG_SLACK 22, ceiling R ~ 5.8e17 (2^59).
//              Best filtering at the lowest bandwidth. Slack: at most 37 odd
//              prime-power terms below 2^59 (3^37 > 2^58), llround deviation
//              <= 0.5/term -> deficit <= 19, +3 margin = 22. Overflow:
//              4*log2(R) + 19 <= 255 holds to 2^59. Minimum missing-prime
//              mass 4*log2(cbrt(2R)) dwarfs slack+overshoot at all practical
//              ranges, so survivors stay scarce.
//   -DSCALE3   u8,  LOG_SCALE 3,  LOG_SLACK 28, ceiling R ~ 1.5e23 (2^77).
//              For searches past the scale-4 ceiling. Slack: at most 48 odd
//              prime-power terms below 2^77 (3^48 > 2^76) -> deficit <= 24,
//              +4 margin = 28. Overflow: 3*log2(R) + 24 <= 255 to 2^77.
//              Missing-prime mass in its intended regime (R > 5.8e17) is
//              3*log2(cbrt(2R)) > 60 units >> 28+24: no survivor flood.
//              Coarser units filter worse at small R; prefer the default
//              anywhere under the scale-4 ceiling.
//   -DACC16    u16, LOG_SCALE 32, LOG_SLACK 48. Wide-margin fallback: with
//              at most 39 odd prime-power terms below 2^63 the deficit is
//              <= 20, far under the 48 budget, and acc peaks near
//              32*63 + 20 << 65535. Guarded to R <= 8.5e18 because the
//              bucket item's 10-bit LOG field needs 32*log2(p) <= 1023,
//              i.e. p < 2^31.97 and R < p^2/2 - past which 2R overflows
//              u64 arithmetic anyway. Doubling the accumulator traffic
//              costs ~10%+ speed; use a u8 config unless you want the
//              extra margin.
#if defined(ACC16) && defined(SCALE3)
#error "ACC16 and SCALE3 are mutually exclusive"
#endif
#if defined(ACC16)
using acc_t = uint16_t;
constexpr int LOG_SCALE = 32;
constexpr int LOG_SLACK = 48;
constexpr double ACC_MAX = 65535.0;
constexpr double ACC_OVERSHOOT = 20.0; // 0.5 * 39 terms (3^39 > 2^61)
constexpr uint64_t RANGE_CAP = 8'500'000'000'000'000'000ULL;
#define ACC_CEIL_MSG "the u16 build's u64-safe range (~8.5e18)"
#elif defined(SCALE3)
using acc_t = uint8_t;
constexpr int LOG_SCALE = 3;
constexpr int LOG_SLACK = 28;
constexpr double ACC_MAX = 255.0;
constexpr double ACC_OVERSHOOT = 24.0; // 0.5 * 48 terms (3^48 > 2^76)
constexpr uint64_t RANGE_CAP = UINT64_MAX;
#define ACC_CEIL_MSG "the scale-3 u8 ceiling (~1.5e23)"
#else
using acc_t = uint8_t;
constexpr int LOG_SCALE = 4;
constexpr int LOG_SLACK = 22;
constexpr double ACC_MAX = 255.0;
constexpr double ACC_OVERSHOOT = 19.0; // 0.5 * 37 terms (3^37 > 2^58)
constexpr uint64_t RANGE_CAP = UINT64_MAX;
#define ACC_CEIL_MSG "the u8 accumulator ceiling (~5.8e17): rebuild with -DSCALE3"
#endif
constexpr uint64_t BIGPOW_CAP = 1'000'000'000'000'000'000ULL;

struct PrimeData
{
    uint64_t magic;
    uint64_t inv_p;   // used by exact_check only
    uint64_t limit;   // used by exact_check only
    uint32_t p;
    uint16_t log2s;   // floor(32*log2(p))
    uint8_t shift;
};

struct Strider        // odd prime power q = p^e <= OPT_BLOCK_SIZE
{
    uint64_t magic;
    uint32_t q;
    uint32_t p;       // base prime (Kummer test / max_p cap)
    uint16_t log2s;   // LOG(p) - one extra factor p per multiple of q
    uint8_t shift;
    bool is_base;     // e == 1
};

struct BigPower       // odd prime power q = p^e > OPT_BLOCK_SIZE, p <= OPT_BLOCK_SIZE
{
    uint64_t q;
    uint16_t log2s;   // LOG(p)
};

// Bucket item (reinsertion scheme, one u64, fully self-contained):
//   bits [0, OFF_BITS)              : offset within chunk
//   bits [OFF_BITS, 2*OFF_BITS)     : stride = min(p_or_q, STRIDE_MASK); exact
//                 for anything that can hit twice per chunk (p < CHUNK_W) -
//                 larger strides saturate, which just drops them after one add
//   bits [2*OFF_BITS, 64)           : LOG(p)  (<= 1023 for any 32-bit prime)
// Population pushes only the FIRST hit per prime; the drain loop adds LOG and
// re-pushes offset+stride (Oliveira e Silva style). Live bucket memory becomes
// one pending item per active prime instead of every hit in the chunk upfront -
// on multi-core boxes this keeps bucket traffic cache-resident instead of
// streaming tens of MB per chunk per thread through DRAM.
constexpr uint64_t CHUNK_MIN = 33554432ULL;
constexpr uint64_t CHUNK_MAX = 67108864ULL;
constexpr uint32_t OFF_BITS = std::countr_zero(CHUNK_MAX) + 1; // holds CHUNK_W at max
constexpr uint64_t OFF_MASK = (1ULL << OFF_BITS) - 1;
constexpr uint32_t STRIDE_SHIFT = OFF_BITS;
constexpr uint64_t STRIDE_MASK = (1ULL << OFF_BITS) - 1;         // exact for any p < CHUNK_W
constexpr uint32_t LG_SHIFT = 2 * OFF_BITS;
static_assert(2 * OFF_BITS + 10 <= 64, "chunk too large: LOG(p) needs 10 bits (CHUNK_MAX <= 2^26)");
static_assert(CHUNK_MAX + 64 <= (1ULL << OFF_BITS), "chunk offsets must fit the item offset field");
struct FastBucket
{
    uint64_t *data;
    uint32_t count;
    uint32_t cap;

    FastBucket() { cap = 0; data = nullptr; count = 0; }
    ~FastBucket() { std::free(data); }
    FastBucket(const FastBucket &) = delete;
    FastBucket &operator=(const FastBucket &) = delete;

    inline void clear() { count = 0; }

    void reserve(uint32_t new_cap)
    {
        if (new_cap > cap) [[unlikely]]
        {
            cap = new_cap;
            data = static_cast<uint64_t *>(std::realloc(data, cap * sizeof(uint64_t)));
        }
    }

    inline void push_back(uint64_t item)
    {
        if (count == cap) [[unlikely]]
            reserve(cap == 0 ? 32768 : cap * 2);
        data[count] = item;
        ++count;
    }
};

struct alignas(64) AlignedAtomic
{
    std::atomic<uint64_t> val{UINT64_MAX};
};

std::vector<PrimeData> primes;
// Population-hot fields split out of the 32-byte PrimeData (which stays for
// exact_check): sequential 21 B/prime streams instead of 32 B with dead weight.
std::vector<uint32_t> pop_p;
std::vector<uint64_t> pop_magic;
std::vector<uint8_t> pop_shift;
std::vector<uint64_t> pop_item; // (min(p,STRIDE_MASK) << STRIDE_SHIFT) | (LOG(p) << LG_SHIFT)
std::vector<Strider> small_striders;
std::vector<BigPower> big_powers;

// Presieve: the primes {3,5,7,11,13} plus the power 9 are folded into a single
// additive pattern with period 45045 = 9*5*7*11*13, applied per block as one
// contiguous (auto-vectorizable) u16 add instead of six scalar strided loops.
// Safe to hard-fold: p_thresh >= cbrt(2*CHUNK_SIZE)+2 ~ 408 and
// max_p >= sqrt(2*CHUNK_SIZE) ~ 8192, so these primes are never Kummer-skipped
// and never above the sieving cap.
constexpr uint32_t PRESIEVE_PERIOD = 45045;
// Second presieve pattern: {17,19,23,29}, period 17*19*23*29 = 215441 (~431 KB
// table, L2-resident). Fused into the same init pass - one extra vector load+add
// removes another ~0.19 scalar strided RMWs per candidate. Same safety argument.
constexpr uint32_t PRESIEVE2_PERIOD = 215441;
// Optional third pattern {31,37,41,43}, period 2018957 (~4 MB table): saves
// another ~0.11 strided RMWs per candidate but the table is L3-resident only.
// Opt-in with -DPRESIEVE3 and A/B it on your hardware.
#ifdef PRESIEVE3
constexpr uint32_t PRESIEVE3_PERIOD = 31u * 37u * 41u * 43u;
std::vector<acc_t> presieve3_pat;
#endif
std::vector<acc_t> presieve_pat;  // length PERIOD + OPT_BLOCK_SIZE
std::vector<acc_t> presieve2_pat; // length PERIOD2 + OPT_BLOCK_SIZE
std::vector<acc_t> tz_pat;        // LOG_SCALE*countr_zero(i & 65535), length 65536 + OPT_BLOCK_SIZE
uint32_t prime_limit = 0;            // primes[] currently covers [2, prime_limit]
constexpr bool in_presieve(uint32_t q)
{
    if (q == 3 || q == 5 || q == 7 || q == 9 || q == 11 || q == 13 || q == 17 || q == 19 || q == 23 || q == 29)
        return true;
#ifdef PRESIEVE3
    if (q == 31 || q == 37 || q == 41 || q == 43)
        return true;
#endif
    return false;
}

struct ActiveStride // per-chunk compacted strider state
{
    uint32_t q;
    uint32_t off;
    uint32_t lg;
};

void emit_prime(uint32_t p)
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
    uint16_t log2s = (uint16_t)std::llround((double)LOG_SCALE * std::log2((double)p));
    primes.emplace_back(PrimeData{(uint64_t)magic_128, inv, UINT64_MAX / p, p, log2s, shift});
    pop_p.push_back(p);
    pop_magic.push_back((uint64_t)magic_128);
    pop_shift.push_back(shift);
    pop_item.push_back((std::min<uint64_t>(p, STRIDE_MASK) << STRIDE_SHIFT) | ((uint64_t)log2s << LG_SHIFT));

    if (p > 2 && p <= OPT_BLOCK_SIZE)
    {
        // strided prime powers q <= OPT_BLOCK_SIZE (except those in the presieve patterns)
        uint64_t q = p;
        bool is_base = true;
        while (q <= OPT_BLOCK_SIZE)
        {
            if (!in_presieve((uint32_t)q))
            {
                uint8_t qshift = std::bit_width((uint32_t)q) - 1;
                unsigned __int128 qmagic = ((unsigned __int128)1 << (64 + qshift)) / q + 1;
                small_striders.push_back(Strider{(uint64_t)qmagic, (uint32_t)q, p, log2s, qshift, is_base});
            }
            is_base = false;
            q *= p;
        }
        // remaining powers q > OPT_BLOCK_SIZE handled via buckets
        while (q <= BIGPOW_CAP)
        {
            big_powers.push_back(BigPower{q, log2s});
            if (q > BIGPOW_CAP / p)
                break;
            q *= p;
        }
    }
}

void get_primes(uint32_t limit)
{
    // Odd-only sieve of Eratosthenes: bit i represents the odd number 2*i+3.
    const uint64_t n_odd = limit >= 3 ? (uint64_t)(limit - 3) / 2 + 1 : 0;
    std::vector<bool> comp(n_odd, false);
    for (uint64_t i = 0; i < n_odd; ++i)
    {
        uint64_t p = 2 * i + 3;
        if (p * p > limit)
            break;
        if (comp[i])
            continue;
        for (uint64_t j = (p * p - 3) / 2; j < n_odd; j += p)
            comp[j] = true;
    }

    if (limit >= 2)
        emit_prime(2);
    for (uint64_t i = 0; i < n_odd; ++i)
        if (!comp[i])
            emit_prime((uint32_t)(2 * i + 3));
    prime_limit = limit;

    std::sort(big_powers.begin(), big_powers.end(),
              [](const BigPower &a, const BigPower &b) { return a.q < b.q; });

    // Build the presieve patterns and the trailing-zero init table.
    {
        const size_t plen = PRESIEVE_PERIOD + OPT_BLOCK_SIZE;
        presieve_pat.assign(plen, 0);
        const uint32_t qs[6] = {3, 9, 5, 7, 11, 13};
        const uint32_t ps[6] = {3, 3, 5, 7, 11, 13};
        for (int t = 0; t < 6; ++t)
        {
            uint16_t lg = (uint16_t)std::llround((double)LOG_SCALE * std::log2((double)ps[t]));
            for (size_t mm = 0; mm < plen; mm += qs[t])
                presieve_pat[mm] += lg;
        }
        const size_t p2len = PRESIEVE2_PERIOD + OPT_BLOCK_SIZE;
        presieve2_pat.assign(p2len, 0);
        const uint32_t qs2[4] = {17, 19, 23, 29};
        for (int t = 0; t < 4; ++t)
        {
            uint16_t lg = (uint16_t)std::llround((double)LOG_SCALE * std::log2((double)qs2[t]));
            for (size_t mm = 0; mm < p2len; mm += qs2[t])
                presieve2_pat[mm] += lg;
        }
#ifdef PRESIEVE3
        const size_t p3len = PRESIEVE3_PERIOD + OPT_BLOCK_SIZE;
        presieve3_pat.assign(p3len, 0);
        const uint32_t qs3[4] = {31, 37, 41, 43};
        for (int t = 0; t < 4; ++t)
        {
            uint16_t lg = (uint16_t)std::llround((double)LOG_SCALE * std::log2((double)qs3[t]));
            for (size_t mm = 0; mm < p3len; mm += qs3[t])
                presieve3_pat[mm] += lg;
        }
#endif
        const size_t tlen = 65536 + OPT_BLOCK_SIZE;
        tz_pat.assign(tlen, 0);
        for (size_t ii = 1; ii < tlen; ++ii)
        {
            uint32_t r = (uint32_t)(ii & 65535);
            tz_pat[ii] = r ? (acc_t)(std::countr_zero(r) * LOG_SCALE) : 0; // 0 -> fixed up at runtime
        }
    }
}

// Append primes in (prime_limit, new_limit] via a segmented odd-only sieve.
// Only called between solves (single-threaded); new primes are > OPT_BLOCK_SIZE,
// so striders / big powers / patterns are already complete.
void extend_primes(uint64_t new_limit_u)
{
    uint32_t new_limit = (uint32_t)std::min<uint64_t>(new_limit_u, (uint64_t)MAXP);
    if (new_limit <= prime_limit)
        return;
    const uint64_t SEG = 1 << 21;
    std::vector<bool> comp;
    for (uint64_t lo = prime_limit + 1; lo <= new_limit; lo += SEG)
    {
        uint64_t hi = std::min<uint64_t>(lo + SEG - 1, new_limit);
        uint64_t first = lo | 1; // first odd in segment
        if (first > hi)
            continue;
        uint64_t cnt = (hi - first) / 2 + 1;
        comp.assign(cnt, false);
        for (const auto &pd : primes)
        {
            uint64_t p = pd.p;
            if (p == 2)
                continue;
            if (p * p > hi)
                break;
            uint64_t m = std::max(p * p, ((lo + p - 1) / p) * p);
            if ((m & 1) == 0)
                m += p; // odd multiples only
            for (; m <= hi; m += 2 * p)
                comp[(m - first) / 2] = true;
        }
        for (uint64_t i = 0; i < cnt; ++i)
            if (!comp[i])
                emit_prime((uint32_t)(first + 2 * i));
    }
    prime_limit = new_limit;
}

// Exact #389 oracle: verifies nu_p((2N-m)!) + nu_p(m!) >= 2*nu_p(N!) for
// every prime, which is the full divisibility condition. Coverage: p = 2 via
// popcounts (Legendre), all p <= m wholesale, then every prime found by
// trial-dividing the run members N-m..N. That set is complete because for
// p > m a carry in m + (N-m) base p can only initiate at digit 0, which
// requires p to divide a run member - so any other prime satisfies the
// condition automatically.
bool exact_check389(uint64_t N, uint64_t m)
{
    uint32_t p_2Nm = std::popcount(2 * N - m);
    uint32_t p_m = std::popcount(m);
    uint32_t p_N = std::popcount(N);
    if (p_2Nm + p_m > 2 * p_N) [[unlikely]]
        return false;

    for (const auto &pd : primes)
    {
        uint32_t p = pd.p;
        if (p == 2)
            continue;
        if (p > m)
            break;

        uint64_t nu_pos = 0;
        uint64_t t1 = 2 * N - m;
        uint64_t magic = pd.magic;
        uint8_t shift = pd.shift;

        while (t1 >= p)
        {
            t1 = (uint64_t)((((unsigned __int128)t1 * magic) >> 64) >> shift);
            nu_pos += t1;
        }

        uint64_t t2 = m;
        while (t2 >= p)
        {
            t2 = (uint64_t)((((unsigned __int128)t2 * magic) >> 64) >> shift);
            nu_pos += t2;
        }

        uint64_t nu_neg = 0;
        uint64_t t3 = N;
        while (t3 >= p)
        {
            t3 = (uint64_t)((((unsigned __int128)t3 * magic) >> 64) >> shift);
            nu_neg += t3;
        }

        if (nu_pos < 2 * nu_neg)
            return false;
    }

    for (uint64_t i = 0; i <= m; ++i)
    {
        uint64_t temp = N - i;
        temp >>= std::countr_zero(temp);
        for (const auto &pd : primes)
        {
            uint32_t p = pd.p;
            if (p == 2)
                continue;

            if (p <= m)
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
                if (temp > m)
                {
                    uint64_t p_val = temp;
                    uint64_t nu_pos = 0;
                    uint64_t t1 = 2 * N - m;
                    while (t1 >= p_val)
                    {
                        t1 /= p_val;
                        nu_pos += t1;
                    }

                    uint64_t nu_neg = 0;
                    uint64_t t3 = N;
                    while (t3 >= p_val)
                    {
                        t3 /= p_val;
                        nu_neg += t3;
                    }

                    if (nu_pos < 2 * nu_neg)
                        return false;
                }
                break;
            }
            uint64_t q = temp * pd.inv_p;
            if (q <= pd.limit)
            {
                uint64_t nu_pos = 0;
                uint64_t t1 = 2 * N - m;
                uint64_t magic = pd.magic;
                uint8_t shift = pd.shift;
                while (t1 >= p)
                {
                    t1 = (uint64_t)((((unsigned __int128)t1 * magic) >> 64) >> shift);
                    nu_pos += t1;
                }

                uint64_t nu_neg = 0;
                uint64_t t3 = N;
                while (t3 >= p)
                {
                    t3 = (uint64_t)((((unsigned __int128)t3 * magic) >> 64) >> shift);
                    nu_neg += t3;
                }

                if (nu_pos < 2 * nu_neg)
                    return false;

                while (q <= pd.limit)
                {
                    temp = q;
                    q = temp * pd.inv_p;
                }
            }
        }
    }
    return true;
}

// dual minima: one scan serves both n's of a K-pair (m2 = 0 disables slot 2)
uint64_t g_m1 = 0, g_m2 = 0;
std::atomic<uint64_t> g_min1{UINT64_MAX}, g_min2{UINT64_MAX};


bool g_worker_mode = false;
uint64_t g_end_L = UINT64_MAX;

template <uint64_t K>
uint64_t solve_impl(uint64_t start_L, uint64_t CHUNK_SIZE)
{

    std::atomic<uint64_t> current_chunk{0};
    std::vector<AlignedAtomic> active_chunks(NUM_THREADS);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto worker = [&](uint32_t thread_index) {
        std::vector<acc_t> acc(OPT_BLOCK_SIZE + 64);
        std::vector<ActiveStride> act;
        act.reserve(small_striders.size());

        uint32_t MAX_CHUNK_W = (uint32_t)(CHUNK_SIZE + K);
        uint32_t TOTAL_BLOCKS = (MAX_CHUNK_W + OPT_BLOCK_SIZE - 1) >> BLOCK_SHIFT;

        std::vector<FastBucket> block_buckets(TOTAL_BLOCKS);
        for (uint32_t b = 0; b < TOTAL_BLOCKS; ++b)
            block_buckets[b].reserve(BUCKET_RESERVE);

        while (true)
        {
            uint64_t chunk_id = current_chunk.fetch_add(1, std::memory_order_relaxed);
            active_chunks[thread_index].val.store(chunk_id, std::memory_order_release);

#ifdef BENCHMARK
            if (chunk_id >= BENCH_CANDS / CHUNK_SIZE)
            {
                active_chunks[thread_index].val.store(UINT64_MAX, std::memory_order_release);
                break;
            }
#endif
            uint64_t L_chunk = start_L + chunk_id * CHUNK_SIZE;
            uint64_t R_chunk = L_chunk + CHUNK_SIZE + K - 1;
            uint32_t CHUNK_W = (uint32_t)(R_chunk - L_chunk + 1);

            // Overflow guard. acc can reach at most LOG_SCALE*log2(x) plus
            // the rounding overshoot (0.5 per odd prime-power term); only the
            // overshoot belongs in this bound - LOG_SLACK is a threshold
            // allowance, not accumulator mass. Ceilings: scale-4 ~2^59 =
            // 5.8e17, scale-3 ~2^77 = 1.5e23, ACC16 capped by the bucket
            // item's 10-bit LOG field.
            if ((double)LOG_SCALE * std::log2((double)R_chunk) + ACC_OVERSHOOT > ACC_MAX ||
                R_chunk > RANGE_CAP) [[unlikely]]
            {
                std::cerr << "\nRange exceeds " ACC_CEIL_MSG ".\n";
                std::exit(1);
            }
            uint64_t stop1 = g_min1.load(std::memory_order_relaxed);
            uint64_t stop2 = g_m2 ? g_min2.load(std::memory_order_relaxed) : 0;
            if (L_chunk >= g_end_L || L_chunk > std::max(stop1, stop2))
            {
                active_chunks[thread_index].val.store(UINT64_MAX, std::memory_order_release);
                break;
            }

            uint64_t max_p = std::max<uint64_t>(std::sqrt(2 * R_chunk) + 1, 2 * K);
            if (max_p > prime_limit) [[unlikely]]
            {
                std::cerr << "\nPrime table exhausted (need " << max_p << ", have " << prime_limit
                          << "). Restart to resume from checkpoint with a larger table.\n";
                std::exit(1);
            }
            auto it = std::upper_bound(primes.begin(), primes.end(), max_p,
                                       [](uint64_t val, const PrimeData &pd) { return val < pd.p; });
            size_t chunk_total_primes = std::distance(primes.begin(), it);

            auto it_large = std::upper_bound(primes.begin(), primes.begin() + chunk_total_primes, OPT_BLOCK_SIZE,
                                             [](uint32_t val, const PrimeData &pd) { return val < pd.p; });
            size_t first_large_prime_idx = std::distance(primes.begin(), it_large);

            uint32_t p_thresh = std::max<uint32_t>(2 * K, (uint32_t)std::cbrt(2.0 * R_chunk) + 2);

            // Phase 1: offsets for the small strided prime powers.
            // - p > max_p is never sieved: such a factor is provably fatal and its
            //   missing mass keeps acc below T; p in (block, max_p] goes to the buckets
            // - base strides of p > p_thresh get the chunk-wide Kummer skip
            const uint32_t small_cap = (uint32_t)std::min<uint64_t>(max_p, OPT_BLOCK_SIZE);
            act.clear();
            for (const Strider &st : small_striders)
            {
                if (st.p > small_cap)
                    continue;
                if (st.is_base && st.p > p_thresh)
                {
                    uint64_t p2 = (uint64_t)st.p * st.p;
                    uint64_t d = (uint64_t)((((unsigned __int128)L_chunk * st.magic) >> 64) >> st.shift);
                    uint64_t c = (uint64_t)((((unsigned __int128)d * st.magic) >> 64) >> st.shift);
                    if (2 * R_chunk < (2 * c + 1) * p2)
                        continue;
                }
                uint64_t num = L_chunk + st.q - 1;
                uint64_t start_c = (uint64_t)((((unsigned __int128)num * st.magic) >> 64) >> st.shift);
                act.push_back(ActiveStride{st.q, (uint32_t)(start_c * st.q - L_chunk), st.log2s});
            }

            auto it_thresh =
                std::upper_bound(primes.begin() + first_large_prime_idx, primes.begin() + chunk_total_primes, p_thresh,
                                 [](uint32_t val, const PrimeData &pd) { return val < pd.p; });
            size_t thresh_idx = std::distance(primes.begin(), it_thresh);

            // Phase 2a: Medium Primes (no Kummer skip possible) - first hit only;
            // the drain loop reinserts subsequent hits.
            for (size_t idx = first_large_prime_idx; idx < thresh_idx; ++idx)
            {
                uint32_t p = pop_p[idx];
                uint64_t num = L_chunk + p - 1;
                uint64_t start_c = (uint64_t)((((unsigned __int128)num * pop_magic[idx]) >> 64) >> pop_shift[idx]);
                uint32_t hit = (uint32_t)(start_c * p - L_chunk);

                if (hit < CHUNK_W)
                    block_buckets[hit >> BLOCK_SHIFT].push_back(pop_item[idx] | hit);
            }

            // Phase 2b: Giant Primes (chunk-wide Kummer skip)
            // The old `L_chunk / p2` was a hardware 64-bit division per giant prime
            // per chunk - millions of them at k >= 14. floor(L/p^2) =
            // floor(floor(L/p)/p), so two magic multiplies replace it.
            for (size_t idx = thresh_idx; idx < chunk_total_primes; ++idx)
            {
                uint32_t p = pop_p[idx];
                const uint64_t magic = pop_magic[idx];
                const uint8_t shift = pop_shift[idx];
                uint64_t p2 = (uint64_t)p * p;
                uint64_t d = (uint64_t)((((unsigned __int128)L_chunk * magic) >> 64) >> shift); // L / p
                uint64_t c = (uint64_t)((((unsigned __int128)d * magic) >> 64) >> shift);       // L / p^2
                if (2 * R_chunk < (2 * c + 1) * p2)
                    continue;

                uint64_t num = L_chunk + p - 1;
                uint64_t start_c = (uint64_t)((((unsigned __int128)num * magic) >> 64) >> shift);
                uint32_t hit = (uint32_t)(start_c * p - L_chunk);

                if (hit < CHUNK_W)
                    block_buckets[hit >> BLOCK_SHIFT].push_back(pop_item[idx] | hit);
            }

            // Phase 2c: big prime powers q = p^e > OPT_BLOCK_SIZE (p <= OPT_BLOCK_SIZE):
            // one extra LOG(p) per multiple of q. Few thousand plain divisions per
            // chunk total - negligible, and it makes power stripping in the hot
            // loop unnecessary.
            for (const auto &bp : big_powers)
            {
                if (bp.q > R_chunk)
                    break;
                uint64_t m = ((L_chunk + bp.q - 1) / bp.q) * bp.q;
                uint64_t base_item = (std::min<uint64_t>(bp.q, STRIDE_MASK) << STRIDE_SHIFT) |
                                     ((uint64_t)bp.log2s << LG_SHIFT);
                if (bp.q <= STRIDE_MASK)
                {
                    // exact stride: first multiple only, rest via reinsertion
                    if (m <= R_chunk)
                        block_buckets[(m - L_chunk) >> BLOCK_SHIFT].push_back(base_item | (m - L_chunk));
                }
                else
                {
                    // saturated stride (drops after one add): push each multiple (at most ~2)
                    for (; m <= R_chunk; m += bp.q)
                        block_buckets[(m - L_chunk) >> BLOCK_SHIFT].push_back(base_item | (m - L_chunk));
                }
            }

            // Phase 3: The Uninterrupted Forward Sweep
            uint32_t overlap = 0, j = 0;
            for (uint32_t b = 0; b < TOTAL_BLOCKS; ++b)
            {
                uint64_t block_L = L_chunk + b * OPT_BLOCK_SIZE;
                uint64_t block_R = std::min(R_chunk, block_L + OPT_BLOCK_SIZE - 1);
                if (block_L > block_R)
                    break;

                uint32_t num_new = (uint32_t)(block_R - block_L + 1);

                // init: 32*countr_zero(x) from a phase-shifted table, fused with the
                // presieve pattern add - one contiguous, auto-vectorized u16 pass.
                acc_t *acc_ptr = acc.data() + overlap;
                const acc_t *tp = tz_pat.data() + (size_t)(block_L & 65535);
                const acc_t *pp = presieve_pat.data() + (size_t)(block_L % PRESIEVE_PERIOD);
                const acc_t *pp2 = presieve2_pat.data() + (size_t)(block_L % PRESIEVE2_PERIOD);
#ifdef PRESIEVE3
                const acc_t *pp3 = presieve3_pat.data() + (size_t)(block_L % PRESIEVE3_PERIOD);
                for (uint32_t idx_new = 0; idx_new < num_new; ++idx_new)
                    acc_ptr[idx_new] = (acc_t)(tp[idx_new] + pp[idx_new] + pp2[idx_new] + pp3[idx_new]);
#else
                for (uint32_t idx_new = 0; idx_new < num_new; ++idx_new)
                    acc_ptr[idx_new] = (acc_t)(tp[idx_new] + pp[idx_new] + pp2[idx_new]);
#endif
                // fixups: multiples of 65536 have tz >= 16 (table holds 0 there)
                for (uint64_t mm = (block_L + 65535) & ~65535ULL; mm <= block_R; mm += 65536)
                {
                    uint32_t j0 = (uint32_t)(mm - block_L);
                    acc_t v = (acc_t)((uint32_t)std::countr_zero(mm) * LOG_SCALE + pp[j0] + pp2[j0]);
#ifdef PRESIEVE3
                    v = (acc_t)(v + pp3[j0]);
#endif
                    acc_ptr[j0] = v;
                }

                // small prime powers: pure strided u16 adds over the compact list
                for (ActiveStride &as : act)
                {
                    uint32_t start_j = as.off;
                    if (start_j < num_new)
                    {
                        const uint32_t q = as.q;
                        const acc_t lg = (acc_t)as.lg;
                        uint32_t jj = start_j;
                        for (; jj < num_new; jj += q)
                            acc_ptr[jj] += lg;
                        as.off = jj - num_new;
                    }
                    else
                    {
                        as.off = start_j - num_new;
                    }
                }

                // bucketed hits: add LOG and forward-schedule the next hit.
                // Reinserts always target a strictly later bucket (stride > block),
                // so iterating this bucket's array while pushing is safe.
                uint32_t b_count = block_buckets[b].count;
                const uint64_t *b_items = block_buckets[b].data;
                for (uint32_t i = 0; i < b_count; ++i)
                {
                    uint64_t item = b_items[i];
                    uint32_t off = (uint32_t)(item & OFF_MASK);
                    acc_ptr[off & BLOCK_MASK] += (acc_t)(item >> LG_SHIFT);
                    uint64_t next = off + ((item >> STRIDE_SHIFT) & STRIDE_MASK);
                    if (next < CHUNK_W) [[likely]]
                        block_buckets[next >> BLOCK_SHIFT].push_back((item & ~OFF_MASK) | next);
                }
                block_buckets[b].clear();

                // threshold for the whole window, based on its smallest value
                uint64_t window_base = block_L - overlap;
                double t_f = (double)LOG_SCALE * std::log2((double)window_base) - LOG_SLACK;
                acc_t T = t_f <= 0.0 ? 0 : (acc_t)t_f;

                uint32_t W_search = overlap + num_new;
                while (j + K < W_search)
                {
                    if (acc[j + K] < T) [[likely]]
                    {
                        j += K + 1;
                    }
                    else
                    {
                        int i = K - 1;
                        while (i >= 0 && acc[j + i] >= T)
                            --i;
                        if (i < 0)
                        {
                            uint64_t n = block_L - overlap + j + K;
                            for (int t2 = 0; t2 < 2; ++t2)
                            {
                                uint64_t mm = t2 ? g_m2 : g_m1;
                                std::atomic<uint64_t> &gm = t2 ? g_min2 : g_min1;
                                if (!mm)
                                    continue;
                                if (n > mm && n < gm.load(std::memory_order_relaxed))
                                {
                                    if (exact_check389(n, mm))
                                    {
                                        uint64_t current = gm.load(std::memory_order_relaxed);
                                        while (n < current &&
                                               !gm.compare_exchange_weak(current, n, std::memory_order_relaxed))
                                        {
                                        }
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
                        acc[i] = acc[W_search - K + i];
                    j -= (W_search - K);
                    overlap = K;
                }
                else
                {
                    overlap = W_search;
                }
            }

            static std::atomic<bool> is_writing{false};
            if (!g_worker_mode && chunk_id > NUM_THREADS && chunk_id % 32 == 0)
            {
                bool expected = false;
                if (is_writing.compare_exchange_strong(expected, true))
                {
                    uint64_t min_active = chunk_id;
                    for (uint32_t t = 0; t < NUM_THREADS; ++t)
                        min_active = std::min(min_active, active_chunks[t].val.load(std::memory_order_acquire));

                    uint64_t safe_L = start_L + min_active * CHUNK_SIZE;
                    std::ofstream fout("checkpoint-389.tmp");
                    if (fout)
                    {
                        fout << K << " " << safe_L << " " << g_min1.load() << " " << g_min2.load() << "\n";
                        fout.close();
                        std::error_code ec;
                        std::filesystem::rename("checkpoint-389.tmp", "checkpoint-389.txt", ec);

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

    return g_min1.load();
}

uint64_t solve(uint64_t k, uint64_t start_L)
{
    // Grow the prime table on demand: exact bound in benchmark mode; in search
    // mode 16x sqrt headroom (covers an answer up to 256x the start; the record
    // ratio so far is 61x at k=11). MAXP is now just the hard cap.
#ifdef BENCHMARK
    uint64_t need = (uint64_t)std::sqrt(2.0 * (double)(start_L + BENCH_CANDS + 64)) + OPT_BLOCK_SIZE;
#else
    uint64_t need = (uint64_t)(16.0 * std::sqrt(2.0 * (double)start_L)) + OPT_BLOCK_SIZE;
    if (g_worker_mode) // bounded range: exact bound, no 16x headroom needed
        need = (uint64_t)std::sqrt(2.0 * (double)(g_end_L + CHUNK_MAX + 64)) + OPT_BLOCK_SIZE;
#endif
    extend_primes(std::max<uint64_t>(need, 1'000'000));

    // Runtime chunk size: small while few primes per chunk, big once the
    // per-prime setup dominates (measured crossover ~ k=14).
#ifdef BENCHMARK
    double est_p = std::sqrt(2.0 * (double)(start_L + BENCH_CANDS));
#else
    double est_p = std::sqrt(2.0 * (double)start_L);
#endif
    uint64_t chunk_size = (est_p >= (double)CHUNK_SWITCH_P) ? CHUNK_MAX : CHUNK_MIN;
    if (const char *e = std::getenv("CHUNK_LOG"))
    {
        int v = std::atoi(e);
        if (v == 25) chunk_size = CHUNK_MIN;
        else if (v == 26) chunk_size = CHUNK_MAX;
    }

    switch (k)
    {
    case 0:
        return solve_impl<0>(start_L, chunk_size);
    case 1:
        return solve_impl<1>(start_L, chunk_size);
    case 2:
        return solve_impl<2>(start_L, chunk_size);
    case 3:
        return solve_impl<3>(start_L, chunk_size);
    case 4:
        return solve_impl<4>(start_L, chunk_size);
    case 5:
        return solve_impl<5>(start_L, chunk_size);
    case 6:
        return solve_impl<6>(start_L, chunk_size);
    case 7:
        return solve_impl<7>(start_L, chunk_size);
    case 8:
        return solve_impl<8>(start_L, chunk_size);
    case 9:
        return solve_impl<9>(start_L, chunk_size);
    case 10:
        return solve_impl<10>(start_L, chunk_size);
    case 11:
        return solve_impl<11>(start_L, chunk_size);
    case 12:
        return solve_impl<12>(start_L, chunk_size);
    case 13:
        return solve_impl<13>(start_L, chunk_size);
    case 14:
        return solve_impl<14>(start_L, chunk_size);
    case 15:
        return solve_impl<15>(start_L, chunk_size);
    case 16:
        return solve_impl<16>(start_L, chunk_size);
    case 17:
        return solve_impl<17>(start_L, chunk_size);
    case 18:
        return solve_impl<18>(start_L, chunk_size);
    case 19:
        return solve_impl<19>(start_L, chunk_size);
    case 20:
        return solve_impl<20>(start_L, chunk_size);
    default:
        return 0;
    }
}

int main(int argc, char **argv)
{
    // Worker mode for the multi-host coordinator: ./binary <n> <start_L> <end_L>
    // (ranges multiples of 2^26; prints "RESULT <n> <min-N-or-0>"; k = N-(n-1))
    if (argc == 4)
    {
        g_worker_mode = true;
        uint64_t n_prob = std::strtoull(argv[1], nullptr, 10);
        uint64_t s = std::strtoull(argv[2], nullptr, 10);
        g_end_L = std::strtoull(argv[3], nullptr, 10);
        uint64_t m = n_prob - 1;
        uint64_t K = m >= 1 ? (m - 1) / 2 : 0;
        g_m1 = m;
        g_m2 = 0;
        g_min1.store(UINT64_MAX);
        g_min2.store(UINT64_MAX);
        detect_threads();
        get_primes(1'000'000);
        uint64_t ans = solve(K, s);
        std::cout << "RESULT " << n_prob << " " << (ans == UINT64_MAX ? 0 : ans) << std::endl;
        return 0;
    }

    detect_threads();
    std::cout << "Generating base prime table (grows on demand, cap " << (uint64_t)MAXP << ")...\n";
    auto start_primes = std::chrono::high_resolution_clock::now();
    get_primes(1'000'000);
    std::cout << "Primes generated in "
              << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_primes).count()
              << " seconds. (" << small_striders.size() << " small strides, " << big_powers.size()
              << " big powers)\n\n";

#ifdef BENCHMARK
    // (K, m1, m2, start) - starts mirror the #396 harness prime densities
    struct BT { uint64_t K, m1, m2, s; };
    std::vector<BT> tests{
        {11, 23, 24, 1'000'000'000'000ULL},
        {12, 25, 26, 5'000'000'000'000ULL},
        {13, 27, 28, 18'000'000'000'000ULL},
    };
    for (const auto &t : tests)
    {
        std::cout << "--- BENCHMARK MODE ---\n";
        std::cout << "Testing K=" << t.K << " (n=" << t.m1 + 1 << "," << t.m2 + 1
                  << ") | Workload: " << BENCH_CANDS / 1000000 << " M candidates\n\n";
        for (int run = 1; run <= RUNS; ++run)
        {
            g_m1 = t.m1; g_m2 = t.m2;
            g_min1.store(UINT64_MAX); g_min2.store(UINT64_MAX);
            auto start = std::chrono::high_resolution_clock::now();
            (void)solve(t.K, t.s);
            double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
            std::cout << "Run " << run << " | Time: " << std::fixed << std::setprecision(4) << std::setw(8)
                      << seconds << " s | Speed: " << std::fixed << std::setprecision(2) << std::setw(8)
                      << (BENCH_CANDS / seconds / 1e6) << " M candidates/s\n";
        }
    }
#else
    uint64_t nmax = 30;
    if (const char *e = std::getenv("NMAX_RT"))
        nmax = std::strtoull(e, nullptr, 10);

    uint64_t start_K = 0;
    uint64_t start_L = 1;
    uint64_t seed1 = UINT64_MAX, seed2 = UINT64_MAX;
    bool resumed = false;
    if (std::filesystem::exists("checkpoint-389.txt"))
    {
        std::ifstream fin("checkpoint-389.txt");
        if (fin >> start_K >> start_L >> seed1 >> seed2)
        {
            std::cout << "--> Resuming from checkpoint: K = " << start_K << ", L = " << start_L << "\n\n";
            resumed = true;
        }
        else { start_K = 0; start_L = 1; seed1 = seed2 = UINT64_MAX; }
    }

    auto emit = [&](uint64_t n_prob, uint64_t m, uint64_t N, double seconds, double cands) {
        std::ostringstream oss;
        oss << "n = " << std::setw(2) << n_prob << " | min k = " << std::setw(15) << (N - m)
            << " | (N = " << N << ") | Time: " << std::fixed << std::setprecision(4) << std::setw(12)
            << seconds << " s | Speed: " << std::fixed << std::setprecision(2) << std::setw(8)
            << (cands / seconds / 1e6) << " M candidates/s\n";
        std::string s = oss.str();
        std::cout << s;
        std::ofstream rf("results-389.txt", std::ios::app);
        if (rf) rf << s;
    };

    if (!resumed && nmax >= 1)
        std::cout << "n =  1 | min k =               1 | (trivial)\n";

    for (uint64_t K = start_K;; ++K)
    {
        uint64_t n1 = 2 * K + 2, n2 = 2 * K + 3;
        if (n1 > nmax)
            break;
        g_m1 = n1 - 1;
        g_m2 = (n2 <= nmax) ? n2 - 1 : 0;
        g_min1.store(K == start_K ? seed1 : UINT64_MAX);
        g_min2.store(K == start_K ? seed2 : UINT64_MAX);
        auto start = std::chrono::high_resolution_clock::now();
        uint64_t N1 = solve(K, K == start_K ? start_L : 1);
        double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        uint64_t N2 = g_min2.load();
        double cands = (double)std::max<uint64_t>(N1, g_m2 ? N2 : 0);
        emit(n1, g_m1, N1, seconds, cands);
        if (g_m2)
            emit(n2, g_m2, N2, seconds, cands);
        std::ofstream fout("checkpoint-389.tmp");
        if (fout)
        {
            fout << (K + 1) << " " << 1 << " " << UINT64_MAX << " " << UINT64_MAX << "\n";
            fout.close();
            std::error_code ec;
            std::filesystem::rename("checkpoint-389.tmp", "checkpoint-389.txt", ec);
        }
    }
    std::error_code ec;
    std::filesystem::remove("checkpoint-389.txt", ec);
#endif
    return 0;
}