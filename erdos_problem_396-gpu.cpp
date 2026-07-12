// Erdos #396 GPU (OpenCL) host for the fixed-point log sieve.
//
// Kernels live in erdos_396_kernels.cl (dual-target: also compiled as C++
// under -DCPU_SIM so the exact kernel logic validates against known answers
// on any machine). The host dlopens libOpenCL.so.1 at runtime through a
// hand-declared loader, so no OpenCL headers or link-time dependency are
// needed to build it.
//
// Per chunk: k_zero clears the u8 global accumulator (4 bytes per u32 word,
// bucket-prime mass only); k_pprep computes per-prime first hits + chunk-wide
// Kummer skips; k_pop_slice runs once per slice (GPU_SLICE_LOG, default 2^23
// candidates = 8 MB) so bucket-hit atomics stay cache-resident; k_bighits
// adds host-enumerated big prime-power hits; k_fused does init + small-prime
// strides + threshold per 16384-candidate segment entirely in SLM; k_runs
// scans mask bytes for runs of K+1 and appends survivor positions. Survivors
// are verified on the CPU with the exact oracle, so reported minima are
// exact.
//
// Chunks run through a depth-2 software pipeline over two command queues:
// while the in-order compute queue executes chunk i, the transfer queue
// uploads chunk i+1's per-chunk tables (into the alternate buffer set) and
// reads back chunk i-1's survivors, and the host does its prep and oracle
// work in the same window. GPU_PIPELINE=0 forces the serial path;
// ERDOS_PROF=1 prints per-chunk per-kernel timings (implies serial so the
// per-stage numbers stay meaningful).
//
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

constexpr int LOG_SCALE = 4;  // fixed-point units per bit of log2
constexpr int LOG_SLACK = 22; // > max llround deficit: <= 37 odd prime-power
                              // terms below 2^59 (3^37 > 2^58) at 0.5/term = 19, +3
// Accumulator bytes peak at LOG_SCALE*log2(x) + rounding overshoot (<= 19);
// keep that under 256:
constexpr uint64_t RANGE_CEIL = 1ULL << 59; // ~5.8e17
static bool g_pipeline = true;   // overlap transfers/host with compute; env GPU_PIPELINE=0 disables
static double g_bin_head = 2.0;  // bin capacity = head * mean records/segment; env GPU_BIN_HEADROOM
static bool g_worker_mode = false;
static uint64_t g_end_L = UINT64_MAX;
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
std::vector<uint16_t> pop_lg;   // LOG(p) for the GPU kernels
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
std::vector<uint8_t> presieve3_pat;
#endif
std::vector<uint8_t> presieve_pat;  // length PERIOD + OPT_BLOCK_SIZE
std::vector<uint8_t> presieve2_pat; // length PERIOD2 + OPT_BLOCK_SIZE
std::vector<uint8_t> tz_pat;        // LOG_SCALE*countr_zero(i & 65535), length 65536 + OPT_BLOCK_SIZE
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
    pop_lg.push_back(log2s);

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
            uint8_t lg = (uint8_t)std::llround((double)LOG_SCALE * std::log2((double)ps[t]));
            for (size_t mm = 0; mm < plen; mm += qs[t])
                presieve_pat[mm] += lg;
        }
        const size_t p2len = PRESIEVE2_PERIOD + OPT_BLOCK_SIZE;
        presieve2_pat.assign(p2len, 0);
        const uint32_t qs2[4] = {17, 19, 23, 29};
        for (int t = 0; t < 4; ++t)
        {
            uint8_t lg = (uint8_t)std::llround((double)LOG_SCALE * std::log2((double)qs2[t]));
            for (size_t mm = 0; mm < p2len; mm += qs2[t])
                presieve2_pat[mm] += lg;
        }
#ifdef PRESIEVE3
        const size_t p3len = PRESIEVE3_PERIOD + OPT_BLOCK_SIZE;
        presieve3_pat.assign(p3len, 0);
        const uint32_t qs3[4] = {31, 37, 41, 43};
        for (int t = 0; t < 4; ++t)
        {
            uint8_t lg = (uint8_t)std::llround((double)LOG_SCALE * std::log2((double)qs3[t]));
            for (size_t mm = 0; mm < p3len; mm += qs3[t])
                presieve3_pat[mm] += lg;
        }
#endif
        const size_t tlen = 65536 + OPT_BLOCK_SIZE;
        tz_pat.assign(tlen, 0);
        for (size_t ii = 1; ii < tlen; ++ii)
        {
            uint32_t r = (uint32_t)(ii & 65535);
            tz_pat[ii] = r ? (uint8_t)(std::countr_zero(r) * LOG_SCALE) : 0; // 0 -> fixed up at runtime
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

template <uint64_t K>
bool exact_check(uint64_t n)
{
    uint32_t nu_2_prod = std::popcount(n - K - 1) - std::popcount(n) + K + 1;
    if ((uint32_t)std::popcount(n) < nu_2_prod) [[unlikely]]
        return false;

    for (const auto &pd : primes)
    {
        uint32_t p = pd.p;
        if (p == 2)
            continue;
        if (p > 2 * K)
            break;

        uint64_t nu_prod = 0;
        uint64_t nu_comb = 0;
        uint64_t power = p;
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


// ===========================================================================
// GPU/simulation backend (v2: fused SLM segment kernel)
// ===========================================================================
#ifndef GPU_CHUNK
#define GPU_CHUNK (1ULL << 27)
#endif
#ifndef SURV_CAP
#define SURV_CAP (1u << 20)
#endif
#define FUSED_LSZ 256

#ifdef CPU_SIM
#include "erdos_396_kernels.cl"
#else
#include <dlfcn.h>
typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_program *cl_program;
typedef struct _cl_kernel *cl_kernel;
typedef struct _cl_event *cl_event;
typedef struct _cl_mem *cl_mem;
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_bitfield;
typedef cl_bitfield cl_device_type, cl_mem_flags, cl_queue_properties;
typedef cl_uint cl_device_info, cl_program_build_info, cl_bool;
#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU (1UL << 2)
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_TRUE 1

static struct CL
{
    cl_int (*GetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
    cl_int (*GetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
    cl_int (*GetDeviceInfo)(cl_device_id, cl_device_info, size_t, void *, size_t *);
    cl_context (*CreateContext)(const intptr_t *, cl_uint, const cl_device_id *, void *, void *, cl_int *);
    cl_command_queue (*CreateCommandQueueWithProperties)(cl_context, cl_device_id, const cl_queue_properties *, cl_int *);
    cl_program (*CreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
    cl_int (*BuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *, void *, void *);
    cl_int (*GetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
    cl_kernel (*CreateKernel)(cl_program, const char *, cl_int *);
    cl_mem (*CreateBuffer)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
    cl_int (*SetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
    cl_int (*EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const void *, void *);
    cl_int (*EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const void *, void *);
    cl_int (*EnqueueFillBuffer)(cl_command_queue, cl_mem, const void *, size_t, size_t, size_t,
                                cl_uint, const void *, void *); // optional (OpenCL 1.2)
    cl_int (*EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const void *, void *);
    cl_int (*Finish)(cl_command_queue);
    cl_int (*Flush)(cl_command_queue);
    cl_int (*ReleaseEvent)(cl_event);
    cl_int (*ReleaseMemObject)(cl_mem);
} CL;

static void load_cl()
{
    const char *lib = std::getenv("OCL_LIB");
    void *h = dlopen(lib ? lib : "libOpenCL.so.1", RTLD_NOW);
    if (!h)
        h = dlopen("libOpenCL.so", RTLD_NOW);
    if (!h)
    {
        std::cerr << "Cannot dlopen libOpenCL.so.1 - install an OpenCL ICD loader "
                     "(use the flake / nix develop), or set OCL_LIB=/path\n";
        std::exit(1);
    }
    auto S = [&](const char *n) {
        void *f = dlsym(h, n);
        if (!f) { std::cerr << "Missing OpenCL symbol " << n << "\n"; std::exit(1); }
        return f;
    };
    CL.GetPlatformIDs = (decltype(CL.GetPlatformIDs))S("clGetPlatformIDs");
    CL.GetDeviceIDs = (decltype(CL.GetDeviceIDs))S("clGetDeviceIDs");
    CL.GetDeviceInfo = (decltype(CL.GetDeviceInfo))S("clGetDeviceInfo");
    CL.CreateContext = (decltype(CL.CreateContext))S("clCreateContext");
    CL.CreateCommandQueueWithProperties = (decltype(CL.CreateCommandQueueWithProperties))S("clCreateCommandQueueWithProperties");
    CL.CreateProgramWithSource = (decltype(CL.CreateProgramWithSource))S("clCreateProgramWithSource");
    CL.BuildProgram = (decltype(CL.BuildProgram))S("clBuildProgram");
    CL.GetProgramBuildInfo = (decltype(CL.GetProgramBuildInfo))S("clGetProgramBuildInfo");
    CL.CreateKernel = (decltype(CL.CreateKernel))S("clCreateKernel");
    CL.CreateBuffer = (decltype(CL.CreateBuffer))S("clCreateBuffer");
    CL.SetKernelArg = (decltype(CL.SetKernelArg))S("clSetKernelArg");
    CL.EnqueueNDRangeKernel = (decltype(CL.EnqueueNDRangeKernel))S("clEnqueueNDRangeKernel");
    CL.EnqueueWriteBuffer = (decltype(CL.EnqueueWriteBuffer))S("clEnqueueWriteBuffer");
    CL.EnqueueFillBuffer = (decltype(CL.EnqueueFillBuffer))dlsym(h, "clEnqueueFillBuffer"); // optional; null -> kernel fallback
    CL.EnqueueReadBuffer = (decltype(CL.EnqueueReadBuffer))S("clEnqueueReadBuffer");
    CL.Finish = (decltype(CL.Finish))S("clFinish");
    CL.Flush = (decltype(CL.Flush))S("clFlush");
    CL.ReleaseEvent = (decltype(CL.ReleaseEvent))S("clReleaseEvent");
    CL.ReleaseMemObject = (decltype(CL.ReleaseMemObject))S("clReleaseMemObject");
}
#define CLCHECK(x)                                                                                 \
    do { cl_int _e = (x); if (_e != CL_SUCCESS) { std::cerr << "OpenCL error " << _e << " at " << __FILE__ << ":" << __LINE__ << "\n"; std::exit(1); } } while (0)
#endif // CPU_SIM

struct Prof
{
    bool on = false;
    double t[8] = {};
    uint64_t chunks = 0;
    static constexpr const char *names[8] = {"host-prep", "upload", "k_zero", "k_pop",
                                             "k_bighits", "k_fused", "k_runs", "readback"};
    void report()
    {
        if (!on || !chunks) return;
        std::cout << "\n[prof] per-chunk averages over " << chunks << " chunks:\n";
        double tot = 0;
        for (int i = 0; i < 8; ++i) tot += t[i];
        for (int i = 0; i < 8; ++i)
            std::cout << "[prof]   " << std::setw(10) << names[i] << ": " << std::fixed
                      << std::setprecision(3) << std::setw(8) << (t[i] / chunks * 1e3) << " ms  ("
                      << std::setprecision(1) << std::setw(5) << (t[i] / tot * 100) << "%)\n";
        std::cout << "[prof]   total " << std::fixed << std::setprecision(3) << (tot / chunks * 1e3)
                  << " ms/chunk\n";
        for (int i = 0; i < 8; ++i) t[i] = 0;
        chunks = 0;
    }
};
static Prof g_prof;
static double now_s()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct GpuCtx
{
    std::vector<uint32_t> surv_host;
    std::vector<uint8_t> mask_hostbuf;

#ifdef CPU_SIM
    std::vector<uint32_t> accp;
    std::vector<uint32_t> bins;
    std::vector<uint32_t> bcnt;
    std::vector<uint8_t> mask;
    std::vector<uint32_t> lacc;

    void init_device() { std::cout << "[CPU_SIM] kernels run single-threaded on the host\n"; }
    void ensure_pop() {}

    uint64_t sim_L = 0; uint32_t sim_W = 0;
    std::vector<uint32_t> sim_surv; bool sim_ovf = false;
    void submit_chunk(int si, uint64_t L, uint64_t R, uint32_t W, uint32_t nseg,
                      const std::vector<uint32_t> &act_q, const std::vector<uint32_t> &act_off,
                      const std::vector<uint32_t> &act_lg, const std::vector<uint32_t> &bh_off,
                      const std::vector<uint32_t> &bh_lg, const std::vector<uint32_t> &Tseg,
                      const std::vector<uint32_t> &ph_tz, const std::vector<uint32_t> &ph1,
                      const std::vector<uint32_t> &ph2, uint32_t first_idx, uint32_t kummer_idx,
                      uint32_t total_idx, uint32_t Kk)
    {
        (void)si;
        sim_L = L; sim_W = W;
        run_chunk(L, R, W, nseg, act_q, act_off, act_lg, bh_off, bh_lg, Tseg, ph_tz, ph1, ph2,
                  first_idx, kummer_idx, total_idx, Kk, sim_surv, sim_ovf);
    }
    void finish_chunk(int si, std::vector<uint32_t> &surv, bool &overflow)
    {
        (void)si;
        surv = sim_surv;
        overflow = sim_ovf;
        if (g_prof.on)
            ++g_prof.chunks;
    }
    void run_chunk(uint64_t L, uint64_t R, uint32_t W, uint32_t nseg,
                   const std::vector<uint32_t> &act_q, const std::vector<uint32_t> &act_off,
                   const std::vector<uint32_t> &act_lg, const std::vector<uint32_t> &bh_off,
                   const std::vector<uint32_t> &bh_lg, const std::vector<uint32_t> &Tseg,
                   const std::vector<uint32_t> &ph_tz, const std::vector<uint32_t> &ph1,
                   const std::vector<uint32_t> &ph2, uint32_t first_idx, uint32_t kummer_idx,
                   uint32_t total_idx, uint32_t Kk, std::vector<uint32_t> &surv,
                   bool &overflow)
    {
        uint32_t np4 = (W + 3) / 4;
        accp.assign(np4, 0);
        mask.resize(W);
        lacc.resize(SEG_WORDS);
        auto RUN = [](uint64_t n, auto fn) {
            for (uint64_t g = 0; g < n; ++g) { g_sim_gid = (uint)g; fn(); }
        };
        uint32_t nseg2 = (W + SEG_SIZE - 1) / SEG_SIZE;
        // capacity model mirrors the GPU path (kept intentionally identical
        // so overflow-fallback coverage is the same in validation)
        uint32_t bin_cap = 64;
        if (total_idx > first_idx && nseg2)
        {
            double max_p = (double)pop_p[total_idx - 1];
            double dens = std::log(std::log(max_p)) - std::log(std::log((double)OPT_BLOCK_SIZE));
            if (dens < 0)
                dens = 0;
            bin_cap = (uint32_t)(g_bin_head * dens * (double)W / nseg2) + 256;
            bin_cap = (bin_cap + 63) & ~63u;
        }
        bins.assign((size_t)nseg2 * bin_cap, 0);
        bcnt.assign(nseg2, 0);
        if (total_idx > first_idx)
            RUN(total_idx - first_idx, [&] { k_scatter(bins.data(), bcnt.data(), accp.data(),
                                                       pop_p.data(), pop_magic.data(), pop_shift.data(),
                                                       pop_lg.data(), first_idx, kummer_idx, total_idx,
                                                       L, 2 * R, W, bin_cap); });
        if (!bh_off.empty())
            RUN(bh_off.size(),
                [&] { k_bighits(accp.data(), bh_off.data(), bh_lg.data(), (uint32_t)bh_off.size()); });
        // fused segment kernel: phases are lane-parallel, so lid=0/lsz=1 covers
        // a whole segment per call (barrier-equivalent: phases run in order)
        for (uint32_t grp = 0; grp < nseg; ++grp)
        {
            phase_init(lacc.data(), 0, 1, grp, tz_pat.data(), presieve_pat.data(),
                       presieve2_pat.data(), ph_tz.data(), ph1.data(), ph2.data(), L, W);
            phase_strides(lacc.data(), 0, 1, grp, act_q.data(), act_off.data(), act_lg.data(),
                          (uint32_t)act_q.size(), W);
            phase_drain(lacc.data(), 0, 1, grp, bins.data(), bcnt.data(), bin_cap);
            phase_mask(lacc.data(), 0, 1, grp, accp.data(), mask.data(), Tseg.data(), W);
        }
        uint32_t cnt = 0;
        surv.assign(SURV_CAP, 0);
        RUN((W + 31) / 32, [&] { k_runs(mask.data(), surv.data(), &cnt, Kk, W, SURV_CAP); });
        if (cnt >= SURV_CAP) { overflow = true; surv.clear(); }
        else { overflow = false; surv.resize(cnt); }
    }

    const uint8_t *mask_host(int si, uint32_t W) { (void)si; (void)W; return mask.data(); }
#else
    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;  // compute (in-order: cross-chunk hazards on
                                   // shared bAcc/bPoff/bBins are serialized here)
    cl_command_queue qx = nullptr; // transfers: uploads for the next chunk and
                                   // readbacks for the previous one overlap compute
    cl_device_id dev = nullptr;
    cl_program prog = nullptr;
    cl_kernel kZero, kScatter, kBighits, kFused, kRuns;
    // per-chunk buffers live in two sets so chunk i+1's uploads and chunk
    // i-1's readbacks never touch the buffers chunk i's kernels are using
    struct ChunkSet
    {
        cl_mem actQ = nullptr, actO = nullptr, actL = nullptr;
        cl_mem phTz = nullptr, ph1 = nullptr, ph2 = nullptr, tseg = nullptr;
        cl_mem bhO = nullptr, bhL = nullptr, out = nullptr, outCnt = nullptr, mask = nullptr;
        size_t capAct = 0, capSeg = 0, capBh = 0, capMask = 0;
        cl_event evC = nullptr; // completion of this chunk's last kernel
        uint64_t L = 0;         // chunk meta for the consume stage
        uint32_t W = 0, nActs = 0, nBh = 0;
    };
    ChunkSet sets[2];
    cl_mem bAcc = nullptr, bTz = nullptr, bPat1 = nullptr, bPat2 = nullptr;

    cl_mem bPopP = nullptr, bPopM = nullptr, bPopS = nullptr, bPopL = nullptr;
    cl_mem bBins = nullptr, bBcnt = nullptr;
    size_t capBins = 0, capBcnt = 0;
    uint32_t cur_bin_cap = 0;
    // capacity = headroom * mean records per segment; mean via Mertens:
    // sum(1/p, OPT_BLOCK..max_p) ~ lnln(max_p) - lnln(OPT_BLOCK)
    uint32_t bin_cap_for(uint32_t total_idx, uint32_t first_idx, uint32_t W, uint32_t nseg2)
    {
        if (total_idx <= first_idx || nseg2 == 0)
            return 64;
        double max_p = (double)pop_p[total_idx - 1];
        double dens = std::log(std::log(max_p)) - std::log(std::log((double)OPT_BLOCK_SIZE));
        if (dens < 0)
            dens = 0;
        double mean = dens * (double)W / nseg2;
        uint32_t cap = (uint32_t)(g_bin_head * mean) + 256;
        return (cap + 63) & ~63u;
    }
    size_t capAcc = 0, popCount = 0;

    void ensure(cl_mem &b, size_t &cap, size_t need)
    {
        if (need <= cap) return;
        if (b) CL.ReleaseMemObject(b);
        cl_int e;
        b = CL.CreateBuffer(ctx, 0x1 /*CL_MEM_READ_WRITE*/, need, nullptr, &e);
        CLCHECK(e);
        cap = need;
    }
    void up(cl_mem b, const void *p, size_t n)
    {
        // blocking, on the transfer queue: the DMA overlaps compute-queue work,
        // and once this returns the data is visible to later-enqueued kernels
        CLCHECK(CL.EnqueueWriteBuffer(qx, b, CL_TRUE, 0, n, p, 0, nullptr, nullptr));
    }
    template <class T> void arg(cl_kernel k, cl_uint i, const T &v)
    {
        CLCHECK(CL.SetKernelArg(k, i, sizeof(T), &v));
    }
    void launch(cl_kernel k, size_t gsz, size_t lsz = 0, cl_event *ev = nullptr)
    {
        CLCHECK(CL.EnqueueNDRangeKernel(q, k, 1, nullptr, &gsz, lsz ? &lsz : nullptr, 0, nullptr, (void *)ev));
    }
    void prof_mark(int slot, double &t0)
    {
        if (!g_prof.on) return;
        CLCHECK(CL.Finish(q));
        double t1 = now_s();
        g_prof.t[slot] += t1 - t0;
        t0 = t1;
    }

    void init_device()
    {
        load_cl();
        cl_platform_id plats[8];
        cl_uint np = 0;
        CLCHECK(CL.GetPlatformIDs(8, plats, &np));
        for (cl_uint i = 0; i < np && !dev; ++i)
        {
            cl_device_id d;
            cl_uint nd = 0;
            if (CL.GetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 1, &d, &nd) == CL_SUCCESS && nd > 0)
                dev = d;
        }
        if (!dev) { std::cerr << "No OpenCL GPU device found (check clinfo / /dev/dri access)\n"; std::exit(1); }
        char name[256] = {0};
        CL.GetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        uint64_t gmem = 0; cl_uint cu = 0;
        CL.GetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, 8, &gmem, nullptr);
        CL.GetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, 4, &cu, nullptr);
        std::cout << "GPU: " << name << " | " << (gmem >> 20) << " MiB | " << cu << " CUs\n";
        cl_int e;
        ctx = CL.CreateContext(nullptr, 1, &dev, nullptr, nullptr, &e); CLCHECK(e);
        q = CL.CreateCommandQueueWithProperties(ctx, dev, nullptr, &e); CLCHECK(e);
        qx = CL.CreateCommandQueueWithProperties(ctx, dev, nullptr, &e); CLCHECK(e);

        std::string src;
        {
            const char *kf = std::getenv("ERDOS_KERNELS");
            std::ifstream f(kf ? kf : "erdos_396_kernels.cl");
            if (!f) { std::cerr << "Cannot open erdos_396_kernels.cl (set ERDOS_KERNELS=path)\n"; std::exit(1); }
            std::ostringstream ss; ss << f.rdbuf(); src = ss.str();
        }
        const char *sp = src.c_str();
        size_t sl = src.size();
        prog = CL.CreateProgramWithSource(ctx, 1, &sp, &sl, &e); CLCHECK(e);
        if (CL.BuildProgram(prog, 1, &dev, "", nullptr, nullptr) != CL_SUCCESS)
        {
            std::vector<char> log(1 << 16);
            size_t ln = 0;
            CL.GetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), &ln);
            std::cerr << "Kernel build failed:\n" << std::string(log.data(), ln) << "\n";
            std::exit(1);
        }
        kZero = CL.CreateKernel(prog, "k_zero", &e); CLCHECK(e);
        kScatter = CL.CreateKernel(prog, "k_scatter", &e); CLCHECK(e);
        kBighits = CL.CreateKernel(prog, "k_bighits", &e); CLCHECK(e);
        kFused = CL.CreateKernel(prog, "k_fused", &e); CLCHECK(e);
        kRuns = CL.CreateKernel(prog, "k_runs", &e); CLCHECK(e);

        size_t c0 = 0;
        ensure(bTz, c0, tz_pat.size()); up(bTz, tz_pat.data(), tz_pat.size());
        c0 = 0; ensure(bPat1, c0, presieve_pat.size()); up(bPat1, presieve_pat.data(), presieve_pat.size());
        c0 = 0; ensure(bPat2, c0, presieve2_pat.size()); up(bPat2, presieve2_pat.data(), presieve2_pat.size());

    }

    void ensure_pop()
    {
        if (pop_p.size() == popCount) return;
        popCount = pop_p.size();
        size_t c;
        c = 0; if (bPopP) CL.ReleaseMemObject(bPopP), bPopP = nullptr; ensure(bPopP, c, popCount * 4); up(bPopP, pop_p.data(), popCount * 4);
        c = 0; if (bPopM) CL.ReleaseMemObject(bPopM), bPopM = nullptr; ensure(bPopM, c, popCount * 8); up(bPopM, pop_magic.data(), popCount * 8);
        c = 0; if (bPopS) CL.ReleaseMemObject(bPopS), bPopS = nullptr; ensure(bPopS, c, popCount * 1); up(bPopS, pop_shift.data(), popCount * 1);
        c = 0; if (bPopL) CL.ReleaseMemObject(bPopL), bPopL = nullptr; ensure(bPopL, c, popCount * 2); up(bPopL, pop_lg.data(), popCount * 2);

    }

    void submit_chunk(int si, uint64_t L, uint64_t R, uint32_t W, uint32_t nseg,
                      const std::vector<uint32_t> &act_q, const std::vector<uint32_t> &act_off,
                      const std::vector<uint32_t> &act_lg, const std::vector<uint32_t> &bh_off,
                      const std::vector<uint32_t> &bh_lg, const std::vector<uint32_t> &Tseg,
                      const std::vector<uint32_t> &ph_tz, const std::vector<uint32_t> &ph1,
                      const std::vector<uint32_t> &ph2, uint32_t first_idx, uint32_t kummer_idx,
                      uint32_t total_idx, uint32_t Kk)
    {
        double t0 = now_s();
        ChunkSet &cs = sets[si];
        cs.L = L; cs.W = W; cs.nActs = (uint32_t)act_q.size(); cs.nBh = (uint32_t)bh_off.size();
        uint32_t np4 = (W + 3) / 4;
        ensure(bAcc, capAcc, (size_t)np4 * 4);
        ensure(cs.mask, cs.capMask, (size_t)W);
        {
            size_t c0 = 0;
            if (!cs.out) ensure(cs.out, c0, (size_t)SURV_CAP * 4);
            c0 = 0;
            if (!cs.outCnt) ensure(cs.outCnt, c0, 4);
        }
        size_t segb = (size_t)nseg * 4;
        if (segb > cs.capSeg)
        {
            size_t c;
            if (cs.phTz) { CL.ReleaseMemObject(cs.phTz); CL.ReleaseMemObject(cs.ph1); CL.ReleaseMemObject(cs.ph2); CL.ReleaseMemObject(cs.tseg); cs.phTz = cs.ph1 = cs.ph2 = cs.tseg = nullptr; }
            c = 0; ensure(cs.phTz, c, segb); c = 0; ensure(cs.ph1, c, segb);
            c = 0; ensure(cs.ph2, c, segb); c = 0; ensure(cs.tseg, c, segb);
            cs.capSeg = segb;
        }
        up(cs.phTz, ph_tz.data(), segb); up(cs.ph1, ph1.data(), segb);
        up(cs.ph2, ph2.data(), segb); up(cs.tseg, Tseg.data(), segb);
        size_t actb = act_q.size() * 4;
        if (actb > cs.capAct)
        {
            size_t c;
            if (cs.actQ) { CL.ReleaseMemObject(cs.actQ); CL.ReleaseMemObject(cs.actO); CL.ReleaseMemObject(cs.actL); cs.actQ = cs.actO = cs.actL = nullptr; }
            c = 0; ensure(cs.actQ, c, actb ? actb : 4); c = 0; ensure(cs.actO, c, actb ? actb : 4); c = 0; ensure(cs.actL, c, actb ? actb : 4);
            cs.capAct = actb ? actb : 4;
        }
        if (actb) { up(cs.actQ, act_q.data(), actb); up(cs.actO, act_off.data(), actb); up(cs.actL, act_lg.data(), actb); }
        size_t bhb = bh_off.size() * 4;
        if (bhb > cs.capBh)
        {
            size_t c;
            if (cs.bhO) { CL.ReleaseMemObject(cs.bhO); CL.ReleaseMemObject(cs.bhL); cs.bhO = cs.bhL = nullptr; }
            c = 0; ensure(cs.bhO, c, bhb); c = 0; ensure(cs.bhL, c, bhb);
            cs.capBh = bhb;
        }
        if (bhb) { up(cs.bhO, bh_off.data(), bhb); up(cs.bhL, bh_lg.data(), bhb); }
        uint32_t zero = 0;
        up(cs.outCnt, &zero, 4);
        prof_mark(1, t0);

        if (CL.EnqueueFillBuffer)
        {
            uint32_t z = 0;
            CLCHECK(CL.EnqueueFillBuffer(q, bAcc, &z, 4, 0, (size_t)np4 * 4, 0, nullptr, nullptr));
        }
        else
        {
            arg(kZero, 0, bAcc); arg(kZero, 1, np4);
            launch(kZero, ((size_t)np4 + 255) / 256 * 256);
        }
        prof_mark(2, t0);

        uint32_t nseg2 = (W + 16383) / 16384; // one bin per 16384-candidate segment
        uint32_t bin_cap = this->bin_cap_for(total_idx, first_idx, W, nseg2);
        {
            size_t needB = (size_t)nseg2 * bin_cap * 4;
            if (needB > capBins)
            {
                if (bBins) { CL.ReleaseMemObject(bBins); bBins = nullptr; }
                size_t c = 0; ensure(bBins, c, needB); capBins = needB;
            }
            size_t needC = (size_t)nseg2 * 4;
            if (needC > capBcnt)
            {
                if (bBcnt) { CL.ReleaseMemObject(bBcnt); bBcnt = nullptr; }
                size_t c = 0; ensure(bBcnt, c, needC); capBcnt = needC;
            }
            if (CL.EnqueueFillBuffer)
            {
                uint32_t z = 0;
                CLCHECK(CL.EnqueueFillBuffer(q, bBcnt, &z, 4, 0, needC, 0, nullptr, nullptr));
            }
            else
            {
                arg(kZero, 0, bBcnt); arg(kZero, 1, nseg2);
                launch(kZero, ((size_t)nseg2 + 255) / 256 * 256);
            }
        }
        if (total_idx > first_idx)
        {
            uint64_t R2 = 2 * R;
            arg(kScatter, 0, bBins); arg(kScatter, 1, bBcnt); arg(kScatter, 2, bAcc);
            arg(kScatter, 3, bPopP); arg(kScatter, 4, bPopM); arg(kScatter, 5, bPopS); arg(kScatter, 6, bPopL);
            arg(kScatter, 7, first_idx); arg(kScatter, 8, kummer_idx); arg(kScatter, 9, total_idx);
            arg(kScatter, 10, L); arg(kScatter, 11, R2); arg(kScatter, 12, W); arg(kScatter, 13, bin_cap);
            launch(kScatter, total_idx - first_idx);
        }
        cur_bin_cap = bin_cap;
        prof_mark(3, t0);

        if (!bh_off.empty())
        {
            uint32_t nb = (uint32_t)bh_off.size();
            arg(kBighits, 0, bAcc); arg(kBighits, 1, cs.bhO); arg(kBighits, 2, cs.bhL); arg(kBighits, 3, nb);
            launch(kBighits, nb);
        }
        prof_mark(4, t0);

        {
            uint32_t na = (uint32_t)act_q.size();
            arg(kFused, 0, bAcc); arg(kFused, 1, cs.mask);
            arg(kFused, 2, bTz); arg(kFused, 3, bPat1); arg(kFused, 4, bPat2);
            arg(kFused, 5, cs.phTz); arg(kFused, 6, cs.ph1); arg(kFused, 7, cs.ph2);
            arg(kFused, 8, cs.tseg); arg(kFused, 9, cs.actQ); arg(kFused, 10, cs.actO); arg(kFused, 11, cs.actL);
            arg(kFused, 12, bBins); arg(kFused, 13, bBcnt); arg(kFused, 14, cur_bin_cap);
            arg(kFused, 15, na); arg(kFused, 16, L); arg(kFused, 17, W);
            launch(kFused, (size_t)nseg * FUSED_LSZ, FUSED_LSZ);
        }
        prof_mark(5, t0);

        uint32_t cap = SURV_CAP;
        arg(kRuns, 0, cs.mask); arg(kRuns, 1, cs.out); arg(kRuns, 2, cs.outCnt);
        arg(kRuns, 3, Kk); arg(kRuns, 4, W); arg(kRuns, 5, cap);
        launch(kRuns, ((size_t)(W + 31) / 32 + 255) / 256 * 256, 0, &cs.evC);
        if (g_prof.on)
        {
            CLCHECK(CL.Finish(q));
            double t1 = now_s();
            g_prof.t[6] += t1 - t0;
            t0 = t1;
        }
        CLCHECK(CL.Flush(q));
    }

    void finish_chunk(int si, std::vector<uint32_t> &surv, bool &overflow)
    {
        double t0 = now_s();
        ChunkSet &cs = sets[si];
        uint32_t cnt = 0;
        CLCHECK(CL.EnqueueReadBuffer(qx, cs.outCnt, CL_TRUE, 0, 4, &cnt, 1, (const void *)&cs.evC, nullptr));
        CL.ReleaseEvent(cs.evC);
        cs.evC = nullptr;
        if (cnt >= SURV_CAP) { overflow = true; surv.clear(); }
        else
        {
            overflow = false;
            surv.resize(cnt);
            if (cnt)
                CLCHECK(CL.EnqueueReadBuffer(qx, cs.out, CL_TRUE, 0, (size_t)cnt * 4, surv.data(), 0, nullptr, nullptr));
        }
        prof_mark(7, t0);
        if (g_prof.on)
            ++g_prof.chunks;
    }

    const uint8_t *mask_host(int si, uint32_t W)
    {
        mask_hostbuf.resize(W);
        CLCHECK(CL.EnqueueReadBuffer(qx, sets[si].mask, CL_TRUE, 0, W, mask_hostbuf.data(), 0, nullptr, nullptr));
        return mask_hostbuf.data();
    }
#endif
};

static GpuCtx g_gpu;

template <uint64_t K>
uint64_t solve_impl(uint64_t start_L)
{
    const uint64_t CHUNK = GPU_CHUNK;
    g_gpu.ensure_pop();

    uint64_t global_min = UINT64_MAX;
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<uint32_t> act_q, act_off, act_lg, bh_off, bh_lg, Tseg, ph_tz, ph1, ph2, surv;
    std::vector<std::array<uint32_t, 3>> act_tmp;

    // Software pipeline (depth 2 unless profiling/GPU_PIPELINE=0): while the
    // device computes chunk i, the host preps + uploads chunk i+1 on the
    // transfer queue and consumes chunk i-1's survivors (including the exact
    // oracle). The in-order compute queue serializes the shared bAcc / bPoff /
    // bBins across chunks; per-chunk buffers alternate between two sets.
#ifdef CPU_SIM
    const uint64_t depth = 1;
#else
    const uint64_t depth = (g_prof.on || !g_pipeline) ? 1 : 2;
#endif
    struct { uint64_t L = 0; uint32_t W = 0; } metas[2];
    uint64_t issued = 0, consumed = 0;
    while (true)
    {
        while (issued < consumed + depth)
        {
            uint64_t chunk_id = issued;
            (void)chunk_id;
#ifdef BENCHMARK
            if (issued >= BENCH_CANDS / CHUNK)
                break;
#endif
            uint64_t L_chunk = start_L + issued * CHUNK;
            if (L_chunk >= g_end_L || L_chunk > global_min)
                break;
            double t_prep = now_s();
            // clamp the final chunk to the worker range so this job never
            // scans candidates owned by a neighbouring job: [start+K, end+K)
            // exactly, matching the CPU worker's tiling contract
            uint64_t span = CHUNK;
            if (g_end_L - L_chunk < span)
                span = g_end_L - L_chunk;
            uint64_t R_chunk = L_chunk + span + K - 1;
            if (R_chunk > RANGE_CEIL) [[unlikely]]
            {
                std::cerr << "\nRange exceeds the u8 accumulator ceiling (~5.8e17).\n";
                std::exit(1);
            }
            uint32_t W = (uint32_t)(R_chunk - L_chunk + 1);
            uint32_t nseg = (W + 16383) / 16384;

            uint64_t max_p = std::max<uint64_t>(std::sqrt(2 * R_chunk) + 1, 2 * K);
            if (max_p > prime_limit) [[unlikely]]
            {
                std::cerr << "\nPrime table exhausted (need " << max_p << ", have " << prime_limit
                          << "). Restart to resume from checkpoint with a larger table.\n";
                std::exit(1);
            }
            auto it = std::upper_bound(primes.begin(), primes.end(), max_p,
                                       [](uint64_t val, const PrimeData &pd) { return val < pd.p; });
            size_t total_idx = std::distance(primes.begin(), it);
            auto it_large = std::upper_bound(primes.begin(), primes.begin() + total_idx, OPT_BLOCK_SIZE,
                                             [](uint32_t val, const PrimeData &pd) { return val < pd.p; });
            size_t first_idx = std::distance(primes.begin(), it_large);
            uint32_t p_thresh = std::max<uint32_t>(2 * K, (uint32_t)std::cbrt(2.0 * R_chunk) + 2);
            auto it_thresh = std::upper_bound(primes.begin() + first_idx, primes.begin() + total_idx, p_thresh,
                                              [](uint32_t val, const PrimeData &pd) { return val < pd.p; });
            size_t kummer_idx = std::distance(primes.begin(), it_thresh);

            // active small strided prime powers (identical logic to CPU v9 Phase 1),
            // sorted by q so neighbouring GPU lanes get equal loop trip counts
            const uint32_t small_cap = (uint32_t)std::min<uint64_t>(max_p, OPT_BLOCK_SIZE);
            act_tmp.clear();
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
                act_tmp.push_back({st.q, (uint32_t)(start_c * st.q - L_chunk), st.log2s});
            }
            std::sort(act_tmp.begin(), act_tmp.end(),
                      [](const auto &a, const auto &b) { return a[0] < b[0]; });
            act_q.clear(); act_off.clear(); act_lg.clear();
            for (const auto &a : act_tmp)
            {
                act_q.push_back(a[0]);
                act_off.push_back(a[1]);
                act_lg.push_back(a[2]);
            }
            // explicit big-power hits
            bh_off.clear(); bh_lg.clear();
            for (const auto &bp : big_powers)
            {
                if (bp.q > R_chunk)
                    break;
                for (uint64_t m = ((L_chunk + bp.q - 1) / bp.q) * bp.q; m <= R_chunk; m += bp.q)
                {
                    bh_off.push_back((uint32_t)(m - L_chunk));
                    bh_lg.push_back(bp.log2s);
                }
            }
            // per-segment thresholds and table phases
            Tseg.resize(nseg); ph_tz.resize(nseg); ph1.resize(nseg); ph2.resize(nseg);
            for (uint32_t s = 0; s < nseg; ++s)
            {
                uint64_t sx = L_chunk + (uint64_t)s * 16384;
                double tf = (double)LOG_SCALE * std::log2((double)sx) - LOG_SLACK;
                Tseg[s] = tf <= 0.0 ? 0u : (uint32_t)tf;
                ph_tz[s] = (uint32_t)(sx & 65535);
                ph1[s] = (uint32_t)(sx % PRESIEVE_PERIOD);
                ph2[s] = (uint32_t)(sx % PRESIEVE2_PERIOD);
            }
            if (g_prof.on)
                g_prof.t[0] += now_s() - t_prep;
            metas[issued & 1].L = L_chunk;
            metas[issued & 1].W = W;
            g_gpu.submit_chunk((int)(issued & 1), L_chunk, R_chunk, W, nseg, act_q, act_off, act_lg,
                               bh_off, bh_lg, Tseg, ph_tz, ph1, ph2, (uint32_t)first_idx,
                               (uint32_t)kummer_idx, (uint32_t)total_idx, (uint32_t)K);
            ++issued;
        }
        if (consumed == issued)
            break;
        int si = (int)(consumed & 1);
        bool overflow = false;
        g_gpu.finish_chunk(si, surv, overflow);
        uint64_t L_chunk = metas[si].L;
        uint32_t W = metas[si].W;
        if (!overflow)
        {
            std::sort(surv.begin(), surv.end());
            for (uint32_t j : surv)
            {
                uint64_t n = L_chunk + j;
                if (n > K && n < global_min && exact_check<K>(n))
                    global_min = n;
            }
        }
        else
        {
            const uint8_t *mask = g_gpu.mask_host(si, W);
            uint32_t run = 0;
            for (uint32_t j = 0; j < W; ++j)
            {
                run = mask[j] ? run + 1 : 0;
                if (run >= K + 1)
                {
                    uint64_t n = L_chunk + j;
                    if (n > K && n < global_min && exact_check<K>(n))
                        global_min = n;
                }
            }
        }
#ifndef BENCHMARK
        if (!g_worker_mode && consumed > 0 && consumed % 8 == 0)
        {
            uint64_t safe_L = L_chunk;
            std::ofstream fout("checkpoint-396.tmp");
            if (fout)
            {
                fout << K << " " << safe_L << "\n";
                fout.close();
                std::error_code ec;
                std::filesystem::rename("checkpoint-396.tmp", "checkpoint-396.txt", ec);
                double speed = (safe_L - start_L) /
                               std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
                std::cout << "\r[Checkpoint] k = " << std::setw(2) << K << " | Candidate L = " << safe_L
                          << " | Speed: " << std::fixed << std::setprecision(2) << (speed / 1e6)
                          << " M candidates/s   " << std::flush;
            }
        }
#endif
        ++consumed;
    }
    g_prof.report();
    return global_min;
}

uint64_t solve(uint64_t k, uint64_t start_L)
{
#ifdef BENCHMARK
    uint64_t need = (uint64_t)std::sqrt(2.0 * (double)(start_L + BENCH_CANDS + 64)) + OPT_BLOCK_SIZE;
#else
    uint64_t need = (uint64_t)(16.0 * std::sqrt(2.0 * (double)start_L)) + OPT_BLOCK_SIZE;
#endif
    extend_primes(std::max<uint64_t>(need, 1'000'000));

    switch (k)
    {
    case 1: return solve_impl<1>(start_L);
    case 2: return solve_impl<2>(start_L);
    case 3: return solve_impl<3>(start_L);
    case 4: return solve_impl<4>(start_L);
    case 5: return solve_impl<5>(start_L);
    case 6: return solve_impl<6>(start_L);
    case 7: return solve_impl<7>(start_L);
    case 8: return solve_impl<8>(start_L);
    case 9: return solve_impl<9>(start_L);
    case 10: return solve_impl<10>(start_L);
    case 11: return solve_impl<11>(start_L);
    case 12: return solve_impl<12>(start_L);
    case 13: return solve_impl<13>(start_L);
    case 14: return solve_impl<14>(start_L);
    case 15: return solve_impl<15>(start_L);
    case 16: return solve_impl<16>(start_L);
    case 17: return solve_impl<17>(start_L);
    case 18: return solve_impl<18>(start_L);
    case 19: return solve_impl<19>(start_L);
    case 20: return solve_impl<20>(start_L);
    default: return 0;
    }
}

int main(int argc, char **argv)
{
    // Worker mode for the multi-host coordinator: ./binary <k> <start_L> <end_L>
    // (prints "RESULT <k> <min-or-0>"; ranges as in the CPU worker)
    if (argc == 4)
    {
        g_worker_mode = true;
        uint64_t kk = std::strtoull(argv[1], nullptr, 10);
        uint64_t s = std::strtoull(argv[2], nullptr, 10);
        g_end_L = std::strtoull(argv[3], nullptr, 10);
        get_primes(1'000'000);
        g_gpu.init_device();
        uint64_t ans = solve(kk, s);
        std::cout << "RESULT " << kk << " " << (ans == UINT64_MAX ? 0 : ans) << std::endl;
        return 0;
    }
    std::cout << "Erdos #396 GPU (OpenCL) solver\n";
    std::cout << "Generating base prime table (grows on demand, cap " << (uint64_t)MAXP << ")...\n";
    auto start_primes = std::chrono::high_resolution_clock::now();
    get_primes(1'000'000);
    std::cout << "Primes generated in "
              << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_primes).count()
              << " seconds. (" << small_striders.size() << " small strides, " << big_powers.size()
              << " big powers)\n";
    g_prof.on = std::getenv("ERDOS_PROF") != nullptr;
    if (const char *e = std::getenv("GPU_PIPELINE"))
        g_pipeline = std::atoi(e) != 0;
    if (const char *e = std::getenv("GPU_BIN_HEADROOM"))
    {
        double v = std::atof(e);
        if (v >= 1.2 && v <= 8.0) g_bin_head = v;
    }
    g_gpu.init_device();
    std::cout << "\n";

#ifdef BENCHMARK
    std::vector<std::pair<uint64_t, uint64_t>> tests{
        {11, 1'000'000'000'000ULL},   {12, 5'000'000'000'000ULL},     {13, 18'000'000'000'000ULL},
        {14, 359'000'000'000'000ULL}, {15, 2'880'000'000'000'000ULL},
    };
    for (const auto &[k, start_L] : tests)
    {
        uint64_t total_candidates = BENCH_CANDS;
        std::cout << "--- BENCHMARK MODE ---\n";
        std::cout << "Testing k=" << k << " | Workload: " << total_candidates / 1000000 << " M candidates\n\n";
        for (int run = 1; run <= RUNS; ++run)
        {
            auto start = std::chrono::high_resolution_clock::now();
            uint64_t ans = solve(k, start_L);
            (void)ans;
            double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
            double speed = total_candidates / seconds;
            std::cout << "Run " << run << " | Time: " << std::fixed << std::setprecision(4) << std::setw(8) << seconds
                      << " s | Speed: " << std::fixed << std::setprecision(2) << std::setw(8) << (speed / 1e6)
                      << " M candidates/s\n";
        }
    }
#else
    uint64_t start_k = 1;
    uint64_t start_L = 1;
    if (std::filesystem::exists("checkpoint-396.txt"))
    {
        std::ifstream fin("checkpoint-396.txt");
        if (fin >> start_k >> start_L)
            std::cout << "--> Resuming from checkpoint: k = " << start_k << ", L_batch = " << start_L << "\n\n";
        else { start_k = 1; start_L = 1; }
    }
    uint64_t kmax_rt = KMAX;
    if (const char *e = std::getenv("KMAX_RT")) kmax_rt = std::strtoull(e, nullptr, 10);
    for (uint64_t k = start_k; k <= kmax_rt; ++k)
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
        if (results_file) results_file << output_str;
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