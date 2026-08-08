// erdos_396_kernels.cl - dual-target kernels for the Erdos #396 log sieve:
//   - compiled as OpenCL C by the GPU driver (host loads this file at runtime)
//   - #included as C++ by the host built with -DCPU_SIM (single-threaded
//     simulation used to validate the exact kernel logic against known answers)
//
// Pipeline per chunk: (zero bin counters) -> k_scatter -> k_fused -> k_runs.
//
// Memory design. All accumulation happens in 16 KB of SLM per 16384-candidate
// segment (u8 at LOG_SCALE = 4 units/bit, packed 4 bytes per u32 word; OpenCL
// has no byte atomics, and adding lg << 8*lane is carry-safe because a fully
// accounted byte never exceeds 255 below the range ceiling ~5.8e17 - the host
// guards this). There is no global accumulator: every mass source reaches the
// segment's SLM through one of
//   phase_init    : LOG_SCALE*ctz(x) + presieve patterns, via phase-shifted
//                   tables
//   phase_strides : small prime powers q <= OPT_BLOCK_SIZE; alignment to the
//                   segment is a magic multiply (host-precomputed magic/shift)
//                   instead of a hardware division, which cost ~0.3 divisions
//                   per candidate on its own
//   phase_drain   : bucket-prime records from this segment's bin, plus the
//                   host-enumerated big prime-power records (CSR by segment)
//
// Bucket primes (p > OPT_BLOCK_SIZE) exploit a structural fact: p exceeds
// four segment lengths, so each prime has AT MOST ONE hit per segment. Their
// hits therefore need no accumulation pass of their own: k_scatter walks each
// prime's hits once per chunk (after the chunk-wide Kummer skip) and appends
// a 32-bit record (segment-local offset | LOG<<16) to the hit segment's bin -
// one atomic counter per segment (L2-resident) reserves the slot, and because
// each bin fills sequentially the scattered 4-byte stores coalesce into ~nseg
// open write streams. Records past the bin capacity are DROPPED (the counter
// still runs): k_scatter tracks the needed capacity in *ovf and the host
// resubmits the whole chunk with a larger capacity, so overflow costs one
// duplicate chunk (statistically rare; capacity is sized ~2x the mean), never
// correctness.
//
// The mask is bit-packed (1 bit per candidate): phase_mask emits one u32 per
// 32 candidates and k_runs finds runs of K+1 set bits word-parallel through a
// 64-bit window (K <= 20 < 32, so one preceding word is enough context). This
// is 16x less mask traffic than byte masks and 32x fewer scan items.
// All math is integer only - the Arc A770 has no native FP64.

#ifdef __OPENCL_VERSION__
/* ---------------- OpenCL C target ---------------- */
#define PHASE_FN static inline void
#define KERNEL __kernel
#define GA __global
#define LA __local
#define GID ((uint)get_global_id(0))
#define ATOMIC_ADD_G(p, v) atomic_add((volatile __global uint *)(p), (uint)(v))
#define ATOMIC_ADD_L(p, v) atomic_add((volatile __local uint *)(p), (uint)(v))
#define ATOMIC_FETCH_ADD_G(p, v) atomic_add((volatile __global uint *)(p), (uint)(v))
#define ATOMIC_MAX_G(p, v) atomic_max((volatile __global uint *)(p), (uint)(v))
#define MULHI64(a, b) mul_hi((ulong)(a), (ulong)(b))
#define CTZ64(x) ((uint)(63 - clz((ulong)(x) & (ulong)(0 - (ulong)(x)))))
#else
/* ---------------- C++ simulation target ----------------
   The sim calls the phase_* functions directly (lid=0, lsz=1 covers a whole
   segment per call, which is valid because each phase is lane-parallel);
   only the thin __kernel entry points are OpenCL-only glue.
   NOTE: ulong must be unsigned long long, not unsigned long - the latter is
   32-bit on Windows (LLP64) and would silently break the 64-bit math. */
typedef unsigned int uint;
typedef unsigned long long ulong;
typedef unsigned short ushort;
typedef unsigned char uchar;
static uint g_sim_gid;
#define PHASE_FN static inline void
#define KERNEL static inline
#define GA
#define LA
#define GID (g_sim_gid)
#define ATOMIC_ADD_G(p, v) ((void)(*(p) += (uint)(v)))
#define ATOMIC_ADD_L(p, v) ((void)(*(p) += (uint)(v)))
#define ATOMIC_FETCH_ADD_G(p, v) sim_fetch_add((p), (uint)(v))
static inline uint sim_fetch_add(uint *p, uint v) { uint o = *p; *p += v; return o; }
#define ATOMIC_MAX_G(p, v) sim_max((p), (uint)(v))
static inline void sim_max(uint *p, uint v) { if (v > *p) *p = v; }
#define MULHI64(a, b) ((ulong)(((unsigned __int128)(a) * (unsigned __int128)(b)) >> 64))
#define CTZ64(x) ((uint)__builtin_ctzll((unsigned long long)(x)))
#endif

#define SEG_SHIFT 14
#define SEG_SIZE 16384u
#define SEG_WORDS 4096u /* u32 words of 4 packed u8 lanes per segment = 16 KB SLM */

/* -------------------------------------------------------------------------
   Fused segment phases. `grp` = segment index, `lid`/`lsz` = local id/size.
   Segment covers chunk-relative candidates [grp*16384, grp*16384+16384) & <W.
   ------------------------------------------------------------------------- */

/* Phase 1: init local acc bytes.  acc8(j) = LOG_SCALE*ctz(x) + presieve1 +
   presieve2 for x = L+j, via phase-shifted tables (ph_* are host-computed
   per-segment table offsets; multiples of 65536 fixed up in code). */
PHASE_FN phase_init(LA uint *lacc, uint lid, uint lsz, uint grp,
                    GA const uchar *tzv, GA const uchar *pat1, GA const uchar *pat2,
                    GA const uint *ph_tz, GA const uint *ph1, GA const uint *ph2,
                    ulong L, uint W)
{
    uint seg_base = grp << SEG_SHIFT;
    uint o_tz = ph_tz[grp], o1 = ph1[grp], o2 = ph2[grp];
    for (uint wi = lid; wi < SEG_WORDS; wi += lsz)
    {
        uint w = 0u;
        uint j0 = seg_base + wi * 4u;
        if (j0 < W)
        {
            for (uint bl = 0u; bl < 4u; ++bl)
            {
                uint j = j0 + bl;
                if (j >= W)
                    break;
                uint o = wi * 4u + bl;
                ulong x = L + (ulong)j;
                uint tz = (((uint)x & 65535u) == 0u) ? (CTZ64(x) << 2) : (uint)tzv[o_tz + o];
                w |= (tz + (uint)pat1[o1 + o] + (uint)pat2[o2 + o]) << (bl << 3);
            }
        }
        lacc[wi] = w;
    }
}

/* Phase 2: strided adds for the small prime powers, into LOCAL memory.
   Host sorts the active list by q so neighbouring lanes get equal trip counts.
   Alignment to the segment start uses the host-precomputed magic/shift for q
   (floor((d+q-1)/q) as a mul_hi) - a hardware division here would cost ~0.3
   divisions per candidate, comparable to the whole bucket-hit workload.
   q is always odd => SLM bank-conflict friendly. */
PHASE_FN phase_strides(LA uint *lacc, uint lid, uint lsz, uint grp,
                       GA const uint *act_q, GA const uint *act_off, GA const uint *act_lg,
                       GA const ulong *act_magic, GA const uchar *act_shift,
                       uint n_act, uint W)
{
    uint seg_start = grp << SEG_SHIFT;
    uint seg_end = seg_start + SEG_SIZE;
    if (seg_end > W)
        seg_end = W;
    for (uint s = lid; s < n_act; s += lsz)
    {
        uint q = act_q[s];
        uint lg = act_lg[s];
        uint j = act_off[s];
        if (j < seg_start)
        {
            ulong num = (ulong)(seg_start - j) + (ulong)q - 1u;
            ulong c = MULHI64(num, act_magic[s]) >> act_shift[s];
            j += (uint)(c * (ulong)q);
        }
        for (; j < seg_end; j += q)
        {
            uint lj = j - seg_start;
            ATOMIC_ADD_L(&lacc[lj >> 2], lg << ((lj & 3u) << 3));
        }
    }
}

/* Phase 2b: drain this segment's bucket-hit bin into local memory, then the
   host-enumerated big prime-power records (same 32-bit format, CSR indexed by
   segment).  Records are dense and read coalesced; each is one SLM add.
   Counts beyond bin_cap correspond to dropped records - the host detects the
   overflow via *ovf and resubmits the chunk, so this path stays exact. */
PHASE_FN phase_drain(LA uint *lacc, uint lid, uint lsz, uint grp,
                     GA const uint *bins, GA const uint *bcnt, uint bin_cap,
                     GA const uint *bh_start, GA const uint *bh_rec)
{
    uint n = bcnt[grp];
    if (n > bin_cap)
        n = bin_cap;
    GA const uint *b = bins + (ulong)grp * bin_cap;
    for (uint i = lid; i < n; i += lsz)
    {
        uint rec = b[i];
        uint lj = rec & (SEG_SIZE - 1u);
        ATOMIC_ADD_L(&lacc[lj >> 2], (rec >> 16) << ((lj & 3u) << 3));
    }
    uint e = bh_start[grp + 1u];
    for (uint i = bh_start[grp] + lid; i < e; i += lsz)
    {
        uint rec = bh_rec[i];
        uint lj = rec & (SEG_SIZE - 1u);
        ATOMIC_ADD_L(&lacc[lj >> 2], (rec >> 16) << ((lj & 3u) << 3));
    }
}

/* Phase 3: threshold each byte against the per-segment T and emit the
   bit-packed mask - one u32 of 32 candidate bits per 8 lacc words.  The SLM
   byte alone is the full accounting (there is no global accumulator), which
   the range ceiling keeps below 256.  Bits at or beyond W stay 0, so no run
   can extend past the window. */
PHASE_FN phase_mask(LA const uint *lacc, uint lid, uint lsz, uint grp,
                    GA uint *mask, GA const uint *Tseg, uint W)
{
    uint seg_base = grp << SEG_SHIFT;
    uint T = Tseg[grp];
    for (uint pw = lid; pw < (SEG_SIZE / 32u); pw += lsz)
    {
        uint j0 = seg_base + pw * 32u;
        if (j0 >= W)
            break;
        uint bits = 0u;
        for (uint b = 0u; b < 32u; ++b)
        {
            uint j = j0 + b;
            if (j >= W)
                break;
            uint lj = pw * 32u + b;
            if (((lacc[lj >> 2] >> ((lj & 3u) << 3)) & 0xFFu) >= T)
                bits |= 1u << b;
        }
        mask[(seg_base >> 5) + pw] = bits;
    }
}

#ifdef __OPENCL_VERSION__
/* OpenCL-only glue: one work-group per segment, acc in SLM, barriers between
   phases.  The phase bodies above are the validated single-source logic. */
KERNEL void k_fused(GA uint *mask,
                    GA const uchar *tzv, GA const uchar *pat1, GA const uchar *pat2,
                    GA const uint *ph_tz, GA const uint *ph1, GA const uint *ph2,
                    GA const uint *Tseg, GA const uint *act_q, GA const uint *act_off,
                    GA const uint *act_lg, GA const ulong *act_magic, GA const uchar *act_shift,
                    GA const uint *bins, GA const uint *bcnt, uint bin_cap,
                    GA const uint *bh_start, GA const uint *bh_rec,
                    uint n_act, ulong L, uint W)
{
    __local uint lacc[SEG_WORDS];
    uint grp = (uint)get_group_id(0);
    uint lid = (uint)get_local_id(0);
    uint lsz = (uint)get_local_size(0);
    phase_init(lacc, lid, lsz, grp, tzv, pat1, pat2, ph_tz, ph1, ph2, L, W);
    barrier(CLK_LOCAL_MEM_FENCE);
    phase_strides(lacc, lid, lsz, grp, act_q, act_off, act_lg, act_magic, act_shift, n_act, W);
    phase_drain(lacc, lid, lsz, grp, bins, bcnt, bin_cap, bh_start, bh_rec);
    barrier(CLK_LOCAL_MEM_FENCE);
    phase_mask(lacc, lid, lsz, grp, mask, Tseg, W);
}
#endif

/* Zero a u32 buffer (bin counters when the runtime lacks clEnqueueFillBuffer). */
KERNEL void k_zero(GA uint *accp, uint np2)
{
    uint i = GID;
    if (i < np2)
        accp[i] = 0u;
}

/* Bucket-prime scatter: one work-item per prime.  Applies the chunk-wide
   Kummer skip (two magic multiplies: a base prime above cbrt(2R) whose
   quotient digit cannot reach (p+1)/2 anywhere in the chunk is provably fatal
   wherever it hits, so its hits are omitted and the missing mass keeps those
   candidates under threshold), computes the first hit, then appends one
   record per hit to the hit segment's bin.  Records past bin_cap are dropped
   and the needed capacity is recorded in *ovf; the host resubmits the chunk
   with a larger capacity, so overflow costs speed, never correctness. */
KERNEL void k_scatter(GA uint *bins, GA uint *bcnt, GA uint *ovf,
                      GA const uint *pp, GA const ulong *pmagic, GA const uchar *pshift,
                      GA const ushort *plg, uint first_idx, uint kummer_idx, uint total_idx,
                      ulong L, ulong R2, uint W, uint bin_cap)
{
    uint idx = first_idx + GID;
    if (idx >= total_idx)
        return;
    ulong p = (ulong)pp[idx];
    ulong magic = pmagic[idx];
    uint shift = (uint)pshift[idx];
    if (idx >= kummer_idx)
    {
        ulong p2 = p * p;
        ulong d = MULHI64(L, magic) >> shift;
        ulong c = MULHI64(d, magic) >> shift;
        if (R2 < (c * 2u + 1u) * p2)
            return;
    }
    ulong num = L + p - 1u;
    ulong start_c = MULHI64(num, magic) >> shift;
    ulong hit64 = start_c * p - L;
    uint lgv = (uint)plg[idx];
    uint pstep = (uint)p; /* p <= sqrt(2R) < 2^32 under the range guard */
    for (ulong h = hit64; h < (ulong)W; h += pstep)
    {
        uint hit = (uint)h;
        uint seg = hit >> SEG_SHIFT;
        uint slot = ATOMIC_FETCH_ADD_G(&bcnt[seg], 1u);
        if (slot < bin_cap)
            bins[(ulong)seg * bin_cap + slot] = (hit & (SEG_SIZE - 1u)) | (lgv << 16);
        else
            ATOMIC_MAX_G(&ovf[0], slot + 1u);
    }
}

/* Find runs of K+1 set mask BITS; a run ending at bit j is the candidate
   n=L+j.  One work-item per 32-candidate mask word: build a 64-bit window
   from this word and the previous one (K <= 20 < 32, so that is enough
   warm-up context), AND-shift K times so bit t survives iff bits t..t+K were
   all set, then emit the run-ends this item owns ([32*gid, 32*gid+32)).
   Nothing is missed or duplicated, and words with no set bits exit early. */
KERNEL void k_runs(GA const uint *mask, GA uint *out, GA uint *out_count, uint Kk, uint W, uint cap)
{
    uint wi = GID;
    uint base = wi * 32u;
    if (base >= W)
        return;
    uint cur = mask[wi];
    if (cur == 0u)
        return; /* a run must end on a set bit */
    ulong v = ((ulong)cur << 32) | (ulong)(wi ? mask[wi - 1u] : 0u);
    for (uint i = 0u; i < Kk; ++i)
        v &= v >> 1;
    /* bit t of v set => candidates base-32+t .. base-32+t+K all set, i.e. a
       run ends at j = base-32+t+K; owned ends have t in [32-K, 64-K), so
       after shifting by 32-K, bit t' maps to j = base + t'. */
    v >>= 32u - Kk;
    for (uint t = 0u; t < 32u && v; ++t, v >>= 1)
    {
        if ((v & 1u) != 0u)
        {
            uint j = base + t;
            if (j < W)
            {
                uint slot = ATOMIC_FETCH_ADD_G(out_count, 1u);
                if (slot < cap)
                    out[slot] = j;
            }
        }
    }
}
