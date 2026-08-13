/**
 * W4A4 GEMM CUDA Kernel
 *
 * Fused INT4 weight × INT4 activation GEMM for MorphoQuant.
 * Activation quantization is fused into the SMEM load path.
 *
 * Math:
 *   smem_A[m,k] = A_int4[m,k] - a_zero[k]          (zero-centered int→fp)
 *   smem_W[n,k] = W_int4[n,k] * w_scale[n]          (dequantized weight)
 *   C = smem_A @ smem_W^T                            (FP16 Tensor Core)
 *   output = C                                       (no epilogue correction needed)
 *
 * a_scale is absorbed into W_int4 during preprocessing (build_w4a4_weights).
 * a_zero is subtracted in the activation load path, so no z_bias is needed.
 */

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

// ============================================================================
// HalfTraits
// ============================================================================

template<typename T> struct HalfTraits {};

template<> struct HalfTraits<__half> {
    static __device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
    static __device__ __forceinline__ __half from_float(float v) { return __float2half(v); }
};
template<> struct HalfTraits<__nv_bfloat16> {
    static __device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
    static __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
};

// ============================================================================
// Tile constants
// ============================================================================
#define TILE_M  64
#define TILE_N  64
#define TILE_K  128
#define WMMA_M  16
#define WMMA_N  16
#define WMMA_K  16
#define FRAGS_M (TILE_M / WMMA_M)   // 4
#define FRAGS_N (TILE_N / WMMA_N)   // 4
#define FRAGS_K (TILE_K / WMMA_K)   // 8
#define WARP_SZ 32

// ============================================================================
// Sign-extend unsigned nibble → signed int8
// ============================================================================
__device__ __forceinline__ int8_t nibble2signed(uint8_t n) {
    return (n >= 8) ? (int8_t)(n - 16) : (int8_t)n;
}

// ============================================================================
// w4a4_gemm_kernel
// ============================================================================
template<typename scalar_t>
__global__ void w4a4_gemm_kernel(
    const scalar_t* __restrict__ x,          // [M, K]
    const uint8_t*  __restrict__ W_packed,   // [N, K/2] packed INT4, row-major
    const float*    __restrict__ act_scale,  // [K]
    const float*    __restrict__ act_zero,   // [K]
    const half*     __restrict__ w_scale,    // [N] (used for dequant in smem_W)
    const float*    __restrict__ z_bias,     // [N] (unused in kernel; for API compat)
    scalar_t*       __restrict__ out,        // [M, N]
    int M, int N, int K,
    float qmax
) {
    using HT = HalfTraits<scalar_t>;

    int block_m = blockIdx.y;
    int block_n = blockIdx.x;
    int row_s = block_m * TILE_M;
    int col_s = block_n * TILE_N;

    // ---- Shared memory ----
    // smem_A: [TILE_M, TILE_K] row-major (for WMMA row_major A load)
    // smem_W: [TILE_N, TILE_K] N-major   (for WMMA col_major B load)
    //   WMMA col_major B[k,n] = ptr[n*ldm+k], ldm≥K
    //   smem_W[n*TILE_K+k] → ptr=&smem_W[n_off*TILE_K+k_off], ldm=TILE_K
    __shared__ half smem_A[TILE_M * TILE_K];
    __shared__ half smem_W[TILE_N * TILE_K];

    // ---- Accumulator fragments ----
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c[FRAGS_M][FRAGS_N];
#pragma unroll
    for (int mi = 0; mi < FRAGS_M; mi++)
#pragma unroll
        for (int ni = 0; ni < FRAGS_N; ni++)
            wmma::fill_fragment(c[mi][ni], __float2half(0.0f));

    int tid = threadIdx.x;
    int nthr = blockDim.x;

    // ---- K loop ----
    for (int kb = 0; kb < K; kb += TILE_K) {
        int ks = min(TILE_K, K - kb);

        // 1. Load A → dequantize (A_int4 - a_zero) → smem_A
        int nA = TILE_M * ks;
        for (int i = tid; i < nA; i += nthr) {
            int lm = i / ks, lk = i % ks;
            int gm = row_s + lm, gk = kb + lk;
            float v = 0.0f;
            if (gm < M && gk < K) {
                v = HT::to_float(x[gm * K + gk]);
                float q = nearbyintf(v / act_scale[gk] + act_zero[gk]);
                q = fminf(fmaxf(q, 0.0f), qmax);
                v = q - act_zero[gk];   // a_scale absorbed in W_int4
            }
            smem_A[lm * ks + lk] = __float2half(v);
        }

        // 2. Load W → unpack INT4 → dequantize → smem_W [TILE_N, TILE_K]
        int nW = TILE_N * (ks / 2);
        for (int i = tid; i < nW; i += nthr) {
            int ln = i / (ks / 2);              // local N (0..TILE_N-1)
            int lb = i % (ks / 2);              // byte index within K_packed
            int lk0 = lb * 2, lk1 = lb * 2 + 1; // two K indices per byte
            int gn = col_s + ln;
            int gk0 = kb + lk0, gk1 = kb + lk1;

            float w0 = 0.0f, w1 = 0.0f;
            if (gn < N) {
                float ws = __half2float(w_scale[gn]);
                if (gk0 < K) {
                    uint8_t b0 = W_packed[gn * (K / 2) + gk0 / 2];
                    uint8_t n0 = (gk0 % 2 == 0) ? (b0 & 0x0F) : ((b0 >> 4) & 0x0F);
                    w0 = (float)nibble2signed(n0) * ws;
                }
                if (gk1 < K) {
                    uint8_t b1 = W_packed[gn * (K / 2) + gk1 / 2];
                    uint8_t n1 = (gk1 % 2 == 0) ? (b1 & 0x0F) : ((b1 >> 4) & 0x0F);
                    w1 = (float)nibble2signed(n1) * ws;
                }
            }
            smem_W[ln * TILE_K + lk0] = __float2half(w0);
            smem_W[ln * TILE_K + lk1] = __float2half(w1);
        }
        __syncthreads();

        // 3. WMMA compute (warp 0 only)
        if (tid < WARP_SZ) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::col_major> bf;

#pragma unroll
            for (int ki = 0; ki < FRAGS_K; ki++) {
                int ko = ki * WMMA_K;
#pragma unroll
                for (int mi = 0; mi < FRAGS_M; mi++) {
                    int mo = mi * WMMA_M;
#pragma unroll
                    for (int ni = 0; ni < FRAGS_N; ni++) {
                        int no = ni * WMMA_N;

                        // A: row-major A[m,k] = ptr[m*lda+k], lda=ks
                        wmma::load_matrix_sync(af,
                            &smem_A[mo * ks + ko], ks);

                        // B: col-major B[k,n] = ptr[n*ldb+k], ldb=TILE_K
                        // smem_W is [TILE_N, TILE_K]
                        wmma::load_matrix_sync(bf,
                            &smem_W[no * TILE_K + ko], TILE_K);

                        wmma::mma_sync(c[mi][ni], af, bf, c[mi][ni]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // 4. Store accumulators → smem_out
    __shared__ half smem_out[TILE_M * TILE_N];

    if (tid < WARP_SZ) {
#pragma unroll
        for (int mi = 0; mi < FRAGS_M; mi++) {
            int mo = mi * WMMA_M;
#pragma unroll
            for (int ni = 0; ni < FRAGS_N; ni++) {
                int no = ni * WMMA_N;
                wmma::store_matrix_sync(
                    &smem_out[mo * TILE_N + no],
                    c[mi][ni], TILE_N, wmma::mem_row_major);
            }
        }
    }
    __syncthreads();

    // 5. Copy to global output
    int nOut = TILE_M * TILE_N;
    for (int i = tid; i < nOut; i += nthr) {
        int lm = i / TILE_N, ln = i % TILE_N;
        int gm = row_s + lm, gn = col_s + ln;
        if (gm < M && gn < N) {
            float v = __half2float(smem_out[i]);
            out[gm * N + gn] = HT::from_float(v);
        }
    }
}

// ============================================================================
// C-linkage launch wrappers
// ============================================================================
extern "C" {

void w4a4_gemm_launch_fp16(
    const void* x, const void* W, const float* as, const float* az,
    const void* ws, const float* zb, void* out,
    int M, int N, int K, float qmax, cudaStream_t s)
{
    dim3 g((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    w4a4_gemm_kernel<__half><<<g, 256, 0, s>>>(
        (const __half*)x, (const uint8_t*)W, as, az,
        (const half*)ws, zb, (__half*)out, M, N, K, qmax);
}

void w4a4_gemm_launch_bf16(
    const void* x, const void* W, const float* as, const float* az,
    const void* ws, const float* zb, void* out,
    int M, int N, int K, float qmax, cudaStream_t s)
{
    dim3 g((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    w4a4_gemm_kernel<__nv_bfloat16><<<g, 256, 0, s>>>(
        (const __nv_bfloat16*)x, (const uint8_t*)W, as, az,
        (const half*)ws, zb, (__nv_bfloat16*)out, M, N, K, qmax);
}

} // extern "C"
