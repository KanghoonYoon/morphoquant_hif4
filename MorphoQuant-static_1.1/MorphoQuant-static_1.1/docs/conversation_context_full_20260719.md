# MorphoQuant 完整对话上下文

> 日期: 2026-07-17 ~ 2026-07-19 | 分支: `static_1.1` | 作者: InitBlue + Claude

---

## 目录

1. [问题链](#一问题链)
2. [融合 CUDA Kernel 实现](#二融合-cuda-kernel-实现)
3. [理论 FLOPs 分析](#三理论-flops-分析)
4. [权重保存优化分析](#四权重保存优化分析)
5. [新增/修改文件清单](#五新增修改文件清单)
6. [踩坑记录](#六踩坑记录)
7. [关键命令](#七关键命令)

---

## 一、问题链

### 1.1 当前 MorphoQuant 为什么没有 FLOPs/延迟优势？

**核心结论：Decode 是 memory-bandwidth-bound，不是 compute-bound。**

```
每 token 读取权重:     16,757 MB
L20 HBM 带宽:          864 GB/s
理想读取时间:           19.4 ms
实际 decode 时间:       23.0 ms
带宽利用率:             84%
──────────────────────────────
Compute 占比:           ~16%
Bandwidth 占比:         ~84%
```

compute 只占 decode 时间的 ~16%，所以减少 compute 操作数对延迟收益极小。优化必须从带宽入手。

### 1.2 为什么 INT8 TensorCore 无法加速 matmul？

三个硬件/语义障碍：

| 障碍 | 详情 |
|------|------|
| **M≥16 约束** | `torch._int_mm` 调用 INT8 TensorCore 要求 M ≥ 16。单 token decode 时 M=1，直接无法调用。这是硬件限制，无法绕过。 |
| **语义不兼容** | Per-channel asymmetric quantization (`x_q = clamp(round(x/s+z), 0, 15)`) 无法映射到标准 INT8 GEMM 的对称量化语义。需要额外的 scale/zero 变换才能对齐。 |
| **带宽收益被抵消** | 即使 INT8 权重减半读取代宽，省出的时间被 on-the-fly dequant 计算开销抵消大部分。 |

L20 Compute Capability 8.9 (Ada Lovelace) 理论峰值：
- FP16 TensorCore: ~59.8 TFLOPS
- INT8 TensorCore: ~118.8 TOPS（但在当前场景下完全用不上）

### 1.3 保存校准后权重能优化吗？

- **✅ 启动时间 −85%** (47s → ~5s)：跳过校准 (25s) + 搜索 (8s) + BNB 解量化 (2s) + 模型加载 (5s)
- **❌ 推理时延无变化**：forward pass 的矩阵运算完全一样，权重已在 GPU HBM 中
- 实现方式：`torch.save(model.state_dict(), path)` → 后续 `torch.load()` 直接加载
- INT8 权重存储方向：理论可省 ~5ms decode，但需要手写 fused dequant+GEMM kernel（工程量大，且 M=1 瓶颈不变）

### 1.4 为什么 matmul 无法利用 INT8 TensorCore 加速（补充微基准验证）

微基准测试结果 (`torch._int_mm` vs `torch.matmul` 在 L20 上)：

| M (batch) | K=3584, N=3584 | INT8 路径 | FP16 路径 | 结论 |
|-----------|---------------|-----------|-----------|------|
| 1 | decode | ❌ 直接报错 | 正常 (~0.15ms) | M=1 无法调用 |
| 16 | prefill edge | ~0.18ms | ~0.22ms | INT8 略快但边际收益小 |
| 128 | prefill | ~0.95ms | ~1.35ms | FP16 TensorCore 性能已足够好 |

**额外约束**：`_int_mm` 要求 K 对齐到 32 的整数倍（`_int_mm` 的 K 维度跨步限制），如果不满足需要显式 padding。

---

## 二、融合 CUDA Kernel 实现

### 2.1 动机

当前 PyTorch 路径每层 5-6 个独立 kernel launch：

```
x.fp32 cast → /scale + zero → round + clamp → -zero * scale → outlier bypass → .to(input_dtype)
  launch 1      launch 2         launch 3         launch 4        launch 5a/5b     launch 6
```

422 层 × 6 launch = **~2,500 kernel launch per forward pass**。对于 decode (M=1) 场景，每次操作的数据量极小 (~1280 元素)，kernel launch overhead (5-10μs/次) 成为主要瓶颈。此外还有中间 tensor (x_fp32, x_scaled, x_int, x_dq) 的 global memory 分配/读写开销。

### 2.2 CUDA Kernel: `morpho_act_quant_dequant_kernel`

单 kernel 融合全部 6 步操作：

```c++
// 1D grid, each thread handles one element
// Template on scalar_t with explicit conversion traits
template<typename scalar_t>
__global__ void morpho_act_quant_dequant_kernel(
    const scalar_t* __restrict__ x,          // [M, K] input
    scalar_t* __restrict__ x_dq,             // [M, K] output
    const float* __restrict__ act_scale,     // [K] per-channel scale
    const float* __restrict__ act_zero,      // [K] per-channel zero point
    const bool* __restrict__ outlier_mask,   // [K] or nullptr
    const float* __restrict__ comp_limit,    // [K] or nullptr
    int64_t M, int64_t K, float qmax,
    bool has_outliers
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;

    int64_t col = idx % K;
    float val = HalfTraits<scalar_t>::to_float(x[idx]);

    // Outlier bypass: original value passes through unquantized
    if (has_outliers && outlier_mask[col]) {
        if (fabsf(val) > comp_limit[col]) {
            x_dq[idx] = HalfTraits<scalar_t>::from_float(val);
            return;
        }
    }

    // Asymmetric quant + dequant
    float s = act_scale[col];
    float z = act_zero[col];
    float q = nearbyintf(val / s + z);
    q = fminf(fmaxf(q, 0.0f), qmax);
    float dq = (q - z) * s;

    x_dq[idx] = HalfTraits<scalar_t>::from_float(dq);
}
```

**HalfTraits 模板类**（必须显式转换，因为 PyTorch 编译时启用了 `-D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__`，禁用了隐式类型转换运算符）：

```cpp
template<typename T> struct HalfTraits;

template<> struct HalfTraits<__half> {
    static __device__ __forceinline__ float to_float(__half v)   { return __half2float(v); }
    static __device__ __forceinline__ __half from_float(float v) { return __float2half(v); }
};

template<> struct HalfTraits<__nv_bfloat16> {
    static __device__ __forceinline__ float to_float(__nv_bfloat16 v)       { return __bfloat162float(v); }
    static __device__ __forceinline__ __nv_bfloat16 from_float(float v)     { return __float2bfloat16(v); }
};

template<> struct HalfTraits<float> {
    static __device__ __forceinline__ float to_float(float v)   { return v; }
    static __device__ __forceinline__ float from_float(float v) { return v; }
};
```

三个 C-linkage launch wrapper:
- `morpho_cuda_launch_fp16` — FP16 kernel
- `morpho_cuda_launch_bf16` — BF16 kernel
- `morpho_cuda_launch_fp32` — FP32 kernel (debug/verification)

### 2.3 JIT 编译

使用 `torch.utils.cpp_extension.load_inline()` 在首次 import 时编译：

```python
def _load_cuda_module():
    """Lazy JIT compilation via load_inline. Returns None on failure (graceful fallback)."""
    return load_inline(
        name="morpho_quant",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
```

编译失败时自动 fallback 到 PyTorch element-wise path（#define `TORCH_FALLBACK`）。

### 2.4 Python Wrapper

```python
def fused_act_quant_dequant(
    x: torch.Tensor,              # [..., K] fp16/bf16/fp32
    act_scale: torch.Tensor,      # [K] fp32
    act_zero: torch.Tensor,       # [K] fp32
    qmax: float,
    outlier_mask: Optional[torch.Tensor] = None,   # [K] bool
    comp_limit: Optional[torch.Tensor] = None,     # [K] fp32
) -> torch.Tensor:
    """Flattens leading dims → [M, K], dispatch to CUDA kernel, reshape back."""
```

### 2.5 MorphoFusedLinear 中的使用

```python
class MorphoFusedLinear(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._cuda_available:
            x_dq = fused_act_quant_dequant(
                x, self.act_scale, self.act_zero, self.qmax,
                self._outlier_mask, self.compensation_limit)
        else:
            x_dq = self._fallback_act_quant(x)
        return torch.matmul(x_dq, self.W_fp16_cache.T)
```

### 2.6 旁路 Bias 补偿机制

参考 `bnb_src/bitsandbytes/quantization_utils/quant_modules.py:960-998` (`QuantAct.forward` inference stage)：

```
Step 1 — 全通道量化:
    x_q[ch] = clamp(round(x[ch] / scale[ch] + zero[ch]), 0, 2^bits - 1)
    x_dq[ch] = (x_q[ch] - zero[ch]) * scale[ch]

Step 2 — 稀疏补偿判定:
    compensation_limit[ch] = max(|best_max[ch]|, |best_min[ch]|) * sparse_buffer_ratio
    sparse_mask[ch] = (|x[ch]| > compensation_limit[ch]) AND outlier_mask[ch]

Step 3 — 旁路替换:
    output[ch] = sparse_mask[ch] ? x_original[ch] : x_dq[ch]
```

**Bias 补偿的数学解释**：对于被 bypass 的离群通道，其对矩阵乘结果的贡献为：

```
正常通道:  y += x_dq[ch] × W[:, ch]            ← 量化值
旁路通道:  y += x_original[ch] × W[:, ch]        ← 原始值 (bypass 直通)

等价于在量化路径上叠加一个补偿 bias:
    bias_ch = (x_original[ch] − x_dq[ch]) × W[:, ch]
```

CUDA kernel 中直接做 in-place replacement（`x_dq[ch] = x_original[ch]`），后续单次 matmul 自动包含补偿效应，无需额外的 bias tensor。

### 2.7 Benchmark 结果

#### Micro-benchmark

| 指标 | PyTorch (element-wise) | CUDA Kernel | 加速比 |
|------|----------------------|-------------|--------|
| 单层 act quant (M=1, K=3584) | 55-167 μs | **7 μs** | **7.9-23.7×** |
| 单层 act quant (M=128, K=3584) | 60-80 μs | **8 μs** | **7.5-10×** |
| 422 层总计 (M=1) | ~23-70 ms | **~3 ms** | **~8-23×** |

#### End-to-end

| 配置 | TTFT (128 tok) | Decode (ms/tok) | Prefill | vs FP16 |
|------|:---:|:---:|:---:|:---:|
| FP16 baseline | ~47 ms | ~23 ms | ~9.2 ms | — |
| Morpho (BNB 4-bit + QuantAct) | ~180 ms | ~55 ms | ~30 ms | +283% |
| Morpho Fused (PyTorch element-wise) | 86.9 ms | 36.8 ms | ~18 ms | +85% |
| **Morpho CUDA** | **46.4 ms** | **~36 ms** | **~9.7 ms** | **−1% / +57% / +5%** |

Morpho CUDA TTFT 已追平 FP16 baseline（prefill 仅慢 5%），但 decode 仍有 ~57% 差距（根因：权重带宽瓶颈——MorphoFusedLinear 的 `W_fp16_cache` 和 FP16 baseline 的权重读取代宽完全一样）。

---

## 三、理论 FLOPs 分析

### 3.1 模型维度 (Qwen2.5-Omni-7B Thinker)

来自 `thinker_config.text_config`：

| 参数 | 值 |
|------|-----|
| hidden_size | 3,584 |
| intermediate_size | 18,944 |
| num_hidden_layers | 28 |
| num_attention_heads | 28 |
| num_key_value_heads | 4 |
| vocab_size | 152,064 |

### 3.2 单层 Linear 矩阵维度

| 层 | 维度 [K, N] | 权重元素数 | 每 token MACs (M=1) |
|----|------------|----------|-------------------|
| q_proj | [3584, 3584] | 12,845,056 | 12,845,056 |
| k_proj | [3584, 512] | 1,835,008 | 1,835,008 |
| v_proj | [3584, 512] | 1,835,008 | 1,835,008 |
| o_proj | [3584, 3584] | 12,845,056 | 12,845,056 |
| gate_proj | [3584, 18944] | 67,895,296 | 67,895,296 |
| up_proj | [3584, 18944] | 67,895,296 | 67,895,296 |
| down_proj | [18944, 3584] | 67,895,296 | 67,895,296 |
| **每层合计** | | **233,046,016** | **233,046,016** |

### 3.3 每 Token Decode FLOPs 详细计算

**FLOPs = 2 × MACs** (每个 MAC = 1 multiply + 1 add)：

```
FP16 Baseline per token (decode, M=1):

q_proj:       2 × 1 × 3584 × 3584   =     25,690,112
k_proj:       2 × 1 × 3584 × 512    =      3,670,016
v_proj:       2 × 1 × 3584 × 512    =      3,670,016
o_proj:       2 × 1 × 3584 × 3584   =     25,690,112
gate_proj:    2 × 1 × 3584 × 18944  =    135,790,592
up_proj:      2 × 1 × 3584 × 18944  =    135,790,592
down_proj:    2 × 1 × 18944 × 3584  =    135,790,592
                                    ─────────────
Per-layer:                              466,092,032  (≈ 0.466 GFLOPs)
28 layers:                           13,050,576,896  (≈ 13.05 GFLOPs)
lm_head (unquantized):                1,090,076,672  (≈  1.09 GFLOPs)
                                    ─────────────
Total decode FLOPs/token:            14,140,653,568  (≈ 14.14 GFLOPs)
```

**Prefill (M=128)：**

```
Per-layer prefill:     128 × 466,092,032  = 59,659,780,096  (≈ 59.66 GFLOPs)
28 layers:             28 × 59.66         = 1,670.47 GFLOPs  (≈ 1.67 TFLOPs)
```

### 3.4 MorphoQuant vs FP16：FLOPs 对比

```
                    Operational FLOPs    BOPs (bit-operations)   Weight BW
                    (操作次数)             (比特操作)               (权重带宽)
────────────────────────────────────────────────────────────────────────────
FP16 Baseline           14.14 G              ~7.24 Tbit²          16.7 GB
MorphoQuant 理论        14.14 G              ~0.45 Tbit²           4.2 GB
MorphoQuant 实际        14.14 G              ~7.24 Tbit²          16.7 GB
────────────────────────────────────────────────────────────────────────────
变化                      0%                理论 −93.75%           理论 −75%
                                                                   实际    0%
```

### 3.5 核心结论

```
┌──────────────────────────────────────────────────────────┐
│               FLOPs 降低百分比                              │
│                                                          │
│  操作次数 (Operational FLOPs):                     0%     │
│  → 矩阵维度 M×K×N 决定，与比特位宽无关                      │
│                                                          │
│  比特操作 (BOPs, 理论 W4A4):                    −93.75%   │
│  → BOPs = 2×M×K×N × bit(A) × bit(W)                    │
│  → FP16: 512×M×K×N  vs  INT4: 32×M×K×N                │
│                                                          │
│  权重内存带宽 (理论 INT4 存储):                   −75%     │
│  → FP16: 2B/elem  vs  INT4: 0.5B/elem                  │
│                                                          │
│  实际有效降低 (当前实现):                          0%     │
│  → matmul 仍是 FP16，权重缓存为 FP16                      │
│  → 节省来自 kernel fusion (延迟优化，非 FLOPs 优化)         │
└──────────────────────────────────────────────────────────┘
```

**FLOPs 从来不降，也降不了。** 原因是：
1. FLOPs 是操作次数，由 `M × K × N` 决定，和用 float16 还是 int4 做乘法无关
2. MorphoQuant（以及几乎所有 W4A16/W4A8 方案）的价值在于**精度降低和带宽节省**，不是操作次数减少
3. 量化论文中声称的 "4×/16× 计算量降低" 通常指 BOPs（比特操作量）或等效吞吐量，不是 FLOPs
4. 当前实现中连带宽节省也没兑现——`W_fp16_cache` 是 FP16 格式，和 baseline 读取代宽完全相同

### 3.6 为什么理论收益无法兑现

| 理论优势 | 障碍 | 是否可解 |
|----------|------|---------|
| BOPs −93.75% | 需要 INT4 TensorCore，但 M≥16 限制使 decode 无法使用 | ❌ 硬件限制 |
| 带宽 −75% | 需要保持 INT4 权重并在 forward 中 on-the-fly dequant | ⚠️ 可行但需手写 fused dequant+GEMM kernel |
| ACT quant latency ↓ | CUDA kernel 融合已实现 | ✅ 已解决 |

### 3.7 三个 FLOPs 指标的形式化定义

```
Operational FLOPs (操作次数):
    FLOPs = Σ_l 2 × M_l × K_l × N_l
    → 不管精度，只数乘法和加法次数
    → 这个值在量化前后不变

BOPs (比特操作量):
    BOPs = Σ_l 2 × M_l × K_l × N_l × b_A × b_W
    → b_A, b_W = 激活/权重的比特位宽
    → FP16→INT4 降低 (16×16)/(4×4) = 16×

Memory-Bandwidth FLOPs (等效带宽操作):
    FLOPs_BW = Σ_l (b_W/8) × K_l × N_l  [bytes read per token]
    → FP16:    2 bytes/elem  × Σ(K_l×N_l)
    → INT4:   0.5 bytes/elem × Σ(K_l×N_l)
    → 降低 75%
```

---

## 四、权重保存优化分析

### 4.1 当前每轮 Benchmark 的启动时间分解

```
加载 BF16 模型 (5 个 4GB shard):         5s
BNB 4-bit 量化 (422 层):                  1.5s
校准 (128 forward passes):               25s   ← 可跳过
搜索 (16 forward passes):                 8s   ← 可跳过
BNB 4→FP16 解量化 (422 层):               2s
构建 422 个 MorphoFusedLinear:           1s
CUDA kernel JIT 编译 (首次):              5s   ← 仅首次
─────────────────────────────────────────────
总计每轮:                                ~47s
```

### 4.2 优化方案

```
当前流程 (每轮 benchmark 重复):
  HF shards → BNB量化 → 校准(128步) → 搜索(16步) → 解量化 → 构建融合层
  ───────────────────────────── 47s ─────────────────────────────

优化后流程 (校准一次，保存，后续直接加载):
  [离线一次] HF shards → BNB量化 → 校准 → 搜索 → 解量化 → torch.save()
  [在线每次] torch.load() → 构建融合层 (或者 torch.load 含完整模型)
  ───── 5s ─────
```

### 4.3 实现方式

```python
# 离线：校准完成后保存
torch.save(model.state_dict(), "/path/to/morpho_quantized.pt")

# 在线：直接加载
model = create_empty_morpho_model(config)  # 只建壳
model.load_state_dict(torch.load("/path/to/morpho_quantized.pt"))
```

每个 `MorphoFusedLinear` 的 state_dict 已包含：
- `W_fp16_cache` — 解量化后的完整 FP16 权重 (12.6 GB)
- `act_scale` / `act_zero` — 逐通道 INT4 量化参数
- `_outlier_mask` / `compensation_limit` — 旁路稀疏补偿参数

### 4.4 效果评估

| | 启动时间 | 推理时延 |
|---|---|---|
| 保存校准后权重 | ✅ **−85%** (47s → 5s) | 无变化 |
| INT8 权重存储 | 无影响 | 理论可省 ~5ms，但需手写 GEMM kernel |
| 当前状态 | 每轮 47s 校准开销 | TTFT 46ms (距 FP16 +5%) |

保存权重优化的是**开发迭代效率**，不改变推理时延——因为 forward pass 中的矩阵运算是完全一样的。

---

## 五、新增/修改文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `modules/morpho_cuda_kernel.py` | **NEW** | CUDA kernel + C++ bindings + Python wrapper |
| `modules/morpho_fused_linear.py` | **NEW** | MorphoFusedLinear 替代 BNB 4bit + QuantAct |
| `modules/latency_probe.py` | **NEW** | CUDA Event 级延迟测量 (LatencyProbe 类) |
| `scripts/run_latency_bench.py` | **NEW** | 统一 benchmark 脚本 (支持 multi-config 对比) |
| `configs/latency_bench/fp16.yaml` | **NEW** | FP16 baseline benchmark 配置 |
| `configs/latency_bench/morpho.yaml` | **NEW** | Morpho (BNB 4bit + QuantAct) 原始路径配置 |
| `configs/latency_bench/morpho_fused.yaml` | **NEW** | MorphoFused (PyTorch element-wise) 配置 |
| `configs/latency_bench/morpho_cuda.yaml` | **NEW** | Morpho CUDA kernel 配置 |
| `config.py` | **MODIFY** | BenchmarkConfig 新增 `fused_kernel`/`use_cuda_kernel` |

### 5.1 文件依赖关系

```
config.py ─────────────────────────────────────────────┐
    │                                                   │
    ▼                                                   │
modules/model_factory.py ──► modules/morpho_fused_linear.py
                                   │
                                   ▼
                            modules/morpho_cuda_kernel.py
                                   │
                                   ▼ (JIT compile)
                            ~/.cache/torch_extensions/morpho_quant/
                                   │
                                   ▼
                            load_inline() → CUDA .so
```

---

## 六、踩坑记录

| # | 问题 | 原因 | 解决 |
|---|------|------|------|
| 1 | `ninja: command not found` | `load_inline` 需要 ninja 构建系统 | `pip install ninja` |
| 2 | `duplicate PYBIND11_MODULE` | `load_inline` 的 `functions` 参数会自动生成 Python bindings，与手写的 `PYBIND11_MODULE` 冲突 | 移除 `functions` 参数，完全由手动 binding 接管 |
| 3 | `__half` / `__nv_bfloat16` 隐式转换编译失败 | PyTorch 构建时启用了 `-D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__`，删除了隐式 conversion operator | 创建 `HalfTraits<T>` 模板结构体，使用显式 intrinsics (`__half2float`、`__float2half`、`__bfloat162float`、`__float2bfloat16`) |
| 4 | `ModuleNotFoundError: No module named 'accelerate'` | 系统 Python 缺少项目依赖 | 使用 conda 环境的 Python：`/opt/conda/envs/MorphoQuant/bin/python` |
| 5 | GPU 0-3 被占用 | CLAUDE.md 规定只能用 GPU 4/5/6 | 所有命令加 `CUDA_VISIBLE_DEVICES=4,5,6` |
| 6 | `_int_mm` 返回空结果 | M < 16 时 cuBLAS INT8 路径直接返回未经初始化的 tensor | M < 16 场景完全跳过 `_int_mm`，使用标准 FP16 matmul |

---

## 七、关键命令

### Benchmark

```bash
# 四路完整对比
CUDA_VISIBLE_DEVICES=4 python scripts/run_latency_bench.py \
  --configs configs/latency_bench/fp16.yaml \
            configs/latency_bench/morpho.yaml \
            configs/latency_bench/morpho_fused.yaml \
            configs/latency_bench/morpho_cuda.yaml \
  --output /private/wy/logs/latency_bench/four_way_comparison.json

# Quick 模式 (仅最短输入长度 128)
CUDA_VISIBLE_DEVICES=4 python scripts/run_latency_bench.py \
  --config configs/latency_bench/morpho_cuda.yaml --quick

# 原始 benchmark (MMMU, ScienceQA, etc.)
python wy_inference_mmmu.py --config configs/qwen2.5-omni-7b/morpho/mmmu_morpho.yaml
```

### 快捷验证

```bash
# 检查 CUDA kernel 是否能编译
python -c "
import torch, sys
sys.path.insert(0, '/private/wy/MorphoQuant')
from modules.morpho_cuda_kernel import is_cuda_available
print('CUDA kernel available:', is_cuda_available())
"

# Smoke test (不需加载大模型)
python tests/smoke_morpho_internvl.py
```

---

## 附录：关键代码位置索引

| 组件 | 文件 | 行号 |
|------|------|------|
| 稀疏补偿机制 (reference) | `bnb_src/.../quant_modules.py` | 960-998 |
| MorphoFusedLinear.forward() | `modules/morpho_fused_linear.py` | forward 方法 |
| CUDA kernel (GPU 代码) | `modules/morpho_cuda_kernel.py` | `_CUDA_SRC` 常量 |
| Python wrapper | `modules/morpho_cuda_kernel.py` | `fused_act_quant_dequant()` |
| BenchmarkConfig | `config.py` | 92-101 |
| 跳过层名单 | `modules/model_factory.py` | `_SKIP_LOCAL_NAMES` |
| Qwen2.5-Omni-7B text config | `pretrained_models/.../config.json` | `thinker_config.text_config` |
