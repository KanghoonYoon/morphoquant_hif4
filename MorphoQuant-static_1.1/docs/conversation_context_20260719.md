# MorphoQuant CUDA Kernel — 对话上下文记录

> 日期: 2026-07-17 ~ 2026-07-19 | 分支: `static_1.1`

## 一、问题链

### 1. 当前 MorphoQuant 为什么没有 FLOPs/延迟优势？

**核心结论：Decode 是 memory-bandwidth-bound，不是 compute-bound。**

| 指标 | 数据 |
|------|------|
| 每 token 读取权重 | 16,757 MB |
| L20 HBM 带宽 | 864 GB/s |
| 理想读取时间 | 19.4 ms |
| 实际 decode 时间 | 23.0 ms |
| **带宽利用率** | **84%** |

compute 只占 decode 时间的 ~16%，所以减少 compute 对延迟收益极小。

### 2. 为什么 INT8 TensorCore 无法加速 matmul？

三个原因：

1. **M≥16 约束**：`_int_mm` 要求 M ≥ 16，单 token decode 时 M=1，无法调用
2. **语义不兼容**：per-channel asymmetric quantization（`x_q = clamp(round(x/s+z), 0, qmax)`）与标准 INT8 GEMM 的对称量化语义不兼容
3. **带宽收益有限**：即使 INT8 权重减半读取代宽，省出的时间被 on-the-fly dequant 开销抵消大半

L20 Compute Capability 8.9 (Ada Lovelace):
- FP16 TensorCore: ~59.8 TFLOPS
- INT8 TensorCore: ~118.8 TOPS（但用不上）

### 3. 保存校准后权重能优化吗？

- **✅ 启动时间 −85%**（47s → 5s）：跳过校准 (25s) + 搜索 (8s) + BNB 解量化 (2s) + HF 加载 (5s)
- **❌ 推理时延无变化**：forward pass 完全一样，权重已在 GPU HBM 中
- 实现方式：`torch.save(model.state_dict(), path)` → 后续 `torch.load()` 直接加载
- INT8 权重存储方向：理论可省 ~5ms decode，但需要手写 GEMM kernel（工程量大）

---

## 二、融合 CUDA Kernel 实现

### 动机

当前 PyTorch 路径每层 5-6 个 kernel launch：
```
x.fp32 cast → /scale + zero → round + clamp → -zero * scale → outlier bypass → .to(input_dtype)
```

422 层 × 6 launch = **~2,500 kernel launch/forward**，每次 5-10μs launch overhead。

### 实现：`morpho_act_quant_dequant_kernel`

单 kernel 融合上述全部 6 步操作：

```c++
// 1D grid, per-element thread
__global__ void morpho_act_quant_dequant_kernel(
    const scalar_t* x, scalar_t* x_dq,
    const float* act_scale, const float* act_zero,
    const bool* outlier_mask, const float* comp_limit,
    int64_t M, int64_t K, float qmax, bool has_outliers
)
```

关键技术细节：
- **HalfTraits 模板**：使用显式转换 intrinsics (`__half2float` / `__float2half` / `__bfloat162float` / `__float2bfloat16`)，因为 PyTorch 编译时启用了 `-D__CUDA_NO_HALF_CONVERSIONS__`
- **Outlier bypass**：`outlier_mask[ch]` boolean 数组 O(1) 查找，`compensation_limit[ch]` 阈值比较
- **JIT 编译**：`torch.utils.cpp_extension.load_inline()` 首次运行时编译，编译失败自动 fallback 到 PyTorch path

### 旁路 Bias 补偿机制

参考 `bnb_src/bitsandbytes/quantization_utils/quant_modules.py:960-998`：

```
正常通道:  y += x_dq[ch] * W[:, ch]           ← 量化值
旁路通道:  y += x_original[ch] * W[:, ch]       ← 原始值 (bypass)

等价 bias: bias_ch = (x_original[ch] - x_dq[ch]) * W[:, ch]
```

CUDA kernel 中直接做 in-place replacement（`x_dq[ch] = x_original[ch]`），后续单次 matmul 自动包含补偿。

### Micro-benchmark 结果

| 指标 | PyTorch (element-wise) | CUDA Kernel | 加速比 |
|------|----------------------|-------------|--------|
| 单层 act quant (M=1) | 55-167 μs | **7 μs** | **7.9-23.7×** |
| 单层 act quant (M=128) | 60-80 μs | **8 μs** | **7.5-10×** |

### End-to-end 结果

| 配置 | TTFT (128 tokens) | Decode (ms/tok) | Prefill |
|------|-------------------|-----------------|---------|
| FP16 baseline | ~47 ms | ~23 ms | ~9.2 ms |
| Morpho Fused (PyTorch) | 86.9 ms | 36.8 ms | ~18 ms |
| **Morpho CUDA** | **46.4 ms** | **~36 ms** | **~9.7 ms** |

Morpho CUDA TTFT 比 PyTorch 版快 1.9×，prefill 距 FP16 仅 5%。

---

## 三、新增/修改文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `modules/morpho_cuda_kernel.py` | NEW | CUDA kernel + C++ bindings + Python wrapper |
| `modules/morpho_fused_linear.py` | NEW | MorphoFusedLinear 替代 BNB 4bit + QuantAct |
| `modules/latency_probe.py` | NEW | CUDA Event 级延迟测量 (LatencyProbe) |
| `scripts/run_latency_bench.py` | NEW | 统一 benchmark 脚本 (支持 multi-config 对比) |
| `configs/latency_bench/fp16.yaml` | NEW | FP16 baseline 配置 |
| `configs/latency_bench/morpho.yaml` | NEW | Morpho (current) 配置 |
| `configs/latency_bench/morpho_fused.yaml` | NEW | MorphoFused (PyTorch) 配置 |
| `configs/latency_bench/morpho_cuda.yaml` | NEW | Morpho CUDA kernel 配置 |
| `config.py` | MODIFY | BenchmarkConfig 新增 `fused_kernel`/`use_cuda_kernel` |

---

## 四、踩坑记录

1. **ninja 未安装** → `pip install ninja`
2. **duplicate PYBIND11_MODULE** → `load_inline` 的 `functions` 参数自动生成 bindings，需移除手动 `PYBIND11_MODULE`
3. **`__CUDA_NO_HALF_CONVERSIONS__`** → PyTorch 禁用隐式 conversion operator，需用显式 intrinsics（HalfTraits 模板）
4. **Python 环境** → 必须用 `/opt/conda/envs/MorphoQuant/bin/python`（conda 环境含 accelerate 等依赖）
5. **GPU 限制** → CLAUDE.md 要求只能用 GPU 4/5/6

---

## 五、关键命令

```bash
# 四路对比 benchmark
CUDA_VISIBLE_DEVICES=4 python scripts/run_latency_bench.py \
  --configs configs/latency_bench/fp16.yaml \
            configs/latency_bench/morpho.yaml \
            configs/latency_bench/morpho_fused.yaml \
            configs/latency_bench/morpho_cuda.yaml

# Quick 模式（仅最短输入长度）
CUDA_VISIBLE_DEVICES=4 python scripts/run_latency_bench.py \
  --config configs/latency_bench/morpho_cuda.yaml --quick
```
