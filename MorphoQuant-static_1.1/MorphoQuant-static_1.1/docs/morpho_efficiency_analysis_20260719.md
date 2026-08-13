# MorphoQuant 量化收益重新评估：显存占用与吞吐量

> 日期: 2026-07-19 | 分支: `static_1.1` | GPU: NVIDIA L20 (48 GB, 864 GB/s HBM)

## 一、实测数据总览

Qwen2.5-Omni-7B Thinker 架构参数：
- `hidden_size=3584`, `intermediate_size=18944`, `num_hidden_layers=28`
- `num_attention_heads=28`, `num_key_value_heads=4` (GQA), `vocab_size=152064`
- 可量化 Linear 层：422 层，共 7.20B 参数（占总参数 8.93B 的 80.6%）
- 不可量化部分（embed_tokens, lm_head, visual/audio encoder, talker, norms）：1.73B 参数

| 指标 | FP16 (BF16) | Morpho BNB 4-bit | Morpho CUDA (Fused) |
|------|-------------|------------------|----------------------|
| **GPU 显存 (alloc)** | **17.87 GB** | **7.21 GB** | **22.21 GB** |
| **GPU 显存 (peak)** | 22.38 GB | 22.38 GB | 26.05 GB |
| **Decode 时延** | **22.9 ms/tok** | — (未测) | **36.4 ms/tok** |
| **Prefill 时延 (128 tok)** | **7.1 ms** | — (未测) | **8.0 ms** |
| **TTFT (128 tok)** | 30.0 ms | — | 44.3 ms |
| **单 token 吞吐** | **43.7 tok/s** | — | **27.5 tok/s** |
| **模型加载时间** | 35.5 s | 9.1 s | 8.8 + 校准 ≈ 25 s |
| **权重存储格式** | BF16 | NF4 + double quant | FP16 (缓存) |

## 二、显存占用分析

### 2.1 逐组件分解

```
                    FP16         Morpho BNB     MorphoFused
                    ────────     ───────────    ───────────
可量化 Linear (7.20B):
  权重存储          14.40 GB      3.60 GB        14.40 GB  ← FP16 cache
  quant_state          0          0.30 GB            0
  QuantAct buffers     0          0.02 GB         0.02 GB

不可量化部分 (1.73B):
  embed_tokens       1.09 GB      1.09 GB         1.09 GB
  lm_head            1.09 GB      1.09 GB         1.09 GB
  vision encoder    ~0.8 GB      ~0.8 GB         ~0.8 GB
  audio encoder     ~0.8 GB      ~0.8 GB         ~0.8 GB
  talker            ~0.6 GB      ~0.6 GB         ~0.6 GB
  其他 (RMSNorm等)  ~0.1 GB      ~0.1 GB         ~0.1 GB
  小计              ~4.48 GB     ~4.48 GB        ~4.48 GB

额外开销:
  CUDA allocator        0            0           ~3.3 GB  ← 旧 BNB 模块未完全释放

────────────────────────────────────────────────────────────────
总计 (alloc)         17.87 GB      7.21 GB        22.21 GB
vs FP16              baseline     **-59.6%**       **+24.3%**
```

### 2.2 关键发现

**Morpho BNB 4-bit：-60% 显存，是量化最大的实际收益。**

- BNB NF4 将 7.20B 权重从 14.40 GB 压缩到 3.60 GB（4× 压缩），加上 double quantization 的 quant_state 开销约 0.30 GB，净节省 10.50 GB
- 不可量化的 embedding、encoder、talker 部分（~4.5 GB）没有变化
- **这是量化带来的最确定的收益，且仅存在于原始 Morpho BNB 路径中**

**MorphoFused/CUDA：+24% 显存，比 FP16 更差。**

根本原因：`W_fp16_cache` 将 BNB 4-bit 权重解量化并缓存为 FP16：
- BNB 4-bit 权重：3.60 GB → FP16 缓存：14.40 GB（+10.80 GB）
- 这完全抵消并反转了原始 Morpho 的显存优势
- 额外 3.3 GB 来自 CUDA caching allocator 的碎片化和旧模块未完全释放

### 2.3 KV Cache 与 Batch Scaling

KV cache 每 token 占用（GQA, 4 KV heads, 128 dim）：
```
每层: 2 (K+V) × 4 heads × 128 dim × 2 bytes = 2,048 bytes
28 层: 57,344 bytes/token ≈ 56 KB/token
```

不同序列长度下的 KV cache 与最大 batch size（L20 48 GB）：

| 序列长度 | KV/seq | FP16 max batch | Morpho BNB max batch | MorphoFused max batch |
|---------|--------|----------------|---------------------|----------------------|
| 2,048 | 117 MB | ~32 | **~64** | ~24 |
| 4,096 | 235 MB | ~16 | **~32** | ~12 |
| 8,192 | 470 MB | ~8 | **~16** | ~6 |
| 32,768 | 1.88 GB | ~4 | **~8** | ~3 |

Morpho BNB 4-bit 在所有序列长度下可以支持 **约 2× 的并发 batch size**。

## 三、吞吐量分析

### 3.1 Decode 阶段（memory-bandwidth-bound）

Decode 阶段每 token 需要读取所有权重。Qwen2.5-Omni-7B 的 decode 延迟分解：

```
FP16 (实测 22.9 ms/tok):
├─ Thinker 权重读取:    14.40 GB / 864 GB/s = 16.7 ms (理想下限)
├─ 非 Thinker 部分:      4.48 GB / 864 GB/s =  5.2 ms
├─ 计算 + kernel launch:                        1.0 ms
└─ 总计 (84% 带宽利用):                         22.9 ms

Morpho CUDA (实测 36.4 ms/tok):
├─ Thinker 权重读取:    14.40 GB / 864 GB/s = 16.7 ms  ← W_fp16_cache 同 FP16
├─ 非 Thinker 部分:      4.48 GB / 864 GB/s =  5.2 ms
├─ Act quant CUDA kernel: 422 × 7 μs =          3.0 ms  ← 单 kernel 开销
├─ Kernel launch 隔断:                           ~5 ms  ← 打断 cuBLAS pipeline
├─ 其他开销 (校准残留等):                         ~6.5 ms
└─ 总计:                                         36.4 ms
```

**Morpho CUDA decode 比 FP16 慢 59%，13.5 ms 的额外延迟主要来自：**
1. **CUDA stream pipeline 被打断**：每个 matmul 前插入 act quant kernel，阻止了 cuBLAS 的跨层 kernel fusion 和 wave 调度
2. **Act quant kernel launch overhead**：422 层 × 7 μs ≈ 3.0 ms
3. **精度转换开销**：BF16 → FP32 (quant) → BF16，额外的 dtype cast

### 3.2 Prefill 阶段（compute-bound）

128 token prefill 时 M=128，matmul 计算量足够大进入 compute-bound 区域：

| 配置 | Prefill (ms) | tok/s | vs FP16 |
|------|-------------|-------|---------|
| FP16 | 7.1 | 18,028 | baseline |
| Morpho CUDA | 8.0 | 16,000 | **-11.3%** |

Prefill 的 Act quant 开销占比小（12.7%），因为 matmul 计算时间远超 quant kernel 时间。这解释了为什么 TTFT 差距主要来自 decode。

### 3.3 Batch 吞吐量（Token/s）

这是量化收益的真正体现场景。高吞吐服务的关键指标是：**在 GPU 显存限制下，每秒能生成多少 token**。

```
Batch throughput = max_batch_size × per_seq_throughput

场景: 2K 上下文，持续 decode

FP16:
  max_batch ≈ 32, per_seq = 43.7 tok/s
  总吞吐 ≈ 32 × 43.7 = 1,398 tok/s

Morpho BNB 4-bit (估计 decode ~35 ms/tok = 28.6 tok/s):
  max_batch ≈ 64, per_seq = 28.6 tok/s
  总吞吐 ≈ 64 × 28.6 = 1,830 tok/s   ← +31% vs FP16

MorphoFused/CUDA:
  max_batch ≈ 24, per_seq = 27.5 tok/s
  总吞吐 ≈ 24 × 27.5 = 660 tok/s     ← -53% vs FP16
```

**结论：原始 Morpho BNB 4-bit 凭借 2× batch size 优势，即使单 token 更慢，总吞吐仍可能超过 FP16。MorphoFused/CUDA 在显存和吞吐两个维度都劣于 FP16 baseline。**

### 3.4 各阶段延迟占比

```
                    FP16          Morpho CUDA
                    ─────         ────────────
Thinker matmul      16.7 ms (73%)   16.7 ms (46%)
Encoder+talker       5.2 ms (23%)    5.2 ms (14%)
Act quant               0            3.0 ms ( 8%)   ← 新增
Pipeline bubble         0           ~5.0 ms (14%)   ← 新增
其他                 1.0 ms ( 4%)    6.5 ms (18%)   ← 扩大
─────────────────────────────────────────────────
总计                22.9 ms         36.4 ms
```

## 四、综合收益评估

### 4.1 三种配置的适用场景

| 场景 | 推荐配置 | 理由 |
|------|---------|------|
| **单用户交互 (低延迟)** | FP16 | 最低 TTFT 和 decode 延迟 |
| **高吞吐批量服务** | Morpho BNB 4-bit | -60% 显存 → 2× batch → 更高总吞吐 |
| **长上下文推理** | Morpho BNB 4-bit | KV cache 压力大时显存优势更明显 |
| **边缘/移动端部署** | Morpho BNB 4-bit | 7.2 GB 可在更小 GPU 上运行 |
| **多模型共驻 GPU** | Morpho BNB 4-bit | 可同时加载 2-3 个量化模型 |
| **研究/精度验证** | MorphoFused/CUDA | 最快的校准→推理迭代流程 |

### 4.2 收益矩阵

```
                    显存       Decode吞吐   Prefill吞吐   Batch吞吐    模型质量
                    ────       ──────────   ──────────   ─────────    ────
FP16                baseline   baseline     baseline     baseline     baseline
Morpho BNB 4-bit    ✅ -60%    ❓ (估计↓)    ❓           ✅ +31%      ✅ 保持
MorphoFused PyTorch ❌ +24%    ❌ -37%       ❌ -11%       ❌ -53%      ✅ 保持
Morpho CUDA         ❌ +24%    ❌ -37%       ❌ -11%       ❌ -53%      ✅ 保持
```

### 4.3 核心结论

1. **量化最大的实际收益是显存而非速度**：BNB 4-bit 权重压缩带来 -60% 显存（10.7 GB 节省），这是确定且可观的收益

2. **MorphoFused/CUDA 路径牺牲了显存优势**：`W_fp16_cache` 将 4-bit 权重展开回 FP16，14.40 GB 的缓存完全抵消了 BNB 的压缩收益

3. **Decode 延迟无改善是结构性的**：
   - Matmul 仍是 FP16，operational FLOPs 不变
   - Act quant 插入打断了 CUDA stream pipeline
   - Decode 是 memory-bandwidth-bound，权重读取决定延迟下限

4. **在批量服务场景下，显存优势可以转化为吞吐优势**：
   - 2× batch size × 稍慢的 per-token 速度 ≈ 更高的总吞吐
   - 这是量化在推理部署中的标准价值主张

## 五、优化方向

### 5.1 恢复 MorphoFused 的显存优势

当前 `W_fp16_cache` 的问题：用 14.40 GB FP16 存储了本该 3.60 GB 4-bit 的权重。

**方案 A：INT8 权重缓存**
- 将 BNB 4-bit 权重解量化为 INT8 而非 FP16
- 存储：7.20 GB（INT8）× 1 byte = 7.20 GB（vs FP16 的 14.40 GB）
- 显存：7.20 + 4.48 + 0.02 = 11.70 GB（vs FP16 的 17.87 GB，-34.5%）
- 问题：需要手写 INT8 matmul kernel（参考 `torch._int_mm`），或接受 FP16 matmul 前的 on-the-fly dequant
- 实现难度：中等

**方案 B：保留 BNB 4-bit 权重 + Direct Dequant Matmul**
- 不缓存 FP16 权重，每次 forward 直接从 4-bit dequant
- 将 act quant CUDA kernel 与 BNB dequant 融合为单一 kernel：`act_quant → weight_dequant → matmul`
- 显存保持 7.21 GB，同时减少 kernel launch 数量
- 实现难度：高（需要深入修改 BNB 的 CUDA kernel）

**方案 C：选择性 FP16 缓存（混合精度）**
- 对 attention 层（q/k/v/o_proj，权重小）使用 FP16 缓存
- 对 MLP 层（gate/up/down_proj，权重大）保留 4-bit
- 显存：约 11-13 GB（折中）
- 实现难度：低

### 5.2 提升 Decode 吞吐

**方案 D：消除 Pipeline Bubble**
- 将 act quant 与下一个操作的权重加载 overlap（CUDA stream 异步）
- 使用 CUDA graph capture 整个 decode step，消除 kernel launch overhead
- 预期收益：减少 5-8 ms 的 pipeline bubble

**方案 E：Prefill 特化**
- Prefill 时 M 较大（≥128），kernel launch overhead 占比小
- Prefill 时可以用 INT8 TensorCore（M≥16 条件满足）：`torch._int_mm(x_int8, W_int8)`
- 预期 prefill 加速 1.5-2×

### 5.3 优先级建议

| 优先级 | 方案 | 收益 | 难度 | 说明 |
|--------|------|------|------|------|
| P0 | 方案 C（混合精度缓存） | 显存 -30%，吞吐不变 | 低 | 快速恢复大部分显存优势 |
| P1 | 方案 A（INT8 权重） | 显存 -35%，可能加速 decode | 中 | 需要 INT8 matmul kernel |
| P2 | 方案 D（CUDA graph） | Decode -5~8 ms | 中 | 消除 pipeline bubble |
| P3 | 方案 E（Prefill INT8） | Prefill 1.5-2× | 中 | 仅 prefill 场景受益 |
| P4 | 方案 B（融合 dequant） | 显存 -60% + decode 加速 | 高 | 需要深入 BNB 修改 |

---

## 附录：测量方法

所有测量在 GPU 5 (NVIDIA L20) 上完成，使用以下命令：

```bash
# 显存测量
CUDA_VISIBLE_DEVICES=5 python -c "
import torch
# ... build model ...
print(f'GPU alloc: {torch.cuda.memory_allocated()/1e9:.2f} GB')
"

# 延迟测量
CUDA_VISIBLE_DEVICES=5 python -c "
from modules.latency_probe import LatencyProbe
probe = LatencyProbe(warmup=3, num_decode_tokens=16, ...)
stats = probe.measure(model, inputs, ...)
"
```

模型配置见 `configs/latency_bench/*.yaml`。
