import yaml
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
    model_path: str = "/private/wy/pretrained_models/Qwen2.5-Omni-3B"
    model_type: str = "qwen2_5_omni"
    quant_method: str = "qlora" # choices: bf16, int8, awq, hqq, qlora, morpho, morpho_withhif8, morpho_withhif4, smoothquant, w4a16, w4a8, quarot, mbq, freeact
    apply_hooks: bool = False
    save_attention_maps: bool = False
    save_per_head_attention: bool = False
    lora_adapter_path: Optional[str] = None  # Path to LoRA adapter weights for QLoRA inference

@dataclass
class QuantConfig:
    simulate_hif8: bool = False
    simulate_hif4: bool = False
    # 4-bit 浮点格式选择 (HiF4 路径): hifx4 (默认) | nvfp4
    fp4_qtype: str = "hifx4"
    # 量化范围控制：全名包含任一子串的 Linear 不做量化。
    # 空 = 除 lm_head 外全部量化（含 visual / audio_tower / talker / token2wav）。
    # 例: ["thinker.visual", "thinker.audio_tower", "talker", "token2wav"] 只量化 LLM。
    skip_module_substrings: List[str] = field(default_factory=list)
    simulate_smoothquant: bool = False
    smoothquant_weight_bits: int = 8
    smoothquant_act_bits: Optional[int] = 8  # None 表示不量化激活 (WxA16)
    activation_bitwidth: int = 4
    keep_special_layer: bool = False
    debug_quant_act: bool = False
    gamma_inf: float = 0.5
    gamma_cos: float = 1.0
    calib_without_audio: bool = False
    calib_without_video: bool = False
    lp_norm: float = 2.0
    use_cosine_loss: bool = False
    sparse_buffer_ratio: float = 0.8
    search_ratio_lower_bound: float = 0.6
    disable_sparse_compensation: bool = False
    disable_boundary_cooptimization: bool = False  # When True, skip boundary search; use raw calib min/max as quantization bounds
    calib_size: int = 256
    search_size: int = 32
    batch_size: int = 8
    outlier_std_threshold: float = 2.0
    force_outlier_quantact_substrings: List[str] = field(default_factory=lambda: [".mlp1."])
    smoothquant_alpha: float = 0.5
    simulate_awq: bool = False
    awq_weight_bits: int = 4
    awq_group_size: int = 128
    awq_n_grid: int = 20
    simulate_mbq: bool = False
    mbq_weight_bits: int = 4
    mbq_act_bits: Optional[int] = None  # None 表示不量化激活 (WxA16)，设为 4 即 W4A4
    mbq_group_size: int = 128
    mbq_n_grid: int = 20
    mbq_modalities: List[str] = field(default_factory=lambda: ["text", "multimodal"])
    simulate_mquant: bool = False
    mquant_weight_bits: int = 4
    mquant_act_bits: int = 4  # W4A4 全静态量化
    mquant_use_quarot: bool = False  # 启用 Hadamard 旋转 (R1+R2) 平滑激活异常值
    mquant_use_rms: bool = False     # 启用 RMS：第一通道 full precision，其余 W4A4 量化
    mquant_rms_threshold: float = 3.0  # RMS auto-detect 阈值
    # QLoRA-specific fields
    qlora_quant_type: str = "nf4"          # nf4 or fp4
    qlora_double_quant: bool = True        # Double quantization for additional memory savings
    qlora_compute_dtype: str = "bfloat16"  # Compute dtype for dequantized weights
    qlora_skip_modules: List[str] = field(default_factory=list)  # Module name substrings to skip 4-bit quantization
    weight_bitwidth: int = 4
    quarot_rotate_v_o: bool = True
    quarot_seed: int = 2025
    quarot_weight_bits: int = 4
    quarot_act_bits: Optional[int] = 4
    # FreeAct
    simulate_freeact: bool = False
    freeact_weight_bits: int = 4
    freeact_act_bits: int = 4
    freeact_r1_ratio: float = 1.0 / 32.0   # vision-unique subspace ratio
    freeact_r2_ratio: float = 1.0 / 32.0   # text-unique subspace ratio
    freeact_calib_epochs: int = 15
    freeact_calib_lr: float = 1e-3
    freeact_rotate_v_o: bool = True
    freeact_seed: int = 2025

@dataclass
class DataConfig:
    only_samples_with_images: bool = False
    num_samples: int = -1 # -1 for all
    target_subject: str = "all"
    dataset_dir: str = "/private/wy/datasets"
    data_root: str = "/Foundation"
    output_file: str = "/private/wy/logs/MorphoQuant/airbench/Foundation_result_qwen2.5_omni.jsonl"
    run_pca: bool = False
    internvl_max_num_tiles: Optional[int] = None
    internvl_calib_max_num_tiles: Optional[int] = None
    internvl_calib_max_images: Optional[int] = None
    internvl_eval_max_images: Optional[int] = None
    internvl_eval_max_num_tiles: Optional[int] = None
    videomme_num_frames: int = 32

@dataclass
class BenchmarkConfig:
    enabled: bool = False
    fused_kernel: bool = False      # Use MorphoFusedLinear instead of current implementation
    use_cuda_kernel: bool = False   # Use fused CUDA kernel for act quant (requires fused_kernel=True)
    use_w4a4: bool = False          # Use W4A4 INT4 GEMM kernel (replaces W_fp16_cache with packed INT4)
    warmup: int = 3
    decode_tokens: int = 32
    input_lengths: List[int] = field(default_factory=lambda: [128, 512, 2048])
    num_trials: int = 3
    profile_layers: bool = False
    output_path: str = ""
    keep_text_layers: Optional[int] = None  # Uniform structural depth pruning for latency/FLOPs candidates

@dataclass
class RunConfig:
    save_dir: str = "/private/wy/pretrained_models/qwen2.5-omni-3b-4bit-calibrated"
    seed: int = 2025

@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    data: DataConfig = field(default_factory=DataConfig)
    run: RunConfig = field(default_factory=RunConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    config_path: str = ""
    config_raw: str = ""

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        data = yaml.safe_load(raw)

        cfg = cls()
        cfg.config_path = path
        cfg.config_raw = raw
        if data.get("model"):
            cfg.model = ModelConfig(**data["model"])
        if data.get("quant"):
            cfg.quant = QuantConfig(**data["quant"])
        if data.get("data"):
            cfg.data = DataConfig(**data["data"])
        if data.get("run"):
            cfg.run = RunConfig(**data["run"])
        if data.get("benchmark"):
            cfg.benchmark = BenchmarkConfig(**data["benchmark"])

        return cfg
