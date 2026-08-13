import os
import re
import torch
import torch.nn as nn
from transformers import BitsAndBytesConfig, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, AutoConfig, AutoModel, AutoTokenizer, AwqConfig
from accelerate import init_empty_weights

from modules.device_utils import get_device
from modules.hif8_layers import replace_hif8_layers_recursive, HiF8Linear, replace_morpho_hif8_layers_recursive, MorphoHiF8Linear
from modules.hif4_layers import replace_hif4_layers_recursive, HiF4Linear, replace_morpho_hif4_layers_recursive, MorphoHiF4Linear, resolve_fp4_qtype_from_config
from modules.smoothquant_layers import replace_smoothquant_layers_recursive
from modules.awq_layers import replace_awq_layers_recursive
from modules.mbq_layers import replace_mbq_layers_recursive
from modules.rtn_layers import replace_rtn_layers_recursive
from modules.quarot_layers import apply_quarot_to_internvl, replace_quarot_rtn_layers_recursive
from modules.freeact_layers import replace_freeact_layers_recursive


class MorphoQuantActWrapper(nn.Module):
    def __init__(self, module, config, name, input_dim, llama_layer=True):
        super().__init__()
        from bitsandbytes.quantization_utils.quant_modules import QuantAct

        self.module = module
        self.layer_name = name
        
        # Initialize quant_activation with defensive fallback
        quant_act = QuantAct(
            activation_bit=config.quant.activation_bitwidth,
            input_dim=input_dim,
            llama_layer=llama_layer,
            count_block=1,
            count_layer=1,
        )
        
        # If QuantAct returned None, create a minimal dummy
        if quant_act is None:
            import warnings
            warnings.warn(f"QuantAct(...) returned None for layer {name}; using DummyQuantAct passthrough.")
            class _DummyQuantAct(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layer_name = name
                def set_gamma(self, gamma_inf=None, gamma_cos=None):
                    self.gamma_inf = gamma_inf
                    self.gamma_cos = gamma_cos
                def set_lp_norm(self, lp_norm=0.5):
                    self.lp_norm = lp_norm
                def set_cosine_loss(self, use_cosine_loss=False):
                    self.use_cosine_loss = use_cosine_loss
                def forward(self, x):
                    return x
            quant_act = _DummyQuantAct()
        
        # Now assign the quant_activation (either real or dummy)
        self.quant_activation = quant_act
        
        # Now set gamma parameters
        self.quant_activation.set_gamma(
            gamma_inf=config.quant.gamma_inf,
            gamma_cos=config.quant.gamma_cos,
        )
        self.quant_activation.set_lp_norm(
            lp_norm=config.quant.lp_norm,
        )
        self.quant_activation.set_cosine_loss(
            use_cosine_loss=config.quant.use_cosine_loss,
        )
        self.quant_activation.layer_name = name

    @property
    def weight(self):
        """Delegate weight access to wrapped module for compatibility."""
        return self.module.weight

    @property
    def bias(self):
        """Delegate bias access to wrapped module for compatibility."""
        return self.module.bias

    def forward(self, x):
        x = self.quant_activation(x)
        return self.module(x)


# Local names of sub-modules that should NOT be wrapped even if they are Linear.
# Uses only the leaf name (last component), not the full path, so that
# q_proj/k_proj/v_proj/o_proj inside attention modules are still wrapped.
_SKIP_LOCAL_NAMES = frozenset({
    "lm_head",       # LLM output head
    "output",        # Generic output layers
    "embed_tokens",  # Embedding layer
    "patch_embed",   # ViT patch embedding
    "patch_proj",    # ViT patch projection
    "pos_embed",     # Position embedding
    "cls_token",     # Class token (learnable)
    "class_token",   # Alternative naming
})


def _wrap_morpho_quantact_recursive(module, config, prefix=""):
    """
    Recursively wrap Linear modules with MorphoQuantActWrapper.
    
    Key design principles:
    1. ALL nn.Linear instances get wrapped EXCEPT those whose local name
       is in the SKIP list (lm_head, output, embed_tokens, etc.)
    2. We ALWAYS recurse into children, so attention container modules
       (self_attn, attention, etc.) still get their q_proj/k_proj/v_proj/o_proj
       sub-modules wrapped.
    3. A wrapped module (MorphoQuantActWrapper) acts as a leaf - we do NOT
       recurse into it since its inner module's children are already handled.
    """
    count = 0
    for name, child in list(module.named_children()):
        fullname = f"{prefix}.{name}" if prefix else name

        # Check if this child's LOCAL name should skip wrapping entirely.
        # We only check the leaf name to avoid blocking attention projection
        # layers whose full path contains "attn" but whose local name is q_proj.
        is_linear = isinstance(child, nn.Linear) or "Linear" in child.__class__.__name__
        should_skip = name.lower() in _SKIP_LOCAL_NAMES

        if is_linear and not should_skip:
            input_dim = getattr(child, "in_features", None)
            if input_dim is None:
                weight = getattr(child, "weight", None)
                input_dim = weight.shape[-1] if weight is not None and hasattr(weight, "shape") else None
            if input_dim is not None:
                llama_layer = True
                wrapped = MorphoQuantActWrapper(child, config, fullname, input_dim=input_dim, llama_layer=llama_layer)
                setattr(module, name, wrapped)
                count += 1
                # After wrapping, this module is now a leaf (MorphoQuantActWrapper).
                # No need to recurse into it.
                continue

        # Always recurse into children (whether skipped or not a Linear).
        count += _wrap_morpho_quantact_recursive(child, config, prefix=fullname)

    return count


def _get_quant_skip_substrings(config):
    return tuple(getattr(config.quant, "skip_module_substrings", None) or ())


def _should_skip_quant_module(name, config):
    return any(substring in name for substring in _get_quant_skip_substrings(config))


def _attach_morpho_quant_activation_to_bnb_linears(model, config):
    """Ensure BNB Linear4bit layers have QuantAct when activation_bitwidth < 16.

    Linear4bit.__init__ defaults ``activation_bit=16`` and skips creating ``quant_activation``.
    Morpho calibrate / prepare / search then find no ``QuantAct`` modules, so ``set_search`` is a no-op
    and the search branch in ``QuantAct.forward`` never runs.
    """
    from bitsandbytes.nn.modules import Linear4bit
    from bitsandbytes.quantization_utils.quant_modules import QuantAct

    bit = int(getattr(config.quant, "activation_bitwidth", 16))
    if bit >= 16:
        return 0

    attached = 0
    for name, module in model.named_modules():
        if not isinstance(module, Linear4bit):
            continue
        if _should_skip_quant_module(name, config):
            continue
        if getattr(module, "quant_activation", None) is not None:
            continue
        module.activation_bit = bit
        module.quant_activation = QuantAct(
            activation_bit=bit,
            input_dim=module.in_features,
            llama_layer=getattr(module, "llama_layer", True),
            count_block=getattr(module, "count_block", 1),
            count_layer=getattr(module, "count_layer", 1),
        )
        module.quant_activation.set_lp_norm(
            lp_norm=config.quant.lp_norm,
        )
        module.quant_activation.set_cosine_loss(
            use_cosine_loss=config.quant.use_cosine_loss,
        )
        attached += 1
    return attached


class ModelBuilder:
    @staticmethod
    def build(config):
        quant_method = config.model.quant_method
        model_type = getattr(config.model, "model_type", "qwen2_5_omni").lower()
        if config.quant.simulate_hif8 and quant_method != 'morpho_withhif8':
            print(">>> HiF8 Simulation ENABLED: Forcing quant_method='bf16'.")
            quant_method = "bf16"
        if config.quant.simulate_hif4 and quant_method not in ('morpho_withhif4',):
            print(">>> HiF4 Simulation ENABLED: Forcing quant_method='bf16'.")
            quant_method = "bf16"
        if config.quant.simulate_smoothquant:
            print(">>> SmoothQuant Simulation ENABLED: Forcing quant_method='bf16'.")
            quant_method = "bf16"
        if config.quant.simulate_awq:
            print(">>> AWQ Simulation ENABLED: Forcing quant_method='bf16'.")
            quant_method = "bf16"
        if config.quant.simulate_mbq:
            print(">>> MBQ Simulation ENABLED: Forcing quant_method='bf16'.")
            quant_method = "bf16"
        if config.quant.simulate_mquant:
            print(">>> MQuant Simulation ENABLED: Forcing quant_method='bf16'.")
            quant_method = "bf16"
        if config.quant.simulate_freeact:
            print(">>> FreeAct Simulation ENABLED: Forcing quant_method='bf16'.")
            quant_method = "bf16"
        model_path = config.model.model_path
        attn_impl = "eager" if config.model.save_attention_maps else None

        if model_type == "internvl2_5":
            processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
            model = ModelBuilder._build_internvl2_5(config, model_path)
            if hasattr(processor, "convert_tokens_to_ids"):
                model.img_context_token_id = processor.convert_tokens_to_ids("<IMG_CONTEXT>")
        else:
            processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)

            if quant_method == "bf16":
                model = ModelBuilder._build_bf16(config, model_path, attn_impl)
            elif quant_method == "int8":
                model = ModelBuilder._build_int8(config, model_path, attn_impl)
            elif quant_method == "awq":
                model = ModelBuilder._build_awq(config, model_path, attn_impl)
            elif quant_method == "hqq":
                model = ModelBuilder._build_hqq(config, model_path, attn_impl)
            elif quant_method in ("morpho", "qvlm"):
                model = ModelBuilder._build_morpho(config, model_path, attn_impl)
            elif quant_method == "morpho_withhif8":
                model = ModelBuilder._build_morpho_withhif8(config, model_path, attn_impl)
            elif quant_method == "morpho_withhif4":
                model = ModelBuilder._build_morpho_withhif4(config, model_path, attn_impl)
            elif quant_method == "qlora":
                model = ModelBuilder._build_qlora(config, model_path, attn_impl)
            else:
                raise ValueError(f"Unknown quantization method: {quant_method}")

        if quant_method in ('morpho', 'qvlm'):
            from bitsandbytes.quantization_utils.quant_modules import QuantAct
            for name, module in model.named_modules():
                if isinstance(module, QuantAct):
                    module.set_gamma(gamma_inf=config.quant.gamma_inf, gamma_cos=config.quant.gamma_cos)
                    module.set_lp_norm(lp_norm=config.quant.lp_norm)
                    module.set_cosine_loss(use_cosine_loss=config.quant.use_cosine_loss)
                    module.layer_name = name
                    module.activation_bit = config.quant.activation_bitwidth
                    
        elif quant_method == 'morpho_withhif8':
            pass # γ will be handled inside our own MorphoHiF8Linear
        elif quant_method == 'morpho_withhif4':
            pass # γ will be handled inside our own MorphoHiF4Linear

        if hasattr(model, "disable_talker"):
            model.disable_talker()
        model.eval()
        
        return model, processor

    @staticmethod
    def _build_internvl2_5(config, model_path):
        quant_method = config.model.quant_method
        print(f"正在加载 InternVL2.5 模型 ({quant_method})...")

        common_kwargs = dict(
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        )

        if quant_method == "bf16":
            model = AutoModel.from_pretrained(model_path, **common_kwargs).eval().cuda()
            if config.quant.simulate_smoothquant:
                print("\n[SmoothQuant Simulation] 开始替换 InternVL2.5 Linear 层...")
                replace_count = replace_smoothquant_layers_recursive(
                    model,
                    skip_substrings=_get_quant_skip_substrings(config),
                    weight_bits=config.quant.smoothquant_weight_bits,
                    act_bits=config.quant.smoothquant_act_bits,
                )
                print(f"[SmoothQuant Simulation] 替换完成，共替换 {replace_count} 层。")
                model.to("cuda")
                print("[SmoothQuant Simulation] 等待校准 (calibrate) 阶段观测激活幅值并完成权重量化。\n")
            if config.quant.simulate_awq:
                print("\n[AWQ Simulation] 开始替换 InternVL2.5 Linear 层...")
                replace_count = replace_awq_layers_recursive(
                    model,
                    skip_substrings=_get_quant_skip_substrings(config),
                    weight_bits=config.quant.awq_weight_bits,
                    group_size=config.quant.awq_group_size,
                    n_grid=config.quant.awq_n_grid,
                )
                print(f"[AWQ Simulation] 替换完成，共替换 {replace_count} 层。")
                model.to("cuda")
                print("[AWQ Simulation] 等待校准 (calibrate) 阶段观测激活幅值并完成分组权重量化。\n")
            if config.quant.simulate_mbq:
                print("\n[MBQ Simulation] 开始替换 InternVL2.5 Linear 层...")
                replace_count = replace_mbq_layers_recursive(
                    model,
                    skip_substrings=_get_quant_skip_substrings(config),
                    weight_bits=config.quant.mbq_weight_bits,
                    act_bits=config.quant.mbq_act_bits,
                    group_size=config.quant.mbq_group_size,
                    n_grid=config.quant.mbq_n_grid,
                )
                print(f"[MBQ Simulation] 替换完成，共替换 {replace_count} 层。")
                model.to("cuda")
                print("[MBQ Simulation] 等待校准 (calibrate) 阶段按模态观测激活幅值并完成分组权重量化。\n")
            if config.quant.simulate_mquant:
                from modules.mquant_layers import replace_mquant_layers_recursive
                print("\n[MQuant Simulation] 开始替换 InternVL2.5 Linear 层 (W{}A{})...".format(
                    config.quant.mquant_weight_bits, config.quant.mquant_act_bits))
                replace_count = replace_mquant_layers_recursive(
                    model,
                    skip_substrings=_get_quant_skip_substrings(config),
                    weight_bits=config.quant.mquant_weight_bits,
                    act_bits=config.quant.mquant_act_bits,
                )
                print(f"[MQuant Simulation] 替换完成，共替换 {replace_count} 层。")
                model.to("cuda")
                print("[MQuant Simulation] 等待校准 (calibrate) 阶段按模态观测激活幅值并完成权重量化。\n")
        elif quant_method == "int8":
            model = AutoModel.from_pretrained(model_path, load_in_8bit=True, **common_kwargs).eval()
        elif quant_method in ("morpho", "morpho_withhif8", "morpho_withhif4"):
            model = AutoModel.from_pretrained(model_path, load_in_4bit=True, **common_kwargs).eval()
            print("InternVL2.5 开始注入 Morpho QuantAct 包装器...")
            replace_count = _wrap_morpho_quantact_recursive(model, config)
            print(f"InternVL2.5 Morpho 注入完成，共包装 {replace_count} 个线性/卷积层。")
        elif quant_method == "qlora":
            model = ModelBuilder._build_qlora(config, model_path, attn_impl=None, model_type="internvl2_5")
        elif quant_method == "awq":
            model = AutoModel.from_pretrained(model_path, load_in_4bit=True, attn_implementation="flash_attention_2", **common_kwargs).eval()
        elif quant_method == "quarot":
            model = AutoModel.from_pretrained(model_path, **common_kwargs).eval()
            print("InternVL2.5 开始应用 QuaRot 旋转融合 (R1" + ("+R2" if config.quant.quarot_rotate_v_o else "") + ")...")
            apply_quarot_to_internvl(model, config)
            print("InternVL2.5 QuaRot 旋转融合完成，开始对语言模型骨干做 RTN 伪量化 (权重逐通道 / 激活逐 token)...")
            replace_count = replace_quarot_rtn_layers_recursive(
                model.language_model,
                skip_substrings=_get_quant_skip_substrings(config),
                weight_bits=config.quant.quarot_weight_bits,
                act_bits=config.quant.quarot_act_bits,
            )
            print(f"InternVL2.5 QuaRot RTN 替换完成，共替换 {replace_count} 层。")
            model = model.cuda()
        elif quant_method == "freeact":
            model = AutoModel.from_pretrained(model_path, **common_kwargs).eval()
            print("InternVL2.5 开始应用 QuaRot 旋转融合 (FreeAct base, R1" + ("+R2" if config.quant.freeact_rotate_v_o else "") + ")...")
            # Reuse QuaRot R1/R2 rotation — temporarily set quarot config fields
            config.quant.quarot_rotate_v_o = config.quant.freeact_rotate_v_o
            config.quant.quarot_seed = config.quant.freeact_seed
            apply_quarot_to_internvl(model, config)
            print("InternVL2.5 QuaRot 旋转融合完成，开始替换为 FreeActLinear...")
            replace_count = replace_freeact_layers_recursive(
                model.language_model,
                skip_substrings=_get_quant_skip_substrings(config),
                weight_bits=config.quant.freeact_weight_bits,
                act_bits=config.quant.freeact_act_bits,
                r1_ratio=config.quant.freeact_r1_ratio,
                r2_ratio=config.quant.freeact_r2_ratio,
                model_root=model,
                calib_epochs=config.quant.freeact_calib_epochs,
                calib_lr=config.quant.freeact_calib_lr,
            )
            print(f"InternVL2.5 FreeAct 替换完成，共替换 {replace_count} 层。")
            model = model.cuda()
        else:
            model = AutoModel.from_pretrained(model_path, **common_kwargs).eval().cuda()

        print("InternVL2.5 模型加载完成！")
        
        # Workaround for transformers >= 4.50: InternLM2ForCausalLM may not inherit
        # GenerationMixin, causing generate() and generation_config to be unavailable.
        # Also patch forward() to accept cache_position for generationmixin compatibility.
        if hasattr(model, 'language_model'):
            lm = model.language_model
            print("[Compatibility] Applying InternLM2 generative compatibility patches...")
            
            from transformers.generation import GenerationMixin, GenerationConfig
            
            # 1. Make the class inherit from GenerationMixin
            lm_class = type(lm)
            if not issubclass(lm_class, GenerationMixin):
                class _PatchedInternLM2(GenerationMixin, lm_class):
                    pass
                lm.__class__ = _PatchedInternLM2
                print("[Compatibility] Patched class to inherit GenerationMixin.")
            
            # 2. Ensure generation_config exists
            if not hasattr(lm, 'generation_config') or lm.generation_config is None:
                if hasattr(lm, 'config'):
                    lm.generation_config = GenerationConfig.from_model_config(lm.config)
                else:
                    lm.generation_config = GenerationConfig()
                print("[Compatibility] Set generation_config.")
            
            # 3. Patch forward() to accept cache_position and other new parameters
            # that GenerationMixin may pass but InternLM2's forward doesn't recognize.
            _original_forward = lm.forward
            def _patched_forward(input_ids, *args, **kwargs):
                # Remove parameters that InternLM2ForCausalLM.forward doesn't support
                # These are added by newer transformers GenerationMixin but older InternLM2 doesn't accept them
                unsupported_params = ['cache_position', 'output_router_logits']
                for param in unsupported_params:
                    kwargs.pop(param, None)
                
                return _original_forward(input_ids, *args, **kwargs)
            lm.forward = _patched_forward
            print("[Compatibility] Patched forward() to filter unsupported parameters.")

            # Newer GenerationMixin versions can pass a non-None
            # past_key_values container whose first key/value tensor is still
            # None. InternLM2's original helper blindly reads
            # past_key_values[0][0].shape, so normalize that case back to
            # "no cache yet".
            if hasattr(lm, "prepare_inputs_for_generation"):
                _original_prepare_inputs = lm.prepare_inputs_for_generation

                def _patched_prepare_inputs_for_generation(
                    input_ids,
                    past_key_values=None,
                    attention_mask=None,
                    inputs_embeds=None,
                    **kwargs,
                ):
                    if past_key_values is not None:
                        try:
                            first_key = past_key_values[0][0]
                        except Exception:
                            first_key = None
                        if first_key is None:
                            past_key_values = None

                    return _original_prepare_inputs(
                        input_ids,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        inputs_embeds=inputs_embeds,
                        **kwargs,
                    )

                lm.prepare_inputs_for_generation = _patched_prepare_inputs_for_generation
                print("[Compatibility] Patched prepare_inputs_for_generation() for empty cache entries.")
            print("[Compatibility] InternLM2 generative compatibility patches complete.")
        
        return model

    @staticmethod
    def _build_awq(config, model_path, attn_impl):
        print("正在加载 AWQ 4-bit 量化模型...")
        quantization_config = AwqConfig(
            bits=4,
            fuse_max_seq_len=1024,
            do_fuse=True,
        )

        try:
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="cuda",
                trust_remote_code=True,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            print(f"transformers 原生 AWQ 加载失败: {e}")
            print("尝试直接加载（假设模型已是 AutoAWQ 处理后保存）...")
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_path,
                device_map="cuda",
                trust_remote_code=True,
                attn_implementation=attn_impl,
            )
        return model

    @staticmethod
    def _build_bf16(config, model_path, attn_impl):
        print("正在加载全精度模型 (BF16)...")
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            device_map=get_device(),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_impl
        )
        print("全精度模型加载完成！")

        if config.quant.simulate_hif8:
            print("\n[HiF8 Simulation] 开始替换 Linear 层...")
            replace_count = replace_hif8_layers_recursive(
                model,
                skip_substrings=_get_quant_skip_substrings(config),
            )
            print(f"[HiF8 Simulation] 替换完成，共替换 {replace_count} 层。")
            
            print("[HiF8 Simulation] 正在初始化并将权重移入 GPU 进行量化...")
            model.to(get_device())
            params_quantized = 0
            for m in model.modules():
                if isinstance(m, HiF8Linear):
                    m.quantize_weight()
                    if m.weight_quantized:
                        params_quantized += 1
            print(f"[HiF8 Simulation] 权重初始化完成 ({params_quantized} layers quantized)。\n")

        if config.quant.simulate_hif4:
            fp4_qtype = resolve_fp4_qtype_from_config(config)
            print(f"\n[FP4 Simulation] dtype={fp4_qtype}，开始替换 Linear 层...")
            replace_count = replace_hif4_layers_recursive(
                model,
                skip_substrings=_get_quant_skip_substrings(config),
                qtype=fp4_qtype,
            )
            print(f"[FP4 Simulation] 替换完成，共替换 {replace_count} 层。")

            print("[FP4 Simulation] 正在初始化并将权重移入 GPU 进行量化...")
            model.to(get_device())
            params_quantized = 0
            for m in model.modules():
                if isinstance(m, HiF4Linear):
                    m.quantize_weight()
                    if m.weight_quantized:
                        params_quantized += 1
            print(f"[FP4 Simulation] 权重初始化完成 ({params_quantized} layers quantized)。\n")

        if config.quant.simulate_smoothquant:
            print("\n[SmoothQuant Simulation] 开始替换 Linear 层...")
            replace_count = replace_smoothquant_layers_recursive(
                model,
                skip_substrings=_get_quant_skip_substrings(config),
                weight_bits=config.quant.smoothquant_weight_bits,
                act_bits=config.quant.smoothquant_act_bits,
            )
            print(f"[SmoothQuant Simulation] 替换完成，共替换 {replace_count} 层。")
            model.to(get_device())
            print("[SmoothQuant Simulation] 等待校准 (calibrate) 阶段观测激活幅值并完成权重量化。\n")

        if config.quant.simulate_awq:
            print("\n[AWQ Simulation] 开始替换 Linear 层...")
            replace_count = replace_awq_layers_recursive(
                model,
                skip_substrings=_get_quant_skip_substrings(config),
                weight_bits=config.quant.awq_weight_bits,
                group_size=config.quant.awq_group_size,
                n_grid=config.quant.awq_n_grid,
            )
            print(f"[AWQ Simulation] 替换完成，共替换 {replace_count} 层。")
            model.to(get_device())
            print("[AWQ Simulation] 等待校准 (calibrate) 阶段观测激活幅值并完成分组权重量化。\n")

        if config.quant.simulate_mbq:
            print("\n[MBQ Simulation] 开始替换 Linear 层...")
            replace_count = replace_mbq_layers_recursive(
                model,
                skip_substrings=_get_quant_skip_substrings(config),
                weight_bits=config.quant.mbq_weight_bits,
                act_bits=config.quant.mbq_act_bits,
                group_size=config.quant.mbq_group_size,
                n_grid=config.quant.mbq_n_grid,
            )
            print(f"[MBQ Simulation] 替换完成，共替换 {replace_count} 层。")
            model.to(get_device())
            print("[MBQ Simulation] 等待校准 (calibrate) 阶段按模态观测激活幅值并完成分组权重量化。\n")

        if config.quant.simulate_mquant:
            # Hadamard 旋转 (R1+R2): 在替换 Linear 之前做离线权重融合
            if getattr(config.quant, 'mquant_use_quarot', False):
                from modules.mquant_quarot import apply_mquant_quarot_to_qwen
                print("\n[MQuant-QuaRot] 开始对 Qwen2.5-Omni Thinker 应用 Hadamard 旋转...")
                apply_mquant_quarot_to_qwen(model, config)
                print("[MQuant-QuaRot] Hadamard 旋转完成。\n")

            from modules.mquant_layers import replace_mquant_layers_recursive
            quarot_tag = " +QuaRot" if getattr(config.quant, 'mquant_use_quarot', False) else ""
            print("\n[MQuant Simulation{}] 开始替换 Linear 层 (W{}A{})...".format(
                quarot_tag, config.quant.mquant_weight_bits, config.quant.mquant_act_bits))
            replace_count = replace_mquant_layers_recursive(
                model,
                skip_substrings=_get_quant_skip_substrings(config),
                weight_bits=config.quant.mquant_weight_bits,
                act_bits=config.quant.mquant_act_bits,
                use_rms=getattr(config.quant, 'mquant_use_rms', False),
                rms_threshold=getattr(config.quant, 'mquant_rms_threshold', 3.0),
            )
            print(f"[MQuant Simulation] 替换完成，共替换 {replace_count} 层。")
            model.to(get_device())
            print("[MQuant Simulation] 等待校准 (calibrate) 阶段按模态观测激活幅值并完成权重量化。\n")

        return model

    @staticmethod
    def _build_int8(config, model_path, attn_impl):
        print("正在加载 8-bit 量化模型...")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation=attn_impl
        )
        print("8-bit 模型加载完成！")
        return model

    @staticmethod
    def _build_hqq(config, model_path, attn_impl):
        print("正在加载 HQQ 4-bit 量化模型...")
        try:
            from hqq.engine.hf import HQQModelForCausalLM
            from hqq.core.quantize import BaseQuantizeConfig
            from hqq.models.base import BaseHQQModel
        except ImportError:
            raise ImportError("请先安装 hqq: pip install hqq")

        class Qwen2_5OmniHQQ(BaseHQQModel):
            @classmethod
            def autoname_modules(cls, model):
                for name, module in model.named_modules():
                    module.name = name

            @classmethod
            def get_linear_tags(cls):
                return [
                    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
                ]

        print("[HQQ] Registering custom handler for Qwen2.5-Omni...")
        HQQModelForCausalLM._HQQ_REGISTRY["Qwen2_5OmniForConditionalGeneration"] = Qwen2_5OmniHQQ
        HQQModelForCausalLM._HQQ_REGISTRY["Qwen2_5OmniModel"] = Qwen2_5OmniHQQ

        quant_config = BaseQuantizeConfig(
            nbits=4, group_size=64, quant_zero=False, quant_scale=False,
            offload_meta=False, view_as_float=False
        )
        
        try:
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
                attn_implementation=attn_impl
            )
            print(f"Quantizing model with HQQ (4-bit, gs=64)...")
            HQQModelForCausalLM.quantize_model_(model, quant_config=quant_config, device="cuda")
            print("HQQ Quantization Complete.")
        except Exception as e:
            print(f"HQQ 量化失败: {e}")
            raise e

        print("HQQ 模型加载完成！")
        return model

    @staticmethod
    def _build_morpho_withhif8(config, model_path, attn_impl):
        print("正在加载全精度模型 (BF16) 用于 Morpho + HiF8 Beta 融合路径...")
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            device_map=get_device(),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_impl
        )
        print("全精度模型加载完成！开始替换为 MorphoHiF8 融合层...")
        
        from modules.hif8_layers import replace_morpho_hif8_layers_recursive, MorphoHiF8Linear
        replace_count = replace_morpho_hif8_layers_recursive(model, config)
        print(f"替换完成，共替换 {replace_count} 层。")
        
        print("正在初始化 MorphoHiF8 权重...")
        model.to(get_device())
        params_quantized = 0
        for m in model.modules():
            if isinstance(m, MorphoHiF8Linear):
                m.quantize_weight()
                if m.weight_quantized:
                    params_quantized += 1
        print(f"MorphoHiF8 权重初始化完成 ({params_quantized} layers quantized)。")
        
        return model

    @staticmethod
    def _build_morpho_withhif4(config, model_path, attn_impl):
        fp4_qtype = resolve_fp4_qtype_from_config(config)
        print(f"正在加载全精度模型 (BF16) 用于 Morpho + FP4 (dtype={fp4_qtype}) Beta 融合路径...")
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            device_map=get_device(),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_impl
        )
        print("全精度模型加载完成！开始替换为 MorphoHiF4 融合层...")

        from modules.hif4_layers import replace_morpho_hif4_layers_recursive, MorphoHiF4Linear
        replace_count = replace_morpho_hif4_layers_recursive(model, config)
        print(f"替换完成，共替换 {replace_count} 层。")

        print("正在初始化 MorphoHiF4 权重...")
        model.to(get_device())
        params_quantized = 0
        for m in model.modules():
            if isinstance(m, MorphoHiF4Linear):
                m.quantize_weight()
                if m.weight_quantized:
                    params_quantized += 1
        print(f"MorphoHiF4 权重初始化完成 ({params_quantized} layers quantized)。")
        
        return model

    @staticmethod
    def _build_morpho(config, model_path, attn_impl):
        print("DABC 模式：正在扫描模型结构以构建量化白名单...")
        auto_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        with init_empty_weights():
            fake_model = Qwen2_5OmniForConditionalGeneration(auto_config)

        modules_to_skip = []
        for name, module in fake_model.named_modules():
            if isinstance(module, nn.Linear):
                if "thinker.model" not in name and "visual" not in name:
                    modules_to_skip.append(name)
                if _should_skip_quant_module(name, config):
                    modules_to_skip.append(name)
                
                if config.quant.keep_special_layer:
                    if re.search(r"thinker\.model\.layers\.([0-2])\.", name):
                        modules_to_skip.append(name)
                        print(f"Skipping quantization for: {name} (Mixed Precision)")

        del fake_model

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            llm_int8_skip_modules=modules_to_skip
        )

        print("正在加载并量化模型...")
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_impl
        )
        n_quant_act = _attach_morpho_quant_activation_to_bnb_linears(model, config)
        if n_quant_act:
            print(
                f"Morpho: 已在 {n_quant_act} 个 BNB Linear4bit 层挂载 QuantAct（activation_bitwidth={config.quant.activation_bitwidth}）。"
            )
        print("加载完成！")
        return model

    @staticmethod
    def _build_qlora(config, model_path, attn_impl, model_type="qwen2_5_omni"):
        quant = config.quant

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=getattr(quant, 'qlora_double_quant', True),
            bnb_4bit_quant_type=getattr(quant, 'qlora_quant_type', 'nf4'),
            bnb_4bit_compute_dtype=getattr(quant, 'qlora_compute_dtype', 'bfloat16'),
            llm_int8_skip_modules=getattr(quant, 'qlora_skip_modules', None) or None,
        )

        print(f"QLoRA 量化模式（{bnb_config.bnb_4bit_quant_type}）：正在加载模型...")

        if model_type == "internvl2_5":
            model = AutoModel.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).eval()
        else:
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="cuda",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=attn_impl,
            )

        # Conditionally attach activation quantization (W4A8 or W4A4)
        bit = int(getattr(quant, 'activation_bitwidth', 16))
        if bit < 16:
            n_attached = _attach_morpho_quant_activation_to_bnb_linears(model, config)
            print(f"QLoRA: 附加 QuantAct 到 {n_attached} 个 Linear4bit 层 (activation_bitwidth={bit})")

        # Optionally load LoRA adapters
        lora_path = getattr(config.model, 'lora_adapter_path', None)
        if lora_path:
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, lora_path)
                print(f"QLoRA LoRA adapters 从 {lora_path} 加载完成。")
            except ImportError:
                print("Warning: peft 未安装，跳过 LoRA adapter 加载。")
            except Exception as e:
                print(f"Warning: LoRA adapter 加载失败: {e}")

        linear4bit_count = sum(1 for m in model.modules() if m.__class__.__name__ in ("Linear4bit", "LinearNF4"))
        print(f"QLoRA 模型加载完成，检测到 {linear4bit_count} 个 4-bit Linear 模块。")
        return model
