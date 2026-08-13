"""
Hardware Latency Probe for GPU Inference Benchmarking.

Provides CUDA-Event-based precise timing for model.generate() calls,
with prefill/decode separation via the two-call difference method.
"""

import torch
import statistics
from dataclasses import dataclass, asdict
from typing import Optional, Callable, List, Dict, Any


@dataclass
class LatencyStats:
    """Single-run latency measurement results."""
    ttft_ms: float = 0.0
    decode_ms_per_token: float = 0.0
    prefill_ms: float = 0.0
    peak_memory_mb: float = 0.0
    num_input_tokens: int = 0
    num_output_tokens: int = 0
    warmup_iters: int = 0
    num_decode_tokens_target: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class LatencyProbe:
    """CUDA Event-based latency probe for model inference.

    Uses a two-call difference method to separate prefill and decode latency
    from HuggingFace generate() which does not natively expose this split.

    Some models (e.g. Qwen2.5-Omni) use non-standard token-limit parameter
    names.  Pass ``token_limit_key`` to match the model's API::

        # Standard HF model
        probe = LatencyProbe(token_limit_key="max_new_tokens")

        # Qwen2.5-Omni
        probe = LatencyProbe(token_limit_key="thinker_max_new_tokens")

    Usage::

        probe = LatencyProbe(warmup=3, num_decode_tokens=32)
        stats = probe.measure(model, inputs)
        print(f"TTFT: {stats.ttft_ms:.1f} ms, Decode: {stats.decode_ms_per_token:.1f} ms/tok")
    """

    def __init__(
        self,
        warmup: int = 3,
        num_decode_tokens: int = 32,
        token_limit_key: str = "max_new_tokens",
    ):
        if warmup < 1:
            raise ValueError("warmup must be >= 1")
        if num_decode_tokens < 2:
            raise ValueError("num_decode_tokens must be >= 2 for the difference method")
        self.warmup = warmup
        self.num_decode_tokens = num_decode_tokens
        self.token_limit_key = token_limit_key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def measure(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        generate_kwargs: Optional[Dict[str, Any]] = None,
    ) -> LatencyStats:
        """Measure prefill + decode latency for a single model.generate() call.

        Args:
            model: The HuggingFace model with a ``generate`` method.
            inputs: Tokenizer outputs (input_ids, attention_mask, etc.) already on GPU.
            generate_kwargs: Extra kwargs forwarded to model.generate().

        Returns:
            LatencyStats with TTFT, decode_per_token, prefill, and peak memory.
        """
        if generate_kwargs is None:
            generate_kwargs = {}

        stats = LatencyStats(
            warmup_iters=self.warmup,
            num_decode_tokens_target=self.num_decode_tokens,
        )
        stats.num_input_tokens = inputs["input_ids"].size(1)

        # ---- Phase 1: Warmup (also triggers lazy weight quant in HiF8/HiF4) ----
        warmup_kwargs = {**generate_kwargs, self.token_limit_key: 4}
        for _ in range(self.warmup):
            _ = model.generate(**inputs, **warmup_kwargs)
        torch.cuda.synchronize()

        # ---- Phase 2: TTFT measurement (token_limit=1) ----
        torch.cuda.reset_peak_memory_stats()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        ttft_kwargs = {**generate_kwargs, self.token_limit_key: 1}
        start_ev.record()
        out1 = model.generate(**inputs, **ttft_kwargs)
        end_ev.record()
        torch.cuda.synchronize()

        stats.ttft_ms = start_ev.elapsed_time(end_ev)
        stats.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # ---- Phase 3: Full decode measurement (token_limit=N) ----
        torch.cuda.reset_peak_memory_stats()

        decode_kwargs = {**generate_kwargs, self.token_limit_key: self.num_decode_tokens}
        start_ev.record()
        outN = model.generate(**inputs, **decode_kwargs)
        end_ev.record()
        torch.cuda.synchronize()

        totalN_ms = start_ev.elapsed_time(end_ev)
        peak_memN_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        stats.peak_memory_mb = max(stats.peak_memory_mb, peak_memN_mb)

        actual_output_tokens = outN.size(1) - stats.num_input_tokens
        stats.num_output_tokens = max(actual_output_tokens, 1)

        # ---- Phase 4: Solve for decode_per_token and prefill ----
        # T₁ = prefill + 1×decode
        # T₂ = prefill + N×decode
        # decode_per_token = (T₂ - T₁) / (N - 1)
        # prefill = T₁ - decode_per_token
        if actual_output_tokens <= 1:
            # Fallback: model didn't generate more than 1 token; can't separate
            stats.decode_ms_per_token = stats.ttft_ms
            stats.prefill_ms = 0.0
        else:
            stats.decode_ms_per_token = (totalN_ms - stats.ttft_ms) / (actual_output_tokens - 1)
            stats.prefill_ms = stats.ttft_ms - stats.decode_ms_per_token

        return stats

    def measure_with_trials(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        generate_kwargs: Optional[Dict[str, Any]] = None,
        num_trials: int = 3,
    ) -> LatencyStats:
        """Run multiple trials and return the median result.

        The median is used (instead of mean) to be robust against OS scheduling
        jitter and occasional thermal throttling spikes.
        """
        if generate_kwargs is None:
            generate_kwargs = {}

        trial_stats: List[LatencyStats] = []
        for _ in range(num_trials):
            trial_stats.append(self.measure(model, inputs, generate_kwargs))

        # Aggregate: use median for timing, max for memory
        ttfts = sorted(s.ttft_ms for s in trial_stats)
        decodes = sorted(s.decode_ms_per_token for s in trial_stats)
        prefills = sorted(s.prefill_ms for s in trial_stats)
        mems = [s.peak_memory_mb for s in trial_stats]

        median_idx = len(ttfts) // 2
        aggregated = LatencyStats(
            ttft_ms=ttfts[median_idx],
            decode_ms_per_token=decodes[median_idx],
            prefill_ms=prefills[median_idx],
            peak_memory_mb=max(mems),
            num_input_tokens=trial_stats[0].num_input_tokens,
            num_output_tokens=trial_stats[0].num_output_tokens,
            warmup_iters=self.warmup,
            num_decode_tokens_target=self.num_decode_tokens,
        )
        return aggregated

    # ------------------------------------------------------------------
    # Per-layer profiling (optional, gated by benchmark.profile_layers)
    # ------------------------------------------------------------------

    @staticmethod
    def attach_layer_probes(model: torch.nn.Module) -> Dict[str, List[float]]:
        """Wrap quant layer forward() methods with CUDA Event timing.

        Returns a shared dict ``layer_name -> [elapsed_ms, ...]`` that is
        populated during forward passes.  Call ``detach_layer_probes`` to
        restore original forwards.

        NOTE: This adds per-layer overhead.  Only use when
        ``benchmark.profile_layers=True``.
        """
        from modules.hif8_layers import HiF8Linear, MorphoHiF8Linear
        from modules.hif4_layers import HiF4Linear, MorphoHiF4Linear
        from modules.smoothquant_layers import SmoothQuantLinear
        from modules.awq_layers import AWQLinear
        from modules.mbq_layers import MBQLinear
        from modules.rtn_layers import RTNQuantLinear
        from modules.mquant_layers import MQuantLinear
        from modules.freeact_layers import FreeActLinear

        _QUANT_LAYER_TYPES = (
            HiF8Linear, MorphoHiF8Linear,
            HiF4Linear, MorphoHiF4Linear,
            SmoothQuantLinear, AWQLinear, MBQLinear,
            RTNQuantLinear, MQuantLinear, FreeActLinear,
        )

        timings: Dict[str, List[float]] = {}
        _originals: Dict[str, object] = {}

        for name, module in model.named_modules():
            if isinstance(module, _QUANT_LAYER_TYPES):
                _originals[name] = module.forward
                timings[name] = []

                def _make_timed_forward(orig_fn, layer_name):
                    def timed_forward(x, *args, **kwargs):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        out = orig_fn(x, *args, **kwargs)
                        end.record()
                        end.synchronize()
                        timings[layer_name].append(start.elapsed_time(end))
                        return out
                    return timed_forward

                module.forward = _make_timed_forward(module.forward, name)

        # Store originals on the model for later cleanup
        model._latency_probe_originals = _originals
        return timings

    @staticmethod
    def detach_layer_probes(model: torch.nn.Module):
        """Restore original forward methods after profiling."""
        originals = getattr(model, '_latency_probe_originals', {})
        for name, orig_fn in originals.items():
            # Navigate to the module and restore
            parts = name.split('.')
            obj = model
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], type(obj)._make_unwrapped(orig_fn) if hasattr(obj, parts[-1]) else None)
            # Actually, simpler: just restore the original forward
            try:
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                leaf = getattr(parent, parts[-1])
                leaf.forward = orig_fn
            except (AttributeError, KeyError):
                pass
        if hasattr(model, '_latency_probe_originals'):
            del model._latency_probe_originals
