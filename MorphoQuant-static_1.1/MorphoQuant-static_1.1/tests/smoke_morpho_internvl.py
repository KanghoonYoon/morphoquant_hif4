import os
import sys
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from modules import model_factory, evaluator
from config import AppConfig
import types
import torch.nn as nn

# Lightweight DummyQuantAct to avoid importing bitsandbytes or repo relative imports
class DummyQuantAct(nn.Module):
    def __init__(self, activation_bit=16, input_dim=128, llama_layer=True, count_block=1, count_layer=1):
        super().__init__()
        self.activation_bit = activation_bit
        self.layer_name = ''
        self._calibrate = False
        self.search = False
        self.dispersion_score = None

    def forward(self, x):
        return x

    def set_calibrate(self, calibrate=True):
        self._calibrate = calibrate

    def set_search(self, search=True):
        self.search = search

    def set_gamma(self, gamma_inf=0.5, gamma_cos=1.0):
        pass

    def set_sparse_buffer_ratio(self, sparse_buffer_ratio=0.8):
        pass

    def compute_dispersion_score(self):
        # create a dummy dispersion score tensor
        self.dispersion_score = torch.rand( (64,) )


def main():
    # minimal config override for InternVL2.5 morpho smoke test
    cfg = AppConfig()
    cfg.model.model_type = "internvl2_5"
    cfg.model.quant_method = "morpho"
    cfg.model.simulate_hif8 = False
    # small calibration size to be quick
    cfg.data = cfg.data or type("C", (), {})()
    cfg.data.calib_size = 2

    print("Building lightweight dummy model for internvl2_5 morpho test (no heavy deps)...")

    # Monkeypatch MorphoQuantActWrapper to use DummyQuantAct to avoid bitsandbytes C-extension
    def _patched_morpho_quantact_init(self, module, config, name, input_dim, llama_layer=True):
        super(model_factory.MorphoQuantActWrapper, self).__init__()
        self.module = module
        self.layer_name = name
        self.quant_activation = DummyQuantAct(
            activation_bit=getattr(config.quant, 'activation_bitwidth', 16),
            input_dim=input_dim,
            llama_layer=llama_layer,
            count_block=1,
            count_layer=1,
        )
        self.quant_activation.set_gamma(
            gamma_inf=getattr(config.quant, 'gamma_inf', 0.5),
            gamma_cos=getattr(config.quant, 'gamma_cos', 1.0),
        )
        self.quant_activation.layer_name = name

    # Apply monkeypatch
    model_factory.MorphoQuantActWrapper.__init__ = _patched_morpho_quantact_init

    # Also ensure evaluator recognizes our DummyQuantAct for isinstance checks
    evaluator.QuantAct = DummyQuantAct

    # Build a small dummy model and inject wrappers
    model = nn.Sequential(
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )
    replace_count = model_factory._wrap_morpho_quantact_recursive(model, cfg)
    print(f"Dummy model Morpho 注入完成，共包装 {replace_count} 个线性/卷积层。")

    # count QuantAct modules in model (use evaluator.QuantAct type for compatibility)
    qa_count = 0
    qa_names = []
    for name, mod in model.named_modules():
        if isinstance(mod, evaluator.QuantAct):
            qa_count += 1
            qa_names.append(name)

    print(f"Found QuantAct modules: {qa_count}")
    if qa_count > 0:
        print("Sample QuantAct names:")
        for n in qa_names[:20]:
            print(" -", n)

    # Simulate calibrate -> prepare sequence directly to avoid heavy dataset/model.generate
    print("Simulating calibrate phase: setting _calibrate flags on QuantAct modules...")
    for name, mod in model.named_modules():
        if isinstance(mod, evaluator.QuantAct):
            mod.set_calibrate(calibrate=True)
            mod.set_search(search=False)

    print("Simulating compute_dispersion_score for QuantAct modules...")
    for name, mod in model.named_modules():
        if isinstance(mod, evaluator.QuantAct):
            try:
                mod.compute_dispersion_score()
                print(f"  [Dispersion] Layer {getattr(mod, 'layer_name', name)}: max score={mod.dispersion_score.max().item():.4f}")
            except Exception as e:
                print(f"  compute_dispersion_score failed for {name}:", e)

    print("Simulating prepare phase: set_search and compute outlier_mask...")
    outlier_std_threshold = getattr(cfg.quant, 'outlier_std_threshold', 2.0)
    for name, mod in model.named_modules():
        if isinstance(mod, evaluator.QuantAct):
            mod.set_search(search=True)
            mod.set_calibrate(calibrate=False)
            if hasattr(mod, 'dispersion_score') and mod.dispersion_score is not None:
                dispersion_score = mod.dispersion_score
                mean_ds = dispersion_score.mean()
                std_ds = dispersion_score.std()
                threshold = mean_ds + outlier_std_threshold * std_ds
                outlier_mask = dispersion_score >= threshold
                num_outliers = int(outlier_mask.sum().item())
                total_channels = int(dispersion_score.numel())
                print(f"Layer {getattr(mod, 'layer_name', name)}: {num_outliers}/{total_channels} channels marked as outliers (threshold={threshold:.4f})")
                mod.register_buffer('outlier_mask', outlier_mask)


if __name__ == "__main__":
    main()
