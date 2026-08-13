import torch
import torch.nn as nn

from quant_cy import QLinear
from quant_cy.utils.utils import replace_linear


# This demo shows how replace_linear converts a normal nn.Module tree into a model
# whose Linear layers are replaced by QLinear.
#
# What this script verifies:
# 1. All target nn.Linear modules are replaced.
# 2. The replaced model still runs bf16 forward on CUDA.
# 3. Backward succeeds and produces finite gradients.
# 4. quant_grad=False means gradient tensors are not quantized during backward.
torch.manual_seed(42)


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # A small nested submodule is enough to demonstrate recursive replacement.
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        return self.fc2(torch.nn.functional.silu(self.fc1(x)))


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(128, 256)
        self.block = ToyBlock()
        self.head = nn.Linear(256, 128)

    def forward(self, x):
        # The model intentionally contains multiple Linear layers at different
        # hierarchy levels so replace_linear has to walk the whole module tree.
        x = self.embed(x)
        x = self.block(x)
        return self.head(x)


# Move the demo model to CUDA and bf16 first, then replace layers in-place.
# This matches common usage where the model is already prepared for GPU execution.
model = ToyModel().cuda().bfloat16()
linear_names = [n for n, m in model.named_modules() if type(m) is nn.Linear]
print("Linear layers before replace:", linear_names)

# w_Q and in_Q are both set to hifx4 for a simple end-to-end example.
# quant_grad=False is important for this demo because the user wants backward to
# keep original gradients without quantization.
replace_linear(model, "hifx4", in_Q="hifx4", quant_grad=False)

replaced_names = [n for n, m in model.named_modules() if isinstance(m, QLinear)]
remaining_linear_names = [n for n, m in model.named_modules() if type(m) is nn.Linear]

print("QLinear layers after replace:", replaced_names)
print("Remaining nn.Linear layers:", remaining_linear_names)

# Structural check: every original Linear layer should now be a QLinear, and
# there should be no plain nn.Linear left in the module tree.
assert replaced_names == linear_names
assert not remaining_linear_names

# Run one bf16 forward/backward step to show the replaced model is executable.
# Shape is [batch, seq, hidden]. We use batch=1 and all dimensions as multiples
# of 128 to better mimic common model-side tensor shapes.
x = torch.randn(1, 128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
out = model(x)

# Compute the loss in fp32 for a stable scalar reduction.
loss = out.float().square().mean()
loss.backward()

# Basic runtime checks: backward should populate input grad, and both outputs and
# gradients should remain finite after quantized forward.
assert x.grad is not None, "Model backward did not produce input gradients"
assert torch.isfinite(out.float()).all().item()
assert torch.isfinite(x.grad.float()).all().item()

# Parameter-gradient checks: every replaced QLinear should still receive a valid
# weight gradient, which is the most practical confirmation that training wiring
# remains intact after replacement.
for name in replaced_names:
    module = dict(model.named_modules())[name]
    assert module.weight.grad is not None, f"{name}.weight grad is missing"
    assert torch.isfinite(module.weight.grad.float()).all().item(), f"{name}.weight grad is not finite"

# Print a few friendly demo signals so the script output is easy to present.
print("Output dtype:", out.dtype)
print("Loss:", float(loss))
print("replace_linear bf16 test passed")
