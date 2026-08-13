import numpy as np
import torch

from quant_cy import QType, quant_dequant_float, quant_func


# This demo focuses on quant_func:
# 1. Forward path: quant_func should produce the same quantized-dequantized result
#    as quant_dequant_float when both use the same qtype.
# 2. Backward path: quant_func is implemented as a straight-through estimator,
#    so dL/dx should be exactly the upstream gradient.
#
# We use bf16 inputs because that is the target demo scenario on CUDA.
np.random.seed(42)
torch.manual_seed(42)

# Use model-like sizes that are all multiples of 128.
# This better matches common tensor shapes seen in real workloads.
M = 128
N = 256

# Build a deterministic floating-point test tensor, then cast to bf16.
# The random distribution mixes Gaussian noise with a small uniform offset so the
# tensor contains both positive and negative values with different magnitudes.
x = (0.2 * np.random.randn(M, N) + np.random.uniform(-0.03, 0.04, (M, N))).astype(np.float32)
x_torch = torch.from_numpy(x).bfloat16()

# grad_seed is the manually constructed upstream gradient used to verify backward.
# Since quant_func backward should be identity on gradients, x.grad should match it.
grad_seed = torch.randint(-3, 4, (M, N), dtype=torch.int32).to(torch.bfloat16)

print(x_torch.shape)

# q_dim=0 means quantization is applied along the first dimension of the 2D tensor.
qtype_str = "hifx4"
print("Qtype string: %s " % (qtype_str))
quant_type = QType(qtype_str).dim(0)

# Reference output:
# force_py=True uses the Python implementation;
# force_fp32=True keeps the reference numerically stable before converting back.
y_ref = quant_dequant_float(x_torch, quant_type, force_py=True, force_fp32=True).cpu().float()

# CUDA path under test:
# quant_func wraps the quantization op in an autograd Function so it can participate
# in backward, unlike a pure no_grad quant_dequant helper.
x_cuda = x_torch.cuda().detach().clone().requires_grad_(True)
grad_seed_cuda = grad_seed.cuda()
y_cuda = quant_func(x_cuda, quant_type, force_py=False)

# Forward check: kernel-backed quant_func should match the quant_dequant reference.
forward_diff = (y_ref - y_cuda.detach().cpu().float()).abs().max().item()
print("ABS diff max (quant_dequant_float <-> quant_func):", forward_diff)

# Use a simple inner-product style loss so the analytical gradient is easy to read:
# if backward is straight-through, grad(x) should be exactly grad_seed.
loss = (y_cuda.float() * grad_seed_cuda.float()).sum()
loss.backward()

assert x_cuda.grad is not None, "quant_func backward did not produce input gradients"

# Backward check: quant_func backward should pass gradients through unchanged.
grad_diff = (x_cuda.grad.float() - grad_seed_cuda.float()).abs().max().item()
print("ABS diff max (input grad <-> upstream grad):", grad_diff)

# For this demo we expect exact match, not just approximate match.
assert forward_diff == 0.0
assert grad_diff == 0.0
print("quant_func bf16 test passed")
