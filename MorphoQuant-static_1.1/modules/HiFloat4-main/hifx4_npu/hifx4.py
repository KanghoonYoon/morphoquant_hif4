import numpy as np
import torch
import torch_npu

import HiF4_NVFP4_v14f16
from quant_cy_npu import QType, quant_dequant_float


np.random.seed(42)
torch.manual_seed(42)

N = 512
M = 512

x = (0.2 * np.random.randn(M, N) + np.random.uniform(-0.03, 0.04, (M, N))).astype(np.float32)
x_torch = torch.from_numpy(x).npu()
print(x.shape)

qtype_str = 'hifx4'
print('Qtype string: %s ' % (qtype_str))
quant_type = QType(qtype_str).dim(0)

y0 = HiF4_NVFP4_v14f16.To_HiFX(x, N=4)
y1 = quant_dequant_float(x_torch.cpu(), quant_type, force_py=True, force_fp32=True).cpu().numpy()
y2 = quant_dequant_float(x_torch, quant_type, force_py=False, force_fp32=True).cpu().numpy()

diff = np.abs(y0 - y1)
print('ABS diff max (numpy <-> torch ):', np.max(diff))
diff = np.abs(y0 - y2)
print('ABS diff max (numpy <-> kernel):', np.max(diff))
diff = np.abs(y1 - y2)
print('ABS diff max (torch <-> kernel):', np.max(diff))

print('Testing zero values')
y0 = HiF4_NVFP4_v14f16.To_HiFX(x * 0, N=4)
y1 = quant_dequant_float(x_torch.cpu() * 0, quant_type, force_py=True, force_fp32=True).cpu().numpy()
y2 = quant_dequant_float(x_torch * 0, quant_type, force_py=False, force_fp32=True).cpu().numpy()
diff = np.abs(y0 - y1)
print('ABS diff max (zero values, numpy <-> torch ):', np.max(diff))
diff = np.abs(y0 - y2)
print('ABS diff max (zero values):', np.max(diff))
diff = np.abs(y1 - y2)
print('ABS diff max (zero values, torch <-> kernel):', np.max(diff))
