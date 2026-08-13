#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*

import torch
import time
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from .quant_utils import *
import sys
from tqdm import tqdm


def _default_device() -> str:
    """CUDA / Ascend NPU / CPU 自动选择。

    与 modules/device_utils.py 逻辑一致，但这里刻意内联，
    避免 bitsandbytes 反向依赖 MorphoQuant 的 modules 包。
    可用 MORPHOQUANT_DEVICE 环境变量强制指定。
    """
    import os

    forced = os.environ.get("MORPHOQUANT_DEVICE", "").strip().lower()
    if forced in ("npu", "cuda", "cpu"):
        return forced
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


last_layer_entropy = 0
# NOTE: 这里原本硬编码 .cuda()，在没有 CUDA 的机器上会让本模块 import 直接失败。
last_layer_distribution = torch.Tensor(np.zeros([1,100,4096])).to(_default_device())
llama_entropy = []
llama_distribution = []

class QuantAct(Module):
    """
    Class to quantize given activations
    """
    def __init__(self,
                 activation_bit=16,
                 # full_precision_flag=False,
                 running_stat=False,
                 # beta=0.9, 
                 input_dim=4096,
                 llama_layer=True,
                 count_block=1, 
                 count_layer=1):
        """
        activation_bit: bit-setting for activation
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(QuantAct, self).__init__()
        self.activation_bit = activation_bit
        # print("Activation quantization bit:", self.activation_bit)
        self.momentum = 0.99
        # self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.llama_layer = llama_layer

        self.init_range = 6.
        self.dim = input_dim
        self.count_block = count_block
        self.count_layer = count_layer
        self.search_flag = True
        self.sample_num = 0
        self.last_entropy = 0
        self.first_search = True
        self.debug = False
        if self.llama_layer == True:
            # llama calibrate search
            self.llama_range_min = torch.Tensor(-self.init_range * np.zeros(self.dim)).to(_default_device())
            self.llama_range_max = torch.Tensor(self.init_range * np.zeros(self.dim)).to(_default_device())
        else:
            # CLIP calibrate search
            CLIP_row_dim = 257 # v1.3
            # CLIP_row_dim = 577 # v1.5 (position_embedding): Embedding(577, 1024)
            self.CLIP_range_min = torch.Tensor(-self.init_range * np.zeros(CLIP_row_dim)).to(_default_device())
            self.CLIP_range_max = torch.Tensor(self.init_range * np.zeros(CLIP_row_dim)).to(_default_device())

        self.layer_name = ''
        self.group_num = 8

        self.act_function = AsymmetricQuantFunction.apply
        self._calibrate = False
        self.search = False

        self.gamma_inf = 0.5
        self.gamma_cos = 1.0
        self.lp_norm = 0.5
        self.use_cosine_loss = False
        self.sparse_buffer_ratio = 0.8
        self.search_ratio_lower_bound = 0.6
        self.search_ema_momentum = 0.5
        self._search_best_score = None
        self._search_best_min = None
        self._search_best_max = None
        self._search_score_ema = None
        self._temp_best_min = None
        self._temp_best_max = None

    def set_calibrate(self, calibrate=True):
        self._calibrate = calibrate

    def set_search(self, search=True):
        self.search = search

    def set_gamma(self, gamma_inf=0.5, gamma_cos=1.0):
        self.gamma_inf = gamma_inf
        self.gamma_cos = gamma_cos
        
    def set_lp_norm(self, lp_norm=0.5):
        self.lp_norm = lp_norm

    def set_cosine_loss(self, use_cosine_loss=False):
        self.use_cosine_loss = use_cosine_loss

    def set_sparse_buffer_ratio(self, sparse_buffer_ratio=0.8):
        self.sparse_buffer_ratio = sparse_buffer_ratio

    def set_search_ratio_lower_bound(self, search_ratio_lower_bound=0.6):
        self.search_ratio_lower_bound = search_ratio_lower_bound

    def set_search_ema_momentum(self, search_ema_momentum=0.5):
        self.search_ema_momentum = search_ema_momentum

    def reset_search_best(self):
        self._search_best_score = None
        self._search_best_min = None
        self._search_best_max = None
        self._search_score_ema = None

    def set_debug(self, debug=False):
        self.debug = debug
        print("QuantAct debug mode set to:", self.debug)

    def analyze_outliers(self, inputs, min_val, max_val, k_ratio=0.001):
        """
        分析 Top-K 离群点的状态：是在量化范围内被保留，还是被截断。
        """
        # 展平输入
        flat_input = inputs.view(-1)
        total_elements = flat_input.numel()
        k = max(1, int(total_elements * k_ratio))
        
        # 找到绝对值最大的 Top-K 元素的索引
        topk_vals, topk_indices = torch.topk(flat_input.abs(), k)
        # 获取原始带符号数值
        topk_orig = flat_input[topk_indices]
        
        # 简化版：只统计全局最大的那些点是否超出了全局最大的阈值范围
        global_min = min_val.min().item() if isinstance(min_val, torch.Tensor) else min_val
        global_max = max_val.max().item() if isinstance(max_val, torch.Tensor) else max_val
        
        outside_mask = (topk_orig > global_max) | (topk_orig < global_min)
        num_outside = outside_mask.sum().item()
        num_inside = k - num_outside
        
        print(f"\n[Layer {self.layer_name} Outlier Analysis (Top {k_ratio*100:.2f}% = {k} points)]")
        print(f"  Thresholds: Min={global_min:.4f}, Max={global_max:.4f}")
        print(f"  Max Val in Input: {topk_orig.abs().max().item():.4f}")
        print(f"  👉 Saved (Inside): {num_inside} ({num_inside/k*100:.1f}%)")
        print(f"  👉 Clipped (Outside): {num_outside} ({num_outside/k*100:.1f}%)")
        
        if num_outside > 0:
            # 粗略估计截断误差（假设是对称截断，只看绝对值越界部分）
            # 实际上每个数值应该对应具体的 channel-wise min/max，这里仅作宏观参考
            avg_clip_err = (topk_orig[outside_mask].abs() - global_max).mean().item()
            print(f"  ⚠️ Avg Max-Clipping Error: {avg_clip_err:.4f}")

    def quantization(self, inputs, quantization_min, quantization_max):
        scale, zero_point = asymmetric_linear_quantization_params(
            self.activation_bit, quantization_min , quantization_max
        )
        # print(inputs.shape[-1], scale.shape[0], inputs.shape[-1]==scale.shape[0])
        if inputs.shape[-1] == scale.shape[0]:
            # print(inputs.shape, scale.shape)  # torch.Size([8, 638, 4096]) torch.Size([4096])
            new_quant_x = torch.round(scale * inputs - zero_point)
            n = 2**(self.activation_bit - 1)
            new_quant_x_1 = 0.5 * ((-new_quant_x - n).abs() - (new_quant_x - (n - 1)).abs() - 1)
            quant_act = (new_quant_x_1 + zero_point) / scale
            if quant_act.dtype != inputs.dtype:
                quant_act = quant_act.to(inputs.dtype)
            return quant_act
        else:            
            # 1. 自动对齐 scale 和 zero_point 维度
            scale_expanded = scale
            zero_point_expanded = zero_point
            
            # 将 [Channel] 扩展为 [1, ..., Channel, ...] 匹配 inputs_calibrate 进行广播
            # target_dim是Channel数对应的维度索引，寻找 scale.shape[0] 在 inputs.shape 中的位置
            dim_match = [i for i, d in enumerate(inputs.shape) if d == scale.shape[0]]
            # 默认假设是在倒数第二维，如果是视觉张量的话。如果没找到就算了
            target_axis = dim_match[-1] if len(dim_match) > 0 else -1

            # 构建广播形状
            view_shape = [1] * inputs.ndim
            if target_axis != -1:
                view_shape[target_axis] = scale.shape[0]
            else:
                #  fallback，如果都没匹配上，假设在倒数第二维
                view_shape[-2] = scale.shape[0]

            scale_expanded = scale.view(*view_shape)
            if isinstance(zero_point, torch.Tensor):
                zero_point_expanded = zero_point.view(*view_shape)

            # 2. 量化计算 (使用广播机制)
            new_quant_x = torch.round(scale_expanded * inputs - zero_point_expanded)
            
            # 3. 反量化计算
            quant_act = (new_quant_x + zero_point_expanded) / scale_expanded

            if quant_act.dtype != inputs.dtype:
                quant_act = quant_act.to(inputs.dtype)

            return quant_act

    def compute_DED(self, p_k, p_k1):
        # calcuate D(k, {k+1}) = -sum_ij p(x_{q,ij}^{(k)}, x_{q,ij}^{(k+1)}) log p(x_{q,ij}^{(k+1)} | x_{q,ij}^{(k)})
        
        p_k = F.normalize(p_k, p=1, dim=1)  
        p_k1 = F.normalize(p_k1, p=1, dim=1)  

        if p_k.shape != p_k1.shape:
            # 如果返回值需要是 Tensor，请确保设备和类型一致
            return torch.tensor(0.0, device=p_k.device, dtype=p_k.dtype)
        else:
            joint_p = p_k * p_k1  
            joint_p = joint_p / joint_p.sum(dim=1, keepdim=True)  
            condition_p = p_k1 / (p_k + 1e-5)  
            condition_p = condition_p / condition_p.sum(dim=1, keepdim=True)
            # print(joint_p, condition_p)
            return -1 * torch.sum(joint_p * torch.log(condition_p + 1e-5), dim=1).mean()
    
    def cal_entropy(self, attn):
        attn = torch.nn.functional.normalize(attn, dim=1)
        # print(attn.shape, self.count_block, self.count_layer)
        return -1 * torch.sum((attn * torch.log(attn+1e-7)), dim=1).mean()
    
    def cal_score_channel(self, quant_act, inputs_calibrate, weight_kl=0.5):
        # calculate the per-channel score

        # MSE
        # [Batch, Seq, Channel] -> [Channel]
        # 注意: 假设 inputs_calibrate 是 [Batch, Seq, Channel]
        reduce_dims = list(range(inputs_calibrate.dim() - 1)) # 除最后一维外的所有维度
        l_mse = (quant_act - inputs_calibrate).pow(2).mean(dim=reduce_dims) 
        # print("l_mse", l_mse.shape)

        # KL Divergence
        # 加 eps 防止除以 0 导致 NaN (尽管 normalize 内部可能有处理，加一层保险)
        # p = F.normalize(inputs_calibrate.abs() + 1e-7, p=1, dim=1) 
        # q = F.normalize(quant_act.abs() + 1e-7, p=1, dim=1)
        
        # # 1. 给 q 加 eps 防止 log(0) -> -inf
        # # 2. reduction='none' 加上 sum(dim=1).mean(dim=0) 以保持 [Channel] 形状
        # #    dim=1 是 Seq 维度 (概率分布的维度)，KL 在这个维度求和
        # log_q = (q + 1e-7).log()
        
        # # [Batch, Seq, Channel] -> [2, 2048, 4096]
        # kl_per_token = F.kl_div(log_q, p, reduction='none') 
        
        # # Sum over distribution dim (Seq=1), Mean over Batch (0)
        # # 如果 dim=1 是 Seq维度
        # l_kl = kl_per_token.sum(dim=1).mean(dim=0) 
        
        # print("l_kl", l_kl.shape)

        # cosine similarity loss(per channel)
        # should be [channel]
        # [Fix] 适配动态 Channel 维度（情况 B 已经确保 x_flat 沿 seq/spatial 维度展开）
        # 让 q_flat / x_flat 正确匹配 [-1, Channel]
        target_dim = self.llama_range_min.shape[0] if self.llama_layer else self.CLIP_range_min.shape[0]
        if inputs_calibrate.shape[-1] == target_dim:
            x_flat = inputs_calibrate.reshape(-1, target_dim).float()
            q_flat = quant_act.reshape(-1, target_dim).float()
        elif inputs_calibrate.ndim >= 2 and inputs_calibrate.shape[-2] == target_dim:
            x_flat = inputs_calibrate.transpose(-1, -2).reshape(-1, target_dim).float()
            q_flat = quant_act.transpose(-1, -2).reshape(-1, target_dim).float()
        else:
            x_flat = inputs_calibrate.reshape(-1, inputs_calibrate.shape[-1]).float()
            q_flat = quant_act.reshape(-1, quant_act.shape[-1]).float()
        
        # [Fix] 添加 eps 防止除零，并手动 clamp 结果
        s_cos = torch.nn.functional.cosine_similarity(x_flat, q_flat, dim=0, eps=1e-6)  # [Channel]
        s_cos = torch.clamp(s_cos, min=-1.0, max=1.0)
        
        # score = l_mse + l_kl * weight_kl
        # print("s_cos!")
        # print(s_cos[s_cos > 1])

        # l_0.5 loss
        lploss = (quant_act-inputs_calibrate).abs().pow(self.lp_norm).mean(dim=reduce_dims)

        if self.use_cosine_loss:
            score = -lploss + self.gamma_cos * s_cos
        else:
            score = -lploss
        # score = -l_mse
        # score = s_cos   # 越接近 1 越好，所以直接用余弦相似度作为分数，搜索时选择最高的那个
        return score
    
    def search_strategy_judge(self):
        self.sample_num += 1
        global last_layer_entropy, llama_entropy
        if last_layer_entropy >= np.mean(llama_entropy) or self.count_block % 3 == 1:
            search_flag = True
        else:
            search_flag = False

        if (self.count_block == 1 and self.count_layer == 1) or self.sample_num <= 1:
            search_flag = True
            llama_entropy = []

        return search_flag

    def calibrate_quantization(self, inputs, init_min=-6, init_max=6):
        if self.llama_layer == True:
            self.search_flag = self.search_strategy_judge()
                
            if self.search_flag:
                # x_min = torch.min(inputs, dim=1)[0].squeeze(dim=0)
                # x_max = torch.max(inputs, dim=1)[0].squeeze(dim=0)
                # --- 🟢 替换为以下智能判断维度的代码 ---
                target_dim = self.llama_range_min.shape[0]  # 这里应该是 1280
                
                # 情况 A: 正常 LLM 格式 [..., Seq, Channel(1280)]
                if inputs.shape[-1] == target_dim:
                    # 展平除最后一维外的所有维度
                    inputs_flat = inputs.reshape(-1, target_dim)
                    x_max = inputs_flat.abs().max(dim=0)[0]
                    x_min = inputs_flat.min(dim=0)[0]
                    
                # 情况 B: 视觉部分转置格式 [..., Channel(1280), Seq]
                elif inputs.ndim >= 2 and inputs.shape[-2] == target_dim:
                    # 先转置成 [..., Seq, Channel]，再展平
                    inputs_transposed = inputs.transpose(-1, -2)
                    inputs_flat = inputs_transposed.reshape(-1, target_dim)
                    x_max = inputs_flat.abs().max(dim=0)[0]
                    x_min = inputs_flat.min(dim=0)[0]
                    
                # 情况 C: 兜底 (如果前面都没匹配上，保持原逻辑防止 crash，虽然可能还是会报错)
                else:
                    # 尝试暴力 reduce 到最后一维
                    inputs_flat = inputs.reshape(-1, inputs.shape[-1])
                    x_max = inputs_flat.abs().max(dim=0)[0]
                    x_min = inputs_flat.min(dim=0)[0]

                # --- 🟢 替换结束 ---

                # in-place operation used on multi-gpus
                # in-place！！search
                # print("LLAMA range updated to minmax")
                self.llama_range_min += -self.llama_range_min + torch.min(self.llama_range_min, x_min)
                self.llama_range_max += -self.llama_range_max + torch.max(self.llama_range_max, x_max)

            quant_act = self.quantization(inputs, self.llama_range_min, self.llama_range_max)
            global last_layer_entropy, last_layer_distribution
            if self.count_layer == 1 or self.count_layer == 7:
                last_layer_entropy = self.cal_entropy(quant_act.abs())
            else:
                last_layer_entropy = self.compute_DED(last_layer_distribution, quant_act.abs())
            last_layer_distribution = quant_act.abs()
            if not np.isnan(last_layer_entropy.item()):
                llama_entropy.append(last_layer_entropy.item())
            # print("last_layer_entropy", last_layer_entropy, self.count_block, self.count_layer)

            return quant_act
        else:
            # row-wise search
            x_min = torch.min(inputs, dim=-1)[0].squeeze(dim=0)
            x_max = torch.max(inputs, dim=-1)[0].squeeze(dim=0)
            # in-place operation used on multi-gpus
            self.CLIP_range_min += -self.CLIP_range_min + torch.min(self.CLIP_range_min, x_min)
            self.CLIP_range_max += -self.CLIP_range_max + torch.max(self.CLIP_range_max, x_max)
            # print(self.CLIP_range_min, self.CLIP_range_max)
            quant_act = self.quantization(inputs, self.CLIP_range_min , self.CLIP_range_max)
            return quant_act
    
    def forward(self, x):
        """
        quantize given activation x
        """
        inputs_calibrate = x.data

        # calibrate stage
        if self._calibrate:
            # === 全新的静态校准数据收集阶段 ===
            target_dim = self.llama_range_min.shape[0] if self.llama_layer else self.CLIP_range_min.shape[0]
            
            # 将多维数据展平为 [N, Channel]
            if inputs_calibrate.shape[-1] == target_dim:
                flat_x = inputs_calibrate.reshape(-1, target_dim)
            elif inputs_calibrate.ndim >= 2 and inputs_calibrate.shape[-2] == target_dim:
                flat_x = inputs_calibrate.transpose(-1, -2).reshape(-1, target_dim)
            else:
                flat_x = inputs_calibrate.reshape(-1, inputs_calibrate.shape[-1])
            
            # 计算当前 Batch 的统计特征
            cur_min = flat_x.min(dim=0)[0]  
            cur_max = flat_x.max(dim=0)[0]
            cur_sum = flat_x.sum(dim=0)
            # 使用 float() 防止平方时范围溢出 (尤其是半精度下)
            cur_sq_sum = (flat_x.float() ** 2).sum(dim=0)
            cur_abs_sum = flat_x.abs().sum(dim=0)
            cur_count = flat_x.shape[0]

            if not hasattr(self, 'calib_count') or self.calib_count == 0:
                self.calib_min = cur_min
                self.calib_max = cur_max
                self.calib_sum = cur_sum
                self.calib_sq_sum = cur_sq_sum
                self.calib_count = cur_count
                self.calib_abs_sum = cur_abs_sum
            else:
                self.calib_min = torch.min(self.calib_min, cur_min)
                self.calib_max = torch.max(self.calib_max, cur_max)
                self.calib_sum += cur_sum
                self.calib_sq_sum += cur_sq_sum
                self.calib_count += cur_count
                self.calib_abs_sum += cur_abs_sum

            # 保留对原有 min/max 变量的更新，使得不修改后续代码的情况下推理也能用
            if self.llama_layer:
                self.llama_range_min = self.calib_min
                self.llama_range_max = self.calib_max
            else:
                self.CLIP_range_min = self.calib_min
                self.CLIP_range_max = self.calib_max

            # 在校准期，不执行量化计算，直接返回原激活值供下一层收集
            return x

        # search stage
        # elif self.search:
        #     # print("search stage")
        #     # 获取通道数
        #     channel_dim = inputs_calibrate.shape[-1]

        #     # 如果你存了 mask 这里可以直接获取
        #     outlier_mask = getattr(self, 'outlier_mask', None)
            
        #     # 1. 确定搜索基准范围
        #     # 优先使用校准收集到的全样本极值，如果没有则使用当前 batch 极值
        #     if hasattr(self, 'calib_min') and hasattr(self, 'calib_max'):
        #         base_min = self.calib_min.clone().to(inputs_calibrate.device).to(inputs_calibrate.dtype)
        #         base_max = self.calib_max.clone().to(inputs_calibrate.device).to(inputs_calibrate.dtype)
        #     else:
        #         x_flat = inputs_calibrate.reshape(-1, channel_dim)
        #         base_min = x_flat.min(dim=0)[0]
        #         base_max = x_flat.max(dim=0)[0]
            
        #     # 根据outlier_mask决定，普通通道使用对称最大范围，离群通道使用对称最小范围作为alpha
        #     sym_edge_max = torch.max(base_min.abs(), base_max.abs())
        #     sym_edge_min = torch.min(base_min.abs(), base_max.abs())
        #     sym_mask = base_min * base_max <= 0

        #     if outlier_mask is not None:
        #         outlier_mask = outlier_mask.to(inputs_calibrate.device).bool()
        #         alpha = torch.where(outlier_mask, sym_edge_min, sym_edge_max)
        #     else:
        #         alpha = sym_edge_max
        #         outlier_mask = torch.zeros_like(base_min, dtype=torch.bool)

        #     sym_min = torch.where(sym_mask, -alpha, base_min)
        #     sym_max = torch.where(sym_mask, alpha, base_max)

        #     # 补偿比例 lambda
        #     sparse_buffer_ratio = self.sparse_buffer_ratio

        #     # 2. 网格搜索（按对数规律取点）
        #     #    在 [search_ratio_lower_bound, 1.0] 区间内按对数规律均匀采样
        #     search_ratio_lower_bound = float(getattr(self, 'search_ratio_lower_bound', 0.6))
        #     num_grid_points = 15  # 对数空间采样点数

        #     # 对数空间采样: 在 [log10(lower), log10(1.0)] 区间均匀取 15 个点
        #     log_low = math.log10(search_ratio_lower_bound)
        #     log_high = math.log10(1.0)  # = 0
        #     ratio_grid = torch.logspace(log_low, log_high, num_grid_points)  # [num_grid_points]

        #     layer_name = getattr(self, 'layer_name', 'Unknown Layer')
        #     batch_score_list = []
        #     # for r_val in tqdm(ratio_grid.tolist(), desc=f"Grid search {layer_name}", leave=False):
        #     for r_val in ratio_grid.tolist():
        #         r_tensor = torch.full_like(base_min, r_val, dtype=torch.float32)

        #         cm  = sym_min * r_tensor;   cmx = sym_max * r_tensor
        #         cm  = cm.to(inputs_calibrate.device).to(inputs_calibrate.dtype)
        #         cmx = cmx.to(inputs_calibrate.device).to(inputs_calibrate.dtype)
        #         q_cur = self.quantization(inputs_calibrate, cm, cmx)
        #         if q_cur.dtype != inputs_calibrate.dtype: q_cur = q_cur.to(inputs_calibrate.dtype)
        #         lim_cur = torch.max(cmx.abs(), cm.abs())
        #         if lim_cur.dim() == 1:
        #             if inputs_calibrate.ndim == 3: lim_cur = lim_cur.view(1, 1, -1)
        #             elif inputs_calibrate.ndim == 2: lim_cur = lim_cur.view(1, -1)
        #         iso_cur = (inputs_calibrate.abs() > (lim_cur * sparse_buffer_ratio)) & outlier_mask.view_as(lim_cur)
        #         score_cur = self.cal_score_channel(torch.where(iso_cur, inputs_calibrate, q_cur), inputs_calibrate)
        #         batch_score_list.append(score_cur.detach())

        #     # 当前 search 由多个 batch/forward 组成：对每个 batch 的完整 ratio-score 网格做滑动平均。
        #     # 最终按滑动平均后的分数逐通道选最优 r，避免单个 batch 的偶然最优值主导搜索结果。
        #     batch_score_grid = torch.stack(batch_score_list, dim=0).float()  # [num_grid_points, channel]
        #     if (
        #         self._search_score_ema is None
        #         or self._search_score_ema.shape != batch_score_grid.shape
        #     ):
        #         self._search_score_ema = batch_score_grid.clone()
        #     else:
        #         momentum = float(getattr(self, 'search_ema_momentum', 0.5))
        #         momentum = max(0.0, min(1.0, momentum))
        #         self._search_score_ema = self._search_score_ema.to(batch_score_grid.device)
        #         self._search_score_ema.mul_(momentum).add_(batch_score_grid, alpha=1 - momentum)

        #     best_grid_idx = torch.argmax(self._search_score_ema, dim=0)
        #     best_score = self._search_score_ema.gather(0, best_grid_idx.unsqueeze(0)).squeeze(0)
        #     ratio_grid = ratio_grid.to(inputs_calibrate.device).to(inputs_calibrate.dtype)
        #     best_r_tensor = ratio_grid[best_grid_idx].to(sym_min.dtype)
        #     best_min = sym_min * best_r_tensor
        #     best_max = sym_max * best_r_tensor

        #     # 3. 固化参数用于推断
        #     self._search_best_score = best_score.detach().clone()
        #     self._search_best_min = best_min.clone()
        #     self._search_best_max = best_max.clone()

        #     self.activation_range_min = self._search_best_min
        #     self.activation_range_max = self._search_best_max
        #     if self.llama_layer:
        #         self.llama_range_min = self._search_best_min
        #         self.llama_range_max = self._search_best_max
        #     else:
        #         self.CLIP_range_min = self._search_best_min
        #         self.CLIP_range_max = self._search_best_max
            
        #     # 保存补偿范围用来推理
        #     self.compensation_limit = torch.max(self._search_best_max.abs(), self._search_best_min.abs()) * sparse_buffer_ratio
                
        #     quant_act_main = self.quantization(inputs_calibrate, self._search_best_min, self._search_best_max)
        #     limit_max_final = torch.max(self._search_best_max.abs(), self._search_best_min.abs())
        #     if limit_max_final.dim() == 1:
        #          if inputs_calibrate.ndim == 3: limit_max_final = limit_max_final.view(1, 1, -1)
        #          elif inputs_calibrate.ndim == 2: limit_max_final = limit_max_final.view(1, -1)
        #     sparse_mask = (inputs_calibrate.abs() > (limit_max_final * sparse_buffer_ratio)) & outlier_mask.view_as(limit_max_final)
        #     quant_act = torch.where(sparse_mask, inputs_calibrate, quant_act_main)
            
        #     # 此处移除了 self.search = False 以支持多 batch search
        #     # 我们在 evaluator.py 的 search loop 结束后手动设为 False
        #     return quant_act

        # elif self.search:
        #     channel_dim = inputs_calibrate.shape[-1]
        #     device = inputs_calibrate.device
        #     dtype = inputs_calibrate.dtype

        #     outlier_mask = getattr(self, 'outlier_mask', None)
        #     if outlier_mask is not None:
        #         outlier_mask = outlier_mask.to(device).bool()
            
        #     # 1. 确定搜索基准范围
        #     if hasattr(self, 'calib_min') and hasattr(self, 'calib_max'):
        #         base_min = self.calib_min.clone().to(device).to(dtype)
        #         base_max = self.calib_max.clone().to(device).to(dtype)
        #     else:
        #         x_flat = inputs_calibrate.reshape(-1, channel_dim)
        #         base_min = x_flat.min(dim=0)[0]
        #         base_max = x_flat.max(dim=0)[0]
            
        #     sym_edge_max = torch.max(base_min.abs(), base_max.abs())
        #     sym_edge_min = torch.min(base_min.abs(), base_max.abs())
        #     sym_mask = base_min * base_max <= 0

        #     if outlier_mask is not None:
        #         alpha = torch.where(outlier_mask, sym_edge_min, sym_edge_max)
        #     else:
        #         alpha = sym_edge_max
        #         outlier_mask = torch.zeros_like(base_min, dtype=torch.bool)

        #     sym_min = torch.where(sym_mask, -alpha, base_min)
        #     sym_max = torch.where(sym_mask, alpha, base_max)

        #     sparse_buffer_ratio = getattr(self, 'sparse_buffer_ratio', 0.8)

        #     # 2. 网格候选集与批量分数存储
        #     ratios = [1.0, 0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6]
        #     num_ratios = len(ratios)
        #     batch_score_grid = torch.empty((num_ratios, channel_dim), device=device, dtype=torch.float32)

        #     # 物理级静音与防 OOM 推断模式
        #     import warnings
        #     with warnings.catch_warnings(), torch.inference_mode():
        #         warnings.simplefilter("ignore")
                
        #         for idx, r_val in enumerate(ratios):
        #             c_min = sym_min * r_val
        #             c_max = sym_max * r_val
                    
        #             c_min_d = c_min.to(device).to(dtype)
        #             c_max_d = c_max.to(device).to(dtype)
                    
        #             q_cand = self.quantization(inputs_calibrate, c_min_d, c_max_d)
        #             if q_cand.dtype != dtype: 
        #                 q_cand = q_cand.to(dtype)
                    
        #             limit_max = torch.max(c_max_d.abs(), c_min_d.abs())
        #             if limit_max.dim() == 1:
        #                 view_shape = [1] * (inputs_calibrate.ndim - 1) + [-1]
        #                 limit_max = limit_max.view(*view_shape)
                    
        #             is_true_outlier = (inputs_calibrate.abs() > (limit_max * sparse_buffer_ratio)) & outlier_mask.view_as(limit_max)
        #             hybrid_prediction = torch.where(is_true_outlier, inputs_calibrate, q_cand)
                    
        #             curr_score = self.cal_score_channel(hybrid_prediction, inputs_calibrate)
        #             batch_score_grid[idx] = curr_score.float()

        #     # [核心机制]: 多 Batch 滑动平均 (EMA) 更新
        #     if getattr(self, '_search_score_ema', None) is None or self._search_score_ema.shape != batch_score_grid.shape:
        #         self._search_score_ema = batch_score_grid.clone()
        #     else:
        #         momentum = float(getattr(self, 'search_ema_momentum', 0.5))
        #         self._search_score_ema.mul_(momentum).add_(batch_score_grid, alpha=1 - momentum)

        #     # 根据 EMA 全局分数选出最优的 ratio 索引
        #     best_idx = torch.argmax(self._search_score_ema, dim=0) # [channel_dim]
        #     ratios_tensor = torch.tensor(ratios, device=device, dtype=sym_min.dtype)
        #     best_r_tensor = ratios_tensor[best_idx]

        #     best_min = sym_min * best_r_tensor
        #     best_max = sym_max * best_r_tensor

        #     # 3. 固化参数用于推断
        #     self.activation_range_min = best_min
        #     self.activation_range_max = best_max
        #     if getattr(self, 'llama_layer', False):
        #         self.llama_range_min = best_min
        #         self.llama_range_max = best_max
        #     else:
        #         self.CLIP_range_min = best_min
        #         self.CLIP_range_max = best_max
            
        #     self.compensation_limit = torch.max(best_max.abs(), best_min.abs()) * sparse_buffer_ratio
                
        #     quant_act_main = self.quantization(inputs_calibrate, best_min, best_max)
        #     limit_max_final = torch.max(best_max.abs(), best_min.abs())
        #     if limit_max_final.dim() == 1:
        #         view_shape = [1] * (inputs_calibrate.ndim - 1) + [-1]
        #         limit_max_final = limit_max_final.view(*view_shape)
                 
        #     sparse_mask = (inputs_calibrate.abs() > (limit_max_final * sparse_buffer_ratio)) & outlier_mask.view_as(limit_max_final)
        #     quant_act = torch.where(sparse_mask, inputs_calibrate, quant_act_main)
            
        #     return quant_act
        
        elif self.search:
            channel_dim = inputs_calibrate.shape[-1]
            device = inputs_calibrate.device
            dtype = inputs_calibrate.dtype

            outlier_mask = getattr(self, 'outlier_mask', None)
            if outlier_mask is not None:
                outlier_mask = outlier_mask.to(device).bool()
            
            # 1. 确定搜索基准范围 (严格依赖全局校准的极值)
            if hasattr(self, 'calib_min') and hasattr(self, 'calib_max'):
                base_min = self.calib_min.clone().to(device).to(dtype)
                base_max = self.calib_max.clone().to(device).to(dtype)
            else:
                # 警告：如果走到这里，说明 Calibration 没做好！
                print("Calibration warning")
                x_flat = inputs_calibrate.reshape(-1, channel_dim)
                base_min = x_flat.min(dim=0)[0]
                base_max = x_flat.max(dim=0)[0]
            
            sym_edge_max = torch.max(base_min.abs(), base_max.abs())
            sym_edge_min = torch.min(base_min.abs(), base_max.abs())
            sym_mask = base_min * base_max <= 0

            if outlier_mask is not None:
                alpha = torch.where(outlier_mask, sym_edge_min, sym_edge_max)
            else:
                alpha = sym_edge_max
                outlier_mask = torch.zeros_like(base_min, dtype=torch.bool)

            sym_min = torch.where(sym_mask, -alpha, base_min)
            sym_max = torch.where(sym_mask, alpha, base_max)
            # asym_min = base_min
            # asym_max = base_max


            sparse_buffer_ratio = getattr(self, 'sparse_buffer_ratio', 0.8)

            # 2. 网格候选集与批量分数存储
            # ratios = [1.0, 0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6]
            # 2. 网格搜索（按对数规律取点）
            #    在 [search_ratio_lower_bound, 1.0] 区间内按对数规律均匀采样
            search_ratio_lower_bound = float(getattr(self, 'search_ratio_lower_bound', 0.4))
            num_grid_points = 15  # 对数空间采样点数

            # 对数空间采样: 在 [log10(lower), log10(1.0)] 区间均匀取 15 个点
            log_low = math.log10(search_ratio_lower_bound)
            log_high = math.log10(1.0)  # = 0
            ratio_grid = torch.logspace(log_low, log_high, num_grid_points)  # [num_grid_points]
            ratios = ratio_grid.tolist()

            num_ratios = len(ratios)
            batch_score_grid = torch.empty((num_ratios, channel_dim), device=device, dtype=torch.float32)
            # print(ratios)

            import warnings
            with warnings.catch_warnings(), torch.inference_mode():
                warnings.simplefilter("ignore")
                
                for idx, r_val in enumerate(ratios):
                    c_min = sym_min * r_val
                    c_max = sym_max * r_val
                    # c_min = asym_min * r_val
                    # c_max = asym_max * r_val
                    
                    c_min_d = c_min.to(device).to(dtype)
                    c_max_d = c_max.to(device).to(dtype)
                    
                    q_cand = self.quantization(inputs_calibrate, c_min_d, c_max_d)
                    if q_cand.dtype != dtype: 
                        q_cand = q_cand.to(dtype)
                    
                    limit_max = torch.max(c_max_d.abs(), c_min_d.abs())
                    if limit_max.dim() == 1:
                        view_shape = [1] * (inputs_calibrate.ndim - 1) + [-1]
                        limit_max = limit_max.view(*view_shape)
                    
                    is_true_outlier = (inputs_calibrate.abs() > (limit_max * sparse_buffer_ratio)) & outlier_mask.view_as(limit_max)
                    hybrid_prediction = torch.where(is_true_outlier, inputs_calibrate, q_cand)
                    
                    curr_score = self.cal_score_channel(hybrid_prediction, inputs_calibrate)
                    batch_score_grid[idx] = curr_score.float()

            # [核心机制]: 多 Batch 滑动平均 (EMA) 更新
            if getattr(self, '_search_score_ema', None) is None or self._search_score_ema.shape != batch_score_grid.shape:
                self._search_score_ema = batch_score_grid.clone()
            else:
                momentum = float(getattr(self, 'search_ema_momentum', 0.5))
                self._search_score_ema.mul_(momentum).add_(batch_score_grid, alpha=1 - momentum)

            # 根据 EMA 全局分数选出最优的 ratio 索引
            best_idx = torch.argmax(self._search_score_ema, dim=0) 
            ratios_tensor = torch.tensor(ratios, device=device, dtype=sym_min.dtype)
            # ratios_tensor = torch.tensor(ratios, device=device, dtype=asym_min.dtype)
            best_r_tensor = ratios_tensor[best_idx]

            best_min = sym_min * best_r_tensor
            best_max = sym_max * best_r_tensor
            # best_min = asym_min * best_r_tensor
            # best_max = asym_max * best_r_tensor

            # =========================================================
            # 【核心修改】：不在这里污染全局推理变量！将最优结果存入临时缓存
            # =========================================================
            self._temp_best_min = best_min
            self._temp_best_max = best_max
                
            # 计算当前 batch 的前向结果，让网络继续走下去
            quant_act_main = self.quantization(inputs_calibrate, best_min, best_max)
            limit_max_final = torch.max(best_max.abs(), best_min.abs())
            if limit_max_final.dim() == 1:
                view_shape = [1] * (inputs_calibrate.ndim - 1) + [-1]
                limit_max_final = limit_max_final.view(*view_shape)
                 
            sparse_mask = (inputs_calibrate.abs() > (limit_max_final * sparse_buffer_ratio)) & outlier_mask.view_as(limit_max_final)
            quant_act = torch.where(sparse_mask, inputs_calibrate, quant_act_main)
            
            return quant_act
        # elif self.search:
        #     channel_dim = inputs_calibrate.shape[-1]
        #     device = inputs_calibrate.device
        #     dtype = inputs_calibrate.dtype
            
        #     # 1. 确定搜索基准范围
        #     if hasattr(self, 'calib_min') and hasattr(self, 'calib_max'):
        #         base_min = self.calib_min.clone().to(device).to(dtype)
        #         base_max = self.calib_max.clone().to(device).to(dtype)
        #     else:
        #         x_flat = inputs_calibrate.reshape(-1, channel_dim)
        #         base_min = x_flat.min(dim=0)[0]
        #         base_max = x_flat.max(dim=0)[0]

        #     outlier_mask = getattr(self, 'outlier_mask', None)
        #     if outlier_mask is not None:
        #         outlier_mask = outlier_mask.to(device).bool()
        #     else:
        #         outlier_mask = torch.zeros_like(base_min, dtype=torch.bool)

        #     # =========================================================
        #     # 【核心修改】：移除所有对称强转逻辑，直接采用真实极值作为非对称边界
        #     # =========================================================
        #     asym_min = base_min
        #     asym_max = base_max

        #     sparse_buffer_ratio = getattr(self, 'sparse_buffer_ratio', 0.8)

        #     # 2. 网格候选集与批量分数存储
        #     ratios = [1.0, 0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6]
        #     num_ratios = len(ratios)
        #     batch_score_grid = torch.empty((num_ratios, channel_dim), device=device, dtype=torch.float32)

        #     # 物理级静音与防 OOM 推断模式
        #     import warnings
        #     with warnings.catch_warnings(), torch.inference_mode():
        #         warnings.simplefilter("ignore")
                
        #         for idx, r_val in enumerate(ratios):
        #             # 【核心修改】：分别按比例缩小非对称的真实的 Min 和 Max
        #             c_min = asym_min * r_val
        #             c_max = asym_max * r_val
                    
        #             c_min_d = c_min.to(device).to(dtype)
        #             c_max_d = c_max.to(device).to(dtype)
                    
        #             q_cand = self.quantization(inputs_calibrate, c_min_d, c_max_d)
        #             if q_cand.dtype != dtype: 
        #                 q_cand = q_cand.to(dtype)
                    
        #             # Limit max 用于计算离群点补偿，依然看绝对值最大的那一头
        #             limit_max = torch.max(c_max_d.abs(), c_min_d.abs())
        #             if limit_max.dim() == 1:
        #                 view_shape = [1] * (inputs_calibrate.ndim - 1) + [-1]
        #                 limit_max = limit_max.view(*view_shape)
                    
        #             is_true_outlier = (inputs_calibrate.abs() > (limit_max * sparse_buffer_ratio)) & outlier_mask.view_as(limit_max)
        #             hybrid_prediction = torch.where(is_true_outlier, inputs_calibrate, q_cand)
                    
        #             curr_score = self.cal_score_channel(hybrid_prediction, inputs_calibrate)
        #             batch_score_grid[idx] = curr_score.float()

        #     # [核心机制]: 多 Batch 滑动平均 (EMA) 更新
        #     if getattr(self, '_search_score_ema', None) is None or self._search_score_ema.shape != batch_score_grid.shape:
        #         self._search_score_ema = batch_score_grid.clone()
        #     else:
        #         momentum = float(getattr(self, 'search_ema_momentum', 0.5))
        #         self._search_score_ema.mul_(momentum).add_(batch_score_grid, alpha=1 - momentum)

        #     # 根据 EMA 全局分数选出最优的 ratio 索引
        #     best_idx = torch.argmax(self._search_score_ema, dim=0) # [channel_dim]
        #     ratios_tensor = torch.tensor(ratios, device=device, dtype=asym_min.dtype)
        #     best_r_tensor = ratios_tensor[best_idx]

        #     # 【核心修改】：固化非对称的最优边界
        #     best_min = asym_min * best_r_tensor
        #     best_max = asym_max * best_r_tensor

        #     # 3. 固化参数用于推断
        #     self.activation_range_min = best_min
        #     self.activation_range_max = best_max
        #     if getattr(self, 'llama_layer', False):
        #         self.llama_range_min = best_min
        #         self.llama_range_max = best_max
        #     else:
        #         self.CLIP_range_min = best_min
        #         self.CLIP_range_max = best_max
            
        #     self.compensation_limit = torch.max(best_max.abs(), best_min.abs()) * sparse_buffer_ratio
                
        #     quant_act_main = self.quantization(inputs_calibrate, best_min, best_max)
        #     limit_max_final = torch.max(best_max.abs(), best_min.abs())
        #     if limit_max_final.dim() == 1:
        #         view_shape = [1] * (inputs_calibrate.ndim - 1) + [-1]
        #         limit_max_final = limit_max_final.view(*view_shape)
                 
        #     sparse_mask = (inputs_calibrate.abs() > (limit_max_final * sparse_buffer_ratio)) & outlier_mask.view_as(limit_max_final)
        #     quant_act = torch.where(sparse_mask, inputs_calibrate, quant_act_main)
            
        #     return quant_act

        # inference stage
        else:
            # inference stage
            # if inputs_calibrate.shape[1] == 1:
            #     # print("seq_len=1")
            #     # 简化为：使用搜索固化下来的 best_min / best_max，并同样进行稀疏补偿 (Sparse Compensation)
            #     if hasattr(self, 'activation_range_min') and hasattr(self, 'activation_range_max'):
            #         best_min = self.activation_range_min.to(inputs_calibrate.device).to(inputs_calibrate.dtype)
            #         best_max = self.activation_range_max.to(inputs_calibrate.device).to(inputs_calibrate.dtype)
            #     else:
            #         best_min = self.llama_range_min.to(inputs_calibrate.device).to(inputs_calibrate.dtype)
            #         best_max = self.llama_range_max.to(inputs_calibrate.device).to(inputs_calibrate.dtype)

            #     quant_act_main = self.quantization(inputs_calibrate, best_min, best_max)
            #     if quant_act_main.dtype != inputs_calibrate.dtype: 
            #         quant_act_main = quant_act_main.to(inputs_calibrate.dtype)

            #     outlier_mask = getattr(self, 'outlier_mask', None)
            #     if hasattr(self, 'compensation_limit') and outlier_mask is not None:
            #         limit_max_final = self.compensation_limit.to(inputs_calibrate.device).to(inputs_calibrate.dtype)
            #         if limit_max_final.dim() == 1:
            #             if inputs_calibrate.ndim == 3: limit_max_final = limit_max_final.view(1, 1, -1)
            #             elif inputs_calibrate.ndim == 2: limit_max_final = limit_max_final.view(1, -1)
                    
            #         sparse_mask = (inputs_calibrate.abs() > limit_max_final) & outlier_mask.view_as(limit_max_final).to(inputs_calibrate.device).bool()
            #         quant_act = torch.where(sparse_mask, inputs_calibrate, quant_act_main)
            #     else:
            #         quant_act = quant_act_main
                    
            #     return quant_act
            # else:
            #     # row-wise  (1, 109, 4096) (1, 109)
            #     if self.llama_layer == True:
            #         # channel-wise                
            #         if self.dim != 4096 or self.count_layer == 4: 
            #             # self.llama_range_min1 = torch.min(inputs_calibrate, dim=1)[0].squeeze(dim=0)
            #             # self.llama_range_max1 = torch.max(inputs_calibrate, dim=1)[0].squeeze(dim=0)

            #             # print("Path 1")
                        
            #             # --- 🟢 最终修复版 (通用逻辑) ---
            #             # 1. 获取通道数 (最后一维)
            #             channel_dim = inputs_calibrate.shape[-1]
                        
            #             # 推理时：使用已固化的量化和补偿范围进行推理
            #             if hasattr(self, 'activation_range_min') and hasattr(self, 'activation_range_max'):
            #                 best_min = self.activation_range_min.to(inputs_calibrate.device).to(inputs_calibrate.dtype)
            #                 best_max = self.activation_range_max.to(inputs_calibrate.device).to(inputs_calibrate.dtype)
            #             else:
            #                 best_min = self.llama_range_min.to(inputs_calibrate.device).to(inputs_calibrate.dtype)
            #                 best_max = self.llama_range_max.to(inputs_calibrate.device).to(inputs_calibrate.dtype)

            #             quant_act_main = self.quantization(inputs_calibrate, best_min, best_max)
            #             if quant_act_main.dtype != inputs_calibrate.dtype: 
            #                 quant_act_main = quant_act_main.to(inputs_calibrate.dtype)

            #             outlier_mask = getattr(self, 'outlier_mask', None)
            #             if hasattr(self, 'compensation_limit') and outlier_mask is not None:
            #                 limit_max_final = self.compensation_limit.to(inputs_calibrate.device).to(inputs_calibrate.dtype)
            #                 if limit_max_final.dim() == 1:
            #                     if inputs_calibrate.ndim == 3: limit_max_final = limit_max_final.view(1, 1, -1)
            #                     elif inputs_calibrate.ndim == 2: limit_max_final = limit_max_final.view(1, -1)
                                
            #                 sparse_mask = (inputs_calibrate.abs() > limit_max_final) & outlier_mask.view_as(limit_max_final).to(inputs_calibrate.device).bool()
            #                 quant_act = torch.where(sparse_mask, inputs_calibrate, quant_act_main)
            #             else:
            #                 quant_act = quant_act_main

            #             self.activation_range_min = best_min
            #             self.activation_range_max = best_max

            #             return quant_act
                    
            #         # Fallback: 如果没有通过 Search 逻辑 (层不匹配或条件未满足)，
            #         # 则使用默认的 llama_range_min/max 进行量化，并初始化 Decode 阶段需要的这两个变量
            #         quant_act = self.quantization(x, self.llama_range_min, self.llama_range_max)
            #         self.activation_range_min = self.llama_range_min
            #         self.activation_range_max = self.llama_range_max
            #         return quant_act
            #     else:
            #         # row-wise
            #         quant_act = self.quantization(x, self.CLIP_range_min , self.CLIP_range_max)
            #         return quant_act

            # --- 新版 Inference Stage ---
            outlier_mask = getattr(self, 'outlier_mask', None)

            # 获取固化的量化范围
            if hasattr(self, 'activation_range_min') and hasattr(self, 'activation_range_max'):
                best_min = self.activation_range_min.to(inputs_calibrate.device).to(inputs_calibrate.dtype)
                best_max = self.activation_range_max.to(inputs_calibrate.device).to(inputs_calibrate.dtype)
            else:
                if self.llama_layer:
                    best_min = self.llama_range_min.to(inputs_calibrate.device).to(inputs_calibrate.dtype)
                    best_max = self.llama_range_max.to(inputs_calibrate.device).to(inputs_calibrate.dtype)
                else:
                    best_min = self.CLIP_range_min.to(inputs_calibrate.device).to(inputs_calibrate.dtype)
                    best_max = self.CLIP_range_max.to(inputs_calibrate.device).to(inputs_calibrate.dtype)

            # 计算量化值
            quant_act_main = self.quantization(inputs_calibrate, best_min, best_max)
            if quant_act_main.dtype != inputs_calibrate.dtype: 
                quant_act_main = quant_act_main.to(inputs_calibrate.dtype)

            # 稀疏补偿
            if hasattr(self, 'compensation_limit') and outlier_mask is not None:
                limit_max_final = self.compensation_limit.to(inputs_calibrate.device).to(inputs_calibrate.dtype)
                # 适配维度
                if limit_max_final.dim() == 1:
                    if inputs_calibrate.ndim == 3: 
                        limit_max_final = limit_max_final.view(1, 1, -1)
                    elif inputs_calibrate.ndim == 2: 
                        limit_max_final = limit_max_final.view(1, -1)
                
                sparse_mask = (inputs_calibrate.abs() > limit_max_final) & outlier_mask.view_as(limit_max_final).to(inputs_calibrate.device).bool()
                quant_act = torch.where(sparse_mask, inputs_calibrate, quant_act_main)
            else:
                quant_act = quant_act_main

            if not torch.isfinite(quant_act).all():
                quant_act = torch.where(torch.isfinite(quant_act), quant_act, inputs_calibrate)

            return quant_act

    def finalize_search(self):
        """
        在多轮 Batch 搜索结束后统一调用。
        将临时保存的全局最优参数真正固化给推理使用。
        """
        if hasattr(self, '_temp_best_min') and hasattr(self, '_temp_best_max') and self._temp_best_min is not None and self._temp_best_max is not None:
            sparse_buffer_ratio = getattr(self, 'sparse_buffer_ratio', 0.8)
            
            self.activation_range_min = self._temp_best_min
            self.activation_range_max = self._temp_best_max
            
            if getattr(self, 'llama_layer', False):
                self.llama_range_min = self._temp_best_min
                self.llama_range_max = self._temp_best_max
            else:
                self.CLIP_range_min = self._temp_best_min
                self.CLIP_range_max = self._temp_best_max
            
            self.compensation_limit = torch.max(self._temp_best_max.abs(), self._temp_best_min.abs()) * sparse_buffer_ratio

    def finalize_without_search(self):
        """
        Skip the boundary search phase entirely and use raw calibration min/max
        as the quantization boundaries. Sparse compensation is still active:
        compensation_limit is computed from the raw calibration bounds.

        This is used for ablation experiments where boundary co-optimization
        is disabled (disable_boundary_cooptimization=True).
        """
        if hasattr(self, 'calib_min') and hasattr(self, 'calib_max') and \
           self.calib_min is not None and self.calib_max is not None:
            sparse_buffer_ratio = getattr(self, 'sparse_buffer_ratio', 0.8)

            # Use raw calibration bounds directly (no grid search)
            self.activation_range_min = self.calib_min.clone()
            self.activation_range_max = self.calib_max.clone()

            if getattr(self, 'llama_layer', False):
                self.llama_range_min = self.calib_min.clone()
                self.llama_range_max = self.calib_max.clone()
            else:
                self.CLIP_range_min = self.calib_min.clone()
                self.CLIP_range_max = self.calib_max.clone()

            # Compute compensation limit from raw calibration bounds
            temp_max_abs = torch.max(self.calib_max.abs(), self.calib_min.abs())
            self.compensation_limit = temp_max_abs * sparse_buffer_ratio
        else:
            layer_name = getattr(self, 'layer_name', 'Unknown')
            print(f"[QuantAct] finalize_without_search: no calibration data for {layer_name}, skipping")

    def compute_dispersion_score(self, eps=1e-5):
        """
        在校准完成后调用，计算各个通道的统计特征以及离群点分数 (Dispersion Score)。
        并将结果直接打印输出或存入 self 供后续混合精度判定使用。
        """
        if not hasattr(self, 'calib_count') or self.calib_count == 0:
            self.dispersion_score = None
            return None
            
        # 1. 计算均值
        mean = self.calib_sum / self.calib_count
        
        # 2. 计算方差 E(X^2) - (E(X))^2 (注意 clamp 防止浮点误差负数)
        variance = (self.calib_sq_sum / self.calib_count) - (mean ** 2)
        variance = torch.clamp(variance, min=0.0)
        
        # 3. 计算绝对值均值 (即 L1 Norm 按样本均值)
        mean_abs = self.calib_abs_sum / self.calib_count
        
        # 4. 计算 Dispersion Score: 极差 / (L1 + eps)
        # dispersion_score = (self.calib_max - self.calib_min).abs() / (mean_abs + eps)

        # log version
        dispersion_score = (self.calib_max - self.calib_min).abs() / torch.log1p(mean_abs + eps)

        # PAR
        # peak_abs = torch.max(self.calib_max.abs(), self.calib_min.abs())
        # dispersion_score = peak_abs / (mean_abs + eps)

        self.dispersion_score = dispersion_score
        
        layer_name = self.layer_name if hasattr(self, 'layer_name') and self.layer_name else "Unknown"
        # 
        # print(f"[{layer_name}] Calib Stats | Count: {self.calib_count} | Global Min: {self.calib_min.min().item():.4f} | Global Max: {self.calib_max.max().item():.4f} | Max Dispersion: {dispersion_score.max().item():.4f}")
        
        return dispersion_score

def find_scale_by_percentile_min(x, percentile=0.9999):
    x_cpu = x.flatten().detach().cpu().numpy()
    max_k = int(x_cpu.size * (1 - percentile))
    # print(max_k)
    return np.partition(x_cpu, max_k)[max_k]

def find_scale_by_percentile_max(x, percentile=0.9999):
    x_cpu = x.flatten().detach().cpu().numpy()
    max_k = int(x_cpu.size * percentile)
    # print(max_k)
    return np.partition(x_cpu, max_k)[max_k]
