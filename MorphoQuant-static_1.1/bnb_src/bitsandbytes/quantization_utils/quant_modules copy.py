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

last_layer_entropy = 0
last_layer_distribution = torch.Tensor(np.zeros([1,100,4096])).cuda()
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
            self.llama_range_min = torch.Tensor(-self.init_range * np.zeros(self.dim)).cuda()
            self.llama_range_max = torch.Tensor(self.init_range * np.zeros(self.dim)).cuda()
        else:
            # CLIP calibrate search
            CLIP_row_dim = 257 # v1.3
            # CLIP_row_dim = 577 # v1.5 (position_embedding): Embedding(577, 1024)
            self.CLIP_range_min = torch.Tensor(-self.init_range * np.zeros(CLIP_row_dim)).cuda()
            self.CLIP_range_max = torch.Tensor(self.init_range * np.zeros(CLIP_row_dim)).cuda()

        self.layer_name = ''
        self.group_num = 8

        self.act_function = AsymmetricQuantFunction.apply
        self._calibrate = False
        self.search = False

        self.gamma_inf = 0.5
        self.gamma_cos = 1.0

    def set_calibrate(self, calibrate=True):
        self._calibrate = calibrate

    def set_search(self, search=True):
        self.search = search

    def set_gamma(self, gamma_inf=0.5, gamma_cos=1.0):
        self.gamma_inf = gamma_inf
        self.gamma_cos = gamma_cos

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
            # print(f"DEBUG: inputs_T shape: {inputs.transpose(1,-1).shape}, scale shape: {scale.shape}")

            # new_quant_x = torch.round(scale * inputs.transpose(1,-1) - zero_point)
            # n = 2**(self.activation_bit - 1)
            # new_quant_x_1 = 0.5 * ((-new_quant_x - n).abs() - (new_quant_x - (n - 1)).abs() - 1)
            # quant_act = (new_quant_x_1 + zero_point) / scale
            # return quant_act.transpose(1,-1)
            
            # 1. 统一变形：将 [352] 变成 [352, 1] 以便广播
            scale_reshaped = scale.view(-1, 1)
            
            if isinstance(zero_point, torch.Tensor) and zero_point.dim() == 1:
                zero_point_reshaped = zero_point.view(-1, 1)
            else:
                zero_point_reshaped = zero_point

            # 2. 量化计算 (使用变形后的变量)
            # inputs.transpose: [352, 1280], scale_reshaped: [352, 1] -> 正常运算
            new_quant_x = torch.round(scale_reshaped * inputs.transpose(1, -1) - zero_point_reshaped)
            
            # ... 这里中间可能有一些 clamp (截断) 的代码，保留原样 ...
            # 假设中间变量叫 new_quant_x_1 或直接用 new_quant_x
            # 这里的示例假设没有中间变量，如果有，请替换变量名
            
            # 3. 反量化计算 (关键修复点！！！！)
            # 必须使用 scale_reshaped 和 zero_point_reshaped，不能用原始的 scale/zero_point
            quant_act = (new_quant_x + zero_point_reshaped) / scale_reshaped

            if quant_act.dtype != inputs.dtype:
                quant_act = quant_act.to(inputs.dtype)

            # 4. 还原形状 (非常重要)
            # 因为一开始 transpose 过了，算完后的形状是 [352, 1280]
            # 必须转置回 [1280, 352] 才能匹配后续网络层的输入
            return quant_act.transpose(1, -1)

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
        x_flat = inputs_calibrate.view(-1, inputs_calibrate.shape[-1]).float()  # [N, Channel]
        q_flat = quant_act.view(-1, quant_act.shape[-1]).float()  # [N, Channel]
        
        # [Fix] 添加 eps 防止除零，并手动 clamp 结果
        s_cos = torch.nn.functional.cosine_similarity(x_flat, q_flat, dim=0, eps=1e-6)  # [Channel]
        s_cos = torch.clamp(s_cos, min=-1.0, max=1.0)
        
        # score = l_mse + l_kl * weight_kl
        # print("s_cos!")
        # print(s_cos[s_cos > 1])

        # l_0.5 loss
        lploss = (quant_act-inputs_calibrate).abs().pow(0.5).mean(dim=reduce_dims)

        # score = -lploss
        # score = -l_mse
        # score = s_cos   # 越接近 1 越好，所以直接用余弦相似度作为分数，搜索时选择最高的那个
        score = -lploss + self.gamma_cos * s_cos  # 组合 Loss，权重可以调整
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
        # print("forward")
        # percentile = 0.9997
        inputs_calibrate = x.data
        # print(self._calibrate, self.first_search, self.search, self.debug)
        if self._calibrate:
            if inputs_calibrate.shape[1] == 1:
                return x
            else:
                global llama_entropy, llama_distribution
                # print(self.first_search)
                if self.search and self.first_search:
                    self.first_search = False
                    if self.llama_layer:
                        # quant_act = self.quantization(inputs_calibrate, self.llama_range_min, self.llama_range_max)
                        quant_act = self.calibrate_quantization(inputs_calibrate)
                        llama_distribution.append(quant_act)
                        entropy = self.cal_entropy(quant_act.abs()).item()
                        if not np.isnan(entropy):
                            llama_entropy.append(entropy)
                    else:
                        quant_act = self.calibrate_quantization(inputs_calibrate)
                        return quant_act


                elif self.search and self.llama_layer == True and self.first_search == False:
                    # print("LLAMA search!")
                    percentile = 1.0
                    best_score = 1e+10
                    best_percentile = 1.0
                    best_max = self.llama_range_max
                    best_min = self.llama_range_min
                    # Original version: range from 1.0 to 0.3
                    # for aa in range(7):
                    #     percentile = 1.0 - (aa * 0.1)
                    #     new_max = self.llama_range_max * (1.0 - (aa * 0.1))
                    #     new_min = self.llama_range_min * (1.0 - (aa * 0.1))

                    # 初始阈值敏感度高
                    for aa in range(31):
                        percentile = 1.0 - (aa * 0.01)  # 从 1.0 到 0.7
                        new_max = self.llama_range_max * percentile
                        new_min = self.llama_range_min * percentile

                        activ_tmp = self.quantization(inputs_calibrate, new_min, new_max)
                        # score = lp_loss(activ_tmp, inputs_calibrate, p=0.5, reduction='all')

                    #     if score < best_score:
                    #         best_max = new_max
                    #         best_min = new_min
                    #         best_score = score
                    # print("LLAMA best percentile:", percentile)

                    
                    # for aa in range(21): 
                    #     percentile = 1.0 - (aa * 0.01)  # 从 1.0 到 0.8
                        
                    #     new_max = self.llama_range_max * percentile
                    #     new_min = self.llama_range_min * percentile

                    #     activ_tmp = self.quantization(inputs_calibrate, new_min, new_max)
                        
                        # --- 🟢 修改 Loss 计算 ---
                        # 1. MSE Loss (L2): 对大误差(截断)比 L0.5 更敏感
                        # 形状: [Batch, Seq, Channel] -> Scalar
                        l_mse = (activ_tmp - inputs_calibrate).pow(2).mean()
                        
                        # 2. Max Error Loss (L_inf): 专门惩罚离群点被截断的情况
                        # 找到所有 Token 中最大的那个误差值
                        l_inf = (activ_tmp - inputs_calibrate).abs().max()
                        
                        # 3. Cosine Loss (保持语义方向)
                        x_flat = inputs_calibrate.view(-1, inputs_calibrate.shape[-1])
                        q_flat = activ_tmp.view(-1, activ_tmp.shape[-1])
                        l_cos = 1.0 - torch.nn.functional.cosine_similarity(x_flat, q_flat, dim=1).mean()
                        
                        # 4. 组合 Loss
                        # 权重是一个玄学，但在搜索阈值时，L_inf 非常重要
                        # 如果 l_inf 很大，说明你把重要的 Outlier 切掉了
                        score = l_mse + 0.1 * l_inf + 1.0 * l_cos
                        # -----------------------

                        if score < best_score:
                            best_score = score
                            best_percentile = percentile
                            best_max = new_max
                            best_min = new_min

                    print(f"Layer {self.count_layer} selected percentile: {best_percentile}")

                    self.llama_range_max = best_max
                    self.llama_range_min = best_min
                    
                elif self.search and self.llama_layer == False and self.first_search == False:
                    print("CLIP search!")
                    best_score = 1e+10
                    best_max = self.CLIP_range_max
                    best_min = self.CLIP_range_min
                    entropyloss = np.mean(llama_entropy)
                    entropyweight = 0.01
                    for aa in range(3):
                        new_max = self.CLIP_range_max * (1.0 - (aa * 0.001))
                        new_min = self.CLIP_range_min * (1.0 - (aa * 0.001))
                        activ_tmp = self.quantization(inputs_calibrate, new_min, new_max)
                        lploss = (activ_tmp-inputs_calibrate).abs().pow(0.5).mean()
                        score = lploss + entropyweight * entropyloss
                        if score < best_score:
                            best_max = new_max
                            best_min = new_min
                            best_score = score
                    self.CLIP_range_max = best_max
                    self.CLIP_range_min = best_min
                else:
                    quant_act = self.calibrate_quantization(inputs_calibrate)
                    return quant_act

        if inputs_calibrate.shape[1] == 1:
            # print("seq_len=1")
            # row-wise  (1, 109, 4096) (1, 109) (8, 1, 4096)
            # print("Bypass quantization for seq_len=1")
            # print(self.debug)
            # if self.debug:
            #     print("Debug QuantAct:")
            #     print("llama_range_min (first 5):", self.llama_range_min[:5])
            #     print("llama_range_max (first 5):", self.llama_range_max[:5])
            #     self.debug = False  # 只打印一次
            activation_catrange_min = torch.cat([self.activation_range_min.unsqueeze(dim=0), inputs_calibrate.squeeze(dim=0)], dim=0)
            activation_catrange_max = torch.cat([self.activation_range_max.unsqueeze(dim=0), inputs_calibrate.squeeze(dim=0)], dim=0)
            
            self.activation_range_min = torch.min(activation_catrange_min, dim=0)[0].squeeze(dim=0)
            self.activation_range_max = torch.max(activation_catrange_max, dim=0)[0].squeeze(dim=0)
            quant_act = self.quantization(x, self.activation_range_min , self.activation_range_max)

            return quant_act
        else:
            # row-wise  (1, 109, 4096) (1, 109)
            if self.llama_layer == True:
                # channel-wise                
                if self.dim != 4096 or self.count_layer == 4: 
                    # self.llama_range_min1 = torch.min(inputs_calibrate, dim=1)[0].squeeze(dim=0)
                    # self.llama_range_max1 = torch.max(inputs_calibrate, dim=1)[0].squeeze(dim=0)

                    # print("Path 1")
                    
                    # --- 🟢 最终修复版 (通用逻辑) ---
                    # 1. 获取通道数 (最后一维)
                    channel_dim = inputs_calibrate.shape[-1]
                    
                    # 2. 将所有前面的维度展平，变成 [N, Channel]
                    # 无论输入是 [Batch, Seq, Channel] 还是 [Tokens, Channel]，都会变成 2D
                    inputs_flat = inputs_calibrate.reshape(-1, channel_dim)
                    
                    # 3. 在第0维 (所有样本/Token) 上取极值，保留第1维 (Channel)
                    # 结果形状必定是 [Channel]，不会变成标量
                    self.llama_range_min1 = torch.min(inputs_flat, dim=0)[0]
                    self.llama_range_max1 = torch.max(inputs_flat, dim=0)[0]
                    # --- 修复结束 ---
                    search = True
                    if search:
                    # if search and hasattr(self, 'layer_name') and any(f"thinker.model.layers.{i}" in self.layer_name for i in range(6)):
                        channel_dim = inputs_calibrate.shape[-1]
                        x_flat = inputs_calibrate.reshape(-1, channel_dim)
                        # N = x_flat.shape[0]

                        # 计算每个通道的原始最小值和最大值
                        base_min = x_flat.min(dim=0)[0] # [Channel]
                        base_max = x_flat.max(dim=0)[0] # [Channel]
                        
                        # 初始化最佳状态 (默认为全范围)
                        best_min = base_min.clone()
                        best_max = base_max.clone()
                        
                        # Initial Quantization
                        q_init = self.quantization(inputs_calibrate, base_min, base_max)
                        if q_init.dtype != inputs_calibrate.dtype: q_init = q_init.to(inputs_calibrate.dtype)
                        
                        sym_edge = torch.min(base_min.abs(), base_max.abs())
                        sym_edge_1 = torch.max(base_min.abs(), base_max.abs())
                        # if base_min * base_max < 0:
                        #     sym_edge = torch.min(base_min.abs(), base_max.abs())
                        #     sym_min = -sym_edge
                        #     sym_max = sym_edge
                        # else:
                        #     sym_min = base_min
                        #     sym_max = base_max
                        sym_mask = base_min * base_max <= 0
                        sym_min = torch.where(sym_mask, -sym_edge, base_min)
                        sym_max = torch.where(sym_mask, sym_edge, base_max)
                        # print(best_min.shape, sym_min.shape)  # [Channel]
                        
                        # Reduce Dims: everything except last dim (Channel)
                        # reduce_dims = list(range(inputs_calibrate.dim() - 1))
                        # Initial MSE is computed on full range quantization
                        # best_mse = (q_init - inputs_calibrate).pow(2).mean(dim=reduce_dims)
                        best_score = self.cal_score_channel(q_init, inputs_calibrate)
                        # print(f"Initial Score (Full Range): {best_score}")
                        # print(best_mse.shape, best_score.shape)  # [Channel]
                        
                        # 3. Grid Search Candidates (收缩比例)
                        # 尝试从两端同时向内收缩，暴力测试多个比例
                        sparse_buffer_ratio = 1.5
                        ratios = [1.0, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2]
                        
                        for r in ratios:
                            # Candidate bounds
                            # 向内收缩: max变小，min变大(如果是负数则绝对值变小)
                            # c_min = base_min * r
                            # c_max = base_max * r

                            # Symmetric Shrinkage (保持对称)
                            c_min = sym_min * r
                            c_max = sym_max * r
                            
                            # 1. Check Sparsity (Feasibility)
                            # 广播比较: [N, C] vs [C] -> [N, C]
                            # 多少点落在了范围之外？
                            # outlier_mask = (x_flat < c_min.unsqueeze(0)) | (x_flat > c_max.unsqueeze(0))
                            # sparsity = outlier_mask.float().mean(dim=0) # [Channel]
                            
                            # 约束：离群点比例必须非常小 (例如 < 0.5%)
                            # 如果超过这个比例，说明切到了主体数据，该方案不可用
                            # is_feasible = sparsity < 0.05
                            
                            # 2. Compute Hybrid MSE for feasible channels
                            # 只计算 Hybrid Result 的量化误差 (Outliers 误差为 0)
                            
                            # (A) Quantize with candidate bounds
                            # Ensure bounds are on correct device/dtype
                            c_min_d = c_min.to(inputs_calibrate.device).to(inputs_calibrate.dtype)
                            c_max_d = c_max.to(inputs_calibrate.device).to(inputs_calibrate.dtype)
                            
                            q_cand = self.quantization(inputs_calibrate, c_min_d, c_max_d)
                            if q_cand.dtype != inputs_calibrate.dtype: q_cand = q_cand.to(inputs_calibrate.dtype)
                            
                            # (B) Hybrid Result Construction
                            # outlier_mask 需要 reshape 回 inputs_calibrate 的形状以便 where
                            # x_flat 是 reshape 过的，outlier_mask 是 [N, C]
                            # inputs_calibrate 是 [Batch, Seq, C]
                            # mask_reshaped = outlier_mask.view_as(inputs_calibrate)
                            
                            # 如果是 Outlier，用原始值；如果是 Inlier，用量化值
                            # hybrid_res = torch.where(mask_reshaped, inputs_calibrate, q_cand)
                            
                            # (C) Calculate MSE [Channel]
                            # curr_mse = (hybrid_res - inputs_calibrate).pow(2).mean(dim=reduce_dims)
                            # curr_score = self.cal_score_channel(q_cand, inputs_calibrate)
                            # 1. 找出 True Outlier Mask
                            limit_max = torch.max(c_max_d.abs(), c_min_d.abs())
                            # 只有维度匹配才能广播，inputs 是 [...]，limit_max 是 [Channel]
                            # 兼容性 reshape
                            if limit_max.dim() == 1:
                                if inputs_calibrate.ndim == 3: limit_max = limit_max.view(1, 1, -1)
                                elif inputs_calibrate.ndim == 2: limit_max = limit_max.view(1, -1)
                            
                            # 加上 Buffer
                            is_true_outlier = inputs_calibrate.abs() > (limit_max * sparse_buffer_ratio)
                            
                            # 2. 混合计算
                            # Outlier -> 用原始值 (误差=0, Score贡献=完美)
                            # Inlier  -> 用 q_cand (误差=量化噪声, Score贡献=提升分辨率后的结果)
                            hybrid_prediction = torch.where(is_true_outlier, inputs_calibrate, q_cand)
                            
                            # 3. 计算分数 (使用 Hybrid 结果!)
                            curr_score = self.cal_score_channel(hybrid_prediction, inputs_calibrate)
                            # 查看 inputs_calibrate 中绝对值大于 400 的channel，search前后的分别得分
                            # key_mask = (sym_edge_1 > 400)
                            # if key_mask.sum().item() > 0:
                            #     # 打印这些关键通道的得分变化
                            #     print(f"\n[Debug] Channels with Symmetric Edge > 400 (Total {key_mask.sum().item()} channels):")
                            #     for idx in torch.where(key_mask)[0]:
                            #         print(f"  Channel {idx.item()}: Before Score={best_score[idx].item()}, After Score={curr_score[idx].item()}")
                            #         print(f"    Min/Max: {base_min[idx].item():.2f}/{base_max[idx].item():.2f} -> {c_min[idx].item():.2f}/{c_max[idx].item():.2f}")
                            #         # 再打印一个l_0.5和l_2的误差看看
                            #         # print(q_cand.shape, inputs_calibrate.shape)
                            #         lploss_before = (q_init[0, :, idx] - inputs_calibrate[0, :, idx]).abs().pow(0.5).mean().item()
                            #         lploss_mid = (q_cand[0, :, idx] - inputs_calibrate[0, :, idx]).abs().pow(0.5).mean().item()
                            #         lploss_after = (hybrid_prediction[0, :, idx] - inputs_calibrate[0, :, idx]).abs().pow(0.5).mean().item()
                            #         print(f"    L0.5 Loss: Before={lploss_before:.4f}, After Quant={lploss_mid:.4f}, After Hybrid={lploss_after:.4f}")
                            #         l2loss_before = (q_init[0, :, idx] - inputs_calibrate[0, :, idx]).pow(2).mean().item()
                            #         l2loss_mid = (q_cand[0, :, idx] - inputs_calibrate[0, :, idx]).pow(2).mean().item()
                            #         l2loss_after = (hybrid_prediction[0, :, idx] - inputs_calibrate[0, :, idx]).pow(2).mean().item()
                            #         print(f"    L2 Loss: Before={l2loss_before:.4f}, After Quant={l2loss_mid:.4f}, After Hybrid={l2loss_after:.4f}")
                            # if r == 1.0:
                            #     print(f"Candidate Ratio: {r:.3f}, Score: {curr_score}")
                            
                            # 3. Update Best
                            # Update if: Feasible AND (MSE < Best_MSE)
                            # update_mask = is_feasible & (curr_mse < best_mse)
                            update_mask = curr_score > best_score
                            # print(f"update_mask for ratio {r:.3f}: {update_mask.sum().item()} channels updated")

                            
                            # Move candidates to correct device for update
                            c_min_dev = c_min.to(inputs_calibrate.device)
                            c_max_dev = c_max.to(inputs_calibrate.device)
                            
                            best_score = torch.where(update_mask, curr_score, best_score)
                            best_min = torch.where(update_mask, c_min_dev, best_min)
                            best_max = torch.where(update_mask, c_max_dev, best_max)

                        # 4. Final Outputs
                        # Fix dtypes
                        target_dtype = inputs_calibrate.dtype
                        if best_min.dtype != target_dtype:
                            best_min = best_min.to(target_dtype)
                            best_max = best_max.to(target_dtype)
                        
                        # Construct Final Sparse Mask based on chosen bounds
                        # Ensure broadcasting support
                        # spand = (best_max - best_min) * 0.5
                        # sparse_mask = (inputs_calibrate > best_max + spand) | (inputs_calibrate < best_min - spand)
                        max_abs_final = torch.max(best_max.abs(), best_min.abs())
                        # 调整形状以匹配 input
                        if max_abs_final.dim() == 1:
                             if inputs_calibrate.ndim == 3: max_abs_final = max_abs_final.view(1, 1, -1)
                             elif inputs_calibrate.ndim == 2: max_abs_final = max_abs_final.view(1, -1)

                        # [关键] 保持与 Search 循环中完全一致的判定标准
                        sparse_mask = inputs_calibrate.abs() > (max_abs_final * sparse_buffer_ratio)
                        # print(sparse_mask.sum().item(), sparse_mask.numel())
                        
                        # --- Logging (Debug) ---
                        # 打印一下选中了多少离群点，确认逻辑生效
                        # sparse_count = sparse_mask.sum().item()
                        # if sparse_count > 0:
                        #     print(f"Layer {self.layer_name if hasattr(self, 'layer_name') else '?'}: Hybrid Selected! Sparsity={sparse_count/sparse_mask.numel():.4%}")
                        
                        # Main Quant
                        quant_act_main = self.quantization(inputs_calibrate, best_min, best_max)
                        if quant_act_main.dtype != target_dtype:
                            quant_act_main = quant_act_main.to(target_dtype)

                        # Hybrid Compensation
                        quant_act = torch.where(sparse_mask, inputs_calibrate, quant_act_main)
                        
                        # Safety Clamp (BF16 safe range)
                        # quant_act = torch.clamp(quant_act, min=-65000, max=65000)

                        self.activation_range_min = best_min
                        self.activation_range_max = best_max

                        # print min/max for debugging
                        # print(f"Original Min/Max (Global): {inputs_calibrate.min().item():.4f}, {inputs_calibrate.max().item():.4f}")
                        # print(f"Selected Min/Max (Global): {best_min.min().item():.4f}, {best_max.max().item():.4f}")
                        # print(f"Final Min/Max (Global): {quant_act.min().item():.4f}, {quant_act.max().item():.4f}")

                        return quant_act
                    else:
                        quant_act = self.quantization(x, self.llama_range_min1 , self.llama_range_max1)
                        self.activation_range_min = self.llama_range_min1
                        self.activation_range_max = self.llama_range_max1
                        return quant_act
                
                # Fallback: 如果没有通过 Search 逻辑 (层不匹配或条件未满足)，
                # 则使用默认的 llama_range_min/max 进行量化，并初始化 Decode 阶段需要的这两个变量
                quant_act = self.quantization(x, self.llama_range_min, self.llama_range_max)
                self.activation_range_min = self.llama_range_min
                self.activation_range_max = self.llama_range_max
                return quant_act
            else:
                # row-wise
                quant_act = self.quantization(x, self.CLIP_range_min , self.CLIP_range_max)
                return quant_act

def calibrate(model, loader, device):
    print('\n==> start calibrate')
    for name, module in model.named_modules():
        if isinstance(module, QuantAct):
            module.set_calibrate(calibrate=True)
    inputs = next(iter(loader))
    # use 1 gpu to calibrate
    inputs = inputs[0].cuda(device, non_blocking=True)
    for i in range(4*8-1):
        inputs1 = next(iter(loader))
        # inputs1, _= next(iter(loader))
        inputs1 = inputs1[0].to(device, non_blocking=True)
        inputs = torch.cat((inputs, inputs1), 0)
    with torch.no_grad():
        model(inputs)
    for name, module in model.named_modules():
        if isinstance(module, QuantAct):
            module.set_calibrate(calibrate=False)
    print('==> end calibrate')
    return model

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
