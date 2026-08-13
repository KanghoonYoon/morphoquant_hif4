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

            # 4. 还原形状 (非常重要)
            # 因为一开始 transpose 过了，算完后的形状是 [352, 1280]
            # 必须转置回 [1280, 352] 才能匹配后续网络层的输入
            return quant_act.transpose(1, -1)

    
    def compute_DED(self, p_k, p_k1):
        """
        calcuate D(k, {k+1}) = -sum_ij p(x_{q,ij}^{(k)}, x_{q,ij}^{(k+1)}) log p(x_{q,ij}^{(k+1)} | x_{q,ij}^{(k)})
        """
        p_k = F.normalize(p_k, p=1, dim=1)  
        p_k1 = F.normalize(p_k1, p=1, dim=1)  

        if p_k.shape != p_k1.shape:
            # 如果返回值需要是 Tensor，请确保设备和类型一致
            return torch.tensor(0.0, device=p_k.device, dtype=p_k.dtype)
        
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
        # print("quant!")
        """
        quantize given activation x
        """
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
                    for aa in range(11):
                        percentile = 1.0 - (aa * 0.03)  # 从 1.0 到 0.7
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
            # row-wise  (1, 109, 4096) (1, 109) (8, 1, 4096)
            # print("Bypass quantization for seq_len=1")
            # print(self.debug)
            if self.debug:
                print("Debug QuantAct:")
                print("llama_range_min (first 5):", self.llama_range_min[:5])
                print("llama_range_max (first 5):", self.llama_range_max[:5])
                self.debug = False  # 只打印一次
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
                    search = False
                    if search:
                        percentile = 1.0
                        best_score = 1e+10
                        best_percentile = 1.0
                        best_max = self.llama_range_max1
                        best_min = self.llama_range_min1

                        for aa in range(11):
                            percentile = 1.0 - (aa * 0.02)
                            new_max = self.llama_range_max1 * percentile
                            new_min = self.llama_range_min1 * percentile

                            activ_tmp = self.quantization(inputs_calibrate, new_min, new_max)
                            
                            loss_func = 1
                            if loss_func == 1:
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
                                # score = l_mse + 0.5 * l_inf + 1.0 * l_cos
                                # 可变权重版本
                                score = l_mse + self.gamma_inf * l_inf + self.gamma_cos * l_cos
                                # -----------------------
                            else:
                                score = lp_loss(activ_tmp, inputs_calibrate, p=0.5, reduction='all')

                            if score < best_score:
                                best_score = score
                                best_percentile = percentile
                                best_max = new_max
                                best_min = new_min
                        # print(best_percentile)
                        quant_act = self.quantization(x, best_min, best_max)
                        self.activation_range_min = best_min
                        self.activation_range_max = best_max
                    else:
                        quant_act = self.quantization(x, self.llama_range_min1 , self.llama_range_max1)
                        self.activation_range_min = self.llama_range_min1
                        self.activation_range_max = self.llama_range_max1

                else:
                    print("Path 2")
                    quant_act = self.quantization(x, self.llama_range_min , self.llama_range_max)
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
