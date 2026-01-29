# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import torch.nn.functional as F_func
import numpy as np

# ===== Time embedding（DDPM 标准）=====
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (B,)
        """
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


############################################
# 1. Diffusion Denoiser (DDPM + U-Net)
############################################
class DiffusionDenoiser(nn.Module):
    """
    Diffusion denoiser (BackNet_k)
    
    论文定义：Ĥ_k = BackNet_k(X_k)
    
    输入 :
        x0 : (B, N, F, T)
    输出 :
        x_denoised_traffic : (B, N, F, T) - 交通流空间（预训练用）
        x_denoised_hidden : (B, N, D, T) - 隐藏空间（主训练用）
    
    重要修改：
    - BackNet 直接输出 F 维（交通流空间），实现真实去噪
    - 添加 F→D 投影层，供主训练阶段使用
    - 预训练时直接在 traffic 空间计算 MSE，无需 input_projector
    """

    def __init__(
            self,
            F_in,
            D,
            diffusion_steps=1000,
            beta_start=1e-4,
            beta_end=2e-2
    ):
        super().__init__()

        self.D = D
        self.F_in = F_in
        self.diffusion_steps = diffusion_steps

        # -------- β schedule --------
        betas = torch.linspace(beta_start, beta_end, diffusion_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

        # -------- Time embedding --------
        self.time_embed = SinusoidalTimeEmbedding(D)
        self.time_mlp = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )

        # -------- U-Net backbone (1D, time axis) --------
        # 注意：现在 BackNet 输出 F 维（交通流空间）
        self.enc1 = nn.Conv1d(F_in, D, kernel_size=3, padding=1)
        self.enc2 = nn.Conv1d(D, D, kernel_size=3, padding=1)

        self.dec1 = nn.Conv1d(D, D, kernel_size=3, padding=1)
        self.dec2 = nn.Conv1d(D, F_in, kernel_size=3, padding=1)  # 直接输出 F 维

        # -------- F→D 投影层（供主训练使用）--------
        self.project_to_d = nn.Conv1d(F_in, D, kernel_size=1)  # 1x1 卷积投影

    def forward(self, x0, use_pure_denoising=True, return_traffic_space=True):
        """
        Diffusion denoiser forward pass
        
        根据论文Algorithm 1：
        Ĥ_k = BackNet_k(X_k)
        
        参数:
            x0: (B,N,F,T) - 输入数据（带噪声）
            use_pure_denoising: bool - 去噪模式选择
            return_traffic_space: bool - 是否返回交通流空间输出
                
        返回:
            如果 return_traffic_space=True:
                x0_hat_traffic: (B,N,F,T) - 去噪后的交通流（F维）
            如果 return_traffic_space=False:
                x0_hat_hidden: (B,N,D,T) - 去噪后的隐藏特征（D维）
        
        重要说明：
        - 预训练模式：return_traffic_space=True，直接用 F 维输出计算 MSE
        - 主训练模式：return_traffic_space=False，用投影到 D 维的输出
        """
        B, N, F, T = x0.shape
        x0 = x0.reshape(B * N, F, T)
        
        if use_pure_denoising:
            # 纯去噪模式（符合论文Algorithm 1）
            # 直接将带噪声数据作为输入，通过U-Net学习去噪
            x = x0
            
            # U-Net前向传播
            h1 = F_func.relu(self.enc1(x))
            h2 = F_func.relu(self.enc2(h1))
            
            h3 = F_func.relu(self.dec1(h2))
            x0_hat_traffic = self.dec2(h3 + h1)  # 直接输出 F 维（交通流空间）
            
            # 投影到 D 维（供主训练使用）
            x0_hat_hidden = self.project_to_d(x0_hat_traffic)
            
        else:
            # DDPM标准模式（用于对比测试）
            # 额外添加噪声
            t = torch.randint(0, self.diffusion_steps, (B * N,), device=x0.device)
            
            eps = torch.randn_like(x0)
            alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
            
            x_t = (
                    torch.sqrt(alpha_bar_t) * x0 +
                    torch.sqrt(1.0 - alpha_bar_t) * eps
            )
            
            t_emb = self.time_embed(t)
            t_emb = self.time_mlp(t_emb).unsqueeze(-1)
            
            h1 = F_func.relu(self.enc1(x_t))
            h2 = F_func.relu(self.enc2(h1 + t_emb))
            h3 = F_func.relu(self.dec1(h2))
            eps_hat = self.dec2(h3 + h1)
            
            x0_hat_traffic = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_hat) / torch.sqrt(alpha_bar_t)
            
            # 投影到 D 维
            x0_hat_hidden = self.project_to_d(x0_hat_traffic)

        # 🔥 安全访问：检查维度并自动修正
        # 确保 x0_hat_traffic 是 3 维张量 (B*N, F, T)
        if x0_hat_traffic.dim() != 3:
            # 如果输出不是 (B*N, F, T)，自动补维度
            x0_hat_traffic = x0_hat_traffic.unsqueeze(-1)  # 保证有 T 维
        
        B_N, F_hat, T_hat = x0_hat_traffic.shape
        if B_N != B * N or F_hat != self.F_in or T_hat != T:
            print(f"[WARNING] DiffusionDenoiser: 输出shape与预期不符: {x0_hat_traffic.shape}, 期望 ({B*N}, {self.F_in}, {T})")
            print(f"  输入原始: x0.shape: {x0.shape}")
            print(f"  原始维度: B={B}, N={N}, F={F}, T={T}")
        
        # Reshape
        x0_hat_traffic = x0_hat_traffic.reshape(B, N, self.F_in, T)
        x0_hat_hidden = x0_hat_hidden.reshape(B, N, self.D, T)
        
        # 🔥 验证：reshap e 后节点数
        if x0_hat_traffic.shape[1] != N:
            print(f"\n[CRITICAL ERROR] DiffusionDenoiser 输出节点数错误！")
            print(f"  输入: x0.shape=(B,N,F,T)=({B},{N},{F},{T})")
            print(f"  x0_hat_traffic.shape={x0_hat_traffic.shape}")
            print(f"  期望节点数: {N}")
            print(f"  实际节点数: {x0_hat_traffic.shape[1]}")
            print("  这说明 U-Net 或 reshape 错误地改变了节点数！")

        # 根据模式返回
        if return_traffic_space:
            return x0_hat_traffic
        else:
            return x0_hat_hidden


############################################
# 2. Temporal Encoder（论文关键修复）
############################################
class TemporalEncoder(nn.Module):
    """
    时间编码器：将 (B,N,F,T) 编码为 (B,N,D)
    
    论文语义：
    - 对每个周期进行独立编码
    - 输出是固定的 D 维特征（不是变长的 T）
    - 用于后续 MAF 在周期维度（3）上做 attention
    """
    def __init__(self, F_in, D, num_nodes=None):
        super().__init__()
        self.F_in = F_in
        self.D = D
        self.num_nodes = num_nodes  # 用于验证，可为 None 跳过检查
        
        # 使用 1D 卷积编码时间维度 (B,N,F,T) → (B,N,D)
        self.conv1 = nn.Conv1d(F_in, D, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(D, D, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)  # 时间池化为1

    def forward(self, x):
        """
        参数: x: (B,N,F,T)
        返回:   (B,N,D)
        """
        B, N, F, T = x.shape
        
        # 🔥 安全检查：只有指定了 num_nodes 时才验证节点数
        if self.num_nodes is not None and N != self.num_nodes:
            print(f"\n[WARNING] TemporalEncoder 输入节点数与预期不符！")
            print(f"  期望节点数: {self.num_nodes}")
            print(f"  实际节点数: {N}")
            print(f"  差异: {self.num_nodes - N}")
            print(f"  x.shape: {x.shape}")
        
        x = x.reshape(B * N, F, T)  # (B*N, F, T)
        
        h1 = F_func.relu(self.conv1(x))  # (B*N, D, T)
        h2 = F_func.relu(self.conv2(h1))  # (B*N, D, T)
        h_pooled = self.pool(h2).squeeze(-1)  # (B*N, D)
        
        result = h_pooled.reshape(B, N, self.D)
        
        # 🔥 验证： reshape 后节点数保持不变
        if self.num_nodes is not None and result.shape[1] != self.num_nodes:
            print(f"\n[CRITICAL ERROR] TemporalEncoder 输出节点数被改变了！")
            print(f"  输入: x.shape={x.shape}")
            print(f"  输出: result.shape={result.shape}")
            print(f"  期望节点数: {self.num_nodes}")
            print(f"  这说明某个操作错误地缩减了节点数！")
        
        return result


############################################
# 3. MDAF: Multi-period Diffusion Attention Fusion（论文级修复）
############################################
class MDAF(nn.Module):
    """
    Multi-period Diffusion Attention Fusion（论文级修复）
    
    输入 :
        X_rec  : (B,N,F,T_rec)  # 近期序列，长度T_rec
        X_hour : (B,N,F,T_hour) # 小时周期序列，长度T_hour
        X_day  : (B,N,F,T_day)  # 日周期序列，长度T_day
    输出 :
        X_mdaf : (B,N,D,T)  # 融合后的D维特征
    
    论文架构（修复后）：
        1. Diffusion denoising (三个 BackNet_k): 去噪 → (B,N,F,T_k)
        2. Temporal Encoder: 编码 → (B,N,D) [关键修复]？？？
        3. MAF fusion: 周期间 attention → (B,N,D) [Attention over 3 periods]
    """

    def __init__(self, F_in, D, num_nodes=None, nhead=1):
        super().__init__()

        # -------- Diffusion Denoisers (U-Net) --------
        self.rec = DiffusionDenoiser(F_in, D)
        self.hour = DiffusionDenoiser(F_in, D)
        self.day = DiffusionDenoiser(F_in, D)

        # -------- Temporal Encoder（关键修复）--------
        self.encoder_rec = TemporalEncoder(F_in, D, num_nodes)
        self.encoder_hour = TemporalEncoder(F_in, D, num_nodes)
        self.encoder_day = TemporalEncoder(F_in, D, num_nodes)

        # -------- MAF: Multi-period Attention Fusion（论文公式 9）--------
        # 关键修复：Attention 在"周期维度"（3）上，不是在节点×时间维度上
        self.maf_attn = nn.MultiheadAttention(
            embed_dim=D, num_heads=nhead, batch_first=True
        )
        
        # Fusion: 注意力输出 → 最终特征
        self.fusion = nn.Linear(D, D)

    def forward(self, x_rec, x_hour, x_day, use_pure_denoising=True, return_traffic_space=False):
        """
        MDAF模块前向传播（论文级修复）
        
        参数:
            x_rec: (B,N,F,T_rec) - 近期序列，F=1
            x_hour: (B,N,F,T_hour) - 小时周期序列，F=1
            x_day: (B,N,F,T_day) - 日周期序列，F=1
            use_pure_denoising: bool - 是否使用纯去噪模式（符合论文）
            return_traffic_space: bool - 是否返回交通流空间输出（用于预训练）
                
        返回:
            如果 return_traffic_space=True:
                (xr, xh, xd): 三个周期的去噪结果 (B,N,1,T_k)
            如果 return_traffic_space=False:
                x_fused: (B,N,D) - 融合后的特征（主训练用）
        """
        B, N, F, _ = x_rec.shape
        
        # -------- 1. 预训练模式：使用去噪后的交通流 --------
        if return_traffic_space:
            # Diffusion denoising（直接返回 F=1 维的交通流）
            xr = self.rec(x_rec, use_pure_denoising=use_pure_denoising, return_traffic_space=True)
            xh = self.hour(x_hour, use_pure_denoising=use_pure_denoising, return_traffic_space=True)
            xd = self.day(x_day, use_pure_denoising=use_pure_denoising, return_traffic_space=True)
            return xr, xh, xd  # (B,N,1,T)
        
        # -------- 2. 主训练模式：论文级修复（关键修改）--------
        # 根据论文，MAF 应该对"原始数据"做编码，而不是对"去噪后的 D 维特征"编码
        # 两种方案：
        # 方案A（推荐）：直接编码原始输入，不通过 BackNet 的 D 维输出
        # 方案B：对 BackNet 输出的 F=1 维交通流编码
        
        # 采用方案B：先去噪（得到 F=1 流量），再编码
        xr = self.rec(x_rec, use_pure_denoising=use_pure_denoising, return_traffic_space=True)
        xh = self.hour(x_hour, use_pure_denoising=use_pure_denoising, return_traffic_space=True)
        xd = self.day(x_day, use_pure_denoising=use_pure_denoising, return_traffic_space=True)
        
        # 验证：确保是 F=1 维
        assert xr.shape[2] == 1, f"Expected F=1, got {xr.shape[2]}"
        assert xh.shape[2] == 1, f"Expected F=1, got {xh.shape[2]}"
        assert xd.shape[2] == 1, f"Expected F=1, got {xd.shape[2]}"
        
        # -------- 3. Temporal Encoding（关键修复）--------
        # 将 (B,N,1,T) 编码为 (B,N,D)
        f_rec = self.encoder_rec(xr)   # (B,N,D)
        f_hour = self.encoder_hour(xh) # (B,N,D)
        f_day = self.encoder_day(xd)   # (B,N,D)

        B, N, D = f_rec.shape

        # -------- 3. MAF Fusion（论文级修复）--------
        # 关键：在周期维度（3）上做 attention，不是在 (N×T) 上
        # 聚合三个周期特征 → (B,N,3,D)
        periods_cat = torch.stack([f_rec, f_hour, f_day], dim=2)  # (B,N,3,D)
        
        # 🔥 关键修复：reshape 成 MHA 可接受的 3D 格式
        # MultiheadAttention 只接受 (batch, seq, embed)
        # 论文语义：batch = B*N, seq = 3 (周期）
        B, N, num_periods, D = periods_cat.shape
        periods_cat = periods_cat.view(B * N, num_periods, D)  # (B*N, 3, D) ✅
        
        # Attention over period dimension (dim=2)
        # 复杂度：O(N × 3² × D) ≈ O(9N)，而不是 O(N²T²)!
        f_fused, _ = self.maf_attn(periods_cat, periods_cat, periods_cat)  # (B*N, 3, D)
        
        # 周期维聚合：MHA 输出 3 个周期，需要聚合为 1 个
        f_fused = f_fused.mean(dim=1)  # (B*N, D)
        
        # 还原节点维度
        x_fused = f_fused.view(B, N, D)  # (B, N, D)
        
        # Fusion: 取attention输出的特征
        x_fused = self.fusion(x_fused)  # (B,N,D)

        return x_fused  # (B,N,D) - 只返回融合后的D维特征


############################################
# 4. MGRC: Multi-Graph Recurrent Convolution
############################################
class MGRC(nn.Module):
    """
    输入 : (B,N,D,T_H)
    输出 : (B,N,D,T_H)
    """

    def __init__(self, num_nodes, D, adj_mx, distance_mx, DEVICE):
        super().__init__()
        self.N = num_nodes
        self.D = D
        self.DEVICE = DEVICE

        # 静态距离矩阵（论文中的 A_dist）
        self.Adist = torch.tensor(distance_mx, dtype=torch.float32, device=DEVICE)

        # 动态图参数（论文中的 E1, E2）
        self.E1 = nn.Parameter(torch.randn(num_nodes, D))
        self.E2 = nn.Parameter(torch.randn(num_nodes, D))

        # 多图融合的2D卷积（公式12）
        self.graph_fusion = nn.Conv2d(2, 1, kernel_size=1)

        # GCN权重（公式13）
        self.W_GCN = nn.Parameter(torch.eye(D))

        # 门控递归单元（公式14）
        self.gru = nn.GRU(D, D, batch_first=True)

    def forward(self, x):
        # x: (B,N,D,T_H)
        B, N, D, T_H = x.shape

        # 动态邻接矩阵（公式10）
        Adyna = torch.softmax(
            torch.relu(self.E1 @ self.E2.T),
            dim=-1
        )  # (N,N)

        # 多图融合（公式12）
        # 将两个邻接矩阵堆叠为 (1, 2, N, N)，通过2D卷积融合
        A_concat = torch.cat([Adyna.unsqueeze(0), self.Adist.unsqueeze(0)], dim=0)  # (2,N,N)
        A_concat = A_concat.unsqueeze(0)  # (1,2,N,N)
        A_F = F_func.relu(self.graph_fusion(A_concat))  # (1,1,N,N)
        A_F = A_F.squeeze(0).squeeze(0)  # (N,N)
        
        # 修正：对融合后的邻接矩阵进行softmax归一化，防止数值不稳定
        A_F = torch.softmax(A_F, dim=-1)

        # 图卷积（公式13）
        x = x.permute(0, 3, 1, 2)  # (B,T_H,N,D)

        gcn_out = []
        for t in range(T_H):
            xt = x[:, t]  # (B,N,D)
            xt = torch.matmul(A_F, xt)  # (B,N,D) - 空间聚合
            xt = torch.matmul(xt, self.W_GCN)  # (B,N,D) - 特征变换
            gcn_out.append(xt)

        x = torch.stack(gcn_out, dim=1)  # (B,T_H,N,D)

        # GRU 建模时间递归（公式14）
        # 修正：GRU应该对每个节点的时间序列建模，而不是对时间步建模节点
        # 输入格式：(batch_size, sequence_length, input_size)
        # 正确的reshape：(B, T_H, N, D) -> (B, N, T_H, D) -> (B*N, T_H, D)
        x = x.reshape(B, T_H, N, D).permute(0, 2, 1, 3)  # (B,N,T_H,D)
        x = x.reshape(B * N, T_H, D)  # (B*N, T_H, D) - 每个节点的T_H个时间步
        x, _ = self.gru(x)
        # reshape回 (B,N,D,T_H)
        x = x.reshape(B, N, T_H, D).transpose(2, 3)  # (B,N,D,T_H)
        
        # 🔥 关键修复：不要再额外 permute，否则 N 会被换成 D！
        # 正确顺序应该是 (B,N,D,T_H)，不能 permute(0,2,3,1) 变成 (B,D,T_H,N)
        return x  # (B,N,D,T_H)


############################################
# 5. STFormer: Spatial-Temporal Transformer 
############################################
class SpatialTransformer(nn.Module):
    """
    空间Transformer模块（公式15-19）
    """
    def __init__(self, D, num_heads=3):
        super().__init__()
        # 多头注意力
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=D, num_heads=num_heads, batch_first=True
        )

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(D, 4 * D),
            nn.ReLU(),
            nn.Linear(4 * D, D)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)

    def forward(self, x, A):
        """
        x: (B,N,D) - 空间特征
        A: (N,N) - 邻接矩阵
        """
        B, N, D = x.shape
        
        # 🔥 调试：追踪节点数
        if A.shape[1] != N:
            print(f"\n[CRITICAL ERROR] SpatialTransformer 节点数不匹配！")
            print(f"  A.shape: {A.shape}")
            print(f"  x.shape: {x.shape}")
            print(f"  预期A节点数: {N}")
            print(f"  实际A节点数: {A.shape[1]}")
            print(f"  差异: {A.shape[1] - N}")
            print("  这说明在数据流中节点数被错误地改变了！")
        
        # 🔥 关键修复：使用 einsum 做批量矩阵乘法
        # torch.matmul 在 batch 维度不匹配时会报错
        # einsum 'ij,bjd->bid' 表示：A(N,N) @ x(B,N,D) → pos(B,N,D)
        # 重要：x 的 N 维必须与 A 的 N 维完全一致！
        pos_encoding = torch.einsum('ij,bjd->bid', A, x)  # (B,N,D)
        x_with_pos = x + pos_encoding

        # 多头注意力（公式16）
        attn_out, _ = self.spatial_attn(x_with_pos, x_with_pos, x_with_pos)  # (B,N,D)

        # Add & Normalize（公式17）
        x = self.norm1(x_with_pos + attn_out)

        # Feed Forward（公式18）
        ffn_out = self.ffn(x)

        # Add & Normalize（公式19）
        x = self.norm2(x + ffn_out)

        return x  # (B,N,D)


class TemporalTransformer(nn.Module):
    """
    时间Transformer模块（公式20-25）
    """
    def __init__(self, D, num_heads=3):
        super().__init__()
        # 时间位置编码（公式21）
        # hour: 1-60, day: 1-24, week: 1-7
        self.hour_emb = nn.Embedding(60, D)
        self.day_emb = nn.Embedding(24, D)
        self.week_emb = nn.Embedding(7, D)
        self.temporal_proj = nn.Linear(D, D)

        # 多头注意力
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=D, num_heads=num_heads, batch_first=True
        )

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(D, 4 * D),
            nn.ReLU(),
            nn.Linear(4 * D, D)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)

    def forward(self, x, hour_idx, day_idx, week_idx):
        """
        x: (B,T,D) - 时间特征
        hour_idx: (B,T) - 小时索引
        day_idx: (B,T) - 日索引
        week_idx: (B,T) - 周索引
        """
        B, T, D = x.shape

        # 时间位置编码（公式21）
        hour_enc = self.hour_emb(hour_idx)  # (B,T,D)
        day_enc = self.day_emb(day_idx)    # (B,T,D)
        week_enc = self.week_emb(week_idx)  # (B,T,D)

        x_with_temp = x + hour_enc + day_enc + week_enc

        # 多头注意力（公式22）
        attn_out, _ = self.temporal_attn(x_with_temp, x_with_temp, x_with_temp)

        # Add & Normalize（公式23）
        x = self.norm1(x_with_temp + attn_out)

        # Feed Forward（公式24）
        ffn_out = self.ffn(x)

        # Add & Normalize（公式25）
        x = self.norm2(x + ffn_out)

        return x


class STFormer(nn.Module):
    """
    时空Transformer模块（公式15-25）

    输入 : (B,N,D,T_H) + 时间索引
    输出 : (B,N,D,T_H)
    """

    def __init__(self, D, num_nodes, num_heads=3, num_layers=2, adj_mx=None):
        super().__init__()
        self.num_layers = num_layers
        self.num_nodes = num_nodes

        # 注册邻接矩阵作为buffer，确保device管理正确
        if adj_mx is not None:
            if isinstance(adj_mx, torch.Tensor):
                self.register_buffer('A', adj_mx)
            else:
                self.register_buffer('A', torch.tensor(adj_mx, dtype=torch.float32))
        else:
            self.register_buffer('A', torch.eye(num_nodes))

        self.spatial_layers = nn.ModuleList([
            SpatialTransformer(D, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        self.temporal_layers = nn.ModuleList([
            TemporalTransformer(D, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x, hour_idx=None, day_idx=None, week_idx=None):
        """
        x: (B,N,D,T_H)
        hour_idx: (B,T_H) - 小时索引，如果为None则自动生成
        day_idx: (B,T_H) - 日索引，如果为None则自动生成
        week_idx: (B,T_H) - 周索引，如果为None则自动生成
        """
        B, N, D, T_H = x.shape
        device = x.device

        # 如果没有提供时间索引，生成默认索引
        if hour_idx is None:
            hour_idx = torch.arange(T_H, device=device).unsqueeze(0).repeat(B, 1) % 60
        if day_idx is None:
            day_idx = torch.arange(T_H, device=device).unsqueeze(0).repeat(B, 1) % 24
        if week_idx is None:
            week_idx = torch.arange(T_H, device=device).unsqueeze(0).repeat(B, 1) % 7

        # 层级处理
        for l in range(self.num_layers):
            # 空间Transformer（处理每个时间步的空间关系）
            # x: (B,N,D,T_H)
            # 🔥 关键修复：MD-GRTN中 T_H=1，必须正确处理reshape
            x_permuted = x.permute(0, 3, 1, 2)  # (B,T_H,N,D)
            x_reshaped = x_permuted.reshape(B * T_H, N, D).contiguous()  # (B*T_H,N,D)
            
            # 传入register_buffer的A矩阵，确保device一致
            # A: (N,N), x: (B*T_H,N,D) → output: (B*T_H,N,D)
            x_spatial = self.spatial_layers[l](x_reshaped, A=self.A)  # (B*T_H,N,D)
            
            x_spatial = x_spatial.reshape(B, T_H, N, D).permute(0, 2, 1, 3)  # (B,N,T_H,D)
            
            # 时间Transformer（处理每个节点的时间关系）
            # 🔥 关键修复：正确处理时间维度
            x_permuted2 = x_spatial.permute(0, 2, 1, 3)  # (B,N,T_H,D) 确保顺序
            x_reshaped2 = x_permuted2.reshape(B * N, T_H, D)  # (B*N,T_H,D)
            
            # 时间索引需要广播到(B*N,T_H)
            hour_idx_broadcast = hour_idx.unsqueeze(1).expand(-1, N, -1).reshape(B * N, T_H)
            day_idx_broadcast = day_idx.unsqueeze(1).expand(-1, N, -1).reshape(B * N, T_H)
            week_idx_broadcast = week_idx.unsqueeze(1).expand(-1, N, -1).reshape(B * N, T_H)
            
            x_temporal = self.temporal_layers[l](x_reshaped2, hour_idx_broadcast, day_idx_broadcast, week_idx_broadcast)  # (B*N,T_H,D)
            
            x_temporal = x_temporal.reshape(B, N, T_H, D).permute(0, 1, 3, 2)  # (B,N,D,T_H)
            x = x_temporal

        return x


############################################
# 6. MD-GRTN 主模型
############################################
class MD_GRTN(nn.Module):
    """
    输入 :
        x_rec   : (B,N,F,T_rec)   # 近期序列
        x_hour  : (B,N,F,T_hour) # 小时周期序列
        x_day   : (B,N,F,T_day)  # 日周期序列
    输出 :
        output : (B,N,T_out)     # 预测的未来T_out个时间步
    """

    def __init__(self, DEVICE, num_nodes, F_in, D, T_out, adj_mx, distance_mx=None):
        super().__init__()
        self.DEVICE = DEVICE
        self.num_nodes = num_nodes
        self.F_in = F_in
        self.D = D
        self.T_out = T_out

        # 静态邻接矩阵（用于空间Transformer）
        if isinstance(adj_mx, np.ndarray):
            self.A = torch.tensor(adj_mx, dtype=torch.float32, device=DEVICE)
        else:
            self.A = adj_mx.to(DEVICE)

        # 距离矩阵（用于MGRC）
        if distance_mx is None:
            self.distance_mx = torch.eye(num_nodes, device=DEVICE)
        elif isinstance(distance_mx, np.ndarray):
            self.distance_mx = torch.tensor(distance_mx, dtype=torch.float32, device=DEVICE)
        else:
            self.distance_mx = distance_mx.to(DEVICE)

        # MDAF模块：多周期扩散注意力融合（论文级修复）
        # 使用修复后论文语义的版本：只在周期维度（3）上做attention
        # 🔥 修复：传递 num_nodes 参数给 MDAF，用于 TemporalEncoder 验证
        self.mdaf = MDAF(F_in, D, num_nodes)

        # MGRC模块：多图循环卷积
        self.mgrc = MGRC(num_nodes, D, adj_mx, distance_mx, DEVICE)

        # STFormer模块：时空Transformer
        # 传入num_nodes和adj_mx，确保A矩阵正确注册和device管理
        self.stformer = STFormer(D, num_nodes, num_heads=4, num_layers=2, adj_mx=adj_mx)

        # 最终预测层（公式26）
        self.predictor = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, T_out)
        )

        self.to(DEVICE)

    def forward(self, x_rec, x_hour, x_day):
        B = x_rec.shape[0]

        # MDAF模块：多周期扩散注意力融合（返回 (B,N,D)）
        x = self.mdaf(x_rec, x_hour, x_day)  # (B,N,D)

        # MGRC模块：多图循环卷积
        # 需要添加时间维度 (B,N,D) → (B,N,D,1)
        x = x.unsqueeze(-1)  # (B,N,D,1)
        x = self.mgrc(x)  # (B,N,D,1)
        # 🔥 关键修复：不要squeeze(-1)，保持(B,N,D,T_H)维度给STFormer
        # 原代码squeeze(-1)会导致STFormer接收(B,N,D)，内部reshape时错误地将D当成了N

        # STFormer模块：时空Transformer（期望输入 (B,N,D,T_H)）
        x = self.stformer(x)  # (B,N,D,T_H)

        # 合并时间维度并预测未来T_out个时间步（公式26）
        # 使用最后一个时间步
        x_final = x[:, :, :, -1]  # (B,N,D)

        # 预测未来T_out个时间步
        output = self.predictor(x_final)  # (B,N,T_out)

        return output


############################################
# 7. make_model
############################################
def make_model(
        DEVICE,
        num_nodes,
        F_in,  # 输入特征维度（三个序列共享相同的特征维度）
        D,  # 隐藏维度
        T_out,  # 输出时间步数（预测的未来时间步数）
        adj_mx,
        distance_mx
):
    """
    创建MD-GRTN模型（论文级修复）

    参数说明：
    - num_nodes: 节点数量 N
    - F_in: 输入特征维度（论文中三个序列共享相同的特征维度）
    - D: 隐藏维度
    - T_out: 输出时间步数（预测的未来时间步数）
    - adj_mx: 邻接矩阵
    - distance_mx: 距离矩阵（可选）
    """
    model = MD_GRTN(
        DEVICE=DEVICE,
        num_nodes=num_nodes,
        F_in=F_in,  # 三个序列共享相同的输入特征维度
        D=D,
        T_out=T_out,
        adj_mx=adj_mx,
        distance_mx=distance_mx
    )

    # 初始化权重
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model
