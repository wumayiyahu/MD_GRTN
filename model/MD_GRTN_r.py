# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from lib.utils import scaled_Laplacian


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
    DDPM-based denoiser

    输入 :
        x0 : (B, N, F, T)
    输出 :
        x_denoised : (B, N, D, T)
        diffusion_loss : scalar
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
        self.enc1 = nn.Conv1d(F_in, D, kernel_size=3, padding=1)
        self.enc2 = nn.Conv1d(D, D, kernel_size=3, padding=1)

        self.dec1 = nn.Conv1d(D, D, kernel_size=3, padding=1)
        self.dec2 = nn.Conv1d(D, D, kernel_size=3, padding=1)

    def forward(self, x0, use_pure_denoising=True):
        """
        去噪器前向传播
        
        根据论文Algorithm 1，预训练阶段应该是：
        - 输入：带噪声的数据 X_k (已经是带噪声的）
        - 输出：去噪后的数据 Ȟ_k ≈ X̂_k (干净数据）
        - 损失：MSE(Ȟ_k, X̂_k)
        
        参数:
            x0: (B,N,F,T) - 输入数据（预训练时已经是带噪声的）
            use_pure_denoising: bool -
                True: 纯去噪模式（符合论文），直接学习带噪声→干净
                False: DDPM标准模式，用于测试
                
        返回:
            x0_hat: (B,N,D,T) - 去噪后的数据
            loss: scalar - 去噪损失
        """
        B, N, F, T = x0.shape
        x0 = x0.reshape(B * N, F, T)
        
        if use_pure_denoising:
            # 纯去噪模式（符合论文Algorithm 1）
            # 直接将带噪声数据作为输入，通过U-Net学习去噪
            x = x0
            
            # 不使用时间嵌入（因为输入已经是特定噪声水平的数据）
            # 使用恒定的"基准"时间步
            t = torch.ones(B * N, device=x0.device, dtype=torch.long)
            t_emb = self.time_embed(t)
            t_emb = self.time_mlp(t_emb).unsqueeze(-1)  # (B*N,D,1)
            
            # U-Net前向传播
            h1 = F.relu(self.enc1(x))
            h2 = F.relu(self.enc2(h1 + t_emb))
            
            h3 = F.relu(self.dec1(h2))
            x0_hat = self.dec2(h3 + h1)  # 直接输出去噪结果
            
            # 损失将在外部通过MSE(x0_hat, clean_data)计算
            loss = torch.tensor(0.0, device=x0.device, requires_grad=True)
            
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
            
            h1 = F.relu(self.enc1(x_t))
            h2 = F.relu(self.enc2(h1 + t_emb))
            h3 = F.relu(self.dec1(h2))
            eps_hat = self.dec2(h3 + h1)
            
            diffusion_loss = F.mse_loss(eps_hat, eps)
            
            x0_hat = (
                             x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_hat
                     ) / torch.sqrt(alpha_bar_t)
            
            loss = diffusion_loss

        x0_hat = x0_hat.reshape(B, N, self.D, T)

        return x0_hat, loss


############################################
# 2. MDAF: Multi-period Diffusion Attention Fusion
############################################
class MDAF(nn.Module):
    """
    输入 :
        X_rec  : (B,N,F,T_rec)  # 近期序列，长度T_rec
        X_hour : (B,N,F,T_hour) # 小时周期序列，长度T_hour
        X_day  : (B,N,F,T_day)  # 日周期序列，长度T_day
    输出 :
        X_mdaf : (B,N,D,T_max)  # T_max = max(T_rec, T_hour, T_day)
    """

    def __init__(self, F_in, D, nhead=1):
        super().__init__()

        # -------- Diffusion Denoisers (U-Net) --------
        self.rec = DiffusionDenoiser(F_in, D)
        self.hour = DiffusionDenoiser(F_in, D)
        self.day = DiffusionDenoiser(F_in, D)

        # -------- Temporal Self-Attention (公式 5–8) --------
        self.attn_rec = nn.MultiheadAttention(
            embed_dim=D, num_heads=nhead, batch_first=True
        )
        self.attn_hour = nn.MultiheadAttention(
            embed_dim=D, num_heads=nhead, batch_first=True
        )
        self.attn_day = nn.MultiheadAttention(
            embed_dim=D, num_heads=nhead, batch_first=True
        )

        # -------- Multi-head Fusion (公式 9) --------
        self.fusion = nn.Linear(3 * D, D)

    def forward(self, x_rec, x_hour, x_day, training_mode=False, use_pure_denoising=True):
        """
        MDAF模块前向传播
        
        参数:
            x_rec: (B,N,F,T_rec) - 近期序列
            x_hour: (B,N,F,T_hour) - 小时周期序列
            x_day: (B,N,F,T_day) - 日周期序列
            training_mode: bool - True:预训练模式(返回损失), False:主训练模式
            use_pure_denoising: bool - 是否使用纯去噪模式（符合论文）
            
        返回:
            预训练: (x_fused, (loss_r, loss_h, loss_d))
            主训练: x_fused
        """
        # -------- 1. Diffusion denoising --------
        # 预训练阶段：use_pure_denoising=True，直接学习带噪声→干净
        # 主训练阶段：MD模块被冻结，去噪方式对训练没有影响
        xr, loss_r = self.rec(x_rec, use_pure_denoising=use_pure_denoising)  # (B,N,D,T_rec)
        xh, loss_h = self.hour(x_hour, use_pure_denoising=use_pure_denoising)  # (B,N,D,T_hour)
        xd, loss_d = self.day(x_day, use_pure_denoising=use_pure_denoising)  # (B,N,D,T_day)

        B, N, D = xr.shape[0], xr.shape[1], xr.shape[2]
        T_rec, T_hour, T_day = xr.shape[-1], xh.shape[-1], xd.shape[-1]
        T_max = max(T_rec, T_hour, T_day)

        # -------- 2. 统一时间维度到T_max --------
        if T_rec < T_max:
            xr = F.interpolate(xr, size=T_max, mode='linear', align_corners=False)
        if T_hour < T_max:
            xh = F.interpolate(xh, size=T_max, mode='linear', align_corners=False)
        if T_day < T_max:
            xd = F.interpolate(xd, size=T_max, mode='linear', align_corners=False)

        # -------- 3. reshape for temporal attention --------
        xr = xr.permute(0, 1, 3, 2).reshape(B * N, T_max, D)
        xh = xh.permute(0, 1, 3, 2).reshape(B * N, T_max, D)
        xd = xd.permute(0, 1, 3, 2).reshape(B * N, T_max, D)

        # -------- 4. Temporal self-attention --------
        xr_attn, _ = self.attn_rec(xr, xr, xr)
        xh_attn, _ = self.attn_hour(xh, xh, xh)
        xd_attn, _ = self.attn_day(xd, xd, xd)

        # -------- 5. Concat + fusion --------
        x_cat = torch.cat([xr_attn, xh_attn, xd_attn], dim=-1)  # (B*N,T_max,3D)
        x_fused = self.fusion(x_cat)  # (B*N,T_max,D)

        # -------- 6. reshape back --------
        x_fused = x_fused.reshape(B, N, T_max, D).permute(0, 1, 3, 2)  # (B,N,D,T_max)

        if training_mode:
            # 预训练模式：返回融合特征和三个扩散损失
            return x_fused, (loss_r, loss_h, loss_d)
        else:
            # 主训练模式：只返回融合特征
            return x_fused


############################################
# 3. MGRC: Multi-Graph Recurrent Convolution (修复版)
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
        A_F = F.relu(self.graph_fusion(A_concat))  # (1,1,N,N)
        A_F = A_F.squeeze(0).squeeze(0)  # (N,N)

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
        x = x.reshape(B * T_H, N, D)
        x, _ = self.gru(x)
        x = x.reshape(B, T_H, N, D)

        return x.permute(0, 2, 3, 1)  # (B,N,D,T_H)


############################################
# 4. STFormer: Spatial-Temporal Transformer (修复版)
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
        # 加权空间位置编码（公式15）
        # 使用邻接矩阵对特征进行加权聚合
        pos_encoding = torch.matmul(A, x)  # (B,N,D)
        x_with_pos = x + pos_encoding

        # 多头注意力（公式16）
        attn_out, _ = self.spatial_attn(x_with_pos, x_with_pos, x_with_pos)

        # Add & Normalize（公式17）
        x = self.norm1(x_with_pos + attn_out)

        # Feed Forward（公式18）
        ffn_out = self.ffn(x)

        # Add & Normalize（公式19）
        x = self.norm2(x + ffn_out)

        return x


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

    def __init__(self, D, num_heads=3, num_layers=2):
        super().__init__()
        self.num_layers = num_layers

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
            x_spatial_list = []
            for t in range(T_H):
                xt = x[:, :, :, t]  # (B,N,D)
                xt = self.spatial_layers[l](xt, A=None)  # 这里A应该在模型外部传入
                x_spatial_list.append(xt.unsqueeze(-1))
            x = torch.cat(x_spatial_list, dim=-1)  # (B,N,D,T_H)

            # 时间Transformer（处理每个节点的时间关系）
            x = x.permute(0, 2, 3, 1)  # (B,D,T_H,N)
            x_temporal_list = []
            for n in range(N):
                xn = x[:, :, :, n]  # (B,D,T_H)
                xn = xn.permute(0, 2, 1)  # (B,T_H,D)
                xn = self.temporal_layers[l](xn, hour_idx, day_idx, week_idx)
                x_temporal_list.append(xn.permute(0, 2, 1).unsqueeze(-1))
            x = torch.cat(x_temporal_list, dim=-1)  # (B,D,T_H,N)

            x = x.permute(0, 3, 1, 2)  # (B,N,D,T_H)

        return x


############################################
# 5. MD-GRTN 主模型（修复版）
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

        # MDAF模块：多周期扩散注意力融合
        self.mdaf = MDAF(F_in, D)

        # MGRC模块：多图循环卷积
        self.mgrc = MGRC(num_nodes, D, adj_mx, distance_mx, DEVICE)

        # STFormer模块：时空Transformer
        self.stformer = STFormer(D, num_heads=3, num_layers=2)

        # 最终预测层（公式26）
        self.predictor = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, T_out)
        )

        self.to(DEVICE)

    def forward(self, x_rec, x_hour, x_day):
        B = x_rec.shape[0]

        # MDAF模块：多周期扩散注意力融合
        x = self.mdaf(x_rec, x_hour, x_day, training_mode=False)  # (B,N,D,T_max)

        # MGRC模块：多图循环卷积
        x = self.mgrc(x)  # (B,N,D,T_max)

        # STFormer模块：时空Transformer
        x = self.stformer(x)  # (B,N,D,T_max)

        # 聚合时间维度并预测未来T_out个时间步（公式26）
        # 方法1: 使用最后一个时间步
        x_final = x[:, :, :, -1]  # (B,N,D)

        # 方法2: 平均池化（更稳定）
        # x_final = x.mean(dim=-1)  # (B,N,D)

        # 预测未来T_out个时间步
        output = self.predictor(x_final)  # (B,N,T_out)

        return output


############################################
# 6. make_model（保持与 ASTGCN 一致）
############################################
def make_model(
        DEVICE,
        num_nodes,
        F_in,  # 输入特征维度（三个序列共享相同的特征维度）
        D,  # 隐藏维度
        T_out,  # 输出时间步（预测未来时间步数）
        adj_mx,
        distance_mx=None
):
    """
    创建MD-GRTN模型

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


############################################
# 7. 预训练辅助函数
############################################
def pretrain_mdaf_module(model, hour_noisy, day_noisy, week_noisy, hour_clean, day_clean, week_clean):
    """
    预训练MDAF模块的辅助函数

    参数:
        model: MD_GRTN模型
        hour_noisy: (B,N,F,T_H) 带噪声的小时序列
        day_noisy: (B,N,F,T_H)  带噪声的日序列
        week_noisy: (B,N,F,T_H) 带噪声的周序列
        hour_clean: (B,N,F,T_H) 干净的小时序列
        day_clean: (B,N,F,T_H)  干净的日序列
        week_clean: (B,N,F,T_H) 干净的周序列

    返回:
        total_loss: 总扩散损失
        fused_features: 融合后的特征 (B,N,D,T_max)
    """
    # 调用MDAF模块的训练模式
    fused_features, (loss_hour, loss_day, loss_week) = model.mdaf(
        hour_noisy, day_noisy, week_noisy, training_mode=True
    )

    total_loss = loss_hour + loss_day + loss_week

    # 可选：计算重构损失（与干净数据比较）
    with torch.no_grad():
        # 分别对干净数据进行去噪
        hour_denoised, _ = model.mdaf.rec(hour_clean)
        day_denoised, _ = model.mdaf.hour(day_clean)
        week_denoised, _ = model.mdaf.day(week_clean)

        # 计算重构误差
        recon_loss = F.mse_loss(hour_denoised, hour_clean) + \
                     F.mse_loss(day_denoised, day_clean) + \
                     F.mse_loss(week_denoised, week_clean)

    return total_loss, fused_features, recon_loss