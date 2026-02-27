# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import torch.nn.functional as F_func
import numpy as np

############################################
# 1. Diffusion Denoiser (BackNet_k)
############################################
class DiffusionDenoiser(nn.Module):
    """
    Diffusion denoiser (BackNet_k)
    使用 LeakyReLU 防止方差塌陷
    """
    def __init__(self, F_in, D, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, negative_slope=0.01):
        super().__init__()
        self.D = D
        self.F_in = F_in
        self.negative_slope = negative_slope

        # ========= diffusion schedule =========
        self.num_timesteps = num_timesteps
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # -------- U-Net backbone (1D, time axis) --------
        self.enc1 = nn.Conv1d(F_in, D, kernel_size=3, padding=1)
        self.enc2 = nn.Conv1d(D, D, kernel_size=3, padding=1)
        self.dec1 = nn.Conv1d(D, D, kernel_size=3, padding=1)
        self.dec2 = nn.Conv1d(D, F_in, kernel_size=3, padding=1)
        self.project_to_d = nn.Conv1d(F_in, D, kernel_size=1)

        # 权重初始化
        for m in [self.enc1, self.enc2, self.dec1, self.dec2, self.project_to_d]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x0, return_traffic_space=True):
        B, N, F, T = x0.shape

        # ---------- reshape for 1D conv ----------
        x_in = x0.reshape(B * N, F, T)

        # ---------- U-Net forward ----------
        h1 = F_func.leaky_relu(self.enc1(x_in), negative_slope=self.negative_slope)
        h2 = F_func.leaky_relu(self.enc2(h1), negative_slope=self.negative_slope)
        h3 = F_func.leaky_relu(self.dec1(h2), negative_slope=self.negative_slope)

        # 残差连接
        x0_hat_traffic = self.dec2(h3 + h1)
        x0_hat_hidden = self.project_to_d(x0_hat_traffic)

        # ---------- reshape 回 (B,N,D,T) ----------
        x0_hat_hidden = x0_hat_hidden.view(B, N, self.D, T)
        x0_hat_traffic = x0_hat_traffic.view(B, N, F, T)

        if return_traffic_space:
            return x0_hat_traffic
        else:
            return x0_hat_hidden

class MDAF(nn.Module):
    """
    Multi-period Attention Fusion (MAF)
    Temporal attention only (on T), no node attention.
    Output keeps temporal dimension for STFormer.
    """
    def __init__(self, F_in, D=96, num_heads=3, attn_dropout=0.0):
        super().__init__()
        assert D % num_heads == 0

        self.D = D
        self.num_heads = num_heads
        self.d_h = D // num_heads

        # Diffusion Denoiser for rec, hour, day
        self.rec  = DiffusionDenoiser(F_in, D)
        self.hour = DiffusionDenoiser(F_in, D)
        self.day  = DiffusionDenoiser(F_in, D)

        # Temporal attention projections
        self.q_linear = nn.ModuleList([nn.Linear(D, D) for _ in range(3)])
        self.k_linear = nn.ModuleList([nn.Linear(D, D) for _ in range(3)])
        self.v_linear = nn.ModuleList([nn.Linear(D, D) for _ in range(3)])
        for m in self.q_linear + self.k_linear + self.v_linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

        self.scale = self.d_h ** -0.5
        self.attn_dropout = nn.Dropout(attn_dropout)

        # WMH
        self.concat_fc = nn.Linear(3 * D, D)
        nn.init.xavier_uniform_(self.concat_fc.weight)
        nn.init.zeros_(self.concat_fc.bias)

        # ===== 新增：LayerNorm =====
        self.norm_rec  = nn.LayerNorm(D)
        self.norm_hour = nn.LayerNorm(D)
        self.norm_day  = nn.LayerNorm(D)
        self.norm_fuse = nn.LayerNorm(D)

    def _temporal_mha(self, x, idx):
        """
        x : (B,N,T,D)
        return : (B,N,T,D)
        """
        B, N, T, D = x.shape
        H = self.num_heads
        d = self.d_h

        x = x.reshape(B * N, T, D)

        Q = self.q_linear[idx](x)
        K = self.k_linear[idx](x)
        V = self.v_linear[idx](x)

        Q = Q.view(B * N, T, H, d).permute(0, 2, 1, 3)
        K = K.view(B * N, T, H, d).permute(0, 2, 1, 3)
        V = V.view(B * N, T, H, d).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(B * N, T, D)
        out = out.view(B, N, T, D)

        return out

    def forward(self, x_rec, x_hour, x_day, return_debug=False):
        # x_* : (B,N,F,T)
        xr = self.rec(x_rec, return_traffic_space=False)  # (B,N,D,T)
        xh = self.hour(x_hour, return_traffic_space=False)
        xd = self.day(x_day, return_traffic_space=False)

        # -> (B,N,T,D)
        xr_t = xr.permute(0,1,3,2).contiguous()
        xh_t = xh.permute(0,1,3,2).contiguous()
        xd_t = xd.permute(0,1,3,2).contiguous()

        # ===== Temporal attention + residual + LN =====
        xr_attn = self._temporal_mha(xr_t, 0)
        xr_attn = self.norm_rec(xr_t + xr_attn)

        xh_attn = self._temporal_mha(xh_t, 1)
        xh_attn = self.norm_hour(xh_t + xh_attn)

        xd_attn = self._temporal_mha(xd_t, 2)
        xd_attn = self.norm_day(xd_t + xd_attn)

        # 拼接
        x_concat = torch.cat([xr_attn, xh_attn, xd_attn], dim=-1)

        # 融合
        x_fused = self.concat_fc(x_concat)

        # ===== 融合后的 residual + LN =====
        shortcut = (xr_attn + xh_attn + xd_attn) / 3.0
        x_fused = self.norm_fuse(x_fused + shortcut)

        if return_debug:
            return x_fused, {
                "xr_attn": xr_attn,
                "xh_attn": xh_attn,
                "xd_attn": xd_attn,
                "x_concat": x_concat,
                "zfused": x_fused
            }

        return x_fused


############################################
# 4. MGRC: Multi-Graph Recurrent Convolution
############################################
class MGRC(nn.Module):
    """
    输入 : (B,N,D,T)
    输出 : (B,N,D,T)
    完全按照论文公式 10 和 12 设计，数值稳定
    """

    def __init__(self, num_nodes, D, distance_mx, DEVICE):
        super().__init__()

        self.N = num_nodes
        self.D = D
        self.DEVICE = DEVICE

        dist = torch.tensor(distance_mx, dtype=torch.float32)
        sigma = dist.std()
        Adist = torch.exp(-(dist ** 2) / (sigma ** 2 + 1e-8))
        Adist.fill_diagonal_(0)
        self.register_buffer("Adist", Adist.to(DEVICE))

        self.E1 = nn.Parameter(torch.randn(num_nodes, 1) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, 1) * 0.1)

        self.graph_fusion = nn.Conv2d(2, 1, kernel_size=1, bias=True)

        self.W_GCN = nn.Parameter(torch.empty(D, D))
        nn.init.xavier_uniform_(self.W_GCN)

        self.W_z = nn.Parameter(torch.empty(D, D))
        self.U_z = nn.Parameter(torch.empty(D, D))
        self.b_z = nn.Parameter(torch.zeros(D))
        
        self.W_r = nn.Parameter(torch.empty(D, D))
        self.U_r = nn.Parameter(torch.empty(D, D))
        self.b_r = nn.Parameter(torch.zeros(D))
        
        self.W_h = nn.Parameter(torch.empty(D, D))
        self.U_h = nn.Parameter(torch.empty(D, D))
        self.b_h = nn.Parameter(torch.zeros(D))
        
        self._init_gru_weights()
        
        self.gru = nn.GRU(D, D, batch_first=True)

    def _init_gru_weights(self):
        for weight in [self.W_z, self.U_z, self.W_r, self.U_r, self.W_h, self.U_h]:
            nn.init.xavier_uniform_(weight)
        for bias in [self.b_z, self.b_r, self.b_h]:
            nn.init.zeros_(bias)

    def forward(self, x, return_debug=False):
        # x: (B,N,D,T)
        B, N, D, T = x.shape

        Adyna = torch.softmax(torch.relu(self.E1 @ self.E2.T), dim=-1)

        A_concat = torch.stack([Adyna, self.Adist], dim=0).unsqueeze(0)
        A_F = self.graph_fusion(A_concat)
        A_F = torch.relu(A_F)
        A_F = A_F.squeeze(0).squeeze(0)
        A_F = A_F.unsqueeze(0).expand(B, -1, -1)

        # ---------- GCN + residual ----------
        x_t = x.permute(0, 3, 1, 2).contiguous()  # (B,T,N,D)
        gcn_out = []

        for t in range(T):
            xt = x_t[:, t]       # (B,N,D)
            xt0 = xt             # ★ residual

            xt = torch.bmm(A_F, xt)
            xt = torch.matmul(xt, self.W_GCN)
            xt = torch.relu(xt)

            xt = xt + xt0        # ★ 关键残差

            gcn_out.append(xt)

        x_spatial = torch.stack(gcn_out, dim=1)  # (B,T,N,D)

        # ---------- 自定义 GRU ----------
        x_gru = x_spatial.permute(0, 2, 1, 3).contiguous()
        x_gru = x_gru.view(B * N, T, D)
        
        h = torch.zeros(B * N, self.D, device=x.device)
        
        outputs = []
        for t in range(T):
            x_t2 = x_gru[:, t, :]
            z_t = torch.sigmoid(x_t2 @ self.W_z + h @ self.U_z + self.b_z)
            r_t = torch.sigmoid(x_t2 @ self.W_r + h @ self.U_r + self.b_r)
            h_tilde = torch.tanh(
                x_t2 @ self.W_h + (r_t * h) @ self.U_h + self.b_h
            )
            h = (1 - z_t) * h + z_t * h_tilde
            outputs.append(h)
        
        x_out = torch.stack(outputs, dim=1)
        x_out = x_out.view(B, N, T, self.D)
        x_out = x_out.permute(0, 1, 3, 2).contiguous()

        if return_debug:
            debug_dict = {
                "Adyna_mean": Adyna.mean().item(),
                "Adyna_std": Adyna.std().item(),
                "AF_mean": A_F.mean().item(),
                "AF_std": A_F.std().item(),
                "output_mean": x_out.mean().item(),
                "output_std": x_out.std().item(),
                "h_mean": h.mean().item(),
                "h_std": h.std().item(),
            }
            return x_out, debug_dict

        return x_out
   
############################################
# 5. STFormer: Spatial-Temporal Transformer 
############################################
class SpatialTransformer(nn.Module):
    """
    空间Transformer模块（公式15-19）
    """
    def __init__(self, num_nodes, D, num_heads=3):
        super().__init__()

        # W_SPE
        self.W_spe = nn.Parameter(torch.randn(num_nodes, D))

        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=D, num_heads=num_heads, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(D, 4 * D),
            nn.ReLU(),
            nn.Linear(4 * D, D)
        )

        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)

    def forward(self, x, A):
        """
        x: (B,N,D)
        A: (N,N)
        """
        B, N, D = x.shape

        # (15)  A X W_SPE
        pos = torch.einsum('ij, jd->id', A, self.W_spe)  # (N, D)
        pos = pos.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)
        x_s1 = x + pos

        x_s1 = x + pos                               # X_S1

        # (16)
        attn_out, _ = self.spatial_attn(x_s1, x_s1, x_s1)

        # (17)
        x_s3 = self.norm1(attn_out + x_s1)

        # (18)
        x_s4 = self.ffn(x_s3)

        # (19)（按标准 Transformer 结构实现）
        y_s = self.norm2(x_s4 + x_s3)

        return y_s

class TemporalTransformer(nn.Module):
    """
    时间Transformer模块（公式21-25）
    """
    def __init__(self, D, num_nodes, num_heads=3):
        super().__init__()

        self.num_nodes = num_nodes

        # W_hour, W_day, W_week ∈ R^{N×1}
        self.w_hour = nn.Parameter(torch.ones(num_nodes,1))
        self.w_day  = nn.Parameter(torch.ones(num_nodes,1))
        self.w_week = nn.Parameter(torch.ones(num_nodes,1))


        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=D, num_heads=num_heads, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(D, 4 * D),
            nn.ReLU(),
            nn.Linear(4 * D, D)
        )

        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)

    def forward(self, x, hour_idx, day_idx, week_idx):
        """
        x: (B, N, T, D)
        """
        B, N, T, D = x.shape

        # (21) 位置编码
        hour_E = hour_idx[0:1].float()  # (1, T)
        hour_enc = self.w_hour @ hour_E  # (N,1) @ (1,T) -> (N,T)
        hour_enc = hour_enc.unsqueeze(0).expand(B, -1, -1)  # (B, N, T)
        
        day_E = day_idx[0:1].float()  # (1, T)
        day_enc = self.w_day @ day_E  # (N, T)
        day_enc = day_enc.unsqueeze(0).expand(B, -1, -1)  # (B, N, T)
        
        week_E = week_idx[0:1].float()  # (1, T)
        week_enc = self.w_week @ week_E  # (N, T)
        week_enc = week_enc.unsqueeze(0).expand(B, -1, -1)  # (B, N, T)
        
        # 扩展特征维度: (B, N, T) -> (B, N, T, D)
        hour_enc = hour_enc.unsqueeze(-1).expand(-1, -1, -1, D)
        day_enc = day_enc.unsqueeze(-1).expand(-1, -1, -1, D)
        week_enc = week_enc.unsqueeze(-1).expand(-1, -1, -1, D)
        
        # 位置编码相加
        x_t2 = x + hour_enc + day_enc + week_enc  # (B, N, T, D)
        
        # 重塑为 (B*N, T, D) 用于MultiheadAttention
        x_t2 = x_t2.reshape(B * N, T, D)

        # (22)
        attn_out, _ = self.temporal_attn(x_t2, x_t2, x_t2)

        # (23)
        x_t3 = self.norm1(attn_out + x_t2)

        # (24)
        x_t4 = self.ffn(x_t3)

        # (25)
        x_out = self.norm2(x_t4 + x_t3)

        # 恢复形状
        x_out = x_out.reshape(B, N, T, D)  # (B, N, T, D)

        return x_out


class STFormer(nn.Module):

    def __init__(self, D, num_nodes, num_heads=3, num_layers=3, adj_mx=None):
        super().__init__()
        self.num_layers = num_layers
        self.num_nodes = num_nodes

        if adj_mx is not None:
            if isinstance(adj_mx, torch.Tensor):
                self.register_buffer('A', adj_mx)
            else:
                self.register_buffer('A', torch.tensor(adj_mx, dtype=torch.float32))
        else:
            self.register_buffer('A', torch.eye(num_nodes))

        self.spatial_layers = nn.ModuleList([
            SpatialTransformer(num_nodes,D, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        self.temporal_layers = nn.ModuleList([
            TemporalTransformer(D, num_nodes, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        # 用于公式(20)
        self.fusion_norm = nn.ModuleList([
            nn.LayerNorm(D) for _ in range(num_layers)  # 使用时需要重塑为 (B*N*T, D)
        ])


    def forward(self, x, hour_idx=None, day_idx=None, week_idx=None):
        """
        x: (B,N,D,T)
        """
        B, N, D, T = x.shape
        device = x.device

        if hour_idx is None:
            hour_idx = torch.arange(T, device=device).unsqueeze(0).repeat(B, 1) % 60
        if day_idx is None:
            day_idx = torch.arange(T, device=device).unsqueeze(0).repeat(B, 1) % 24
        if week_idx is None:
            week_idx = torch.arange(T, device=device).unsqueeze(0).repeat(B, 1) % 7

        # 转换为内部统一格式: (B, N, T, D) 用于时间处理
        x = x.permute(0, 1, 3, 2).contiguous()  # (B, N, T, D)
        
        for l in range(self.num_layers):
            x_prev = x  # (B, N, T, D)

            # -------- Spatial --------
            # 空间处理需要 (B*T, N, D) 格式
            x_sp = x_prev.permute(0, 2, 1, 3).contiguous()  # (B, T, N, D)
            x_sp = x_sp.reshape(B * T, N, D)
            
            y_s = self.spatial_layers[l](x_sp, self.A)  # (B*T, N, D)
            y_s = y_s.reshape(B, T, N, D).permute(0, 2, 1, 3)  # (B, N, T, D)

            # -------- 公式 (20) 融合 --------
            # LayerNorm 需要输入形状 [*, D]，重塑为 (B*N*T, D)
            residual = x_prev + y_s  # (B, N, T, D)
            residual_flat = residual.reshape(-1, D)
            normalized = self.fusion_norm[l](residual_flat)
            x_t1 = normalized.reshape(B, N, T, D)  # (B, N, T, D)

            # -------- Temporal --------
            # TemporalTransformer 接受 (B, N, T, D)
            x_t = self.temporal_layers[l](x_t1, hour_idx, day_idx, week_idx)  # (B, N, T, D)
            
            x = x_t  # 更新为当前层输出

        # 转换回原始格式 (B, N, D, T)
        output = x.permute(0, 1, 3, 2).contiguous()  # (B, N, D, T)
        return output



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

    def __init__(self, DEVICE, num_nodes, F_in, D, T_out, adj_mx, distance_mx=None, num_heads=3, num_layers=3):
        super().__init__()
        self.DEVICE = DEVICE
        self.num_nodes = num_nodes
        self.F_in = F_in
        self.D = D
        self.T_out = T_out

        # 距离矩阵（用于MGRC）
        if distance_mx is None:
            self.distance_mx = torch.eye(num_nodes, device=DEVICE)
        elif isinstance(distance_mx, np.ndarray):
            self.distance_mx = torch.tensor(distance_mx, dtype=torch.float32, device=DEVICE)
        else:
            self.distance_mx = distance_mx.to(DEVICE)

        # MDAF
        self.mdaf = MDAF(F_in, D)

        # MGRC
        self.mgrc = MGRC(num_nodes, D,  distance_mx, DEVICE)

        # STFormer
        # 传入num_nodes和adj_mx，确保A矩阵正确注册和device管理
        self.stformer = STFormer(D, num_nodes, num_heads=num_heads, num_layers=num_layers, adj_mx=adj_mx)

        # 最终预测层（公式26）
        self.predictor = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, T_out)
        )

        self.to(DEVICE)

    def forward(self, x_rec, x_hour, x_day, return_debug=False):

        debug_info = {}
        debug_mdaf = None
        debug_mgrc = None

        # ---------- MDAF模块 ----------
        if return_debug:
            x, debug_mdaf = self.mdaf(x_rec, x_hour, x_day, return_debug=True)
            debug_info['mdaf'] = debug_mdaf
        else:
            x = self.mdaf(x_rec, x_hour, x_day)

        # (B,N,T,D) -> (B,N,D,T)
        x = x.permute(0,1,3,2).contiguous()

        # ---------- MGRC模块 ----------
        if return_debug:
            x, debug_mgrc = self.mgrc(x, return_debug=True)
            debug_info['mgrc'] = debug_mgrc
        else:
            x = self.mgrc(x)

        # ---------- STFormer模块 ----------
        x = self.stformer(x)

        # ---------- 最后时间步 ----------
        x_final = x[:, :, :, -1]
        output = self.predictor(x_final)

        if return_debug:
            return output, debug_info

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
        distance_mx,
        num_heads=3,
        num_layers=3
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
    - num_heads: 注意力头的数量
    - num_layers: STFormer 层数
    """
    model = MD_GRTN(
        DEVICE=DEVICE,
        num_nodes=num_nodes,
        F_in=F_in,  # 三个序列共享相同的输入特征维度
        D=D,
        T_out=T_out,
        adj_mx=adj_mx,
        distance_mx=distance_mx,
        num_heads=num_heads,
        num_layers=num_layers
    )

    # 初始化权重
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model
