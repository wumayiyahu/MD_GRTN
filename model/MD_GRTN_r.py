# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian


############################################
# 1. Diffusion Denoiser (工程稳定版)
############################################
class DiffusionDenoiser(nn.Module):
    """
    输入 : (B, N, F, T)
    输出 : (B, N, D, T)
    """
    def __init__(self, F_in, D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(F_in, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )

    def forward(self, x):
        # x: (B,N,F,T)
        x = x.permute(0, 1, 3, 2)      # (B,N,T,F)
        x = self.net(x)               # (B,N,T,D)
        return x.permute(0, 1, 3, 2)  # (B,N,D,T)


############################################
# 2. MDAF: Multi-period Diffusion Attention Fusion
############################################
class MDAF(nn.Module):
    """
    输入 :
        X_rec  : (B,N,F,T)
        X_hour : (B,N,F,T)
        X_day  : (B,N,F,T)
    输出 :
        X_mdaf : (B,N,D,T)
    """
    def __init__(self, F_in, D):
        super().__init__()
        self.rec = DiffusionDenoiser(F_in, D)
        self.hour = DiffusionDenoiser(F_in, D)
        self.day = DiffusionDenoiser(F_in, D)

        # 用可学习权重做周期融合（比简单平均更合理）
        self.alpha = nn.Parameter(torch.ones(3))

    def forward(self, x_rec, x_hour, x_day):
        xr = self.rec(x_rec)      # (B,N,D,T)
        xh = self.hour(x_hour)
        xd = self.day(x_day)

        w = torch.softmax(self.alpha, dim=0)
        x = w[0] * xr + w[1] * xh + w[2] * xd

        return x                  # (B,N,D,T)


############################################
# 3. MGRC: Multi-Graph Recurrent Convolution
############################################
class MGRC(nn.Module):
    """
    输入 : (B,N,D,T)
    输出 : (B,N,D,T)
    """
    def __init__(self, num_nodes, D, adj_mx, DEVICE):
        super().__init__()
        self.N = num_nodes
        self.D = D
        self.DEVICE = DEVICE

        # 静态图
        self.Adist = torch.tensor(adj_mx, dtype=torch.float32, device=DEVICE)

        # 动态图参数（论文中的 E1, E2）
        self.E1 = nn.Parameter(torch.randn(num_nodes, D))
        self.E2 = nn.Parameter(torch.randn(num_nodes, D))

        # 递归建模时间
        self.gru = nn.GRU(D, D, batch_first=True)

    def forward(self, x):
        # x: (B,N,D,T)
        B, N, D, T = x.shape

        # 动态邻接矩阵
        Adyna = torch.softmax(
            torch.relu(self.E1 @ self.E2.T),
            dim=-1
        )                          # (N,N)

        A = self.Adist + Adyna     # (N,N)

        x = x.permute(0, 3, 1, 2)  # (B,T,N,D)

        gcn_out = []
        for t in range(T):
            xt = x[:, t]           # (B,N,D)
            xt = torch.matmul(A, xt)  # (B,N,D)
            gcn_out.append(xt)

        x = torch.stack(gcn_out, dim=1)   # (B,T,N,D)

        # GRU 建模时间递归
        x = x.reshape(B*T, N, D)
        x, _ = self.gru(x)
        x = x.reshape(B, T, N, D)

        return x.permute(0, 2, 3, 1)  # (B,N,D,T)


############################################
# 4. STFormer: Spatial-Temporal Transformer
############################################
class STFormer(nn.Module):
    """
    输入 : (B,N,D,T)
    输出 : (B,N,D,T)
    """
    def __init__(self, D, nhead=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        # x: (B,N,D,T)
        B, N, D, T = x.shape
        x = x.permute(0, 1, 3, 2)      # (B,N,T,D)
        x = x.reshape(B*N, T, D)
        x = self.encoder(x)
        x = x.reshape(B, N, T, D)
        return x.permute(0, 1, 3, 2)  # (B,N,D,T)


############################################
# 5. MD-GRTN 主模型
############################################
class MD_GRTN(nn.Module):
    """
    输入 : (B,N,F,T)
    输出 : (B,N,T_out)
    """
    def __init__(self, DEVICE, num_nodes, F_in, D, T_out, adj_mx):
        super().__init__()
        self.mdaf = MDAF(F_in, D)
        self.mgrc = MGRC(num_nodes, D, adj_mx, DEVICE)
        self.stformer = STFormer(D, nhead=4, num_layers=2)

        self.proj = nn.Conv2d(
            in_channels=D,
            out_channels=T_out,
            kernel_size=(1, 1)
        )

        self.to(DEVICE)

    def forward(self, x):
        # x: (B,N,F,T)
        x = self.mdaf(x, x, x)      # (B,N,D,T)
        x = self.mgrc(x)            # (B,N,D,T)
        x = self.stformer(x)        # (B,N,D,T)
        x = self.proj(x)            # (B,N,T_out,T)
        return x[..., -1]           # (B,N,T_out)


############################################
# 6. make_model（保持与 ASTGCN 一致）
############################################
def make_model(
    DEVICE,
    num_nodes,
    F_in,
    D,
    T_out,
    adj_mx
):
    model = MD_GRTN(
        DEVICE=DEVICE,
        num_nodes=num_nodes,
        F_in=F_in,
        D=D,
        T_out=T_out,
        adj_mx=adj_mx
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model
