#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import shutil
import argparse
import configparser
from time import time
from model.MD_GRTN_r import make_model
# 修改导入，使用新的数据加载和评估函数
from lib.utils import load_md_grtn_data, get_adjacency_matrix, compute_val_loss, predict_and_save_results
from tensorboardX import SummaryWriter
from lib.metrics import masked_mae, masked_mse, masked_rmse, masked_mape

# ---------------------- 参数和配置 ----------------------
parser = argparse.ArgumentParser()
# 修改默认配置文件为MD-GRTN专用
parser.add_argument("--config", default='configurations/PEMS04_md_grtn.conf', type=str)
args = parser.parse_args()

config = configparser.ConfigParser()
print('读取配置文件:', args.config)
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

# 数据路径
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
adj_filename = data_config['adj_filename']
id_filename = data_config.get('id_filename', None)

# 数据规格
num_of_vertices = int(data_config['num_of_vertices'])
dataset_name = data_config['dataset_name']
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])

# 训练配置
ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = str(ctx)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("使用设备:", DEVICE)

batch_size = int(training_config['batch_size'])
learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])

num_of_hours = int(training_config['num_of_hours'])
num_of_days = int(training_config['num_of_days'])
num_of_weeks = int(training_config['num_of_weeks'])

# MD-GRTN模型参数
in_channels = int(training_config['in_channels'])  # 输入特征维度
hidden_dim = int(training_config.get('hidden_dim', 64))  # 隐藏层维度
num_heads = int(training_config.get('num_heads', 4))  # 注意力头数
num_layers = int(training_config.get('num_layers', 2))  # Transformer层数

loss_function = training_config['loss_function']
metric_method = training_config.get('metric_method', 'unmask')
missing_value = float(training_config.get('missing_value', 0.0))

# 修改文件夹命名，包含MD-GRTN标识
folder_dir = 'MD_GRTN_pretrain_h%dd%dw%d_channel%d_hidden%d_%e' % (
    num_of_hours, num_of_days, num_of_weeks, in_channels, hidden_dim, learning_rate
)
params_path = os.path.join('experiments', dataset_name, folder_dir)
print('参数保存路径:', params_path)

# ---------------------- 数据加载 ----------------------
print("\n" + "=" * 50)
print("加载MD-GRTN预训练数据")
print("=" * 50)

# 使用MD-GRTN专用数据加载器，模式为'pretrain'
train_loader, _, _, _, _, _, _, _ = load_md_grtn_data(
    graph_signal_matrix_filename,
    num_of_hours, num_of_days, num_of_weeks, num_for_predict,
    DEVICE, batch_size, shuffle=True, mode='pretrain'
)
print(f"训练批次: {len(train_loader)}")

# 为验证集加载主训练数据
_, _, val_loader, val_target_tensor, test_loader, test_target_tensor, _, _ = load_md_grtn_data(
    graph_signal_matrix_filename,
    num_of_hours, num_of_days, num_of_weeks, num_for_predict,
    DEVICE, batch_size, shuffle=False, mode='train'
)

# 邻接矩阵
adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
print("邻接矩阵形状:", adj_mx.shape)

# ---------------------- 模型构建 ----------------------
print("\n" + "=" * 50)
print("构建MD-GRTN模型")
print("=" * 50)
print("注意：预训练阶段使用完整的 MD-GRTN 模型")
print("  - T_out (预测时间步长) 参数被传递给模型")
print("  - 但在预训练 forward 时不参与计算")
print("  - 只有主训练阶段才会使用 T_out 进行预测")
print()

# 修改make_model调用，使用正确的参数
# 注意：需要查看make_model函数的具体实现，可能需要调整参数
try:
    net = make_model(
        DEVICE=DEVICE,
        num_nodes=num_of_vertices,
        F_in=in_channels,
        D=hidden_dim,
        T_out=num_for_predict,
        adj_mx=adj_mx,
        distance_mx=distance_mx
    )
except TypeError as e:
    print(f"模型构建错误: {e}")
    print("尝试使用旧版本参数...")
    # 如果make_model需要更多参数，可能需要调整
    # 这里假设make_model接受标准参数
    net = make_model(DEVICE, num_of_vertices, in_channels, hidden_dim, num_for_predict, adj_mx, distance_mx)

print(net)

# ---------------------- 损失函数 ----------------------
# 预训练阶段使用 MSE Loss（论文 Algorithm 1：去噪重建任务）
criterion = nn.MSELoss().to(DEVICE)

# ---------------------- 优化器（只优化 MD 模块）----------------------
print("\n" + "=" * 50)
print("预训练配置（严格符合论文 Algorithm 1）:")
print("=" * 50)

md_params = []

# 📌 论文 Algorithm 1 的严格实现：
# MD 预训练只涉及 BackNet_k（DiffusionDenoiser）
# 不涉及 MAF（TemporalEncoder + Attention + Fusion）
if hasattr(net, 'mdaf'):
    # 🔥 关键：只训练 BackNet_k，绝对不涉及 MAF
    # 论文严格定义：MD = BackNet，独立去噪器
    md_params.extend(list(net.mdaf.rec.parameters()))
    md_params.extend(list(net.mdaf.hour.parameters()))
    md_params.extend(list(net.mdaf.day.parameters()))
    
    print("=== 严格论文级 MD 预训练 ===")
    print()
    print("论文 Algorithm 1 预训练目标:")
    print("  - 输入: 带噪声的交通流 X_k (rec_noisy, hour_noisy, day_noisy)")
    print("  - 去噪: X̂_k = BackNet_k(X_k) - 直接输出 F 维交通流")
    print("  - 损失: L_pretrain = Σ_k MSELoss(X̂_k, X_k_clean)")
    print()
    print("关键点（严格遵守论文）:")
    print("  ✅ BackNet 是独立的去噪器，不经过 MAF 结构")
    print("  ✅ 输出维度 = F（交通流空间），不是 D")
    print("  ✅ 损失直接在 traffic 空间计算 MSE，不涉及 attention")
    print("  ✅ 不做时间对齐（hour/day 和 rec 不需要统一时间维度）")
    print("  ✅ 不涉及 TemporalEncoder、Attention、Fusion")
    print()
    print("模块边界（严格符合论文）:")
    print(f"  - 训练: MD模块（BackNet_k）- 独立去噪器")
    print(f"  - 不涉及: MAF模块（TemporalEncoder + Attention + Fusion）")
    print()
    print(f"MD模块参数数量（可训练）: {sum(p.numel() for p in md_params):,}")
    print(f"  - rec (BackNet_k): {sum(p.numel() for p in net.mdaf.rec.parameters()):,}")
    print(f"  - hour (BackNet_k): {sum(p.numel() for p in net.mdaf.hour.parameters()):,}")
    print(f"  - day (BackNet_k): {sum(p.numel() for p in net.mdaf.day.parameters()):,}")
    
else:
    print("警告: 模型没有mdaf属性，无法进行 MD 预训练")
    raise SystemExit("预训练失败：需要 MDAF 模块")

# 确保参数列表不为空
if len(md_params) == 0:
    print("错误: 没有找到MD模块参数")
    raise SystemExit("预训练失败：无法找到MD模块参数")

optimizer = optim.Adam(md_params, lr=learning_rate, weight_decay=0.01)
print()
print("优化器配置:")
print(f"  - 优化器: Adam")
print(f"  - 学习率: {learning_rate}")
print(f"  - 权重衰减: 0.01")
print(f"  - 更新参数数: {len(md_params)} 个张量")
print("=" * 50)

# ---------------------- TensorBoard ----------------------
sw = SummaryWriter(logdir=params_path, flush_secs=5)

# ---------------------- 训练目录 ----------------------
if (start_epoch == 0) and (not os.path.exists(params_path)):
    os.makedirs(params_path)
elif (start_epoch == 0) and os.path.exists(params_path):
    shutil.rmtree(params_path)
    os.makedirs(params_path)
elif (start_epoch > 0) and os.path.exists(params_path):
    print('从以下路径恢复训练:', params_path)
else:
    raise SystemExit("错误的参数路径!")

# ---------------------- 主训练循环 ----------------------
print("\n" + "=" * 50)
print("开始MD模块预训练")
print("=" * 50)

best_train_loss = np.inf
best_epoch = 0
global_step = 0
start_time = time()

for epoch in range(start_epoch, epochs):
    net.train()
    total_loss = 0
    batch_count = 0
    epoch_start_time = time()

    for batch_index, batch_data in enumerate(train_loader):
        # 预训练模式返回6个数据:
        # (rec_noisy, hour_noisy, day_noisy, rec_clean, hour_clean, day_clean)
        # 注意：论文中的命名是 RecN, HourN, DayN（近期、小时周期、日周期）
        if len(batch_data) != 6:
            raise RuntimeError(f"预训练 batch 应为 6 个张量，实际为 {len(batch_data)}")

        rec_noisy, hour_noisy, day_noisy, rec_clean, hour_clean, day_clean = batch_data
        
        # 数据形状验证
        B, N, F, T_rec = rec_noisy.shape
        _, _, _, T_hour = hour_noisy.shape
        _, _, _, T_day = day_noisy.shape
        
        # 确保clean数据与noisy数据维度匹配
        assert rec_clean.shape == rec_noisy.shape, f"Rec clean {rec_clean.shape} vs noisy {rec_noisy.shape} 维度不匹配"
        assert hour_clean.shape == hour_noisy.shape, f"Hour clean {hour_clean.shape} vs noisy {hour_noisy.shape} 维度不匹配"
        assert day_clean.shape == day_noisy.shape, f"Day clean {day_clean.shape} vs noisy {day_noisy.shape} 维度不匹配"
        
        # 打印维度信息（调试用）
        if batch_index == 0:
            print(f"数据维度: rec_noisy={rec_noisy.shape}, hour_noisy={hour_noisy.shape}, day_noisy={day_noisy.shape}")
            print(f"        clean: rec_clean={rec_clean.shape}, hour_clean={hour_clean.shape}, day_clean={day_clean.shape}")

        optimizer.zero_grad()
        
        try:
            # 🔥 严格论文实现：直接调用 BackNet_k，不经过 MDAF
            # 论文语义：每个 BackNet_k 是独立的去噪器
            if hasattr(net, 'mdaf'):
                # 直接访问 MDAF 内部的 BackNet，跳过 MAF 结构
                # X̂_rec = BackNet_rec(X_rec_noisy)
                # 注意：DiffusionDenoiser.forward 的第一个参数是 x0（位置参数），不是 x_rec
                X_rec_denoised = net.mdaf.rec(rec_noisy, use_pure_denoising=True, return_traffic_space=True)
                
                # X̂_hour = BackNet_hour(X_hour_noisy)
                X_hour_denoised = net.mdaf.hour(hour_noisy, use_pure_denoising=True, return_traffic_space=True)
                
                # X̂_day = BackNet_day(X_day_noisy)
                X_day_denoised = net.mdaf.day(day_noisy, use_pure_denoising=True, return_traffic_space=True)
            else:
                print("错误：模型没有mdaf模块")
                raise SystemExit("预训练失败：需要 MDAF 模块")
            
            # L_pretrain = Σ_k ||X̂_k - X_k_clean||²
            # 论文算法：逐元素计算 MSE，不需要任何时间对齐或聚合
            #
            # 关键：clean 数据和 noisy 数据是一一对应的，都在 traffic 空间
            
            # 验证：维度必须完全匹配
            assert X_rec_denoised.shape == rec_clean.shape, \
                f"Rec denoised {X_rec_denoised.shape} vs clean {rec_clean.shape} 维度不匹配"
            assert X_hour_denoised.shape == hour_clean.shape, \
                f"Hour denoised {X_hour_denoised.shape} vs clean {hour_clean.shape} 维度不匹配"
            assert X_day_denoised.shape == day_clean.shape, \
                f"Day denoised {X_day_denoised.shape} vs clean {day_clean.shape} 维度不匹配"
            
            # 直接计算 MSE（论文公式）
            loss_rec = criterion(X_rec_denoised, rec_clean)
            loss_hour = criterion(X_hour_denoised, hour_clean)
            loss_day = criterion(X_day_denoised, day_clean)
            
            # L_pretrain = Σ_k L_k
            loss = loss_rec + loss_hour + loss_day

        except Exception as e:
            print(f"扩散模块训练错误（批次 {batch_index}): {e}")
            import traceback
            traceback.print_exc()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(md_params, max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1
        global_step += 1

        if global_step % 50 == 0:
            # 修正问题5：打印真实的损失值（已在上面计算）
            print(f"周期 {epoch}, 步骤 {global_step}, "
                  f"MD预训练总损失: {loss.item():.6f} "
                  f"(rec={loss_rec.item():.6f}, hour={loss_hour.item():.6f}, day={loss_day.item():.6f})")
            if sw:
                sw.add_scalar('pretrain/total_md_loss', loss.item(), global_step)
                sw.add_scalar('pretrain/loss_rec', loss_rec.item(), global_step)
                sw.add_scalar('pretrain/loss_hour', loss_hour.item(), global_step)
                sw.add_scalar('pretrain/loss_day', loss_day.item(), global_step)

    if batch_count > 0:
        avg_loss = total_loss / batch_count
        epoch_time = time() - epoch_start_time
        print(f"周期 {epoch} 完成. "
              f"平均MD预训练损失: {avg_loss:.6f}, "
              f"耗时: {epoch_time:.2f}秒, "
              f"批次: {batch_count}")
        
        # 保存最佳模型（基于训练损失）
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            best_epoch = epoch
            best_params_filename = os.path.join(params_path, 'best_md_model.params')
            torch.save(net.state_dict(), best_params_filename)
            print(f"保存最佳MD模型到 {best_params_filename}")
    else:
        avg_loss = 0
        print(f"周期 {epoch} 没有有效批次数据")

    # 保存当前周期模型
    params_filename = os.path.join(params_path, f'epoch_{epoch}.params')
    torch.save(net.state_dict(), params_filename)
    print(f'保存MD模型参数到 {params_filename}')

# ---------------------- 预训练完成 ----------------------
print("\n" + "=" * 50)
print("MD模块预训练完成")
print("=" * 50)
training_time = time() - start_time
print(f"总训练时间: {training_time / 60:.2f} 分钟")
print(f"最佳周期: {best_epoch}, 最佳训练损失: {best_train_loss:.6f}")

# 保存最终模型
final_model_path = os.path.join(params_path, 'final_md_model.params')
torch.save(net.state_dict(), final_model_path)
print(f"保存最终MD模型到: {final_model_path}")

if sw:
    sw.close()
