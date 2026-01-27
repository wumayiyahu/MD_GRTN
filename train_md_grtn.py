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
# 修改导入
from lib.utils import load_md_grtn_data, get_adjacency_matrix, compute_val_loss, predict_and_save_results
from tensorboardX import SummaryWriter
from lib.metrics import masked_mae, masked_mse, masked_rmse, masked_mape, masked_huber_loss

# ---------------------- 参数 ----------------------
parser = argparse.ArgumentParser()
# 修改默认配置文件
parser.add_argument("--config", default='configurations/PEMS04_md_grtn.conf', type=str)
args = parser.parse_args()

config = configparser.ConfigParser()
print('读取配置文件:', args.config)
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
adj_filename = data_config['adj_filename']
id_filename = data_config.get('id_filename', None)
num_of_vertices = int(data_config['num_of_vertices'])
dataset_name = data_config['dataset_name']
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])

# GPU
ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = str(ctx)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("使用设备:", DEVICE)

# 训练参数
batch_size = int(training_config['batch_size'])
learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])

# 根据论文符号定义的参数说明：
# num_of_hours  → 论文中的 Rec (最近连续时间)
# num_of_days   → 论文中的 Hour (小时周期，24小时模式)
# num_of_weeks  → 论文中的 Day (日周期，7天模式)
#
# 数据流：
# X_Rec  (num_of_hours步)  → 传入第一个参数 (rec_data)
# X_Hour (num_of_days步)   → 传入第二个参数 (hour_data)
# X_Day  (num_of_weeks步)  → 传入第三个参数 (day_data)

num_of_hours = int(training_config['num_of_hours'])   # 论文 Rec 周期
num_of_days = int(training_config['num_of_days'])    # 论文 Hour 周期
num_of_weeks = int(training_config['num_of_weeks'])  # 论文 Day 周期

# MD-GRTN模型参数
in_channels = int(training_config['in_channels'])
hidden_dim = int(training_config.get('hidden_dim', '64'))
num_heads = int(training_config.get('num_heads', '4')) #应该是3
num_layers = int(training_config.get('num_layers', '2'))

# 参数验证
if hidden_dim % num_heads != 0:
    print(f"警告: hidden_dim={hidden_dim} 不能被 num_heads={num_heads} 整除")
    print(f"自动调整 num_heads 为 4 (64 可被 4 整除)")
    num_heads = 4

loss_function = training_config['loss_function']
metric_method = training_config['metric_method']
missing_value = float(training_config.get('missing_value', 0.0))

# 修改文件夹命名
folder_dir = 'MD_GRTN_main_h%dd%dw%d_channel%d_hidden%d_%e' % (
    num_of_hours, num_of_days, num_of_weeks, in_channels, hidden_dim, learning_rate
)
params_path = os.path.join('experiments', dataset_name, folder_dir)
print('参数保存路径:', params_path)

# ---------------------- 数据加载 ----------------------
print("\n" + "=" * 50)
print("加载MD-GRTN主训练数据")
print("=" * 50)

# 使用MD-GRTN专用数据加载器，模式为'train'
# 返回：train_loader, train_target, val_loader, val_target, test_loader, test_target
# 数据映射（根据论文符号）：
# - num_of_hours  → X_Rec (最近连续时间)
# - num_of_days   → X_Hour (小时周期，24小时模式)
# - num_of_weeks  → X_Day (日周期，7天模式)
train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _, _ = load_md_grtn_data(
    graph_signal_matrix_filename,
    num_of_hours, num_of_days, num_of_weeks, num_for_predict,
    DEVICE, batch_size, shuffle=True, mode='train'
)

print(f"数据集信息（根据论文符号定义）:")
print(f"  X_Rec (num_of_hours={num_of_hours}): 最近连续时间")
print(f"  X_Hour (num_of_days={num_of_days}): 小时周期（24小时模式）")
print(f"  X_Day (num_of_weeks={num_of_weeks}): 日周期（7天模式）")
print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}, 测试批次: {len(test_loader)}")

# 邻接矩阵和距离矩阵
adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
print("邻接矩阵形状:", adj_mx.shape)
print("距离矩阵形状:", distance_mx.shape if distance_mx is not None else "无")

# ---------------------- 模型构建 ----------------------
print("\n" + "=" * 50)
print("构建MD-GRTN模型")
print("=" * 50)

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
    print(f"模型构建错误（可能缺少distance_mx参数）: {e}")
    print("尝试不传distance_mx...")
    net = make_model(
        DEVICE=DEVICE,
        num_nodes=num_of_vertices,
        F_in=in_channels,
        D=hidden_dim,
        T_out=num_for_predict,
        adj_mx=adj_mx
    )

print(net)

# ---------------------- 加载预训练权重（MD模块）----------------------
print("\n" + "=" * 50)
print("主训练配置（根据论文Algorithm 1）:")
print("-" * 50)
print("论文符号定义与模型映射:")
print("  - Rec (X_Rec): 最近连续时间，由num_of_hours参数确定")
print("  - Hour (X_Hour): 小时周期，由num_of_days参数确定（24小时模式）")
print("  - Day (X_Day): 日周期，由num_of_weeks参数确定（7天模式）")
print("-" * 50)
print("  - 训练MAF模块参数: θ_MAF")
print("  - 训练MGRC模块参数: θ_MGRC")
print("  - 训练STFormer模块参数: θ_ST")
print("  - 冻结MD模块参数: θ_MD（来自预训练）")
print("    - 冻结rec去噪器 (处理 X_Rec)")
print("    - 冻结hour去噪器 (处理 X_Hour)")
print("    - 冻结day去噪器 (处理 X_Day)")
print("=" * 50)

# 查找预训练模型
pretrain_dir = params_path.replace('main', 'pretrain')
pretrain_model_path = None

if os.path.exists(pretrain_dir):
    # 查找最佳模型或最后周期模型
    best_model_path = os.path.join(pretrain_dir, 'best_md_model.params')
    last_model_path = os.path.join(pretrain_dir, f'epoch_{epochs - 1}.params')

    if os.path.exists(best_model_path):
        pretrain_model_path = best_model_path
        print(f"找到最佳预训练模型: {best_model_path}")
    elif os.path.exists(last_model_path):
        pretrain_model_path = last_model_path
        print(f"找到最后周期预训练模型: {last_model_path}")
    else:
        print(f"预训练目录 {pretrain_dir} 中没有找到模型文件")

if pretrain_model_path and os.path.exists(pretrain_model_path):
    print(f"加载预训练模型: {pretrain_model_path}")
    pretrain_state_dict = torch.load(pretrain_model_path, map_location=DEVICE)

    # 只加载MDAF中MD模块的权重（扩散去噪器：rec, hour, day）
    model_state_dict = net.state_dict()
    loaded_weights = []
    
    for name, param in pretrain_state_dict.items():
        # 只加载MD模块的权重（mdaf.rec.*/mdaf.hour.*/mdaf.day.*）
        # 不加载MAF模块权重（mdaf.attn_*/mdaf.fusion.*）
        if ('mdaf.rec.' in name or 'mdaf.hour.' in name or 'mdaf.day.' in name):
            if name in model_state_dict:
                model_state_dict[name] = param
                loaded_weights.append(name)

    net.load_state_dict(model_state_dict)
    print(f"成功加载 {len(loaded_weights)} 个MD模块权重")
else:
    print("警告: 未找到预训练模型，将从头开始训练（包括MD模块）")

# ---------------------- 损失函数 ----------------------
if loss_function.lower() == 'masked_mae':
    criterion = masked_mae
elif loss_function.lower() == 'masked_mse':
    criterion = masked_mse
elif loss_function.lower() == 'huber_loss':
    criterion = masked_huber_loss
else:
    criterion = nn.MSELoss().to(DEVICE)

# ---------------------- 优化器（训练非MD模块） ----------------------
# 根据论文Algorithm 1，主训练阶段：
# 论文符号定义：
# - Rec (X_Rec): 最近连续时间，由num_of_hours确定，由mdaf.rec去噪器处理
# - Hour (X_Hour): 小时周期，由num_of_days确定，由mdaf.hour去噪器处理
# - Day (X_Day): 日周期，由num_of_weeks确定，由mdaf.day去噪器处理
#
# - 冻结MD模块参数：mdaf.rec.*/mdaf.hour.*/mdaf.day.*
# - 训练MAF模块参数：mdaf.attn_*/mdaf.fusion.*
# - 训练MGRC模块参数：mgrc.*
# - 训练STFormer模块参数：stformer.*
# - 训练预测器参数：predictor.*

print("\n配置主训练参数...")
trainable_params = []
frozen_params = []
maf_params = []
mgrc_params = []
stformer_params = []

for name, param in net.named_parameters():
    # 冻结MD模块的扩散去噪器参数（θ_MD）
    if ('mdaf.rec.' in name or 'mdaf.hour.' in name or 'mdaf.day.' in name):
        param.requires_grad = False
        frozen_params.append(param)
        print(f"  冻结: {name}")
    else:
        # 训练其他所有模块参数（θ_MAF, θ_MGRC, θ_ST, θ_TT）
        param.requires_grad = True
        trainable_params.append(param)
        
        # 分类统计
        if 'mdaf.' in name:
            maf_params.append(param)
            print(f"  训练(MAF): {name}")
        elif 'mgrc.' in name:
            mgrc_params.append(param)
            print(f"  训练(MGRC): {name}")
        elif 'stformer.' in name:
            stformer_params.append(param)

print(f"\n参数统计:")
print(f"  冻结的MD模块（扩散去噪器）: {sum(p.numel() for p in frozen_params):,}")
print(f"  可训练的MAF模块参数: {sum(p.numel() for p in maf_params):,}")
print(f"  可训练的MGRC模块参数: {sum(p.numel() for p in mgrc_params):,}")
print(f"  可训练的STFormer模块参数: {sum(p.numel() for p in stformer_params):,}")
print(f"  可训练参数总计: {sum(p.numel() for p in trainable_params):,}")

optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

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
print("开始MD-GRTN主训练")
print("=" * 50)

best_val_loss = np.inf
best_epoch = 0
global_step = 0
start_time = time()

for epoch in range(start_epoch, epochs):
    net.train()
    total_loss = 0
    batch_count = 0

    for batch_index, batch_data in enumerate(train_loader):
        # 主训练模式返回4个数据:
        # (x_rec, x_hour, x_day, labels)
        # 根据论文符号定义：
        # - x_rec   = X_Rec  (最近连续时间，num_of_hours步)
        # - x_hour  = X_Hour (小时周期，num_of_days步，24小时模式)
        # - x_day   = X_Day  (日周期，num_of_weeks步，7天模式)
        # - labels  = Y (预测标签)
        if len(batch_data) != 4:
            print(f"警告: 批次数据长度应为4，实际为{len(batch_data)}")
            continue

        x_rec, x_hour, x_day, labels = batch_data

        optimizer.zero_grad()

        # MD-GRTN前向传播（根据论文Algorithm 1 Line 12-14）
        # 输入：三个周期的带噪声交通流 X_RecN, X_HourN, X_DayN
        # MDAF模块内部会：
        #   1. 使用预训练的MD模块对带噪声数据进行去噪
        #   2. 使用训练中的MAF模块对去噪后的特征进行融合
        # 输出：未来T_out个时间步的预测 (B, N, T_out)
        try:
            outputs = net(x_rec, x_hour, x_day)
        except Exception as e:
            import traceback
            print(f"\n前向传播错误（批次 {batch_index}): {e}")
            print(f"  x_rec (X_Rec).shape: {x_rec.shape if x_rec is not None else 'None'}")
            print(f"  x_hour (X_Hour).shape: {x_hour.shape if x_hour is not None else 'None'}")
            print(f"  x_day (X_Day).shape: {x_day.shape if x_day is not None else 'None'}")
            print("完整错误堆栈:")
            traceback.print_exc()
            continue

        # 计算损失（使用Huber损失或指定的损失函数）
        if isinstance(criterion, nn.Module):
            loss = criterion(outputs, labels)
        else:
            loss = criterion(outputs, labels, missing_value)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1
        global_step += 1

        if global_step % 50 == 0:
            print(f"周期 {epoch}, 步骤 {global_step}, 训练损失: {loss.item():.6f}")
            if sw:
                sw.add_scalar('training_loss', loss.item(), global_step)

    if batch_count > 0:
        avg_loss = total_loss / batch_count
        print(f"周期 {epoch} 完成. 平均训练损失: {avg_loss:.6f}")
    else:
        avg_loss = 0
        print(f"周期 {epoch} 没有有效批次数据")

    # ---------------------- 验证集评估 ----------------------
    if val_loader is not None:
        net.eval()
        with torch.no_grad():
            val_loss = compute_val_loss(
                net, val_loader, criterion,
                masked_flag=False, missing_value=missing_value,
                sw=sw, epoch=epoch, limit=None
            )
            print(f"周期 {epoch} 验证损失: {val_loss:.6f}")

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_params_filename = os.path.join(params_path, 'best_model.params')
            torch.save(net.state_dict(), best_params_filename)
            print(f"保存最佳模型到 {best_params_filename}")

    # 保存当前周期模型
    params_filename = os.path.join(params_path, f'epoch_{epoch}.params')
    torch.save(net.state_dict(), params_filename)
    print(f'保存模型参数到 {params_filename}')

# ---------------------- 测试集评估 ----------------------
print("\n" + "=" * 50)
print("使用最佳模型评估测试集...")
print("=" * 50)

if test_loader is not None:
    # 加载最佳模型
    best_params_filename = os.path.join(params_path, 'best_model.params')
    if os.path.exists(best_params_filename):
        net.load_state_dict(torch.load(best_params_filename))
        print(f"加载最佳模型: {best_params_filename}")
    else:
        # 如果没有最佳模型，使用最后一个周期的模型
        params_filename = os.path.join(params_path, f'epoch_{epochs - 1}.params')
        net.load_state_dict(torch.load(params_filename))
        print(f"加载最后周期模型: {params_filename}")

    # 测试集评估
    results = predict_and_save_results(
        net, test_loader, test_target_tensor, best_epoch,
        metric_method=metric_method, params_path=params_path, type='test'
    )

    if results:
        print("\n测试集最终评估结果:")
        print(f"MAE: {results[-3]:.4f}")
        print(f"RMSE: {results[-2]:.4f}")
        print(f"MAPE: {results[-1]:.4f}%")

        # 保存结果到文件
        result_file = os.path.join(params_path, 'final_results.txt')
        with open(result_file, 'w') as f:
            f.write(f"最佳周期: {best_epoch}\n")
            f.write(f"最佳验证损失: {best_val_loss:.6f}\n")
            f.write(f"测试集MAE: {results[-3]:.4f}\n")
            f.write(f"测试集RMSE: {results[-2]:.4f}\n")
            f.write(f"测试集MAPE: {results[-1]:.4f}%\n")
        print(f"结果保存到: {result_file}")

print("MD-GRTN主训练完成.")
print(f"总训练时间: {(time() - start_time) / 60:.2f} 分钟")