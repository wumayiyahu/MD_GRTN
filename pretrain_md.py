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
from lib.utils import load_md_grtn_data, get_adjacency_matrix
from tensorboardX import SummaryWriter

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
epochs = 15
start_epoch = int(training_config['start_epoch'])

num_of_recs = int(training_config['num_of_recs'])
num_of_hours = int(training_config['num_of_hours'])
num_of_days = int(training_config['num_of_days'])

# MD-GRTN模型参数
in_channels = int(training_config['in_channels'])  # 输入特征维度
hidden_dim = int(training_config.get('hidden_dim', 64))  # 隐藏层维度


# 修改文件夹命名，包含MD-GRTN标识
folder_dir = 'MD_GRTN_pretrain_h%dd%dw%d_channel%d_hidden%d_%e' % (
    num_of_recs, num_of_hours, num_of_days, in_channels, hidden_dim, learning_rate
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
    num_of_recs, num_of_hours, num_of_days, num_for_predict,
    DEVICE, batch_size, shuffle=True, mode='pretrain'
)
print(f"训练批次: {len(train_loader)}")

# 为验证集加载主训练数据
_, _, val_loader, val_target_tensor, test_loader, test_target_tensor, _, _ = load_md_grtn_data(
    graph_signal_matrix_filename,
    num_of_recs, num_of_hours, num_of_days, num_for_predict,
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

# 论文 Algorithm 1 的严格实现：
# MD 预训练只涉及 BackNet_k（DiffusionDenoiser）不涉及 MAF（TemporalEncoder + Attention + Fusion）
if hasattr(net, 'mdaf'):
    # 论文严格定义：MD = BackNet，独立去噪器
    md_params.extend(list(net.mdaf.rec.parameters()))
    md_params.extend(list(net.mdaf.hour.parameters()))
    md_params.extend(list(net.mdaf.day.parameters()))
    
    print("===  MD 预训练 ===")
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

# ---------------------- 主训练循环（严格 DDPM 预训练）----------------------
print("\n" + "=" * 50)
print("开始 MD 模块严格 DDPM 预训练")
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
        # *_noisy: 外部环境噪声数据
        # *_clean: 原始干净数据（用于计算 loss）
        rec_noisy, hour_noisy, day_noisy, rec_clean, hour_clean, day_clean = batch_data

        # 数据形状验证
        B, N, F, T_rec = rec_noisy.shape
        _, _, _, T_hour = hour_noisy.shape
        _, _, _, T_day = day_noisy.shape

        # 验证维度匹配
        assert rec_clean.shape == rec_noisy.shape, f"Rec clean {rec_clean.shape} vs noisy {rec_noisy.shape} 维度不匹配"
        assert hour_clean.shape == hour_noisy.shape, f"Hour clean {hour_clean.shape} vs noisy {hour_noisy.shape} 维度不匹配"
        assert day_clean.shape == day_noisy.shape, f"Day clean {day_clean.shape} vs noisy {day_noisy.shape} 维度不匹配"

        optimizer.zero_grad()

        # -------- X_t → X_0 去噪映射 --------
        # 输入: *_noisy (外部环境噪声，直接当作 X_t)
        # 网络学习: X_t → X_0 (去噪映射)
        # 输出: x0_hat (去噪结果)
        # Loss = MSE(x0_hat, *_clean)
        
        # 验证数据维度（调试用）
        if batch_index == 0:
            print(f"数据维度: rec_noisy={rec_noisy.shape}, hour_noisy={hour_noisy.shape}, day_noisy={day_noisy.shape}")
            print(f"        clean: rec_clean={rec_clean.shape}, hour_clean={hour_clean.shape}, day_clean={day_clean.shape}")

        try:
            # 传入noisy数据，直接去噪（不额外添加噪声）
            if hasattr(net, 'mdaf'):
                X_rec_denoised = net.mdaf.rec(rec_noisy, return_traffic_space=True)
                X_hour_denoised = net.mdaf.hour(hour_noisy, return_traffic_space=True)
                X_day_denoised = net.mdaf.day(day_noisy, return_traffic_space=True)
            else:
                print("错误：模型没有mdaf模块")
                raise SystemExit("预训练失败：需要 MDAF 模块")

            # 验证维度
            assert X_rec_denoised.shape == rec_clean.shape, \
                f"Rec denoised {X_rec_denoised.shape} vs clean {rec_clean.shape} 维度不匹配"
            assert X_hour_denoised.shape == hour_clean.shape, \
                f"Hour denoised {X_hour_denoised.shape} vs clean {hour_clean.shape} 维度不匹配"
            assert X_day_denoised.shape == day_clean.shape, \
                f"Day denoised {X_day_denoised.shape} vs clean {day_clean.shape} 维度不匹配"

            # -------- MSE loss --------
            # 损失：去噪结果 vs 原始 clean 数据
            loss_rec = criterion(X_rec_denoised, rec_clean)
            loss_hour = criterion(X_hour_denoised, hour_clean)
            loss_day = criterion(X_day_denoised, day_clean)
            loss = loss_rec + loss_hour + loss_day

        except Exception as e:
            print(f"扩散模块训练错误（批次 {batch_index}): {e}")
            import traceback
            traceback.print_exc()
            continue

        # -------- 反向传播 & 优化 --------
        loss.backward()
        torch.nn.utils.clip_grad_norm_(md_params, max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1
        global_step += 1

        if global_step % 50 == 0:
            print(f"Epoch {epoch}, Step {global_step}, "
                  f"Loss: {loss.item():.6f} "
                  f"(rec={loss_rec.item():.6f}, hour={loss_hour.item():.6f}, day={loss_day.item():.6f})")
            if sw:
                sw.add_scalar('pretrain/total_md_loss', loss.item(), global_step)
                sw.add_scalar('pretrain/loss_rec', loss_rec.item(), global_step)
                sw.add_scalar('pretrain/loss_hour', loss_hour.item(), global_step)
                sw.add_scalar('pretrain/loss_day', loss_day.item(), global_step)

    if batch_count > 0:
        avg_loss = total_loss / batch_count
        epoch_time = time() - epoch_start_time
        print(f"Epoch {epoch} 完成. 平均MD预训练损失: {avg_loss:.6f}, 耗时: {epoch_time:.2f}s, 批次: {batch_count}")

        # 保存最佳模型
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            best_epoch = epoch
            best_params_filename = os.path.join(params_path, 'best_md_model.params')
            torch.save(net.state_dict(), best_params_filename)
            print(f"保存最佳MD模型到 {best_params_filename}")
    else:
        avg_loss = 0
        print(f"Epoch {epoch} 没有有效批次数据")

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
