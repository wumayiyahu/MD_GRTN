import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import eigs


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    获取邻接矩阵

    Parameters
    ----------
    distance_df_filename: str, 包含边信息的csv文件路径
    num_of_vertices: int, 节点数量
    id_filename: str, 节点ID映射文件（可选）

    Returns
    ----------
    A: np.ndarray, 邻接矩阵
    distaneA: np.ndarray, 距离矩阵
    '''
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, adj_mx  # 如果是npy文件，同时作为邻接矩阵和距离矩阵
    else:
        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

        # 通用CSV处理（支持 from,to,cost 格式）
        with open(distance_df_filename, 'r') as f:
            f.readline()  # 跳过标题行
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                # 尝试将列值转换为数字
                try:
                    i = int(float(row[0]))  # 处理可能是浮点数或字符串的情况
                    j = int(float(row[1]))
                    distance = float(row[2])
                    
                    # 检查索引是否有效（严格大于等于0，小于num_of_vertices）
                    if i < 0 or i >= num_of_vertices or j < 0 or j >= num_of_vertices:
                        continue
                    
                    # 设置邻接矩阵和距离矩阵
                    A[i, j] = 1
                    A[j, i] = 1  # 无向图
                    distaneA[i, j] = distance
                    distaneA[j, i] = distance  # 无向图
                except (ValueError, IndexError) as e:
                    continue
        
        return A, distaneA


def load_md_grtn_data(graph_signal_matrix_filename, num_of_recs, num_of_hours, num_of_days, num_for_predict, DEVICE, batch_size,
                      shuffle=True, mode='train'):
    '''
    为MD-GRTN模型准备数据
    返回三个独立周期的数据和对应的带噪声数据
    Returns
    ----------
    train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, target_mean, target_std
    '''

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
    dirpath = os.path.dirname(graph_signal_matrix_filename)
    
    # MD-GRTN专用数据文件（包含预测步数num_for_predict参数）
    # 注意：文件名格式从 prepareData.py 改为包含 _p{num_for_predict}
    filename = os.path.join(dirpath, file + '_md_grtn' +
                            '_w' + str(num_of_days) +
                            '_d' + str(num_of_hours) +
                            '_h' + str(num_of_recs) +
                            '_p' + str(num_for_predict) + '.npz')
    
    print('加载MD-GRTN数据文件:', filename)
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"MD-GRTN数据文件不存在: {filename}. 请先运行数据预处理脚本。")

    file_data = np.load(filename)
    
    # 加载标准化参数（用于评估时反归一化）
    target_mean = float(file_data['target_mean'])
    target_std = float(file_data['target_std'])
    print(f"加载标准化参数: mean={target_mean:.4f}, std={target_std:.4f}")

    # 根据模式加载数据
    if mode == 'pretrain':
        # 预训练：加载带噪声和无噪声数据
        print(f"\n[DEBUG] 加载预训练数据键: {list(file_data.keys())}")
        train_rec_noisy = file_data['train_rec_noisy']
        train_hour_noisy = file_data['train_hour_noisy']
        train_day_noisy = file_data['train_day_noisy']
        train_rec = file_data['train_rec']
        train_hour = file_data['train_hour']
        train_day = file_data['train_day']
        
        print(f"\n[DEBUG] 原始数据形状:")
        print(f"  train_rec_noisy (噪声): {train_rec_noisy.shape}")
        print(f"  train_hour_noisy (噪声): {train_hour_noisy.shape}")
        print(f"  train_day_noisy (噪声): {train_day_noisy.shape if train_day_noisy is not None else 'None'}")
        print(f"  train_rec (干净): {train_rec.shape}")
        print(f"  train_hour (干净): {train_hour.shape}")
        print(f"  train_day (干净): {train_day.shape if train_day is not None else 'None'}")

        # 转换为张量
        def to_tensor(data):
            if data is None:
                return None
            return torch.from_numpy(data).type(torch.FloatTensor).to(DEVICE)
 
        # 训练集
        train_rec_noisy_tensor = to_tensor(train_rec_noisy)
        train_hour_noisy_tensor = to_tensor(train_hour_noisy)
        train_day_noisy_tensor = to_tensor(train_day_noisy)
        train_rec_tensor = to_tensor(train_rec)
        train_hour_tensor = to_tensor(train_hour)
        train_day_tensor = to_tensor(train_day)

        # 创建预训练数据集
        class MDGRTNPretrainDataset(torch.utils.data.Dataset):
            """
            根据 Algorithm 1，预训练阶段：
            - 输入：Noisy traffic flow features [X_Rec, X_Hour, X_Day]
            - 监督：Noise-free traffic flow features [X̂_Rec, X̂_Hour, X̂_Day] 仅用于计算MSE损失
            """
            def __init__(self, rec_noisy, hour_noisy, day_noisy, rec_clean, hour_clean, day_clean):
                self.rec_noisy = rec_noisy  # (B, N, F, T) 带噪声 
                self.hour_noisy = hour_noisy  # (B, N, F, T)
                self.day_noisy = day_noisy    # (B, N, F, T) 
                # 干净数据：监督信号（必须与去噪输出维度匹配）
                self.rec_clean = rec_clean   # (B, N, F, T) 干净数据
                self.hour_clean = hour_clean # (B, N, F, T) 
                self.day_clean = day_clean   # (B, N, F, T) 

            def __len__(self):
                return len(self.rec_noisy)

            def __getitem__(self, idx):
                """
                模型期望输入: (N, F, T)，DataLoader会添加batch维度
                返回: (rec_noisy, hour_noisy, day_noisy, rec_clean, hour_clean, day_clean)
                """
                # 提取第一个特征（流量特征）
                rec_noisy = self.rec_noisy[idx][:, 0:1, :]  # (N, 1, T) - X_Rec (最近连续时间)
                hour_noisy = self.hour_noisy[idx][:, 0:1, :]  # (N, 1, T) - X_Hour (小时周期)
                day_noisy = self.day_noisy[idx][:, 0:1, :]    # (N, 1, T) - X_Day (日周期)
                
                # 提取对应的干净数据（监督信号）- 关键：从 clean 张量中获取
                # 确保索引一致，无论 DataLoader 是否打乱顺序
                rec_clean = self.rec_clean[idx][:, 0:1, :]   # (N, 1, T) - X̂_Rec (干净的原始数据)
                hour_clean = self.hour_clean[idx][:, 0:1, :] # (N, 1, T) - X̂_Hour (干净的原始数据)
                day_clean = self.day_clean[idx][:, 0:1, :]   # (N, 1, T) - X̂_Day (干净的原始数据)
                
                return rec_noisy, hour_noisy, day_noisy, rec_clean, hour_clean, day_clean
            
        # 创建预训练集：传入噪声数据（模型输入）和干净数据（监督信号）
        train_dataset = MDGRTNPretrainDataset(
            train_rec_noisy_tensor,  # 噪声数据：X_Rec 
            train_hour_noisy_tensor,   # 噪声数据：X_Hour 
            train_day_noisy_tensor,  # 噪声数据：X_Day 
            train_rec_tensor,        # 干净数据：X̂_Rec
            train_hour_tensor,         # 干净数据：X̂_Hour 
            train_day_tensor         # 干净数据：X̂_Day 
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        return train_loader, None, None, None, None, None, target_mean, target_std

    else:
        # 主训练/测试：根据论文Algorithm 1，所有阶段都使用带噪声数据作为输入
        # 论文从头到尾只有一个输入分布：noisy
        
        # 训练集输入：带噪声数据
        train_rec = file_data['train_rec_noisy']
        train_hour = file_data['train_hour_noisy']
        train_day = file_data['train_day_noisy'] 
        train_target = file_data['train_target']
        
        # 验证集输入：带噪声数据
        val_rec = file_data['val_rec_noisy'] 
        val_hour = file_data['val_hour_noisy'] 
        val_day = file_data['val_day_noisy']
        val_target = file_data['val_target']
        
        # 测试集输入：带噪声数据
        test_rec = file_data['test_rec_noisy'] 
        test_hour = file_data['test_hour_noisy'] 
        test_day = file_data['test_day_noisy'] 
        test_target = file_data['test_target']

        # 转换为张量
        def to_tensor(data):
            if data is None:
                return None
            return torch.from_numpy(data).type(torch.FloatTensor).to(DEVICE)

        # 训练集
        train_rec_tensor = to_tensor(train_rec)
        train_hour_tensor = to_tensor(train_hour)
        train_day_tensor = to_tensor(train_day)
        train_target_tensor = to_tensor(train_target)

        # 验证集
        val_rec_tensor = to_tensor(val_rec)
        val_hour_tensor = to_tensor(val_hour)
        val_day_tensor = to_tensor(val_day)
        val_target_tensor = to_tensor(val_target)

        # 测试集
        test_rec_tensor = to_tensor(test_rec)
        test_hour_tensor = to_tensor(test_hour)
        test_day_tensor = to_tensor(test_day)
        test_target_tensor = to_tensor(test_target)

        # 创建主训练数据集
        class MDGRTNTrainDataset(torch.utils.data.Dataset):
            def __init__(self, rec, hour, day, target):
                """
                参数:
                    rec: 近期序列 (B, N, F, T) - X_Rec，
                    hour: 小时周期序列 (B, N, F, T) - X_Hour
                    day: 日周期序列 (B, N, F, T) - X_Day
                    target: 目标数据 (B, N, T_out)
                """
                self.rec = rec  # X_Rec: 最近连续时间
                self.hour = hour  # X_Hour: 小时周期(24小时模式)
                self.day = day  # X_Day: 日周期(7天模式)
                self.target = target

            def __len__(self):
                return len(self.target)

            def __getitem__(self, idx):
                """
                返回单个样本的数据
                模型期望输入: (B, N, F, T)，DataLoader会添加batch维度
                
                返回: (x_rec, x_hour, x_day, labels)
                - x_rec: X_Rec (最近连续时间)
                - x_hour: X_Hour (小时周期)
                - x_day: X_Day (日周期)
                """
                # 处理可能为None的日数据
                if self.day is not None:
                    day_data = self.day[idx]  # (N, F, T) - X_Day
                else:
                    # 创建零张量，使用rec的形状
                    day_data = torch.zeros_like(self.rec[idx][:, :, :1])  # 只取1个时间步，避免浪费内存
                
                # 数据形状已经是 (N, 1, T)，F=1（流量特征）
                rec_data = self.rec[idx][:, 0:1, :]   # (N, 1, T) - X_Rec
                hour_data = self.hour[idx][:, 0:1, :]   # (N, 1, T) - X_Hour
                day_data = day_data[:, 0:1, :]          # (N, 1, T) - X_Day
                
                return rec_data, hour_data, day_data, self.target[idx]

        # 创建数据加载器
        train_dataset = MDGRTNTrainDataset(
            train_rec_tensor,  # X_Rec (最近连续时间)
            train_hour_tensor,   # X_Hour (小时周期)
            train_day_tensor,  # X_Day (日周期)
            train_target_tensor
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = MDGRTNTrainDataset(
            val_rec_tensor,   # X_Rec
            val_hour_tensor,    # X_Hour
            val_day_tensor,   # X_Day
            val_target_tensor
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = MDGRTNTrainDataset(
            test_rec_tensor,   # X_Rec
            test_hour_tensor,    # X_Hour
            test_day_tensor,   # X_Day
            test_target_tensor
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"训练集 - X_Rec(num_of_recs): {train_rec_tensor.size()}, X_Hour(num_of_hours): {train_hour_tensor.size()}, X_Day(num_of_days): {train_day_tensor.size() if train_day_tensor is not None else 'None'}, 目标: {train_target_tensor.size()}")
        print(f"验证集 - X_Rec: {val_rec_tensor.size()}, X_Hour: {val_hour_tensor.size()}, X_Day: {val_day_tensor.size() if val_day_tensor is not None else 'None'}, 目标: {val_target_tensor.size()}")
        print(f"测试集 - X_Rec: {test_rec_tensor.size()}, X_Hour: {test_hour_tensor.size()}, X_Day: {test_day_tensor.size() if test_day_tensor is not None else 'None'}, 目标: {test_target_tensor.size()}")

        return (train_loader, train_target_tensor,
                val_loader, val_target_tensor,
                test_loader, test_target_tensor,
                target_mean, target_std)


def compute_val_loss_md_grtn(net, val_loader, criterion, sw, epoch, limit=None):
    '''
    为MD-GRTN模型计算验证损失

    Parameters
    ----------
    net: MD-GRTN模型
    val_loader: torch.utils.data.DataLoader
    criterion: 损失函数
    masked_flag: bool, 是否使用masked损失
    missing_value: float, 缺失值
    sw: tensorboardX.SummaryWriter
    epoch: int, 当前周期
    limit: int, 限制批次数量

    Returns
    ----------
    validation_loss: float, 验证损失
    '''
    net.eval()

    with torch.no_grad():
        val_loader_length = len(val_loader)
        tmp = []  # 记录所有batch的loss

        for batch_index, batch_data in enumerate(val_loader):
            # MD-GRTN: batch_data包含三个输入和一个目标 (x_rec, x_hour, x_day, labels)
            if len(batch_data) != 4:
                print(f"错误: MD-GRTN期望4个数据，但得到{len(batch_data)}个")
                continue

            x_rec, x_hour, x_day, labels = batch_data

            # 调用MD-GRTN模型（需要三个输入）
            try:
                outputs = net(x_rec, x_hour, x_day)
            except Exception as e:
                print(f"前向传播错误: {e}")
                continue

            # 计算损失
            loss = criterion(outputs, labels)

            tmp.append(loss.item())

            if batch_index % 100 == 0:
                print(f'验证批次 {batch_index + 1} / {val_loader_length}, 损失: {loss.item():.4f}')

            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp) if tmp else 0

        if sw is not None:
            sw.add_scalar('validation_loss', validation_loss, epoch)

    return validation_loss


def to_numpy(x):
    """
    辅助函数：将 PyTorch tensor 转换为 NumPy 数组
    解决 CUDA tensor 无法直接保存的问题
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def predict_and_save_results_md_grtn(net, data_loader, global_step, params_path, type='test',
                                            target_mean=None, target_std=None, null_val=0.0):
    """
    为MD-GRTN模型预测并保存结果（使用 masked 指标）

    Parameters
    ----------
    net: MD-GRTN模型
    data_loader: torch.utils.data.DataLoader
    global_step: int, 当前全局步数
    params_path: str, 结果保存路径
    type: str, 'train', 'val', 或 'test'
    target_mean: float, 标准化均值（用于反归一化）
    target_std: float, 标准化标准差（用于反归一化）
    null_val: float, 计算指标时忽略的值（如0或nan）
    
    Returns
    ----------
    excel_list: list, 包含所有评估指标的列表
    """
    net.eval()
    need_inverse = target_mean is not None and target_std is not None
    if need_inverse:
        print(f"\n✅ 使用反标准化计算指标: mean={target_mean:.4f}, std={target_std:.4f}")
    else:
        print("\n评估模式：使用原始数据空间计算指标")

    prediction_list, target_list = [], []

    with torch.no_grad():
        for batch_index, batch_data in enumerate(data_loader):
            if len(batch_data) != 4:
                print(f"错误: 期望4个数据，但得到{len(batch_data)}")
                continue
            x_rec, x_hour, x_day, labels = batch_data

            try:
                outputs = net(x_rec, x_hour, x_day)
            except Exception as e:
                print(f"前向传播错误: {e}")
                continue

            prediction_list.append(outputs.detach().cpu().numpy())
            target_list.append(labels.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print(f'预测数据集批次 {batch_index + 1} / {len(data_loader)}')

    # 合并所有 batch
    predictions = np.concatenate(prediction_list, 0)
    targets = np.concatenate(target_list, 0)

    # 反标准化
    if need_inverse:
        predictions = predictions * target_std + target_mean
        targets = targets * target_std + target_mean

    # 保存结果
    output_filename = os.path.join(params_path, f'output_epoch_{global_step}_{type}')
    np.savez(output_filename, prediction=predictions, data_target_tensor=targets)
    print(f'结果已保存到: {output_filename}')

    # 逐时间步计算 masked 指标
    excel_list = []
    T_out = predictions.shape[2]

    for i in range(T_out):
        gt = targets[:, :, i]
        pred = predictions[:, :, i]

        # ------------------ masked 指标 ------------------
        # mask：忽略 null_val 或 nan
        if np.isnan(null_val):
            mask = ~np.isnan(gt)
        else:
            mask = gt != null_val
        mask = mask.astype(float)
        mask /= np.mean(mask)  # normalize
        mask = np.nan_to_num(mask)

        mae = np.nan_to_num(mask * np.abs(pred - gt))
        rmse = np.nan_to_num(mask * (pred - gt)**2)
        mape = np.nan_to_num(mask * np.abs((pred - gt) / (gt + 1e-5)))

        mae_val = np.mean(mae)
        rmse_val = np.sqrt(np.mean(rmse))
        mape_val = np.mean(mape)

        print(f'当前周期: {global_step}, 预测第 {i} 个时间点')
        print('MAE: %.2f, RMSE: %.2f, MAPE: %.2f%%' % (mae_val, rmse_val, mape_val * 100))

        excel_list.extend([mae_val, rmse_val, mape_val])

    return excel_list