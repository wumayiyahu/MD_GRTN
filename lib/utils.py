import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import eigs
from .metrics import masked_mape_np, masked_mae, masked_mse, masked_rmse, masked_mae_test, masked_rmse_test


def re_normalization(x, mean, std):
    """反归一化"""
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    """最大最小归一化"""
    x = 1. * (x - _min) / (_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    """反最大最小归一化"""
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


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
        return adj_mx, None
    else:
        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

        if id_filename:
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA
        else:
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def scaled_Laplacian(W):
    '''
    计算缩放拉普拉斯矩阵

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N是节点数量

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    计算Chebyshev多项式

    Parameters
    ----------
    L_tilde: 缩放拉普拉斯矩阵, np.ndarray, shape (N, N)
    K: Chebyshev多项式的最大阶数

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), 长度: K, 从T_0到T_{K-1}
    '''
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def load_md_grtn_data(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size,
                      shuffle=True, mode='train'):
    '''
    为MD-GRTN模型准备数据
    返回三个独立周期的数据和对应的带噪声数据

    Parameters
    ----------
    graph_signal_matrix_filename: str, 原始数据文件路径
    num_of_hours: int, 小时周期数
    num_of_days: int, 日周期数
    num_of_weeks: int, 周周期数
    DEVICE: torch.device
    batch_size: int
    shuffle: bool
    mode: str, 'pretrain' 或 'train' 或 'test'

    Returns
    ----------
    train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std
    '''

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
    dirpath = os.path.dirname(graph_signal_matrix_filename)

    # MD-GRTN专用数据文件
    filename = os.path.join(dirpath, file + '_md_grtn' +
                            '_w' + str(num_of_weeks) +
                            '_d' + str(num_of_days) +
                            '_h' + str(num_of_hours) + '.npz')

    print('加载MD-GRTN数据文件:', filename)

    if not os.path.exists(filename):
        raise FileNotFoundError(f"MD-GRTN数据文件不存在: {filename}. 请先运行数据预处理脚本。")

    file_data = np.load(filename)

    # 根据模式加载数据
    if mode == 'pretrain':
        # 预训练：加载带噪声和无噪声数据
        train_hour_noisy = file_data['train_hour_noisy']
        train_day_noisy = file_data['train_day_noisy']
        train_week_noisy = file_data['train_week_noisy'] if 'train_week_noisy' in file_data else None
        train_hour = file_data['train_hour']
        train_day = file_data['train_day']
        train_week = file_data['train_week'] if 'train_week' in file_data else None

        # 转换为张量
        def to_tensor(data):
            if data is None:
                return None
            return torch.from_numpy(data).type(torch.FloatTensor).to(DEVICE)

        # 训练集
        train_hour_noisy_tensor = to_tensor(train_hour_noisy)
        train_day_noisy_tensor = to_tensor(train_day_noisy)
        train_week_noisy_tensor = to_tensor(train_week_noisy)
        train_hour_tensor = to_tensor(train_hour)
        train_day_tensor = to_tensor(train_day)
        train_week_tensor = to_tensor(train_week)

        # 创建预训练数据集
        class MDGRTNPretrainDataset(torch.utils.data.Dataset):
            def __init__(self, hour_noisy, day_noisy, week_noisy, hour_clean, day_clean, week_clean):
                self.hour_noisy = hour_noisy  # (B, N, F, T) 带噪声
                self.day_noisy = day_noisy    # (B, N, F, T)
                self.week_noisy = week_noisy  # (B, N, F, T)
                self.hour_clean = hour_clean  # (B, N, F, T) 干净数据
                self.day_clean = day_clean    # (B, N, F, T)
                self.week_clean = week_clean  # (B, N, F, T)

            def __len__(self):
                return len(self.hour_noisy) if self.hour_noisy is not None else 0

            def __getitem__(self, idx):
                """
                返回预训练所需的6个数据
                模型期望输入: (N, F, T)，DataLoader会添加batch维度
                """
                # 提取第一个特征（流量特征）
                rec_noisy = self.hour_noisy[idx][:, :, 0:1, :]  # (N, 1, T)
                hour_noisy = self.day_noisy[idx][:, :, 0:1, :]  # (N, 1, T)
                day_noisy = self.week_noisy[idx][:, :, 0:1, :]   # (N, 1, T)
                
                rec_clean = self.hour_clean[idx][:, :, 0:1, :]   # (N, 1, T)
                hour_clean = self.day_clean[idx][:, :, 0:1, :]   # (N, 1, T)
                day_clean = self.week_clean[idx][:, :, 0:1, :]   # (N, 1, T)
                
                return rec_noisy, hour_noisy, day_noisy, rec_clean, hour_clean, day_clean

        train_dataset = MDGRTNPretrainDataset(
            train_hour_noisy_tensor, train_day_noisy_tensor, train_week_noisy_tensor,
            train_hour_tensor, train_day_tensor, train_week_tensor
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        return train_loader, None, None, None, None, None, None, None

    else:
        # 主训练/测试：根据论文Algorithm 1，主训练阶段使用带噪声数据作为输入
        # 训练集：使用带噪声数据作为输入（与预训练一致）
        # 验证集和测试集：使用干净数据（评估模型在干净输入下的性能）
        
        # 训练集输入：带噪声数据
        train_hour = file_data['train_hour_noisy']
        train_day = file_data['train_day_noisy']
        train_week = file_data['train_week_noisy'] if 'train_week_noisy' in file_data else None
        train_target = file_data['train_target']
        
        # 验证集输入：干净数据（不添加噪声）
        val_hour = file_data['val_hour']
        val_day = file_data['val_day']
        val_week = file_data['val_week'] if 'val_week' in file_data else None
        val_target = file_data['val_target']
        
        # 测试集输入：干净数据（不添加噪声）
        test_hour = file_data['test_hour']
        test_day = file_data['test_day']
        test_week = file_data['test_week'] if 'test_week' in file_data else None
        test_target = file_data['test_target']

        # 转换为张量
        def to_tensor(data):
            if data is None:
                return None
            return torch.from_numpy(data).type(torch.FloatTensor).to(DEVICE)

        # 训练集
        train_hour_tensor = to_tensor(train_hour)
        train_day_tensor = to_tensor(train_day)
        train_week_tensor = to_tensor(train_week)
        train_target_tensor = to_tensor(train_target)

        # 验证集
        val_hour_tensor = to_tensor(val_hour)
        val_day_tensor = to_tensor(val_day)
        val_week_tensor = to_tensor(val_week)
        val_target_tensor = to_tensor(val_target)

        # 测试集
        test_hour_tensor = to_tensor(test_hour)
        test_day_tensor = to_tensor(test_day)
        test_week_tensor = to_tensor(test_week)
        test_target_tensor = to_tensor(test_target)

        # 创建主训练数据集
        class MDGRTNTrainDataset(torch.utils.data.Dataset):
            def __init__(self, hour, day, week, target):
                self.hour = hour  # 近期序列 (B, N, F, T)
                self.day = day    # 日周期序列 (B, N, F, T)
                self.week = week  # 周周期序列 (B, N, F, T)，可能为None
                self.target = target  # 目标数据 (B, N, T_out)

            def __len__(self):
                return len(self.target)

            def __getitem__(self, idx):
                """
                返回单个样本的数据
                模型期望输入: (B, N, F, T)，DataLoader会添加batch维度
                """
                # 处理可能为None的周数据
                if self.week is not None:
                    week_data = self.week[idx]  # (N, F, T)
                else:
                    # 创建零张量，使用hour的形状
                    week_data = torch.zeros_like(self.hour[idx][:, :, :1])  # 只取1个时间步，避免浪费内存
                
                # 移除F维度中多余的特征（只使用第一个特征）
                # 数据形状从 (N, F, T) 中的 F=1（流量特征）
                hour_data = self.hour[idx][:, :, 0:1, :]   # (N, 1, T)
                day_data = self.day[idx][:, :, 0:1, :]      # (N, 1, T)
                week_data = week_data[:, :, 0:1, :]          # (N, 1, T)
                
                return hour_data, day_data, week_data, self.target[idx]

        # 创建数据加载器
        train_dataset = MDGRTNTrainDataset(train_hour_tensor, train_day_tensor, train_week_tensor, train_target_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        val_dataset = MDGRTNTrainDataset(val_hour_tensor, val_day_tensor, val_week_tensor, val_target_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = MDGRTNTrainDataset(test_hour_tensor, test_day_tensor, test_week_tensor, test_target_tensor)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"训练集 - 小时序列(带噪声): {train_hour_tensor.size()}, 目标: {train_target_tensor.size()}")
        print(f"验证集 - 小时序列(干净): {val_hour_tensor.size()}, 目标: {val_target_tensor.size()}")
        print(f"测试集 - 小时序列(干净): {test_hour_tensor.size()}, 目标: {test_target_tensor.size()}")

        return (train_loader, train_target_tensor,
                val_loader, val_target_tensor,
                test_loader, test_target_tensor,
                None, None)  # MD-GRTN每个周期单独归一化，不返回全局mean/std


def compute_val_loss_md_grtn(net, val_loader, criterion, masked_flag, missing_value, sw, epoch, limit=None):
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
    limit: int, 限制批次数量（可选）

    Returns
    ----------
    validation_loss: float, 验证损失
    '''
    net.eval()

    with torch.no_grad():
        val_loader_length = len(val_loader)
        tmp = []  # 记录所有batch的loss

        for batch_index, batch_data in enumerate(val_loader):
            # MD-GRTN: batch_data包含三个输入和一个目标 (x_hour, x_day, x_week, labels)
            if len(batch_data) != 4:
                print(f"错误: MD-GRTN期望4个数据，但得到{len(batch_data)}个")
                continue

            x_hour, x_day, x_week, labels = batch_data

            # 调用MD-GRTN模型（需要三个输入）
            try:
                outputs = net(x_hour, x_day, x_week)
            except Exception as e:
                print(f"前向传播错误: {e}")
                continue

            # 计算损失
            if masked_flag:
                loss = criterion(outputs, labels, missing_value)
            else:
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


def predict_and_save_results_md_grtn(net, data_loader, data_target_tensor, global_step, metric_method,
                                     params_path, type='test'):
    '''
    为MD-GRTN模型预测并保存结果

    Parameters
    ----------
    net: MD-GRTN模型
    data_loader: torch.utils.data.DataLoader
    data_target_tensor: torch.Tensor, 目标数据
    global_step: int, 当前全局步数
    metric_method: str, 'mask' 或 'unmask'
    params_path: str, 结果保存路径
    type: str, 'train', 'val', 或 'test'

    Returns
    ----------
    excel_list: list, 包含所有评估指标的列表
    '''
    net.eval()

    with torch.no_grad():
        data_target_tensor = data_target_tensor.cpu().numpy()
        loader_length = len(data_loader)

        prediction = []
        input_hour_list, input_day_list, input_week_list = [], [], []

        for batch_index, batch_data in enumerate(data_loader):
            # MD-GRTN: batch_data包含三个输入和一个目标
            if len(batch_data) != 4:
                print(f"错误: MD-GRTN期望4个数据，但得到{len(batch_data)}个")
                continue

            x_hour, x_day, x_week, labels = batch_data

            # 保存输入数据用于分析
            input_hour_list.append(x_hour[:, :, 0:1].cpu().numpy())
            input_day_list.append(x_day[:, :, 0:1].cpu().numpy())
            input_week_list.append(x_week[:, :, 0:1].cpu().numpy())

            # 前向传播
            try:
                outputs = net(x_hour, x_day, x_week)
            except Exception as e:
                print(f"前向传播错误: {e}")
                continue

            prediction.append(outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print(f'预测数据集批次 {batch_index + 1} / {loader_length}')

        # 合并结果
        if input_hour_list:
            input_hour = np.concatenate(input_hour_list, 0)
            input_day = np.concatenate(input_day_list, 0)
            input_week = np.concatenate(input_week_list, 0)
        else:
            input_hour = input_day = input_week = None

        if prediction:
            prediction = np.concatenate(prediction, 0)
        else:
            prediction = None

        print(f'输入小时序列: {input_hour.shape if input_hour is not None else "None"}')
        print(f'输入日序列: {input_day.shape if input_day is not None else "None"}')
        print(f'输入周序列: {input_week.shape if input_week is not None else "None"}')
        print(f'预测结果: {prediction.shape if prediction is not None else "None"}')
        print(f'目标数据: {data_target_tensor.shape}')

        # 保存结果
        output_filename = os.path.join(params_path, f'output_epoch_{global_step}_{type}')

        save_dict = {
            'prediction': prediction,
            'data_target_tensor': data_target_tensor
        }

        if input_hour is not None:
            save_dict['input_hour'] = input_hour
        if input_day is not None:
            save_dict['input_day'] = input_day
        if input_week is not None:
            save_dict['input_week'] = input_week

        np.savez(output_filename, **save_dict)
        print(f'结果已保存到: {output_filename}')

        # 计算评估指标
        if prediction is not None and data_target_tensor.shape[0] == prediction.shape[0]:
            excel_list = []
            prediction_length = prediction.shape[2]

            # 逐时间点计算指标
            for i in range(prediction_length):
                assert data_target_tensor.shape[0] == prediction.shape[0]
                print(f'当前周期: {global_step}, 预测第 {i} 个时间点')

                if metric_method == 'mask':
                    mae = masked_mae_test(data_target_tensor[:, :, i], prediction[:, :, i], 0.0)
                    rmse = masked_rmse_test(data_target_tensor[:, :, i], prediction[:, :, i], 0.0)
                    mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i], 0)
                else:
                    mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i])
                    rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
                    mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i], 0)

                print(f'MAE: {mae:.4f}')
                print(f'RMSE: {rmse:.4f}')
                print(f'MAPE: {mape:.4f}')
                excel_list.extend([mae, rmse, mape])

            # 整体结果
            if metric_method == 'mask':
                mae = masked_mae_test(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0.0)
                rmse = masked_rmse_test(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0.0)
                mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
            else:
                mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
                rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
                mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)

            print(f'整体 MAE: {mae:.4f}')
            print(f'整体 RMSE: {rmse:.4f}')
            print(f'整体 MAPE: {mape:.4f}')
            excel_list.extend([mae, rmse, mape])
            print(f'评估结果列表: {excel_list}')

            return excel_list
        else:
            print('错误: 预测结果与目标数据形状不匹配')
            return None


# MD-GRTN专用函数别名（保持接口一致性）
def compute_val_loss(net, val_loader, criterion, masked_flag, missing_value, sw, epoch, limit=None):
    '''MD-GRTN验证损失计算函数'''
    return compute_val_loss_md_grtn(net, val_loader, criterion, masked_flag, missing_value, sw, epoch, limit)


def predict_and_save_results(net, data_loader, data_target_tensor, global_step, metric_method, params_path,
                             type='test'):
    '''MD-GRTN预测和保存结果函数'''
    return predict_and_save_results_md_grtn(net, data_loader, data_target_tensor, global_step,
                                            metric_method, params_path, type)