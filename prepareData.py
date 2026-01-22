import os
import numpy as np
import argparse
import configparser
import random


def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    搜索历史数据索引
    '''
    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    获取单个样本的多周期数据
    '''
    week_sample, day_sample, hour_sample = None, None, None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None
        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None
        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None
        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


def add_comprehensive_traffic_noise(clean_data, noise_config):
    '''
    添加综合交通噪声，模拟真实传感器数据问题
    
    Parameters:
    -----------
    clean_data: np.ndarray, 干净数据
    noise_config: dict, 噪声配置
    
    Returns:
    --------
    noisy_data: np.ndarray, 带噪声的数据
    '''
    noisy_data = clean_data.copy()
    n_samples, n_nodes, n_features, n_timesteps = noisy_data.shape
    
    # 1. 高斯噪声（传感器测量误差）
    if noise_config.get('gaussian_noise_level', 0) > 0:
        gaussian_noise = np.random.normal(
            0, 
            noise_config['gaussian_noise_level'], 
            noisy_data.shape
        )
        noisy_data += gaussian_noise
    
    # 2. 块状缺失（传感器短期故障）
    if noise_config.get('block_missing_rate', 0) > 0:
        block_size = noise_config.get('block_size', 6)  # 默认30分钟
        for sample_idx in range(n_samples):
            if random.random() < noise_config['block_missing_rate']:
                # 随机选择起始时间和传感器
                start_time = random.randint(0, n_timesteps - block_size)
                node_idx = random.randint(0, n_nodes - 1)
                
                end_time = min(start_time + block_size, n_timesteps)
                # 设置为0或NaN
                noisy_data[sample_idx, node_idx, :, start_time:end_time] = 0
    
    # 3. 随机点缺失（瞬时故障）
    if noise_config.get('point_missing_rate', 0) > 0:
        mask = np.random.rand(*noisy_data.shape) < noise_config['point_missing_rate']
        noisy_data[mask] = 0
    
    # 4. 异常值（交通事件：事故、施工等）
    if noise_config.get('outlier_rate', 0) > 0:
        outlier_mask = np.random.rand(*noisy_data.shape) < noise_config['outlier_rate']
        n_outliers = outlier_mask.sum()
        
        if n_outliers > 0:
            # 两种异常值：突然增加（70%）或突然减少（30%）
            outlier_type = np.random.rand(n_outliers) > 0.3  # 70%增加，30%减少
            
            # 异常值强度：增加1.5-3倍，减少到0.1-0.5倍
            increase_factors = np.random.uniform(1.5, 3.0, n_outliers)
            decrease_factors = np.random.uniform(0.1, 0.5, n_outliers)
            
            # 应用异常值
            outlier_values = noisy_data[outlier_mask].copy()
            for i in range(n_outliers):
                if outlier_type[i]:
                    outlier_values[i] *= increase_factors[i]  # 突然增加
                else:
                    outlier_values[i] *= decrease_factors[i]  # 突然减少
            
            noisy_data[outlier_mask] = outlier_values
    
    # 5. 平滑噪声（传感器漂移）
    if noise_config.get('drift_noise', False):
        # 添加缓慢变化的漂移
        time_axis = np.arange(n_timesteps) / n_timesteps
        for sample_idx in range(n_samples):
            for node_idx in range(n_nodes):
                drift = np.random.normal(0, 0.05) * time_axis
                noisy_data[sample_idx, node_idx, 0, :] += drift  # 只对流量特征添加漂移
    
    # 6. 保持数据范围合理（防止负值）
    noisy_data = np.maximum(noisy_data, 0)
    
    return noisy_data


def normalize_data(train_data, val_data, test_data):
    '''
    对数据进行Z-score归一化
    '''
    # 计算训练集的统计量
    mean = train_data.mean(axis=(0, 1, 3), keepdims=True)
    std = train_data.std(axis=(0, 1, 3), keepdims=True)
    
    # 避免除零
    std = np.where(std == 0, 1.0, std)
    
    def normalize(x):
        return (x - mean) / std
    
    train_norm = normalize(train_data)
    val_norm = normalize(val_data)
    test_norm = normalize(test_data)
    
    return {'mean': mean, 'std': std}, train_norm, val_norm, test_norm


def prepare_md_grtn_dataset(original_data_path, 
                           num_of_weeks=1, num_of_days=1, num_of_hours=3,
                           num_for_predict=12, points_per_hour=12,
                           noise_config=None, save_path=None):
    '''
    为MD-GRTN准备数据集
    
    Parameters:
    -----------
    original_data_path: str, 原始数据文件路径（.npz格式）
    num_of_weeks/days/hours: int, 各周期历史长度
    num_for_predict: int, 预测步长
    points_per_hour: int, 每小时数据点数
    noise_config: dict, 噪声配置
    save_path: str, 保存路径
    
    Returns:
    --------
    dataset_dict: dict, 包含所有数据的字典
    '''
    
    # 默认噪声配置（模拟真实交通数据）
    if noise_config is None:
        noise_config = {
            'gaussian_noise_level': 0.05,      # 5%的高斯噪声
            'block_missing_rate': 0.02,        # 2%的块状缺失
            'block_size': 6,                   # 30分钟的缺失块
            'point_missing_rate': 0.01,        # 1%的随机点缺失
            'outlier_rate': 0.03,              # 3%的异常值
            'drift_noise': True                # 添加漂移噪声
        }
    
    print("=" * 60)
    print("MD-GRTN 数据处理")
    print("=" * 60)
    print(f"原始数据: {original_data_path}")
    print(f"时间周期: 周={num_of_weeks}, 天={num_of_days}, 小时={num_of_hours}")
    print(f"预测步长: {num_for_predict}")
    print(f"噪声配置: {noise_config}")
    print("-" * 60)
    
    # 1. 加载原始数据（假设已经是修复和插值后的"干净"数据）
    print("1. 加载原始数据...")
    data_seq = np.load(original_data_path)['data']  # (T, N, F)
    print(f"   数据形状: {data_seq.shape}")
    print(f"   时间步数: {data_seq.shape[0]}")
    print(f"   传感器数: {data_seq.shape[1]}")
    print(f"   特征数: {data_seq.shape[2]}")
    
    # 2. 生成多周期样本
    print("2. 生成多周期样本...")
    all_samples = []
    
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        
        if sample[0] is None and sample[1] is None and sample[2] is None:
            continue
            
        week_sample, day_sample, hour_sample, target = sample
        
        # 转换为 (N, F, T) 格式并存储
        sample_data = []
        
        if week_sample is not None:
            week_sample = week_sample.transpose((1, 2, 0))  # (N, F, T)
            sample_data.append(('week', week_sample))
        
        if day_sample is not None:
            day_sample = day_sample.transpose((1, 2, 0))  # (N, F, T)
            sample_data.append(('day', day_sample))
            
        if hour_sample is not None:
            hour_sample = hour_sample.transpose((1, 2, 0))  # (N, F, T)
            sample_data.append(('hour', hour_sample))
            
        target = target.transpose((1, 2, 0))[:, :, :]  # (N, F, T)
        sample_data.append(('target', target))
        sample_data.append(('timestamp', idx))
        
        all_samples.append(sample_data)
    
    print(f"   生成样本数: {len(all_samples)}")
    
    # 3. 重新组织数据结构
    print("3. 重新组织数据结构...")
    
    # 初始化列表
    week_samples = [] if num_of_weeks > 0 else None
    day_samples = [] if num_of_days > 0 else None
    hour_samples = [] if num_of_hours > 0 else None
    targets = []
    timestamps = []
    
    for sample_data in all_samples:
        for data_type, data in sample_data:
            if data_type == 'week' and week_samples is not None:
                week_samples.append(data)
            elif data_type == 'day' and day_samples is not None:
                day_samples.append(data)
            elif data_type == 'hour' and hour_samples is not None:
                hour_samples.append(data)
            elif data_type == 'target':
                targets.append(data[:, 0, :])  # 只取第一个特征（流量）(N, T)
            elif data_type == 'timestamp':
                timestamps.append(data)
    
    # 转换为数组并调整维度
    if week_samples is not None:
        week_data = np.array(week_samples)  # (B, N, F, T)
        # 为了一致性，我们只使用第一个特征
        week_data = week_data[:, :, 0:1, :]  # (B, N, 1, T)
    else:
        week_data = None
        
    if day_samples is not None:
        day_data = np.array(day_samples)
        day_data = day_data[:, :, 0:1, :]
    else:
        day_data = None
        
    if hour_samples is not None:
        hour_data = np.array(hour_samples)
        hour_data = hour_data[:, :, 0:1, :]
    else:
        hour_data = None
        
    targets = np.array(targets)  # (B, N, T)
    timestamps = np.array(timestamps)
    
    print(f"   周周期数据: {week_data.shape if week_data is not None else 'None'}")
    print(f"   日周期数据: {day_data.shape if day_data is not None else 'None'}")
    print(f"   小时周期数据: {hour_data.shape if hour_data is not None else 'None'}")
    print(f"   目标数据: {targets.shape}")
    
    # 4. 按7:1:2比例分割（符合MD-GRTN论文）
    print("4. 数据分割 (7:1:2)...")
    total_samples = len(targets)
    train_end = int(total_samples * 0.7)
    val_end = int(total_samples * 0.8)
    
    def split_dataset(data):
        if data is None:
            return None, None, None
        return data[:train_end], data[train_end:val_end], data[val_end:]
    
    # 分割各数据集
    train_week, val_week, test_week = split_dataset(week_data)
    train_day, val_day, test_day = split_dataset(day_data)
    train_hour, val_hour, test_hour = split_dataset(hour_data)
    train_target, val_target, test_target = split_dataset(targets)
    train_timestamp, val_timestamp, test_timestamp = split_dataset(timestamps)
    
    print(f"   训练集: {len(train_target)} 样本")
    print(f"   验证集: {len(val_target)} 样本")
    print(f"   测试集: {len(test_target)} 样本")
    
    # 5. 归一化
    print("5. 数据归一化...")
    
    week_stats, train_week_norm, val_week_norm, test_week_norm = None, None, None, None
    day_stats, train_day_norm, val_day_norm, test_day_norm = None, None, None, None
    hour_stats, train_hour_norm, val_hour_norm, test_hour_norm = None, None, None, None
    
    if train_week is not None:
        week_stats, train_week_norm, val_week_norm, test_week_norm = normalize_data(
            train_week, val_week, test_week
        )
        print(f"   周周期归一化完成")
        
    if train_day is not None:
        day_stats, train_day_norm, val_day_norm, test_day_norm = normalize_data(
            train_day, val_day, test_day
        )
        print(f"   日周期归一化完成")
        
    if train_hour is not None:
        hour_stats, train_hour_norm, val_hour_norm, test_hour_norm = normalize_data(
            train_hour, val_hour, test_hour
        )
        print(f"   小时周期归一化完成")
    
    # 6. 生成噪声版本（用于预训练）
    print("6. 生成噪声数据（模拟真实传感器噪声）...")
    
    train_week_noisy, train_day_noisy, train_hour_noisy = None, None, None
    
    if train_week_norm is not None:
        train_week_noisy = add_comprehensive_traffic_noise(train_week_norm, noise_config)
        print(f"   周周期噪声数据生成完成")
        
    if train_day_norm is not None:
        train_day_noisy = add_comprehensive_traffic_noise(train_day_norm, noise_config)
        print(f"   日周期噪声数据生成完成")
        
    if train_hour_norm is not None:
        train_hour_noisy = add_comprehensive_traffic_noise(train_hour_norm, noise_config)
        print(f"   小时周期噪声数据生成完成")
    
    # 7. 准备返回的数据字典
    dataset_dict = {
        'train': {
            'week': train_week_norm,
            'day': train_day_norm,
            'hour': train_hour_norm,
            'week_noisy': train_week_noisy,
            'day_noisy': train_day_noisy,
            'hour_noisy': train_hour_noisy,
            'target': train_target,
            'timestamp': train_timestamp,
        },
        'val': {
            'week': val_week_norm,
            'day': val_day_norm,
            'hour': val_hour_norm,
            'target': val_target,
            'timestamp': val_timestamp,
        },
        'test': {
            'week': test_week_norm,
            'day': test_day_norm,
            'hour': test_hour_norm,
            'target': test_target,
            'timestamp': test_timestamp,
        },
        'stats': {
            'week': week_stats,
            'day': day_stats,
            'hour': hour_stats,
        },
        'config': {
            'num_of_weeks': num_of_weeks,
            'num_of_days': num_of_days,
            'num_of_hours': num_of_hours,
            'num_for_predict': num_for_predict,
            'points_per_hour': points_per_hour,
            'noise_config': noise_config,
            'total_samples': total_samples,
            'train_samples': len(train_target),
            'val_samples': len(val_target),
            'test_samples': len(test_target),
        }
    }
    
    # 8. 保存数据
    if save_path is not None:
        print("7. 保存数据...")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 准备保存字典
        save_dict = {}
        
        # 添加训练数据
        if train_week_norm is not None:
            save_dict.update({
                'train_week': train_week_norm,
                'train_week_noisy': train_week_noisy,
                'week_mean': week_stats['mean'] if week_stats else 0,
                'week_std': week_stats['std'] if week_stats else 1,
            })
            
        if train_day_norm is not None:
            save_dict.update({
                'train_day': train_day_norm,
                'train_day_noisy': train_day_noisy,
                'day_mean': day_stats['mean'] if day_stats else 0,
                'day_std': day_stats['std'] if day_stats else 1,
            })
            
        if train_hour_norm is not None:
            save_dict.update({
                'train_hour': train_hour_norm,
                'train_hour_noisy': train_hour_noisy,
                'hour_mean': hour_stats['mean'] if hour_stats else 0,
                'hour_std': hour_stats['std'] if hour_stats else 1,
            })
        
        # 添加验证和测试数据
        if val_week_norm is not None:
            save_dict.update({
                'val_week': val_week_norm,
                'test_week': test_week_norm,
            })
            
        if val_day_norm is not None:
            save_dict.update({
                'val_day': val_day_norm,
                'test_day': test_day_norm,
            })
            
        if val_hour_norm is not None:
            save_dict.update({
                'val_hour': val_hour_norm,
                'test_hour': test_hour_norm,
            })
        
        # 添加目标数据和时间戳
        save_dict.update({
            'train_target': train_target,
            'val_target': val_target,
            'test_target': test_target,
            'train_timestamp': train_timestamp,
            'val_timestamp': val_timestamp,
            'test_timestamp': test_timestamp,
        })
        
        # 添加配置信息
        save_dict.update({
            'config_num_weeks': num_of_weeks,
            'config_num_days': num_of_days,
            'config_num_hours': num_of_hours,
            'config_num_predict': num_for_predict,
            'config_points_per_hour': points_per_hour,
        })
        
        # 保存到文件
        np.savez_compressed(save_path, **save_dict)
        print(f"   数据保存到: {save_path}")
    
    print("=" * 60)
    print("MD-GRTN数据处理完成！")
    print("=" * 60)
    
    return dataset_dict


def main():
    '''主函数'''
    parser = argparse.ArgumentParser(description='MD-GRTN数据预处理')
    parser.add_argument('--config', type=str, default='configurations/PEMS_MD_GRTN.conf',
                       help='配置文件路径')
    parser.add_argument('--original_data', type=str, 
                       help='原始数据路径（覆盖配置文件设置）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认使用原始数据所在目录）')
    parser.add_argument('--noise_level', type=float, default=0.05,
                       help='噪声水平（0-1）')
    parser.add_argument('--missing_rate', type=float, default=0.02,
                       help='缺失率（0-1）')
    
    args = parser.parse_args()
    
    # 读取配置文件
    config = configparser.ConfigParser()
    print(f"读取配置文件: {args.config}")
    config.read(args.config)
    
    # 获取数据配置
    data_config = config['Data']
    training_config = config['Training']
    
    # 使用命令行参数或配置文件
    if args.original_data:
        original_data_path = args.original_data
    else:
        original_data_path = data_config['graph_signal_matrix_filename']
    
    # 获取参数
    num_of_vertices = int(data_config['num_of_vertices'])
    points_per_hour = int(data_config['points_per_hour'])
    num_for_predict = int(data_config['num_for_predict'])
    
    num_of_weeks = int(training_config.get('num_of_weeks', 1))
    num_of_days = int(training_config.get('num_of_days', 1))
    num_of_hours = int(training_config.get('num_of_hours', 3))
    
    # 噪声配置
    noise_config = {
        'gaussian_noise_level': args.noise_level,
        'block_missing_rate': args.missing_rate,
        'block_size': 6,
        'point_missing_rate': args.missing_rate * 0.5,
        'outlier_rate': args.missing_rate * 1.5,
        'drift_noise': True
    }
    
    # 生成输出文件名
    dataset_name = os.path.basename(original_data_path).replace('.npz', '')
    output_filename = f"{dataset_name}_md_grtn_w{num_of_weeks}_d{num_of_days}_h{num_of_hours}_p{num_for_predict}.npz"
    
    # 如果没有指定输出目录，使用原始数据所在的目录
    if args.output_dir is None:
        output_path = os.path.join(os.path.dirname(original_data_path), output_filename)
    else:
        output_path = os.path.join(args.output_dir, output_filename)
    
    # 创建数据
    dataset = prepare_md_grtn_dataset(
        original_data_path=original_data_path,
        num_of_weeks=num_of_weeks,
        num_of_days=num_of_days,
        num_of_hours=num_of_hours,
        num_for_predict=num_for_predict,
        points_per_hour=points_per_hour,
        noise_config=noise_config,
        save_path=output_path
    )
    
    print("\n数据统计:")
    print(f"  训练集样本: {dataset['config']['train_samples']}")
    print(f"  验证集样本: {dataset['config']['val_samples']}")
    print(f"  测试集样本: {dataset['config']['test_samples']}")
    print(f"  总样本数: {dataset['config']['total_samples']}")
    print(f"\n输出文件: {output_path}")


if __name__ == "__main__":
    main()