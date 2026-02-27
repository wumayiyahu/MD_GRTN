import os
import numpy as np
import argparse
import configparser

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


def get_sample_indices(data_sequence, num_of_days, num_of_hours, num_of_recs,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    获取单个样本的多周期数据
    
    参数说明：
    - num_of_days: 论文中的 Day 周期数（日周期，7天模式）
    - num_of_hours: 论文中的 Hour 周期数（小时周期，24小时模式）
    - num_of_recs: 论文中的 Rec 周期数（最近连续时间）
    
    返回：
    - day_sample: Day周期数据（7天前的数据）
    - hour_sample: Hour周期数据（24小时前的数据）
    - rec_sample: Rec周期数据（最近连续时间）
    '''
    day_sample, hour_sample, rec_sample = None, None, None
 
    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return day_sample, hour_sample, rec_sample, None
 
    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days, #搜索周期数据索引
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not day_indices:
            return None, None, None, None
        day_sample = np.concatenate([data_sequence[i: j]  #提取周期数据特征
                                      for i, j in day_indices], axis=0)
 
    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not hour_indices:
            return None, None, None, None
        hour_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in hour_indices], axis=0)
 
    if num_of_recs > 0:
        # 根据 MD-GRTN 论文，Rec 表示最近连续的时间序列
        # 应该取 label_start_idx 前面的 num_of_recs * num_for_predict 个时间点
        # 直接取连续的 num_of_recs 个 num_for_predict 长度的序列
        # 保证 Rec/Hour/Day 的输入时间维度都是 num_for_predict
        rec_start_idx = label_start_idx - num_of_recs * num_for_predict
        if rec_start_idx < 0:
            return None, None, None, None
        rec_sample = data_sequence[rec_start_idx: label_start_idx]
 
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]
 
    return day_sample, hour_sample, rec_sample, target


def add_comprehensive_traffic_noise(clean_data, noise_config):
    '''
    添加高斯噪声（在原始数据上添加）
    Parameters:
    -----------
    clean_data: np.ndarray, 干净的原始数据
    Returns:
    --------
    noisy_data: np.ndarray, 添加高斯噪声后的数据
    '''
    noisy_data = clean_data.copy()
    
    # 添加纯高斯噪声
    gaussian_mean = noise_config.get('gaussian_mean', 0)
    gaussian_std = noise_config.get('gaussian_std', 10)  # 默认PEMS=10
    
    gaussian_noise = np.random.normal(
        gaussian_mean,
        gaussian_std,
        noisy_data.shape
    )
    noisy_data += gaussian_noise
    
    return noisy_data


def prepare_md_grtn_dataset(original_data_path,
                           num_of_days=1, num_of_hours=1, num_of_recs=3,
                           num_for_predict=12, points_per_hour=12,
                           noise_config=None, save_path=None):
    '''
    为MD-GRTN准备数据集
    Parameters:
    -----------
    original_data_path: str, 原始数据文件路径（.npz格式）
    num_of_days: int, 论文中的 Day 周期数（日周期，7天模式）
    num_of_hours: int, 论文中的 Hour 周期数（小时周期，24小时模式）
    num_of_recs: int, 论文中的 Rec 周期数（最近连续时间）
    num_for_predict: int, 预测步长
    points_per_hour: int, 每小时数据点数
    noise_config: dict, 噪声配置
    save_path: str, 保存路径
    Returns:
    --------
    dataset_dict: dict, 包含所有数据的字典
    '''

    if noise_config is None:
        noise_config = {
            'dataset_type': 'PEMS',              # 'PEMS' 或 'SZTaxi'
            'gaussian_mean': 0,                  # 固定为0
            'gaussian_std': 10 if 'PEMS' in original_data_path.upper() else 2,  # PEMS=10, SZTaxi=2
        }
    
    print("=" * 60)
    print("MD-GRTN 数据处理")
    print("=" * 60)
    print(f"原始数据: {original_data_path}")
    print(f"时间周期: Day(7天)={num_of_days}, Hour(24小时)={num_of_hours}, Rec(连续)={num_of_recs}")
    print(f"预测步长: {num_for_predict}")
    print(f"噪声配置: {noise_config}")
    print("-" * 60)
    
    # 1. 加载原始数据（已经是修复和插值后的"干净"数据）
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
        sample = get_sample_indices(data_seq, num_of_days, num_of_hours,
                                    num_of_recs, idx, num_for_predict,
                                    points_per_hour)
        
        if sample[0] is None and sample[1] is None and sample[2] is None:
            continue
            
        day_sample, hour_sample, rec_sample, target = sample
        
        # 转换为 (N, F, T) 格式并存储

        sample_data = []
        
        if day_sample is not None:
            day_sample = day_sample.transpose((1, 2, 0))  # (N, F, T) - 论文中的 Day 周期
            sample_data.append(('day', day_sample))
        
        if hour_sample is not None:
            hour_sample = hour_sample.transpose((1, 2, 0))  # (N, F, T) - 论文中的 Hour 周期
            sample_data.append(('hour', hour_sample))
            
        if rec_sample is not None:
            rec_sample = rec_sample.transpose((1, 2, 0))  # (N, F, T) - 论文中的 Rec 周期
            sample_data.append(('rec', rec_sample))
            
        target = target.transpose((1, 2, 0))[:, :, :]  # (N, F, T)
        sample_data.append(('target', target))
        sample_data.append(('timestamp', idx))
        
        all_samples.append(sample_data)
    
    print(f"   生成样本数: {len(all_samples)}")
    
    # 3. 重新组织数据结构
    print("3. 重新组织数据结构...")
    
    # 初始化列表
    day_samples = [] if num_of_days > 0 else None
    hour_samples = [] if num_of_hours > 0 else None
    rec_samples = [] if num_of_recs > 0 else None
    targets = []
    timestamps = []
    
    for sample_data in all_samples:
        for data_type, data in sample_data:
            # 注意：data_type 名称保持为内部标识符，但实际对应关系：
            # 保存到文件时，将按论文符号重新命名
            if data_type == 'day' and day_samples is not None:
                day_samples.append(data)  # 这是 Day 周期数据(7天)
            elif data_type == 'hour' and hour_samples is not None:
                hour_samples.append(data)  # 这是 Hour 周期数据(24小时)
            elif data_type == 'rec' and rec_samples is not None:
                rec_samples.append(data)  # 这是 Rec 周期数据(最近连续)
            elif data_type == 'target':
                targets.append(data[:, 0, :])  # 只取第一个特征（流量）(N, T)
            elif data_type == 'timestamp':
                timestamps.append(data)
    
    # 转换为数组并调整维度
    if day_samples is not None:
        day_data = np.array(day_samples)  # (B, N, F, T) - 将被保存为 X_Day(7天周期)
        # 为了一致性，我们只使用第一个特征
        day_data = day_data[:, :, 0:1, :]  # (B, N, 1, T)
    else:
        day_data = None
        
    if hour_samples is not None:
        hour_data = np.array(hour_samples)  # 将被保存为 X_Hour(24小时周期)
        hour_data = hour_data[:, :, 0:1, :]
    else:
        hour_data = None
        
    if rec_samples is not None:
        rec_data = np.array(rec_samples)  # 将被保存为 X_Rec(最近连续周期)
        rec_data = rec_data[:, :, 0:1, :]
    else:
        rec_data = None
        
    targets = np.array(targets)  # (B, N, T)
    timestamps = np.array(timestamps)
    
    print(f"   周周期数据: {day_data.shape if day_data is not None else 'None'}")
    print(f"   日周期数据: {hour_data.shape if hour_data is not None else 'None'}")
    print(f"   小时周期数据: {rec_data.shape if rec_data is not None else 'None'}")
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
    train_day, val_day, test_day = split_dataset(day_data)
    train_hour, val_hour, test_hour = split_dataset(hour_data)
    train_rec, val_rec, test_rec = split_dataset(rec_data)
    train_target, val_target, test_target = split_dataset(targets)
    train_timestamp, val_timestamp, test_timestamp = split_dataset(timestamps)
    
    print(f"   训练集: {len(train_target)} 样本")
    print(f"   验证集: {len(val_target)} 样本")
    print(f"   测试集: {len(test_target)} 样本")

    clean_target_mean = train_target.mean()
    clean_target_std = train_target.std()

    
    # 5. 在原始数据上添加高斯噪声
    print("5. 生成噪声数据（在原始数据上添加高斯噪声）...")
    print(f"   噪声配置：gaussian_mean={noise_config.get('gaussian_mean', 0)}, gaussian_std={noise_config.get('gaussian_std', 10)}")
    print("-" * 60)
    
    # 训练集噪声数据
    train_day_noisy, train_hour_noisy, train_rec_noisy = None, None, None
    
    if train_day is not None:
        train_day_noisy = add_comprehensive_traffic_noise(train_day, noise_config)
        print(f"   训练集日周期：添加高斯噪声完成")
        
    if train_hour is not None:
        train_hour_noisy = add_comprehensive_traffic_noise(train_hour, noise_config)
        print(f"   训练集小时周期：添加高斯噪声完成")
        
    if train_rec is not None:
        train_rec_noisy = add_comprehensive_traffic_noise(train_rec, noise_config)
        print(f"   训练集最近连续周期：添加高斯噪声完成")
    
    # 验证集噪声数据
    val_day_noisy, val_hour_noisy, val_rec_noisy = None, None, None
    
    if val_day is not None:
        val_day_noisy = add_comprehensive_traffic_noise(val_day, noise_config)
        print(f"   验证集日周期：添加高斯噪声完成")
        
    if val_hour is not None:
        val_hour_noisy = add_comprehensive_traffic_noise(val_hour, noise_config)
        print(f"   验证集小时周期：添加高斯噪声完成")
        
    if val_rec is not None:
        val_rec_noisy = add_comprehensive_traffic_noise(val_rec, noise_config)
        print(f"   验证集最近连续周期：添加高斯噪声完成")
        
    
    # 测试集噪声数据
    test_day_noisy, test_hour_noisy, test_rec_noisy = None, None, None
    
    if test_day is not None:
        test_day_noisy = add_comprehensive_traffic_noise(test_day, noise_config)
        print(f"   测试集日周期：添加高斯噪声完成")
        
    if test_hour is not None:
        test_hour_noisy = add_comprehensive_traffic_noise(test_hour, noise_config)
        print(f"   测试集小时周期：添加高斯噪声完成")
        
    if test_rec is not None:
        test_rec_noisy = add_comprehensive_traffic_noise(test_rec, noise_config)
        print(f"   测试集最近连续周期：添加高斯噪声完成")
 
    # 6. ✅ Z-Score 标准化（使用训练集 target 统计）
    print("6. 执行 Z-Score 标准化（对干净数据和噪声数据都进行标准化）...")
    print("=" * 60)
    
    # 只用训练集的 target 统计 mean/std
    target_mean = train_target.mean()
    target_std = train_target.std()
    print(f"   训练集 target 统计:")
    print(f"   mean = {target_mean:.4f}")
    print(f"   std  = {target_std:.4f}")
    print("-" * 60)
    
    # 标准化函数
    def zscore(x, mean, std):
        return (x - mean) / std
    
    # 标准化训练集干净数据
    print("   标准化训练集干净数据...")
    if train_day is not None:
        train_day = zscore(train_day, clean_target_mean, clean_target_std)
    if train_hour is not None:
        train_hour = zscore(train_hour, clean_target_mean, clean_target_std)
    if train_rec is not None:
        train_rec = zscore(train_rec, clean_target_mean, clean_target_std)
    train_target = zscore(train_target, clean_target_mean, clean_target_std)
    
    # 标准化验证集干净数据
    print("   标准化验证集干净数据...")
    if val_day is not None:
        val_day = zscore(val_day, clean_target_mean, clean_target_std)
    if val_hour is not None:
        val_hour = zscore(val_hour, clean_target_mean, clean_target_std)
    if val_rec is not None:
        val_rec = zscore(val_rec, clean_target_mean, clean_target_std)
    val_target = zscore(val_target, clean_target_mean, clean_target_std)
    
    # 标准化测试集干净数据
    print("   标准化测试集干净数据...")
    if test_day is not None:
        test_day = zscore(test_day, clean_target_mean, clean_target_std)
    if test_hour is not None:
        test_hour = zscore(test_hour, clean_target_mean, clean_target_std)
    if test_rec is not None:
        test_rec = zscore(test_rec, clean_target_mean, clean_target_std)
    test_target = zscore(test_target, clean_target_mean, clean_target_std)
    
    # 标准化训练集噪声数据
    print("   标准化训练集噪声数据...")
    if train_day_noisy is not None:
        train_day_noisy = zscore(train_day_noisy, target_mean, target_std)
    if train_hour_noisy is not None:
        train_hour_noisy = zscore(train_hour_noisy, target_mean, target_std)
    if train_rec_noisy is not None:
        train_rec_noisy = zscore(train_rec_noisy, target_mean, target_std)
    
    # 标准化验证集噪声数据
    print("   标准化验证集噪声数据...")
    if val_day_noisy is not None:
        val_day_noisy = zscore(val_day_noisy, target_mean, target_std)
    if val_hour_noisy is not None:
        val_hour_noisy = zscore(val_hour_noisy, target_mean, target_std)
    if val_rec_noisy is not None:
        val_rec_noisy = zscore(val_rec_noisy, target_mean, target_std)
    
    # 标准化测试集噪声数据
    print("   标准化测试集噪声数据...")
    if test_day_noisy is not None:
        test_day_noisy = zscore(test_day_noisy, target_mean, target_std)
    if test_hour_noisy is not None:
        test_hour_noisy = zscore(test_hour_noisy, target_mean, target_std)
    if test_rec_noisy is not None:
        test_rec_noisy = zscore(test_rec_noisy, target_mean, target_std)
    
    print("   ✅ 标准化完成，验证统计量：")
    print(f"      train_target: mean={train_target.mean():.6f}, std={train_target.std():.6f}")
    print(f"      val_target: mean={val_target.mean():.6f}, std={val_target.std():.6f}")
    print(f"      test_target: mean={test_target.mean():.6f}, std={test_target.std():.6f}")
    print("-" * 60)
    
    # 6. 准备返回的数据字典
    dataset_dict = {
        'train': {
            'day': train_day, # 原始干净数据
            'hour': train_hour,
            'rec': train_rec,
            'day_noisy': train_day_noisy,# 带噪声数据（模型输入）
            'hour_noisy': train_hour_noisy,
            'rec_noisy': train_rec_noisy,
            'target': train_target,        # 原始预测目标数据
            'timestamp': train_timestamp,
        },
        'val': {           
            'day': val_day,# 原始干净数据
            'hour': val_hour,
            'rec': val_rec, 
            'day_noisy': val_day_noisy,# 带噪声数据（模型输入）
            'hour_noisy': val_hour_noisy,
            'rec_noisy': val_rec_noisy,
            'target': val_target,          # 原始预测目标数据
            'timestamp': val_timestamp,
        },
        'test': {            
            'day': test_day, # 原始干净数据
            'hour': test_hour,
            'rec': test_rec, 
            'day_noisy': test_day_noisy,# 带噪声数据（模型输入）
            'hour_noisy': test_hour_noisy,
            'rec_noisy': test_rec_noisy,
            'target': test_target,         # 原始预测目标数据
            'timestamp': test_timestamp,
        },
        'stats': {
            'day': None,
            'hour': None,
            'rec': None,
            'target': None,
        },
        'config': {
            'num_of_days': num_of_days,
            'num_of_hours': num_of_hours,
            'num_of_recs': num_of_recs,
            'num_for_predict': num_for_predict,
            'points_per_hour': points_per_hour,
            'noise_config': noise_config,
            'total_samples': total_samples,
            'train_samples': len(train_target),
            'val_samples': len(val_target),
            'test_samples': len(test_target),
        }
    }
    
    # 7. 保存数据
    if save_path is not None:
        print("6. 保存数据...")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 准备保存字典
        save_dict = {}
        
        # 保存标准化参数（用于评估时反归一化）这里用训练集的干净数据
        save_dict['target_mean'] = clean_target_mean
        save_dict['target_std'] = clean_target_std
        
        # 添加训练数据       
        if train_day is not None:
            save_dict.update({
                'train_day': train_day,
                'train_day_noisy': train_day_noisy,
            })
            
        if train_hour is not None:
            save_dict.update({
                'train_hour': train_hour,
                'train_hour_noisy': train_hour_noisy,
            })

        if train_rec is not None:
            save_dict.update({
                'train_rec': train_rec,
                'train_rec_noisy': train_rec_noisy,
            })
        
        # 添加验证和测试数据
        if val_day is not None:
            save_dict.update({
                'val_day': val_day,
                'val_day_noisy': val_day_noisy,
                'test_day': test_day,
                'test_day_noisy': test_day_noisy,
            })
            
        if val_hour is not None:
            save_dict.update({
                'val_hour': val_hour,
                'val_hour_noisy': val_hour_noisy,
                'test_hour': test_hour,
                'test_hour_noisy': test_hour_noisy,
            })
        
        if val_rec is not None:
            save_dict.update({
                'val_rec': val_rec,
                'val_rec_noisy': val_rec_noisy,
                'test_rec': test_rec,
                'test_rec_noisy': test_rec_noisy,
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
            'config_num_days': num_of_days,
            'config_num_hours': num_of_hours,
            'config_num_recs': num_of_recs,
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
    
    num_of_days = int(training_config.get('num_of_days', 1))
    num_of_hours = int(training_config.get('num_of_hours', 1))
    num_of_recs = int(training_config.get('num_of_recs', 1))
    
    # 噪声类型：纯高斯噪声
    dataset_upper = original_data_path.upper()
    if 'PEMS' in dataset_upper:
        gaussian_std = 10  # PEMS数据集
    else:
        gaussian_std = 2   # SZTaxi数据集
    
    noise_config = {
        'gaussian_mean': 0,
        'gaussian_std': gaussian_std,
    }
    
    print(f"噪声配置：gaussian_mean={noise_config['gaussian_mean']}, gaussian_std={noise_config['gaussian_std']}")
    
    # 生成输出文件名
    dataset_name = os.path.basename(original_data_path).replace('.npz', '')
    output_filename = f"{dataset_name}_md_grtn_w{num_of_days}_d{num_of_hours}_h{num_of_recs}_p{num_for_predict}.npz"
    
    # 如果没有指定输出目录，使用原始数据所在的目录
    if args.output_dir is None:
        output_path = os.path.join(os.path.dirname(original_data_path), output_filename)
    else:
        output_path = os.path.join(args.output_dir, output_filename)
    
    # 创建数据
    dataset = prepare_md_grtn_dataset(
        original_data_path=original_data_path,
        num_of_days=num_of_days,
        num_of_hours=num_of_hours,
        num_of_recs=num_of_recs,
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