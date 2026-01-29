#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os

print("=" * 60)
print("检查数据文件")
print("=" * 60)

# 配置参数（从 PEMS04_astgcn.conf）
num_of_weeks = 1
num_of_days = 1
num_of_hours = 12
num_for_predict = 12

expected_filename = f"data/PEMS04/PEMS04_md_grtn_w{num_of_weeks}_d{num_of_days}_h{num_of_hours}_p{num_for_predict}.npz"
print(f"期望的文件名: {expected_filename}")
print(f"文件存在: {os.path.exists(expected_filename)}")

if os.path.exists(expected_filename):
    data = np.load(expected_filename)
    print(f"\n数据文件中的键: {list(data.keys())}")
    
    # 检查训练数据维度
    keys_to_check = [
        'train_hour_noisy', 'train_hour',
        'train_day_noisy', 'train_day',
        'train_week_noisy', 'train_week'
    ]
    
    print("\n" + "=" * 60)
    print("数据维度检查")
    print("=" * 60)
    
    for key in keys_to_check:
        if key in data:
            arr = data[key]
            print(f"{key}: {arr.shape}")
        else:
            print(f"{key}: [不存在]")
    
    print("\n" + "=" * 60)
    print("问题诊断")
    print("=" * 60)
    print("配置文件中的参数:")
    print(f"  num_of_weeks = {num_of_weeks}  （日周期，7天×24小时×num_of_weeks = {7*24*num_of_weeks} 步）")
    print(f"  num_of_days = {num_of_days}   （小时周期，24小时×num_of_days = {24*num_of_days} 步）")
    print(f"  num_of_hours = {num_of_hours}  （近期连续，1小时×num_of_hours = {num_of_hours} 步）")
    print(f"  num_for_predict = {num_for_predict} （预测目标，12步）")
    
