'''
|水生哺乳动物| 海狸，海豚，水獭，海豹，鲸鱼|0-4
|鱼| 水族馆的鱼，比目鱼，射线，鲨鱼，鳟鱼|5-9
|花卉| 兰花，罂粟花，玫瑰，向日葵，郁金香| 10-14
|食品容器| 瓶子，碗，罐子，杯子，盘子|15-19
|水果和蔬菜| 苹果，蘑菇，橘子，梨，甜椒|20-24
|家用电器| 时钟，电脑键盘，台灯，电话机，电视机|25-29
|家用家具| 床，椅子，沙发，桌子，衣柜|30-34
|昆虫| 蜜蜂，甲虫，蝴蝶，毛虫，蟑螂|35-39
|大型食肉动物| 熊，豹，狮子，老虎，狼|40-44
|大型人造户外用品| 桥，城堡，房子，路，摩天大楼|45-49
|大自然的户外场景| 云，森林，山，平原，海|50-54
|大杂食动物和食草动物| 骆驼，牛，黑猩猩，大象，袋鼠|55-59
|中型哺乳动物| 狐狸，豪猪，负鼠，浣熊，臭鼬|60-64
|非昆虫无脊椎动物| 螃蟹，龙虾，蜗牛，蜘蛛，蠕虫|65-69
|人| 宝贝，男孩，女孩，男人，女人|70-74
|爬行动物| 鳄鱼，恐龙，蜥蜴，蛇，乌龟|75-79
|小型哺乳动物| 仓鼠，老鼠，兔子，母老虎，松鼠|80-84
|树木| 枫树，橡树，棕榈，松树，柳树|85-89
|车辆1| 自行车，公共汽车，摩托车，皮卡车，火车|90-94
|车辆2| 割草机，火箭，有轨电车，坦克，拖拉机|95-99
[96,93,80,42,84,68,44,42,97,99]
cifar10 
# cifar10 0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer, 
5:dog, 6: frog, 7: horse, 8: ship, 9: truck
'''

import numpy as np
import os
import random
import torch
import torchvision
import torchvision.transforms as transforms
import copy

random.seed(1)
np.random.seed(1)

# CIFAR-100到CIFAR-10的类别映射


def generate_cifar100(dir_path, datasize, classes=np.arange(10)):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    # 加载 CIFAR-100 数据
    full_dataset = torchvision.datasets.CIFAR100(root=dir_path, train=True, transform=transform, download=True)
    
    # 选择映射到CIFAR-10的10个类别
    selected_cifar10_classes = [96,93,80,42,84,68,44,43,97,99]

    # 创建新的数据集，只包括选定的10个类别
    cifar10_dataset = create_cifar10_dataset(full_dataset, selected_cifar10_classes)
    
    # 生成部分数据集
    partial_dataset,total_samples = generate_partial_data(cifar10_dataset, classes, datasize)
    print(f"Total samples in the partial dataset: {total_samples}")
    # 保存部分数据集
    save_partial_dataset(dir_path, partial_dataset)

# def generate_partial_data(dataset, classes, datasize=5000):
#     targets = dataset.targets
#     if isinstance(targets, list):
#         targets = np.array(targets)
#     data_indices = []
#     for c in classes:
#         idx_c = list(np.where(targets == c)[0])
#         data_indices.extend(idx_c)
#     data_indices = np.random.choice(data_indices, size=datasize, replace=False)
    
#     partial_dataset = copy.deepcopy(dataset)
#     partial_dataset.data = partial_dataset.data[data_indices]
#     if isinstance(partial_dataset.targets, list):
#         partial_dataset.targets = np.array(partial_dataset.targets)
#     partial_dataset.targets = partial_dataset.targets[data_indices]
    
#     return partial_dataset
def generate_partial_data(dataset, classes, datasize=1000):
    targets = dataset.targets
    if isinstance(targets, list):
        targets = np.array(targets)
    
    data_indices = []
    total_samples = 0

    for c in classes:
        idx_c = list(np.where(targets == c)[0])
        sample_size = min(len(idx_c), datasize // len(classes))  # Adjust sample size for each class
        if sample_size > 0:
            selected_indices = np.random.choice(idx_c, size=sample_size, replace=False)
            data_indices.extend(selected_indices)
            total_samples += sample_size

    if len(data_indices) == 0:
        raise ValueError("No data found for the specified classes")

    partial_dataset = copy.deepcopy(dataset)
    partial_dataset.data = partial_dataset.data[data_indices]
    partial_dataset.targets = np.array(partial_dataset.targets)[data_indices]
    
    return partial_dataset, total_samples 
def save_partial_dataset(dir_path, dataset):
    images = dataset.data  # No need to convert, as it's already a numpy array
    labels = np.array(dataset.targets)  # Convert targets to numpy array if not already
    np.savez_compressed(os.path.join(dir_path, '0.npz'), x=images, y=labels)


def create_cifar10_dataset(dataset, selected_classes):
    filtered_data = []
    filtered_targets = []

    for i in range(len(dataset.targets)):
        if dataset.targets[i] in selected_classes:
            filtered_data.append(dataset.data[i])
            filtered_targets.append(selected_classes.index(dataset.targets[i]))

    cifar10_dataset = copy.deepcopy(dataset)
    cifar10_dataset.data = np.array(filtered_data)
    cifar10_dataset.targets = filtered_targets
    
    return cifar10_dataset

# 示例调用
dir_path = "dataset/cifar100_public/"
datasize = 1000  # 指定的数据量
classes = np.arange(10)  # 指定的类别
generate_cifar100(dir_path, datasize, classes)

