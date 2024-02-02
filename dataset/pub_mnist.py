import numpy as np
import os
import random
import torch
import torchvision
import torchvision.transforms as transforms
import copy

random.seed(1)
np.random.seed(1)

# ...[保留原有函数定义，如 check]...

def generate_mnist(dir_path, datasize, classes=np.arange(10)):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

#     transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=3),  # 将1个通道转换为3个通道
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    # 加载 MNIST 数据
    full_dataset = torchvision.datasets.MNIST(root=dir_path, train=True, transform=transform, download=True)
    
    # 生成部分数据集
    partial_dataset = generate_partial_data(full_dataset, classes, datasize)
    
    # 保存部分数据集
    save_partial_dataset(dir_path, partial_dataset)

def generate_partial_data(dataset, classes, datasize=5000):
    targets = dataset.targets
    if isinstance(targets, list):
        targets = np.array(targets)
    data_indices = []
    for c in classes:
        idx_c = list(np.where(targets == c)[0])
        data_indices.extend(idx_c)
    data_indices = np.random.choice(data_indices, size=datasize, replace=False)
    
    partial_dataset = copy.deepcopy(dataset)
    partial_dataset.data = partial_dataset.data[data_indices]
    if isinstance(partial_dataset.targets, list):
        partial_dataset.targets = np.array(partial_dataset.targets)
    partial_dataset.targets = partial_dataset.targets[data_indices]
    
    return partial_dataset

def save_partial_dataset(dir_path, dataset):
    # 保存数据集为 .npz 文件
    images = dataset.data.numpy()
    labels = dataset.targets.numpy()
    np.savez_compressed(os.path.join(dir_path, '0.npz'), x=images, y=labels)

# 示例调用
dir_path = "dataset/mnist_public/"
datasize = 5000  # 指定的数据量
classes = np.arange(10)  # 指定的类别
generate_mnist(dir_path, datasize, classes)
