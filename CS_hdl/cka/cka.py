import torch
import numpy as np
from torchvision.models import googlenet, resnet50, resnet18
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, ToTensor
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # 导入 tqdm

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义一个函数来提取模型的中间层输出
def get_intermediate_layers(model, input_data):
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach())

    hooks = []
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            hooks.append(layer.register_forward_hook(hook_fn))

    model(input_data)
    for hook in hooks:
        hook.remove()

    return activations

# 定义 CKA 计算函数
def centered_kernel_alignment(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    norm_X = torch.norm(X)
    norm_Y = torch.norm(Y)

    # 分批计算 (X.T @ Y)
    batch_size = 100  # 根据需要调整批次大小
    result = 0
    for i in tqdm(range(0, X.size(1), batch_size), desc="Computing CKA", leave=False):  # 使用 tqdm 包装循环
        for j in range(0, Y.size(1), batch_size):
            result += (X[:, i:i+batch_size].T @ Y[:, j:j+batch_size]).norm() ** 2

    return result / (norm_X * norm_Y) ** 2

# 加载数据集4+
transform = Compose([Resize((224, 224)), ToTensor()])
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
subset_dataset = Subset(dataset, range(5))  # 只使用前5个样本
dataloader = DataLoader(subset_dataset, batch_size=1, shuffle=True)  # 减小批次大小

# 加载模型
googlenet_model = googlenet(pretrained=True).to(device).eval()
resnet_model = resnet18(pretrained=True).to(device).eval()

# 获取一个批次的数据
data, _ = next(iter(dataloader))
data = data.to(device)

# 提取中间层输出
googlenet_layers = get_intermediate_layers(googlenet_model, data)
resnet_layers = get_intermediate_layers(resnet_model, data)

# 计算并收集每一层的 CKA
cka_scores = []
for i, (googlenet_layer, resnet_layer) in enumerate(tqdm(zip(googlenet_layers, resnet_layers), total=len(googlenet_layers), desc="Comparing Layers")):  # 使用 tqdm 包装循环
    googlenet_layer = googlenet_layer.view(googlenet_layer.size(0), -1)
    resnet_layer = resnet_layer.view(resnet_layer.size(0), -1)

    cka_score = centered_kernel_alignment(googlenet_layer, resnet_layer)
    cka_scores.append(cka_score)
    print(f'Layer {i + 1}: CKA Score = {cka_score:.4f}')

# 将结果保存到图片中
plt.figure(figsize=(10, 8))
sns.heatmap(np.array(cka_scores).reshape(-1, 1), annot=True, cmap="YlOrRd", cbar=False, linewidths=1, annot_kws={"size": 10, "weight": "bold"})
plt.title("CKA Layerwise Similarity between ResNet and GoogLeNet", pad=20)
plt.xlabel("ResNet Layers")
plt.yticks(np.arange(len(cka_scores)) + 0.5, [f"Layer {i+1}" for i in range(len(cka_scores))], rotation=0)
plt.tight_layout()
plt.savefig("layerwise_cka_similarity_google2resnet18.png", dpi=300)
