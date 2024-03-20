import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from sklearn.metrics.pairwise import polynomial_kernel
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import torch.nn as nn
from torchvision.datasets import MNIST

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match the input size of the models
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    
    transforms.Normalize((0.5,), (0.5,))  # Normalize to match the expected input of the models
])

# Create MNIST dataset
mnist_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
subset_size = 1000  # 使用数据集的前 10000 个样本
mnist_subset = torch.utils.data.Subset(mnist_dataset, range(subset_size))
loader = DataLoader(mnist_subset, batch_size=8, shuffle=True)

def cka(X, Y):
    X_kernel = polynomial_kernel(X)
    Y_kernel = polynomial_kernel(Y)
    hsic_X = np.trace(np.dot(X_kernel, X_kernel))
    hsic_Y = np.trace(np.dot(Y_kernel, Y_kernel))
    hsic_XY = np.trace(np.dot(X_kernel, Y_kernel))
    return hsic_XY / np.sqrt(hsic_X * hsic_Y)

def get_features(model, layer_or_seq, loader):
    features = []
    def hook(module, input, output):
        # 全局平均池化
        features.append(output.flatten(1).detach().cpu().numpy())
    handles = []
    if isinstance(layer_or_seq, nn.Sequential):
        # 如果是Sequential容器，为容器中的每个层注册钩子
        for layer in layer_or_seq:
            handle = layer.register_forward_hook(hook)
            handles.append(handle)
    else:
        # 如果是单个层，直接注册钩子
        handle = layer_or_seq.register_forward_hook(hook)
        handles.append(handle)
    with torch.no_grad():
        for inputs, _ in loader:
            _ = model(inputs.to(device))
    for h in handles:
        h.remove()
    return np.concatenate(features, axis=0)
def compare_layers(model1, model2, loader, device):
    # 在这里更换不同的模型结构
    models1_layers = [model1.layer1, model1.layer2, model1.layer3, model1.layer4]
    models2_layers = [nn.Sequential(
                    model2.conv1,
                ), nn.Sequential(
                    model2.inception3a,
                    model2.inception3b,
                ),nn.Sequential(
                    model2.inception4a,
                    model2.inception4b,
                    model2.inception4c,
                    model2.inception4d,
                ),nn.Sequential(
                    model2.inception5a,
                    model2.inception5b,
                )]
  
    # alexnet type
    #  models2_layers = [model2.features[2], model2.features[5], model2.features[7], model2.features[10]]
    
    assert len(models1_layers) == len(models2_layers), "Number of layers to compare must be the same."
    
    similarities = []
    
    # Iterate over the layers
    for i, (model1_layer, model2_layer) in enumerate(zip(models1_layers, models2_layers)):
        print(f"Comparing layer {i+1}/{len(models1_layers)}...")
        
        # Get features from the current layers
        model1_features = get_features(model1, model1_layer, loader)
        model2_features = get_features(model2, model2_layer, loader)
        layer_similarities = []
        for j in range(model1_features.shape[0]):
            similarity = cka(model1_features[j:j+1, :], model2_features[j:j+1, :])
            layer_similarities.append(similarity)
        # Compute CKA similarity
        # similarity = cka(model1_features, model2_features)
        avg_similarity = np.mean(layer_similarities)
        similarities.append(avg_similarity)
        
        print(f"Layer {i+1} CKA Similarity: {similarity}")
    
    return similarities

# Load pre-trained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# resnet = models.resnet18(pretrained=True).to(device).eval()
# alexnet = models.alexnet(pretrained=True).to(device).eval()
model1 = models.resnet18(pretrained=True).to(device).eval()
model2 = models.googlenet(pretrained=True).to(device).eval()
# Use your data loader
similarities = compare_layers(model1, model2, loader, device)
print(f"CKA Similarity: {similarities}")

# ... your existing code ...

plt.figure(figsize=(10, 8))
sns.heatmap(np.array(similarities).reshape(-1, 1), annot=True, cmap="YlOrRd", cbar=False, linewidths=1, annot_kws={"size": 10, "weight": "bold"})
plt.title("CKA Layerwise Similarity between ResNet and googleNet", pad=20)
plt.xlabel("ResNet Layers")
plt.yticks(np.arange(len(similarities)) + 0.5, [f"Layer {i+1}" for i in range(len(similarities))], rotation=0)
plt.tight_layout()
plt.savefig("layerwise_cka_similarity.png", dpi=300)
