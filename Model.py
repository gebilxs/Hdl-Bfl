import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
def model_judge(args):
    if args.model_name == 'mnist_2nn':
        args.model_name = Mnist_2NN
    elif args.model_name ==  'mnist_cnn':
        args.model_name = Mnist_CNN
    elif args.model_name == 'cnn':
        if "mnist" in args.dataset:
            args.model_name = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
    print(args.model_name)
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head
        
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out

class Mnist_2NN(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(784, 200)
		self.fc2 = nn.Linear(200, 200)
		self.fc3 = nn.Linear(200, 10)

	def forward(self, inputs):
		tensor = F.relu(self.fc1(inputs))
		tensor = F.relu(self.fc2(tensor))
		tensor = self.fc3(tensor)
		return tensor
	

class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


class Mnist_CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.fc1 = nn.Linear(7*7*64, 512)
		self.fc2 = nn.Linear(512, 10)

	def forward(self, inputs):
		tensor = inputs.view(-1, 1, 28, 28)
		tensor = F.relu(self.conv1(tensor))
		tensor = self.pool1(tensor)
		tensor = F.relu(self.conv2(tensor))
		tensor = self.pool2(tensor)
		tensor = tensor.view(-1, 7*7*64)
		tensor = F.relu(self.fc1(tensor))
		tensor = self.fc2(tensor)
		return tensor

class alexnet_mnist(nn.Module):  
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding='same'),
            nn.ReLU()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((3,3))
            

        self.fc = nn.Sequential(OrderedDict([
            ('drop1', nn.Dropout(p=0.5)),
            ('fcin', nn.Linear(256 * 3 * 3, 512)),
            ('relu', nn.ReLU()),
            ('fcout', nn.Linear(512, num_classes)),
        ]))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.fc(out)
        return out