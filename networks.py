import warnings
warnings.filterwarnings("ignore")

from torch import nn
import torch
import numpy as np

class MLP(nn.Module):

    def __init__(self, num_features, num_classes):
        super(MLP, self).__init__()
        #nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        #class torch.nn.Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(num_features, 274)
        self.fc2 = nn.Linear(274, 135)
        self.fc3 = nn.Linear(135, 65)
        self.fc4 = nn.Linear(65, 30)
        self.fc5 = nn.Linear(30, 10)
        self.fc6 = nn.Linear(10, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        
        
        return out
        
# data = torch.rand(2,274).float()
# model = MLP(batch_size = 2, num_features=data.shape[1], num_classes=2)
# out = model(data)
# print(out)