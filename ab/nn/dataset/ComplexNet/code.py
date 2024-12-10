import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ComplexConv2d(3, 10, 5, 1)
        self.bn  = ComplexBatchNorm2d(10)
        self.conv2 = ComplexConv2d(10, 20, 5, 1)
        self.fc1 = ComplexLinear(500, 500)
        self.fc2 = ComplexLinear(500, 10)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        x = self.bn(x)
        x = complex_relu(self.conv2(x))
        x = complex_max_pool2d(x, 2, 2)
        x = x.view(-1,500)
        x = self.fc1(x)
        x = complex_relu(x)
        x = self.fc2(x)
        x = x.abs()
        x = F.log_softmax(x, dim=1)
        return x