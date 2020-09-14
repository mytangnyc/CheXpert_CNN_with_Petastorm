import torch
from torch import nn

class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)


'''
Used as transition step to connect convolution layer to fully connected layer
'''
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.adp = nn.AdaptiveAvgPool2d(sz)
        self.amp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x):
        return torch.cat([self.amp(x), self.adp(x)], 1)