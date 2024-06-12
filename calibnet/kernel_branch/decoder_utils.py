import torch
from torch import nn
from torch.nn import functional as F
from detectron2.layers import Conv2d
from torch.nn import init
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill
import math
import random


class CoarsePredictor(nn.Module):
    def __init__(self, in_channel,):
        super().__init__()
        middle_dim = in_channel//2
        self.conv1 = Conv2d(in_channel, middle_dim, kernel_size=3, stride=1, padding=1)
        self.norm = nn.GroupNorm(32, middle_dim)
        self.activation = nn.GELU()
        self.conv2 = Conv2d(middle_dim, 1, kernel_size=1)

        nn.init.xavier_uniform_(self.conv1.weight, gain=1)
        nn.init.xavier_uniform_(self.conv2.weight, gain=1)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, features):
        out = []
        if isinstance(features, list):
            for i in range(len(features)):
                out.append(self.conv2(self.activation(self.norm(self.conv1(features[i])))))
        else:
            out.append(self.conv2(self.activation(self.norm(self.conv1(features)))))
        return out

