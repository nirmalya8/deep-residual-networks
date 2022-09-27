#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from torch import Tensor
from typing import Type

'''
Here, we have broken up the ResNet18 model into the different
residual blocks. Each residual block has a fixed kernel size, i.e. 3X3
and two of these kernels are applied in each layer before the input
is added to the output to create the identity. We start by coding the
residual block.
'''

class ResidualBlock(nn.Module):
    def __init__(self, in_channels:int,out_channels:int,stride:int=1,expansion:int=1,downsample: nn.Module=None):
        super(ResidualBlock,self).__init__()
        self.expansion = expansion
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size = 3, padding=1,bias=False) 
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
    

    def forward(self,x:Tensor)-> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out+identity
        out = self.relu(out)

        return out


'''
Now that the residual block has been built, we need to build up the
full ResNet with the initial 7X7 kernel with 64 filters at the begining
and the classification head consisting of the average pooling layer
and the softmax activation function.
'''

class ResNet(nn.Module):
    def __init__(self,img_channels:int,num_layers:int,block:Type[ResidualBlock],num_classes:int=1000):

