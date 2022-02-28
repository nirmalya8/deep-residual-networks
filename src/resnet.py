import torch
import torch.nn as nn

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

# Making the Convolution block with Auto Padding
class autoPadConv2D(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.padding = (self.kernel_size[0]//2, self.kernel_size[1]//2)
conv3x3 = partial(autoPadConv2D,kernel_size=3, bias=False)

# conv = conv3x3(in_channels=32, out_channels=64)
# print(conv)

# To choose from a variety of activation functions
def choose_activation(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

print(choose_activation('leaky_relu'))
print(choose_activation('selu'))