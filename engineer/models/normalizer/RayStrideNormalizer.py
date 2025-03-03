import torch.nn as nn
import torch
from engineer.models.registry import DEPTH
import numpy as np
from ..common import ConvBlock
import torch.nn.functional as F

@DEPTH.register_module
class RayStrideNormalizer(nn.Module):
    def __init__(self, filter_channels, last_op, factor):
        super(RayStrideNormalizer, self).__init__()

        self.filters = nn.ModuleList()

        for l in range(0, len(filter_channels)-1):
            self.filters.append(nn.Conv1d(
                filter_channels[l],
                filter_channels[l+1],
                1))
            
        self.factor = factor

        if last_op == 'sigmoid':
            self.last_op = nn.Sigmoid()
        elif last_op == 'tanh':
            self.last_op = nn.Tanh()
        else:
            raise NotImplementedError("only sigmoid function could "
                "be used in terms of sigmoid")

        self.name = "RayStrideNormalizer"
        self.input_para=dict(
            filter_channels=filter_channels,
            last_op=last_op,
            factor=factor
        )

    def forward(self, x)->torch.Tensor:
        for i, f in enumerate(self.filters):
            x = f(x)
            if i != len(self.filters) - 1:
                x = F.leaky_relu(x)  
        return self.last_op(x)*self.factor
    
    @property
    def name(self):
        __repr = "{}(Parameters: ".format(self.__name)
        for key in self.input_para.keys():
            __repr+="{}:{}, ".format(key,self.input_para[key])
        __repr=__repr[:-2]
        return __repr+')'
    
    @name.setter
    def name(self,v):
        self.__name = v