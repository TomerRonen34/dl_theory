import torch
import torch.nn.functional as F
from typing import Tuple
from initializers import get_init_func


class Dropout:
    def __init__(self, drop_probability: float):
        self.phase = "train"
        self.drop_probability = drop_probability

    def forward(self, x):
        if self.phase == "train":
            drop_mask = torch.rand_like(x) < self.drop_probability
            res = torch.where(drop_mask,  torch.zeros((1) ), x)
            res /= 1 - self.drop_probability
            return res
        else:
            return x

    def set_phase(self, phase: str):
        self.phase = phase


class FullyConnectedLayer:
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 with_bias: bool,
                 init_type: str,
                 init_gaussian_std: float = None,
                 dropout_drop_probability: float = 0.):
        self.phase = "train"
        self.with_bias = with_bias
        self.init_func = get_init_func(init_type, init_gaussian_std)
        self.W = self.init_func((input_size, output_size))
        if self.with_bias:
            self.b = torch.zeros(1, output_size, requires_grad=True)
        self.dropout = None
        if dropout_drop_probability != 0.:
            self.dropout = Dropout(dropout_drop_probability)

    def forward(self, x):
        res = torch.matmul(x, self.W)
        if self.with_bias:
            res = res + self.b
        if self.dropout is not None:
            res = self.dropout.forward(res)
        return res

    def trainable_params(self):
        params = [self.W]
        if self.with_bias:
            params.append(self.b)
        return params

    def set_phase(self, phase: str):
        self.phase = phase
        if self.dropout is not None:
            self.dropout.set_phase(phase)


class ConvLayer:
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int,
                 stride: int=1,
                 with_bias: bool = True,
                 init_type: str = 'xavier',
                 init_gaussian_std: float = None,
                 dropout_drop_probability: float = 0.):
        self.phase = "train"
        self.with_bias = with_bias
        self.init_func = get_init_func(init_type, init_gaussian_std)
        self.W = self.init_func((out_channels, in_channels, kernel_size, kernel_size)) #  [out_channel, in_channel, kernel_H, kernel_W]
        self.b = None
        self.stride = stride
        self.padding = [padding]
        if self.with_bias:
            self.b = torch.zeros(1, out_channels, requires_grad=True)
        self.dropout = None
        if dropout_drop_probability != 0.:
            self.dropout = Dropout(dropout_drop_probability)
    def forward(self, x):
        return F.conv2d(x, weight=self.W, bias=self.b, stride = self.stride, padding=self.padding)

    def trainable_params(self):
        params = [self.W]
        if self.with_bias:
            params.append(self.b)
        return params

    def set_phase(self, phase: str):
        self.phase = phase
        if self.dropout is not None:
            self.dropout.set_phase(phase)