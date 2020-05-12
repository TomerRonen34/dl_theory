import torch.nn.functional as F
from initializers import *
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self, phase: str = 'train'):
        self.phase = phase

    def trainable_params(self):
        return []

    def set_phase(self, phase: str):
        self.phase = phase

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Dropout(Layer):
    def __init__(self, drop_probability: float):
        super().__init__()
        self.phase = "train"
        self.drop_probability = drop_probability

    def forward(self, x):
        if self.phase == "train":
            drop_mask = torch.rand_like(x) < self.drop_probability
            res = torch.where(drop_mask, torch.zeros(1, device=device), x)
            res /= 1 - self.drop_probability
            return res
        else:
            return x


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch.relu(x)


class FullyConnectedLayer(Layer):
    def __init__(self, input_size: int, output_size: int, init_type: str, with_bias: bool = True,
                 init_gaussian_std: float = None, dropout_drop_probability: float = 0.):
        super().__init__()
        self.with_bias = with_bias
        self.init_func = get_init_func(init_type, init_gaussian_std)
        self.W = self.init_func((input_size, output_size))
        if self.with_bias:
            self.b = zero_init((1, output_size))
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


class ConvLayer(Layer):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int = 0,
                 init_type: str = 'xavier',
                 stride: int = 1,
                 with_bias: bool = True,
                 init_gaussian_std: float = None):
        super().__init__()
        self.with_bias = with_bias
        self.init_func = get_init_func(init_type, init_gaussian_std)
        self.W = self.init_func(
            (out_channels, in_channels, kernel_size, kernel_size))  # [out_channel, in_channel, kernel_H, kernel_W]
        self.stride = stride
        self.padding = [padding]
        self.b = None
        if self.with_bias:
            self.b = zero_init(out_channels)

    def forward(self, x):
        return F.conv2d(x, weight=self.W, bias=self.b, stride=self.stride, padding=self.padding)

    def trainable_params(self):
        params = [self.W]
        if self.with_bias:
            params.append(self.b)
        return params


class MaxPool2D(Layer):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        return F.max_pool2d(x, self.kernel_size, ceil_mode=True)


class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class ResidualLayer(Layer):
    def __init__(self, layer: Layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer.forward(x) + x
