import torch
from torch.autograd import Variable
from initializers import gaussian_init, xavier_init
from functools import partial


class FullyConnectedClassifier:
    def __init__(self,
                 num_classes: int,
                 input_size: int,
                 hidden_size: int,
                 num_hidden_layers: int,
                 activation: str = "relu",
                 init_type: str = "gaussian",
                 init_gaussian_std: float = 0.001,
                 dropout_drop_probability: float = 0.):
        """
        activation: one of ["relu", "tanh"]
        init_type: one of ["gaussian", "xavier"]
        init_gaussian_std: ignored if init_type != "gaussian"
        """
        self.phase = "train"
        self.__choose_activation_func(activation)
        self.classification_layer = FullyConnectedLayer(input_size=hidden_size,
                                                        output_size=num_classes,
                                                        with_bias=True,
                                                        init_type=init_type,
                                                        init_gaussian_std=init_gaussian_std,
                                                        dropout_drop_probability=dropout_drop_probability)
        first_hidden_layer = FullyConnectedLayer(input_size=input_size,
                                                 output_size=hidden_size,
                                                 with_bias=True,
                                                 init_type=init_type,
                                                 init_gaussian_std=init_gaussian_std,
                                                 dropout_drop_probability=dropout_drop_probability)
        self.hidden_layers = [first_hidden_layer]
        if num_hidden_layers > 1:
            for _ in range(num_hidden_layers - 1):
                intermediate_hidden_layer = FullyConnectedLayer(input_size=hidden_size,
                                                                output_size=hidden_size,
                                                                with_bias=True,
                                                                init_type=init_type,
                                                                init_gaussian_std=init_gaussian_std,
                                                                dropout_drop_probability=dropout_drop_probability)
                self.hidden_layers.append(intermediate_hidden_layer)

        self.layers = self.hidden_layers + [self.classification_layer]

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = hidden_layer.forward(x)
            x = self.activation(x)
        logits = self.classification_layer.forward(x)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def predict(self, x):
        probs = self.predict_proba(x)
        pred_labels = probs.argmax(dim=1)
        return pred_labels

    def predict_proba(self, x):
        phase = self.phase
        self.set_phase("eval")
        probs = self.forward(x)
        self.set_phase(phase)
        return probs

    def trainable_params(self):
        params = []
        for layer in self.layers:
            _params = layer.trainable_params()
            params.extend(_params)
        return params

    def set_phase(self, phase: str):
        self.phase = phase
        for layer in self.layers:
            layer.set_phase(phase)

    def __choose_activation_func(self, activation: str):
        if activation.lower() == "relu":
            self.activation = torch.relu
        elif activation.lower() == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError('activation should be one of ["relu", "tanh"]')


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
        self.__choose_init_func(init_type, init_gaussian_std)
        self.W = Variable(self.init_func((input_size, output_size)), requires_grad=True)
        if self.with_bias:
            self.b = Variable(torch.zeros(1, output_size), requires_grad=True)
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

    def __choose_init_func(self, init_type: str, init_gaussian_std: float):
        if init_type.lower() == "gaussian":
            if init_gaussian_std is None:
                raise ValueError("init_gaussian_std can't be None when using gaussian initialization")
            self.init_func = partial(gaussian_init, std=init_gaussian_std, mean=0.)
        elif init_type.lower() == "xavier":
            self.init_func = xavier_init
        else:
            raise ValueError('init_type should be one of ["gaussian", "xavier"]')


class Dropout:
    def __init__(self, drop_probability: float):
        self.phase = "train"
        self.drop_probability = drop_probability

    def forward(self, x):
        if self.phase == "train":
            drop_mask = torch.rand_like(x) < self.drop_probability
            res = torch.where(drop_mask, torch.Tensor([0.]), x)
            res /= 1 - self.drop_probability
            return res
        else:
            return x

    def set_phase(self, phase: str):
        self.phase = phase


def _example():
    layer = FullyConnectedLayer(input_size=32 * 32 * 3,
                                output_size=256,
                                with_bias=True,
                                init_type="gaussian",
                                init_gaussian_std=1.)
    x = torch.rand(100, 32 * 32 * 3)
    res = layer.forward(x)

    net = FullyConnectedClassifier(num_classes=10,
                                   input_size=32 * 32 * 3,
                                   hidden_size=256,
                                   num_hidden_layers=2,
                                   activation="relu",
                                   init_type="gaussian",
                                   init_gaussian_std=0.05)
    probs = net.forward(x)

    import numpy as np
    probs_np = np.around(probs.data.numpy(), 2)
    print(probs_np)
    pass


if __name__ == '__main__':
    _example()
