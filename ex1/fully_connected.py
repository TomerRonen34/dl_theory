import torch
from layers import FullyConnectedLayer
from time import time

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
        self.classification_layer = FullyConnectedLayer(input_size=hidden_size, output_size=num_classes,
                                                        init_type=init_type, with_bias=True,
                                                        init_gaussian_std=init_gaussian_std,
                                                        dropout_drop_probability=dropout_drop_probability)
        first_hidden_layer = FullyConnectedLayer(input_size=input_size, output_size=hidden_size, init_type=init_type,
                                                 with_bias=True, init_gaussian_std=init_gaussian_std,
                                                 dropout_drop_probability=dropout_drop_probability)
        self.hidden_layers = [first_hidden_layer]
        if num_hidden_layers > 1:
            for _ in range(num_hidden_layers - 1):
                intermediate_hidden_layer = FullyConnectedLayer(input_size=hidden_size, output_size=hidden_size,
                                                                init_type=init_type, with_bias=True,
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
        t = time()
        with torch.no_grad():
            probs = self.forward(x)
        print(f'eval took {time()-t} seconds')
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


def _example():
    layer = FullyConnectedLayer(input_size=32 * 32 * 3, output_size=256, init_type="gaussian", with_bias=True,
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
