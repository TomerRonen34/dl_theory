from layers import *
from collections import namedtuple

ConvLayerParams = namedtuple("ConvLayerParams", "num_kernels kernel_size padding_size")


class CNNClassifier:
    def __init__(self,
                 num_classes: int,
                 hidden_size: int,
                 conv_layers_params,
                 init_type: str = "gaussian",
                 init_gaussian_std: float = 0.001,
                 dropout_drop_probability: float = 0.,
                 residual_net=False):
        """
        init_type: one of ["gaussian", "xavier"]
        init_gaussian_std: ignored if init_type != "gaussian"
        """

        assert len(conv_layers_params) >= 2, 'num of convolution layer should be bigger then 2'
        self.phase = "train"
        if residual_net:
            self.__init_residual_net(num_classes, hidden_size,
                                     conv_layers_params,
                                     init_type, init_gaussian_std,
                                     dropout_drop_probability)
        else:
            self.__init_basic_net(num_classes, hidden_size,
                                  conv_layers_params,
                                  init_type, init_gaussian_std,
                                  dropout_drop_probability)

    def __init_basic_net(self, num_classes, hidden_layer_size, conv_layers_params: list, init_type='xavier',
                         init_gaussian_std=None, dropout_drop_probability=0.):
        self.layers = []
        in_channels = 3
        for conv_params in conv_layers_params:
            conv = ConvLayer(in_channels=in_channels, out_channels=conv_params.num_kernels,
                             kernel_size=conv_params.kernel_size,
                             padding=conv_params.padding_size,
                             init_type=init_type, init_gaussian_std=init_gaussian_std)
            in_channels = conv_params.num_kernels
            self.layers.append(conv)
            self.layers.append(ReLU())

        fc_layer = FullyConnectedLayer(input_size=7 * 7 * conv_layers_params[-1].num_kernels,
                                       output_size=hidden_layer_size,
                                       init_type=init_type, init_gaussian_std=init_gaussian_std)
        classification_layer = FullyConnectedLayer(input_size=hidden_layer_size, output_size=num_classes,
                                                   init_type=init_type, init_gaussian_std=init_gaussian_std)
        self.layers.insert(-2, MaxPool2D(kernel_size=2))
        self.layers.append(MaxPool2D(kernel_size=2))
        self.layers += [Flatten(), Dropout(dropout_drop_probability), fc_layer, ReLU(),
                        Dropout(dropout_drop_probability), classification_layer]

    def __init_residual_net(self, num_classes, hidden_layer_size, conv_layers_params: list, init_type='xavier',
                            init_gaussian_std=None, dropout_drop_probability=0.):
        self.layers = []
        in_channels = 3
        for i in range(len(conv_layers_params)):
            conv_params = conv_layers_params[i]
            conv = ConvLayer(in_channels=in_channels, out_channels=conv_params.num_kernels,
                             kernel_size=conv_params.kernel_size,
                             padding=conv_params.padding_size,
                             init_type=init_type, init_gaussian_std=init_gaussian_std)
            in_channels = conv_params.num_kernels
            if len(conv_layers_params) - i > 2 and i != 0:
                res_layer = ResidualLayer(conv)
                self.layers.append(res_layer)
            else:
                self.layers.append(conv)
            self.layers.append(ReLU())

        fc_layer = FullyConnectedLayer(input_size=7 * 7 * conv_layers_params[-1].num_kernels,
                                       output_size=hidden_layer_size,
                                       init_type=init_type, init_gaussian_std=init_gaussian_std)
        classification_layer = FullyConnectedLayer(input_size=hidden_layer_size, output_size=num_classes,
                                                   init_type=init_type, init_gaussian_std=init_gaussian_std)
        self.layers.insert(-2, MaxPool2D(kernel_size=2))
        self.layers.append(MaxPool2D(kernel_size=2))
        self.layers += [Flatten(), Dropout(dropout_drop_probability), fc_layer, ReLU(),
                        Dropout(dropout_drop_probability), classification_layer]

    def forward(self, x):
        for hidden_layer in self.layers:
            x = hidden_layer.forward(x)
        probs = torch.softmax(x, dim=-1)
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
