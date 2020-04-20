from layers import *

class CNNClassifier:
    def __init__(self,
                 num_classes: int=10,
                 input_size: int=32,
                 hidden_size: int,
                 num_cnn_layers : int=2,
                 activation: str = "relu",
                 init_type: str = "gaussian",
                 init_gaussian_std: float = 0.001,
                 dropout_drop_probability: float = 0.):
        """
        activation: one of ["relu", "tanh"]
        init_type: one of ["gaussian", "xavier"]
        init_gaussian_std: ignored if init_type != "gaussian"
        """
        if num_cnn_layers < 2:
            raise ValueError('num of convolution layer should be bigger then 2')
        self.phase = "train"
        self.__choose_activation_func(activation)
        self.hidden_layers = []
        self.__init_additional_cnn_layers()
        self.__init_basic_net()
        self.cnn_layers = []
        for _ in range(num_cnn_layers - 2): # layer which keep the input the same size
            intermediate_hidden_layer = FullyConnectedLayer(input_size=hidden_size,
                                                            output_size=hidden_size,
                                                            with_bias=True,
                                                            init_type=init_type,
                                                            init_gaussian_std=init_gaussian_std,
                                                            dropout_drop_probability=dropout_drop_probability)
            self.hidden_layers.append(intermediate_hidden_layer)

        self.layers = self.hidden_layers + [self.classification_layer]
        self.dropout = False
        if dropout_drop_probability != 0.:
            self.dropout = Dropout(dropout_drop_probability)

    def __init_additional_cnn_layers(self):
        pass

    def __init_basic_net(self,input_size=32, kernels_num=(64,12),kernels_size=(3,3),num_classes=10, hidden_layer_size=784,init_type, init_gaussian_std
                         ):
        last_cnn = ConvLayer(in_channels=kernels_num[0], out_channels=kernels_num[1])
        self.classification_layer = FullyConnectedLayer(input_size=hidden_layer_size,
                                                        output_size=num_classes,
                                                        with_bias=True,
                                                        init_type=init_type,
                                                        init_gaussian_std=init_gaussian_std)
        self.cnn_layers += []
     in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int,
                 init_type: str,
                 stride: int=1,
                 with_bias: bool = True,
                 init_gaussian_std: float = None)

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = hidden_layer.forward(x)
            x = self.activation(x)
            if self.dropout:
                x = self.dropout.forward(x)
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