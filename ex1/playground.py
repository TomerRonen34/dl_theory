import torch
from fully_connected import FullyConnectedClassifier
from cifar_loader import prepare_cifar_data_for_vector_classifier
from training import fit_classifier

X_train, y_train, X_test, y_test, class_names = (
    prepare_cifar_data_for_vector_classifier(dataset_dir="cifar-10-batches-py",
                                             cache_dir="data_cache",
                                             subsample_fraction=0.1))

grid_search_params = {
    "model": {
        "init_gaussian_std": [1e-2, 1e-3, 1e-4]
    },
    "optimizer": {
        "lr": [1e-3, 1e-4, 1e-5],
        "momentum": [0.8, 0.9, 0.95]

    }
}

num_classes = len(class_names)
input_size = X_train.shape[1]
hidden_size = 256
num_hidden_layers = 1
activation = "relu"
init_type = "gaussian"
epochs = 5
batch_size = 32
seed = 34
save_dir = "models"

net = FullyConnectedClassifier(num_classes=len(class_names),
                               input_size=X_train.shape[1],
                               hidden_size=hidden_size,
                               num_hidden_layers=num_hidden_layers,
                               activation=activation,
                               init_type=init_type,
                               **grid_search_params["model"])

optimizer = torch.optim.SGD(net.trainable_params(),
                            **grid_search_params["optimizer"])

fit_classifier(net, optimizer, X_train, y_train, epochs, batch_size, seed,
               X_test=X_test,
               y_test=y_test)
