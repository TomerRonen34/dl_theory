import os
import os.path as osp
import torch
import pickle
import json
from fully_connected import FullyConnectedClassifier
from cifar_loader import prepare_cifar_data_for_vector_classifier
from training import fit_classifier


def train_models():
    save_dir = osp.join("models", "fully_connected")
    dataset_dir = "cifar-10-batches-py"
    cache_dir = "data_cache"

    subsample_fraction = 0.1
    grid_init_gaussian_std = [1e-1, 1e-2, 1e-3]
    grid_learning_rate = [1e-2, 1e-3, 1e-4]
    grid_momentum = [0., 0.5, 0.9]

    X_train, y_train, X_test, y_test, class_names = (
        prepare_cifar_data_for_vector_classifier(dataset_dir,
                                                 cache_dir,
                                                 subsample_fraction))

    num_models = len(grid_init_gaussian_std) * len(grid_learning_rate) * len(grid_momentum)
    i_model = 1

    for init_gaussian_std in grid_init_gaussian_std:
        for learning_rate in grid_learning_rate:
            for momentum in grid_momentum:
                print(f"\nmodel {i_model}/{num_models}")
                print(f"init_gaussian_std: {init_gaussian_std}  learning_rate: {learning_rate}"
                      f"  momentum: {momentum}")
                model_name = f"fully_connected_{i_model}"
                train_and_eval_fully_connected_model(X_train, y_train, X_test, y_test, class_names,
                                                     save_dir, model_name,
                                                     init_gaussian_std, learning_rate, momentum)
                i_model += 1


def train_and_eval_fully_connected_model(X_train, y_train, X_test, y_test, class_names,
                                         save_dir, model_name,
                                         init_gaussian_std, learning_rate, momentum):
    epochs = 30
    num_classes = len(class_names)
    input_size = X_train.shape[1]
    hidden_size = 256
    num_hidden_layers = 1
    activation = "relu"
    init_type = "gaussian"
    batch_size = 32
    seed = 34

    net = FullyConnectedClassifier(num_classes,
                                   input_size,
                                   hidden_size,
                                   num_hidden_layers,
                                   activation,
                                   init_type,
                                   init_gaussian_std)

    optimizer = torch.optim.SGD(net.trainable_params(),
                                lr=learning_rate,
                                momentum=momentum)

    metrics = fit_classifier(net,
                             optimizer,
                             X_train,
                             y_train,
                             epochs,
                             batch_size,
                             seed,
                             X_test,
                             y_test)

    hyper_params = dict(
        init_gaussian_std=init_gaussian_std,
        learning_rate=learning_rate,
        momentum=momentum,
        epochs=epochs,
        num_classes=num_classes,
        input_size=input_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        activation=activation,
        init_type=init_type,
        batch_size=batch_size,
        seed=seed)

    save_model(net,
               metrics,
               hyper_params,
               model_name,
               save_dir)


def save_model(net, metrics, hyper_params, model_name, save_dir):
    model_dir = osp.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    with open(osp.join(model_dir, "net.pkl"), 'wb') as f:
        pickle.dump(net, f)

    with open(osp.join(model_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(osp.join(model_dir, "hyper_params.json"), 'w') as f:
        json.dump(hyper_params, f, indent=2)


if __name__ == '__main__':
    train_models()
