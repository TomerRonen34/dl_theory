from utils import batchify
from losses import cross_entropy_loss
import os
import os.path as osp
import torch
import pickle
import json
from fully_connected import FullyConnectedClassifier
import numpy as np


def train_and_eval_fully_connected_model(X_train, y_train, X_test, y_test, class_names, save_dir, model_name,
                                         weight_decay=0., dropout_drop_probability=0.,
                                         learning_rate=0.001, momentum=0.9,
                                         init_type="xavier", init_gaussian_std=0.001,
                                         hidden_size=256, num_hidden_layers=1, activation="relu",
                                         epochs=100, batch_size=32, seed=34):
    num_classes = len(class_names)
    input_size = X_train.shape[1]

    net = FullyConnectedClassifier(num_classes,
                                   input_size,
                                   hidden_size,
                                   num_hidden_layers,
                                   activation,
                                   init_type,
                                   init_gaussian_std,
                                   dropout_drop_probability)

    optimizer = torch.optim.SGD(net.trainable_params(),
                                lr=learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)

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
        model_name=model_name,
        dropout_drop_probability=dropout_drop_probability,
        weight_decay=weight_decay,
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


def fit_classifier(net,
                   optimizer,
                   X_train,
                   y_train,
                   epochs,
                   batch_size=32,
                   seed=None,
                   X_test=None,
                   y_test=None):
    metrics = {
        "loss": [],
        "weights_l2": [],
        "grad_l2": [],
        "train_accuracy": [],
    }
    if (X_test is not None) and (y_test is not None):
        metrics["test_accuracy"] = []

    for epoch in range(epochs):
        epoch_seed = seed + epoch if seed is not None else None
        batches = batchify(X_train, y_train, batch_size, seed=epoch_seed)

        loss_per_batch = []
        for i_batch, (X_batch, y_batch) in enumerate(batches):
            # forward
            X_batch = torch.FloatTensor(X_batch)
            y_batch = torch.LongTensor(y_batch)
            probs = net.forward(X_batch)
            loss = cross_entropy_loss(probs, y_batch)

            # gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_per_batch.append(loss.item())

        # metrics & progress report
        metrics["loss"].append(np.mean(loss_per_batch))
        weights = optimizer.param_groups[0]["params"]
        flat_weights = torch.cat([torch.flatten(x) for x in weights])
        weights_l2 = torch.sqrt(torch.sum(flat_weights ** 2))
        flat_grad = torch.cat([torch.flatten(x.grad) for x in weights])
        grad_l2 = torch.sqrt(torch.sum(flat_grad ** 2))
        metrics["weights_l2"].append(weights_l2.item())
        metrics["grad_l2"].append(grad_l2.item())

        pred_train = net.predict(torch.FloatTensor(X_train))
        accuracy_train = (pred_train == torch.LongTensor(y_train)).float().mean()
        metrics["train_accuracy"].append(accuracy_train.item())

        report = f"epoch {epoch + 1}/{epochs}" \
                 f"   loss: {loss.item():.3f}" \
                 f"   train_accuracy: {accuracy_train.item():.2f}"
        if (X_test is not None) and (y_test is not None):
            pred_test = net.predict(torch.FloatTensor(X_test))
            accuracy_test = (pred_test == torch.LongTensor(y_test)).float().mean()
            metrics["test_accuracy"].append(accuracy_test.item())
            report += f"   test_accuracy: {accuracy_test.item():.2f}"
        report += f"   weights_l2: {weights_l2.item():.3f}" \
                  f"   grad_l2: {grad_l2.item():.3f}"
        print(report)

    return metrics


def save_model(net, metrics, hyper_params, model_name, save_dir):
    model_dir = osp.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    with open(osp.join(model_dir, "net.pkl"), 'wb') as f:
        pickle.dump(net, f)

    with open(osp.join(model_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(osp.join(model_dir, "hyper_params.json"), 'w') as f:
        json.dump(hyper_params, f, indent=2)
