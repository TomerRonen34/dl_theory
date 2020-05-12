from utils import batchify
from losses import cross_entropy_loss
import os
import os.path as osp
import torch
import pickle
import json
import math
from utils import RandomStateContextManager
from fully_connected import FullyConnectedClassifier
from typing import *


def train_and_eval_fully_connected_model(X_train, y_train, X_test, y_test, class_names, save_dir, model_name,
                                         weight_decay=0., dropout_drop_probability=0.,
                                         optimizer_type="sgd", learning_rate=0.001, sgd_momentum=0.9,
                                         init_type="xavier", init_gaussian_std=0.01,
                                         hidden_size=256, num_hidden_layers=1, activation="relu",
                                         epochs=100, batch_size=32, seed=34, num_reports=5):
    with RandomStateContextManager(seed):
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

        if optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(net.trainable_params(),
                                        lr=learning_rate,
                                        momentum=sgd_momentum,
                                        weight_decay=weight_decay)
        elif optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(net.trainable_params(),
                                         lr=learning_rate)
        else:
            raise ValueError('optimizer_type must be one of ["sgd", "adam]')

        metrics = fit_classifier(net,
                                 optimizer,
                                 X_train,
                                 y_train,
                                 epochs,
                                 batch_size,
                                 seed,
                                 X_test,
                                 y_test,
                                 num_reports)

        hyper_params = dict(
            model_name=model_name,
            optimizer_type=optimizer_type,
            dropout_drop_probability=dropout_drop_probability,
            weight_decay=weight_decay,
            init_gaussian_std=init_gaussian_std,
            learning_rate=learning_rate,
            sgd_momentum=sgd_momentum,
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
                   y_test=None,
                   num_reports=10):
    metrics = {
        "train_loss": [],
        "train_accuracy": []
    }
    if (X_test is not None) and (y_test is not None):
        metrics.update({
            "test_loss": [],
            "test_accuracy": []
        })
    metrics.update({
        "weights_l2": [],
        "grad_l2": []
    })

    for curr_epoch in range(1, epochs + 1):
        epoch_seed = seed + curr_epoch if seed is not None else None
        batches = batchify(X_train, y_train, batch_size, seed=epoch_seed)

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

        # update metrics
        weights_l2, grad_l2 = _calculate_l2_norms(optimizer)
        metrics["weights_l2"].append(weights_l2)
        metrics["grad_l2"].append(grad_l2)

        train_loss, train_accuracy = eval_classifier(net, X_train, y_train)
        metrics["train_loss"].append(train_loss)
        metrics["train_accuracy"].append(train_accuracy)

        if (X_test is not None) and (y_test is not None):
            test_loss, test_accuracy = eval_classifier(net, X_test, y_test)
            metrics["test_loss"].append(test_loss)
            metrics["test_accuracy"].append(test_accuracy)

        if _should_report_progress(curr_epoch, epochs, num_reports):
            _report_progress(curr_epoch, epochs, metrics)

    return metrics


def _should_report_progress(curr_epoch: int,
                            epochs: int,
                            num_reports: int):
    should_report_at_all = (num_reports > 0)
    report_every = math.ceil(epochs / num_reports)
    is_reportable_epoch = ((curr_epoch % report_every == 0)
                           or (curr_epoch == 1)
                           or (curr_epoch == epochs))
    should_report = should_report_at_all and is_reportable_epoch
    return should_report


def _report_progress(curr_epoch: int,
                     epochs: int,
                     metrics: Dict[str, List[float]]):
    curr_metrics = {name: values[-1] for name, values in metrics.items()}
    report = f"epoch: {curr_epoch}/{epochs} | "
    report += " | ".join([f"{name}: {value:.2f}" for name, value in curr_metrics.items()])
    print(report)


def _calculate_l2_norms(optimizer) -> Tuple[float, float]:
    weights = optimizer.param_groups[0]["params"]
    flat_weights = torch.cat([torch.flatten(x) for x in weights])
    weights_l2 = torch.sqrt(torch.sum(flat_weights ** 2)).item()
    flat_grad = torch.cat([torch.flatten(x.grad) for x in weights])
    grad_l2 = torch.sqrt(torch.sum(flat_grad ** 2)).item()
    return weights_l2, grad_l2


def eval_classifier(net, X, y):
    X, y = torch.FloatTensor(X), torch.LongTensor(y)
    probs = net.predict_proba(X)
    pred_labels = probs.argmax(axis=1)

    loss = cross_entropy_loss(probs, y).item()
    accuracy = (pred_labels == y).float().mean().item()
    return loss, accuracy


def save_model(net, metrics, hyper_params, model_name, save_dir):
    model_dir = osp.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    with open(osp.join(model_dir, "net.pkl"), 'wb') as f:
        pickle.dump(net, f)

    with open(osp.join(model_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(osp.join(model_dir, "hyper_params.json"), 'w') as f:
        json.dump(hyper_params, f, indent=2)
