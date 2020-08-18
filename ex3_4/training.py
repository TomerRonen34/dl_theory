from typing import List, Dict, Tuple, Iterable
import math

from pprint import pprint
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from utils import RandomStateContextManager


def fit_classifier(net: nn.Module,
                   trainloader: DataLoader,
                   testloader: DataLoader,
                   epochs: int = 10,
                   num_reports_per_epoch: int = 10,
                   random_seed: int = 34
                   ) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    with RandomStateContextManager(random_seed):
        optimizer = Adam(net.parameters())
        loss_object = CrossEntropyLoss()
        training_metrics = {
            "train_loss_per_batch": [],
            "train_accuracy_per_batch": []
        }

        batches_per_epoch = len(trainloader)
        for i_epoch in range(1, epochs + 1):
            for i_batch, (X_batch, y_batch) in enumerate(trainloader, start=1):
                # forward
                logits = net.forward(X_batch)
                loss = loss_object(logits, y_batch)
                train_accuracy = _compute_accuracy(logits, y_batch)

                # gradient step
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # update training metrics
                training_metrics["train_loss_per_batch"].append(loss.item())
                training_metrics["train_accuracy_per_batch"].append(train_accuracy)

                # progress report
                if _should_report_progress(
                        i_batch, num_reports_per_epoch, batches_per_epoch):
                    _report_progress(i_epoch, epochs, batches_per_epoch, i_batch, training_metrics)

        final_model_metrics = _calculate_final_model_metrics(net, trainloader, testloader)
        pprint(final_model_metrics)
        return final_model_metrics, training_metrics


def _calculate_final_model_metrics(net: nn.Module,
                                   trainloader: DataLoader,
                                   testloader: DataLoader
                                   ) -> Dict[str, float]:
    train_loss, train_accuracy, train_accuracy_per_class = _eval_classifier(net, trainloader)
    test_loss, test_accuracy, test_accuracy_per_class = _eval_classifier(net, testloader)
    num_params = _count_params(net)
    final_model_metrics = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "train_accuracy_per_class": train_accuracy_per_class,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_accuracy_per_class": test_accuracy_per_class,
        "num_params": num_params
    }
    return final_model_metrics


def _should_report_progress(i_batch: int,
                            num_reports_per_epoch: int,
                            batches_per_epoch: int):
    should_report_at_all = (num_reports_per_epoch > 0)
    report_every = math.ceil(batches_per_epoch / num_reports_per_epoch)
    is_reportable_batch = ((i_batch % report_every == 0)
                           or (i_batch == 1)
                           or (i_batch == batches_per_epoch))
    should_report = should_report_at_all and is_reportable_batch
    return should_report


def _report_progress(i_epoch: int,
                     epochs: int,
                     batches_per_epoch: int,
                     i_batch: int,
                     metrics: Dict[str, List[float]]):
    curr_metrics = {name: values[-1] for name, values in metrics.items()}
    report = f"epoch {i_epoch}/{epochs} batch {i_batch}/{batches_per_epoch}: | "
    report += " | ".join([f"{name}: {value:.2f}" for name, value in curr_metrics.items()])
    print(report)


def _compute_accuracy(logits, labels):
    pred_labels = logits.argmax(axis=1)
    accuracy = (pred_labels == labels).float().mean().item()
    return accuracy


def _eval_classifier(net, dataloader):
    net.eval()
    with torch.no_grad():
        logits, labels = _infer_net(net, dataloader)
        pred_labels = logits.argmax(axis=1)

        loss = CrossEntropyLoss()(logits, labels).item()
        accuracy = _accuracy(pred_labels, labels)
        accuracy_per_class = _accuracy_per_class(pred_labels, labels, dataloader.dataset.classes)
    net.train()
    return loss, accuracy, accuracy_per_class


def _infer_net(net, dataloader):
    logits_per_batch, labels_per_batch = zip(*[
        (net.forward(X_batch), y_batch)
        for X_batch, y_batch in dataloader
    ])
    logits = torch.cat(logits_per_batch)
    labels = torch.cat(labels_per_batch)
    return logits, labels


def _accuracy(pred_labels: torch.LongTensor,
              labels: torch.LongTensor
              ) -> float:
    accuracy = (pred_labels == labels).float().mean().item()
    return accuracy


def _accuracy_per_class(pred_labels: torch.LongTensor,
                        labels: torch.LongTensor,
                        classes: Iterable[str]
                        ) -> Dict[str, float]:
    accuracy_per_class = {}
    for i_class, class_name in enumerate(classes):
        class_pred_labels = (pred_labels == i_class).long()
        class_labels = (labels == i_class).long()
        accuracy = _accuracy(class_pred_labels, class_labels)
        accuracy_per_class[class_name] = accuracy
    return accuracy_per_class


def _count_params(module: nn.Module) -> int:
    flat_params = torch.cat([torch.flatten(x)
                             for x in module.parameters()])
    num_params = flat_params.numel()
    return num_params
