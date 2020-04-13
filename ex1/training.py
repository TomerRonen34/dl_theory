import torch
from utils import batchify
from losses import cross_entropy_loss


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

        # metrics & progress report
        metrics["loss"].append(loss.item())
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
