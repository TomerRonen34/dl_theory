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
                   report_progress=True,
                   X_test=None,
                   y_test=None,
                   print_gradient_amplitudes=False):
    for epoch in range(epochs):
        epoch_seed = seed + epoch if seed is not None else None
        batches = batchify(X_train, y_train, batch_size, seed=epoch_seed)

        num_batches = len(batches)
        epoch_loss = 0.
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

            if print_gradient_amplitudes:
                trainable_params = optimizer.param_groups[0]["params"]
                grads_amplitude = [param.grad.abs().mean().item() for param in trainable_params]
                print("grads_amplitude:", grads_amplitude)

            epoch_loss += loss.item()

        epoch_loss /= num_batches

        # report progress
        if report_progress:
            pred_labels = net.predict(torch.FloatTensor(X_train)).data.numpy()
            accuracy = (pred_labels == y_train).mean()
            report = f"epoch {epoch + 1}/{epochs}   loss: {epoch_loss:.3f}   train_accuracy: {accuracy:.2f}"
            if (X_test is not None) and (y_test is not None):
                pred_labels = net.predict(torch.FloatTensor(X_test)).data.numpy()
                accuracy = (pred_labels == y_test).mean()
                report += f"   test_accuracy: {accuracy:.2f}"
            print(report)
