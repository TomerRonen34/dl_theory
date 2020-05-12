import torch
from utils import device


def cross_entropy_loss(probs, targets):
    """
    :param probs: float[samples, classes]
    :param targets: int[samples]
    """
    eps = torch.Tensor([1e-10]).to(device=device)
    probs = torch.max(probs, eps)

    correct_class_prob = probs[range(len(probs)), targets]
    loss = - torch.log(correct_class_prob)
    loss = torch.mean(loss)

    return loss


def MSE_loss(preds, targets):
    loss = (preds - targets) ** 2
    loss = loss.mean()
    return loss


def _test_cross_entropy_loss():
    num_samples = 1000
    num_classes = 10

    probs = torch.rand(num_samples, num_classes)
    probs = probs / probs.sum(dim=1, keepdim=True)
    targets = torch.randint(low=0, high=num_classes, size=(num_samples,))

    torch_loss = torch.nn.NLLLoss()(torch.log(probs), targets)
    my_loss = cross_entropy_loss(probs, targets)

    if torch.allclose(torch_loss, my_loss):
        print("test_cross_entropy_loss:  Success!")
    else:
        print("test_cross_entropy_loss:  Failure")


def _test_MSE_loss():
    n = 1000
    preds = torch.rand(n)
    targets = torch.rand(n)

    torch_loss = torch.nn.MSELoss()(preds, targets)
    my_loss = MSE_loss(preds, targets)

    if torch.allclose(torch_loss, my_loss):
        print("test_MSE_loss:  Success!")
    else:
        print("test_MSE_loss:  Failure")


if __name__ == '__main__':
    _test_cross_entropy_loss()
    _test_MSE_loss()
