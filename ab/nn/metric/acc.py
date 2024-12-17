import torch


def compute(outputs, targets):
    """
    Compute accuracy for classification tasks.
    :param outputs: Model predictions.
    :param targets: Ground truth labels.
    :return: Accuracy value.
    """
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct, total