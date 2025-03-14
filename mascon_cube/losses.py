import torch


def normalized_l1_loss(predicted: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Computes the L1 loss between labels and predicted for some scaling parameter c

    Args:
        predicted (torch.Tensor): model predictions
        labels (torch.Tensor): ground truth labels

    Returns:
        [torch.Tensor]: loss
    """
    c = torch.sum(torch.mul(labels, predicted)) / torch.sum(torch.pow(predicted, 2))
    return torch.sum(torch.abs(torch.sub(labels, c * predicted))) / len(labels)


def l1_loss(predicted: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Computes th L1 loss between labels and predicted

    Args:
        predicted (torch.Tensor): model predictions
        labels (torch.Tensor): ground truth labels

    Returns:
        [torch.Tensor]: loss
    """
    return torch.sum(torch.abs(torch.sub(labels, predicted))) / len(labels)
