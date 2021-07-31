import torch
from torch import nn


class LabelSmoothingLoss(nn.Module):
    """Cross Entropy with Label Smoothing.
    
    Attributes:
        num_classes (int): Number of target classes.
        smoothing (float, optional): Smoothing fraction constant, in the range (0.0, 1.0). Defaults to 0.1.
        dim (int, optional): Dimension across which to apply loss. Defaults to -1.
    """

    def __init__(self, num_classes: int, smoothing : float = 0.1, dim : int = -1):
        """Initializer for LabelSmoothingLoss.

        Args:
            num_classes (int): Number of target classes.
            smoothing (float, optional): Smoothing fraction constant, in the range (0.0, 1.0). Defaults to 0.1.
            dim (int, optional): Dimension across which to apply loss. Defaults to -1.
        """
        super().__init__()
        
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = num_classes
        self.dim = dim

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): Model predictions, of shape (batch_size, num_classes).
            target (torch.Tensor): Target tensor of shape (batch_size).

        Returns:
            torch.Tensor: Loss.
        """

        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))