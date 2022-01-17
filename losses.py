import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
import torchmetrics


class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0, pos_weight=None):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight

    @staticmethod
    def _smooth(targets, n_labels, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(
            targets, inputs.size(-1), self.smoothing)
        loss = F.binary_cross_entropy_with_logits(
            inputs, targets, self.weight, pos_weight=self.pos_weight)
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


class RMSELoss(torchmetrics.Metric):

    def __init__(self):
        super().__init__()
        self.add_state("sum_squared_errors",
                       torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_observations", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        self.sum_squared_errors += torch.sum((preds - target) ** 2)
        self.n_observations += preds.numel()

    def compute(self):
        return torch.sqrt(self.sum_squared_errors / self.n_observations)
