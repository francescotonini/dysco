import torch

from src.utils.logger import get_logger

log = get_logger(__name__)


class Classifier(torch.nn.Module):
    def __init__(
        self,
        tau: float = 1.0,
    ):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        query: torch.Tensor,
        supports: torch.Tensor,
        op: str = "non_rare",
    ):
        assert op in ["rare", "non_rare"]

        query = query / query.norm(dim=-1, keepdim=True)
        supports = supports / supports.norm(dim=-1, keepdim=True)

        if supports.dim() == 2:
            supports = supports.unsqueeze(0)

        Q, _ = query.shape
        N, C, _ = supports.shape

        if op == "rare":
            similarity = query @ supports.reshape(N * C, -1).T
            similarity = similarity / self.tau if self.tau != 1.0 else similarity
            logits = similarity.reshape(Q, N, C)
            logits = logits.mean(dim=1)
        elif op == "non_rare":
            supports = supports.mean(dim=0)
            supports = supports / supports.norm(dim=-1, keepdim=True)
            similarity = query @ supports.T
            similarity = similarity / self.tau if self.tau != 1.0 else similarity
            logits = similarity

        return logits
