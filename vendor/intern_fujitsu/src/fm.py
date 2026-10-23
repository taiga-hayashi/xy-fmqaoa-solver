import torch
import torch.nn as nn
import numpy as np
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TorchFM(nn.Module):
    def __init__(self, d: int, k: int):
        super().__init__()
        self.V = nn.Parameter(torch.randn(d, k, device=device))
        self.lin = nn.Linear(d, 1).to(device)
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, active_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if active_idx is not None:
            E = self.V[active_idx]  # (B,F,K)
            summed = torch.sum(E, dim=1)
            term1 = torch.sum(summed * summed, dim=1)
            term2 = torch.sum(E * E, dim=(1, 2))
            interaction = 0.5 * (term1 - term2)

            lin_w = self.lin.weight.view(-1)
            linear = lin_w[active_idx].sum(dim=1) + self.lin.bias.view(1)
            return interaction + linear.view(-1)

        term1_inner = x @ self.V
        term1 = torch.sum(term1_inner * term1_inner, dim=1, keepdim=True)
        term2_inner = x.pow(2) @ self.V.pow(2)
        term2 = torch.sum(term2_inner, dim=1, keepdim=True)
        interaction = 0.5 * (term1 - term2)
        return (interaction + self.lin(x)).view(-1)


def train_factorization_machine(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    d: int,
    k: int = 2,
    epochs: int = 120,
    learning_rate: float = 0.1,
    *,
    optimizer_name: str = "Adam",
    weight_decay: float = 0.0,
) -> TorchFM:
    """Fit a fresh FM with MSE; defaults preserve the original Adam policy.

    AdamW applies decoupled decay to all parameters, including the bias.
    No scheduler or early stopping is applied.
    """
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs must be positive and learning_rate must be greater than zero.")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative.")
    optimizers = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}
    if optimizer_name not in optimizers:
        raise ValueError("optimizer_name must be Adam or AdamW.")
    model = TorchFM(d=d, k=k)
    optimizer = optimizers[optimizer_name](
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    return model
