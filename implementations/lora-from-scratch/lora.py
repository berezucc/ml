"""LoRA: wrap a frozen linear with two low-rank trainable matrices."""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear (or HF Conv1D) and add B @ A as a low-rank update."""

    def __init__(self, base: nn.Module, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        # HF GPT-2 Conv1D stores weight as (in, out); nn.Linear stores (out, in).
        if isinstance(base, nn.Linear):
            in_features, out_features = base.in_features, base.out_features
        else:
            in_features, out_features = base.weight.shape

        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.empty(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        # B starts at zero, so the LoRA branch is a no-op at init.
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling


def apply_lora(model: nn.Module, target_names, r: int, alpha: float) -> int:
    """Swap every submodule whose name ends in one of target_names with a LoRALinear.

    Returns the number of layers wrapped.
    """
    targets = tuple(target_names)
    swapped = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in targets and not isinstance(child, LoRALinear):
                setattr(parent, child_name, LoRALinear(child, r=r, alpha=alpha))
                swapped += 1
    return swapped


def trainable_param_counts(model: nn.Module):
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


if __name__ == "__main__":
    torch.manual_seed(0)
    base = nn.Linear(64, 128)
    wrapped = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(2, 64)
    # B is zero at init => output must equal the base linear exactly.
    assert torch.allclose(wrapped(x), base(x), atol=1e-6)
    # Only A and B should be trainable.
    trainable = [n for n, p in wrapped.named_parameters() if p.requires_grad]
    assert set(trainable) == {"A", "B"}, trainable
    print("lora.py self-checks passed")
