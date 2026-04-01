import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_hyperbolic.lorentz import Lorentz
from vit_hyperbolic.lorentz_utils import LorentzLayerNorm
from layers.LLinear import LorentzFullyConnected


torch.manual_seed(0)


class LorentzMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.fc1 = LorentzFullyConnected(
            in_features=config["hidden_size"] + 1,
            out_features=config["mlp_dim"] + 1,
            manifold=config["manifold"],
            reset_params=config["reset_params"],
            a_default=config["a_default"],
        )
        self.ln = LorentzLayerNorm(
            manifold=config["manifold"],
            dim=config["mlp_dim"],
        )
        self.fc2 = LorentzFullyConnected(
            in_features=config["mlp_dim"] + 1,
            out_features=config["hidden_size"] + 1,
            manifold=config["manifold"],
            reset_params=config["reset_params"],
            a_default=config["a_default"],
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln(x)
        x = self.fc2(x)
        return x


def manifold_check(manifold: Lorentz, x: torch.Tensor, name: str = "x"):
    with torch.no_grad():
        k = manifold.k().to(x.device, x.dtype)
        mink = manifold.inner(x, x, keepdim=True)
        target = -1.0 / k
        max_err = (mink - target).abs().max().item()
        min_time = x[..., 0].min().item()
        print(f"{name}: max |<x,x>_L + 1/k| = {max_err:.3e}, min x0 = {min_time:.3e}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifold = Lorentz(k=1.0).to(device)

    config = {
        "manifold": manifold,
        "hidden_size": 16,
        "mlp_dim": 32,
        "reset_params": "lorentz_kaiming",
        "a_default": 0.0,
    }

    model = LorentzMLP(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 8
    seq_len = 4
    tangent_dim = config["hidden_size"]

    print("Starting minimal LayerNorm run test")

    for step in range(10):
        optimizer.zero_grad()

        # Build random tangent vectors at the origin, then map to Lorentz manifold
        x_tan = 0.1 * torch.randn(batch_size, seq_len, tangent_dim, device=device)
        x = manifold.expmap0(x_tan)

        # Forward pass
        y = model(x)

        # Simple dummy loss that stays in ambient coordinates
        # Enough to test forward/backward/optimizer step.
        loss = y[..., 1:].pow(2).mean()

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        loss.backward()
        optimizer.step()

        manifold_check(manifold, x, name=f"input step {step}")
        manifold_check(manifold, y, name=f"output step {step}")

        print(f"step {step:02d} | loss = {loss.item():.6f}")

    print("LayerNorm minimal run completed successfully.")


if __name__ == "__main__":
    main()