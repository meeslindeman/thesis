import torch.optim as optim
from geoopt import ManifoldParameter
from geoopt.optim import RiemannianAdam


class DualOptimizer:
    """
    AdamW for Euclidean params + RiemannianAdam for ManifoldParameters
    """

    def __init__(self, opt_euc, opt_hyp):
        self.optimizers = [opt_euc, opt_hyp]

    def zero_grad(self, set_to_none=False):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()


def build_optimizer(model, cfg) -> DualOptimizer:
    manifold_params = [p for p in model.parameters() if isinstance(p, ManifoldParameter)]
    euclidean_params = [p for p in model.parameters() if not isinstance(p, ManifoldParameter)]

    opt_euc = optim.AdamW(euclidean_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    opt_hyp = RiemannianAdam(manifold_params, lr=cfg.hyp_lr, weight_decay=cfg.hyp_weight_decay, stabilize=cfg.stabilize)
    return DualOptimizer(opt_euc, opt_hyp)
