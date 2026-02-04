import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from hyperbolic.lorentz import Lorentz
from layers.LLinear import LorentzFullyConnected

fc = LorentzFullyConnected(
    in_features=9,          
    out_features=9,     
    manifold=Lorentz(0.1), 
    reset_params="lorentz_kaiming", 
    a_default=0.0
)

manifold = Lorentz(0.1)
x_euc = torch.randn(4, 8)
x_lor = manifold.expmap0(x_euc)

out = fc(x_lor)
print(out)