import oneflow.experimental.nn as nn
import oneflow.experimental as flow
import math

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        tmp = flow.Tensor([math.sqrt(2 / math.pi)], device=x.device)
        return 0.5 * x * (1 + flow.tanh(tmp) * (x + 0.044715 * x.pow(3.0)))
