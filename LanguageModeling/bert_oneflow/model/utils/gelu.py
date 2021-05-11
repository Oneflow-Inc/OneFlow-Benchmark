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
        # return 0.5 * x * (1 + flow.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * flow.pow(x, 3))))
        xx = flow.Tensor([math.sqrt(2 / math.pi)])
        return 0.5 * x * (1 + xx.tanh() * (x + 0.044715 * x.pow(3)))