import oneflow.nn as nn
import oneflow as flow
import math

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return x
        # TODO: flow.tanh flow.pow
        # return 0.5 * x * (1 + flow.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * flow.pow(x, 3))))
