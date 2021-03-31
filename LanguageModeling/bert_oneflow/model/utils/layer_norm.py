import oneflow.nn as nn
import oneflow as flow


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        # self.a_2 = nn.Parameter(flow.ones(features))
        # self.b_2 = nn.Parameter(flow.zeros(features))
        m0 = flow.Zeros()
        m1 = flow.Ones()
        self.a_2 = m1(features)
        self.b_2 = m0(features)

    def forward(self, x):
        # TODO : Tensor.mean; Tensor.std
        # mean = x.mean(-1, keepdim=True)
        # std = x.std(-1, keepdim=True)
        # return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return x
