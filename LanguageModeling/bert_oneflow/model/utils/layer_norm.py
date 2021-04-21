import oneflow.nn as nn
import oneflow as flow


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        # self.a_2 = nn.Parameter(flow.tmp.ones(features))
        # self.b_2 = nn.Parameter(flow.tmp.zeros(features))
        m0 = flow.Zeros(dtype=flow.float32)
        m1 = flow.Ones(dtype=flow.float32)
        self.a_2 = m1(features)
        self.b_2 = m0(features)

    def forward(self, x): # x input/output >> shape flow.Size([16, 20, 256])
        print("LayerNorm >> x.shape >>>>>>>>>>>>>", x.shape)
        mean = x.mean(2, keepdim=True)
        std = x.npstd(2)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
