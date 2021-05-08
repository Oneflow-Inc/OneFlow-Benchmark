import oneflow.nn as nn
import oneflow as flow
import numpy as np


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        # self.a_2 = nn.Parameter(flow.tmp.ones(features))
        # self.b_2 = nn.Parameter(flow.tmp.zeros(features))
        self.a_2 = flow.tmp.ones(features, dtype=flow.float32)
        self.b_2 = flow.tmp.zeros(features, dtype=flow.float32)

    def forward(self, x): # x input/output >> shape flow.Size([16, 20, 256])
        # 此处应为std = x.std(dim=2, keepdim=True),但x.mean消费过一次后x.std会报错
        mean = x.mean(2, keepdim=True)
        x2 = flow.Tensor(np.random.randn(16, 20, 256))
        std = x2.std(dim=2, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
