import oneflow.experimental.nn as nn
import oneflow.experimental as flow
import numpy as np


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a_2 = flow.ones(features, dtype=flow.float32)
        self.b_2 = flow.zeros(features, dtype=flow.float32)
        # TODO: nn.Parameter 放开会报错
        # self.a_2 = nn.Parameter(flow.ones(features, dtype=flow.float32))
        # self.b_2 = nn.Parameter(flow.zeros(features, dtype=flow.float32))


    def forward(self, x): # x input/output >> shape flow.Size([16, 20, 256])
        mean = x.mean(2, keepdim=True)   # TODO: 原参数为x.mean(-1, keepdim=True)，需对齐torch
        std = x.std(dim=2, keepdim=True) # 同上， 原参数为dim=-1，此处不支持，需对齐pytorch
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
