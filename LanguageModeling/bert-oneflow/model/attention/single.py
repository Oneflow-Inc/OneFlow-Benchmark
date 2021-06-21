import oneflow.experimental.nn as nn
import oneflow.experimental as flow
import math
import numpy as np

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def __init__(self):
        super().__init__()
        self.softmax = flow.nn.Softmax(dim = -1)
 
    def forward(self, query, key, value, mask=None, dropout=None): # q k v shape >> flow.Size([16, 8, 20, 32])
        x = flow.matmul(query, key.transpose(-2, -1))
        scores = x / math.sqrt(query.size(-1))

        if mask is not None:
            # scores = scores.masked_fill(mask == 0, -1e9)
            mask = flow.Tensor((mask.numpy() == 0).astype(np.int8), dtype=flow.int, device=scores.device)
            scores = scores.masked_fill(mask, -1e9)

        p_attn = self.softmax(scores)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return flow.matmul(p_attn, value), p_attn
