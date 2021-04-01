import oneflow.nn as nn
import oneflow as flow
import math
import oneflow.nn as nn

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def __init__(self):
        super().__init__()
        self.matmul = flow.tmp.MatMul()
 
    def forward(self, query, key, value, mask=None, dropout=None): # q k v shape >> flow.Size([16, 8, 20, 32])
        # TODO: Tensor.masked_fill; flow.math.softmax; flow.matmul(dim>2的多维情况下报错)
        print("Enter Attention module >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  forward()")

        # scores = flow.matmul(query, key.transpose(-2, -1)) \
        #          / math.sqrt(query.size()[-1])
        x = flow.tmp.transpose(key, perm=[0, 1, 3, 2])
        scores = self.matmul(query, x) / math.sqrt(query.size()[-1])

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = flow.math.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return self.matmul(p_attn, value), p_attn
