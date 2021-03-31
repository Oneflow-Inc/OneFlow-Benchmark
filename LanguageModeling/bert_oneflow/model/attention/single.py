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
        self.matmul = flow.MatMul()
 
    def forward(self, query, key, value, mask=None, dropout=None): # q k v shape >> flow.Size([16, 8, 20, 32])
        # TODO: Tensor.masked_fill; flow.math.softmax; flow.matmul(已有,需要对齐行为,dim>2的多维情况下似乎存在问题)
        print("Enter Attention module >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  forward()")

        # scores = flow.matmul(query, key.transpose(-2, -1)) \
        #          / math.sqrt(query.size()[-1])
        aa = flow.tmp.transpose(key, perm=[0, 1, 3, 2])
        print("aa.shape >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", aa.shape)
        bb = self.matmul(query, aa)
        print("bb.shape >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", bb.shape)
        scores = bb / math.sqrt(query.size()[-1])

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = flow.math.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return self.matmul(p_attn, value), p_attn
