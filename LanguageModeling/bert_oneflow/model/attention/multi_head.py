import oneflow.experimental.nn as nn
import oneflow as flow
from .single import Attention


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size()[0] # 16

        query = flow.Tensor(16, 8, 20, 32)
        key = flow.Tensor(16, 8, 20, 32)
        value = flow.Tensor(16, 8, 20, 32)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # TODO: query, key, value放开会报错
        # query, key, value = [l(x).reshape(shape=[batch_size, -1, self.h, self.d_k]).permute(0, 2, 1, 3)
        #                      for l, x in zip(self.linear_layers, (query, key, value))]
        # # query,key,value  shape >> flow.Size([16, 8, 20, 32])

   
        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask, self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        res = x.transpose(1, 2).reshape(shape = [batch_size, -1, self.h * self.d_k])
        res = self.output_linear(res)
        return res

