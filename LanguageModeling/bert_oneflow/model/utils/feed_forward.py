import oneflow.nn as nn
import oneflow as flow
from .gelu import GELU


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # d_model,d_ff >>  256,1024
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):   # x.shape >> flow.Size([16, 20, 256])
        return self.dropout(self.activation(x))
        # return self.w_2(self.dropout(self.activation(self.w_1(x))))
    
