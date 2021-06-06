import oneflow.experimental.nn as nn
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
        # self.activation = nn.GELU() # NOTE: torch原实现为调用GELU()、此处直接可用已有module(nn.GELU)代替

    def forward(self, x):   # x.shape >> flow.Size([16, 20, 256])
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    
