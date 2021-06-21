import oneflow.experimental as flow
import oneflow.experimental.nn as nn
import numpy as np
import math


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = flow.zeros(size=(max_len, d_model), requires_grad=False)

        position = flow.arange(0, max_len, dtype=flow.float).unsqueeze(1)
        div_term = (flow.arange(0, d_model, 2, dtype=flow.float)* -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = flow.sin(position * div_term)
        pe[:, 1::2] = flow.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', flow.Tensor(pe))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)  # shape >> flow.Size([16, 20, 256])
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)      # shape >> flow.Size([1, 20, 256])
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)       # shape >> flow.Size([16, 20, 256])


        self.dropout = flow.nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label): # sequence/segment_label .shape >>> flow.Size([16, 20])
        

        x = self.segment(segment_label) + self.token(sequence) + self.position(sequence)


        return self.dropout(x)
