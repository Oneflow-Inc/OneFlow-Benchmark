import oneflow as flow
import oneflow.nn as nn
import numpy as np


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        # pe = flow.zeros(max_len, d_model).float()
        pe = flow.Tensor(np.zeros((max_len, d_model)))
        # position = flow.arange(0, max_len).float().unsqueeze(1)
        a = np.arange(0, max_len, dtype=np.float32).astype(np.float32)
        position = flow.Tensor(np.expand_dims(a, 1))
        # div_term = (flow.arange(0, d_model, 2).float() * -(flow.math.log(10000.0) / d_model)).exp()
        b = np.exp((np.log(10000.0) / d_model))
        c = np.arange(0, d_model, 2).astype(np.float32)
        div_term = flow.Tensor(b * c)


        pe = pe.numpy()
        pe[:, 0::2] = flow.sin(position * div_term).numpy()
        pe[:, 1::2] = flow.cos(position * div_term).numpy()

        # pe = pe.unsqueeze(0)
        pe = np.expand_dims(pe, 0)
        pe = flow.Tensor(pe)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size()[1]]


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

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)  # shape >> flow.Size([16, 20, 256])
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)      # shape >> flow.Size([1, 20, 256])
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)       # shape >> flow.Size([16, 20, 256])


        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label): # shape >>> flow.Size([16, 20])
        print("Enter BERTEmbedding module >>>>>>>>>>>>>>>>>>>>>>>>>> forward()...")
        x = flow.Tensor(16, 20, 256) + flow.Tensor(1, 20, 256) + flow.Tensor(16, 20, 256)
        # x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)
