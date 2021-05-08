import oneflow.nn as nn

from .transformer import TransformerBlock
from .embedding import BERTEmbedding
import numpy as np
import oneflow as flow


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        
    def forward(self, x, segment_info): # x.shape >> flow.Size([16, 20])
        print("Enter BERT module >>>>>>>>>>>>>>>>>>>>>>>>>> forward()...")
        # attention masking for padded token
        
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  # shape >> flow.Size([16, 1, 20, 20])
        # mask = mask.repeat(1, 8, 1, 1) # shape >> flow.Size([16, 8, 20, 20])

        # aaa = (x > 0).unsqueeze(1)
        aaa = flow.Tensor(16, 1, 20, dtype=flow.float)
        # TODO 1.Tensor > 0有问题，暂时替换为：x > flow.tmp.zeros(size = (16, 20), dtype=flow.float32)
        # 2. Tensor.repeat 目前不支持flow.int类型的参数
        mask = aaa.repeat(sizes=(1, x.size()[1], 1)).unsqueeze(1).repeat(sizes=(1, 8, 1, 1))
    
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        print("Enter TransformerBlock module >>>>>>>>>>>>>>>>>>>>>>>>>> for loop...")
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
