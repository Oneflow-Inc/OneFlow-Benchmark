from collections import OrderedDict
import oneflow as flow
import oneflow.nn as nn
from typing import Any


__all__ = ["WideAndDeep", "wide_and_deep"]


class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size, split_axis=0):
        # TODO: name and split_axis for weight
        super(Embedding, self).__init__(vocab_size, embed_size, padding_idx=0)
        for param in self.parameters():
            nn.init.uniform_(param, a=-0.05, b=0.05)

    def forward(self, indices):
        # indices = flow.parallel_cast(indices, distribute=flow.distribute.broadcast())
        embedding = flow._C.gather(self.weight, indices, axis=0)
        return embedding.view(-1, embedding.shape[-1] * embedding.shape[-2])


class Dense(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, dropout_rate: float = 0.5
    ) -> None:
        super(Dense, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
        )
        for name, param in self.named_parameters():
            if name.endswith("weight"):
                nn.init.xavier_uniform_(param)
            elif name.endswith("bias"):
                nn.init.zeros_(param)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x = self.features(x)
        return x

class WideAndDeep(nn.Module):
    def __init__(self, FLAGS) -> None:
        super(WideAndDeep, self).__init__()
        self.FLAGS = FLAGS
        self.wide_embedding = Embedding(vocab_size=FLAGS.wide_vocab_size, embed_size=1)
        self.deep_embedding = Embedding(
            vocab_size=FLAGS.deep_vocab_size,
            embed_size=FLAGS.deep_embedding_vec_size,
            split_axis=1,
        )
        deep_feature_size = (
            FLAGS.deep_embedding_vec_size * FLAGS.num_deep_sparse_fields
            + FLAGS.num_dense_fields
        )
        self.linear_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"fc{i}",
                        Dense(
                            deep_feature_size if i == 0 else FLAGS.hidden_size,
                            FLAGS.hidden_size,
                            FLAGS.deep_dropout_rate,
                        ),
                    )
                    for i in range(FLAGS.hidden_units_num)
                ]
            )
        )
        self.deep_scores = nn.Linear(FLAGS.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, dense_fields, wide_sparse_fields, deep_sparse_fields
    ) -> flow.Tensor:
        wide_embedding = self.wide_embedding(wide_sparse_fields)
        wide_scores = flow.sum(wide_embedding, dim=1, keepdim=True)
        deep_embedding = self.deep_embedding(deep_sparse_fields)
        deep_features = flow.cat([deep_embedding, dense_fields], dim=1)
        deep_features = self.linear_layers(deep_features)
        deep_scores = self.deep_scores(deep_features)
        return self.sigmoid(wide_scores + deep_scores)


def wide_and_deep(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> WideAndDeep:
    r"""WideAndDeep model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1606.07792>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on WideAndDeep
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = WideAndDeep(**kwargs)
    return model
