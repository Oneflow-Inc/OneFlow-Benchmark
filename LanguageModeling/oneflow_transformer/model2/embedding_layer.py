# Copyright 2018 MLBenchmark Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of embedding layer with shared weights."""

import oneflow as flow
import oneflow.typing as tp
import numpy as np
from oneflow_transformer.model2 import model_utils


class EmbeddingSharedWeights(object):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(self, vocab_size, hidden_size):
        super(EmbeddingSharedWeights, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.Build_EmbeddingLayer(vocab_size=self.vocab_size,
                                  embedding_size=self.hidden_size,
                                  word_embedding_name="Embedding_Layer")

    def Build_EmbeddingLayer(self,
                             vocab_size,
                             embedding_size=128,
                             word_embedding_name="Embedding_Layer"):
        """
        Build a Embedding Layer
        :param input_ids_blob:The input ID Blob
        :param vocab_size: The input Vocab size
        :param embedding_size: The embedding Size
        :param initializer_range: The range of Initializer, Use flow.truncated_normal
        :param word_embedding_name: The name of Embedding variable
        :return: The output and the Embedding table.
        """
        self.embedding_table = flow.get_variable(name=word_embedding_name + "_Embed",
                                                 shape=[vocab_size, embedding_size],
                                                 dtype=flow.float32,
                                                 initializer=flow.random_normal_initializer(0,
                                                                                            self.hidden_size ** -0.5))

    def __call__(self, x):
        """
        Get embeddings of x
        :param x: An flow.int64 Tensor with shape [batchsize, length]
        :return: embeddings: float32 tensor with shape [batch_size, length, embedding_size]
                 padding: float32 tensor with shape [batch_size, length] indicating the
                 locations of the padding tokens in x.
        """
        with flow.scope.namespace("embedding"):
            embeddings = flow.gather(self.embedding_table, x, axis=0)

            # Scale embedding by the sqrt of the hidden size
            embeddings *= self.hidden_size ** 0.5

            # Create binary array of size [batch_size, length]
            # where 1 = padding, 0 = not padding
            padding = model_utils.get_padding(x)

            # Set all padding embedding values to 0
            embeddings *= flow.expand_dims(1 - padding, -1)
            return embeddings

    def linear(self, x):
        """ Computes logits by running x through a linear layer.

            Args:
              x: A float32 tensor with shape [batch_size, length, hidden_size]
            Returns:
              float32 tensor with shape [batch_size, length, vocab_size].
        """
        with flow.scope.namespace("presoftmax_linear"):
            batch_size = x.shape[0]
            length = x.shape[1]

            x = flow.reshape(x, [-1, self.hidden_size])
            logits = flow.matmul(x, self.embedding_table, transpose_b=True)

            return flow.reshape(logits, [batch_size, length, self.vocab_size])


# test
# @flow.global_function()
# def embed(x: tp.Numpy.Placeholder(shape=(1, 3), dtype=flow.int32)) -> tp.Numpy:
#     with flow.scope.namespace("multi"):
#         Embedlayer = EmbeddingSharedWeights(vocab_size=50, hidden_size=10)
#
#         out = Embedlayer(x)
#
#         return out

# if __name__ == "__main__":
#     check = flow.train.CheckPoint()
#     check.init()
#
#     x = np.array([[1, 1, 2]]).astype(np.int32)
#
#     out = embed(x)
#     print(out.shape)
#     print(out)
