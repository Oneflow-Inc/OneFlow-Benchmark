"""
Build Attention Layer
author: zzk
"""
import oneflow as flow
import oneflow.typing as tp
import numpy as np


class Attention(object):
    """
    Build Multi-head Attention Layer
    """

    def __init__(self, hidden_size, num_heads, attention_dropout, train=True):
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of "
                             "heads.")

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.train = train

        self.use_bias = False

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.

        The tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.

        Args:
          x: A tensor with shape [batch_size, length, hidden_size]

        Returns:
          A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
        """
        with flow.scope.namespace("split_heads"):
            batch_size = x.shape[0]
            length = x.shape[1]

            # Calculate depth of last dimension after it has been split.
            depth = (self.hidden_size // self.num_heads)

            # Split the last dimension
            x = flow.reshape(x, [batch_size, length, self.num_heads, depth])

            # Transpose the result
            return flow.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.

        Args:
          x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

        Returns:
          A tensor with shape [batch_size, length, hidden_size]
        """
        with flow.scope.namespace("combine_heads"):
            batch_size = x.shape[0]
            length = x.shape[2]
            x = flow.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            return flow.reshape(x, [batch_size, length, self.hidden_size])

    def _build_dense(self, x, unit, name="dense_"):
        """
        Build Dense Layer
        :param x:
        :return:
        """
        self.init_range = 0.2
        self.init = flow.truncated_normal_initializer(self.init_range)
        self.reg = flow.regularizers.l2(0.01)

        return flow.layers.dense(x,
                                 units=unit,
                                 kernel_initializer=self.init,
                                 kernel_regularizer=self.reg,
                                 use_bias=self.use_bias,
                                 name=name + "w")

    def __call__(self, x, y, bias, cache=None):
        """Apply attention mechanism to x and y.

            Args:
              x: a tensor with shape [batch_size, length_x, hidden_size]
              y: a tensor with shape [batch_size, length_y, hidden_size]
              bias: attention bias that will be added to the result of the dot product.
              cache: (Used during prediction) dictionary with tensors containing results
                of previous attentions. The dictionary must have the items:
                    {"k": tensor with shape [batch_size, i, key_channels],
                     "v": tensor with shape [batch_size, i, value_channels]}
                where i is the current decoded length.

            Returns:
              Attention layer output with shape [batch_size, length_x, hidden_size]
            """
        # Linearly project the query (q), key (k) and value (v) using different
        # learned projections. This is in preparation of splitting them into
        # multiple heads. Multi-head attention uses multiple queries, keys, and
        # values rather than regular attention (which uses a single q, k, v).
        q = self._build_dense(x, self.hidden_size, name="dense_q")
        k = self._build_dense(y, self.hidden_size, name="dense_k")
        v = self._build_dense(y, self.hidden_size, name="dense_v")

        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = flow.concat([cache["k"], k], axis=1)
            v = flow.concat([cache["v"], v], axis=1)

            # Update cache
            cache["k"] = k
            cache["v"] = v

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5

        logits = flow.matmul(q, k, transpose_b=True)
        logits += bias
        weights = flow.nn.softmax(logits, name="attention_weights")
        if self.train:
            weights = flow.nn.dropout(weights, self.attention_dropout)

        attention_output = flow.matmul(weights, v)

        # Recombine heads --> [batch_size, length, hidden_size]
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self._build_dense(attention_output, unit=self.hidden_size, name="output_transform")

        return attention_output


class SelfAttention(Attention):
    """Multiheaded self-attention layer."""

    def __call__(self, x, bias, cache=None):
        """
        Self Attention just set the 'x' and 'y' all as 'x'
        :param x: Input Tensor
        :param bias: The attention bias
        :param cache: The cache
        :return:
        """
        return super(SelfAttention, self).__call__(x, x, bias, cache)


# # test
# if __name__ == "__main__":
#     @flow.global_function()
#     def multi_attention() -> tp.Numpy:
#         with flow.scope.namespace("multi"):
#             mha = SelfAttention(512, 8, 0.1, True)
#
#             x = flow.get_variable("x",
#                                   shape=(1, 60, 512),
#                                   initializer=flow.zeros_initializer(),
#                                   )
#             bias_blob = flow.get_variable("bias_blob",
#                                   shape=(1, 8, 60, 60),
#                                   initializer=flow.zeros_initializer(),
#                                   )
#             out = mha(x, bias=bias_blob)
#
#             return out
#
#     out = multi_attention()
#     print(out.shape)
