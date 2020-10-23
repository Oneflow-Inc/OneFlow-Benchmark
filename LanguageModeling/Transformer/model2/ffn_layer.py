"""Implementation of fully connected network."""
import oneflow as flow
import oneflow.typing as tp
import numpy as np


class FeedFowardNetwork(object):
    """Fully connected feedforward network."""

    def __init__(self, hidden_size, filter_size, relu_dropout, train=True):
        super(FeedFowardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.train = train

        self.use_bias = True

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

    def __call__(self, x, padding=None):
        # Retrieve dynamically known shapes
        batch_size = x.shape[0]
        length = x.shape[1]

        if padding is not None:
            with flow.scope.namespace("remove_padding"):
                # Flatten padding to [batch_size*length]
                pad_mask = flow.reshape(padding, [-1])

                nonpad_ids = flow.cast(flow.where(pad_mask < 1e-9), dtype=flow.int32)
                # nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))

                # Reshape x to [batch_size*length, hidden_size] to remove padding
                x = flow.reshape(x, [-1, self.hidden_size])
                x = flow.gather_nd(x, indices=nonpad_ids)

                # Reshape x from 2 dimensions to 3 dimensions.

                # TODO:Maybe has a batch axis error in there
                x = flow.expand_dims(x, axis=0)

        output = self._build_dense(x, self.filter_size, name="filter_layer")
        if self.train:
            # In TensorFlow the param means `keep_prob` and use `1-dropout`,
            # but our dropout means drop rate so i just use dropout !
            output = flow.nn.dropout(output, self.relu_dropout)
        if padding is not None:
            with flow.scope.namespace("re_add_padding"):
                output = flow.squeeze(output, axis=[0, ])
                output = flow.scatter_nd(
                    indices=nonpad_ids,
                    updates=output,
                    shape=[batch_size * length, self.hidden_size]
                )
                output = flow.reshape(output, [batch_size, length, self.hidden_size])
        return output


# # Test
# @flow.global_function()
# def test_FFN(x: tp.Numpy.Placeholder(shape=(1, 4), dtype=flow.float32)) -> tp.Numpy:
#     FFN = FeedFowardNetwork(10, 20, 0.1)
#     out = FFN(x)
#     return out
#
#
# if __name__ == "__main__":
#     check = flow.train.CheckPoint()
#     check.init()
#
#     x = np.array([[0, 2, 0, 5]]).astype(np.float32)
#     out = test_FFN(x)
#     print(out.shape)
