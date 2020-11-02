"""
Test model_utils
"""
import oneflow as flow
import oneflow.typing as tp
import model_utils
import numpy as np
import unittest

NEG_INF = -1e9


@flow.unittest.skip_unless_1n1d()
class test_model_utils(flow.unittest.TestCase):
    def test_get_padding(test_case):
        x = np.array([[1, 0, 0, 0, 2], [3, 4, 0, 0, 0], [0, 5, 6, 0, 7]]).astype(np.float32)

        @flow.global_function()
        def do_get_pad(input_x: tp.Numpy.Placeholder(shape=x.shape, dtype=flow.float32)
                       ) -> tp.Numpy:
            pad_val = model_utils.get_padding(input_x, 0)
            return pad_val

        out = do_get_pad(x)
        assert np.array_equal(out, [[0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [1, 0, 0, 1, 0]])

    def test_padding_bias(test_case):
        x = np.array([[1, 0, 0, 0, 2], [3, 4, 0, 0, 0], [0, 5, 6, 0, 7]]).astype(np.float32)

        @flow.global_function()
        def do_get_pad_bias(input_x: tp.Numpy.Placeholder(shape=x.shape, dtype=flow.float32)
                            ) -> tp.Numpy:
            pad_bias = model_utils.get_padding_bias(input_x)
            flatten_pad_bias = flow.reshape(pad_bias, shape=(3, 5))

            return flatten_pad_bias

        out = do_get_pad_bias(x)
        assert np.array_equal([[0, NEG_INF, NEG_INF, NEG_INF, 0],
                               [0, 0, NEG_INF, NEG_INF, NEG_INF],
                               [NEG_INF, 0, 0, NEG_INF, 0]], out)

    def test_get_decoder_self_attention_bias(test_case):
        x = np.array([[1, 0, 0, 0, 2], [3, 4, 0, 0, 0], [0, 5, 6, 0, 7]]).astype(np.float32)
        length = len(x[0])

        @flow.global_function()
        def do_get_decoder_self_attention_bias() -> tp.Numpy:
            att_bias = model_utils.get_decoder_self_attention_bias(length)

            return att_bias

        # Init Variable, Because we use get_variable to get ones matrix.
        check = flow.train.CheckPoint()
        check.init()

        out = do_get_decoder_self_attention_bias()

        assert np.array_equal([[[[0, NEG_INF, NEG_INF, NEG_INF, NEG_INF],
                                 [0, 0, NEG_INF, NEG_INF, NEG_INF],
                                 [0, 0, 0, NEG_INF, NEG_INF],
                                 [0, 0, 0, 0, NEG_INF],
                                 [0, 0, 0, 0, 0]]]], out)


if __name__ == "__main__":
    unittest.main()