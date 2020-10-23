import oneflow as flow
import oneflow.typing as tp


def build_inband(x, num_lower, num_upper):
    m, n = x.shape[-2], x.shape[-1]

    inband_matrix = flow.constant(value=0.0, dtype=flow.float32,
                                  shape=(m, n))

    # TODO We cannot Use Oneflow to build a band_matrix