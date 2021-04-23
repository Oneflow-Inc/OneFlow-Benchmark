import oneflow as flow
import oneflow.nn as nn
import oneflow_api
from typing import Optional, List, Tuple

# TODO(Liang Depeng): oneflow's `CrossEntropyLoss` module call the `Build()` method
#                     in `forward` which raise error when running the module multiply times. 
class CrossEntropyLoss(flow.nn.Module):
    def __init__(self, depth: int, reduction: str = "mean",) -> None:
        super().__init__()
        assert reduction in [
            "sum",
            "none",
            "mean",
            None,
        ], "only 'sum', 'mean' and None supported by now"

        self.reduction = reduction

        self._op = (
            flow.builtin_op("sparse_softmax_cross_entropy")
            .Input("prediction")
            .Input("label")
            .Attr("depth", depth)
            .Output("prob")
            .Output("out")
            .Build()
        )

        self._sum_op = (
            flow.builtin_op("reduce_sum")
            .Input("input_tensor")
            .Output("output_tensor")
            .Attr("keepdims", False)
            .Attr("axis", [0,])
            .Build()
        )

    def forward(self, input, target):
        prob, out = self._op(input, target)
        sums = self._sum_op(out)[0]
        reduce_count = 1
        for dim in out.shape:
            reduce_count *= dim
        return flow.mul(sums, 1.0 / reduce_count)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        # TODO(Liang Depeng): oneflow does not support `flow.cat` yet
        self.cat = (
            flow.builtin_op("concat")
                .Input("in", 2)
                .Attr("axis", 1)
                .Attr("max_dim_size", input_size + hidden_size)
                .Output("out")
                .Build()
        )

    def forward(self, input, hidden):
        # NOTE(Liang Depeng): original torch implementation 
        # combined = torch.cat((input, hidden), 1)
        combined = self.cat(input, hidden)[0]
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        # output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        # NOTE(Liang Depeng): original torch implementation 
        # return torch.zeros(1, self.hidden_size)
        hidden = flow.Tensor(1, self.hidden_size, device=oneflow_api.device("cuda"))
        flow.nn.init.zeros_(hidden)
        return hidden
