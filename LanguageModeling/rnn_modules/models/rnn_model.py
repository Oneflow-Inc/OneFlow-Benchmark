import oneflow as flow
import oneflow.nn as nn
import oneflow_api
from typing import Optional, List, Tuple

class CrossEntropyLoss(flow.nn.Module):
    def __init__(
        self,
        depth: int,
        weight=None,
        ignore_index: int = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if weight != None:
            raise ValueError("Argument weight is not supported yet")
        if ignore_index != None:
            raise ValueError("Argument ignore_index is not supported yet")
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
            .Output("prob")
            .Output("out")
            .Attr("depth", depth).Build()
        )

        if self.reduction == "mean":
            self._reduce = flow.Mean()
        elif self.reduction == "sum":
            self._reduce = flow.Sum()
        else:
            self._reduce = None

    def forward(self, input, target):
        self._op = self._op
        prob, out = self._op(input, target)
        if self._reduce != None:
            return self._reduce(out)
        else:
            return out

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        # NOTE(Liang Depeng): module support flow.cat 
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
