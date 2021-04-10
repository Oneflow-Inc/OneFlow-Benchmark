import oneflow as flow
import oneflow.nn as nn
import oneflow_api

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
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        # NOTE(Liang Depeng): original torch implementation 
        # return torch.zeros(1, self.hidden_size)
        hidden = flow.Tensor(1, self.hidden_size, device=oneflow_api.device("cuda"))
        flow.nn.init.zeros_(hidden)
        return hidden
