import oneflow as flow
import oneflow_api
from utils.dataset import all_letters, n_letters

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    # NOTE(Liang Depeng): original torch implementations
    # tensor = torch.zeros(1, n_letters)
    # tensor[0][letterToIndex(letter)] = 1
    tensor = flow.Tensor(n_letters, device=oneflow_api.device("cuda"))
    flow.nn.init.zeros_(tensor)
    tensor[letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    # NOTE(Liang Depeng): original torch implementations
    # tensor = torch.zeros(len(line), 1, n_letters)
    # for li, letter in enumerate(line):
    #     tensor[li][0][letterToIndex(letter)] = 1
    tensor = flow.Tensor(len(line), n_letters, device=oneflow_api.device("cuda"))
    flow.nn.init.zeros_(tensor)
    for li, letter in enumerate(line):
        tensor[li,letterToIndex(letter)] = 1
    return tensor

# flow.enable_eager_execution()
# print(letterToTensor('J'))
# print(lineToTensor('Jones').size())
