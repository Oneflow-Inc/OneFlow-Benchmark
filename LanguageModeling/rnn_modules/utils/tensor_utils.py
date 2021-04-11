import oneflow as flow
import oneflow_api

from utils.dataset import *

import random

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    # NOTE(Liang Depeng): oneflow does not provide `flow.zeros`
    # tensor = torch.zeros(1, n_letters)
    # tensor[0][letterToIndex(letter)] = 1
    tensor = flow.Tensor(n_letters, device=oneflow_api.device("cuda"))
    flow.nn.init.zeros_(tensor)
    tensor[letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    # NOTE(Liang Depeng): oneflow does not provide `flow.zeros`
    # tensor = torch.zeros(len(line), 1, n_letters)
    # for li, letter in enumerate(line):
    #     tensor[li][0][letterToIndex(letter)] = 1
    tensor = flow.Tensor(len(line), n_letters, device=oneflow_api.device("cuda"))
    flow.nn.init.zeros_(tensor)
    for li, letter in enumerate(line):
        # NOTE(Liang Depeng): oneflow Tensor does not support tensor[li][letterToIndex(letter)] = 1
        tensor[li, letterToIndex(letter)] = 1
    return tensor

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    # category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    category_tensor = flow.Tensor([all_categories.index(category)], dtype=flow.int64)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor
