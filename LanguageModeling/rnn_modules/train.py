import oneflow as flow

from utils.dataset import *
from utils.tensor_utils import *
from models.rnn_model import RNN, CrossEntropyLoss

import time
import math
import numpy as np

flow.env.init()
flow.enable_eager_execution()

def train(category_tensor, line_tensor, rnn, criterion, learning_rate):
    hidden = rnn.initHidden()

    # TODO(Liang Depeng): oneflow Module does not have `zero_grad` method
    # rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)

    # TODO(Liang Depeng): oneflow Tensor does not implement `backward` method yet
    # loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    # TODO(Liang Depeng): oneflow Tensor does not support inplace `add_` yet
    # for p in rnn.parameters():
    #     p.data.add_(p.grad.data, alpha=-learning_rate)

    # NOTE(Liang Depeng): oneflow Tensor does not have `item` method yet
    # return output, loss.item()
    return output.softmax(), loss.numpy()[0]

n_iters = 100000
print_every = 50
plot_every = 10
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

dataset_path = "./data/names"
n_categories = processDataset(dataset_path)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
criterion = CrossEntropyLoss(n_categories)

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return now, '%ds' % s

start = time.time()

# refer to: https://blog.csdn.net/Nin7a/article/details/107631078
def topk_(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort, topk_index_sort

def categoryFromOutput(output):
    # TODO(Liang Depeng): oneflow does not provide the same `topk`
    #                     operation as pytorch, which also return the index.
    #                     Using a numpy implementation instead.

    # top_n, top_i = output.topk(1)
    # category_i = top_i[0].item()
    top_n, top_i = topk_(output.numpy(), 1)
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor, rnn, criterion, learning_rate)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        start, time_str = timeSince(start)
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('iter: %d / %f%%, time_for_every_%d_iter: %s, loss: %.4f, predict: %s / %s, correct? %s' % (iter, float(iter) / n_iters * 100, print_every, time_str, loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

