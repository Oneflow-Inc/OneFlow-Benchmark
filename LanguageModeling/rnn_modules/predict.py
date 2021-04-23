import oneflow as flow

from utils.dataset import *
from utils.tensor_utils import *
from models.rnn_model import RNN

flow.env.init()
flow.enable_eager_execution()

dataset_path = "./data/names"
n_categories = processDataset(dataset_path)
print(letterToTensor('J'))
print(lineToTensor('Jones').size())
for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line, line_tensor.shape)

    
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)


input = lineToTensor('Albert')
# NOTE(Liang Depeng): original torch implementation
# hidden = torch.zeros(1, n_hidden)
hidden = flow.Tensor(1, n_hidden, device=oneflow_api.device("cuda"))
flow.nn.init.ones_(hidden)
print(input)
print(input[0])
output, next_hidden = rnn(input[0], hidden)
print(output.numpy())



