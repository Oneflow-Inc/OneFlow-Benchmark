import sys
import re

# 文件路径
file_path = sys.argv[1]

# 存储 train 模式下的 throughput
print(file_path)
train_throughputs = []

# 正则表达式模式匹配 train 模式下的 throughput
pattern = re.compile(r'\[train\][^|]*?throughput:\s(\d+\.\d+)')

# 读取文件并提取需要的信息
with open(file_path, 'r') as file:
    for line in file:
        matches = pattern.findall(line)
        for match in matches:
            throughput = float(match)
            train_throughputs.append(throughput)

# 计算平均 throughput
if train_throughputs:
    average_throughput = sum(train_throughputs) / len(train_throughputs)
    print(f'The average throughput for [train] mode is: {average_throughput:.6f}')
else:
    print('No [train] mode throughputs found.')
