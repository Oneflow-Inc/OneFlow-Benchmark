import numpy as np
import math

if __name__ == "__main__":
    model = 'vgg'
    throughput_file = '{0}txt/{0}_average_throughput.txt'.format(model)
    strategy_file = '{0}txt/{0}_model_strategy.txt'.format(model)

    model_strategy = []
    with open(strategy_file, 'r') as f:
        line = f.readline()
        while line:
            model_strategy.append(line.split(", "))
            line = f.readline()

    ave_throughput = np.loadtxt(throughput_file)

    nrows = min(len(ave_throughput), len(model_strategy)-1)
    if nrows > 0:
        ncols = min(len(ave_throughput[0]), len(model_strategy[0])-1)
        strategy2id = {}
        id2throughput = []
        for i in range(nrows):
            for j in range(ncols):
                throughput = ave_throughput[i][j]
                if throughput > 1.0:
                    strategy = model_strategy[i][j]
                    if not strategy in strategy2id:
                        strategy2id[strategy] = len(id2throughput)
                        id2throughput.append([])
                    id2throughput[strategy2id[strategy]].append(throughput)
        
        id2stat = [[0]*4 for i in range(len(id2throughput))]

        for strategy in strategy2id:
            id = strategy2id[strategy]
            n = len(id2throughput[id])
            if n==0:
                continue
            l = math.floor((n-1)*0.2)
            r = math.ceil(n*0.8)
            if n>2:
                l = max(l, 1)
                r = min(n-1, r)
            
            id2throughput[id].sort()

            print(l, r, n, id)
            print(id2throughput[id])
            id2stat[id][0] = sum(id2throughput[id][l:r])/(r-l)
            id2stat[id][1] = id2throughput[id][math.floor((l+r-1)/2)]
            id2stat[id][2] = id2throughput[id][l]
            id2stat[id][3] = id2throughput[id][r-1]

        sort_strategy = sorted(strategy2id.items(), key = lambda item: -id2stat[item[1]][0])

        with open('{0}txt/{0}_sorted_strategy.txt'.format(model), 'w') as f:
            for pair in sort_strategy:
                id = pair[1]
                f.write("Strategy: {0}, Ave: {1}, Median: {2}, {3} ~ {4} \n".format(pair[0], id2stat[id][0], id2stat[id][1], id2stat[id][2], id2stat[id][3]))

        print(strategy2id)
        print(id2stat)
        print(sort_strategy)

    





