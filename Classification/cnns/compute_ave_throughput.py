import math
import numpy as np

if __name__ == "__main__":
    nrows = 13
    ncols = 13
    model = 'vgg'
    ave_throughput = [[0]*ncols for i in range(nrows)]
    for i in range(nrows):
        for j in range(ncols):
            throughput = []
            with open('{0}txt/{0}_{1}_{2}.txt'.format(model,i,j), 'r', encoding='utf-8') as f:
                for line in f:
                    split_line = line.split()
            
                    if len(split_line)==13 and split_line[11]=='samples/s:':
                        throughput.append(float(split_line[12]))

            num = len(throughput)
            if num>0:
                throughput.sort()
                l = math.floor(num*0.1)
                r = math.ceil(num*0.9)
                ave_throughput[i][j] = sum(throughput[l:r])/(r-l)
            print(i,j,":", ave_throughput[i][j])

    
    np.savetxt('{0}txt/{0}_average_throughput.txt'.format(model), ave_throughput)



