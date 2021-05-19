import argparse
import re
import json

from ctypes import *



def collect_loss(log_file, gpu_num):
    print("loss : ",log_file)

    f = open(log_file,"r")  
    lines = f.readlines()#读取全部内容  
    total_loss = []
    mlm_loss = []
    nsp_loss = []
    throughput =[]
    memory=[]

    pattern = re.compile(r"step:\s*(\d+)\s*,\s*total_loss:\s*(\d+\.?\d+)\s*,\s*mlm_loss:\s*(\d+\.?\d+)\s*,\s*nsp_loss:\s*(\d+\.?\d+)\s*,\s*throughput:\s*(\d+\.?\d+)\s*")
    for line in lines:
        if(line.split(':')[0] == 'step'):
            # print(line)
            
            match = pattern.match(line)
            if match:
                # print(match.groups())
                total_loss.append(match.group(2))
                mlm_loss.append(match.group(3))
                nsp_loss.append(match.group(4))
                throughput.append(match.group(5))
        if(line.split(' [MiB]\\n')[0] == 'b\'memory.used'):
            str_tmp = line.split(' [MiB]\\n')[1]
            # print(str_tmp)
            for i in range(gpu_num):
                memory.append(str_tmp.split(' MiB\\n')[i])

    return total_loss, mlm_loss, nsp_loss,throughput, memory


def main():
    parser = argparse.ArgumentParser(description="collect GPU device memory usage")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--mem_file", type=str, default=None)
    parser.add_argument("--out_file", type=str, default=None)
    parser.add_argument("--gpu_num", type=int, default=1)

    args = parser.parse_args()

    total_loss, mlm_loss, nsp_loss,throughput, memory = collect_loss(args.log_file, args.gpu_num)
    # print(total_loss)
    # print(mlm_loss)
    # print(nsp_loss)
    # print(throughput)
    out={}
    out['total_loss'] = total_loss
    out['mlm_loss'] = mlm_loss
    out['nsp_loss'] = nsp_loss
    out['throughput'] = throughput
    out['memory'] = memory
    # print(out)

    string = json.dumps(out)
    with open(args.out_file,'w')as f:
        f.write(string)


if __name__ == "__main__":
    # libc = CDLL("libc.so.6")
    # msg = "Hello, world!\n"
    # libc.printf("Testing: %s \n", msg)
    main()
