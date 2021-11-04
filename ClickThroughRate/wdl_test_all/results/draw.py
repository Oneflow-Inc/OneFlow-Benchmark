from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


dirpath='/home/shiyunxiao/of_benchmark/OneFlow-Benchmark/ClickThroughRate/wdl_test_all'

def benchmark_n1g1_loss():
    '''
    draw n1g1 benchmark line chart of loss
    '''
    filename='n1g1_old_100loss'
    legendname='benchmark_n1g1_loss'
    csvpath='wide_deep_test/old/%s.csv'%filename
    imgpath='wide_deep_test/img/%s.jpg'%filename
    df = pd.read_csv(csvpath)
    y=df.loc[:,['loss']].values
    x = list(range(100))
    plt.figure(figsize=(8, 5))
    plt.xlabel('iter times')
    plt.ylabel('loss')
    plt.plot(x,y)
    plt.legend([legendname])
    plt.savefig(imgpath,dpi=400)

def n1n1_loss():
    '''draw n1g1 benchmark,eager,graph,ddp line chart of loss'''
    filename='n1g1_ddp_eager_graph_benchmark_100loss'
    benchmark_csvpath=dirpath+'/results/old/n1g1_benchmark.csv'
    ddp_csvpath=dirpath+'/results/new/n1g1_ddp/record_32_0.csv'
    eager_csvpath=dirpath+'/results/new/n1g1_eager/record_32_0.csv'
    graph_csvpath=dirpath+'/results/new/n1g1_graph/record_32_0.csv'
    imgpath=dirpath+'/results/img/%s.jpg'%filename
    df_benchmark=pd.read_csv(benchmark_csvpath)
    df_ddp = pd.read_csv(ddp_csvpath)
    df_eager = pd.read_csv(eager_csvpath)
    df_graph = pd.read_csv(graph_csvpath)
    y_benchmark=df_benchmark.loc[:,['loss']].values
    y_ddp=df_ddp.loc[:,['loss']].values
    y_eager=df_eager.loc[:,['loss']].values
    y_graph=df_graph.loc[:,['loss']].values
    x = list(range(100))
    plt.figure(figsize=(8, 5))
    plt.xlabel('iter times')
    plt.ylabel('loss')
    plt.plot(x,y_benchmark,label='benchmark')
    plt.plot(x,y_ddp,label='ddp')
    plt.plot(x,y_eager,label='eager')
    plt.plot(x,y_graph,label='graph')
    plt.legend()
    plt.savefig(imgpath,dpi=400)

def n1g1_lat():
    '''
    draw n1g1 benchmark,eager,graph,ddp line chart of latency
    '''
    filename='n1g1_ddp_eager_graph_lat'
    imgpath='wide_deep_test/img/%s.jpg'%filename
    n1g1_ddp_mem=dirpath+'/results/new/n1g1_ddp_mem/record_16384_0.csv'
    n1g1_eager_mem=dirpath+'/results/new/n1g1_eager_mem/record_16384_0.csv'
    n1g1_graph_mem=dirpath+'/results/new/n1g1_graph_mem/record_16384_0.csv'
    old_n1g1_mem=dirpath+'/results/old/n1g1_benchmark_mem.csv'
    n1g1_ddp_mem=pd.read_csv(n1g1_ddp_mem)
    n1g1_eager_mem = pd.read_csv(n1g1_eager_mem)
    n1g1_graph_mem = pd.read_csv(n1g1_graph_mem)
    old_n1g1_mem = pd.read_csv(old_n1g1_mem)
    n1g1_ddp_mem=n1g1_ddp_mem.loc[:,['latency/ms']].values
    n1g1_eager_mem=n1g1_eager_mem.loc[:,['latency/ms']].values
    n1g1_graph_mem=n1g1_graph_mem.loc[:,['latency/ms']].values
    old_n1g1_mem=old_n1g1_mem.loc[:,['latency/ms']].values
    x = list(range(100))
    plt.figure(figsize=(8, 5))
    plt.xlabel('iter times')
    plt.ylabel('latency/ms')
    plt.plot(x,n1g1_ddp_mem,label='ddp')
    plt.plot(x,n1g1_eager_mem,label='eager')
    plt.plot(x,n1g1_graph_mem,label='graph')
    plt.plot(x,old_n1g1_mem,label='benchmark')
    plt.legend()
    plt.savefig(imgpath,dpi=400)

def n1g1_lat_avg():
    '''
    draw n1g1 benchmark,eager,graph,ddp table of latency
    '''
    filename='n1g1_ddp_eager_graph_lat'
    imgpath=dirpath+'/results/img/%s.jpg'%filename
    n1g1_ddp_mem=dirpath+'/results/new/n1g1_ddp_mem/record_16384_0.csv'
    n1g1_eager_mem=dirpath+'/results/new/n1g1_eager_mem/record_16384_0.csv'
    n1g1_graph_mem=dirpath+'/results/new/n1g1_graph_mem/record_16384_0.csv'
    old_n1g1_mem=dirpath+'/results/old/n1g1_benchmark_mem.csv'
    n1g1_ddp_mem=pd.read_csv(n1g1_ddp_mem)
    n1g1_eager_mem = pd.read_csv(n1g1_eager_mem)
    n1g1_graph_mem = pd.read_csv(n1g1_graph_mem)
    old_n1g1_mem = pd.read_csv(old_n1g1_mem)
    n1g1_ddp_mem=n1g1_ddp_mem.loc[:,['latency/ms']].values[20:100]
    n1g1_eager_mem=n1g1_eager_mem.loc[:,['latency/ms']].values[20:100]
    n1g1_graph_mem=n1g1_graph_mem.loc[:,['latency/ms']].values[20:100]
    old_n1g1_mem=old_n1g1_mem.loc[:,['latency/ms']].values[20:100]
    return (np.mean(n1g1_ddp_mem),
        np.mean(n1g1_eager_mem),
        np.mean(n1g1_graph_mem),
        np.mean(old_n1g1_mem))


def n1g1_mem():
    '''draw n1g1 benchmark,eager,graph,ddp line of memory'''
    filename='n1g1_ddp_eager_graph_mem'
    imgpath='wide_deep_test/img/%s.jpg'%filename
    n1g1_ddp_mem=dirpath+'/results/new/n1g1_ddp_mem/record_16384_0.csv'
    n1g1_eager_mem=dirpath+'/results/new/n1g1_eager_mem/record_16384_0.csv'
    n1g1_graph_mem=dirpath+'/results/new/n1g1_graph_mem/record_16384_0.csv'
    old_n1g1_mem=dirpath+'/results/old/n1g1_benchmark_mem.csv'
    n1g1_ddp_mem=pd.read_csv(n1g1_ddp_mem)
    n1g1_eager_mem = pd.read_csv(n1g1_eager_mem)
    n1g1_graph_mem = pd.read_csv(n1g1_graph_mem)
    old_n1g1_mem = pd.read_csv(old_n1g1_mem)
    n1g1_ddp_mem=n1g1_ddp_mem.loc[:,['memory_usage/MB']].values
    n1g1_eager_mem=n1g1_eager_mem.loc[:,['memory_usage/MB']].values
    n1g1_graph_mem=n1g1_graph_mem.loc[:,['memory_usage/MB']].values
    old_n1g1_mem=old_n1g1_mem.loc[:,['memory_usage_0/MB']].values
    x = list(range(100))
    plt.figure(figsize=(8, 5))
    plt.xlabel('iter times')
    plt.ylabel('memory_usage/MB')
    plt.plot(x,n1g1_ddp_mem,label='ddp')
    plt.plot(x,n1g1_eager_mem,label='eager')
    plt.plot(x,n1g1_graph_mem,label='graph')
    plt.plot(x,old_n1g1_mem,label='benchmark')
    plt.legend()
    plt.savefig(imgpath,dpi=400)

def n1g1_mem_avg():
    '''draw n1g1 benchmark,eager,graph,ddp table of memory'''
    filename='n1g1_ddp_eager_graph_mem'
    n1g1_ddp_mem=dirpath+'/results/new/n1g1_ddp_mem/record_16384_0.csv'
    n1g1_eager_mem=dirpath+'/results/new/n1g1_eager_mem/record_16384_0.csv'
    n1g1_graph_mem=dirpath+'/results/new/n1g1_graph_mem/record_16384_0.csv'
    old_n1g1_mem=dirpath+'/results/old/n1g1_benchmark_mem.csv'
    n1g1_ddp_mem=pd.read_csv(n1g1_ddp_mem)
    n1g1_eager_mem = pd.read_csv(n1g1_eager_mem)
    n1g1_graph_mem = pd.read_csv(n1g1_graph_mem)
    old_n1g1_mem = pd.read_csv(old_n1g1_mem)
    n1g1_ddp_mem=n1g1_ddp_mem.loc[:,['memory_usage/MB']].values[20:100]
    n1g1_eager_mem=n1g1_eager_mem.loc[:,['memory_usage/MB']].values[20:100]
    n1g1_graph_mem=n1g1_graph_mem.loc[:,['memory_usage/MB']].values[20:100]
    old_n1g1_mem=old_n1g1_mem.loc[:,['memory_usage_0/MB']].values[20:100]
    return (np.mean(n1g1_ddp_mem),
    np.mean(n1g1_eager_mem),
    np.mean(n1g1_graph_mem),
    np.mean(old_n1g1_mem))

def all_n1g8_loss(column='loss'):
    '''draw n1g8 benchmark,eager,graph,ddp line of latency or memory'''
    gpu_num=2
    filename='n1g8_ddp_eager_graph_%s'%(column[:3])
    imgpath=dirpath+'/results/img/%s.jpg'%filename
    plt.figure(figsize=(8, 5))
    plt.xlabel('iter times')
    plt.ylabel(column)
    x = list(range(100))
    #calculate average lat in 8 devices per batch
    for path in [dirpath+'/results/new/n1g8_ddp_mem',dirpath+'/results/new/n1g8_eager_mem',dirpath+'/results/new/n1g8_graph_mem']:
        data=np.zeros(100)
        for i in range(gpu_num):
            csv_path=os.path.join(path,'record_16384_%s.csv'%i)
            df = pd.read_csv(csv_path)
            the_data=df.loc[:,[column]].values.flatten()
            data+=the_data
        data=data/gpu_num
        label_str=path.split('''/''')[-1].split('_')[-2]
        plt.plot(x,data,label=label_str)

    old_csv_path=dirpath+'/results/old/n1g8_benchmark_mem.csv'
    df=pd.read_csv(old_csv_path)
    if column=='latency/ms' or column=='loss':
        data=df.loc[:,[column]].values.flatten()
        plt.plot(x,data,label='benchmark')
    else:
        data=df.iloc[:,-gpu_num:].values
        data=np.mean(data,axis=1).flatten()
        plt.plot(x,data,label='benchmark')
    plt.legend()
    plt.savefig(imgpath,dpi=400)


def n1g8_lat_mem(column='latency/ms'):
    '''draw n1g8 benchmark,eager,graph,ddp line of latency or memory'''
    filename='n1g8_ddp_eager_graph_%s'%(column[:3])
    imgpath='wide_deep_test/img/%s.jpg'%filename
    plt.figure(figsize=(8, 5))
    plt.xlabel('iter times')
    plt.ylabel(column)
    x = list(range(100))
    #calculate average lat in 8 devices per batch
    global dirpath
    for dirpath in [dirpath+'/results/new/n1g8_ddp_mem',dirpath+'/results/new/n1g8_eager_mem',dirpath+'/results/new/n1g8_graph_mem']:
        data=np.zeros(100)
        for i in range(8):
            csv_path=os.path.join(dirpath,'record_16384_%s.csv'%i)
            df = pd.read_csv(csv_path)
            the_data=df.loc[:,[column]].values.flatten()
            data+=the_data
        data=data/8
        label_str=dirpath.split('''/''')[-1].split('_')[-2]
        plt.plot(x,data,label=label_str)

    old_csv_path=dirpath+'/results/old/n1g8_benchmark_mem.csv'
    df=pd.read_csv(old_csv_path)
    if column=='latency/ms' or column=='loss':
        data=df.loc[:,[column]].values.flatten()
        plt.plot(x,data,label='benchmark')
    else:
        data=df.iloc[:,-8:-1].values
        data=np.mean(data,axis=1).flatten()
        plt.plot(x,data,label='benchmark')
    plt.legend()
    plt.savefig(imgpath,dpi=400)
    
def n1g8_lat_mem_avg(column='latency/ms'):
    gpu_num=2
    '''draw n1g8 benchmark,eager,graph,ddp table of latency or memory'''
    #calculate average lat in 8 devices per batch
    return_result=[]
    for path in [dirpath+'/results/new/n1g8_ddp_mem',dirpath+'/results/new/n1g8_eager_mem',dirpath+'/results/new/n1g8_graph_mem']:
        data=np.zeros(100)
        for i in range(gpu_num):
            csv_path=os.path.join(path,'record_16384_%s.csv'%i)
            df = pd.read_csv(csv_path)
            the_data=df.loc[:,[column]].values.flatten()
            data+=the_data
        data=data/gpu_num
        return_result.append(np.mean(data[20:100]))


    old_csv_path=dirpath+'/results/old/n1g8_benchmark_mem.csv'
    df=pd.read_csv(old_csv_path)
    if column=='latency/ms' or column=='loss':
        data=df.loc[:,[column]].values.flatten()
        return_result.append(np.mean(data[20:100]))
    else:
        data=df.iloc[:,-gpu_num:].values
        data=np.mean(data,axis=1).flatten()
        return_result.append(np.mean(data[20:100]))
    return tuple(return_result)

def n1g1_tabel():
    (ddp_lat,eager_lat,graph_lat,old_lat)=( round(num,3) for num in n1g1_lat_avg())
    (ddp_mem,eager_mem,graph_mem,old_mem)=( round(num,3) for num in n1g1_mem_avg()) 
    base_info=[['n1g1','32','0'],['n1g1','32','0']]
    data=[[ddp_lat,eager_lat,graph_lat,old_lat],
    [ddp_mem,eager_mem,graph_mem,old_mem]]
    data=np.concatenate((base_info,data),axis=1)
    df=pd.DataFrame(data,columns=['gpu','batch_size','dropout','ddp','eager','graph','old'],index=['latency/ms','memory_usage/MB'])
    print(df)

def n1g8_tabel():
    (ddp_lat,eager_lat,graph_lat,old_lat)=( round(num,3) for num in n1g8_lat_mem_avg('latency/ms'))
    (ddp_mem,eager_mem,graph_mem,old_mem)=( round(num,3) for num in n1g8_lat_mem_avg('memory_usage/MB')) 
    base_info=[['n1g8','16384','0.5'],['n1g8','16384','0.5']]
    data=[[ddp_lat,eager_lat,graph_lat,old_lat],
    [ddp_mem,eager_mem,graph_mem,old_mem]]
    data=np.concatenate((base_info,data),axis=1)
    df=pd.DataFrame(data,columns=['gpu','batch_size','dropout','ddp','eager','graph','old'],index=['latency/ms','memory_usage/MB'])
    print(df)

if __name__ == "__main__":
    n1n1_loss()
    all_n1g8_loss('loss')
    n1g8_tabel()
    n1g1_tabel()
    