import json
import matplotlib.pyplot as plt
import argparse
import numpy as np

def read_file(file1, file2):

    with open(file1,'r') as load_f:
        dict1 = json.load(load_f)

    with open(file2,'r') as load_f:
        dict2 = json.load(load_f)


    return dict1, dict2

def analysis_loss(y3, f32):
    # if f32 == 1:
    #     iter = len(y3)

    # else:
    # len = y3.shape[0]
    # min = y3.min()
    # max = y3.max()
    # mean = np.mean(y3)
    # var = np.var(y3) 

    # random = np.random.randn(len)/10000.0
    # print(random)
    # print(y3)
    # lst=[y3,random]
    # res=np.corrcoef(lst)
    # print('----->', res)

    len = y3.shape[0]

    if f32 == 1:
        iter = np.count_nonzero(y3==0)
        tmp = iter/len
        print('count zeor = ', iter)
        if iter/len > 0.6:
            print('Test passed')
            return 1
        else:
            print('Test failed')
            return 0
    else:

        mean = np.mean(y3)
        var = np.var(y3) 
        print('F16---->', abs(mean), var)

        if abs(mean) < 0.001 and var < 0.00001:
            print('Test passed')
            return 1
        else:
            print('Test failed')
            return 0

def drawing_loss(dict1, dict2, f32, image):


    m1 = dict1["memory"]
    m2 = dict2["memory"]
    
    print(m1)
    print(m2)
    table_len = len(m1)
    row_labels = ['old','new']
    table_vals = [m1,m2]


    y1 = dict1["total_loss"]
    y2 = dict2["total_loss"]
    y1=list(map(float,y1))
    y2=list(map(float,y2))
    x = np.arange(1, len(y1)+1)
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.subtract(y1,y2)
    # v =list(map(lambda x,y:x - y))

    result = analysis_loss(y3, f32)

    print(x)
    print(y1)
    print(y2)
    print(y3)
    fig = plt.figure(figsize=(24,8), dpi=150)
    plt.figure(1) 
    ax = fig.add_subplot()


    ax1 = plt.subplot(121)
    plt.xlabel('iterations')
    plt.plot(x,y1,color='red',label='Diachronic version')
    plt.plot(x,y2,color='blue',label='Current version')
    plt.title('Loss comparison')

    plt.legend(loc='best')
    
    ax2 = plt.subplot(122)
    plt.xlabel('iterations')
    plt.plot(x,y3,color='red')  
    plt.title('Loss difference')
    plt.table(cellText=table_vals, rowLabels=row_labels, colWidths=[0.05]*table_len, loc='best')

    plt.suptitle(image.split('/')[1].split('.')[0],fontsize=20,x=0.5,y=0.98)

    if result == 1:
        plt.text(0.9, 1,'PASS', fontsize=50, color='blue', transform=ax.transAxes)
    else:
        plt.text(0.9, 1,'FAILED',fontsize=50,color='red',transform=ax.transAxes)
    plt.savefig(image)
    

# def analysis_f32(dict1, dict2):
#     return 1
# def analysis_f16(dict1, dict2):
#     return 1

def main():
    print('test')
    parser = argparse.ArgumentParser(description="Compare and analyze training results and output icons")
    parser.add_argument("--cmp1_file", type=str, default=None)
    parser.add_argument("--cmp2_file", type=str, default=None)
    parser.add_argument("--out_file", type=str, default=None)
    parser.add_argument("--f32", type=int, default=0)
    args = parser.parse_args()

    # print('---------------------')
    dict1, dict2 = read_file(args.cmp1_file, args.cmp2_file)

    # if args.f32 == 1:
    #     result = analysis_f32(dict1, dict2)
    # else:
    #     result = analysis_f16(dict1, dict2)

    
    drawing_loss(dict1, dict2, args.f32, args.out_file)


if __name__ == "__main__":
    main()