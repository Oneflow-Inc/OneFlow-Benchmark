

if __name__ == "__main__":
    nrows = 13
    ncols = 13
    model = 'resnet'
    op_names = ['System-RegularizeGradient-L1L2-659',
        'Resnet-res2_1_branch2a-weight_optimizer',
        'Resnet-res3_3_branch2c-weight_optimizer', 
        'esnet-res5_0_branch2a-weight_optimizer', 
        'Resnet-res5_2_branch2b-weight_optimizer', 
        'Resnet-res5_2_branch2b_bn-gamma_optimizer',
        'Softmax_69']
    op_marker = '(^_^)'
    nop = len(op_names)
    sbps = [{} for i in range(nop)]
    type = {}
    sbp_buffer = [0] * nop
    model_strategy = [['Z']*ncols for i in range(nrows)]

    for i in range(nrows):
        for j in range(ncols):
            with open('{0}txt/{0}_{1}_{2}.txt'.format(model,i,j), 'r', encoding='utf-8') as f:
                line = f.readline()
                while line:
                    readnewline = True
                    if op_marker in line:
                        for k in range(nop):
                            op_name = op_names[k]
                            if op_name in line:
                                readnewline = False
                                f.readline()
                                line = f.readline()
                                sbp = ''
                                while not op_marker in line:
                                    sbp += ', ' + line
                                    line = f.readline()

                                if not sbp in sbps[k]:
                                    sbps[k][sbp] = len(sbps[k])
                                sbp_buffer[k] = sbps[k][sbp]

                    if readnewline:
                        line = f.readline()

                type_num = 0
                for sbp_num in sbp_buffer:
                    type_num = type_num * 10 + sbp_num

                if not type_num in type:
                    type[type_num] = chr(len(type)+65)
                model_strategy[i][j] = type[type_num]

    print(model_strategy)
    with open('{0}txt/{0}_model_strategy.txt'.format(model), 'w') as f:
        for rows in model_strategy:
            for col in rows:
                f.write("%s, " % col)
            f.write("\n")



