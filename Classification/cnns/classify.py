

if __name__ == "__main__":
    # decide model
    nrows = 13
    ncols = 13
    model = 'vgg'

    # chose operators for distinguishing different strategies.
    if model == 'resnet':
        op_names = ['System-RegularizeGradient-L1L2-659',
            'Resnet-res2_1_branch2a-weight_optimizer',
            'Resnet-res3_3_branch2c-weight_optimizer', 
            'esnet-res5_0_branch2a-weight_optimizer', 
            'Resnet-res5_2_branch2b-weight_optimizer', 
            'Resnet-res5_2_branch2b_bn-gamma_optimizer',
            'Softmax_69']
        
    if model == 'vgg':
        op_names = ['Reshape_28',
            'conv2_bias_optimizer',
            'conv2_bn-beta_optimizer',
            'conv2_weight_optimizer',
            'conv6_weight_optimizer',
            'dense2-weight_optimizer',
            'dense1-weight_optimizer',
            'conv0_bn_grad']

    # each starting line of operator contains a marker
    op_marker = '(^_^)'

    nop = len(op_names)
    for i in range(nop):
        op_names[i] += ' ' + op_marker

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
                                # collect all the lines with sbps, different strategies have different sbp lines for one of the operators
                                line = f.readline()
                                sbp = ''
                                while not op_marker in line:
                                    sbp += ', ' + line
                                    line = f.readline()
                                # associate the sbp with a number
                                if not sbp in sbps[k]:
                                    sbps[k][sbp] = len(sbps[k])
                                sbp_buffer[k] = sbps[k][sbp]

                    if readnewline:
                        line = f.readline()

                type_num = 0
                for sbp_num in sbp_buffer:
                    type_num = type_num * 10 + sbp_num
                # associate the type with a characteristic
                if not type_num in type:
                    type[type_num] = chr(len(type)+65)
                model_strategy[i][j] = type[type_num]

    print(model_strategy)
    with open('{0}txt/{0}_model_strategy.txt'.format(model), 'w') as f:
        for rows in model_strategy:
            for col in rows:
                f.write("%s, " % col)
            f.write("\n")



