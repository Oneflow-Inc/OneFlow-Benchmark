import argparse


parser = argparse.ArgumentParser(description="flags for realse diff")
parser.add_argument(
    "--release_log_file", type=str, default="./logs/resnet_training.log")
parser.add_argument(
    "--base_release_log_file", type=str, default="./resnet50_fp16_b192_1n8g_50E_logs/resnet_training.log")
args = parser.parse_args()


def extract_info_from_file(log_file):
    '''
    model = resnet50
    batch_size_per_device = 128
    gpu_num_per_node = 8
    num_nodes = 2
    train: epoch 0, iter 20, loss: 7.087004, top_1: 0.000000, top_k: 0.000000, samples/s: 3988.891 1597933942.9863544
    train: epoch 0, iter 120, loss: 1.050499, top_1: 1.000000, top_k: 1.000000, samples/s: 5917.583 1597933977.6064055
    '''
    # extract info from file name
    throughput_dict = {}
    loss_dict = {}
    with open(log_file, 'r') as f:
        for line in f.readlines():
            ss = line.split(' ')
            if ss[0] == 'train:': 
                it = 'epoch_'+ss[2][:-1]+'_iter_'+ss[4][:-1] 
                throughput_dict[it] = float(ss[-2].strip())
                loss_dict[it+''] = float(ss[6][:-1].strip())

    return loss_dict, throughput_dict             


if __name__ == "__main__":
    release_loss_dict , release_throughput_dict = extract_info_from_file(args.release_log_file)
    base_release_loss_dict , base_release_throughput_dict = extract_info_from_file(args.base_release_log_file)
    for (k,v) in  release_loss_dict.items():
        if k in base_release_loss_dict and k in base_release_throughput_dict:
            print(k, ', diff loss :', (release_loss_dict[k]-base_release_loss_dict[k]), ', diff throughput : ', (release_throughput_dict[k]-base_release_throughput_dict[k]))


