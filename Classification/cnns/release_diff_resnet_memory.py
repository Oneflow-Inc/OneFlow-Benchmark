import argparse


parser = argparse.ArgumentParser(description="flags for realse diff")
parser.add_argument(
    "--release_log_file", type=str, default="./logs/resent_fp16_b192_1ng8_50e.mem")
parser.add_argument(
    "--base_release_log_file", type=str, default="./resnet50_fp16_b192_1n8g_50E_logs/resnet_fp16_b192_1n8g_50E.mem")
args = parser.parse_args()


def extract_info_from_file(log_file):
    '''
    gpt0 max memory usage: 10651.06M
    gpt1 max memory usage: 10655.06M
    gpt2 max memory usage: 10655.06M
    gpt3 max memory usage: 10635.06M
    gpt4 max memory usage: 10635.06M
    gpt5 max memory usage: 10675.06M
    gpt6 max memory usage: 10675.06M
    gpt7 max memory usage: 10675.06M
    '''
    # extract info from file name
    max_memory_usage = 0
    with open(log_file, 'r') as f:
        for line in f.readlines():
            ss = line.split(' ')
            if len(ss) == 5 and max_memory_usage < float(ss[-1].strip()[:-1]):
                max_memory_usage = float(ss[-1].strip()[:-1])

    return max_memory_usage            


if __name__ == "__main__":
    release_max_memory_usage = extract_info_from_file(args.release_log_file)
    base_max_memory_usage = extract_info_from_file(args.base_release_log_file)
    print('release max memory usage: ', release_max_memory_usage, 'M, base max memory usage: ', base_max_memory_usage, 'M, diff :', (release_max_memory_usage-base_max_memory_usage))
