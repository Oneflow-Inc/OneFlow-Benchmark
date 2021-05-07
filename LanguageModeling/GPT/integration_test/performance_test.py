import argparse
from test_util import choose_and_run_test_cases


dropout_rate = 0.1
seq_len = 2048

fixed_args = {
    'dataset': '/data/gpt/gpt_sample_dataset_text_document',
    # 'dataset': '/dataset/gpt/datasets/output_data_text_document',
    'vocab-size': 50257,
    'make-vocab-size-divisible-by': 128,
    'hidden-dropout': dropout_rate,
    'attention-dropout': dropout_rate,
    # 'fp16': True,
    'optimizer': 'adamw',
    'lr-decay-style': 'cosine',
    'learning-rate': 0.00015,
    'lr-decay-iters': 320000,
    'lr-warmup-iters': 3200,
    'min-lr': 1e-5,
    'train-iters': 510,
    'log-interval': 10,
    'metric-print-format': 'table',
    'node-ips': '10.11.0.2,10.11.0.3,10.11.0.4,10.11.0.5',
    # 'node-ips': '192.168.1.15,192.168.1.16,192.168.1.13,192.168.1.12',
    'log': './output',
    'seq-length': seq_len,
    # 'max-position-embeddings': seq_len,
}


configurable_args = {
    'hidden-size': 1024,
    'num-attention-heads': 16,
    'num-layers': 16,
    'num-nodes': 1,
    'num-gpus-per-node': 1,
    'micro-batch-size': 2,
    'num-accumulation-steps': 1,
    'tensor-model-parallel-size': 1,
    'pipeline-model-parallel-size': 1,
    # 'global_batch_size': 2,
}
 

def gen_performance_test_args(row):
    env = {
        'PYTHONUNBUFFERED': 1,
        'NCCL_DEBUG': 'INFO',
        # 'ONEFLOW_DEBUG_MODE': 1,
        # 'ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE': 1,
    }
    args = dict(configurable_args)
    args.update(fixed_args)

    test_case = row['case']
    for k in configurable_args:
        if k in row:
            test_case = test_case + '_' + k[0] + str(row[k])
            args[k] = row[k]
            # print(f'update {k} with new value {row[k]}')
    print(args['num-accumulation-steps'])
    if int(args['num-accumulation-steps']) > 1:
        args['train-iters'] = 130
    return args, env, test_case


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="./integration_test/config_of_performance_test.csv",
        help="config csv file path for performance test",
    )
    parser.add_argument(
        "--python_cmd",
        type=str,
        default="python3",
        help="python command path",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="python command path",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="/workspace",
        help="workkspace path",
    )
    args = parser.parse_args()
    choose_and_run_test_cases(args.cfg, gen_performance_test_args, port=args.port,
                              python_cmd=args.python_cmd, workspace=args.workspace)