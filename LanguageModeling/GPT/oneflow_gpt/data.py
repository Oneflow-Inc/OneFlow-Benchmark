import oneflow as flow

from . import distribute
from .config import get_args


def get_train_val_test_num_samples(split, num_samples):
    assert len(split) == 3
    total = sum(split)
    return [int((s / total) * num_samples) for s in split]


class GPTDataLoader(object):
    def __init__(self, name):
        self.name = name
        args = get_args()
        assert args.dataset is not None
        self.dataset = args.dataset
        self.batch_size = args.global_batch_size // args.num_accumulation_steps
        self.seq_length = args.seq_length
        self.seed = args.seed
        self.split = args.split
        self.num_samples = args.train_samples

    def __call__(self):
        with distribute.data_placement_scope():
            x = flow.data.megatron_gpt_mmap_data_loader(
                data_file_prefix=self.dataset,
                seq_length=self.seq_length,
                num_samples=self.num_samples,
                batch_size=self.batch_size,
                dtype=flow.int64,
                shuffle=True,
                random_seed=self.seed,
                split_sizes=self.split,
                split_index=0,
                parallel_distribution=distribute.get_data_parallel_dist(),
                name=self.name,
            )

        # embedding is on pipeline first stage
        with distribute.layer_placement_scope(0):
            data = flow.slice(x, begin=(None, 0), size=(None, self.seq_length))

        # loss is on pipeline last stage
        with distribute.layer_placement_scope(-1):
            labels = flow.slice(x, begin=(None, 1), size=(None, self.seq_length))

        return data, labels
