import numpy as np
import random
import oneflow as flow
import string
from typing import Tuple, List, Union
from data_utils.utils import InputFeatures, InputExample

domain_list = {
    'g1': ['sst-2', 'mr', 'cr'],
    'g2': ['mnli', 'snli'],
    'g3': ['mrpc', 'qqp']
}

class_list = {
    'g1': [0, 1],
    'sst-2': [0, 1],
    'mr': [0, 1],
    'cr': [0, 1],
    'g2': ['contradiction', 'entailment', 'neutral'],
    'mnli': ['contradiction', 'entailment', 'neutral'],
    'snli': ['contradiction', 'entailment', 'neutral'],
    'g3': [0, 1],
    'mrpc': [0, 1],
    'qqp': [0, 1],
}

# 在 cross-task中，每个group的task需要使用对应的prompt-encoder，因此需要为每个task给予一个group内部的下标编号，用于指定对应的prompt-encoder
task_to_id = {
    'SST-2': 0,
    'mr': 1,
    'cr': 2,
    'mnli': 0,
    'snli': 1,
    'mrpc': 0,
    'qqp': 1,
}


# 名称对齐
# 终端传入的参数会被默认小写，而磁盘目录会有大写，因此要做一个转换
data_to_name = {
    'SST-2': 'SST-2',
    'sst-5': 'sst-5',
    'mr': 'mr',
    'cr': 'cr',
    'mpqa': 'mpqa',
    'subj': 'subj',
    'trec': 'trec',
    'CoLA': 'CoLA',
    'MRPC': 'MRPC',
    'QQP': 'QQP',
    'STS-B': 'STS-B',
    'MNLI': 'MNLI',
    'SNLI': 'SNLI',
    'QNLI': 'QNLI',
    'RTE': 'RTE',
    'sst-2': 'SST-2',
    'cola': 'CoLA',
    'mrpc': 'MRPC',
    'qqp': 'QQP',
    'sts-b': 'STS-B',
    'mnli': 'MNLI',
    'snli': 'SNLI',
    'qnli': 'QNLI',
    'rte': 'RTE',
    'g1': 'g1',
    'g2': 'g2',
    'g3': 'g3',
    'g4': 'g4'
}

class Preprocessor(object):
    """

    """

    def __init__(self, config, tokenizer, seed: int = 42):
        """
        Create a new PVP.

        :param wrapper: the wrapper for the underlying language model
        :param verbalizer_file: an optional file that contains the verbalizer to be used
        :param seed: a seed to be used for generating random numbers if necessary
        """
        self.config = config
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        self.label_map = {label: i for i, label in enumerate(self.config.label_list)}

    @property
    def mask(self) -> str:
        """Return the underlying LM's mask token"""
        return '[MASK]'

    @property
    def mask_id(self) -> int:
        """Return the underlying LM's mask id"""
        return self.tokenizer.mask_token_id

    @property
    def max_num_verbalizers(self) -> int:
        """Return the maximum number of verbalizers across all labels"""
        # 正常情况下结果为1
        return max(len(self.verbalize(label)) for label in self.config.label_list)

    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True

    # @staticmethod
    # def remove_final_punc(s: Union[str, Tuple[str, bool]]):
    #     """Remove the final punctuation mark"""
    #     if isinstance(s, tuple):
    #         return remove_final_punc(s[0]), s[1]
    #     return s.rstrip(string.punctuation)
    #
    # @staticmethod
    # def lowercase_first(s: Union[str, Tuple[str, bool]]):
    #     """Lowercase the first character"""
    #     if isinstance(s, tuple):
    #         return self.lowercase_first(s[0]), s[1]
    #     return s[0].lower() + s[1:]

    def encode(self, example: InputExample, priming: bool = False, labeled: bool = False) \
            -> Tuple[List[int], List[int]]:
        """
        Encode an input example using this pattern-verbalizer pair.
        将输入的句子样本转化为feature

        :param example: the input example to encode
        :param priming: whether to use this example for priming
        :param labeled: if ``priming=True``, whether the label should be appended to this example
        :return: A tuple, consisting of a list of input ids and a list of token type ids
        """
        # 获得预训练分词工具
        tokenizer = self.tokenizer
        # 不同的Task有不同的PVP get_parts方法，获得相应的成分。
        # 例如parts_a = [texta, 'x', 'x', MASK, '.]
        # block_flag_a = [0. 1, 0, 0]
        # parts_a, parts_b, block_flag_a, block_flag_b = self.get_parts(example)

        text_a, text_b = example.text_a, example.text_b
        text_a = self.shortenable(text_a)
        parts_a = [text_a]
        parts_b = []
        if text_b is not None:
            text_b = self.shortenable(text_b)
            parts_b = [text_b]
        # kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        # parts_a = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_a if x]
        parts_a = [(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)), s) for x, s in parts_a if x]

        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)), s) for x, s in parts_b if x]

        # self.truncate(parts_a, parts_b, max_length=self.config.seq_length)
        num_special = self.tokenizer.num_special_tokens_to_add(bool(parts_b))

        # 根据最大长度对text进行截断
        # print('parts_b=', parts_b)
        self.truncate(parts_a, parts_b, max_length=self.config.seq_length - num_special)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        # tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else None
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else []

        if tokens_b:
            input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)
        else:
            input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a)
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a)

        ### return input_ids, token_type_ids
        return input_ids, token_type_ids


    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def truncate(self, parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]], max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - max_length # 总长度如果超过设定的最大程度，则删除

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)

    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        # print('input_ids=', input_ids)
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels



    def get_input_features(self, example: InputExample, labelled: bool, priming: bool = False,
                           **kwargs) -> InputFeatures:
        # 获得PVP（模板句子+label mapping）
        input_ids, token_type_ids = self.encode(example)

        attention_mask = [1] * len(input_ids)
        padding_length = self.config.seq_length - len(input_ids)

        if padding_length < 0:
            raise ValueError(f"Maximum sequence length is too small, got {len(input_ids)} input ids")

        input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)  # wordid序列+padding
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        assert len(input_ids) == self.config.seq_length
        assert len(attention_mask) == self.config.seq_length
        assert len(token_type_ids) == self.config.seq_length

        example_label = example.label
        example_task = example.task  # add by wjn 表示当前样本所属的task
        # add by wjn 只有当数字型的label（0,1），可能真实标签是字符串（'0', '1'），因此需要进行转换判断
        if example_label not in self.label_map.keys():
            if type(example_label) == int:
                example_label = str(example_label)
            elif type(example_label) == str:
                example_label = int(example_label)

        label = self.label_map[example_label] if example.label is not None else -100  # 当前样本的类标
        task = task_to_id[example_task]  # add by wjn 表示当前task对应group内的编号
        # task = example_task
        logits = example.logits if example.logits else [-1]

        # if labelled:
        #     # 获得一个序列中[MASK]所在的索引
        #     # eg 长度为5的序列[-1 -1 1 -1 -1]，可知第3个token为[MASK]
        #     mlm_labels = self.pvp.get_mask_positions(input_ids)
        # else:
        #     mlm_labels = [-1] * self.config.seq_length

        # masked_lm_positions = mlm_labels.index(1)
        # # print('masked_lm_positions=', masked_lm_positions)
        # label_word = self.pvp.verbalize(example_label)[0]
        # masked_lm_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(label_word))[
        #     0]  # list [label word token id]
        # masked_lm_weights = 1.0

        return InputFeatures(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             task=task,
                             label=label,
                             logits=logits,
                             idx=example.guid,)