# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import numpy as np
from tokenizer.tokenizer import build_tokenizer


def build_dataset(args):
    """Helper function to select and build dataset."""
    if args.task == "LAMBADA":
        return _build_lambada_dataset(args)

    raise NotImplementedError("dataset for {} task is not " "implemented.".format(task))


class _LambadaDataset:
    def __init__(self, path, pad_idx, tokenizer, seq_len, strict=False):
        print("> building lambada dataset from {} ...".format(path))
        self.seq_len = seq_len
        self.pad_idx = pad_idx
        self.tokenizer = tokenizer
        self.strict = strict

        self.tokens = []
        self.labels = []
        with open(path, "r") as f:
            for line in f.readlines():
                text = json.loads(line)["text"]
                tokens, labels = self.get_tokens(text)
                self.tokens.append(tokens)
                self.labels.append(labels)

    def get_tokens(self, text):
        if not self.strict:
            tokens = self.tokenizer.tokenize(text)
            return tokens[:-1], [tokens[-1]]
        last_token = text.split()[-1]
        start_idx = text.rfind(last_token)
        beginning_tokens = self.tokenizer.tokenize(text[:start_idx].strip())
        last_token = self.tokenizer.tokenize(" " + last_token)
        return beginning_tokens, last_token

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        num_tokens = len(tokens)
        pad_mask = [0] * num_tokens
        labels = self.labels[idx]
        pad_mask += [1] * len(labels)
        tokens = tokens + labels
        num_tokens = len(tokens)
        if num_tokens < self.seq_len + 1:
            num_pad = self.seq_len + 1 - num_tokens
            pad_mask += [0] * (num_pad)
            tokens += [self.pad_idx] * num_pad
        pad_mask = np.array(pad_mask[1:])

        return {"text": np.array(tokens), "pad_mask": pad_mask}


def _build_lambada_dataset(args):
    """Build lambada dataset."""
    tokenizer = build_tokenizer(args)

    assert len(args.valid_data) == 1
    val_dataset = _LambadaDataset(
        args.valid_data[0],
        tokenizer.eod,
        tokenizer,
        args.seq_length,
        args.strict_lambada,
    )
    print(" > found {} samples.".format(len(val_dataset)))

    return val_dataset
