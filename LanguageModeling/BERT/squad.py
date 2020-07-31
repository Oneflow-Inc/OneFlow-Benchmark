"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import oneflow as flow
import bert as bert_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util


def SQuAD(
    input_ids_blob,
    input_mask_blob,
    token_type_ids_blob,
    vocab_size,
    seq_length=512,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=16,
    initializer_range=0.02,
    ):

    backbone = bert_util.BertBackbone(
        input_ids_blob=input_ids_blob,
        input_mask_blob=input_mask_blob,
        token_type_ids_blob=token_type_ids_blob,
        vocab_size=vocab_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=type_vocab_size,
        initializer_range=initializer_range,
    )

    with flow.scope.namespace("cls-squad"):
        final_hidden = backbone.sequence_output()
        final_hidden_matrix = flow.reshape(final_hidden, [-1, hidden_size])
        logits = bert_util._FullyConnected(
                    final_hidden_matrix,
                    hidden_size,
                    units=2,
                    weight_initializer=bert_util.CreateInitializer(initializer_range),
                    name='output')
        logits = flow.reshape(logits, [-1, seq_length, 2])

        start_logits = flow.slice(logits, [None, None, 0], [None, None, 1])
        end_logits = flow.slice(logits, [None, None, 1], [None, None, 1])

    return start_logits, end_logits
