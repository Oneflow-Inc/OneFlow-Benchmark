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


def GlueBERT(
    input_ids_blob,
    input_mask_blob,
    token_type_ids_blob,
    label_blob,
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
    label_num=2,
    replace_prob=None,
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
    pooled_output = PooledOutput(
        sequence_output=backbone.sequence_output(),
        hidden_size=hidden_size,
        initializer_range=initializer_range
    )
    loss, _, logit_blob = _AddClassficationLoss(
        input_blob=pooled_output,
        label_blob=label_blob,
        hidden_size=hidden_size,
        label_num=label_num,
        initializer_range=initializer_range,
        scope_name='classification'
    )

    return loss, logit_blob


def PooledOutput(sequence_output, hidden_size, initializer_range):
    with flow.scope.namespace("bert-pooler"):
        first_token_tensor = flow.slice(
            sequence_output, [None, 0, 0], [None, 1, -1])
        first_token_tensor = flow.reshape(
            first_token_tensor, [-1, hidden_size])
        pooled_output = bert_util._FullyConnected(
            first_token_tensor,
            input_size=hidden_size,
            units=hidden_size,
            weight_initializer=bert_util.CreateInitializer(initializer_range),
            name="dense",
        )
        pooled_output = flow.math.tanh(pooled_output)
    return pooled_output


def _AddClassficationLoss(input_blob, label_blob, hidden_size, label_num, initializer_range,
                          scope_name='classification'):
    with flow.scope.namespace(scope_name):
        output_weight_blob = flow.get_variable(
            name="output_weights",
            shape=[label_num, hidden_size],
            dtype=input_blob.dtype,
            # initializer=bert_util.CreateInitializer(initializer_range),
            initializer=flow.random_normal_initializer(
                mean=0.0, stddev=initializer_range, seed=None, dtype=None)
        )
        output_bias_blob = flow.get_variable(
            name="output_bias",
            shape=[label_num],
            dtype=input_blob.dtype,
            initializer=flow.constant_initializer(0.0),
        )
        logit_blob = flow.matmul(
            input_blob, output_weight_blob, transpose_b=True)
        logit_blob = flow.nn.bias_add(logit_blob, output_bias_blob)
        pre_example_loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logit_blob, labels=label_blob
        )
        loss = pre_example_loss
        return loss, pre_example_loss, logit_blob
