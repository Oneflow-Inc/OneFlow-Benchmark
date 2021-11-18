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

### Meta Fine-tuning
def MFTBERT(
    input_ids_blob,
    input_mask_blob,
    token_type_ids_blob,
    label_blob,
    input_domain,
    vocab_size,
    input_weight,
    layer_indexes=[-1],
    num_domains=3,
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
    lambda_=0.1, # 两个loss的权重
    replace_prob=None,
    get_output=False, # add by wjn，当为True时，表示只获取BERT的embedding
):
    backbone = bert_util.BertBackbone( # 创建BERT基础模型
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
    if get_output:
        last_layer_output = backbone.sequence_output() # [bz, len, dim]
        # indices = flow.tensor([0])
        # cls_output = flow.gather(last_layer_output, indices=indices, axis=1) # [bz, dim] 取CLS对应的embedding
        cls_output = flow.math.reduce_mean(last_layer_output, axis=1)
        # print('cls_output.shape=', cls_output.shape)
        return cls_output

    # Meta Fine-tuning: CLS classification loss
    pooled_output = PooledOutput(
        sequence_output=backbone.sequence_output(),
        hidden_size=hidden_size,
        initializer_range=initializer_range
    )
    # 添加分类损失函数
    cls_loss, _, logit_blob = _AddClassficationLoss(
        input_blob=pooled_output,
        label_blob=label_blob,
        hidden_size=hidden_size,
        label_num=label_num,
        initializer_range=initializer_range,
        scope_name='classification'
    )

    ## Meta Fine-tuing: Corrupted Domain Classification loss
    corrupted_loss = _AddCorruptedDomainCLSLoss(backbone.all_encoder_layers_, label_blob, input_domain, hidden_size, num_domains,
                               layer_indexes, initializer_range)

    # 对corrupted_loss进行加权求和
    corrupted_loss = flow.math.multiply(corrupted_loss, input_weight)

    return cls_loss + lambda_ * corrupted_loss, logit_blob
    # return cls_loss, logit_blob



### standard Fine-tuning
def BERT(
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
    backbone = bert_util.BertBackbone( # 创建BERT基础模型
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

    # Meta Fine-tuning: CLS classification loss
    pooled_output = PooledOutput(
        sequence_output=backbone.sequence_output(),
        hidden_size=hidden_size,
        initializer_range=initializer_range
    )
    # 添加分类损失函数
    cls_loss, _, logit_blob = _AddClassficationLoss(
        input_blob=pooled_output,
        label_blob=label_blob,
        hidden_size=hidden_size,
        label_num=label_num,
        initializer_range=initializer_range,
        scope_name='classification'
    )

    return cls_loss, logit_blob




# BERT的CLS token进行分类
def PooledOutput(sequence_output, hidden_size, initializer_range):
    with flow.scope.namespace("bert-pooler"):
        first_token_tensor = flow.slice( # 切片操作，提取每个样本的第一个token（[CLS]）对应的表征向量
            sequence_output, [None, 0, 0], [None, 1, -1])
        first_token_tensor = flow.reshape(
            first_token_tensor, [-1, hidden_size])
        pooled_output = bert_util._FullyConnected( # 添加一个全连接层
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
        logit_blob = flow.matmul(  # output_weight_blob先转置，再与input_bob相乘
            input_blob, output_weight_blob, transpose_b=True) # [batch_size, label_num]
        logit_blob = flow.nn.bias_add(logit_blob, output_bias_blob)
        # 获得每个样本的loss [batch_size]
        pre_example_loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logit_blob, labels=label_blob
        )
        loss = pre_example_loss
        # print('loss.shape=', loss.shape) # [bz, ]
        loss = flow.math.reduce_mean(loss)
        # print('loss.shape=', loss.shape) # [1, ]
        return loss, pre_example_loss, logit_blob





def _AddCorruptedDomainCLSLoss(input_blob, label_blob, input_domain, hidden_size, num_domains,
                               layer_indexes, initializer_range,
                               scope_name='corrupted_domain_classification'):
    '''
    input_blob: [layer_num, batch_size, seq_length, hidden_size]
    '''
    # print('len(input_blob)=', len(input_blob))
    domain_logits = dict()
    total_domain_loss = 0.
    with flow.scope.namespace(scope_name):
        domain_embedded_matrix = flow.get_variable("domain_projection", [num_domains, hidden_size],
                                                 initializer=flow.truncated_normal_initializer(stddev=0.02))
        domain_embedded = flow.gather(params=domain_embedded_matrix, indices=input_domain, axis=0)
        domain_embedded = flow.reshape(domain_embedded, shape=[-1, hidden_size])
        # print('domain_embedded.shape=', domain_embedded.shape) # [bz, dim]
        for layer_index in layer_indexes:
            content_tensor = flow.math.reduce_mean(input_blob[layer_index], axis=1)
            content_tensor_with_domains = domain_embedded + content_tensor
            # print('content_tensor.shape=', content_tensor.shape) # [bz, dim]
            domain_weights = flow.get_variable("domain_weights", [num_domains, hidden_size], initializer=flow.truncated_normal_initializer(stddev=0.02))
            domain_bias = flow.get_variable("domain_bias", [num_domains], initializer=flow.zeros_initializer())
            # print('content_tensor_with_domains.shape=', content_tensor_with_domains.shape) # [bz, dim]
            current_domain_logits = flow.matmul(content_tensor_with_domains, domain_weights, transpose_b=True)
            current_domain_logits = flow.nn.bias_add(current_domain_logits, domain_bias)

            # domain_logits["domain_logits_"+str(layer_index)] = current_domain_logits

            # 计算当前layer对应的loss
            shuffle_domain_labels = flow.random.shuffle(input_domain) # 随机生成错误的domain标签
            shuffle_domain_labels = flow.reshape(shuffle_domain_labels, shape=[-1])
            # print('shuffle_domain_labels.shape=', shuffle_domain_labels.shape) # [bz]
            shuffle_domain_labels = flow.squeeze(shuffle_domain_labels)
            # one_hot_labels = flow.one_hot(shuffle_domain_labels, depth=num_domains, dtype=flow.float32)
            # print('current_domain_logits.shape=', current_domain_logits.shape) # [bz, domain_num]
            # print('one_hot_labels.shape=', one_hot_labels.shape)
            domain_loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                logits=current_domain_logits, labels=shuffle_domain_labels
            )
            total_domain_loss += domain_loss
        total_domain_loss = total_domain_loss / len(layer_indexes)
        return total_domain_loss
