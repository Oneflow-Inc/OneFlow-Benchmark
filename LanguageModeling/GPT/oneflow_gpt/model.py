import math
import numpy as np
import oneflow.compatible.single_client as flow

from oneflow_gpt import distribute
from oneflow_gpt.config import get_args


class GPTModel(object):
    def __init__(self, name):
        self.name = name

        args = get_args()
        self.batch_size = args.global_batch_size // args.num_accumulation_steps
        self.seq_length = args.seq_length
        self.hidden_size = args.hidden_size
        self.vocab_size = args.padded_vocab_size

        self.embedding = Embedding(
            self.batch_size, self.seq_length, self.hidden_size, self.vocab_size
        )
        self.transformer = Transformer(
            self.batch_size, self.seq_length, self.hidden_size
        )

    def __call__(self, tokens):
        """
        tokens shape: (batch_size, seq_length)
        dp sbp: S(0)
        2d sbp: [S(0), B]
        """
        assert len(tokens.shape) == 2
        assert tokens.shape[0] == self.batch_size
        assert tokens.shape[1] == self.seq_length

        with flow.scope.namespace(self.name):
            hidden_states, token_embeddings = self.embedding(tokens)
            h = self.transformer(hidden_states)
            lgs = self.logits(h, token_embeddings)

        return lgs

    def logits(self, hidden_states, token_embeddings):
        """
        shape sig: (batch_size * seq_length, hidden_size) x (hidden_size, vocab_size)(transposed)
            -> (batch_size * seq_length, vocab_size)
        dp sbp sig: S(0) x B -> S(0)
        2d sbp sig: [S(0), B] x [B, S(1)](transposed) -> [S(0), S(1)]
        """
        assert len(hidden_states.shape) == 3
        assert np.prod(hidden_states.shape[0:2]) == self.batch_size * self.seq_length
        assert hidden_states.shape[-1] == self.hidden_size
        assert len(token_embeddings.shape) == 2
        assert token_embeddings.shape[0] == self.vocab_size
        assert token_embeddings.shape[1] == self.hidden_size

        with distribute.layer_placement_scope(-1):
            if (
                hidden_states.shape[0] == self.seq_length
                and hidden_states.shape[1] == self.batch_size
            ):
                # [s, b, H] -> [b, s, H]
                h = flow.transpose(hidden_states, [1, 0, 2])
            elif (
                hidden_states.shape[0] == self.batch_size
                and hidden_states.shape[1] == self.seq_length
            ):
                h = hidden_states
            else:
                raise ValueError(f"invalid hidden states shape {hidden_states.shape}")

            # [s, b, H] or [b, s, H] -> [b * s, H]
            h = flow.flatten(h, start_dim=0, end_dim=1)
            # 2d sbp sig: [S(0), B] x [B, S(1)](transposed) -> [S(0), S(1)]
            # grad 2d sbp sig: [S(0), S(1)] x [B, S(0)] -> [S(0), P] -> [S(0), B]
            h = distribute.backward_p2b_parallel_cast(h)
            lgs = flow.matmul(h, token_embeddings, transpose_b=True)

        return lgs


class Embedding(object):
    def __init__(self, batch_size, seq_length, hidden_size, vocab_size):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        args = get_args()
        self.embedding_dropout_rate = args.hidden_dropout
        self.use_fp16 = args.fp16

        self.wpe_initializer = flow.random_normal_initializer(
            stddev=args.init_method_std
        )
        self.wte_initializer = flow.random_normal_initializer(
            stddev=args.init_method_std
        )

    def __call__(self, tokens):
        """
        tokens shape: (batch_size, seq_length)
        dp sbp: S(0)
        2d sbp: [S(0), B]
        """
        assert len(tokens.shape) == 2
        assert tokens.shape[0] == self.batch_size
        assert tokens.shape[1] == self.seq_length

        with distribute.layer_placement_scope(0):
            wpe = flow.get_variable(
                "wpe",
                shape=(self.seq_length, self.hidden_size),
                initializer=self.wpe_initializer,
                parallel_distribution=distribute.get_wpe_parallel_dist(),
            )
            wte = flow.get_variable(
                "wte",
                shape=(self.vocab_size, self.hidden_size),
                initializer=self.wte_initializer,
                parallel_distribution=distribute.get_wte_parallel_dist(),
            )

            # 2d sbp sig: [B, S(0)] x [S(0), B] -> [S(0), P] -> [S(0), B]
            # grad 2d sbp sig: [S(0), B](dy) x [S(0), B](index) x [B, S(0)](x)
            #                   -> [P, S(0)](dx) -> [B, S(0)](wte_grad)
            if self.use_fp16:
                h = flow.gather(flow.amp_white_identity(wte), tokens)
                wpe = flow.amp_white_identity(wpe)
            else:
                h = flow.gather(wte, tokens)

            h = distribute.forward_p2b_parallel_cast(h) + wpe
            h = flow.nn.dropout(
                h, rate=self.embedding_dropout_rate, name="embd_dropout"
            )

        return h, wte


class Transformer(object):
    def __init__(self, batch_size, seq_length, hidden_size):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size

        args = get_args()
        self.multihead_attention_fusion = args.multihead_attention_fusion
        self.num_layers = args.num_layers
        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(
                TransformerLayer(
                    f"h{i}",
                    i + 1,
                    batch_size,
                    seq_length,
                    hidden_size,
                    initializer=flow.random_normal_initializer(
                        stddev=args.init_method_std
                    ),
                    output_layer_initializer=flow.random_normal_initializer(
                        stddev=(args.init_method_std / math.sqrt(2.0 * self.num_layers))
                    ),
                )
            )

    def __call__(self, hidden_states):
        """
        hidden_states shape: (batch_size, seq_length, hidden_size)
        data parallel sbp: S(0)
        2d sbp: [S(0), B]
        """
        assert len(hidden_states.shape) == 3
        assert hidden_states.shape[0] == self.batch_size
        assert hidden_states.shape[1] == self.seq_length
        assert hidden_states.shape[2] == self.hidden_size

        if self.multihead_attention_fusion:
            with distribute.layer_placement_scope(0):
                # [b, s, H] -> [s, b, H] for multihead_attention_fusion
                h = flow.transpose(hidden_states, [1, 0, 2])
        else:
            h = hidden_states

        for i in range(self.num_layers):
            with distribute.layer_placement_scope(i):
                h = self.layers[i](h)

        # final layernorm
        with distribute.layer_placement_scope(-1):
            h = layernorm("layernorm_f", h)

        return h


class TransformerLayer(object):
    def __init__(
        self,
        name,
        layer_id,
        batch_size,
        seq_length,
        hidden_size,
        initializer=None,
        output_layer_initializer=None,
    ):
        self.name = name
        self.layer_id = layer_id
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size

        args = get_args()
        self.enable_profiling = args.profile_transformer_layer
        self.attn = SelfAttention(
            layer_id,
            batch_size,
            seq_length,
            hidden_size,
            args.hidden_dropout,
            initializer,
            output_layer_initializer,
        )
        self.mlp = MLP(
            batch_size,
            seq_length,
            hidden_size,
            args.hidden_dropout,
            initializer,
            output_layer_initializer,
        )

        self.checkpoint_activations = args.checkpoint_activations

    def __call__(self, hidden_states):
        """
        hidden_states shape: (batch_size, seq_length, hidden_size)
        data parallel sbp: S(0)
        2d sbp: [S(0), B]
        """
        assert len(hidden_states.shape) == 3
        assert hidden_states.shape[-1] == self.hidden_size
        assert np.prod(hidden_states.shape[:-1]) == self.batch_size * self.seq_length

        h = hidden_states
        with flow.scope.namespace(self.name):
            if self.enable_profiling:
                h = flow.profiler.nvtx_start(h, mark_prefix=f"transformer-{self.name}")

            h = flow.identity(h)
            with flow.experimental.scope.config(
                checkpointing=self.checkpoint_activations
            ):
                # input layernorm
                norm1 = layernorm("layernorm_1", h)
                # attention
                h = h + self.attn(norm1)
                # output layernorm
                norm2 = layernorm("layernorm_2", h)
                # mlp
                h = h + self.mlp(norm2)

            if self.enable_profiling:
                h = flow.profiler.nvtx_end(h, mark_prefix=f"transformer-{self.name}")

        return h


class SelfAttention(object):
    def __init__(
        self,
        layer_id,
        batch_size,
        seq_length,
        hidden_size,
        hidden_dropout_rate,
        initializer=None,
        output_layer_initializer=None,
    ):
        self.layer_id = layer_id
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.hidden_dropout_rate = hidden_dropout_rate
        self.initializer = initializer
        self.output_layer_initializer = output_layer_initializer

        args = get_args()
        self.num_heads = args.num_attention_heads
        self.head_size = args.hidden_size // args.num_attention_heads
        self.attention_dropout_rate = args.attention_dropout
        self.scale_tril_softmax_dropout_fusion = args.scale_tril_softmax_dropout_fusion
        self.bias_dropout_fusion = args.bias_dropout_fusion
        self.multihead_attention_fusion = args.multihead_attention_fusion

        self.norm_factor = math.sqrt(float(self.head_size))
        self.coeff = 1.0
        if args.apply_query_key_layer_scaling:
            self.coeff = float(self.layer_id)
            self.norm_factor *= self.coeff

    def query_key_value(self, h):
        """
        Split input to q, k, v and split hidden states into heads,
            shape: (batch_size, seq_length, hidden_size)
                -> (batch_size, seq_length, num_attn_heads, head_size)
                -> (batch_size, num_attn_heads, seq_length, head_size)
        """
        assert len(h.shape) == 3

        # Note: 3 is between num_heads and head_size that ensure the features of heads of q, k, v is contiguously arranged
        new_shape = (
            h.shape[0],
            h.shape[1],
            self.num_heads,
            3 * self.head_size,
        )
        if h.shape[0] == self.seq_length and h.shape[1] == self.batch_size:
            perm = [1, 2, 0, 3]
        elif h.shape[0] == self.batch_size and h.shape[1] == self.seq_length:
            perm = [0, 2, 1, 3]
        else:
            raise ValueError

        h = flow.reshape(h, new_shape)
        q, k, v = (
            flow.transpose(
                flow.slice(
                    h,
                    begin=[None, None, None, i * self.head_size],
                    size=[None, None, None, self.head_size],
                ),
                perm=perm,
            )
            for i in range(3)
        )
        return q, k, v

    def multihead_attn(self, q, k, v):
        """
        q, k, v shape: (batch_size, num_attn_heads, seq_length, head_size)
        """
        assert all(len(x.shape) == 4 for x in (q, k, v))
        assert all(x.shape[0] == self.batch_size for x in (q, k, v))
        assert all(x.shape[1] == self.num_heads for x in (q, k, v))
        assert all(x.shape[2] == self.seq_length for x in (q, k, v))
        assert all(x.shape[3] == self.head_size for x in (q, k, v))

        # q * k: batch_matmul
        # shape sig: (b, n, s, h) x (b, n, h, s)(transposed) -> (b, n, s, s)
        # data parallel sbp sig: S(0) x S(0) -> S(0)
        # 2d sbp sig: [S(0), S(1)] x [S(0), S(1)] -> [S(0), S(1)]
        qmk = flow.matmul(q, k, transpose_b=True, alpha=(1.0 / self.norm_factor))
        qmk = self.tril_softmax_dropout(qmk)
        # w * v: batch_matmul
        # shape sig: (b, n, s, s) x (b, n, s, h) -> (b, n, s, h)
        # data parallel sbp sig: S(0) x S(0) -> S(0)
        # 2d sbp sig: [S(0), S(1)] x [S(0), S(1)] -> [S(0), S(1)]
        return flow.matmul(qmk, v)

    def tril_softmax_dropout(self, x):
        if self.scale_tril_softmax_dropout_fusion:
            x = flow.math.fused_scale_tril_softmax_dropout(
                x,
                diagonal=0,
                scale=self.coeff,
                fill_value=float("-inf"),
                rate=self.attention_dropout_rate,
            )
        else:
            x = flow.math.fused_scale_tril(
                x, fill_value=float("-inf"), scale=self.coeff,
            )
            x = flow.nn.softmax(x, axis=-1)
            x = flow.nn.dropout(x, rate=self.attention_dropout_rate)

        return x

    def fused_multihead_attn(self, h):
        assert len(h.shape) == 3
        assert h.shape[0] == self.seq_length
        assert h.shape[1] == self.batch_size
        assert h.shape[2] == self.hidden_size * 3

        qmk, v = flow.nn.fused_self_attention_query_mul_key_and_value(
            h, head_size=self.head_size, alpha=(1.0 / self.norm_factor)
        )
        qmk = self.tril_softmax_dropout(qmk)
        return flow.matmul(qmk, v)

    def __call__(self, hidden_states):
        # hidden_states shape: (batch_size, seq_length, hidden_size)
        # or (seq_length, batch_size, hidden_size) [seq_len dim leading]
        # data parallel sbp: S(0)
        # 2d sbp: [S(0), B]
        assert len(hidden_states.shape) == 3
        assert hidden_states.shape[-1] == self.hidden_size
        if (
            hidden_states.shape[0] == self.batch_size
            and hidden_states.shape[1] == self.seq_length
        ):
            is_seq_len_dim_leading = False
        elif (
            hidden_states.shape[0] == self.seq_length
            and hidden_states.shape[1] == self.batch_size
        ):
            is_seq_len_dim_leading = True
        else:
            raise ValueError(f"invalid hidden states shape {hidden_states.shape}")

        h = hidden_states
        with flow.scope.namespace("attn"):
            h = col_parallel_linear(
                "c_attn", h, self.hidden_size * 3, weight_initializer=self.initializer,
            )
            if self.multihead_attention_fusion:
                h = self.fused_multihead_attn(h)
            else:
                q, k, v = self.query_key_value(h)
                h = self.multihead_attn(q, k, v)

            if is_seq_len_dim_leading:
                # (b, n, s, h) -> (s, b, n, h)
                h = flow.transpose(h, [2, 0, 1, 3])
            else:
                # (b, n, s, h) -> (b, s, n, h)
                h = flow.transpose(h, [0, 2, 1, 3])

            # (b, s, n, h) -> (b, s, H) or (s, b, n, h) -> (s, b, H)
            h = flow.flatten(h, start_dim=2)
            h = row_parallel_linear(
                "c_proj",
                h,
                self.hidden_size,
                weight_initializer=self.output_layer_initializer,
                dropout_rate=self.hidden_dropout_rate,
                bias_dropout_fusion=self.bias_dropout_fusion,
            )

        return h


class MLP(object):
    def __init__(
        self,
        batch_size,
        seq_length,
        hidden_size,
        hidden_dropout_rate,
        initializer=None,
        output_layer_initializer=None,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.hidden_dropout_rate = hidden_dropout_rate
        self.initializer = initializer
        self.output_layer_initializer = output_layer_initializer

        args = get_args()
        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.bias_dropout_fusion = args.bias_dropout_fusion

    def __call__(self, hidden_states):
        # hidden_states shape: (batch_size, seq_length, hidden_size)
        # data parallel sbp: S(0)
        # 2d sbp: [S(0), B]
        assert hidden_states.shape[-1] == self.hidden_size

        h = hidden_states
        with flow.scope.namespace("mlp"):
            h = col_parallel_linear(
                "c_fc",
                h,
                self.hidden_size * 4,
                weight_initializer=self.initializer,
                need_gelu=True,
                bias_gelu_fusion=self.bias_gelu_fusion,
            )
            h = row_parallel_linear(
                "c_proj",
                h,
                self.hidden_size,
                weight_initializer=self.output_layer_initializer,
                dropout_rate=self.hidden_dropout_rate,
                bias_dropout_fusion=self.bias_dropout_fusion,
            )

        # output hidden states shape: (batch_size * seq_length, hidden_size)
        # data parallel sbp: S(0)
        # 2d sbp: [S(0), B]
        return h


class ParallelSparseSoftmaxCrossEntropyLoss(object):
    def __init__(self, name="loss"):
        self.name = name

        args = get_args()
        self.batch_size = args.global_batch_size // args.num_accumulation_steps
        self.seq_length = args.seq_length
        self.vocab_size = args.padded_vocab_size

    def __call__(self, logits, labels):
        """
        logits shape: (batch_size * seq_length, vocab_size)
        logits dp sbp: S(0)
        logits 2d sbp: [S(0), S(1)]
        labels shape: (batch_size, seq_length)
        labels dp sbp: S(0)
        labels 2d sbp: [S(0), B]
        """
        assert len(logits.shape) == 3 or len(logits.shape) == 2
        if len(logits.shape) == 3:
            assert logits.shape[0] == self.batch_size
            assert logits.shape[1] == self.seq_length
            assert logits.shape[2] == self.vocab_size
        elif len(logits.shape) == 2:
            assert logits.shape[0] == self.batch_size * self.seq_length
            assert logits.shape[1] == self.vocab_size
        else:
            raise ValueError(f"invalid logits shape {logits.shape}")

        assert len(labels.shape) == 2
        assert labels.shape[0] == self.batch_size
        assert labels.shape[1] == self.seq_length

        with flow.scope.namespace(self.name):
            with distribute.layer_placement_scope(-1):
                if len(logits.shape) == 2:
                    labels = flow.flatten(labels)

                if distribute.get_dist_util().tensor_model_parallel_size > 1:
                    loss = flow.nn.distributed_sparse_softmax_cross_entropy_with_logits(
                        labels, logits
                    )
                else:
                    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                        labels, logits
                    )
                    # loss = flow.amp_white_identity(loss)

                loss = flow.math.reduce_mean(loss)

        return loss


def layernorm(
    name,
    x,
    axis=-1,
    epsilon=1e-5,
    params_parallel_dist=distribute.get_layernorm_params_parallel_dist(),
):
    """
    Normalize to mean = 0, std = 1, then do a diagonal affine transform.
    """
    begin_norm_axis = axis
    begin_params_axis = -1
    param_shape = x.shape[begin_params_axis:]

    with flow.scope.namespace(name):
        beta = flow.get_variable(
            name="beta",
            shape=param_shape,
            dtype=x.dtype,
            initializer=flow.constant_initializer(0.0),
            trainable=True,
            model_name="beta",
            reuse=False,
            parallel_distribution=params_parallel_dist,
        )

        gamma = flow.get_variable(
            name="gamma",
            shape=param_shape,
            dtype=x.dtype,
            initializer=flow.constant_initializer(1.0),
            trainable=True,
            model_name="gamma",
            reuse=False,
            parallel_distribution=params_parallel_dist,
        )

    return flow.nn.layer_norm(
        x, gamma, beta, begin_norm_axis, begin_params_axis, epsilon, name
    )


def get_linear_params(
    name,
    input_size,
    output_size,
    dtype,
    weight_initializer=flow.random_normal_initializer(stddev=0.02),
    bias_initializer=flow.constant_initializer(0.0),
    weight_parallel_dist=None,
    bias_parallel_dist=None,
):
    with flow.scope.namespace(name):
        weight = flow.get_variable(
            name="weight",
            shape=(input_size, output_size),
            dtype=dtype,
            initializer=weight_initializer,
            parallel_distribution=weight_parallel_dist,
        )
        bias = flow.get_variable(
            name="bias",
            shape=(output_size,),
            dtype=dtype,
            initializer=bias_initializer,
            parallel_distribution=bias_parallel_dist,
        )

    return weight, bias


def col_parallel_linear(
    name,
    x,
    output_size,
    weight_initializer,
    bias_initializer=flow.constant_initializer(0.0),
    weight_parallel_dist=distribute.get_col_linear_weight_parallel_dist(),
    bias_parallel_dist=distribute.get_col_linear_bias_parallel_dist(),
    need_gelu=False,
    bias_gelu_fusion=True,
):
    w, b = get_linear_params(
        name,
        x.shape[-1],
        output_size,
        x.dtype,
        weight_initializer=weight_initializer,
        bias_initializer=bias_initializer,
        weight_parallel_dist=weight_parallel_dist,
        bias_parallel_dist=bias_parallel_dist,
    )
    # 2d sbp sig: [S(0), B] x [B, S(1)] -> [S(0), S(1)]
    # data grad 2d sbp sig: [S(0), S(1)] x [B, S(0)](transposed) -> [S(0), P] -> [S(0), B]
    x = distribute.backward_p2b_parallel_cast(x)
    x = flow.matmul(x, w)
    if need_gelu:
        if bias_gelu_fusion:
            x = flow.nn.fused_bias_add_gelu(x, b, data_format="NHC")
        else:
            x = flow.nn.bias_add(x, b, data_format="NHC")
            x = flow.math.gelu(x)
    else:
        x = flow.nn.bias_add(x, b, data_format="NHC")

    return x


def row_parallel_linear(
    name,
    x,
    output_size,
    weight_initializer,
    bias_initializer=flow.constant_initializer(0.0),
    weight_parallel_dist=distribute.get_row_linear_weight_parallel_dist(),
    bias_parallel_dist=distribute.get_row_linear_bias_parallel_dist(),
    dropout_rate=0.1,
    bias_dropout_fusion=True,
):
    w, b = get_linear_params(
        name,
        x.shape[-1],
        output_size,
        x.dtype,
        weight_initializer=weight_initializer,
        bias_initializer=bias_initializer,
        weight_parallel_dist=weight_parallel_dist,
        bias_parallel_dist=bias_parallel_dist,
    )
    # 2d sbp sig: [S(0), S(1)] x [B, S(0)] -> [S(0), P] -> [S(0), B]
    # data grad 2d sbp sig: [S(0), B] x [B, S(1)](transposed) -> [S(0), S(1)]
    x = flow.matmul(x, w)
    x = distribute.forward_p2b_parallel_cast(x)
    if bias_dropout_fusion:
        x = flow.nn.fused_bias_add_dropout(x, b, data_format="NHC", rate=dropout_rate)
    else:
        x = flow.nn.bias_add(x, b, data_format="NHC")
        x = flow.nn.dropout(x, rate=dropout_rate)

    return x
