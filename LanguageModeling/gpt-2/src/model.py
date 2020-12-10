import os
import json
import math
import numpy as np
import oneflow as flow


def softmax(x, axis=-1):
    return flow.nn.softmax(x, axis=axis)


def gelu(x):
    return flow.math.gelu(x)


def norm(x, axis=-1, epsilon=1e-5, name=None):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    assert len(x.shape) == 3
    y = flow.layers.layer_norm(x, begin_norm_axis=axis, epsilon=epsilon, name=name)
    return flow.flatten(y, start_dim=0, end_dim=2)


def linear(name, x, nf, *, model_parallel_mode=None, w_init_stdev=0.02):
    assert len(x.shape) == 2
    nx = x.shape[-1]
    w_sbp = None
    b_sbp = None

    if model_parallel_mode == "s1":
        w_sbp = flow.distribute.split(1)
        b_sbp = flow.distribute.split(0)
    elif model_parallel_mode == "s0":
        w_sbp = flow.distribute.split(0)
        b_sbp = flow.distribute.broadcast()
    else:
        pass

    with flow.scope.namespace(name):
        w = flow.get_variable(
            name="weight",
            shape=(nx, nf),
            dtype=x.dtype,
            initializer=flow.random_normal_initializer(stddev=w_init_stdev),
            distribute=w_sbp,
        )
        b = flow.get_variable(
            name="bias",
            shape=(nf,),
            dtype=x.dtype,
            initializer=flow.constant_initializer(0.0),
            distribute=b_sbp,
        )

        c = flow.matmul(x, w, name="matmul")
        if w.split_axis == 0 and x.split_axis == 1:
            c = flow.parallel_cast(
                c,
                distribute=flow.distribute.broadcast(),
                gradient_distribute=flow.distribute.broadcast(),
            )
        c = flow.nn.bias_add(c, b, name="biasadd")
        c = flow.reshape(c, (b, s, nf), name="reshape")

    return c


class GPT2(object):
    def __init__(self, args, name="gpt2"):
        self.name = name
        self.n_vocab = args.padded_vocab_size
        self.n_ctx = args.n_ctx
        self.n_embd = args.n_embd
        self.n_head = args.n_head
        self.n_layer = args.n_layer
        self.seq_len = args.seq_len
        self.batch_size = args.batch_size
        self.embedding_dropout = args.embedding_dropout
        self.attention_dropout = args.attention_dropout
        self.hidden_dropout = args.hidden_dropout
        self.wte_split = args.wte_split
        self.wpe_split = args.wpe_split
        self.embd_parallel = args.embd_parallel
        self.decoder_parallel = args.decoder_parallel
        self.use_fp16 = args.use_fp16
        self.use_big_c_attn = args.use_big_c_attn
        self.checkpoint_activations = args.checkpoint_activations
        self.checkpoint_matmul = args.checkpoint_matmul
        self.checkpoint_variable = args.checkpoint_variable

        assert self.n_embd % self.n_head == 0

    def forward(self, x, past=None):
        assert len(x.shape) == 2
        assert x.shape[0] == self.batch_size
        assert x.shape[-1] == self.seq_len

        outputs = {}
        presents = []

        with flow.scope.namespace(self.name):
            h, wte = self.embedding(x)
            for i in range(self.n_layer):
                h, present = self.transformer_layer(f"h{i}", h, past=past)
                presents.append(present)

            outputs["presents"] = presents
            h = norm(h, name="layernorm_f")
            logits = flow.matmul(h, wte, transpose_b=True)
            outputs["logits"] = logits  # maybe reshape

        return outputs

    def embedding(self, x):
        """ position embedding and token embedding
        """
        wpe_sbp, wte_sbp = None, None
        if self.embd_parallel:
            wpe_sbp = flow.distribute.broadcast()
            wte_sbp = flow.distribute.split(0)

        wpe = flow.get_variable(
            "wpe",
            shape=(self.n_ctx, self.n_embd),
            initializer=flow.random_normal_initializer(stddev=0.01),
            distribute=wpe_sbp,
        )
        wte = flow.get_variable(
            "wte",
            shape=(self.n_vocab, self.n_embd),
            initializer=flow.random_normal_initializer(stddev=0.02),
            distribute=wte_sbp,
        )

        if self.use_fp16:
            # x = flow.amp_white_identity(x)
            wpe = flow.amp_white_identity(wpe)
            wte = flow.amp_white_identity(wte)

        if self.embd_parallel:
            x = flow.parallel_cast(x, distribute=flow.distribute.broadcast())

        h = flow.gather(wte, x, name="embd_gather")
        h = h + wpe
        h = flow.nn.dropout(h, rate=self.embedding_dropout, name="embd_dropout")

        if self.embd_parallel:
            h = flow.parallel_cast(
                h,
                distribute=flow.distribute.broadcast(),
                gradient_distribute=flow.distribute.split(0),
            )

        return h, wte

    def transformer_layer(self, name, x, *, past):
        assert len(x.shape) == 3
        b, s, e = tuple(x.shape)
        assert b == self.batch_size
        assert s == self.seq_len
        assert e == self.n_embd

        with flow.scope.namespace(name):
            x = flow.identity(x)
            with flow.experimental.scope.config(
                checkpointing=self.checkpoint_activations
            ):
                x = norm(x, name="layernorm_1")
                a, present = self.attn(x, past=past)
                x = x + a
                x = norm(x, name="layernorm_2")
                m = self.mlp(x)
                x = x + m

        return x, present

    def attn(self, x, *, past):
        assert len(x.shape) == 2
        e = x.shape[-1]

        def split_heads(x):
            # From [batch, sequence, features] to [batch, heads, sequence, features]
            assert len(x.shape) == 3
            b, s, _ = tuple(x.shape)
            x = flow.reshape(x, (b, s, self.n_head, -1))
            return flow.transpose(x, perm=[0, 2, 1, 3])

        def merge_heads(x):
            # Reverse of split_heads
            x = flow.transpose(x, [0, 2, 1, 3])
            b, s, _, _ = x.shape
            return flow.reshape(x, (b * s, -1))

        def multihead_attn(q, k, v):
            # q, k, v have shape [batch, heads, sequence, features]
            w = flow.matmul(q, k, transpose_b=True)
            w = flow.math.fused_scale_tril(
                w, fill_value=float("-inf"), scale=(1.0 / math.sqrt(float(v.shape[-1])))
            )
            w = softmax(w)
            w = flow.nn.dropout(w, rate=self.attention_dropout, name="attn_dropout")
            a = flow.matmul(w, v)
            return a

        with flow.scope.namespace("attn"):
            parallel_mode = "s0" if self.decoder_parallel else None
            if self.use_big_c_attn:
                c = linear("c_attn", x, e * 3, model_parallel_mode=parallel_mode)
                assert len(c.shape) == 3
                assert c.shape[-1] == e * 3
                b, s, _ = tuple(c.shape)
                c = flow.reshape(c, (b, s, e, 3))
                q = flow.slice(
                    c, begin=[None, None, None, 0], size=[None, None, None, 1]
                )
                k = flow.slice(
                    c, begin=[None, None, None, 1], size=[None, None, None, 1]
                )
                v = flow.slice(
                    c, begin=[None, None, None, 2], size=[None, None, None, 1]
                )
            else:
                q = linear("q_attn", x, e, model_parallel_mode=parallel_mode)
                k = linear("k_attn", x, e, model_parallel_mode=parallel_mode)
                v = linear("v_attn", x, e, model_parallel_mode=parallel_mode)

            q, k, v = map(split_heads, [q, k, v])
            present = []  # TODO: tf.stack([k, v], axis=1)
            if past is not None:
                pk, pv = tf.unstack(past, axis=1)
                k = tf.concat([pk, k], axis=-2)
                v = tf.concat([pv, v], axis=-2)
            a = multihead_attn(q, k, v)
            a = merge_heads(a)

            parallel_mode = "s1" if self.decoder_parallel else None
            a = linear("c_proj", a, e, model_parallel_mode=parallel_mode)
            a = flow.nn.dropout(a, rate=self.hidden_dropout, name="hidden_dropout_1")
            return a, present

    def mlp(self, x):
        assert len(x.shape) == 2
        e = x.shape[-1]

        with flow.scope.namespace("mlp"):
            parallel_mode = "s1" if self.decoder_parallel else None
            h = linear("c_fc", x, e * 4, model_parallel_mode=parallel_mode)
            h = gelu(h)
            assert h.shape[-1] == e * 4
            parallel_mode = "s0" if self.decoder_parallel else None
            h = linear("c_proj", h, e, model_parallel_mode=parallel_mode)
            h = flow.nn.dropout(h, rate=self.hidden_dropout, name="hidden_dropout_2")
            return h

    def loss(self, labels, logits, loss_parallel=False):
        assert len(labels.shape) == 2
        assert len(logits.shape) == 3
        b, s = tuple(labels.shape)
        assert labels.shape == logits.shape[:-1]
        assert logits.shape[-1] == self.n_vocab
        assert b == self.batch_size
        assert s == self.seq_len

        with flow.scope.namespace("loss"):
            labels = flow.slice(labels, begin=(None, 1), size=(None, s - 1))
            labels = flow.pad(labels, paddings=((0, 0), (0, 1)), constant_value=0.0)

            if loss_parallel:
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels,
                    logits=logits.with_distribute(flow.distribute.split(1)),
                )
            else:
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

            loss = flow.slice(loss, begin=(None, 0), size=(None, s - 1))
            return flow.math.reduce_mean(loss)
