import os
import json
import math
import numpy as np
import oneflow as flow

#def default_hparams():
#    return {
#        n_vocab=0,
#        n_ctx=1024,
#        n_embd=768,
#        n_head=12,
#        n_layer=12,
#    }

def softmax(x, axis=-1):
    return flow.nn.softmax(x, axis=axis)

def gelu(x):
    return flow.math.gelu(x)

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    return flow.layers.layer_norm(x, name=scope, begin_norm_axis=axis, epsilon=epsilon)

def conv1d(x, scope, nf, *, w_init_stdev=0.02, split=None):
    with flow.scope.namespace(scope):
        *start, nx = x.shape
        w = flow.get_variable(name='w', shape=[nx, nf], dtype=x.dtype,
                              initializer=flow.random_normal_initializer(stddev=w_init_stdev))
        b = flow.get_variable(name='b', shape=[nf], dtype=x.dtype,
                              initializer=flow.constant_initializer(0.0))
        split = None
        if split == 0:
            w = flow.parallel_cast(w, distribute=flow.distribute.split(0))
        elif split == 1:
            x = flow.parallel_cast(x, distribute=flow.distribute.broadcast())
            w = flow.parallel_cast(w, distribute=flow.distribute.split(1))
            b = flow.parallel_cast(b, distribute=flow.distribute.split(0))

        c = flow.matmul(flow.reshape(x, [-1, nx]), w)
        c = flow.nn.bias_add(c, b)
        if split == 0:
            c = flow.parallel_cast(c, distribute=flow.distribute.broadcast())
        return flow.reshape(c, start + [nf])

def mlp(x, scope, n_state):
    with flow.scope.namespace(scope):
        nx = x.shape[-1]
        h = conv1d(x, 'c_fc', n_state, split=1)
        h = gelu(h)
        return conv1d(h, 'c_proj', nx, split=0)


class GPT2(object):
    def __init__(self, args, scope='model'):
        self.scope = scope
        self.n_vocab = args.n_vocab
        self.n_ctx = args.n_ctx
        self.n_embd = args.n_embd
        self.n_head = args.n_head
        self.n_layer = args.n_layer
        self.sequence = args.seq_len
        self.batch = args.batch_size_per_device * args.gpu_num_per_node * args.num_nodes

    def forward(self, X, past=None, split=None):
        with flow.scope.namespace(self.scope):
            results = {}
            flow.identity_n([X])
            wpe = flow.get_variable('wpe', [self.n_ctx, self.n_embd],
                                    initializer=flow.random_normal_initializer(stddev=0.01))
            wte = flow.get_variable('wte', [self.n_vocab, self.n_embd],
                                    initializer=flow.random_normal_initializer(stddev=0.02),
                                    #distribute=flow.distribute.split(0),
            )
            split = False#True
            if split:
                wte = flow.parallel_cast(wte, distribute=flow.distribute.split(0))
                X = flow.parallel_cast(X, distribute=flow.distribute.broadcast())
            h = flow.gather(wte, X)# + flow.reshape(wpe, shape=(1, self.n_ctx, self.n_embd))
            if split:
                h = flow.parallel_cast(h, distribute=flow.distribute.split(0),
                                       gradient_distribute=flow.distribute.broadcast())
            h = h + flow.reshape(wpe, shape=(1, self.n_ctx, self.n_embd))
            presents = []
            for layer in range(self.n_layer):
                h, present = self.block(h, 'h%d' % layer, past=past)
                presents.append(present)
            results['presents'] = presents
            h = norm(h, 'ln_f')

            *start, _ = h.shape
            h_flat = flow.reshape(h, [-1, self.n_embd])
            #if split:
            #    wte = flow.parallel_cast(wte, distribute=flow.distribute.broadcast())
            if split:
                h_flat = flow.parallel_cast(h_flat, distribute=flow.distribute.broadcast())
            logits = flow.matmul(h_flat, wte, transpose_b=True)
            if split:
                logits = flow.parallel_cast(logits, distribute=flow.distribute.split(0))
            logits = flow.reshape(logits, start + [self.n_vocab])
            results['logits'] = logits
        return results 

    #def vocab_embedding(self, X, split=None):
    def block(self, x, scope, *, past):
        with flow.scope.namespace(scope):
            nx = x.shape[-1]
            assert nx == self.n_embd
            a, present = self.attn(norm(x, 'ln_1'), 'attn', nx, past=past)
            x = x + a
            m = mlp(norm(x, 'ln_2'), 'mlp', nx*4)
            x = x + m
            return x, present

    def attn(self, x, scope, n_state, *, past):
        assert len(x.shape) == 3  # Should be [batch, sequence, features]
        assert n_state % self.n_head == 0
        if past is not None:
            assert len(past.shape) == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]
    
        def split_heads(x):
            # From [batch, sequence, features] to [batch, heads, sequence, features]
            *start, _ = x.shape
            return flow.transpose(flow.reshape(x, start + [self.n_head, -1]), perm=[0, 2, 1, 3])
    
        def merge_heads(x):
            # Reverse of split_heads
            x = flow.transpose(x, [0, 2, 1, 3])
            *start, _, _ = x.shape
            return flow.reshape(x, start + [-1])
    
        def mask_attn_weights(w):
            # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
            assert len(w.shape) == 4
            _, _, nd, ns = w.shape
            b = flow.math.tril(flow.constant(value=int(-1), dtype=flow.int32, shape=(nd, ns)), ns-nd)
            b = b + flow.ones_like(like=b, dtype=flow.int32)
            b = flow.reshape(b, [1, 1, nd, ns])
            w = flow.masked_fill(w, b, float('-inf'))
            return w
    
        def multihead_attn(q, k, v):
            # q, k, v have shape [batch, heads, sequence, features]
            w = flow.matmul(q, k, transpose_b=True)
            w = w * (1.0 / math.sqrt(float(v.shape[-1])))
    
            w = mask_attn_weights(w)
            w = softmax(w)
            a = flow.matmul(w, v)
            return a

        with flow.scope.namespace(scope):
            #c = conv1d(x, 'c_attn', n_state*3)
            #q, k, v = map(split_heads, tf.split(c, 3, axis=2))
            q = conv1d(x, 'q_attn', n_state, split=1)
            k = conv1d(x, 'k_attn', n_state, split=1)
            v = conv1d(x, 'v_attn', n_state, split=1)
            q, k, v = map(split_heads, [q, k, v])

            present = [] # TODO: tf.stack([k, v], axis=1)
            if past is not None:
                pk, pv = tf.unstack(past, axis=1)
                k = tf.concat([pk, k], axis=-2)
                v = tf.concat([pv, v], axis=-2)
            a = multihead_attn(q, k, v)
            a = merge_heads(a)
            a = conv1d(a, 'c_proj', n_state, split=0)
            return a, present

