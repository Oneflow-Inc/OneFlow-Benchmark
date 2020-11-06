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

def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with flow.scope.namespace(scope):
        *start, nx = x.shape
        w = flow.get_variable(name='w', shape=[nx, nf], dtype=x.dtype,
                              initializer=flow.random_normal_initializer(stddev=w_init_stdev))
        b = flow.get_variable(name='b', shape=[nf], dtype=x.dtype, 
                              initializer=flow.constant_initializer(0.0))

        c = flow.matmul(flow.reshape(x, [-1, nx]), w)
        c = flow.nn.bias_add(c, b)
        return flow.reshape(c, start + [nf]) 

def mlp(x, scope, n_state):
    with flow.scope.namespace(scope):
        nx = x.shape[-1]
        h = conv1d(x, 'c_fc', n_state)
        h = gelu(h)
        return conv1d(h, 'c_proj', nx)


class GPT2(object):
    def __init__(self, args, scope='model'):
        self.scope = scope
        with open(os.path.join(args.cfg_dir, 'hparams.json')) as f:
            hparams = json.load(f)
            print('hparams', hparams)
            self.n_vocab = hparams['n_vocab']
            self.n_ctx = hparams['n_ctx']
            self.n_embd = hparams['n_embd']
            self.n_head = hparams['n_head']
            self.n_layer = hparams['n_layer']
        self.batch, self.sequence = args.batch_size, args.seq_len
    
    def forward(self, X, past=None):
        with flow.scope.namespace(self.scope):
            results = {}
            wpe = flow.get_variable('wpe', [self.n_ctx, self.n_embd],
                                    initializer=flow.random_normal_initializer(stddev=0.01))
            wte = flow.get_variable('wte', [self.n_vocab, self.n_embd],
                                    initializer=flow.random_normal_initializer(stddev=0.02))

            h = flow.gather(wte, X) + flow.reshape(wpe, shape=(1, self.n_ctx, self.n_embd))
            presents = []
            for layer in range(self.n_layer):
                h, present = self.block(h, 'h%d' % layer, past=past)
                presents.append(present)
            results['presents'] = presents
            h = norm(h, 'ln_f')

            *start, _ = h.shape
            h_flat = flow.reshape(h, [-1, self.n_embd])
            logits = flow.matmul(h_flat, wte, transpose_b=True)
            logits = flow.reshape(logits, start + [self.n_vocab])
            results['logits'] = logits
        return results 

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
            b = flow.masked_fill(w, b, float('-inf'))
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
            q = conv1d(x, 'q_attn', n_state)
            k = conv1d(x, 'k_attn', n_state)
            v = conv1d(x, 'v_attn', n_state)
            q, k, v = map(split_heads, [q, k, v])

            present = [] # TODO: tf.stack([k, v], axis=1)
            if past is not None:
                pk, pv = tf.unstack(past, axis=1)
                k = tf.concat([pk, k], axis=-2)
                v = tf.concat([pv, v], axis=-2)
            a = multihead_attn(q, k, v)
            a = merge_heads(a)
            a = conv1d(a, 'c_proj', n_state)
            return a, present


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present


def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def model(hparams, X, past=None, scope='model'):#, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            if layer == 10:
                tf.add_to_collection('checkpoints', h)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')
        print(h)

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results
