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
        #split = None
        *start, nx = x.shape
        w_sbp = flow.distribute.split(split) if split == 0 or split == 1 else None
        b_sbp = flow.distribute.split(0) if split == 1 else None
        print(scope, w_sbp, b_sbp)
        w = flow.get_variable(name='weight', shape=[nx, nf], dtype=x.dtype,
                              initializer=flow.random_normal_initializer(stddev=w_init_stdev),
                              distribute=w_sbp)
        b = flow.get_variable(name='bias', shape=[nf], dtype=x.dtype,
                              initializer=flow.constant_initializer(0.0),
                              distribute=b_sbp)
        
        if split == 1:
            x = flow.parallel_cast(x, distribute=flow.distribute.broadcast())

        c = flow.matmul(flow.reshape(x, [-1, nx], name='reshape1'), w)
        c = flow.nn.bias_add(c, b)
        if split == 0:
            c = flow.parallel_cast(c, distribute=flow.distribute.broadcast())
        return flow.reshape(c, start + [nf], name='reshape2')


class GPT2(object):
    def __init__(self, args, scope='model'):
        self.scope = scope
        #self.n_vocab = args.n_vocab
        self.n_vocab = args.padded_vocab_size
        self.n_ctx = args.n_ctx
        self.n_embd = args.n_embd
        self.n_head = args.n_head
        self.n_layer = args.n_layer
        self.sequence = args.seq_len
        self.batch = args.batch_size_per_device * args.gpu_num_per_node * args.num_nodes
        self.embedding_dropout = args.embedding_dropout
        self.attention_dropout = args.attention_dropout
        self.output_dropout = args.output_dropout
        assert (args.wte_split in [0, 1]) == (args.wpe_split in [0, 1])
        self.wte_split = args.wte_split
        self.wpe_split = args.wpe_split
        self.decoder_model_parallel = args.decoder_model_parallel
        self.use_fp16 = args.use_fp16
        self.checkpoint_activations = args.checkpoint_activations

    def forward(self, X, past=None, split=None):
        with flow.scope.namespace(self.scope):
            results = {}
            flow.identity_n([X])
            embd_model_parallel = self.wte_split in [0, 1]
            wte_sbp = flow.distribute.split(self.wte_split) if self.wte_split in [0, 1] else None 
            wpe_sbp = flow.distribute.split(self.wpe_split) if self.wpe_split in [0, 1] else None 
            print(wte_sbp, wpe_sbp)
            wpe = flow.get_variable('wpe', [self.n_ctx, self.n_embd], distribute=wpe_sbp, 
                                    initializer=flow.random_normal_initializer(stddev=0.01))
            wte = flow.get_variable('wte', [self.n_vocab, self.n_embd], distribute=wte_sbp,
                                    initializer=flow.random_normal_initializer(stddev=0.02))
            if embd_model_parallel:
                X = flow.parallel_cast(X, distribute=flow.distribute.broadcast())

            if self.use_fp16:
                X = flow.amp_white_identity(X)
            h = flow.gather(wte, X)# + flow.reshape(wpe, shape=(1, self.n_ctx, self.n_embd))
            h = h + flow.reshape(wpe, shape=(1, self.n_ctx, self.n_embd))
            h =  flow.nn.dropout(h, rate=self.embedding_dropout)
            presents = []
            for layer in range(self.n_layer):
                h, present = self.block(h, 'h%d' % layer, past=past)
                presents.append(present)
            results['presents'] = presents
            h = norm(h, 'layernorm_f')

            *start, _ = h.shape
            h_flat = flow.reshape(h, [-1, self.n_embd])
            if embd_model_parallel:
                h_flat = flow.parallel_cast(h_flat, distribute=flow.distribute.broadcast())
            logits = flow.matmul(h_flat, wte, transpose_b=True)

            if embd_model_parallel:
                logits = flow.parallel_cast(logits, distribute=wte_sbp)
            logits = flow.reshape(logits, start + [self.n_vocab])
            results['logits'] = logits
        return results 

    #def vocab_embedding(self, X, split=None):
    def block(self, x, scope, *, past):
        with flow.scope.namespace(scope):
            x = flow.identity(x)
            with flow.experimental.scope.config(checkpointing = self.checkpoint_activations):
                nx = x.shape[-1]
                assert nx == self.n_embd
                a, present = self.attn(norm(x, 'layernorm_1'), 'attn', nx, past=past)
                x = x + a
                m = self.mlp(norm(x, 'layernorm_2'), 'mlp', nx*4)
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
            return flow.math.tril(w, fill_value=float('-inf'))
    
        def multihead_attn(q, k, v):
            # q, k, v have shape [batch, heads, sequence, features]
            w = flow.matmul(q, k, transpose_b=True)
            w = w * (1.0 / math.sqrt(float(v.shape[-1])))
    
            w = mask_attn_weights(w)
            w = softmax(w)
            w = flow.nn.dropout(w, rate=self.attention_dropout)
            a = flow.matmul(w, v)
            return a

        with flow.scope.namespace(scope):
            split = 1 if self.decoder_model_parallel else None 
            c = conv1d(x, 'c_attn', n_state*3, split=split)
            q = flow.slice(c, begin=[None, None, 0], size=[None, None, n_state])
            k = flow.slice(c, begin=[None, None, n_state], size=[None, None, n_state])
            v = flow.slice(c, begin=[None, None, 2*n_state], size=[None, None, n_state])
            q, k, v = map(split_heads, [q, k, v])

            present = [] # TODO: tf.stack([k, v], axis=1)
            if past is not None:
                pk, pv = tf.unstack(past, axis=1)
                k = tf.concat([pk, k], axis=-2)
                v = tf.concat([pv, v], axis=-2)
            a = multihead_attn(q, k, v)
            a = merge_heads(a)
            split = 0 if self.decoder_model_parallel else None 
            a = conv1d(a, 'c_proj', n_state, split=split)
            a = flow.nn.dropout(a, rate=self.output_dropout)
            return a, present

    def mlp(self, x, scope, n_state):
        with flow.scope.namespace(scope):
            split = 1 if self.decoder_model_parallel else None 
            nx = x.shape[-1]
            h = conv1d(x, 'c_fc', n_state, split=split)
            h = gelu(h)
            split = 0 if self.decoder_model_parallel else None 
            h = conv1d(h, 'c_proj', nx, split=split)
            return flow.nn.dropout(h, rate=self.output_dropout)
