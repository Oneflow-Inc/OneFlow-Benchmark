import math
import oneflow as flow

def layer_norm_2D(
    inputs,
    center: bool = True,
    scale: bool = True,
    trainable: bool = True,
    begin_norm_axis: int = 1,
    begin_params_axis: int = -1,
    epsilon: float = 1e-5,
    parallel_hierarchy=[2, 2], 
    parallel_distribution=["B", "B"],
    name: str = "LayerNorm",
):
    if center is False and scale is False:
        trainable = False

    beta = None
    gamma = None

    param_shape = inputs.shape[begin_params_axis:]
    if center:
        with flow.scope.namespace(name):
            beta = flow.get_variable(
                name="beta",
                shape=param_shape,
                dtype=inputs.dtype,
                initializer=flow.constant_initializer(0.0),
                trainable=trainable,
                model_name="beta",
                reuse=False,
                parallel_hierarchy=parallel_hierarchy,
                parallel_distribution=parallel_distribution,
            )
    if scale:
        with flow.scope.namespace(name):
            gamma = flow.get_variable(
                name="gamma",
                shape=param_shape,
                dtype=inputs.dtype,
                initializer=flow.constant_initializer(1.0),
                trainable=trainable,
                model_name="gamma",
                reuse=False,
                parallel_hierarchy=parallel_hierarchy,
                parallel_distribution=parallel_distribution,
            )

    op_builder = (
        flow.user_op_builder(name)
        .Op("layer_norm")
        .Input("x", [inputs])
        .Output("y")
        .Output("mean")
        .Output("inv_variance")
    )
    if beta is not None:
        op_builder.Input("beta", [beta])
    if gamma is not None:
        op_builder.Input("gamma", [gamma])
        op_builder.Output("normalized")
    op_builder.Attr("center", center)
    op_builder.Attr("scale", scale)
    op_builder.Attr("begin_norm_axis", begin_norm_axis)
    op_builder.Attr("begin_params_axis", begin_params_axis)
    op_builder.Attr("epsilon", epsilon)
    return op_builder.Build().InferAndTryRun().RemoteBlobList()[0]


def softmax(x, axis=-1):
    return flow.nn.softmax(x, axis=axis)


def gelu(x):
    return flow.math.gelu(x)


def flatten(x, start_dim=0, end_dim=-1):
    # flow.flatten need to be added to auto mixed precision clear list
    # we use reshape as a stand in for flatten
    # return flow.flatten(x, start_dim, end_dim)
    ndim = len(x.shape)
    if start_dim < 0:
        start_dim += ndim
    if end_dim < 0:
        end_dim += ndim
    assert start_dim >= 0
    assert start_dim <= end_dim
    assert end_dim <= ndim

    dims = []
    for i, dim_size in enumerate(x.shape):
        if i <= start_dim or i > end_dim:
            dims.append(dim_size)
        elif i > start_dim and i <= end_dim:
            dims[-1] *= dim_size
        else:
            raise ValueError
    
    print(f"flatten start={start_dim} end={end_dim} from {x.shape} to {dims}")
    return flow.reshape(x, dims)


def norm(x, axis=-1, epsilon=1e-5, name=None):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    assert len(x.shape) == 3, name
    y = flow.layers.layer_norm(x, begin_norm_axis=axis, epsilon=epsilon, name=name)
    return flatten(y, start_dim=0, end_dim=1)


def norm_2d(x, axis=-1, epsilon=1e-5, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"], name=None):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    assert len(x.shape) == 3, name
    y = layer_norm_2D(x, begin_norm_axis=axis, epsilon=epsilon, parallel_hierarchy=parallel_hierarchy, parallel_distribution=parallel_distribution, name=name)
    #return flatten(y, start_dim=0, end_dim=1)
    return flow.flatten(y, 0, 1) #can't process sbp in reshape op.cpp, use flatten op


def col_parallel_linear(name, x, nf, parallel_hierarchy, *, w_init_stdev=0.02):
    if len(parallel_hierarchy) == 1:
        weight_parallel_distribution = ["S(1)"]
        bias_parallel_distribution = ["S(0)"]
        #x_grad_parallel_distribution = ["S(0)"]
    elif len(parallel_hierarchy) == 2:
        weight_parallel_distribution = ["B", "S(1)"]
        bias_parallel_distribution = ["B", "S(0)"]
        #x_grad_parallel_distribution = ["S(0)", "B"]
    else:
        assert 0

    with flow.scope.namespace(name):
        if len(x.shape) == 3:
            x = flow.flatten(x, start_dim=0, end_dim=1)
        else:
            assert len(x.shape) == 2
        nx = x.shape[-1]

        weight = flow.get_variable(
            name="weight",
            shape=(nx, nf),
            dtype=x.dtype,
            initializer=flow.random_normal_initializer(stddev=w_init_stdev),
            parallel_hierarchy=parallel_hierarchy,
            parallel_distribution=weight_parallel_distribution,
        )

        bias = flow.get_variable(
            name="bias",
            shape=(nf,),
            dtype=x.dtype,
            initializer=flow.constant_initializer(0.0),
            parallel_hierarchy=parallel_hierarchy,
            parallel_distribution=bias_parallel_distribution,
        )

        c = flow.matmul(x, weight, name="matmul")
        print("c.shape", c.shape, "x.shape", x.shape, "weight.shape", weight.shape, "bias.shape", bias.shape)
        c = flow.nn.bias_add(c, bias, name="biasadd")
        print("c.shape", c.shape)


    return c


def row_parallel_linear(name, x, nf, parallel_hierarchy, *, w_init_stdev=0.02):
    if len(parallel_hierarchy) == 1:
        weight_parallel_distribution = ["S(0)"]
        bias_parallel_distribution = ["B"]
        c_parallel_distribution = ["B"]
    elif len(parallel_hierarchy) == 2:
        weight_parallel_distribution = ["B", "S(0)"]
        bias_parallel_distribution = ["B", "B"]
        c_parallel_distribution = ["S(0)", "B"]
    else:
        assert 0

    with flow.scope.namespace(name):
        if len(x.shape) == 3:
            x = flow.flatten(x, start_dim=0, end_dim=1)
        else:
            assert len(x.shape) == 2
        nx = x.shape[-1]

        weight = flow.get_variable(
            name="weight",
            shape=(nx, nf),
            dtype=x.dtype,
            initializer=flow.random_normal_initializer(stddev=w_init_stdev),
            parallel_hierarchy=parallel_hierarchy,
            parallel_distribution=weight_parallel_distribution,
        )
        bias = flow.get_variable(
            name="bias",
            shape=(nf,),
            dtype=x.dtype,
            initializer=flow.constant_initializer(0.0),
            parallel_hierarchy=parallel_hierarchy,
            parallel_distribution=bias_parallel_distribution,
        )
        c = flow.matmul(x, weight, name="matmul")
        c = flow.hierarchical_parallel_cast(
            c, parallel_hierarchy=parallel_hierarchy, 
            parallel_distribution=c_parallel_distribution,
            grad_mode="manual",
            grad_parallel_hierarchy=parallel_hierarchy,
            grad_parallel_distribution=c_parallel_distribution
        )
        c = flow.nn.bias_add(c, bias, name="biasadd")

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
        self.embd_parallel_hierarchy = args.embd_parallel_hierarchy
        self.attn_parallel_hierarchy = args.attn_parallel_hierarchy 
        self.use_fp16 = args.use_fp16
        self.use_big_fc = args.use_big_fc
        self.checkpoint_activations = args.checkpoint_activations
        self.checkpoint_matmul = args.checkpoint_matmul
        self.checkpoint_variable = args.checkpoint_variable

        assert self.n_embd % self.n_head == 0


    def forward(self, x, past=None):
        assert len(x.shape) == 2
        b, s = x.shape
        assert x.shape[0] == self.batch_size
        assert x.shape[-1] == self.seq_len

        outputs = {}
        presents = []

        with flow.scope.namespace(self.name):
            h, wte = self.embedding(x)
            # h(S0, B) wte(B, S0)

            # set h SBP before decoder layer 
            pd = ["S(0)", "B"] if len(self.attn_parallel_hierarchy) == 2 else ["S(0)"]
            #h = flow.hierarchical_parallel_cast(
            #    h, parallel_hierarchy=self.attn_parallel_hierarchy, 
            #    parallel_distribution=pd,
            #)
            for i in range(self.n_layer):
                h, present = self.encoder_layer(f"h{i}", h, past=past)
                presents.append(present)
            outputs["presents"] = presents
            #h = norm(h, name="layernorm_f")
            h = norm_2d(h, parallel_hierarchy = [2, 2], parallel_distribution=["B", "B"], name="layernorm_f") #[S0, B]

            wte = flow.hierarchical_parallel_cast(
                    wte, parallel_hierarchy=[2, 2], 
                    parallel_distribution=["B", "S(0)"],
                    grad_mode="manual",
                    grad_parallel_hierarchy=[2, 2],
                    grad_parallel_distribution=["B", "S(0)"]
            )
            h = flow.hierarchical_parallel_cast(
                    h, parallel_hierarchy=[2, 2], 
                    parallel_distribution=["S(0)", "B"],
                    grad_mode="manual",
                    grad_parallel_hierarchy=[2, 2],
                    grad_parallel_distribution=["S(0)", "B"]
            )#for layernorm_f dy is B not P
            logits = flow.matmul(h, wte, transpose_b=True) #h(S0, B) wte(B, S0) out(S0, S1)  h shape (4096, 768) wte shape (50688, 768) logits shape (4096, 50688)
            print("h shape", h.shape, "wte shape", wte.shape, "logits shape", logits.shape)
            logits = flow.hierarchical_parallel_cast(
                logits, parallel_hierarchy=[2, 2], 
                parallel_distribution=["S(0)", "S(0)"],
                grad_mode="manual",
                grad_parallel_hierarchy=[2, 2],
                grad_parallel_distribution=["S(0)", "S(1)"]
            )
            outputs["logits"] = logits

        return outputs


    def embedding(self, x):
        """ position embedding and token embedding
        """
        if len(self.embd_parallel_hierarchy) == 1:
            wte_parallel_distribution = ["S(0)"]
        elif len(self.embd_parallel_hierarchy) == 2:
            wte_parallel_distribution = ["B", "S(0)"]
        else:
            assert 0, '1D, 2D SBP only'
        wpe_parallel_distribution = ["B" for _ in self.embd_parallel_hierarchy]

        wpe = flow.get_variable(
            "wpe",
            shape=(self.n_ctx, self.n_embd),
            initializer=flow.random_normal_initializer(stddev=0.01),
            parallel_hierarchy=[2, 2],
            parallel_distribution=["B", "B"],
        )
        wte = flow.get_variable(
            "wte",
            shape=(self.n_vocab, self.n_embd),
            initializer=flow.random_normal_initializer(stddev=0.02),
            parallel_hierarchy=[2, 2],
            parallel_distribution=["B", "S(0)"],
        )

        if self.use_fp16:
            # x = flow.amp_white_identity(x)
            wpe = flow.amp_white_identity(wpe)
            wte = flow.amp_white_identity(wte)

        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], 
            parallel_distribution=["S(0)", "B"],
            grad_mode="manual",
            grad_parallel_hierarchy=[4],
            grad_parallel_distribution=["S(0)"]
        )
        wte_model = flow.hierarchical_parallel_cast(
            wte, parallel_hierarchy=[2, 2], 
            parallel_distribution=["B", "S(0)"],
            grad_mode="manual",
            grad_parallel_hierarchy=[2, 2],
            grad_parallel_distribution=["B", "S(0)"]
        )
        h = flow.gather(wte_model, x, name="embd_gather")
        h = flow.hierarchical_parallel_cast(
            h, parallel_hierarchy=[2, 2], 
            parallel_distribution=["S(0)", "B"],
            grad_mode="manual",
            grad_parallel_hierarchy=[2, 2],
            grad_parallel_distribution=["S(0)", "B"]
        )
        h = h + wpe
        h = flow.nn.dropout(h, rate=self.embedding_dropout, name="embd_dropout")
        return h, wte

    def encoder_layer(self, name, x, *, past):
        assert len(x.shape) == 3
        b, s, e = x.shape
        assert b == self.batch_size
        assert s == self.seq_len
        assert e == self.n_embd

        with flow.scope.namespace(name):
            x = flow.identity(x)
            with flow.experimental.scope.config(
                checkpointing=self.checkpoint_activations
            ):
                #norm1 = norm(x, name="layernorm_1")
                norm1 = norm_2d(x, parallel_hierarchy = [2, 2], parallel_distribution=["B", "B"], name="layernorm_1")
                a, present = self.attn(norm1, past=past)
                x = x + a
                norm2 = norm_2d(x, parallel_hierarchy = [2, 2], parallel_distribution=["B", "B"], name="layernorm_2")
                print("norm2 shape", norm2.shape)
                m = self.mlp(norm2)
                m = flow.hierarchical_parallel_cast(
                    m, parallel_hierarchy=[2, 2], 
                    parallel_distribution=["S(0)", "B"],
                    grad_mode="manual",
                    grad_parallel_hierarchy=[4],
                    grad_parallel_distribution=["S(0)"]
                )
                x = x + m

        return x, present

    def attn(self, x, *, past):
        e = x.shape[-1]

        def split_heads(x):
            # From [batch, sequence, features] to [batch, heads, sequence, features]
            x = flow.reshape(x, (self.batch_size, self.seq_len, self.n_head, -1))
            return flow.transpose(x, perm=[0, 2, 1, 3])

        def merge_heads(x):
            # Reverse of split_heads
            x = flow.transpose(x, [0, 2, 1, 3])
            return flow.flatten(x, start_dim=2, end_dim=3)

        def multihead_attn(q, k, v):
            # q, k, v have shape [batch, heads, sequence, features]
            w = flow.matmul(q, k, transpose_b=True) #q (b,n,s,h)["S(0)", "S(1)"] k(b,n,s,h)["S(0)", "S(1)"]  -> out(b,n,s,s) ["S(0)", "S(1)"]
            w = flow.math.fused_scale_tril(
                w, fill_value=float("-inf"), scale=(1.0 / math.sqrt(float(v.shape[-1])))
            )
            w = softmax(w) #(b,n,s,s) ["S(0)", "S(1)"] axis=-1
            w = flow.nn.dropout(w, rate=self.attention_dropout)
            a = flow.matmul(w, v) #w(b,n,s,s) ["S(0)", "S(1)"]  v(b,n,s,h)["S(0)", "S(1)"] -> a(b,n,s,h)["S(0)", "S(1)"]
            return a

        with flow.scope.namespace("attn"):
            if self.use_big_fc:
                c = col_parallel_linear("c_attn", x, e * 3, self.attn_parallel_hierarchy)
                assert len(c.shape) == 2
                assert c.shape[-1] == e * 3
                bs = c.shape[0]
                c = flow.reshape(c, (bs, e, 3))
                q = flow.slice(
                    c, begin=[None, None, 0], size=[None, None, 1]
                )
                k = flow.slice(
                    c, begin=[None, None, 1], size=[None, None, 1]
                )
                v = flow.slice(
                    c, begin=[None, None, 2], size=[None, None, 1]
                )
            else:
                x = flow.hierarchical_parallel_cast(
                    x, parallel_hierarchy=[2, 2], 
                    parallel_distribution=["S(0)", "B"],
                    grad_mode="manual",
                    grad_parallel_hierarchy=[2, 2],
                    grad_parallel_distribution=["S(0)", "B"]
                ) #for grad P->B
                q = col_parallel_linear("q_attn", x, e, [2, 2]) # x ["S(0)", "B"] w[B,S1] b[B,S0] -> [S0, S1]
                k = col_parallel_linear("k_attn", x, e, [2, 2])
                v = col_parallel_linear("v_attn", x, e, [2, 2])

            q, k, v = map(split_heads, [q, k, v])
            # TODO: tf.stack([k, v], axis=1)
            present = []  
            if past is not None:
                pk, pv = tf.unstack(past, axis=1)
                k = tf.concat([pk, k], axis=-2)
                v = tf.concat([pv, v], axis=-2)

            print("q k v shape", q.shape, k.shape, v.shape)
            a = multihead_attn(q, k, v) #(b,n,s,h)[S0, S1]
            print("before merge_heads a", a.shape) #(b,n,s,h)[S0, S1]
            a = merge_heads(a)
            print("after merge_heads a", a.shape)  #(b,s,e)[S0, S2]
            a = row_parallel_linear("c_proj", a, e, [2, 2])
            a = flow.nn.dropout(a, rate=self.hidden_dropout) #[S0,B]
            a = flow.reshape(a, (self.batch_size, self.seq_len, self.n_embd))
            return a, present

    def mlp(self, x):
        assert len(x.shape) == 2
        e = x.shape[-1]

        with flow.scope.namespace("mlp"):
            h = col_parallel_linear("c_fc", x, e * 4, [2, 2])
            h = gelu(h)
            assert h.shape[-1] == e * 4
            h = row_parallel_linear("c_proj", h, e, [2, 2])
            h = flow.nn.dropout(h, rate=self.hidden_dropout)
            h = flow.hierarchical_parallel_cast(
                x, parallel_hierarchy=[4], 
                parallel_distribution=["S(0)"],
                grad_mode="manual",
                grad_parallel_hierarchy=[2, 2],
                grad_parallel_distribution=["S(0)", "B"]
            ) # to 1d for reshape
            h = flow.reshape(h, (self.batch_size, self.seq_len, self.n_embd))
            return h

    def loss(self, labels, logits, parallel_loss=False):
        assert len(labels.shape) == 2
        b, s = labels.shape
        assert len(logits.shape) == 2
        bs, v = logits.shape
        assert b == self.batch_size
        assert s == self.seq_len
        assert bs == b * s
        assert v == self.n_vocab
        print("labels", labels.shape, labels.dtype)
        labels = flow.hierarchical_parallel_cast(
            labels, parallel_hierarchy=[2, 2], 
            parallel_distribution=["S(0)", "S(0)"],
            grad_mode="manual",
            grad_parallel_hierarchy=[4],
            grad_parallel_distribution=["S(0)"]
        )
        with flow.scope.namespace("loss"):
            labels = flow.slice(labels, begin=(None, 1), size=(None, s - 1))

            labels = flow.pad(labels, paddings=((0, 0), (0, 1)), constant_value=0.0)

            labels = flow.flatten(labels)

            parallel_loss=False
            if parallel_loss:
                # split vocab dim (v)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels,
                    logits=logits.with_distribute(flow.distribute.split(1)),
                )
            else:
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

            loss = flow.hierarchical_parallel_cast(
                loss, parallel_hierarchy=[4], 
                parallel_distribution=["S(0)"],
                grad_mode="manual",
                grad_parallel_hierarchy=[2, 2],
                grad_parallel_distribution=["S(0)", "S(0)"]
                )#to 1d for reshape
            loss = flow.reshape(loss, (b, s))
            loss = flow.hierarchical_parallel_cast(
                loss, parallel_hierarchy=[2, 2], 
                parallel_distribution=["S(0)", "S(0)"],
                grad_mode="manual",
                grad_parallel_hierarchy=[4],
                grad_parallel_distribution=["S(0)"]
                )#to 2d after reshape
            loss = flow.slice(loss, begin=(None, 0), size=(None, s - 1))
            loss = flow.hierarchical_parallel_cast(
                loss, parallel_hierarchy=[4], 
                parallel_distribution=["S(0)"],
                grad_mode="manual",
                grad_parallel_hierarchy=[2, 2],
                grad_parallel_distribution=["S(0)", "S(0)"]
                )
            return flow.math.reduce_mean(loss)
