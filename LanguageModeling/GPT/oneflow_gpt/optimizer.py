import oneflow.compatible.single_client as flow


def get_lr_scheduler(args):
    # set up warmup strategy
    warmup = None
    if args.lr_warmup_iters is not None and args.lr_warmup_iters > 0:
        warmup = flow.optimizer.warmup.linear(args.lr_warmup_iters, 0)

    lr_decay_alpha = args.min_lr / args.lr
    # set up learning rate scheduler
    if args.lr_decay_style == "cosine" and args.lr_decay_iters is not None:
        lr_scheduler = flow.optimizer.CosineScheduler(
            base_lr=args.lr,
            steps=args.lr_decay_iters,
            alpha=lr_decay_alpha,
            warmup=warmup,
        )
    else:
        raise NotImplementedError("not supported yet")

    return lr_scheduler


def make_optimizer(args):
    lr_scheduler = get_lr_scheduler(args)

    loss_scale_policy = None
    if args.fp16:
        if args.loss_scale is not None:
            loss_scale_policy = flow.optimizer.loss_scale.static_loss_scale(
                args.loss_scale
            )
        else:
            loss_scale_policy = flow.optimizer.loss_scale.dynamic_loss_scale(
                initial_loss_scale=args.initial_loss_scale,
                increment_period=args.loss_scale_window,
            )

    if args.optimizer == "adamw":
        optimizer = flow.optimizer.AdamW(
            lr_scheduler,
            do_bias_correction=True,
            loss_scale_policy=loss_scale_policy,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            epsilon=args.adam_eps,
            weight_decay_excludes=["bias", "LayerNorm", "layernorm"],
            weight_decay=args.weight_decay,
            grad_clipping=flow.optimizer.grad_clipping.by_global_norm(args.clip_grad),
        )
    else:
        raise NotImplementedError("not supported yet")

    return optimizer
