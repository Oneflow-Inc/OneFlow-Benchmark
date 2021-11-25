import contextlib
import oneflow.compatible.single_client as flow

from oneflow_gpt.config import get_args

_DIST_UTIL = None


class _DistributeUtil(object):
    def __init__(self):
        args = get_args()
        self._init_parallel_size(args)
        self._init_placement_group(args)
        self._init_parallel_hierarchy()

    def _init_parallel_size(self, args):
        self.world_size_ = args.num_gpus_per_node * args.num_nodes

        # Tensor model parallel size.
        self.tensor_model_parallel_size_ = min(
            args.tensor_model_parallel_size, self.world_size_
        )
        assert self.world_size_ % self.tensor_model_parallel_size_ == 0, (
            f"world size ({self.world_size_})"
            " is not divisible by "
            " tensor model parallel size ({self.tensor_model_parallel_size_})"
        )

        # Pipeline model parallel size.
        self.pipeline_model_parallel_size_ = min(
            args.pipeline_model_parallel_size,
            (self.world_size_ // args.tensor_model_parallel_size),
        )

        self.model_parallel_size = (
            self.pipeline_model_parallel_size_ * self.tensor_model_parallel_size_
        )
        assert self.world_size_ % self.model_parallel_size == 0, (
            f"world size ({self.world_size_}) is not divisible by"
            " tensor parallel size ({self.tensor_model_parallel_size_}) times,"
            " pipeline paralle size ({self.pipeline_model_parallel_size_})"
        )

        # Data parallel size
        self.data_parallel_size_ = self.world_size_ // self.model_parallel_size

    def _init_placement_group(self, args):
        node_device_ids = self._init_node_device_ids(args)
        num_devices_per_stage = self.world_size_ // self.pipeline_model_parallel_size_
        stages_device_ids = [
            node_device_ids[i : (i + num_devices_per_stage)]
            for i in range(0, self.world_size_, num_devices_per_stage)
        ]

        assert args.num_layers % self.pipeline_model_parallel_size_ == 0, (
            f"number of layers ({args.num_layers}) is not divisible by"
            " pipeline parallel size ({self.pipeline_model_parallel_size_})"
        )
        num_layers_per_stage = args.num_layers // self.pipeline_model_parallel_size_
        self.layers_stage_ids_ = [
            i // num_layers_per_stage for i in range(args.num_layers)
        ]
        self.layers_device_ids_ = [
            stages_device_ids[i // num_layers_per_stage] for i in range(args.num_layers)
        ]

    def _init_node_device_ids(self, args):
        # ['0:0', '0:1', '0:2', '0:3', '1:0', '1:1', '1:2', '1:3']
        node_ids = [i // args.num_gpus_per_node for i in range(self.world_size_)]
        device_ids = list(range(args.num_gpus_per_node)) * args.num_nodes
        node_device_ids = [f"{n}:{d}" for n, d in zip(node_ids, device_ids)]
        return node_device_ids

    def _init_parallel_hierarchy(self):
        if self.is_hybrid_parallel():
            self.parallel_hierarchy_ = [
                self.data_parallel_size_,
                self.tensor_model_parallel_size_,
            ]
        else:
            self.parallel_hierarchy_ = None

    @property
    def parallel_hierarchy(self):
        return self.parallel_hierarchy_

    @property
    def tensor_model_parallel_size(self):
        return self.tensor_model_parallel_size_

    def get_layer_placement(self, layer_idx):
        return self.layers_device_ids_[layer_idx]

    def get_layer_stage(self, layer_idx):
        return self.layers_stage_ids_[layer_idx]

    def is_model_parallel(self):
        return self.tensor_model_parallel_size_ > 1

    def is_data_parallel(self):
        return self.data_parallel_size_ > 1

    def is_pipeline_parallel(self):
        return self.pipeline_model_parallel_size_ > 1

    def is_hybrid_parallel(self):
        return self.is_model_parallel() and self.is_data_parallel()

    def is_non_parallel(self):
        return not self.is_model_parallel() and not self.is_data_parallel()


def get_dist_util():
    global _DIST_UTIL
    if _DIST_UTIL is None:
        _DIST_UTIL = _DistributeUtil()
    return _DIST_UTIL


@contextlib.contextmanager
def layer_placement_scope(layer_idx, device="gpu"):
    dist_util = get_dist_util()
    with flow.scope.placement(
        device, dist_util.get_layer_placement(layer_idx), dist_util.parallel_hierarchy,
    ):
        if dist_util.is_pipeline_parallel():
            with flow.experimental.scope.config(
                pipeline_stage_id_hint=dist_util.get_layer_stage(layer_idx)
            ):
                yield
        else:
            yield


@contextlib.contextmanager
def data_placement_scope():
    with layer_placement_scope(0, "cpu"):
        yield


def _is_sbp_str(s):
    if not isinstance(s, str):
        return False

    if s in ("B", "P"):
        return True

    if s.startswith("S(") and s[-1] == ")":
        return True

    return False


def _gen_parallel_dist_by_2d_sbp(sbp_list):
    assert isinstance(sbp_list, list)
    assert len(sbp_list) == 2
    assert all(_is_sbp_str(sbp) for sbp in sbp_list)

    dist_util = get_dist_util()
    if dist_util.is_hybrid_parallel():
        return sbp_list
    elif dist_util.is_data_parallel():
        return sbp_list[:1]
    elif dist_util.is_model_parallel():
        return sbp_list[1:]
    elif dist_util.is_non_parallel():
        return []
    else:
        raise NotImplementedError


def get_data_parallel_dist():
    return _gen_parallel_dist_by_2d_sbp(["S(0)", "B"])


def get_wpe_parallel_dist():
    return _gen_parallel_dist_by_2d_sbp(["B", "B"])


def get_wte_parallel_dist():
    return _gen_parallel_dist_by_2d_sbp(["B", "S(0)"])


def get_col_linear_weight_parallel_dist():
    return _gen_parallel_dist_by_2d_sbp(["B", "S(1)"])


def get_col_linear_bias_parallel_dist():
    return _gen_parallel_dist_by_2d_sbp(["B", "S(0)"])


def get_row_linear_weight_parallel_dist():
    return _gen_parallel_dist_by_2d_sbp(["B", "S(0)"])


def get_row_linear_bias_parallel_dist():
    return _gen_parallel_dist_by_2d_sbp(["B", "B"])


def get_layernorm_params_parallel_dist():
    return _gen_parallel_dist_by_2d_sbp(["B", "B"])


def _infer_split_axis(x):
    if len(x.shape) == 2:
        return 0

    if len(x.shape) == 3:
        if x.shape[0] == x.shape[-1]:
            return -1

        args = get_args()
        if x.shape[0] == args.seq_length:
            return 1

        if x.shape[1] == args.seq_length:
            return 0

    return -1


def forward_p2b_parallel_cast(x):
    split_axis = _infer_split_axis(x)
    if split_axis < 0:
        raise RuntimeError("can't infer split axis")

    sbps = [f"S({split_axis})", "B"]
    parallel_dist = _gen_parallel_dist_by_2d_sbp(sbps)

    dist_util = get_dist_util()
    if dist_util.is_hybrid_parallel() or dist_util.is_model_parallel():
        # forward: [S(0), P] cast to [S(0), B], allreduce
        # backward: [S(0), B] cast to [S(0), B], identity
        # forward: P -> B, allreduce
        # backward: B -> B, identity
        x = flow.hierarchical_parallel_cast(
            x,
            nd_sbp=parallel_dist,
            grad_mode="manual",
            grad_nd_sbp=parallel_dist,
        )
    elif dist_util.is_data_parallel():
        # parallel cast: S(0) -> S(0), identity
        pass
    elif dist_util.is_non_parallel():
        # no need to cast, identity
        pass
    else:
        raise NotImplementedError

    return x


def backward_p2b_parallel_cast(x):
    split_axis = _infer_split_axis(x)
    if split_axis < 0:
        raise RuntimeError("can't infer split axis")

    sbps = [f"S({split_axis})", "B"]
    parallel_dist = _gen_parallel_dist_by_2d_sbp(sbps)

    dist_util = get_dist_util()
    if dist_util.is_hybrid_parallel():
        # forward: [S(0), B] cast to [S(0), B], identity
        # backward: [S(0), P] cast to [S(0), B], for layernorm grad not supporting P, cast from P to B
        x = flow.hierarchical_parallel_cast(
            x,
            nd_sbp=parallel_dist,
            grad_mode="manual",
            grad_nd_sbp=parallel_dist,
        )
    elif dist_util.is_data_parallel():
        # parallel cast: S(0) -> S(0), identity
        pass
    elif dist_util.is_model_parallel():
        # auto cast by choicing P -> B or P -> S(0), according to order value it should be former
        pass
    elif dist_util.is_non_parallel():
        # no need to cast, identity
        pass
    else:
        raise NotImplementedError

    return x


def output_parallel_cast(x, device="gpu"):
    dist_util = get_dist_util()
    if dist_util.is_hybrid_parallel():
        with flow.scope.placement(device, dist_util.get_layer_placement(-1)):
            x = flow.hierarchical_parallel_cast(x, nd_sbp=["B"])

    return x


def input_data_parallel_cast(x):
    dist_util = get_dist_util()
    if dist_util.is_hybrid_parallel():
        x = flow.hierarchical_parallel_cast(
            x, nd_sbp=get_data_parallel_dist(),
        )

    return x
