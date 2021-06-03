import argparse
import os
import numpy as np
import torch
import meta_pb2 as meta_pb

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument(
    "--py_model_dir",
    type=str,
    default="./iter_0500000/mp_rank_00/model_optim_rng.pt",
    help="Path the PyTorch checkpoint file path.",
)
parser.add_argument(
    "--of_dump_path",
    type=str,
    default="./convert_pt_to_of_gpt_release",
    help="Path to the output OneFlow model.",
)

args = parser.parse_args()


def _SaveWeightBlob2File(blob, op_name, var="out", meta="meta"):
    folder = os.path.join(args.of_dump_path, op_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, var)
    f = open(filename, "wb")
    f.write(blob.tobytes())
    meta_info = meta_pb.Meta()
    meta_info.shape.dim[:] = blob.shape
    meta_info.data_type = meta_pb.kFloat
    filename = os.path.join(folder, meta)
    f = open(filename, "w")
    f.write(str(meta_info))
    f.close()
    np.save(filename, blob)


def _SaveWeightBlob2FileExtend(blob, op_name, var="out", meta="meta"):
    _SaveWeightBlob2File(blob.numpy(), op_name, var="out", meta="meta")
    _SaveWeightBlob2File(np.ones_like(blob), op_name + "-v", var="out", meta="meta")
    _SaveWeightBlob2File(np.zeros_like(blob), op_name + "-m", var="out", meta="meta")


def convert():
    path = args.py_model_dir
    state_dict = torch.load(path, map_location="cpu")
    for model_key, model_value in state_dict["model"]["language_model"][
        "transformer"
    ].items():
        if len(model_value.shape) > 1:
            model_value = torch.transpose(model_value, 0, 1)
        model_value = model_value.float()
        op_name_list = model_key.split(".")
        if "layers." in model_key:
            op_name = model_key.replace("layers.", "model-")
            op_name = op_name.replace(
                "-%s." % (op_name_list[1]), "-h%s-" % (op_name_list[1])
            )
        else:
            op_name = model_key.replace("final_layernorm.", "model-layernorm_f-")
        op_name = op_name.replace("input_layernorm.", "layernorm_1-")
        op_name = op_name.replace("post_attention_layernorm.", "layernorm_2-")
        op_name = op_name.replace("attention.", "attn-")
        op_name = op_name.replace("query_key_value.", "c_attn-")
        op_name = op_name.replace("dense.", "c_proj-")
        op_name = op_name.replace("mlp.dense_h_to_4h.", "mlp-c_fc-")
        op_name = op_name.replace("mlp.dense_4h_to_h.", "mlp-c_proj-")
        if (
            "layernorm_1" in op_name
            or "layernorm_2" in op_name
            or "layernorm_f" in op_name
        ):
            op_name = op_name.replace("-weight", "-gamma")
            op_name = op_name.replace("-bias", "-beta")
        elif "-c_attn-" in op_name:
            model_dim = model_value.chunk(3, dim=-1)
            _SaveWeightBlob2FileExtend(
                model_dim[0], op_name.replace("-c_attn-", "-q_attn-")
            )
            _SaveWeightBlob2FileExtend(
                model_dim[1], op_name.replace("-c_attn-", "-k_attn-")
            )
            _SaveWeightBlob2FileExtend(
                model_dim[2], op_name.replace("-c_attn-", "-v_attn-")
            )
        print(model_key, "-" * 8, op_name)
        _SaveWeightBlob2FileExtend(model_value, op_name)

    _SaveWeightBlob2FileExtend(
        state_dict["model"]["language_model"]["embedding"]["position_embeddings"][
            "weight"
        ].float(),
        "model-wpe",
    )
    _SaveWeightBlob2FileExtend(
        state_dict["model"]["language_model"]["embedding"]["word_embeddings"][
            "weight"
        ].float(),
        "model-wte",
    )


if __name__ == "__main__":
    convert()
