import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from oneflow_gpt.config import get_args


def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title="tasks")

    group.add_argument("--task", type=str, required=True, help="Task name.")
    group.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of finetunning epochs. Zero results in " "evaluation only.",
    )
    group.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help="Pretrained checkpoint used for finetunning.",
    )
    group.add_argument(
        "--keep-last",
        action="store_true",
        help="Keep the last batch (maybe incomplete) in" "the data loader",
    )
    group.add_argument(
        "--train-data",
        nargs="+",
        default=None,
        help="Whitespace separated paths or corpora names " "for training.",
    )
    group.add_argument(
        "--valid-data", nargs="*", default=None, help="path(s) to the validation data."
    )
    group.add_argument(
        "--overlapping-eval",
        type=int,
        default=32,
        help="Sliding window for overlapping evaluation.",
    )
    group.add_argument(
        "--strict-lambada",
        action="store_true",
        help="Use more difficult formulation of lambada.",
    )
    parser.add_argument(
        "--vocab-file", type=str, default=None, help="Path to the vocab file."
    )
    parser.add_argument(
        "--merge-file", type=str, default=None, help="Path to the BPE merge file."
    )
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        default=None,
        choices=["BertWordPieceLowerCase", "BertWordPieceCase", "GPT2BPETokenizer"],
        help="What type of tokenizer to use.",
    )
    parser.add_argument(
        "--reset-position-ids",
        action="store_true",
        help="Reset posistion ids after end-of-document token.",
    )
    parser.add_argument(
        "--reset-attention-mask",
        action="store_true",
        help="Reset self attention maske after " "end-of-document token.",
    )
    parser.add_argument(
        "--eod-mask-loss",
        action="store_true",
        help="Mask loss for the end of document tokens.",
    )

    return parser


if __name__ == "__main__":

    args = get_args(extra_args_provider=get_tasks_args)

    if args.task in ["LAMBADA"]:
        from zeroshot_gpt.evaluate import main
    else:
        raise NotImplementedError("Task {} is not implemented.".format(args.task))

    main(args)
