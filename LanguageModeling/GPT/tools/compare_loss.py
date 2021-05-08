import os
import re
import sys


def parse_losses_for_log_file(
    log_file, loss_pattern, step_pattern, max_step, verbose=False
):
    if not os.path.isfile(log_file):
        raise ValueError(f"log file {log_file} do not exist")

    loss_dict = {}
    with open(log_file, "rt") as f:
        for line in f:
            step = None
            loss = None

            m = re.search(loss_pattern, line.strip())
            if m:
                loss = float(m.group(1))
            elif verbose:
                print(f"not found loss in line: {line.strip()}")
            else:
                pass

            m = re.search(step_pattern, line.strip())
            if m:
                step = int(m.group(1))
            elif verbose:
                print(f"not found step in line: {line.strip()}")
            else:
                pass

            if loss is not None and step is not None:
                assert step not in loss_dict
                loss_dict[step] = loss
                if len(loss_dict) >= max_step:
                    break

    return loss_dict


def plot_losses_comparison(oneflow_log_file, openai_log_file, verbose=False):
    import matplotlib.pyplot as plt

    loss_pattern = r"loss=[+-]?((\d+(\.\d+)?)|(\.\d+))"
    of_step_pattern = r"step=(\d+)"
    of_loss_dict = parse_losses_for_log_file(
        oneflow_log_file, loss_pattern, of_step_pattern, 100, verbose
    )

    oa_step_pattern = r"\[(\d+)\s\|\s\d+\.\d+\]"
    oa_loss_dict = parse_losses_for_log_file(
        openai_log_file, loss_pattern, oa_step_pattern, 100, verbose
    )

    if verbose:
        print("of_loss_dict:", of_loss_dict)
        print("oa_loss_dict:", oa_loss_dict)

    plt.plot(*zip(*sorted(of_loss_dict.items())))
    plt.plot(*zip(*sorted(oa_loss_dict.items())))
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise ValueError

    loss_pattern = r"loss=[+-]?((\d+(\.\d+)?)|(\.\d+))"
    # step_pattern = r"step=(\d+)"
    step_pattern = r"\[(\d+)\s\|\s\d+\.\d+\]"
    losses = parse_losses_for_log_file(sys.argv[1], loss_pattern, step_pattern)
    print(losses)
