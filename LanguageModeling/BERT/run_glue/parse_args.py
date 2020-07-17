import argparse
from datetime import datetime


def parse_args():
  parser = argparse.ArgumentParser(description="flags for bert-theseus")

  # resouce
  parser.add_argument("--gpu_num_per_node", type=int, default=1)
  parser.add_argument("--node_num", type=int, default=1)
  parser.add_argument("--node_list", type=str, default=None)

  # train
  parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
  parser.add_argument("--weight_decay_rate", type=float, default=0.01, help="weight l2 decay parameter")
  parser.add_argument("--batch_size_per_device", type=int, default=16)
  parser.add_argument("--iter_num", type=int, default=10, help="total iterations to run")
  parser.add_argument("--skip_iter_num", type=int, default=1,
                      help="number of skipping iterations for benchmark purpose.")
  parser.add_argument("--log_every_n_iter", type=int, default=1, help="print loss every n iteration")
  parser.add_argument("--task", type=str, default='CoLA')
  parser.add_argument("--data_dir", type=str, default=None)
  parser.add_argument("--val_data_dir", type=str, default=None)
  parser.add_argument("--test_data_dir", type=str, default=None)
  parser.add_argument("--data_part_num", type=int, default=32, help="data part number in dataset")
  parser.add_argument("--enable_auto_mixed_precision", default=False, type=lambda x: (str(x).lower() == 'true'))

  # log and resore/save
  parser.add_argument("--loss_print_every_n_iter", type=int, default=80, required=False,
                      help="print loss every n iteration")
  parser.add_argument("--model_save_every_n_iter", type=int, default=200, required=False,
                      help="save model every n iteration")
  parser.add_argument("--model_save_dir", type=str, required=False, help="model save directory",
                      default="./output/model_save-{}".format(str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))))
  parser.add_argument("--save_last_snapshot", type=bool, default=False, required=False,
                      help="save model snapshot for last iteration")
  parser.add_argument("--model_load_dir", type=str, default=None, required=False, help="model load directory")
  parser.add_argument("--log_dir", type=str, default="./output", required=False, help="log info save directory")
  parser.add_argument("--save_and_break", type=bool, default=False, required=False,
                help="init model , save model and break")
  # bert
  parser.add_argument("--seq_length", type=int, default=128)
  parser.add_argument("--num_hidden_layers", type=int, default=12)
  parser.add_argument("--num_attention_heads", type=int, default=12)
  parser.add_argument("--max_position_embeddings", type=int, default=512)
  parser.add_argument("--type_vocab_size", type=int, default=2)
  parser.add_argument("--vocab_size", type=int, default=30522)
  parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
  parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
  parser.add_argument("--hidden_size_per_head", type=int, default=64)
  parser.add_argument("--warmup_batches", type=int, default=100)
  parser.add_argument("--lr_decay_num", type=int, default=1000)
  parser.add_argument("--lr_decay_num_same_as_iter_num",
          default=False, type=(lambda x: str(x).lower() == 'true'))



  args = parser.parse_args()

  return args
