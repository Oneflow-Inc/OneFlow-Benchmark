import argparse

from torch.utils.data import DataLoader
# from oneflow.utils.data import DataLoader
import oneflow as flow
from model.bert import BERT
from trainer.pretrain import BERTTrainer
from dataset.dataset import BERTDataset
from dataset.vocab import WordVocab

# eager mode
flow.enable_eager_execution()

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=False, type=str, default='data/corpus.small', help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default='data/corpus.small', help="test set for evaluate train set")
    parser.add_argument("-v", "--vocab_path", required=False, default='data/vocab.small', type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-o", "--output_path", required=False, default='output/', type=str, help="ex)output/bert.model")

    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=20, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=16, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()

    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len,
                                corpus_lines=args.corpus_lines, on_memory=args.on_memory)

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = BERTDataset(args.test_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory) \
        if args.test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    print("Building BERT model")
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

    print("Creating BERT Trainer")
    trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=10)

    print("Trainer build finished!")
    print("Training Start......")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        # print("Saving model...")
        trainer.save(epoch, args.output_path)
        if test_data_loader is not None:
            print("Running testing...")
            trainer.test(epoch)

main()
