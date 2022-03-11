import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(
        datasets[TRAIN],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=datasets[TRAIN].collate_fn,
    )
    val_loader = DataLoader(
        datasets[DEV],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=datasets[DEV].collate_fn,
    )
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        datasets[TRAIN].num_classes,
    ).to(args.device)

    # # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        tacc = 0
        tloss = 0

        model.train()
        for i, batch in enumerate(train_loader):
            pred = model(batch[0].to(args.device))
            loss = criterion(pred, batch[1].to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tloss += loss.item()
            tacc += torch.mean((torch.argmax(pred, dim=1) == batch[1].to(args.device)).float(), dim=0).item()
        tloss /= len(train_loader)
        tacc /= len(train_loader)

        epoch_pbar.set_postfix({'tacc': tacc, 'tloss': tloss})
        # print(f'tacc: {tacc:.3f} | tloss: {tloss:.3f}')
        #     # TODO: Training loop - iterate over train dataloader and update model weights
        #     # TODO: Evaluation loop - calculate accuracy and save model weights

    # # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="/data/ADL/hw1/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="/data/ADL/hw1/cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(args)
    main(args)
