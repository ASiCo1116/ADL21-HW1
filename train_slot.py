import json
import pickle
import datetime
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from dataset import SlotTagDataset
from model import SlotClassifier
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def accuracy(y_pred, y_true, lengths):
    tacc = 0
    jacc = 0
    for pred, true, length in zip(y_pred, y_true, lengths):
        hits = int(torch.sum((torch.argmax(pred[:, :length], dim=0) == true[:length]).float()).item())
        tacc += hits
        jacc += int(hits == length)
    return tacc, jacc

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SlotTagDataset] = {
        split: SlotTagDataset(split_data, vocab, tag2idx, args.max_len)
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
    model = SlotClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        args.max_len,
        datasets[TRAIN].num_classes,
    ).to(args.device)

    # # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50], gamma=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    train_len = len(train_loader)
    val_len = len(val_loader)

    train_token_len = sum([l for batch in train_loader for l in batch[-1]])
    val_token_len = sum([l for batch in val_loader for l in batch[-1]])

    best_joint_acc = 0
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        token_tacc = 0
        joint_tacc = 0
        tloss = 0
        model.train()
        for i, batch in tqdm(enumerate(train_loader), total=train_len, desc='Train iter', leave=False):
            text, tag, _, length = batch
            tag = tag.to(args.device)
            pred = model(text.to(args.device))
            loss = criterion(pred, tag)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tloss += loss.item()
            token_acc, joint_acc = accuracy(pred, tag, length)
            token_tacc += token_acc
            joint_tacc += joint_acc

        tloss /= train_len
        token_tacc /= train_token_len
        joint_tacc /= len(datasets[TRAIN])

        token_vacc = 0
        joint_vacc = 0
        vloss = 0
        model.eval()
        for i, batch in tqdm(enumerate(val_loader), total=val_len, desc='Valid iter', leave=False):
            text, tag, _, length = batch
            tag = tag.to(args.device)

            with torch.no_grad():
                pred = model(text.to(args.device))
                loss = criterion(pred, tag)
                token_acc, joint_acc = accuracy(pred, tag, length)

                vloss += loss.item()
                token_vacc += token_acc
                joint_vacc += joint_acc
        
        vloss /= val_len
        token_vacc /= val_token_len
        joint_vacc /= len(datasets[DEV])

        if joint_vacc > best_joint_acc:
            best_joint_acc = joint_vacc
            torch.save(model.state_dict(), f'{args.ckpt_dir}/best.ckpt')
            epoch_pbar.write(f'Save best joint acc {joint_vacc:.5f} @ epoch {epoch}')

        if epoch % args.save_step == 0 and epoch != 0:
            torch.save(model.state_dict(), f'{args.ckpt_dir}/{epoch}.ckpt')
        
        if epoch == args.num_epoch - 1:
            torch.save(model.state_dict(), f'{args.ckpt_dir}/latest.ckpt')

        epoch_pbar.set_postfix({'token tacc': f'{token_tacc:.5f}', 'joint tacc': f'{joint_tacc:.5f}', 'tloss': f'{tloss:.5f}', 'token vacc': f'{token_vacc:.5f}', 'joint vacc': f'{joint_vacc:.5f}', 'vloss': f'{vloss:.5f}'})

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="/data/ADL/hw1/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="/data/ADL/hw1/cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )
    parser.add_argument(
        "--save_step",
        type=int,
        default=10,
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
