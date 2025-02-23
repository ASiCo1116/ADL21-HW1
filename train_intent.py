import json
import pickle
import datetime
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    set_seeds(args.seed)
    writer = SummaryWriter()
    writer.add_text('args', str(vars(args)))
    
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
        args.max_len,
        datasets[TRAIN].num_classes,
    ).to(args.device)

    # # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50], gamma=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        tacc = 0
        tloss = 0
        model.train()
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train iter', leave=False):
            text, intent, _ = batch
            intent = intent.to(args.device)
            pred = model(text.to(args.device))
            loss = criterion(pred, intent)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tloss += loss.item()
            acc = torch.mean((torch.argmax(pred, dim=1) == intent).float(), dim=0).item()
            tacc += acc

            writer.add_scalar('Train/loss', loss.item(), i + epoch * len(train_loader))
            writer.add_scalar('Train/acc', acc, i + epoch * len(train_loader))

        tloss /= len(train_loader)
        tacc /= len(train_loader)

        vacc = 0
        vloss = 0
        model.eval()
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='Valid iter', leave=False):
            text, intent, _ = batch
            intent = intent.to(args.device)
            with torch.no_grad():
                pred = model(text.to(args.device))
                loss = criterion(pred, intent)
            vloss += loss.item()
            acc = torch.mean((torch.argmax(pred, dim=1) == intent).float(), dim=0).item()
            vacc += acc

            writer.add_scalar('Valid/loss', loss.item(), i + epoch * len(val_loader))
            writer.add_scalar('Valid/acc', acc, i + epoch * len(val_loader))

        vloss /= len(val_loader)
        vacc /= len(val_loader)

        # scheduler.step()

        if vacc > best_acc:
            best_acc = vacc
            torch.save(model.state_dict(), f'{args.ckpt_dir}/best.ckpt')
            epoch_pbar.write(f'Save best vacc {vacc:.5f} @ epoch {epoch}')

        if epoch % args.save_step == 0 and epoch != 0:
            torch.save(model.state_dict(), f'{args.ckpt_dir}/{epoch}.ckpt')
        
        if epoch == args.num_epoch - 1:
            torch.save(model.state_dict(), f'{args.ckpt_dir}/latest.ckpt')

        epoch_pbar.set_postfix({'tacc': f'{tacc:.5f}', 'tloss': f'{tloss:.5f}', 'vacc': f'{vacc:.5f}', 'vloss': f'{vloss:.5f}'})

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
    parser.add_argument(
        "--save_step",
        type=int,
        default=10,
    )
    parser.add_argument("--seed", type=int, default=1116)

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
