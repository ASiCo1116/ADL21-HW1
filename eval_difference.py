from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from seqeval.metrics import accuracy_score

import json
import pickle
import csv
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import numpy as np

from dataset import SlotTagDataset
from model import SlotClassifier
from utils import Vocab

def accuracy(y_pred, y_true, lengths):
    tacc = 0
    jacc = 0
    for pred, true, length in zip(y_pred, y_true, lengths):
        hits = int(torch.sum((pred[:length] == true[:length]).float()).item())
        tacc += hits
        jacc += int(hits == length)
    return tacc/sum(lengths), jacc/len(lengths)

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SlotTagDataset(data, vocab, tag2idx, args.max_len)

    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    
    model = SlotClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        args.max_len,
        dataset.num_classes,
    ).to(args.device)

    model.eval()

    ckpt = torch.load(args.ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt)
    model.to(args.device)

    tags = []
    pred_slot = []
    lengths = []
    for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Test iter"):
        model.initHidden(batch[0].size(0), args.device)

        text, tag, _, length = batch
        tags.append(tag)
        lengths.append(length)

        text = text.to(args.device)

        with torch.no_grad():
            logits = model(text)
            pred = torch.argmax(logits, dim=1)
        pred_slot.append(pred.to('cpu'))

    pred_slot = torch.cat(pred_slot)
    tags = torch.cat(tags)
    lengths = np.concatenate(lengths)

    token_acc, joint_acc = accuracy(pred_slot, tags, lengths)
    print(f'token acc: {token_acc:.3f}')
    print(f'joint acc: {joint_acc:.3f}')

    y_true = []
    y_pred = []
    for tag, slot, length in zip(np.array(tags), np.array(pred_slot), lengths):
        y_true.append(list(map(dataset.idx2label, tag[:length])))
        y_pred.append(list(map(dataset.idx2label, slot[:length])))
        
    print(accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, mode='strict', scheme=IOB2))
    

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)