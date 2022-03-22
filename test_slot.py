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

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SlotTagDataset(data, vocab, tag2idx, args.max_len)

    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # # TODO: init model and move model to target device(cpu / gpu)
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

    pred_ids = []
    pred_slot = []
    pred_length = []
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Test iter"):
        model.initHidden(batch[0].size(0), args.device)

        text, ids, length = batch
        text = text.to(args.device)
        pred_ids.append(ids)
        pred_length.append(length)

        with torch.no_grad():
            logits = model(text)
            pred = torch.argmax(logits, dim=1)
        pred_slot.append(pred.to('cpu').numpy())

    pred_ids = np.concatenate(pred_ids)
    pred_slot = np.concatenate(pred_slot)
    pred_length = np.concatenate(pred_length)
    # results = np.hstack((pred_ids, pred_slot))

    with open(args.pred_file, 'w') as fw:
        writer = csv.writer(fw)
        writer.writerow(['id', 'tags'])
        for id, tag, length in zip(pred_ids, pred_slot, pred_length):
            tags = " ".join(list(map(dataset.idx2label, tag[:length])))
            writer.writerow([id, tags])

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
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

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