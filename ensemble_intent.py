import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from unittest import result
from tqdm import tqdm
import numpy as np
import csv

import torch
from torch.utils.data import DataLoader

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    ckpt_path = ['best_intent_87984.ckpt', 'best_intent_88322.ckpt', 'best_intent_88721.ckpt']
    all_logits = []
    for j in range(3):

        model = SeqClassifier(
            embeddings,
            768,
            4 if j != 2 else 5,
            0.1,
            True,
            128,
            dataset.num_classes,
        )
        model.eval()

        # load weights into model
        ckpt = torch.load(ckpt_path[j], map_location=args.device)
        model.load_state_dict(ckpt)
        model.to(args.device)

        # TODO: predict dataset
        pred_ids = []
        pred_logits = []
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Test iter"):
            text, ids = batch
            pred_ids.append(ids)
            with torch.no_grad():
                logits = model(text.to(args.device))
                pred_logits.append(logits.cpu())
                # print(logits.size())
                # import sys; sys.exit()
                # pred = torch.argmax(logits, dim=1).cpu()
            # pred_logits.append(list(map(dataset.idx2label, pred.numpy())))

        pred_ids = np.concatenate(pred_ids).reshape(-1, 1)
        pred_logits = torch.cat(pred_logits).reshape(-1, 150)
        all_logits.append(pred_logits)
        # results = np.hstack((pred_ids, pred_intent))
    pred = sum(all_logits)
    pred = torch.argmax(pred, dim=1).numpy()
    pred = np.array(list(map(dataset.idx2label, pred))).reshape(-1, 1)
    results = np.hstack((pred_ids, pred))
    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as fw:
        writer = csv.writer(fw)
        writer.writerow(['id', 'intent'])
        writer.writerows(results)

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
        default="./cache/intent/",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

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
    main(args)
