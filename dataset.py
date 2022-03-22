from typing import List, Dict

import torch
from torch.utils.data import Dataset

from utils import Vocab, pad_to_len


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        texts, intents, ids = [], [], []
        for sample in samples:
            if "intent" in sample.keys():
                intents.append(self.label2idx(sample["intent"]))
            texts.append(sample["text"])
            ids.append(sample["id"])
        texts = self.vocab.encode_batch(texts, self.max_len)
        if "intent" in samples[0].keys():
            return torch.LongTensor(texts), torch.LongTensor(intents), ids
        else:
            return torch.LongTensor(texts), ids

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

class SlotTagDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        texts, tags, ids, lengths = [], [], [], []
        for sample in samples:
            if "tags" in sample.keys():
                tags.append(list(map(self.label2idx, sample["tags"])))
            texts.append(sample["tokens"])
            ids.append(sample["id"])
            lengths.append(len(sample["tokens"]))
        texts = self.vocab.encode_batch(texts, self.max_len)
        tags = pad_to_len(tags, self.max_len, self.label2idx('O'))
        if "tags" in samples[0].keys():
            return torch.LongTensor(texts), torch.LongTensor(tags), ids, lengths
        else:
            return torch.LongTensor(texts), ids, lengths

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

if __name__ == '__main__':
    import pickle
    import json
    from pathlib import Path

    with open("/home/ubuntu/data/ADL/hw1/cache/slot/vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    tag_idx_path = Path("/home/ubuntu/data/ADL/hw1/cache/slot/tag2idx.json")
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())
    std = SlotTagDataset(json.loads(Path("/home/ubuntu/data/ADL/hw1/slot/train.json").read_text()), vocab, tag2idx, 64)

    from torch.utils.data import DataLoader
    dtl = DataLoader(std, batch_size=2, collate_fn=std.collate_fn)
    print(next(iter(dtl))[0].size())

    # with open("/home/ubuntu/data/ADL/hw1/cache/intent/vocab.pkl", "rb") as f:
    #     vocab: Vocab = pickle.load(f)
    # tag_idx_path = Path("/home/ubuntu/data/ADL/hw1/cache/intent/intent2idx.json")
    # tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())
    # std = SeqClsDataset(json.loads(Path("/home/ubuntu/data/ADL/hw1/intent/train.json").read_text()), vocab, tag2idx, 64)

    # from torch.utils.data import DataLoader
    # dtl = DataLoader(std, batch_size=2, collate_fn=std.collate_fn)
    # print(next(iter(dtl))[0])