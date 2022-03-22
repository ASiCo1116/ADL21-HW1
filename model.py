from typing import Dict

import torch
import torch.nn as nn


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        input_size: int,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.lstm = nn.LSTM(
            embeddings.size(1),
            hidden_size,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.clf = nn.Sequential(
            nn.Linear(input_size * hidden_size * 2 if bidirectional else input_size * hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_class),
        )

    def forward(self, batch):
        # TODO: implement model forward
        out = self.embed(batch)
        out, (h, c) = self.lstm(out)
        return self.clf(out.reshape(out.size(0), -1))

class SlotClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        input_size: int,
        num_class: int,
    ) -> None:
        super(SlotClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.lstm = nn.LSTM(
            embeddings.size(1),
            hidden_size,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.clf = nn.Sequential(
            nn.Linear(hidden_size * 2 if self.bidirectional else hidden_size, num_class),
        )

    def initHidden(self, batch_size, device):
        return torch.zeros(2 * self.num_layers if self.bidirectional else self.num_layers, batch_size, self.hidden_size, device=device)

    def forward(self, batch):
        # TODO: implement model forward
        out = self.embed(batch)
        out, (h, c) = self.lstm(out)
        return self.clf(out).permute(0, 2, 1)