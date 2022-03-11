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
        )
        bidirection = 2 if bidirectional else 1
        self.clf = nn.Linear(hidden_size * bidirection, num_class)

    # @property
    # def encoder_output_size(self) -> int:
    #     # TODO: calculate the output dimension of rnn
    #     raise NotImplementedError

    def forward(self, batch):
        # TODO: implement model forward
        out = self.embed(batch)
        out, (_, _) = self.lstm(out)
        return self.clf(out[-1])
