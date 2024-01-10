from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):

    def __init__(self, args: dict) -> None:
        super(MLP, self).__init__()
    
    def forward(self, inputs: torch.tensor) -> Any:
        pass


class Transformer(nn.Module):

    def __init__(self, args: dict) -> None:
        super(Transformer, self).__init__()

    def forward(self, inputs: torch.tensor):
        pass


class EmbeddingLayer(nn.Module):

    def __init__(self, args: dict) -> None:
        super(EmbeddingLayer, self).__init__()
    
    def forward(self, inputs: torch.tensor):
        pass


if __name__ == "__main__":
    pass
