import torch
import torch.nn as nn
import torch.nn.functional as func


class MockData(object):

    def __init__(self) -> None:
        pass

    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        pass

class MLP(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class DeepFM(nn.Module):

    def __init__(self) -> None:
        super(DeepFM, self).__init__()
        self.net = nn.Sequential(nn.Linear(21, 23), nn.ReLU(), nn.Linear(15, 1))
        pass

    def forward(self, inputs: torch.tensor):
        
        return self.net(inputs)


if __name__ == "__main__":
    print("Hello World.")

