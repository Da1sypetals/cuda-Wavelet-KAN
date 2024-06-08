import torch
import torch.nn as nn
# from .layer import MorletWaveletKanLayer as Layer
from .layer import MexhatWaveletKanLayer as Layer


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Layer(768, 256)
        self.layer2 = Layer(256, 128)
        self.layer3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x














