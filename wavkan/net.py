import torch
import torch.nn as nn
# from .layer import MorletWaveletKanLayer as Layer
from .layer import MexhatWaveletKanLayer as Layer


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = Layer(784, 256)
        # self.layer2 = Layer(256, 256)
        self.layer3 = Layer(256, 10)

    def forward(self, x):
        x = self.layer1(x)
        # x = self.layer2(x)
        x = self.layer3(x)
        return x














