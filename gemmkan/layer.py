'''This is a sample code for the simulations of the paper:
Bozorgasl, Zavareh and Chen, Hao, Wav-KAN: Wavelet Kolmogorov-Arnold Networks (May, 2024)

https://arxiv.org/abs/2405.12832
and also available at:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4835325
We used efficient KAN notation and some part of the code:https://github.com/Blealtan/efficient-kan

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from .wavelet import mexhat_wavelet


class MexhatWaveletKanLayer(nn.Module):

    @property
    def minimum_dim_size(self):
        return 128

    def __init__(self, in_features, out_features, assertion=True):

        assert in_features % self.minimum_dim_size == 0, f"Require (in_features % {self.minimum_dim_size} == 0)!"
        assert out_features % self.minimum_dim_size == 0, f"Require (in_features % {self.minimum_dim_size} == 0)!"

        super().__init__()

        self.assertion = assertion
        self.in_features = in_features
        self.out_features = out_features

        self.scale = nn.Parameter(torch.ones(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(in_features, out_features))

        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.base_activation = nn.SiLU()

        # Batch normalization
        self.bn = nn.BatchNorm1d(out_features)


    def forward(self, x):
        
        if self.assertion:
            assert x.size(0) % self.minimum_dim_size == 0, f"Require (in_features % {self.minimum_dim_size} == 0)!"

        x = mexhat_wavelet(x, self.scale, self.bias, self.weight)

        return self.bn(x)
