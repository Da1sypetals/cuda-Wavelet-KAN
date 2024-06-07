import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *
from torch.autograd import Function
import einops as ein

import gemm_ops

class MexHatWaveletFunction(Function):
    @staticmethod
    def forward(ctx, x, scale, bias, weight):

        ctx.save_for_backward(x, scale, bias, weight)

        return gemm_ops.forward(x, scale, bias, weight)

    
    @staticmethod
    def backward(ctx, grad_output):
        x, scale, bias, weight = ctx.saved_tensors

        grad_x, grad_scale, grad_bias, grad_weight = gemm_ops.backward(grad_output, x, scale, bias, weight)

        return grad_x, grad_scale, grad_bias, grad_weight

    
def mexhat_wavelet(x, scale, bias, weight):
    return MexHatWaveletFunction.apply(x, scale, bias, weight)


