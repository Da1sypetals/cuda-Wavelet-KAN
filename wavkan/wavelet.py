import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *
from torch.autograd import Function
import einops as ein

# import faster_ops 
import mexhat_ops
import morlet_ops

class MorletWaveletFunction(Function):
    @staticmethod
    def forward(ctx, x, scale, bias, weight):

        ctx.save_for_backward(x, scale, bias, weight)

        return morlet_ops.forward(x, scale, bias, weight)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, scale, bias, weight = ctx.saved_tensors

        grad_x, grad_scale, grad_bias, grad_weight = morlet_ops.backward(grad_output, x, scale, bias, weight)

        return grad_x, grad_scale, grad_bias, grad_weight


    
def morlet_wavelet(x, scale, bias, weight):
    return MorletWaveletFunction.apply(x, scale, bias, weight)


class MexHatWaveletFunction(Function):
    @staticmethod
    def forward(ctx, x, scale, bias, weight):

        ctx.save_for_backward(x, scale, bias, weight)

        return mexhat_ops.forward(x, scale, bias, weight)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, scale, bias, weight = ctx.saved_tensors

        grad_x, grad_scale, grad_bias, grad_weight = mexhat_ops.backward(grad_output, x, scale, bias, weight)

        return grad_x, grad_scale, grad_bias, grad_weight

    
def mexhat_wavelet(x, scale, bias, weight):
    return MexHatWaveletFunction.apply(x, scale, bias, weight)

