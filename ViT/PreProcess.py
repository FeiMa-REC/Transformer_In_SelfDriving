import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreProcess(nn.Module):
    def __init__(self,image_size, patch_size, patch_dim, dim):
        super(PreProcess, self).__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.patch_num = (image_size // patch_size)**2
        ...