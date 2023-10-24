import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat  # einops和einsum用于以直观的方式对张量进行操作
from einops.layers.torch import Rearrange

# 定义pair函数用于检查t是否为tuple, 若不是则返回t的同值元组
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class MLP(nn.Module):
    def __init__(self,input_channels, output_channels):
        super(self, MLP).__init__()
        self.fc1 = nn.Linear(in_features=input_channels, out_features=output_channels)

    def forward(self,x):
        ...

if __name__ == "__main__":
    t = 224
    r = pair(t)
    print(r)