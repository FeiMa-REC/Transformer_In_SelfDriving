import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat  # einops和einsum用于以直观的方式对张量进行操作
from einops.layers.torch import Rearrange

# 定义pair函数用于检查t是否为tuple, 若不是则返回t的同值元组
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ViT(nn.Module):
    '''
        args:
            image_size : 输入图片的大小
            patch_size : 输入图片被切割成的patch大小，patch的个数必须大于16
            num_classes : 分类数
            dim : 线性投影后的维度
            depth : encoder 的深度
            heads : 注意力头的个数
            mlp_dim : 分类层的维度
            channels : 输入图片的通道数
            pool : 选用 CLS embedding 或者 mean 平均池化
    '''
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(ViT, self).__init__()
    
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, "Image dimensions must be divisible by the patch size."

        num_patchs = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, "pool type must be either cls or mean."

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h, p1) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.rand(1, num_patchs + 1, dim))
        self.cls_token = nn.Parameter(torch.rand(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = ...

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes*2),
            nn.GELU(),
            nn.Linear(num_classes*2, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ =  x.shape

        cls_token = repeat(self.cls_token, '1 1 d -> b, 1, d', b = b)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)

if __name__ == "__main__":
    ...