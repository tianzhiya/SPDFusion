import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numbers
from einops import rearrange


class TransformerBlock(nn.Module):
    def __init__(self, dim_2, dim, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        # self.conv2 = nn.Conv2d(dim, dim_2, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.chanelDownR = nn.Conv2d(dim, 16, (1, 1))
        self.chanelDownS = nn.Conv2d(dim_2, 16, (1, 1))
        self.chanelNorm1 = LayerNorm(16, LayerNorm_type)
        self.chanelattn = Attention(16, num_heads, bias)
        self.chanelffn = FeedForward(16, ffn_expansion_factor, bias)
        self.chanelnorm2 = LayerNorm(16, LayerNorm_type)
        self.chanelUp = nn.Conv2d(16, 48, (1, 1))

    def forward(self, input_R, input_S):
        chanleN = input_R.shape[1]
        if (chanleN > 32):
            input_R_down = self.chanelDownR(input_R)
            input_S = F.interpolate(input_S, [input_R_down.shape[2], input_R_down.shape[3]])
            input_S = self.chanelDownS(input_S)
            # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
            input_R_down = self.chanelNorm1(input_R_down)
            input_R_down = self.chanelNorm1(input_S)
            input_R = input_R + self.chanelUp(self.chanelattn(input_R_down, input_S))
            input_R = input_R + self.chanelUp(self.chanelffn(self.chanelnorm2(input_R_down), self.chanelnorm2(input_S)))
            return input_R

        # input_ch = input_R.size()[1]
        input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]])
        input_S = self.conv1(input_S)
        # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
        input_R = self.norm1(input_R)
        input_S = self.norm1(input_S)
        input_R = input_R + self.attn(input_R, input_S)
        input_R = input_R + self.ffn(self.norm2(input_R), self.norm2(input_S))

        return input_R


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class FeedForward_1(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_1, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.project_in_R = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.project_in_S = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        # self.dwconv_R = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.dwconv_S = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        x = self.project_in_R(x)
        y = self.project_in_S(y)
        # x = self.project_in(x)
        # x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x) * F.gelu(y)
        x = self.project_out(x)
        return x


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


if __name__ == '__main__':
    shape1 = (2, 256, 60, 80)
    random_tensorSeg = torch.randn(shape1)

    shape2 = (2, 16, 480, 640)
    random_tensorVis = torch.randn(shape2)

    att_block_1 = TransformerBlock(256, 16)
    input_R = att_block_1(random_tensorVis, random_tensorSeg)
