import torch
import torch.nn as nn

from PathArgs import PathArgs


class saBlock(nn.Module):
    def __init__(self):
        super(saBlock, self).__init__()
        self.spatianlAttention = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, labVisOneHot, labIrOneHot, decodex):
        tensor = labVisOneHot.squeeze().long()
        # 统计每个标签值出现的次数
        label_counts = torch.bincount(tensor.view(-1), minlength=PathArgs.classN)

        # 计算每个位置的权重值
        weights = label_counts[tensor] / tensor.numel()

        spAtt = self.spatianlAttention(decodex)
        spAtt = spAtt * PathArgs.weightConstant + (1 - PathArgs.weightConstant) * weights
        out = spAtt * decodex + decodex
        out = self.relu(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
