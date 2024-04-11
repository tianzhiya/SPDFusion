# SematicGraphAttention.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numbers
from einops import rearrange


class SematicGraphAttention(nn.Module):
    def __init__(self):
        super(SematicGraphAttention, self).__init__()

        # localVisSegF1,localIrSegF1,x_vis_p,x_inf_p

    def forward(self, labVisOneHot, labIrOneHot, decodex):
        qHW = labVisOneHot
        kHW = labIrOneHot
        decodex = decodex.to('cuda')  # 将张量移动到GPU上

        qHW = qHW[0]
        kHW = kHW[0]
        # v = torch.randn(480, 640)

        # 点乘计算注意力分数
        attention_scores = torch.matmul(qHW.unsqueeze(1), kHW.unsqueeze(-1)).squeeze()  # 注意力分数形状为（480,）

        # softmax归一化操作
        attention_weights = F.softmax(attention_scores, dim=0)

        # 转换为形状（480, 640）的张量
        attention_weights = attention_weights.unsqueeze(1).expand(-1, 640)  # 注意力权重形状为（480, 640）

        attention_weightsHW = attention_weights.to('cuda')
        decodeFBathSize = decodex.shape[0]
        decodeFChanelSize = decodex.shape[1]

        chanelDataList = []
        batchDataList = []

        for i in range(decodeFBathSize):
            data = decodex[i, :, :, :]
            for j in range(decodeFChanelSize):
                featureHW = data[j, :, :]
                featureHW = featureHW + torch.mul(featureHW, attention_weightsHW)

                chanelDataList.append(featureHW)

            chanelTensor = torch.stack(chanelDataList)
            batchDataList.append(chanelTensor)
        result = torch.stack(batchDataList)
        return  result
