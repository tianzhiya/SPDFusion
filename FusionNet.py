# coding:utf-8
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from koila import lazy
import gc

from FeatureMapTransformer import FeatureMapTransformer
from SegResultAware.saBlock import saBlock, SpatialAttention
from SemanticAware.SematicGraphAttention import SematicGraphAttention

from SemanticAware.TransformerBlock import TransformerBlock
from SemanticAware.UnetTransformer import UnetTransformer_0, UnetTransformer_1, UnetTransformer_2


class ConvBnLeakyRelu2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.tanh(self.conv(x)) / 2 + 0.5


class ConvLeakyRelu2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x


class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return self.conv(x)


class DenseBlock(nn.Module):
    def __init__(self, channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2 * channels, channels)

    def forward(self, x):
        x = torch.cat((x, self.conv1(x)), dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        return x


class RGBD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RGBD, self).__init__()
        self.dense = DenseBlock(in_channels)
        self.convdown = Conv1(3 * in_channels, out_channels)
        self.sobelconv = Sobelxy(in_channels)
        self.convup = Conv1(in_channels, out_channels)

    def forward(self, x):
        x1 = self.dense(x)
        x1 = self.convdown(x1)
        x2 = self.sobelconv(x)
        x2 = self.convup(x2)
        return F.leaky_relu(x1 + x2, negative_slope=0.1)


class FusionNet(nn.Module):
    def __init__(self, output):
        super(FusionNet, self).__init__()
        vis_ch = [16, 32, 48]
        inf_ch = [16, 32, 48]
        output = 1

        self.att_block1 = FeatureMapTransformer(in_dim=48, sr_ratio=4)
        self.att_block2 = FeatureMapTransformer(in_dim=48, sr_ratio=4)
        self.att_block3 = FeatureMapTransformer(in_dim=48, sr_ratio=2)

        self.att_block_1 = TransformerBlock(64, 16)
        self.att_block_2 = TransformerBlock(128, 32)
        self.att_block_3 = TransformerBlock(256, 48)

        self.unetTransformer_0 = UnetTransformer_0()
        self.unetTransformer_1 = UnetTransformer_1()
        self.unetTransformer_2 = UnetTransformer_2()

        self.vis_conv = ConvLeakyRelu2d(1, vis_ch[0])
        self.vis_rgbd1 = RGBD(vis_ch[0], vis_ch[1])
        self.vis_rgbd2 = RGBD(vis_ch[1], vis_ch[2])

        self.inf_conv = ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = RGBD(inf_ch[0], inf_ch[1])
        self.inf_rgbd2 = RGBD(inf_ch[1], inf_ch[2])
        self.sematicGraphAttention1 = SematicGraphAttention()
        self.sematicGraphAttention2 = SematicGraphAttention()
        self.sematicGraphAttention3 = SematicGraphAttention()
        self.sematicGraphAttention4 = SematicGraphAttention()

        self.saBlock = saBlock()
        self.SemanticEmbeddModule1 = SemanticEmbeddModule(norm_nc=64, label_nc=1024, nhidden=64)
        self.SemanticEmbeddModule2 = SemanticEmbeddModule(norm_nc=32, label_nc=2048, nhidden=32)

        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2] + inf_ch[2], vis_ch[1] + vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1] + inf_ch[1], vis_ch[0] + inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0] + inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnTanh2d(vis_ch[0], output)

    def getDecoderAttionF(self, x, classAtionF):
        classAtionF = classAtionF.to('cuda:0')
        mulResult = torch.mul(classAtionF, x)
        addResult = torch.add(x, mulResult)
        mean = addResult.mean(dim=(0, 2, 3), keepdim=True)
        std = addResult.std(dim=(0, 2, 3), keepdim=True)
        normalized_tensor = (addResult - mean) / std
        return normalized_tensor

    def forward(self, image_vis, image_ir, loadSegF, labVisOneHot, labIrOneHot):
        localVisSegF1 = loadSegF.getVisFeature1()
        localSecondVisSegF2 = loadSegF.getVisFeature2()
        localThirdVisSegF3 = loadSegF.getVisFeature3()

        localIrSegF1 = loadSegF.getIrFeature1()
        localIrSegF2 = loadSegF.getIrFeature2()
        localIrSegF3 = loadSegF.getIrFeature3()

        localVisSegF1 = localVisSegF1.to('cuda:0')
        localSecondVisSegF2 = localSecondVisSegF2.to('cuda:0')
        localThirdVisSegF3 = localThirdVisSegF3.to('cuda:0')

        localIrSegF1 = localIrSegF1.to('cuda:0')
        localIrSegF2 = localIrSegF2.to('cuda:0')
        localIrSegF3 = localIrSegF3.to('cuda:0')

        segDecoderF = loadSegF.getSegDecoderF()

        x_vis_origin = image_vis[:, :1]
        x_inf_origin = image_ir
        # encode
        x_vis_p = self.vis_conv(x_vis_origin)
        x_inf_p = self.inf_conv(x_inf_origin)
        labVisOneHot = labVisOneHot.to('cuda:0')
        labIrOneHot = labIrOneHot.to('cuda:0')

        visTsFormerF = self.att_block_1(x_vis_p, localVisSegF1)
        irTsFormerF = self.att_block_1(x_inf_p, localVisSegF1)

        x_vis_p1 = self.vis_rgbd1(visTsFormerF)
        x_inf_p1 = self.inf_rgbd1(irTsFormerF)
        visTsFormerF = self.att_block_2(x_vis_p1, localSecondVisSegF2)
        irTsFormerF = self.att_block_2(x_inf_p1, localIrSegF2)

        x_vis_p2 = self.vis_rgbd2(visTsFormerF)
        x_inf_p2 = self.inf_rgbd2(irTsFormerF)

        visTsFormerF = self.att_block_3(x_vis_p2, localThirdVisSegF3)
        irTsFormerF = self.att_block_3(x_inf_p2, localThirdVisSegF3)

        # decode
        x = torch.cat((visTsFormerF, irTsFormerF), dim=1)
        x = self.saBlock(labVisOneHot, labIrOneHot, x)

        x = self.decode4(x)

        ownsampled_tensor = torch.nn.functional.avg_pool2d(x, kernel_size=2)
        ownsampled_tensor = torch.nn.functional.avg_pool2d(ownsampled_tensor, kernel_size=2)
        segX1 = self.SemanticEmbeddModule1(ownsampled_tensor, segDecoderF[0])
        upsampled_tensor = torch.nn.functional.interpolate(segX1, scale_factor=2, mode='bilinear',
                                                           align_corners=False)
        upsampled_tensor = torch.nn.functional.interpolate(upsampled_tensor, scale_factor=2, mode='bilinear',
                                                           align_corners=False)
        if upsampled_tensor.size() != x.size():
            upsampled_tensor = F.interpolate(upsampled_tensor, size=x.size()[2:], mode='bilinear', align_corners=False)
        x = x + upsampled_tensor
        x = self.decode3(x)
        ownsampled_tensor2 = torch.nn.functional.avg_pool2d(x, kernel_size=2)
        ownsampled_tensor2 = torch.nn.functional.avg_pool2d(ownsampled_tensor2, kernel_size=2)
        ownsampled_tensor2 = torch.nn.functional.avg_pool2d(ownsampled_tensor2, kernel_size=2)
        segX2 = self.SemanticEmbeddModule2(ownsampled_tensor2, segDecoderF[1])
        upsampled_tensor2 = torch.nn.functional.interpolate(segX2, scale_factor=2, mode='bilinear',
                                                            align_corners=False)
        upsampled_tensor2 = torch.nn.functional.interpolate(upsampled_tensor2, scale_factor=2, mode='bilinear',
                                                            align_corners=False)
        upsampled_tensor2 = torch.nn.functional.interpolate(upsampled_tensor2, scale_factor=2, mode='bilinear',
                                                            align_corners=False)
        if upsampled_tensor2.size() != x.size():
            upsampled_tensor2 = F.interpolate(upsampled_tensor2, size=x.size()[2:], mode='bilinear',
                                              align_corners=False)
        x = x + upsampled_tensor2

        x = self.saBlock(labVisOneHot, labIrOneHot, x)
        x = self.decode2(x)
        x = self.saBlock(labVisOneHot, labIrOneHot, x)

        x = self.decode1(x)
        return x


class SemanticEmbeddModule(nn.Module):

    def __init__(self, norm_nc, label_nc, nhidden=64):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=norm_nc)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = self.bn(normalized * (1 + gamma)) + beta
        return out


def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(2, 4, 480, 640).astype(np.float32))
    model = FusionNet(output=1)
    y = model(x)
    print('output shape:', y.shape)
    assert y.shape == (2, 1, 480, 640), 'output shape (2,1,480,640) is expected!'
    print('test ok!')


if __name__ == '__main__':
    unit_test()
