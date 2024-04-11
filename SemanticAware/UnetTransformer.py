import torch
import torch.nn as nn

from SemanticAware.TransformerBlock import TransformerBlock


# 将形为（2,16,480，,640）经过多次下采样得到形如（2,64,240，320）的tensor ，再上采样至（2，16,480,640）

class UnetTransformer_0(nn.Module):
    def __init__(self):
        super(UnetTransformer_0, self).__init__()

        # 下采样部分
        self.downsample = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.att_block = TransformerBlock(64, 64)

        # 上采样部分
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x, seg):
        # 下采样
        x = self.downsample(x)
        awareF = self.att_block(x, seg)
        # 上采样
        x = self.upsample(awareF)

        return x


# 采用Unet网络结构，将形为（2,32,480，,640）经过2次下采样得到形如（2,128,120.160）的tensor    ，再上采样至（2，32,480,640）    pytorch
class UnetTransformer_1(nn.Module):
    def __init__(self):
        super(UnetTransformer_1, self).__init__()

        # 下采样部分
        self.downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.att_block = TransformerBlock(128, 128)
        # 上采样部分
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)

        )

    def forward(self, x, seg):
        # 下采样
        x = self.downsample(x)
        awareF = self.att_block(x, seg)
        # 上采样
        x = self.upsample(awareF)

        return x


# 采用Unet网络结构，将形为（2,48,480，,640）经过多次下采样得到形如（2,256,60，80）的tensor    ，再上采样至（2，48,480,640

class UnetTransformer_2(nn.Module):
    def __init__(self):
        super(UnetTransformer_2, self).__init__()

        # 下采样部分
        self.downsample1 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.downsample2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.downsample3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.att_block = TransformerBlock(256, 256)

        # 上采样部分
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(64, 48, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x, seg):
        # 下采样
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        awareF = self.att_block(x, seg)
        # 上采样
        x = self.upsample1(awareF)
        x = self.upsample2(x)
        x = self.upsample3(x)

        return x
