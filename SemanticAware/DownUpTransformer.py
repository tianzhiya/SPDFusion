import functools
import torch
import torch.nn as nn

from SemanticAware import arch_util
from TransformerBlock import TransformerBlock


class DownUpTransformer(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True, cnn=True):
        super(DownUpTransformer, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        self.cnn = cnn
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(16, 64, 3, 2, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(32, 128, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(48, 256, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)

        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)

        self.upconv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf * 2, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64 * 2, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.att_block_1 = TransformerBlock(64, 64)
        self.att_block_2 = TransformerBlock(96, 64)
        self.att_block_3 = TransformerBlock(192, 64)
        # self.att_block_4 = TransformerBlock(192, 64)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.transformer = Encoder_patch66(d_model=1024, d_inner=2048, n_layers=6)
        self.recon_trunk_light = arch_util.make_layer(ResidualBlock_noBN_f, 6)

    def forward(self, x, seg_fea, index):

        if (index == 0):
            L1_fea_1 = self.lrelu(self.conv_first_1(x))
            fea1 = self.feature_extraction(L1_fea_1)
            out_noise = self.recon_trunk(fea1)
            out_noise = self.att_block_1(out_noise, seg_fea[0])
            out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
            out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))
        if (index == 1):
            L1_fea_2 = self.lrelu(self.conv_first_2(x))
            fea2 = self.feature_extraction(L1_fea_2)
            out_noise = self.recon_trunk(fea2)
            out_noise = self.att_block_1(out_noise, seg_fea[1])
            out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
            out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))
        if (index == 2):
            L1_fea_3 = self.lrelu(self.conv_first_3(x))
            fea3 = self.feature_extraction(L1_fea_3)
            out_noise = self.recon_trunk(fea3)
            out_noise = self.att_block_1(out_noise, seg_fea[2])
            out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
            out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))

        return out_noise


if __name__ == '__main__':
    shape = (2, 32, 480, 640)
    x = torch.randn(shape)
    low_light_transformer_seg = DownUpTransformer(nf=128, HR_in=True)

    tensor1 = torch.rand(1, 64, 240, 320)
    tensor2 = torch.rand(1, 128, 120, 160)
    tensor3 = torch.rand(1, 256, 60, 80)

    seg_fea = [tensor1, tensor2, tensor3]
    index = 1
    low_light_transformer_seg(x=x, seg_fea=seg_fea, index=index)
