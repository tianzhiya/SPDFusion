import os

from Seg_build_BiSeNet import BiSeNet
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def selectHalfFeatureRand(input_tensor, N):
    # 获取输入张量的通道数
    num_channels = input_tensor.size(1)
    # 计算要选择的特征图数量
    selected_channels = num_channels // N
    # 使用torch.randperm生成一个随机排列索引
    random_indices = torch.randperm(num_channels)[:selected_channels]

    # 从输入张量中选择指定数量的特征图
    selected_features = input_tensor[:, random_indices, :, :]
    return selected_features


def selectNFeatureMapByImportant(data, N):
    # 计算通道的重要程度（这里采用通道的平均值作为重要程度）
    channel_mean = torch.mean(data, dim=(2, 3))  # 形状为(2, 1024)

    # 找到最重要的N个通道的索引
    top_channels = torch.topk(channel_mean, k=N, dim=1)[1]  # 形状为(2, N)

    # 生成向量形式的索引
    indices = torch.arange(data.shape[1], device=top_channels.device).unsqueeze(
        0)  # 形状为(1, 1024)，确保与top_channels张量在相同的设备上
    # 使用索引向量提取最重要的48个通道并保持原始形状
    output = torch.index_select(data, dim=1, index=indices.squeeze(0)[top_channels.flatten()]).view(
        top_channels.shape[0], -1, data.shape[2], data.shape[3])
    output = output.to('cpu')
    output = torch.index_select(output, dim=1, index=torch.arange(N))
    return output


def changeIrTo3Chanel(image_ir):
    rgb_tensor = image_ir.repeat(1, 3, 1, 1)
    return rgb_tensor


class LoadSegFeature:
    def __init__(self, image_vis_ycrcb, image_ir):
        # 构造函数
        self.image_vis_ycrcb = image_vis_ycrcb
        self.image_ir = changeIrTo3Chanel(image_ir)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        visModel = BiSeNet(12, "resnet101")
        irModel = BiSeNet(12, "resnet101")
        #
        for param in visModel.parameters():
            param.requires_grad = False
        for param in irModel.parameters():
            param.requires_grad = False

        if torch.cuda.is_available():
            self.visSegMode = torch.nn.DataParallel(visModel).cuda()
            self.irSegMode = torch.nn.DataParallel(irModel).cuda()
            self.visSegMode(self.image_vis_ycrcb)
            self.irSegMode(self.image_ir)

    def getVisFeature1(self):
        if isinstance(self.visSegMode, torch.nn.DataParallel):
            visFeature1 = self.visSegMode.module.firstFeature  # (list 1,(2,64,240,320))
            feature = visFeature1[0][0]
            # visReshapeFeature1 = selectNFeatureMapByImportant(visFeature1[0], 48)
            # visReshapeFeature1 = F.interpolate(visFeature1[0], size=(480, 640), mode='bilinear')
        return feature

    def getVisFeature2(self):
        if isinstance(self.visSegMode, torch.nn.DataParallel):
            visFeature2 = self.visSegMode.module.secondFeature  # (Tensor (2,128,120,160)
            # visReshapeFeature2 = selectNFeatureMapByImportant(visFeature2, 48)
            # visReshapeFeature2 = F.interpolate(visFeature2, size=(480, 640), mode='bilinear')
        return visFeature2[0][0]

    def getVisFeature3(self):
        if isinstance(self.visSegMode, torch.nn.DataParallel):
            visFeature3 = self.visSegMode.module.thirdFeature  # (Tensor (2,256,60,80)
            # visFeature3 = selectNFeatureMapByImportant(visFeature3, 64)
            # visReshapeFeature3 = F.interpolate(visFeature3, size=(480, 640), mode='bilinear')
        return visFeature3[0][0]

    def getIrFeature1(self):
        if isinstance(self.irSegMode, torch.nn.DataParallel):
            irFeature1 = self.irSegMode.module.firstFeature  # (list 1,(2,64,240,320))
            # visReshapeFeature1 = selectNFeatureMapByImportant(visFeature1[0], 48)
            # irReshapeFeature1 = F.interpolate(irFeature1[0], size=(480, 640), mode='bilinear')
        return irFeature1[0][0]

    def getIrFeature2(self):
        if isinstance(self.irSegMode, torch.nn.DataParallel):
            irFeature2 = self.irSegMode.module.secondFeature  # (Tensor (2,128,120,160)
            # irReshapeFeature2 = F.interpolate(irFeature2, size=(480, 640), mode='bilinear')
        return irFeature2[0][0]

    def getIrFeature3(self):
        if isinstance(self.irSegMode, torch.nn.DataParallel):
            irFeature3 = self.irSegMode.module.thirdFeature  # (Tensor (2,256,60,80)
            # irFeature3 = selectNFeatureMapByImportant(irFeature3, 32)
            # irReshapeFeature3 = F.interpolate(irFeature3, size=(480, 640), mode='bilinear')
        return irFeature3[0][0]

    def getSegDecoderF(self):
        if isinstance(self.visSegMode, torch.nn.DataParallel):
            segDecodeF = self.visSegMode.module.segDecode
            return segDecodeF
