import torch
import torch.nn.functional as F

from Seg_DivImage import imageTo255, getMulPicSegResult


class ClassAttion():
    def __init__(self, image_vis, image_ir):
        super(ClassAttion, self).__init__()
        self.image_vis = image_vis
        self.image_ir = image_ir
        self.segLabVisOneHot = getMulPicSegResult(imageTo255(image_vis))[0]
        self.segLabIrOneHot = getMulPicSegResult(imageTo255(self.changeIrTo3Chanel(image_ir)))[0]

        self.segGraphAttionVis = getMulPicSegResult(imageTo255(image_vis))[1]
        self.segGraphAttionIr = getMulPicSegResult(imageTo255(self.changeIrTo3Chanel(image_ir)))[1]

    def getVisAttentionWeight(self):
        # pixel_counts = [torch.sum(torch.eq(seg_map, label)) for label in class_labels]
        attention = F.sigmoid(self.personCarOtherSet0(self.segLabVisOneHot))
        return attention

    def getIrAttentionWeight(self):
        attention = F.sigmoid(self.personCarOtherSet0(self.segLabIrOneHot))
        return attention

    def getMaxVisIrAttentionWeight(self):
        visAttentionWeight = self.getVisAttentionWeight()
        irAtttentionWeight = self.getIrAttentionWeight()
        max_values, _ = torch.max(torch.stack([visAttentionWeight, irAtttentionWeight]), dim=0)
        return max_values

    def getMinVisIrAttentionWeight(self):
        visAttentionWeight = self.getVisAttentionWeight()
        irAtttentionWeight = self.getIrAttentionWeight()
        min_values, _ = torch.min(torch.stack([visAttentionWeight, irAtttentionWeight]), dim=0)
        return min_values

    def changeIrTo3Chanel(self, image_ir):
        rgb_tensor = image_ir.repeat(1, 3, 1, 1)
        return rgb_tensor

    def personCarOtherSet0(self, segLab):
        mask = (segLab != 2) & (segLab != 5)
        segLab[mask] = 0
        segLab[segLab == 2] = -2
        segLab[segLab == 5] = -2
        return segLab

    def getSegGraphAttionVis(self):
        return self.segGraphAttionVis

    def getSegGraphAttionIr(self):
        return self.segGraphAttionIr

    def getLabVisOneHot(self):
        return self.segLabVisOneHot

    def getsegLabIrOneHot(self):
        return self.segLabIrOneHot
