import torch
from torch import nn


class FeatureMapTransformer(nn.Module):
    def __init__(self, in_dim=2048, sr_ratio=1):
        super(FeatureMapTransformer, self).__init__()
        input_dim = in_dim
        self.chanel_in = input_dim

        self.convChannel16To48 = nn.Conv2d(in_channels=16, out_channels=48, kernel_size=(1, 1), stride=(1, 1))
        self.convChannel32To48 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(1, 1), stride=(1, 1))
        self.convChannel48To16 = nn.Conv2d(in_channels=48, out_channels=16, kernel_size=(1, 1), stride=(1, 1))
        self.convChannel48To32 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=(1, 1), stride=(1, 1))

        self.query_convrd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.key_convrd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.value_convrd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)

        self.query_convdr = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.key_convdr = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.value_convdr = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)

        self.sr_ratio = sr_ratio
        dim = in_dim

        if sr_ratio > 1:
            self.sr_q = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_k = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_v = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)

            self.norm_q = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)
            self.norm_k = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)
            self.norm_v = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)

            self.sr_kk = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_vv = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_kk = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)
            self.norm_vv = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)

        self.gamma_rd = nn.Parameter(torch.zeros(1))
        self.gamma_dr = nn.Parameter(torch.zeros(1))
        self.gamma_x = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Conv2d(dim * 2, dim // 2, kernel_size=1)
        self.fc2 = nn.Conv2d(dim // 2, dim * 2, kernel_size=1)
        self.merge_conv1x1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, 1), self.relu)

    # localVisSegF1,localIrSegF1,x_vis_p,x_inf_p
    def forward(self, visSegF, irSegF, x_vis_p, x_ir_p):
        # xr, xd = x[0].unsqueeze(dim=0), x[1].unsqueeze(dim=0)
        # visSegKey, visQuery = visSegF, x_vis_p  # torch.Size([2, 64, 72, 72]) torch.Size([2, 64, 72, 72])
        m_vis_seg_batchsize, vis_seg_C, vis_seg_width, vis_seg_height = visSegF.size()
        m_vis_batchsize, vis_C, vis_width, vis_height = x_vis_p.size()
        if vis_C == 16:
            visQuery = self.convChannel16To48(x_vis_p)
            vis_C_Change = visQuery.size()[1]
        elif vis_C == 32:
            visQuery = self.convChannel32To48(x_vis_p)
            vis_C_Change = visQuery.size()[1]
        else:
            visQuery = x_vis_p
            vis_C_Change = vis_C

        query_vis = self.query_convrd(visQuery).view(m_vis_seg_batchsize, -1, vis_width * vis_height).permute(0, 2, 1)
        if self.sr_ratio > 1:
            key_d = self.norm_k(self.sr_k(visSegF))
            value_d = self.norm_v(self.sr_v(visSegF))
            key_d = self.key_convrd(key_d).view(m_vis_seg_batchsize, -1,
                                                vis_seg_width // self.sr_ratio * vis_seg_height // self.sr_ratio)
            value_d = self.value_convrd(value_d).view(m_vis_seg_batchsize, -1,
                                                      vis_seg_width // self.sr_ratio * vis_seg_height // self.sr_ratio)
        else:
            key_d = self.key_convrd(visSegF).view(m_vis_seg_batchsize, -1, vis_seg_width * vis_seg_height)
            value_d = self.value_convrd(x_ir_p).view(m_vis_seg_batchsize, -1, vis_seg_width * vis_seg_height)
        attention_rd = self.softmax(torch.bmm(query_vis, key_d))
        out_rd = torch.bmm(value_d, attention_rd.permute(0, 2, 1))
        out_rd = out_rd.view(m_vis_batchsize, vis_C_Change, vis_width, vis_height)
        if vis_C == 16:
            out_rd = self.convChannel48To16(out_rd)
        if vis_C == 32:
            out_rd = self.convChannel48To32(out_rd)

        ir_batchsize, ir_C, ir_width, ir_height = x_ir_p.size()
        if ir_C == 16:
            query_ir = self.convChannel16To48(x_ir_p)
            ir_C_Change = query_ir.size()[1]
        elif ir_C == 32:
            query_ir = self.convChannel32To48(x_ir_p)
            ir_C_Change = query_ir.size()[1]
        else:
            query_ir = x_ir_p
            ir_C_Change = ir_C

        m_ir_seg_batchsize, ir_seg_C, ir_seg_width, ir_seg_height = irSegF.size()

        query_d = self.query_convdr(query_ir).view(ir_batchsize, -1, ir_width * ir_height).permute(0, 2, 1)
        if self.sr_ratio > 1:
            key_r = self.norm_kk(self.sr_kk(irSegF))
            value_r = self.norm_vv(self.sr_vv(irSegF))
            key_r = self.key_convdr(key_r).view(m_ir_seg_batchsize, -1,
                                                ir_seg_width // self.sr_ratio * ir_seg_height // self.sr_ratio)
            value_r = self.value_convdr(value_r).view(m_ir_seg_batchsize, -1,
                                                      ir_seg_width // self.sr_ratio * ir_seg_height // self.sr_ratio)
        else:
            key_r = self.key_convdr(irSegF).view(m_ir_seg_batchsize, -1, ir_seg_width * ir_seg_height)
            value_r = self.value_convdr(irSegF).view(m_ir_seg_batchsize, -1, ir_seg_width * ir_seg_height)
        attention_dr = self.softmax(torch.bmm(query_d, key_r))
        out_dr = torch.bmm(value_r, attention_dr.permute(0, 2, 1))
        out_dr = out_dr.view(ir_batchsize, ir_C_Change, ir_width, ir_height)

        if ir_C == 16:
            out_dr = self.convChannel48To16(out_dr)
        if ir_C == 32:
            out_dr = self.convChannel48To32(out_dr)

        out_rd = self.gamma_rd * out_rd + x_vis_p
        out_rd = self.relu(out_rd)

        out_dr = self.gamma_dr * out_dr + x_ir_p
        out_dr = self.relu(out_dr)
        return out_rd, out_dr
