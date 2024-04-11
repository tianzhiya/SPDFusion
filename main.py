# This is a sample Python script.
import argparse
import os.path as osp
import os
import time

import cv2
from torch.autograd import Variable

from ClassAttion import ClassAttion
from FusionNet import FusionNet
import torch
from torch.utils.data import DataLoader

from LoadSegFeature import LoadSegFeature
from Models import D_VI, D_IR
from Seg_DivImage import getMulPicSegResult, class_map_to_mask, imageTo255
from TaskFusion_dataset import Fusion_dataset
from loss import BackgroudLoss, SalientLoss
import datetime
import numpy as np
from PIL import Image
import torch.optim as optim
from args import args

# Press the green button in the gutter to run the script.
VisibleDiscrim = D_VI().cuda()
G = FusionNet(1).cuda()
D_ir = D_IR().cuda()

optimizerD_vi = optim.Adam(VisibleDiscrim.parameters(), args.d_lr)
optimizerD_ir = optim.Adam(D_ir.parameters(), args.d_lr)


def reset_grad(g_optimizer, dir_optimizer, dvi_optimizer):
    dir_optimizer.zero_grad()
    dvi_optimizer.zero_grad()
    g_optimizer.zero_grad()


def np_to_tensor(array):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor = torch.from_numpy(array).to(device)
    return tensor


def train_fusion(i):
    lr_start = 0.001
    modelpth = './model'
    Method = 'Fusion'
    modelpth = os.path.join(modelpth, Method)
    fusionmodel = eval('FusionNet')(output=1)

    fusionmodel.cuda()
    fusionmodel.train()
    optimizerG = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    train_dataset = Fusion_dataset('train')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)
    print(train_loader.n_iter)
    backgroudLoss = BackgroudLoss()
    salientLoss = SalientLoss()

    epoch = 10
    benCiStartTime = globStartTime = time.time()
    print("Train Fusion Model start...")

    for epo in range(0, epoch):
        for it, (image_vis, image_ir, name) in enumerate(train_loader):
            classAttion = ClassAttion(image_vis, image_ir)
            labVisOneHot = classAttion.getLabVisOneHot()
            labIrOneHot = classAttion.getsegLabIrOneHot()

            fusionmodel.train()
            reset_grad(optimizerG, optimizerD_ir, optimizerD_vi)
            image_vis = Variable(image_vis).cuda()
            image_vis_y = image_vis[:, :1, :, :]
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(image_ir).cuda()

            loadSegF = LoadSegFeature(image_vis_ycrcb, image_ir)

            outFusionImage_y = fusionmodel(image_vis_ycrcb, image_ir, loadSegF, labVisOneHot, labIrOneHot)
            fusionImage_ycrcb = torch.cat(
                (outFusionImage_y, image_vis_ycrcb[:, 1:2, :, :],
                 image_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )

            vi_discrim_loss = trainDiscriminatorVis(image_vis_y, optimizerG, outFusionImage_y)

            ir_d_loss = trainDiscriminatorIr(image_ir, optimizerG, outFusionImage_y)

            dir_g_adversarial_loss, dvi_g_adversarial_loss, gloss_total, loss_contain_fusion = trainGeneratorNet(
                backgroudLoss, salientLoss, image_ir, image_vis, image_vis_ycrcb, optimizerG, outFusionImage_y)

            min_loss = float('inf')
            if gloss_total.item() < min_loss:
                min_loss = gloss_total.item()
                fusion_model_file = os.path.join(modelpth, 'fusion_model.pth')
                torch.save(fusionmodel.state_dict(), fusion_model_file)

            printTrainInfo(benCiStartTime, dir_g_adversarial_loss, dvi_g_adversarial_loss, epo, epoch, globStartTime,
                           gloss_total, ir_d_loss, it, loss_contain_fusion, train_loader, vi_discrim_loss)

        min_loss = float('inf')
    if loss_contain_fusion.item() < min_loss:
        min_loss = loss_contain_fusion.item()
        fusion_model_file = os.path.join(modelpth, 'fusion_model.pth')
        torch.save(fusionmodel.state_dict(), fusion_model_file)


def printTrainInfo(benCiStartTime, dir_g_adversarial_loss, dvi_g_adversarial_loss, epo, epoch, globStartTime,
                   gloss_total, ir_d_loss, it, loss_contain_fusion, train_loader, vi_discrim_loss):
    endTime = time.time()
    t_intv, glob_t_intv = endTime - benCiStartTime, endTime - globStartTime
    currentIter = train_loader.n_iter * epo + it + 1
    eta = int((train_loader.n_iter * epoch - currentIter)
              * (glob_t_intv / (currentIter)))
    eta = str(datetime.timedelta(seconds=eta))
    if currentIter % 10 == 0:
        msg = ', '.join(
            [
                'step: {cur_it}/{max_it}',
                '总gloss_total: {gloss_total:.4f}',
                '判别器可见光vi_d_loss: {vi_discrim_loss:.4f}',
                '判别器红外ir_d_loss: {ir_d_loss:.4f}',
                '生成器loss_total: {loss_total:.4f}',
                "其中loss_contain_fusion:{loss_contain_fusion:.4f}",
                "dir_g_adversarial_loss:{dir_g_adversarial_loss:.4f}",
                "dvi_g_adversarial_loss：{dvi_g_adversarial_loss:.4f}",
                'eta: {eta}',
                'time: {time:.4f}',
            ]
        ).format(
            cur_it=currentIter,
            gloss_total=gloss_total,
            max_it=train_loader.n_iter * epoch,
            vi_discrim_loss=vi_discrim_loss.item(),
            ir_d_loss=ir_d_loss.item(),
            loss_total=gloss_total.item(),
            loss_contain_fusion=loss_contain_fusion,
            dir_g_adversarial_loss=dir_g_adversarial_loss,
            dvi_g_adversarial_loss=dvi_g_adversarial_loss,
            time=t_intv,
            eta=eta,
        )
        print(msg)
        benCiStartTime = endTime


def getSalientMask(predictSegPicMask):
    data_integer = np.round(predictSegPicMask).to(torch.int)
    salientMask = torch.where(data_integer == 5, torch.tensor(1), torch.tensor(0))
    return salientMask


def trainGeneratorNet(backgroudLoss, salientLoss, image_ir, image_vis, image_vis_ycrcb, optimizerG, outFusionImage_y):
    image_visForSeg = imageTo255(image_vis)
    predictSegPicMask = getMulPicSegResult(image_visForSeg)[0]
    salientMask = getSalientMask(predictSegPicMask)
    num_classes = 12
    predictSegPicMaskBatchSize = predictSegPicMask.size(0)
    loss_contain_fusion_T = 0.0

    for k in range(predictSegPicMaskBatchSize):
        imageMask = predictSegPicMask[k]
        imageMask = imageMask.squeeze(0)
        resultImage = outFusionImage_y[k]
        for i in range(num_classes):
            nClassMasks = class_map_to_mask(imageMask, num_classes)
            nClassMasks = np_to_tensor(nClassMasks)
            for j in range(nClassMasks.size(0)):
                outFusionImage_fragment = nClassMasks[j] * resultImage
                originalImage_fragment = nClassMasks[j] * image_vis_ycrcb
                if (num_classes != 5):
                    backgroudLossTotal, backgroudLoss_in, backgroudLoss_grad = backgroudLoss(
                        originalImage_fragment, image_ir, outFusionImage_fragment, 0
                    )
                    loss_contain_fusion_T = loss_contain_fusion_T + backgroudLossTotal.item()
                else:
                    salientLossTotal, salientLoss_in, salientLoss_grad = salientLoss(
                        originalImage_fragment, image_ir, outFusionImage_fragment, 0
                    )
                    loss_contain_fusion_T = loss_contain_fusion_T + salientLossTotal.item()

    dir_g_adversarial_loss = -D_ir(outFusionImage_y).mean()
    dvi_g_adversarial_loss = -VisibleDiscrim(outFusionImage_y).mean()
    g_adversarial_loss = (dir_g_adversarial_loss + dvi_g_adversarial_loss)
    genArg = 1
    gloss_total = backgroudLossTotal + genArg * g_adversarial_loss
    reset_grad(optimizerG, optimizerD_ir, optimizerD_vi)
    gloss_total.backward()
    optimizerG.step()
    return dir_g_adversarial_loss, dvi_g_adversarial_loss, gloss_total, backgroudLossTotal


def trainDiscriminatorIr(image_ir, optimizerG, outFusionImage_y):
    for _ in range(2):
        gamma_ = 10
        all_ir_d_loss = 0
        for _ in range(2):
            D_out_ir = D_ir(image_ir)
            D_loss_ir = - torch.mean(D_out_ir)

            fusionImageDiscrimOut = D_ir(outFusionImage_y.detach())
            discrim_fusionImg_loss = fusionImageDiscrimOut.mean()

            alpha_ir = torch.rand(image_ir.size(0), 1, 1, 1).cuda().expand_as(image_ir)
            interpolated_ir = Variable(alpha_ir * image_ir.data + (1 - alpha_ir) * outFusionImage_y.data,
                                       requires_grad=True)
            Dir_interpolated = D_ir(interpolated_ir)
            grad_ir = torch.autograd.grad(outputs=Dir_interpolated,
                                          inputs=interpolated_ir,
                                          grad_outputs=torch.ones(Dir_interpolated.size()).cuda(),
                                          retain_graph=True,
                                          create_graph=True,
                                          only_inputs=True)[0]
            grad_ir = grad_ir.view(grad_ir.size(0), -1)
            grad_ir_l2norm = torch.sqrt(torch.sum(grad_ir ** 2, dim=1))
            Dir_penalty = torch.mean((grad_ir_l2norm - 1) ** 2)

            ir_d_loss = D_loss_ir + discrim_fusionImg_loss + Dir_penalty * gamma_
            all_ir_d_loss += ir_d_loss.item()

            reset_grad(optimizerG, optimizerD_ir, optimizerD_vi)
            ir_d_loss.backward(retain_graph=True)
            optimizerD_ir.step()
    return ir_d_loss


def trainDiscriminatorVis(image_vis_y, optimizerG, outFusionImage_y):
    gamma_ = 10
    all_vi_d_loss = 0

    for _ in range(2):
        visibleDiscrimOut = VisibleDiscrim(image_vis_y)
        discrim_vi_loss = - torch.mean(visibleDiscrimOut)

        fusionImageDiscrimOut = VisibleDiscrim(outFusionImage_y.detach())
        discrim_fusionImg_loss = fusionImageDiscrimOut.mean()
        alpha_vi = torch.rand(image_vis_y.size(0), 1, 1, 1).cuda().expand_as(image_vis_y)
        interpolated_vi = Variable(alpha_vi * image_vis_y.data + (1 - alpha_vi) * outFusionImage_y.data,
                                   requires_grad=True)
        Dvi_interpolated = VisibleDiscrim(interpolated_vi)
        grad_vi = torch.autograd.grad(outputs=Dvi_interpolated,
                                      inputs=interpolated_vi,
                                      grad_outputs=torch.ones(Dvi_interpolated.size()).cuda(),
                                      retain_graph=True,
                                      create_graph=True,
                                      only_inputs=True)[0]
        grad_vi = grad_vi.view(grad_vi.size(0), -1)
        grad_vi_l2norm = torch.sqrt(torch.sum(grad_vi ** 2, dim=1))

        Dvi_penalty = torch.mean((grad_vi_l2norm - 1) ** 2)

        vi_discrim_loss = discrim_vi_loss + discrim_fusionImg_loss + Dvi_penalty * gamma_
        all_vi_d_loss += vi_discrim_loss.item()

        reset_grad(optimizerG, optimizerD_ir, optimizerD_vi)
        vi_discrim_loss.backward(retain_graph=True)
        optimizerD_vi.step()
    return vi_discrim_loss


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def test_fusion(type='MSRS'):
    fusion_model_path = './model/Fusion/fusion_model.pth'
    fused_dir = os.path.join('./Fusion', type)
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    fusionmodel = eval('FusionNet')(output=1)
    fusionmodel.eval()
    # if args.gpu >= 0:
    # fusionmodel.cuda(args.gpu)
    fusionmodel.cuda()
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    print('done!')
    test_dataset = Fusion_dataset(type)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir, name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images_vis = images_vis.cuda(args.gpu)
                images_ir = images_ir.cuda(args.gpu)
            images_vis_ycrcb = RGB2YCrCb(images_vis)
            classAttion = ClassAttion(images_vis, images_ir)

            labVisOneHot = classAttion.getLabVisOneHot()
            labIrOneHot = classAttion.getsegLabIrOneHot()

            loadSegF = LoadSegFeature(images_vis_ycrcb, images_ir)
            fusionResult_y = fusionmodel(images_vis_ycrcb, images_ir, loadSegF, labVisOneHot, labIrOneHot)
            fusionResult_ycrcb = torch.cat(
                (fusionResult_y, images_vis_ycrcb[:, 1:2, :,
                                 :], images_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusionResult_RGB = YCrCb2RGB(fusionResult_ycrcb)
            ones = torch.ones_like(fusionResult_RGB)
            zeros = torch.zeros_like(fusionResult_RGB)
            fusionResult_RGB = torch.where(fusionResult_RGB > ones, ones, fusionResult_RGB)
            fusionResult_RGB = torch.where(
                fusionResult_RGB < zeros, zeros, fusionResult_RGB)
            fused_image = fusionResult_RGB.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                    np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = image.squeeze()
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='SDFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=4)
    parser.add_argument('--isTrain', '-C', type=int, default=0)
    args = parser.parse_args()
    if (args.isTrain):
        for i in range(1):
            train_fusion(i)
    else:
        with torch.no_grad():
            test_fusion('MSRS')
