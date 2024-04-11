import argparse

from PathArgs import PathArgs
from Seg_build_BiSeNet import BiSeNet
import os
import torch
import cv2
from imgaug import augmenters as iaa
from PIL import Image
from torchvision import transforms
import numpy as np

from Seg_utils import get_label_info, reverse_one_hot, colour_code_segmentation


def getSegResult(inputImage):
    # build model
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', action='store_true', default=False, help='predict on image')
    parser.add_argument('--video', action='store_true', default=False, help='predict on video')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='The path1 to the pretrained weights of model')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path1 model you are using.')
    parser.add_argument('--num_classes', type=int, default=12, help='num of object classes (with void)')
    parser.add_argument('--data', type=str, default=None, help='Path to image or video for prediction')
    parser.add_argument('--crop_height', type=int, default=640, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=480, help='Width of cropped/resized input image to network')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--csv_path', type=str, default=None, required=True, help='Path to label info csv file')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='Path to save predict image')

    params = [
        '--image',
        '--data', '00539D.png',
        '--checkpoint_path', './best_dice_loss_miou_0.655.pth',
        '--cuda', '0',
        '--csv_path', './class_dict.csv',
        '--save_path', 'demoVi.png',
        '--context_path', 'resnet18'
    ]
    args = parser.parse_args(params)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model if exists
    print('load model from %s ...' % args.checkpoint_path)
    model.module.load_state_dict(torch.load(args.checkpoint_path))
    print('Done!')

    # predict on image
    image = inputImage
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resize = iaa.Scale({'height': args.crop_height, 'width': args.crop_width})
    resize_det = resize.to_deterministic()
    image = resize_det.augment_image(image)
    image = Image.fromarray(image).convert('RGB')
    image = transforms.ToTensor()(image)
    image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image).unsqueeze(0)
    # read csv label path1
    label_info = get_label_info(args.csv_path)
    # predict
    model.eval()
    predict = model(image).squeeze()
    predictSegPic = reverse_one_hot(predict)

    return predictSegPic


def getMulPicSegResult(inputImage):
    path = PathArgs()
    # build model
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', action='store_true', default=False, help='predict on image')
    parser.add_argument('--video', action='store_true', default=False, help='predict on video')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='The path1 to the pretrained weights of model')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path1 model you are using.')
    parser.add_argument('--num_classes', type=int, default=12, help='num of object classes (with void)')
    parser.add_argument('--data', type=str, default=None, help='Path to image or video for prediction')
    parser.add_argument('--crop_height', type=int, default=path.mImageHeight,
                        help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=path.mImageWidth,
                        help='Width of cropped/resized input image to network')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--csv_path', type=str, default=None, required=True, help='Path to label info csv file')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='Path to save predict image')

    params = [
        '--image',
        '--data', '00539D.png',
        '--checkpoint_path', './best_dice_loss_miou_0.655.pth',
        '--cuda', '0',
        '--csv_path', './class_dict.csv',
        '--save_path', 'demoVi.png',
        '--context_path', 'resnet18'
    ]
    args = parser.parse_args(params)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model if exists
    model.module.load_state_dict(torch.load(args.checkpoint_path))

    imageBatchSize = inputImage.size(0)
    result = torch.empty(imageBatchSize, inputImage.size(2), inputImage.size(3))
    for i in range(imageBatchSize):
        # predict on image
        image = inputImage[i]
        image = image.squeeze(0)
        image = image.permute(1, 2, 0)
        tensor_detached = image.detach()
        image_np = tensor_detached.cpu().numpy()
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        height = image_np.shape[0]
        width = image_np.shape[1]
        resize = iaa.Scale({'height': height, 'width': width})
        resize_det = resize.to_deterministic()
        image = resize_det.augment_image(image)
        image = Image.fromarray(np.uint8(image)).convert('RGB')
        image = transforms.ToTensor()(image)
        image1 = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image).unsqueeze(0)
        # read csv label path1
        label_info = get_label_info(args.csv_path)
        # predict
        model.eval()
        predict = model(image1).squeeze()
        predictW=predict.size(1)
        if predictW!=width:
            predict = torch.nn.functional.interpolate(predict.unsqueeze(0), size=(height, width), mode='bilinear',
                                                      align_corners=True)
            predict = predict.squeeze(0)

        predictSegPic = reverse_one_hot(predict)
        result[i] = predictSegPic
    return result, predict


def class_map_to_mask(class_map, num_classes):
    """
    将语义分割的分类图转换为针对每个类别的掩码。

    参数：
        class_map：语义分割的分类图，大小为(height, width)。
        num_classes：类别数。

    返回值：
        masks：掩码数组，大小为(num_classes, height, width)。
    """
    masks = np.zeros((num_classes, class_map.shape[0], class_map.shape[1]), dtype=np.uint8)
    for i in range(num_classes):
        masks[i] = (class_map == i).cpu().numpy().astype(np.uint8)
    return masks


def imageTo255(image_RGB):
    ones = torch.ones_like(image_RGB)
    zeros = torch.zeros_like(image_RGB)
    image_RGB = torch.where(image_RGB > ones, ones, image_RGB)
    image_RGB = torch.where(
        image_RGB < zeros, zeros, image_RGB)
    image = image_RGB.cpu().numpy()
    image = (image - np.min(image)) / (
            np.max(image) - np.min(image)
    )
    image = torch.from_numpy(np.uint8(255.0 * image))

    return image


if __name__ == '__main__':
    params = [
        '--image',
        '--data', '00539D.png',
        '--checkpoint_path', './best_dice_loss_miou_0.655.pth',
        '--cuda', '0',
        '--csv_path', './class_dict.csv',
        '--save_path', 'demoVi.png',
        '--context_path', 'resnet18'
    ]

    image = cv2.imread('00539D.png', -1)
    predictSegPic = getSegResult(image)
    num_classes = 12
    for i in range(num_classes):
        masks = class_map_to_mask(predictSegPic, num_classes)
        print(masks)
