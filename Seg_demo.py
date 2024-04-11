import argparse
from Seg_build_BiSeNet import BiSeNet
import os
import torch
import cv2
from imgaug import augmenters as iaa
from PIL import Image
from torchvision import transforms
import numpy as np
from Seg_utils import reverse_one_hot, get_label_info, colour_code_segmentation


def predict_on_image(model, args):
    # pre-processing on image
    image = cv2.imread(args.data, -1)
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
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
        feature = model.firstFeature  # (list 1,(1,256,90,120))
        feature2 = model.secondFeature  # (Tensor (1,256,45,60)
        feature3 = model.thirdFeature  # (Tensor (1,512,23,30)
    predict = reverse_one_hot(predict)

    predict = colour_code_segmentation(np.array(predict.cpu()), label_info)
    predict = cv2.resize(np.uint8(predict), (960, 720))
    cv2.imwrite(args.save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', action='store_true', default=False, help='predict on image')
    parser.add_argument('--video', action='store_true', default=False, help='predict on video')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='The path1 to the pretrained weights of model')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path1 model you are using.')
    parser.add_argument('--num_classes', type=int, default=12, help='num of object classes (with void)')
    parser.add_argument('--data', type=str, default=None, help='Path to image or video for prediction')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--csv_path', type=str, default=None, required=True, help='Path to label info csv file')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='Path to save predict image')

    args = parser.parse_args(params)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model if exists
    print('load model from %s ...' % args.checkpoint_path)
    model.module.load_state_dict(torch.load(args.checkpoint_path))
    print('Done!')

    # predict on image
    if args.image:
        predict_on_image(model, args)

    # predict on video
    if args.video:
        pass


if __name__ == '__main__':
    params = [
        '--image',
        '--data', '00318.png',
        '--checkpoint_path', './best_dice_loss_miou_0.655.pth',
        '--cuda', '0',
        '--csv_path', './class_dict.csv',
        '--save_path', 'demo.png',
        '--context_path', 'resnet18'
    ]

    tensor = torch.randn(2, 2, 3)
    print(tensor)
    main(params)
