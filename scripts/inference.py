import argparse
import os.path

import cv2
import torch
import torch.backends.cudnn

import matplotlib.pyplot as plt

from torch.nn import DataParallel

from stacked_hourglass import hg1, hg2, hg8, HumanPosePredictor


def main(args):
    # Select the hardware device to use for inference.
    if torch.cuda.is_available():
        device = torch.device('cuda', torch.cuda.current_device())
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # Disable gradient calculations.
    torch.set_grad_enabled(False)

    pretrained = not args.model_file

    if pretrained:
        print('No model weights file specified, using pretrained weights instead.')

    # Create the model, downloading pretrained weights if necessary.
    if args.arch == 'hg1':
        model = hg1(pretrained=pretrained)
    elif args.arch == 'hg2':
        model = hg2(pretrained=pretrained)
    elif args.arch == 'hg8':
        model = hg8(pretrained=pretrained)
    else:
        raise Exception('unrecognised model architecture: ' + args.model)
    model = model.to(device)

    if not pretrained:
        assert os.path.isfile(args.model_file)
        print('Loading model weights from file: {}'.format(args.model_file))
        checkpoint = torch.load(args.model_file)
        state_dict = checkpoint['state_dict']
        if sorted(state_dict.keys())[0].startswith('module.'):
            model = DataParallel(model)
        model.load_state_dict(state_dict)

    # Initialise the MPII validation set dataloader.
    # val_dataset = Mpii(args.image_path, is_train=False)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
    #                         num_workers=args.workers, pin_memory=True)

    # Generate predictions for the validation set.
    # _, _, predictions = do_validation_epoch(val_loader, model, device, Mpii.DATA_INFO, args.flip)

    model = hg1(pretrained=True)
    predictor = HumanPosePredictor(model, device='cpu')
    # my_image = image_loader("../inference-img/1.jpg")
    # joints = image_inference(predictor, image_path=None, my_image=my_image)
    # imshow(my_image, joints=joints)
    if args.camera == False:
        inference_video(predictor, "../inference-video/R6llTwEh07w.mp4")

    elif args.camera:
        inference_video(predictor, 0)

    # Report PCKh for the predictions.
    # print('\nFinal validation PCKh scores:\n')
    # print_mpii_validation_accuracy(predictions)


global global_img
global_img = None
# 输入图片地址
from PIL import Image, ImageDraw
# import torch
from torchvision import transforms

# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()
from timeit import default_timer as timer


# 返回tensor变量
def image_loader(image_name):
    image = Image.open(image_name)
    return image


def image_to_tensor(image):
    image.convert('RGB')
    image = loader(image).unsqueeze(0)
    if torch.cuda.is_available():
        device = torch.device('cuda', torch.cuda.current_device())
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    return image.to(device, torch.float)


def image_inference(predictor, image_path="../inference-img/1.jpg", my_image=None):
    # if you want to load PIL image without opening a saved image, please explicitly parse image_path=None!
    if image_path is not None:
        my_image = image_loader(image_path)
    my_image_tensor = image_to_tensor(my_image)
    joints = predictor.estimate_joints(my_image_tensor, flip=True).numpy()[0]
    # print(joints)
    return joints


def inference_video(predictor, video_path=0):
    global global_img

    video_capture = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        tic = timer()
        ret, frame = video_capture.read()
        if ret:
            print(idx)
            if (global_img is not None):
                global_img.remove()
            idx = idx + 1
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            joints = image_inference(predictor, None, image)
            global_img = imshow(image, None, joints)
        else:
            exit(0)
        toc = timer()
        print("frame" + str(idx) + " time", toc - tic)  # 输出的时间，秒为单位


def imshow_from_tensor(tensor, title=None, joints=None):
    tic = timer()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    r = image.size[0] / 100

    draw = ImageDraw.Draw(image)
    for joint in joints:
        x = joint[0]
        y = joint[1]
        draw.chord((x - r, y - r, x + r, y + r), 0, 360, (255, 0, 0), (0, 255, 0))
    plt.imshow(image)
    # if title is not None:
    #     plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    toc = timer()
    print("frame" + str(idx) + " time", toc - tic)  # 输出的时间，秒为单位


def imshow(image, title=None, joints=None):
    draw = ImageDraw.Draw(image)
    r = image.size[0] / 100
    for joint in joints:
        x = joint[0]
        y = joint[1]
        draw.chord((x - r, y - r, x + r, y + r), 0, 360, (255, 0, 0), (0, 255, 0))
    ret = plt.imshow(image)
    # if title is not None:
    #     plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a stacked hourglass model.')
    parser.add_argument('--image-path', required=True, type=str,
                        help='path to MPII Human Pose images')
    parser.add_argument('--camera', required=True, default=False, type=bool,
                        help='use real-time video capture or not')
    parser.add_argument('--arch', metavar='ARCH', default='hg1',
                        choices=['hg1', 'hg2', 'hg8'],
                        help='model architecture')
    parser.add_argument('--model-file', default='', type=str, metavar='PATH',
                        help='path to saved model weights')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--batch-size', default=6, type=int, metavar='N',
                        help='batch size')
    parser.add_argument('--flip', dest='flip', action='store_true',
                        help='flip the input during validation')

    main(parser.parse_args())
