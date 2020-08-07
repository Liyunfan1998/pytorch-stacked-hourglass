import argparse
import os
import cv2
import torch
import torch.backends.cudnn
import matplotlib.pyplot as plt
from torch.nn import DataParallel
import sys

# setting PYTHONPATH
# print(os.path.abspath('../hourglass-pkg/src/stacked_hourglass'))
# os.system("export PYTHONPATH=\"${PYTHONPATH}:" + os.path.abspath('../hourglass-pkg/src/stacked_hourglass') + "\"")
# os.system("export PYTHONPATH=\"${PYTHONPATH}:" + os.path.abspath('./stacked_hourglass') + "\"")
# sys.path.append(os.path.abspath('..') + '/src/stacked_hourglass')  # ??
# sys.path.append(os.path.abspath('.') + '/stacked_hourglass')  # ??
from Tools.stacked_hourglass import hg1, HumanPosePredictor
from PIL import Image, ImageDraw
from torchvision import transforms
from timeit import default_timer as timer
from time import sleep


class embedded_inference:

    def __init__(self):
        self.global_img = None
        self._run_flag = True

        # loader使用torchvision中自带的transforms函数
        self.loader = transforms.Compose([
            transforms.ToTensor()])
        self.unloader = transforms.ToPILImage()
        self.parser = argparse.ArgumentParser(description='Evaluate a stacked hourglass model.')
        self.parser.add_argument('--image-path', required=False, type=str,
                                 help='path to MPII Human Pose images')
        self.parser.add_argument('--camera', required=False, default=True, type=bool,
                                 help='use real-time video capture or not')
        self.parser.add_argument('--arch', metavar='ARCH', default='hg1',
                                 choices=['hg1', 'hg2', 'hg8'],
                                 help='model architecture')
        self.parser.add_argument('--model-file', default='', type=str, metavar='PATH',
                                 help='path to saved model weights')
        self.parser.add_argument('--workers', default=4, type=int, metavar='N',
                                 help='number of data loading workers')
        self.parser.add_argument('--batch-size', default=6, type=int, metavar='N',
                                 help='batch size')
        self.parser.add_argument('--flip', dest='flip', action='store_true',
                                 help='flip the input during validation')
        # self.main(self.parser.parse_args())

    def main(self, args=None):
        # Select the hardware device to use for inference.
        if torch.cuda.is_available():
            device = torch.device('cuda', torch.cuda.current_device())
            torch.backends.cudnn.benchmark = True

        else:
            device = torch.device('cpu')
        map_location = device

        # Disable gradient calculations.
        torch.set_grad_enabled(False)

        pretrained = not args.model_file

        if pretrained:
            print('No model weights file specified, using pretrained weights instead.')

        # Create the model, downloading pretrained weights if necessary.
        model = hg1(pretrained=pretrained)
        model = model.to(device)
        """
        if not pretrained:
            assert os.path.isfile(args.model_file)
            print('Loading model weights from file: {}'.format(args.model_file))
            checkpoint = torch.load(args.model_file, map_location=map_location)
            state_dict = checkpoint['state_dict']
            if sorted(state_dict.keys())[0].startswith('module.'):
                model = DataParallel(model)
            model.load_state_dict(state_dict)
        """
        model = hg1(pretrained=True)
        predictor = HumanPosePredictor(model, device='cpu')
        # my_image = image_loader("../inference-img/1.jpg")
        # joints = image_inference(predictor, image_path=None, my_image=my_image)
        # self.imshow(my_image, joints=joints)
        if args.camera == False:
            self.inference_video(predictor, "../inference-video/R6llTwEh07w.mp4")
        elif args.camera:
            self.inference_video(predictor, 0)

    def draw_joints(self, joints, cv2image=None):
        # cv2image = zeros((300, 300, 3)) if cv2image is None
        r = int(cv2image.shape[0] / 100)
        red = (255, 0, 0)
        for joint in joints:
            x = joint[0]
            y = joint[1]
            cv2.circle(cv2image, (x, y), r, red, -1)

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

    def image_loader(self, image_name):
        image = Image.open(image_name)
        return image

    def image_to_tensor(self, image):
        # 返回tensor变量
        image.convert('RGB')
        image = self.loader(image)
        print(type(image))
        image = image.unsqueeze(0)
        print(type(image))
        if torch.cuda.is_available():
            device = torch.device('cuda', torch.cuda.current_device())
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device('cpu')
        return image.to(device, torch.float)

    def image_inference(self, predictor, image_path="../inference-img/1.jpg", my_image=None):
        # if you want to load PIL image without opening a saved image, please explicitly parse image_path=None!
        if image_path is not None:
            my_image = self.image_loader(image_path)
        my_image_tensor = self.image_to_tensor(my_image)
        joints = predictor.estimate_joints(my_image_tensor, flip=True).numpy()[0]
        # print(joints)
        return joints

    def tensor_inference(self, predictor, my_image_tensor, my_image=None):
        return predictor.estimate_joints(my_image_tensor, flip=True).numpy()[0]

    """
    def inference_video(self, predictor, video_path=0):
        video_capture = cv2.VideoCapture(video_path)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
        idx = 0
        while True:
            tic = timer()
            ret, frame = cap = video_capture.read()
            # print(type(frame))
            if ret:
                if (self.global_img is not None):
                    self.global_img.remove()
                idx = idx + 1
                inference_tic = timer()
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                joints = self.image_inference(predictor, None, image)
                inference_toc = timer()
                self.global_img = self.imshow(image, None, joints)
                # print(type(self.global_img))
            else:
                exit(0)
            toc = timer()
            print("frame" + str(idx) + "; total time", toc - tic, "; inference time:",
                  inference_toc - inference_tic)  # 输出的时间，秒为单位
    """

    def inference_video(self, predictor, video_path=0):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
        cap.set(cv2.CAP_PROP_FPS, 10)
        device = torch.device('cpu')
        idx = 0
        # capture from web cam
        while self._run_flag:
            tic = timer()
            ret, frame = cap.read()
            if ret:
                idx = idx + 1
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # https://stackoverflow.com/a/55468544/6622587
                inference_tic = timer()
                image_tensor = self.loader(rgbImage).unsqueeze(0).to(device, torch.float)
                inference_toc = timer()
                """
                frame: 
                rgbImage: 'numpy.ndarray'
                image_tensor: torch tensor
                """
                joints = self.tensor_inference(predictor, image_tensor)
                self.draw_joints(joints, rgbImage)
                # print(type(rgbImage))
                # sleep(0.001)
                cv2.imshow('inference', rgbImage)
                if cv2.waitKey(1) == ord('q'):
                    exit(0)
            else:
                exit(0)
            toc = timer()
            print("frame" + str(idx) + "; total time", toc - tic, "; inference time:",
                  inference_toc - inference_tic)  # 输出的时间，秒为单位
        # shut down capture system
        cap.release()

    def imshow_from_tensor(self, tensor, title=None, joints=None):
        tic = timer()
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = self.unloader(image)
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
        print("imshow_from_tensor time", toc - tic)  # 输出的时间，秒为单位

    def imshow(self, image, title=None, joints=None):
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
    video_inf = embedded_inference()
    video_inf.main(video_inf.parser.parse_args())
