"""
ImageNet Validation Script
Adapted from https://github.com/rwightman/pytorch-image-models
The script is further extend to evaluate VOLO
"""
import argparse
import os
import csv
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy
import models

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')


from models.volo import *
from utils import load_pretrained_weights 

from torchvision import datasets, models, transforms
import PIL
from PIL import Image
import cv2
import ipdb

class opt:
    weight_path = "weights/d1_224_84.2.pth.tar"
    arch = "efficientnet-b2"
    conf_thres = 0.5
    device = 'cpu'
    image_size = 224  #'inference size (pixels)'
    advprop = False
    pretrained = False


image_size = opt.image_size

if opt.advprop:
    normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
else:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])


# Applying Transforms to the Data
image_transforms = {
    'train': transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ]),
    'valid': transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ]),
    'test': transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
}

# create model
model = volo_d1()

# load the pretrained weights
# change num_classes based on dataset, can work for different image size 
# as we interpolate the position embeding for different image size.
load_pretrained_weights(model, opt.weight_path, use_ema=False, 
                        strict=False, num_classes=1000)

# ipdb.set_trace()


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')

parser.add_argument('-ic', '--image_classification', type=str, default='images/000000050811.jpg', help='分类模型测试图片')
parser.add_argument('--model', '-m', metavar='NAME', default='volo_d1',
                    help='model architecture (default: dpn92)')
parser.add_argument('--checkpoint', default='weights/', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')

##image classification
def inference_classification(test_image):
    '''
    Function to predict the class of a single test image
    Parameters
        :param model: Model to test
        :param test_image_name: Test image

    '''

    transform = image_transforms['test']

    # test_image = Image.open(test_image_name)
    # test_image_np = cv2.imread(image_name)
    # test_image = Image.fromarray(test_image_np)  # 这里test_image_np为原来的numpy数组类型的输入

    test_image_tensor = transform(test_image)
    test_image_tensor = test_image_tensor.view(1, 3, image_size, image_size)
    # if torch.cuda.is_available():
    #     test_image_tensor = test_image_tensor.view(1, 3, image_size, image_size).cuda()
    # else:
    #     test_image_tensor = test_image_tensor.view(1, 3, image_size, image_size)

    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ipdb.set_trace()
        # ps = torch.exp(out)
        ps = torch.nn.functional.softmax(out, dim =1)
    
    return 0, 0

def main():
    setup_default_logging()
    args = parser.parse_args()

    image_path = args.image_classification
    assert os.path.exists(image_path), 'image_path is not exist'
    image_cv = cv2.imread(image_path)
    assert image_cv is not None, 'image_cv imread Failed'
    image_pil = Image.fromarray(image_cv)
    classification_start = time.time()
    cls_cn, score = inference_classification(image_pil)


if __name__ == '__main__':
    main()
