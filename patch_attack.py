import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

import cv2
import numpy as np

from pytorch_auto_drive.utils.lr_schedulers.builder import LR_SCHEDULERS
from pytorch_auto_drive.utils.common import load_checkpoint
from pytorch_auto_drive.utils.args import read_config
from pytorch_auto_drive.utils.models import MODELS
from pytorch_auto_drive.utils.losses import LOSSES
from pytorch_auto_drive.utils.runners.base import BaseTrainer
import torch


CONFIG=os.path.dirname(os.getcwd()) + '/pytorch_auto_drive/configs/lane_detection/resa/resnet50_culane.py'
cfg = read_config(CONFIG)
model = MODELS.from_dict(cfg['model'])

optimizer = BaseTrainer.get_optimizer(cfg['optimizer'], model)
# lr_scheduler = LR_SCHEDULERS.from_dict(cfg['lr_scheduler'], model, len(cfg['train_loader'])

loss_config = loss = dict(
    name='LaneLossSeg',
    ignore_index=255,
    weight=[0.4, 1, 1, 1, 1]
)

loss = LOSSES.from_dict(loss_config)
num_classes = cfg['train']['num_classes']
input_size = cfg['train']['input_size']

load_checkpoint(net=model, optimizer=None, lr_scheduler=None, filename=os.path.dirname(os.getcwd()) + '/../../resnet50_resa_culane_20211016.pt', strict=False)

clip_values = (0, 255)

# create a classifier
from estimator import MyPyTorchClassifier
import pytorch_auto_drive.utils.transforms.functional as F
from PIL import Image

classifier = MyPyTorchClassifier(
    model=model,
    loss=loss,
    clip_values=clip_values,
    optimizer=optimizer,
    input_shape=(1, input_size[0], input_size[1]),
    nb_classes=num_classes,
)

image = Image.open('../../lanefitting/example_input.jpg')
orig_sizes = (image.height, image.width)
original_img = F.to_tensor(image).clone().unsqueeze(0)
image = F.resize(image, size=input_size) # type: ignore

model_in = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))

model_in = model_in.view(image.size[1], image.size[0], len(image.getbands()))
model_in = (
    model_in.permute((2, 0, 1)).contiguous().float().div(255).unsqueeze(0).numpy()
)

def save_image(image, path):
    image = image.transpose((1, 2, 0))
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    image = cv2.resize(image, (orig_sizes[1], orig_sizes[0]))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)

print("Starting Fast Gradient Method attack")
from art.attacks.evasion import FastGradientMethod
attack = FastGradientMethod(estimator=classifier, eps=0.01)
x_test_adv = attack.generate(x=model_in)
print("Completed Fast Gradient Method attack")

# # save image
# save_image(x_test_adv[0], 'adv_img_fast_gradient.jpg')

# from art.attacks.evasion import BasicIterativeMethod
# attack = BasicIterativeMethod(estimator=classifier, eps=0.01)
# x_test_adv = attack.generate(x=model_in)
# save_image(x_test_adv[0], 'adv_img_iterative.jpg')

print("Starting patch attack")
from adversarial_patch_pytorch import MyAdversarialPatchPyTorch
attack = MyAdversarialPatchPyTorch(estimator=classifier, max_iter=1000)
x_test_adv = attack.generate(x=model_in)
save_image(x_test_adv[0], 'adv_img_patch.jpg')
print("Completed patch attack")