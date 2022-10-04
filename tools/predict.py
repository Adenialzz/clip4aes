import torch
import timm
import cv2
import numpy as np
import sys

from get_score import get_score


image_path = sys.argv[1]
# arch = 'swin_base_patch4_window7_224'
arch = 'resnet50'

model = timm.create_model(arch, pretrained=False, num_classes=10)
# ckpt = torch.load('/media/song/iaa/weights/swin_base_patch4_window7_224_lr1e-3/model_10.pth', map_location='cpu')
ckpt = torch.load('/media/song/iaa/weights/resnet50_lr1e-3/model_15.pth', map_location='cpu')
state_dict = ckpt['state_dict']
model.load_state_dict(state_dict)
softmax = torch.nn.Softmax(dim=1)


img = cv2.imread(image_path)[:, :, ::-1]
img = cv2.resize(img, (224, 224))
img = img.transpose(2, 0, 1).astype(np.float32)  # HWC -> CHW
tensor_img = torch.from_numpy(img)[None]

logits = model(tensor_img)
prob = softmax(logits)
score = get_score(prob, 'cpu')
print(prob)
print(score)
