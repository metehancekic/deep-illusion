from PIL import Image
import numpy as np

import torch
import torchvision.models as models


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


model = models.wide_resnet50_2(pretrained=True).to(device)

im_frame = Image.open("images/" + 'panda.png')
np_frame = np.array(im_frame.getdata()).reshape(224, 224, 3)/255

np_frame = (np_frame - mean)/std

img = torch.from_numpy(np_frame).float().to(device).permute(2, 0, 1).view(1, 3, 224, 224)

out = model(img)
