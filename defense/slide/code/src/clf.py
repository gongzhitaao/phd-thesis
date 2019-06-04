import imageio
import numpy as np

import torch as th
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

clf = models.inception_v3(pretrained=True)
clf.eval()

pp = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
softmax = nn.Softmax(dim=1)

img = imageio.imread('tmp/get/hello.png')
img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.

with th.no_grad():
    x = th.FloatTensor(img)
    x = pp(x)
    x = th.unsqueeze(x, dim=0)
    logits = clf(x)
    ybar = softmax(logits).detach().cpu().numpy().flatten()

print(np.min(ybar))
print(np.argmax(ybar), np.max(ybar))
print(ybar[779])
