import os

import numpy as np

import torch as th
import torch.nn as nn

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.models import GoogLeNet
from torch.utils.data import DataLoader

from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # noqa: E402
import matplotlib.pyplot as plt


def fgsm(net, x, eps=0.03, clip=[-1, 1]):
    x.requires_grad = True
    out = net(x)
    targets = th.argmax(out.logits, dim=-1)
    crit = th.nn.CrossEntropyLoss()
    loss = crit(out.logits, targets)
    loss.backward()
    adv = x + eps * x.grad
    return adv.detach()


normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
trans = transforms.Compose([transforms.ToTensor(), normalize])

unnormalize = transforms.Normalize(
    (-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010),
    (1 / 0.2023, 1 / 0.1994, 1 / 0.2010))

data_test = CIFAR10(os.path.expanduser('~/data/large'),
                    train=False,
                    download=True,
                    transform=trans)

data_test_loader = DataLoader(data_test,
                              batch_size=8,
                              shuffle=False,
                              num_workers=8,
                              pin_memory=True)

device = 'cuda:7'
net = GoogLeNet(num_classes=10)
net.to(device)
crit = nn.CrossEntropyLoss()

net.load_state_dict(th.load('tmp/cifar10.pth'))
net.eval()
for p in net.parameters():
    p.requires_grad = False

total_correct = 0
acc, loss = 0, 0.0

pbar = tqdm(data_test_loader, ascii=True)
pbar.set_postfix(loss='inf')
crit = th.nn.CrossEntropyLoss()

EPS = 0.1

for batch, (images, labels) in enumerate(pbar):
    X_batch = images.to(device)
    X_batch.requires_grad = True

    logits = net(X_batch)
    ypred = th.argmax(logits, dim=1)
    ytrue = labels.numpy().flatten()
    print(ypred.detach().cpu().numpy() == ytrue)

    loss = crit(logits, ypred)
    loss.backward()
    X_adv = X_batch + EPS * X_batch.grad.sign()
    X_adv = th.clamp(X_adv, 0, 1).detach()

    logits = net(X_adv)
    yadv = np.argmax(logits.detach().cpu().numpy(), axis=1)
    print(yadv == ytrue)

    ind = np.where(yadv != ytrue)[0]

    x0 = th.stack([unnormalize(x) for x in images[ind]]).detach().numpy()
    x1 = th.stack([unnormalize(x) for x in X_adv[ind]]).detach().cpu().numpy()
    y1 = yadv

    break

# B, C, W, H
x0 = np.moveaxis(x0, 1, -1)
x1 = np.moveaxis(x1, 1, -1)

for i, (a, b, c) in enumerate(zip(x0, x1, y1)):
    a = (np.squeeze(a) * 255).astype(np.uint8)
    b = (np.squeeze(b) * 255).astype(np.uint8)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(a)
    plt.axis('off')
    fig.tight_layout()
    plt.savefig(f'tmp/get/cifar10-good-{i}.jpg', bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(b)
    ax.text(0.85,
            0.85,
            c,
            horizontalalignment='center',
            fontsize=70,
            color='red',
            verticalalignment='center',
            transform=ax.transAxes)
    plt.axis('off')
    fig.tight_layout()
    plt.savefig(f'tmp/get/cifar10-bad-{i}.jpg', bbox_inches='tight')
