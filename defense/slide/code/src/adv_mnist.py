import os

import numpy as np

import torch as th
import torch.nn as nn

from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # noqa: E402
import matplotlib.pyplot as plt

from lenet import LeNet5

data_test = MNIST(os.path.expanduser('~/data/large'),
                  train=False,
                  download=False,
                  transform=transforms.Compose(
                      [transforms.Resize((32, 32)),
                       transforms.ToTensor()]))

data_test_loader = DataLoader(data_test,
                              batch_size=8,
                              num_workers=8,
                              pin_memory=True,
                              shuffle=True)

device = 'cuda:7'
net = LeNet5()
net.to(device)
crit = nn.CrossEntropyLoss()

net.load_state_dict(th.load('tmp/mnist.pth'))
net.eval()
for p in net.parameters():
    p.requires_grad = False

total_correct = 0
acc, loss = 0, 0.0

pbar = tqdm(data_test_loader, ascii=True)
pbar.set_postfix(loss='inf')
crit = th.nn.CrossEntropyLoss()

EPS = 0.2

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

    x0 = images.detach().numpy()[ind]
    x1 = X_adv.detach().cpu().numpy()[ind]
    y1 = yadv

    print(x0.shape)
    print(x1.shape)

    break

for i, (a, b, c) in enumerate(zip(x0, x1, y1)):
    a = (np.squeeze(a) * 255).astype(np.uint8)
    b = (np.squeeze(b) * 255).astype(np.uint8)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(a, cmap='gray', interpolation='none')
    plt.axis('off')
    fig.tight_layout()
    plt.savefig(f'tmp/get/good-{i}.jpg', bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(b, cmap='gray', interpolation='none')
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
    plt.savefig(f'tmp/get/bad-{i}.jpg', bbox_inches='tight')
