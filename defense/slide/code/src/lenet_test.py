import os

import numpy as np

import torch as th
import torch.nn as nn

from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

from lenet import LeNet5

data_test = MNIST(
    os.path.expanduser('~/data/large'),
    train=False,
    download=False,
    transform=transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor()]))

data_test_loader = DataLoader(
    data_test, batch_size=1024, num_workers=8, pin_memory=True)

device = 'cuda:0'
net = LeNet5()
net.to(device)
crit = nn.CrossEntropyLoss()

net.load_state_dict(th.load('tmp/mnist.pth'))
net.eval()
total_correct = 0
acc, loss = 0, 0.0

pbar = tqdm(data_test_loader, ascii=True)
pbar.set_postfix(loss='inf')
for batch, (images, labels) in enumerate(pbar):
    X_batch, y_batch = images.to(device), labels.to(device)
    logits = net(X_batch)
    loss += crit(logits, y_batch).sum()
    yy = np.argmax(logits.detach().cpu().numpy(), axis=1)
    y = labels.numpy().flatten()
    acc += np.sum(y == yy)

    if batch % 10 == 0:
        pbar.set_postfix(loss='{:.4f}'.format(loss.detach().cpu()))

loss = loss.detach().cpu() / len(data_test)
acc /= len(data_test)
print('Test avg Loss: {:.6f}, acc: {:.4f}'.format(loss, acc))
