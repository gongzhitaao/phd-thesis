import os

import torch as th
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.models import GoogLeNet
from torch.utils.data import DataLoader

from tqdm import tqdm

trans = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
data_train = CIFAR10(os.path.expanduser('~/data/large'),
                     download=True,
                     transform=trans)

data_train_loader = DataLoader(data_train,
                               batch_size=512,
                               shuffle=True,
                               num_workers=8,
                               pin_memory=True)

device = 'cuda:0'
net = GoogLeNet(num_classes=10)

if True:
    net.load_state_dict(th.load('tmp/cifar10.pth'))

net.to(device)
crit = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=1e-3)
net.train()
epochs = 10
pbar = tqdm(range(epochs), ascii=True)
pbar.set_postfix(loss='inf')
for _ in pbar:
    for batch, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()
        X_batch, y_batch = images.to(device), labels.to(device)
        out = net(X_batch)
        loss = crit(out.logits, y_batch)
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            pbar.set_postfix(loss='{:.6f}'.format(loss.detach().cpu()))

th.save(net.state_dict(), 'tmp/cifar10.pth')
