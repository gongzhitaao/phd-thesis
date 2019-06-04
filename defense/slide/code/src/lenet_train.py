import os

import torch as th
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

from lenet import LeNet5

data_train = MNIST(
    os.path.expanduser('~/data/large'),
    download=True,
    transform=transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor()]))

data_train_loader = DataLoader(
    data_train, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)

device = 'cuda:0'
net = LeNet5()
net.to(device)
crit = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=2e-3)
net.train()
epochs = 10
pbar = tqdm(range(epochs), ascii=True)
pbar.set_postfix(loss='inf')
for _ in pbar:
    for batch, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()
        X_batch, y_batch = images.to(device), labels.to(device)
        ybar = net(X_batch)
        loss = crit(ybar, y_batch)
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            pbar.set_postfix(loss='{:.6f}'.format(loss.detach().cpu()))

th.save(net.state_dict(), 'tmp/mnist.pth')
