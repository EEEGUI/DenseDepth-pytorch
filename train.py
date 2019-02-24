from model import DDNet
from loss import criterion
import torch.optim as optim
import torch
from utils import scale_up
import torch.nn as nn

def train():
    device = torch.device('cuda')
    net = DDNet().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i in range(100):
            # get the inputs
            inputs = torch.randn(1, 3, 640, 480).to(device)
            labels = inputs[:, :1, :, :].to(device)
            # labels = torch.randn(1, 1, 640, 480).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i+1) % 5 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

    print('Finished Training')

if __name__ == '__main__':
    train()