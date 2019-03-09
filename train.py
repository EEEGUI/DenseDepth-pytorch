from model import DDNet
from loss import criterion
import torch.optim as optim
import torch
from Dataloader import dataset_loader
from utils import scale_up
import torch.nn as nn

def train():
    device = torch.device('cuda')
    net = DDNet().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, batch_sample in enumerate(dataset_loader):
            # get the inputs
            inputs = batch_sample['image'].to(device)
            labels = batch_sample['depth'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i+1) % 1 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1))
                running_loss = 0.0

    print('Finished Training')

if __name__ == '__main__':
    train()