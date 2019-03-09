from model import DDNet
from loss import criterion
import torch.optim as optim
import torch
from Dataloader import dataset_loader
from tensorboardX import SummaryWriter


def train():
    writer = SummaryWriter('log')
    device = torch.device('cuda')
    net = DDNet().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    n_batch = len(dataset_loader)
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

            writer.add_scalar('Train/Loss', loss.item(), epoch*n_batch + (i+1))
            # print statistics
            running_loss += loss.item()
            if (i+1) % 1 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1))
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), 'densedepth.pt')
if __name__ == '__main__':
    train()