import torch
import torch.nn as nn
from torchvision import models


class DDNet(nn.Module):
    def __init__(self):
        super(DDNet, self).__init__()
        self.up_channels = [64, 64, 128, 256]
        self.densenet = models.densenet169(pretrained=True)

        self.slice = nn.Sequential(self.densenet.features[:3])

        self.encoder_out_channels = self.densenet.classifier.in_features

        self.decoder_in_channels = int(self.encoder_out_channels // 2)

        self.conv1 = nn.Conv2d(self.encoder_out_channels, self.decoder_in_channels, kernel_size=1)

        self.upproject1 = UpProject(self.decoder_in_channels + self.up_channels[-1], int(self.decoder_in_channels/2))
        self.upproject2 = UpProject(int(self.decoder_in_channels/2) + self.up_channels[-2], int(self.decoder_in_channels/4))
        self.upproject3 = UpProject(int(self.decoder_in_channels/4) + self.up_channels[-3], int(self.decoder_in_channels/8))
        self.upproject4 = UpProject(int(self.decoder_in_channels/8) + self.up_channels[-4], int(self.decoder_in_channels/16))

        self.conv2 = nn.Conv2d(int(self.decoder_in_channels/16), 1, kernel_size=3, stride=1, padding=(1, 1))

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        # Encoder
        up1 = self.slice(x)

        up2 = self.densenet.features.pool0(up1)

        x = self.densenet.features.denseblock1(up2)
        up3 = self.densenet.features.transition1(x)

        x = self.densenet.features.denseblock2(up3)
        up4 = self.densenet.features.transition2(x)

        x = self.densenet.features.denseblock3(up4)
        x = self.densenet.features.transition3(x)

        x = self.densenet.features.denseblock4(x)
        x = self.densenet.features.norm5(x)

        x = self.conv1(x)

        # Decoder
        x = self.upproject1(x, up4)
        x = self.upproject2(x, up3)
        x = self.upproject3(x, up2)
        x = self.upproject4(x, up1)

        x = self.conv2(x)

        x = self.upsample(x)

        return x


class UpProject(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpProject, self).__init__()
        self.bilinear_upsample_2d = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.leak_relu = nn.LeakyReLU(0.2)

    def forward(self, x, up):
        x = self.bilinear_upsample_2d(x)
        x = torch.cat([x, up], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.leak_relu(x)

        return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 1280, 384)
    net = DDNet()

    def count_param(net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(num_params / 1e6)

    count_param(net)
