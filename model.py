import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F

import extractors


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, n_classes=18, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax()
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        return self.final(p), self.classifier(auxiliary)


class PSPDepthNet(nn.Module):
    def __init__(self, n_classes=1, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50',
                 pretrained=True):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        return self.final(p), self.classifier(auxiliary)




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
    net = PSPDepthNet()
    net2 = DDNet()
    # print(net(x))
    def count_param(net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(num_params / 1e6)

    count_param(net)
    count_param(net2)