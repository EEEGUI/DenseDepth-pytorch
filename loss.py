import torch
import torch.nn.functional as F
import pytorch_ssim


def criterion(y_true, y_pred, theta=0.1, max_depth_val=1000.0/10.0):
    l_depth = torch.mean(torch.abs(y_true-y_pred))

    dx_pred, dy_pred = gradient(y_pred)
    dx_true, dy_true = gradient(y_true)

    l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_true - dx_pred))

    ssim_loss = pytorch_ssim.SSIM()
    l_ssim = torch.clamp(1 - ssim_loss(y_true, y_pred), 0, 1)

    return theta * l_depth + l_edges + l_ssim


def gradient(x):
    h_x = x.size()[-2]
    w_x = x.size()[-1]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    return torch.abs(r-l), torch.abs(t-b)


if __name__ == '__main__':
    yt = (torch.randn(4, 1, 348, 1280))
    yp = (torch.randn(4, 1, 348, 1280))
    print(criterion(yp, yt))
