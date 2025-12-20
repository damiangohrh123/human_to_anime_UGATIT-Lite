# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# ----------------------------
# Conv / Residual / Norm blocks
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, norm='in', activation='relu'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        
        if norm == 'in':
            self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        elif norm == 'bn':
            self.norm = nn.BatchNorm2d(out_ch)
        else:
            self.norm = nn.Identity()
        
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, norm='in', activation='relu'),
            ConvBlock(channels, channels, norm='in', activation='none')
        )

    def forward(self, x):
        return x + self.block(x)

# ----------------------------
# ILN / AdaILN
# ----------------------------
class ILN(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.rho = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        in_mean = x.mean([2,3], keepdim=True)
        in_var = x.var([2,3], keepdim=True)
        ln_mean = x.mean([1,2,3], keepdim=True)
        ln_var = x.var([1,2,3], keepdim=True)

        out = self.rho * ((x - in_mean) / torch.sqrt(in_var + self.eps)) + \
              (1 - self.rho) * ((x - ln_mean) / torch.sqrt(ln_var + self.eps))
        out = out * self.gamma + self.beta
        return out

class AdaILN(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.rho = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, gamma, beta):
        in_mean = x.mean([2,3], keepdim=True)
        in_var = x.var([2,3], keepdim=True)
        ln_mean = x.mean([1,2,3], keepdim=True)
        ln_var = x.var([1,2,3], keepdim=True)

        out = self.rho * ((x - in_mean) / torch.sqrt(in_var + self.eps)) + \
              (1 - self.rho) * ((x - ln_mean) / torch.sqrt(ln_var + self.eps))
        out = out * gamma + beta
        return out

# ----------------------------
# CAM attention block
# ----------------------------
class CAMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gap_fc = nn.Linear(channels, 1, bias=False)
        self.gmp_fc = nn.Linear(channels, 1, bias=False)
        self.conv1x1 = nn.Conv2d(channels*2, channels, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.size()
        gap = F.adaptive_avg_pool2d(x, 1).view(b,c)
        gmp = F.adaptive_max_pool2d(x, 1).view(b,c)
        gap_logit = self.gap_fc(gap)
        gmp_logit = self.gmp_fc(gmp)
        # get weights (shape: [c, 1]) -> expand to multiply
        gap_weight = list(self.gap_fc.parameters())[0]   # weight shape (c, 1)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        # apply channel-wise weighting
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        cam = torch.cat([gap, gmp], dim=1)
        cam = self.relu(self.conv1x1(cam))
        return cam, gap_logit + gmp_logit

# ----------------------------
# Generator
# ----------------------------
class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=4):
        super().__init__()
        self.down = nn.Sequential(
            ConvBlock(input_nc, ngf, 7, 1, 3, norm='in', activation='relu'),
            ConvBlock(ngf, ngf*2, 4, 2, 1, norm='in', activation='relu'),
            ConvBlock(ngf*2, ngf*4, 4, 2, 1, norm='in', activation='relu')
        )
        # Residual blocks
        res_blocks = []
        for _ in range(n_blocks):
            res_blocks.append(ResBlock(ngf*4))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.cam = CAMBlock(ngf*4)

        # Up-sampling
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(ngf*4, ngf*2, 3, 1, 1, norm='in', activation='relu'),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(ngf*2, ngf, 3, 1, 1, norm='in', activation='relu'),
            nn.Conv2d(ngf, output_nc, 7, 1, 3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.down(x)
        x = self.res_blocks(x)
        x, cam_logit = self.cam(x)
        x = self.up(x)
        return x, cam_logit

# ----------------------------
# Discriminator
# ----------------------------
class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(input_nc, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, 1, 4, 1, 1)  # PatchGAN output
        )

    def forward(self, x):
        return self.model(x)

__all__ = ["Generator", "Discriminator"]
