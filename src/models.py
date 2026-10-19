import torch
from torch import nn


def _downsample(in_channels: int, out_channels: int, normalize: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not normalize)
    ]
    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _upsample(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class InpaintingGenerator(nn.Module):
    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = _downsample(4, c, normalize=False)
        self.enc2 = _downsample(c, c * 2)
        self.enc3 = _downsample(c * 2, c * 4)
        self.enc4 = _downsample(c * 4, c * 8)
        self.enc5 = _downsample(c * 8, c * 8)
        self.dec1 = _upsample(c * 8, c * 8)
        self.dec2 = _upsample(c * 16, c * 4)
        self.dec3 = _upsample(c * 8, c * 2)
        self.dec4 = _upsample(c * 4, c)
        self.output = nn.Sequential(
            nn.ConvTranspose2d(c * 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, masked_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(torch.cat((masked_image, mask), dim=1))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        d1 = self.dec1(e5)
        d2 = self.dec2(torch.cat((d1, e4), dim=1))
        d3 = self.dec3(torch.cat((d2, e3), dim=1))
        d4 = self.dec4(torch.cat((d3, e2), dim=1))
        return self.output(torch.cat((d4, e1), dim=1))


class PatchDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.network = nn.Sequential(
            nn.Conv2d(4, c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((image, mask), dim=1))
